# rt_patchtst_onefile.py
# -*- coding: utf-8 -*-
"""
One-file production-ready RT (15min, 6h=24 steps) forecasting with:
- PatchTST-style history encoder
- Future exogenous driver (internal lag1d / lag1d_residual) => wide future matrix
- RT baseline + residual learning (stabilizes spikes)
- Optional future DA path injection (recommended)

Main modes:
  --mode fit     : train + eval + save ckpt/meta
  --mode predict : load ckpt/meta and output next-24-step RT forecast for a given asof

Design goals:
- No leakage: cutoff_time determined by last valid timestamps of required cols.
- Robustness: strict NaN defense + optional drop-invalid + fallback exog.
- Deployable: single file, reproducible meta.json, deterministic seed.

Authoring notes:
- This script does NOT use any '*预测值' columns from Excel.
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Constants
# -----------------------------
STEP_MINUTES = 15
HORIZON = 24          # 6h
LOOKBACK = 192        # 2d (96*2)
DAY_STEPS = 96
SEED = 42


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def dt_floor(ts: pd.Series, minutes: int = 15) -> pd.Series:
    return ts.dt.floor(f"{minutes}min")

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def infer_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = df.columns.tolist()
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        hits = [x for x in cols if c in x]
        if len(hits) == 1:
            return hits[0]
    return None

def check_15min_continuity(t: pd.Series) -> None:
    dt_diff = t.diff().dropna()
    if not (dt_diff == pd.Timedelta(minutes=STEP_MINUTES)).all():
        bad = dt_diff[dt_diff != pd.Timedelta(minutes=STEP_MINUTES)]
        raise ValueError(
            f"Time is not strictly {STEP_MINUTES}min continuous. Found {len(bad)} irregular intervals.\n"
            f"Fix by reindexing to 15min grid before running."
        )

def last_valid_timestamp(time_index: pd.Series, values: pd.Series) -> Optional[pd.Timestamp]:
    a = pd.to_numeric(values, errors="coerce").to_numpy()
    good = np.isfinite(a)
    if not good.any():
        return None
    return pd.Timestamp(time_index.iloc[np.where(good)[0][-1]])

def compute_cutoff_time(
    df: pd.DataFrame, time_col: str, required_cols: List[str]
) -> Tuple[pd.Timestamp, Dict[str, Optional[str]], Dict[str, str]]:
    last_map: Dict[str, Optional[str]] = {}
    missing: Dict[str, str] = {}
    times: List[pd.Timestamp] = []

    for c in required_cols:
        if c not in df.columns:
            last_map[c] = None
            missing[c] = "missing_col"
            continue
        lv = last_valid_timestamp(df[time_col], df[c])
        if lv is None:
            last_map[c] = None
            missing[c] = "all_nan_or_nonfinite"
            continue
        last_map[c] = lv.isoformat(sep=" ")
        times.append(lv)

    if len(times) == 0:
        raise ValueError(f"[CUTOFF] No valid required columns: {required_cols} / missing={missing}")
    cutoff_time = min(times)
    return cutoff_time, last_map, missing

def make_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    n = len(df)
    valid_i = np.arange(LOOKBACK - 1, n - HORIZON, dtype=np.int64)
    t_dec = df[time_col].iloc[valid_i].reset_index(drop=True)
    tgt_max_time = t_dec + pd.to_timedelta(HORIZON * STEP_MINUTES, unit="m")
    date_tgt_max = pd.to_datetime(tgt_max_time.dt.date)
    return pd.DataFrame({"t_dec": t_dec, "tgt_max_time": tgt_max_time, "date_tgt_max": date_tgt_max})

def split_by_tgtmax_day(
    idx: pd.DataFrame, val_days: int, test_days: int, require_full_day: bool
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    day_counts = idx.groupby("date_tgt_max").size().sort_index()
    eligible_days = day_counts[day_counts == 96].index if require_full_day else day_counts.index
    eligible_days = pd.Series(eligible_days).sort_values().reset_index(drop=True)

    if len(eligible_days) < (val_days + test_days + 1):
        raise ValueError(
            f"[SPLIT] Not enough eligible days. eligible_days={len(eligible_days)} need>={val_days+test_days+1}\n"
            f"day_counts_tail:\n{day_counts.tail(14)}"
        )

    test_start = eligible_days.iloc[-test_days]
    val_start = eligible_days.iloc[-(test_days + val_days)]
    last_eligible = eligible_days.iloc[-1]

    out = idx.copy()
    out["split"] = "train"
    out.loc[out["date_tgt_max"] >= val_start, "split"] = "val"
    out.loc[out["date_tgt_max"] >= test_start, "split"] = "test"

    if require_full_day:
        out = out[out["date_tgt_max"].isin(set(eligible_days))].reset_index(drop=True)

    info = {
        "val_start": str(pd.Timestamp(val_start).date()),
        "test_start": str(pd.Timestamp(test_start).date()),
        "last_eligible_day": str(pd.Timestamp(last_eligible).date()),
        "require_full_day": str(require_full_day),
    }
    return out, info

def get_dec_pos(df: pd.DataFrame, idx: pd.DataFrame, time_col: str) -> np.ndarray:
    pos = pd.Series(np.arange(len(df), dtype=np.int64), index=df[time_col]).to_dict()
    return idx["t_dec"].map(pos).astype(np.int64).values

def add_time_features(t_dec: pd.Series) -> np.ndarray:
    # returns [N, 6]: hour_sin/cos, dow_sin/cos, month_sin/cos
    t = pd.to_datetime(t_dec)
    hour = t.dt.hour.values + t.dt.minute.values / 60.0
    dow = t.dt.dayofweek.values.astype(np.float32)
    month = t.dt.month.values.astype(np.float32)

    hour_rad = 2 * np.pi * hour / 24.0
    dow_rad = 2 * np.pi * dow / 7.0
    mon_rad = 2 * np.pi * (month - 1) / 12.0

    feats = np.stack([
        np.sin(hour_rad), np.cos(hour_rad),
        np.sin(dow_rad),  np.cos(dow_rad),
        np.sin(mon_rad),  np.cos(mon_rad),
    ], axis=1).astype(np.float32)
    return feats

def safe_to_numeric(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)

def masked_np_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return (np.nan, np.nan)
    return (
        float(mean_absolute_error(y_true[m], y_pred[m])),
        float(np.sqrt(mean_squared_error(y_true[m], y_pred[m])))
    )


# -----------------------------
# Column detection
# -----------------------------
@dataclass
class ColMap:
    time: str
    rt: str
    da: str
    load: str
    pv: str
    line: str
    ru: str
    rd: str
    wind: Optional[str] = None
    wg_total: Optional[str] = None
    space: Optional[str] = None
    local_gen: Optional[str] = None

def detect_colmap(df: pd.DataFrame) -> ColMap:
    time = infer_col(df, ["datetime", "时间", "日期时间"])
    rt   = infer_col(df, ["实时出清电价", "实时电价", "RT", "实时价格"])
    da   = infer_col(df, ["日前出清电价", "日前电价", "DA", "日前价格"])

    load = infer_col(df, ["系统负荷实际值"])
    pv   = infer_col(df, ["光伏实际值"])
    line = infer_col(df, ["联络线实际值"])
    ru   = infer_col(df, ["上旋备用实际值"])
    rd   = infer_col(df, ["下旋备用实际值"])

    wind = infer_col(df, ["风电实际值"])
    wg_total = infer_col(df, ["风光总加实际值"])
    space = infer_col(df, ["竞价空间实际值"])
    local_gen = infer_col(df, ["地方电厂发电实际值"])

    req = {"time": time, "rt": rt, "da": da, "load": load, "pv": pv, "line": line, "ru": ru, "rd": rd}
    missing = [k for k, v in req.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable cols head: {df.columns.tolist()[:80]}")

    return ColMap(
        time=time, rt=rt, da=da, load=load, pv=pv, line=line, ru=ru, rd=rd,
        wind=wind, wg_total=wg_total, space=space, local_gen=local_gen
    )

def load_excel(data_path: str) -> Tuple[pd.DataFrame, ColMap]:
    df = pd.read_excel(data_path, engine="openpyxl", sheet_name=0)
    colmap = detect_colmap(df)

    df[colmap.time] = pd.to_datetime(df[colmap.time], errors="coerce")
    df = df.dropna(subset=[colmap.time]).copy()
    df[colmap.time] = dt_floor(df[colmap.time], STEP_MINUTES)

    df = df.sort_values(colmap.time).drop_duplicates(colmap.time, keep="last").reset_index(drop=True)
    check_15min_continuity(df[colmap.time])
    return df, colmap


# -----------------------------
# Scalers
# -----------------------------
@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

def fit_scaler_from_train_rows(series_mat: np.ndarray) -> Scaler:
    mean = np.nanmean(series_mat, axis=0)
    std = np.nanstd(series_mat, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return Scaler(mean=mean.astype(np.float32), std=std.astype(np.float32))


# -----------------------------
# Exogenous driver (internal)
# -----------------------------
def build_exog_wide(
    df: pd.DataFrame,
    idx: pd.DataFrame,
    colmap: ColMap,
    future_vars: List[str],
    method: str,
    max_iter: int,
    fallback_direct: bool,
) -> pd.DataFrame:
    """
    Build wide future exog predictions for each t_dec in idx:
      columns: t_dec, var_h01..var_h24
    Supported:
      - lag1d          : pred = value(t_dec - 1day + h)
      - lag1d_residual : pred = lag1d + median(residual_by_h) (train-only)
    """
    time_col = colmap.time
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    var_to_col = {
        "load": colmap.load,
        "pv": colmap.pv,
        "line": colmap.line,
        "ru": colmap.ru,
        "rd": colmap.rd,
        "wind": colmap.wind,
        "wg_total": colmap.wg_total,
        "space": colmap.space,
        "local_gen": colmap.local_gen,
    }

    pos_map = pd.Series(np.arange(len(df), dtype=np.int64), index=df[time_col]).to_dict()
    dec_pos = idx["t_dec"].map(pos_map).astype(np.int64).values

    out = pd.DataFrame({"t_dec": idx["t_dec"].values})

    def lag1d_pred(arr: np.ndarray, dec_pos_: np.ndarray, h: int) -> np.ndarray:
        src = dec_pos_ - DAY_STEPS + h
        pred = np.full(len(dec_pos_), np.nan, dtype=np.float32)
        m = (src >= 0) & (src < len(arr))
        pred[m] = arr[src[m]]
        return pred

    idx_train = idx[idx["split"] == "train"].reset_index(drop=True)
    dec_pos_train = idx_train["t_dec"].map(pos_map).astype(np.int64).values

    for v in future_vars:
        col = var_to_col.get(v, None)
        if col is None or col not in df.columns:
            raise ValueError(f"[EXOG] future var {v} has no actual column mapping in Excel.")

        arr = safe_to_numeric(df[col])
        med_res = np.zeros(HORIZON, dtype=np.float32)

        if method == "lag1d_residual":
            for h in range(1, HORIZON + 1):
                base = lag1d_pred(arr, dec_pos_train, h)
                tgt_idx = dec_pos_train + h
                ok = (tgt_idx >= 0) & (tgt_idx < len(arr))
                y_true = np.full(len(dec_pos_train), np.nan, dtype=np.float32)
                y_true[ok] = arr[tgt_idx[ok]]
                m = np.isfinite(base) & np.isfinite(y_true)
                if m.sum() < 200:
                    med_res[h-1] = 0.0
                else:
                    med_res[h-1] = np.nanmedian((y_true[m] - base[m]).astype(np.float32))

        for h in range(1, HORIZON + 1):
            base_all = lag1d_pred(arr, dec_pos, h)
            if method == "lag1d":
                pred = base_all
            elif method == "lag1d_residual":
                pred = base_all + med_res[h-1]
            else:
                raise ValueError(f"[EXOG] unknown method={method}")

            if fallback_direct and method == "lag1d_residual":
                fb = base_all
                mfb = ~np.isfinite(pred) & np.isfinite(fb)
                pred[mfb] = fb[mfb]

            out[f"{v}_h{h:02d}"] = pred.astype(np.float32)

        nan_cnt = int(np.isnan(out[[f"{v}_h{h:02d}" for h in range(1, HORIZON + 1)]].to_numpy()).sum())
        if nan_cnt > 0 and not fallback_direct:
            raise RuntimeError(f"[EXOG] NaN exists in wide for var={v}. Consider --exog_fallback_direct")

    return out


# -----------------------------
# RT baseline (stabilizer)
# -----------------------------
def build_rt_baseline_wide(
    df: pd.DataFrame,
    idx: pd.DataFrame,
    colmap: ColMap,
    rt_method: str,
    use_future_da: bool,
) -> pd.DataFrame:
    """
    Build baseline RT forecast wide for each t_dec:
      columns: t_dec, rtbase_h01..rtbase_h24
    rt_method:
      - lag1d          : baseline = RT(t-1day + h)
      - lag1d_residual : baseline = lag1d + median(RT - lag1d) by horizon (train-only)
    """
    time_col = colmap.time
    df = df.sort_values(time_col).reset_index(drop=True)
    pos_map = pd.Series(np.arange(len(df), dtype=np.int64), index=df[time_col]).to_dict()

    dec_pos = idx["t_dec"].map(pos_map).astype(np.int64).values
    rt = safe_to_numeric(df[colmap.rt])
    da = safe_to_numeric(df[colmap.da])

    def lag1d(arr: np.ndarray, dec_pos_: np.ndarray, h: int) -> np.ndarray:
        src = dec_pos_ - DAY_STEPS + h
        pred = np.full(len(dec_pos_), np.nan, dtype=np.float32)
        m = (src >= 0) & (src < len(arr))
        pred[m] = arr[src[m]]
        return pred

    out = pd.DataFrame({"t_dec": idx["t_dec"].values})

    idx_train = idx[idx["split"] == "train"].reset_index(drop=True)
    dec_pos_train = idx_train["t_dec"].map(pos_map).astype(np.int64).values

    med_res = np.zeros(HORIZON, dtype=np.float32)
    if rt_method == "lag1d_residual":
        for h in range(1, HORIZON + 1):
            base = lag1d(rt, dec_pos_train, h)
            tgt_idx = dec_pos_train + h
            ok = (tgt_idx >= 0) & (tgt_idx < len(rt))
            y_true = np.full(len(dec_pos_train), np.nan, dtype=np.float32)
            y_true[ok] = rt[tgt_idx[ok]]

            m = np.isfinite(y_true) & np.isfinite(base)
            if m.sum() >= 200:
                med_res[h-1] = np.nanmedian((y_true[m] - base[m]).astype(np.float32))
            else:
                med_res[h-1] = 0.0

    for h in range(1, HORIZON + 1):
        base_all = lag1d(rt, dec_pos, h)
        if rt_method == "lag1d":
            pred = base_all
        elif rt_method == "lag1d_residual":
            pred = base_all + med_res[h-1]
        else:
            raise ValueError(f"[RTBASE] unknown rt_method={rt_method}")
        out[f"rtbase_h{h:02d}"] = pred.astype(np.float32)

    return out


# -----------------------------
# Dataset
# -----------------------------
class RtDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        idx: pd.DataFrame,
        colmap: ColMap,
        hist_cols: List[str],
        exog_wide: pd.DataFrame,
        future_vars: List[str],
        rtbase_wide: pd.DataFrame,
        use_future_da: bool,
        scaler_hist: Scaler,
        scaler_future: Dict[str, Tuple[float, float]],
        scaler_rt_delta: Tuple[float, float],
        split: str,
        drop_invalid_samples: bool,
        hist_ratio_min: float,
        fut_ratio_min: float,
    ):
        self.df = df
        self.colmap = colmap
        self.hist_cols = hist_cols
        self.future_vars = future_vars
        self.use_future_da = use_future_da
        self.scaler_hist = scaler_hist
        self.scaler_future = scaler_future
        self.mu_delta, self.sd_delta = scaler_rt_delta
        self.split = split

        sub = idx[idx["split"] == split].reset_index(drop=True).copy()
        self.idx_raw = sub

        time_col = colmap.time
        self.dec_pos = get_dec_pos(df, sub, time_col)

        exog_wide = exog_wide.copy()
        exog_wide["t_dec"] = pd.to_datetime(exog_wide["t_dec"])
        self.exog = exog_wide.set_index("t_dec")

        rtbase_wide = rtbase_wide.copy()
        rtbase_wide["t_dec"] = pd.to_datetime(rtbase_wide["t_dec"])
        self.rtbase = rtbase_wide.set_index("t_dec")

        self.y_rt = safe_to_numeric(df[colmap.rt])
        self.y_da = safe_to_numeric(df[colmap.da])

        self.hist_full = np.stack([safe_to_numeric(df[c]) for c in hist_cols], axis=1)  # [T, C_hist]

        keep_mask = np.ones(len(sub), dtype=bool)
        if drop_invalid_samples:
            for i in range(len(sub)):
                dp = int(self.dec_pos[i])
                xh = self.hist_full[dp - LOOKBACK + 1: dp + 1, :]
                hist_ratio = np.isfinite(xh).mean()

                t_dec = pd.Timestamp(sub.loc[i, "t_dec"])
                xf = self._get_future_exog(t_dec)
                fut_ratio = np.isfinite(xf).mean()

                rb = self._get_rtbase(t_dec)
                rb_ratio = np.isfinite(rb).mean()

                if (hist_ratio < hist_ratio_min) or (fut_ratio < fut_ratio_min) or (rb_ratio < 1.0):
                    keep_mask[i] = False

            sub = sub[keep_mask].reset_index(drop=True)
            self.dec_pos = self.dec_pos[keep_mask]
            self.idx = sub
        else:
            self.idx = sub

        self.time_feats = add_time_features(self.idx["t_dec"])

    def __len__(self) -> int:
        return len(self.idx)

    def _get_future_exog(self, t_dec: pd.Timestamp) -> np.ndarray:
        H, C = HORIZON, len(self.future_vars)
        out = np.full((H, C), np.nan, dtype=np.float32)
        if t_dec not in self.exog.index:
            return out
        row = self.exog.loc[t_dec]
        for j, v in enumerate(self.future_vars):
            for h in range(1, H + 1):
                key = f"{v}_h{h:02d}"
                out[h-1, j] = np.float32(row[key]) if key in row.index else np.nan
        return out

    def _get_rtbase(self, t_dec: pd.Timestamp) -> np.ndarray:
        out = np.full((HORIZON,), np.nan, dtype=np.float32)
        if t_dec not in self.rtbase.index:
            return out
        row = self.rtbase.loc[t_dec]
        for h in range(1, HORIZON + 1):
            key = f"rtbase_h{h:02d}"
            out[h-1] = np.float32(row[key]) if key in row.index else np.nan
        return out

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        dp = int(self.dec_pos[i])
        t_dec = pd.Timestamp(self.idx.loc[i, "t_dec"])

        x_hist = self.hist_full[dp - LOOKBACK + 1: dp + 1, :]
        xh = pd.DataFrame(x_hist).ffill().bfill().to_numpy(dtype=np.float32)
        x_hist = self.scaler_hist.transform(xh)

        x_time = self.time_feats[i]

        x_fut = self._get_future_exog(t_dec)
        xf = pd.DataFrame(x_fut).ffill().bfill().to_numpy(dtype=np.float32)
        for j, v in enumerate(self.future_vars):
            mu, sd = self.scaler_future[v]
            xf[:, j] = (xf[:, j] - mu) / sd

        if self.use_future_da:
            da_future = self.y_da[dp + 1: dp + 1 + HORIZON].astype(np.float32)
            da_future = pd.Series(da_future).ffill().bfill().to_numpy(dtype=np.float32)
            mu_da = float(np.nanmean(da_future))
            sd_da = float(np.nanstd(da_future))
            sd_da = sd_da if (np.isfinite(sd_da) and sd_da > 1e-6) else 1.0
            da_norm = ((da_future - mu_da) / sd_da).astype(np.float32)[:, None]
            xf = np.concatenate([xf, da_norm], axis=1)

        rt_base = self._get_rtbase(t_dec)
        rt_base = pd.Series(rt_base).ffill().bfill().to_numpy(dtype=np.float32)

        y = self.y_rt[dp + 1: dp + 1 + HORIZON].astype(np.float32)
        y = pd.Series(y).ffill().bfill().to_numpy(dtype=np.float32)

        delta = (y - rt_base).astype(np.float32)
        delta_n = (delta - self.mu_delta) / self.sd_delta

        return {
            "x_hist": torch.from_numpy(x_hist.astype(np.float32)),
            "x_time": torch.from_numpy(x_time.astype(np.float32)),
            "x_fut":  torch.from_numpy(xf.astype(np.float32)),
            "rt_base": torch.from_numpy(rt_base.astype(np.float32)),
            "y": torch.from_numpy(delta_n.astype(np.float32)),
        }


# -----------------------------
# Model
# -----------------------------
class PatchTSTResidual(nn.Module):
    def __init__(
        self,
        c_hist: int,
        c_fut: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        patch_len: int = 16,
        time_dim: int = 6,
        horizon: int = HORIZON,
    ):
        super().__init__()
        assert LOOKBACK % patch_len == 0
        self.horizon = horizon
        self.patch_len = patch_len

        self.hist_patch = nn.Conv1d(c_hist, d_model, kernel_size=patch_len, stride=patch_len, bias=True)
        n_patches = LOOKBACK // patch_len
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.fut_in = nn.Linear(c_fut, d_model)
        self.fut_pos = nn.Parameter(torch.zeros(1, horizon, d_model))
        self.fut_encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, n_layers // 2))

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.fut_pos, std=0.02)

    def forward(self, x_hist: torch.Tensor, x_time: torch.Tensor, x_fut: torch.Tensor) -> torch.Tensor:
        B, L, C = x_hist.shape
        xh = x_hist.permute(0, 2, 1)
        tok = self.hist_patch(xh)
        tok = tok.permute(0, 2, 1) + self.pos
        tok = self.drop(tok)
        enc = self.encoder(tok)
        ctx = enc.mean(dim=1)

        ft = self.fut_in(x_fut) + self.fut_pos
        ft = self.drop(ft)
        ft = self.fut_encoder(ft)
        fut = ft.mean(dim=1)

        tm = self.time_mlp(x_time)

        z = torch.cat([ctx, fut, tm], dim=1)
        out = self.head(z)
        return out


# -----------------------------
# Loss
# -----------------------------
def masked_huber(y_hat: torch.Tensor, y_true: torch.Tensor, delta: float = 2.0) -> torch.Tensor:
    mask = torch.isfinite(y_true) & torch.isfinite(y_hat)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=y_hat.device)
    return torch.nn.functional.huber_loss(y_hat[mask], y_true[mask], delta=delta, reduction="mean")


# -----------------------------
# Train / Eval helpers
# -----------------------------
def predict_all(model: nn.Module, dl: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, yh, rb = [], [], []
    with torch.no_grad():
        for b in dl:
            x_hist = b["x_hist"].to(device)
            x_time = b["x_time"].to(device)
            x_fut  = b["x_fut"].to(device)
            y      = b["y"].cpu().numpy()
            rt_base = b["rt_base"].cpu().numpy()

            yhat = model(x_hist, x_time, x_fut).cpu().numpy()
            ys.append(y)
            yh.append(yhat)
            rb.append(rt_base)
    return np.concatenate(ys), np.concatenate(yh), np.concatenate(rb)

def plot_test_true_pred_curve(
    run_id: str,
    d_eval: str,
    idx_split: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefer_last_days: int = 7,
    use_h: int = 24,
) -> None:
    """
    画 test 集：t_dec 维度下某一 horizon(h) 的真实 vs 预测曲线
    idx_split 必须与 y_true/y_pred 的行顺序对齐，并包含: t_dec, date_tgt_max
    """
    if idx_split is None or len(idx_split) == 0:
        print("[PLOT] skip: empty idx_split", flush=True)
        return

    days = pd.Series(pd.to_datetime(idx_split["date_tgt_max"]).unique()).sort_values()
    if len(days) >= prefer_last_days:
        keep_days = set(days.iloc[-prefer_last_days:].tolist())
        mask = idx_split["date_tgt_max"].isin(keep_days).to_numpy()
        tag = f"last{prefer_last_days}d"
    else:
        mask = np.ones(len(idx_split), dtype=bool)
        tag = "alltest"

    h = int(use_h) - 1
    t = pd.to_datetime(idx_split.loc[mask, "t_dec"].values)
    yt = y_true[mask, h]
    yp = y_pred[mask, h]

    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() == 0:
        print(f"[PLOT] skip: no finite points (tag={tag})", flush=True)
        return

    plt.figure()
    plt.plot(t[m], yt[m], label="true")
    plt.plot(t[m], yp[m], label="pred")
    plt.legend()
    plt.xlabel("t_dec")
    plt.ylabel("RT price")
    plt.title(f"{run_id} | test | h{use_h:02d} {tag}")
    out_path = os.path.join(d_eval, f"{run_id}_test_h{use_h:02d}_{tag}.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print("[PLOT] saved:", out_path, flush=True)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", type=str, default="fit", choices=["fit", "predict"])

    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="outputs_rt_patchtst")
    ap.add_argument("--run_id", type=str, default="rt_onefile")

    ap.add_argument("--val_days", type=int, default=5)
    ap.add_argument("--test_days", type=int, default=6)
    ap.add_argument("--require_full_day", action="store_true")

    ap.add_argument("--cutoff_mode", type=str, default="rt_chain", choices=["rt_chain", "exog_only"])

    ap.add_argument("--future_vars", type=str, default="load,pv,line,ru,rd")
    ap.add_argument("--exog_driver_method", type=str, default="lag1d_residual", choices=["lag1d", "lag1d_residual"])
    ap.add_argument("--exog_max_iter", type=int, default=500)
    ap.add_argument("--exog_fallback_direct", action="store_true")

    ap.add_argument("--rt_method", type=str, default="lag1d_residual", choices=["lag1d", "lag1d_residual"])
    ap.add_argument("--use_future_da", action="store_true")

    ap.add_argument("--drop_invalid_samples", action="store_true")
    ap.add_argument("--hist_ratio_min", type=float, default=0.98)
    ap.add_argument("--fut_ratio_min", type=float, default=0.98)

    # model
    ap.add_argument("--patch_len", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    # optim
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # predict mode
    ap.add_argument("--ckpt_path", type=str, default="")
    ap.add_argument("--asof", type=str, default="", help="asof t_dec, e.g. '2025-12-15 23:45:00' (15min aligned)")
    ap.add_argument("--out_csv", type=str, default="")

    # eval plot options
    ap.add_argument("--plot_test_curve", action="store_true", help="If set, save test true vs pred curve.")
    ap.add_argument("--plot_test_h", type=int, default=24, help="Which horizon step to plot for test curve (1..24).")
    ap.add_argument("--plot_last_days", type=int, default=7, help="Prefer last K days for test curve; fallback to alltest.")

    args = ap.parse_args()
    set_seed(SEED)

    out_root = os.path.join(args.outdir, args.run_id)
    d_meta = os.path.join(out_root, "meta")
    d_ckpt = os.path.join(out_root, "ckpt")
    d_eval = os.path.join(out_root, "eval")
    d_exog = os.path.join(out_root, "exog_pred")
    for d in [d_meta, d_ckpt, d_eval, d_exog]:
        ensure_dir(d)

    print(f"[LOAD] excel={args.data_path}", flush=True)
    df, colmap = load_excel(args.data_path)
    print(f"[DATA] shape={df.shape} time_range={df[colmap.time].min()} ~ {df[colmap.time].max()}", flush=True)
    print(f"[COLMAP] time={colmap.time} rt={colmap.rt} da={colmap.da}", flush=True)

    hist_cols = [colmap.rt, colmap.da, colmap.load, colmap.pv, colmap.line, colmap.ru, colmap.rd]
    for opt in [colmap.wind, colmap.wg_total, colmap.space, colmap.local_gen]:
        if opt is not None:
            hist_cols.append(opt)

    if args.cutoff_mode == "rt_chain":
        required = [colmap.rt, colmap.da, colmap.load, colmap.pv, colmap.line, colmap.ru, colmap.rd]
        for opt in [colmap.wind, colmap.wg_total, colmap.space, colmap.local_gen]:
            if opt is not None:
                required.append(opt)
    else:
        required = [colmap.load, colmap.pv, colmap.line, colmap.ru, colmap.rd]
        for opt in [colmap.wind, colmap.wg_total, colmap.space, colmap.local_gen]:
            if opt is not None:
                required.append(opt)

    cutoff_time, last_valid_map, missing_map = compute_cutoff_time(df, colmap.time, required)
    print(f"[CUTOFF] mode={args.cutoff_mode} cutoff_time={cutoff_time}", flush=True)
    if missing_map:
        print(f"[CUTOFF] missing/allnan: {missing_map}", flush=True)

    idx = make_index(df, colmap.time)
    before = len(idx)
    idx = idx[idx["tgt_max_time"] <= cutoff_time].reset_index(drop=True)
    print(f"[INDEX] raw={before} after_cutoff={len(idx)} dropped={before-len(idx)}", flush=True)

    idx, split_info = split_by_tgtmax_day(idx, args.val_days, args.test_days, args.require_full_day)
    print("[SPLIT] counts:\n", idx["split"].value_counts(), flush=True)
    print("[SPLIT] info:", split_info, flush=True)

    future_vars = [x.strip() for x in args.future_vars.split(",") if x.strip()]
    if not future_vars:
        raise ValueError("future_vars is empty.")

    print(f"[EXOG] Training internal exog forecaster: vars={future_vars} method={args.exog_driver_method} "
          f"max_iter={args.exog_max_iter} fallback_direct={args.exog_fallback_direct}", flush=True)
    exog_wide = build_exog_wide(
        df=df, idx=idx, colmap=colmap,
        future_vars=future_vars,
        method=args.exog_driver_method,
        max_iter=args.exog_max_iter,
        fallback_direct=args.exog_fallback_direct,
    )
    exog_path = os.path.join(d_exog, f"exog_pred_wide_{args.run_id}.parquet")
    exog_wide.to_parquet(exog_path, index=False)
    print(f"[EXOG] saved wide: {exog_path}", flush=True)

    rtbase_wide = build_rt_baseline_wide(
        df=df, idx=idx, colmap=colmap,
        rt_method=args.rt_method,
        use_future_da=args.use_future_da,
    )

    t_train_max = idx.loc[idx["split"] == "train", "t_dec"].max()
    train_mask = (df[colmap.time] <= t_train_max).to_numpy()

    hist_full = np.stack([safe_to_numeric(df[c]) for c in hist_cols], axis=1)
    scaler_hist = fit_scaler_from_train_rows(hist_full[train_mask, :])

    var_to_col = {
        "load": colmap.load, "pv": colmap.pv, "line": colmap.line, "ru": colmap.ru, "rd": colmap.rd,
        "wind": colmap.wind, "wg_total": colmap.wg_total, "space": colmap.space, "local_gen": colmap.local_gen
    }
    scaler_future: Dict[str, Tuple[float, float]] = {}
    for v in future_vars:
        col = var_to_col.get(v, None)
        if col is None or col not in df.columns:
            raise ValueError(f"[SCALER] no actual column for future var={v}")
        arr = safe_to_numeric(df[col])[train_mask]
        mu = float(np.nanmean(arr))
        sd = float(np.nanstd(arr))
        sd = sd if (np.isfinite(sd) and sd > 1e-8) else 1.0
        mu = mu if np.isfinite(mu) else 0.0
        scaler_future[v] = (mu, sd)

    pos_map = pd.Series(np.arange(len(df), dtype=np.int64), index=df[colmap.time]).to_dict()
    dec_pos_train = idx[idx["split"] == "train"]["t_dec"].map(pos_map).astype(np.int64).values
    rt = safe_to_numeric(df[colmap.rt])

    rb_train = rtbase_wide.merge(idx[idx["split"] == "train"][["t_dec"]], on="t_dec", how="inner").set_index("t_dec")
    idx_train = idx[idx["split"] == "train"].reset_index(drop=True)
    rb_train = rb_train.loc[pd.to_datetime(idx_train["t_dec"])]

    deltas = []
    for h in range(1, HORIZON + 1):
        y = rt[dec_pos_train + h]
        b = rb_train[f"rtbase_h{h:02d}"].to_numpy(dtype=np.float32)
        m = np.isfinite(y) & np.isfinite(b)
        if m.sum() > 0:
            deltas.append((y[m] - b[m]).astype(np.float32))
    d_all = np.concatenate(deltas) if deltas else np.array([0.0], dtype=np.float32)
    mu_d = float(np.nanmean(d_all))
    sd_d = float(np.nanstd(d_all))
    sd_d = sd_d if (np.isfinite(sd_d) and sd_d > 1e-8) else 1.0
    mu_d = mu_d if np.isfinite(mu_d) else 0.0
    scaler_rt_delta = (mu_d, sd_d)

    ds_train = RtDataset(
        df=df, idx=idx, colmap=colmap,
        hist_cols=hist_cols,
        exog_wide=exog_wide,
        future_vars=future_vars,
        rtbase_wide=rtbase_wide,
        use_future_da=args.use_future_da,
        scaler_hist=scaler_hist,
        scaler_future=scaler_future,
        scaler_rt_delta=scaler_rt_delta,
        split="train",
        drop_invalid_samples=args.drop_invalid_samples,
        hist_ratio_min=args.hist_ratio_min,
        fut_ratio_min=args.fut_ratio_min,
    )
    ds_val = RtDataset(
        df=df, idx=idx, colmap=colmap,
        hist_cols=hist_cols,
        exog_wide=exog_wide,
        future_vars=future_vars,
        rtbase_wide=rtbase_wide,
        use_future_da=args.use_future_da,
        scaler_hist=scaler_hist,
        scaler_future=scaler_future,
        scaler_rt_delta=scaler_rt_delta,
        split="val",
        drop_invalid_samples=args.drop_invalid_samples,
        hist_ratio_min=args.hist_ratio_min,
        fut_ratio_min=args.fut_ratio_min,
    )
    ds_test = RtDataset(
        df=df, idx=idx, colmap=colmap,
        hist_cols=hist_cols,
        exog_wide=exog_wide,
        future_vars=future_vars,
        rtbase_wide=rtbase_wide,
        use_future_da=args.use_future_da,
        scaler_hist=scaler_hist,
        scaler_future=scaler_future,
        scaler_rt_delta=scaler_rt_delta,
        split="test",
        drop_invalid_samples=args.drop_invalid_samples,
        hist_ratio_min=args.hist_ratio_min,
        fut_ratio_min=args.fut_ratio_min,
    )

    device = torch.device(args.device)

    if args.mode == "predict":
        ckpt_path = args.ckpt_path if args.ckpt_path else os.path.join(d_ckpt, "best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

        model = PatchTSTResidual(
            c_hist=len(hist_cols),
            c_fut=len(future_vars) + (1 if args.use_future_da else 0),
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            patch_len=args.patch_len,
        ).to(device)

        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        model.eval()

        if args.asof:
            asof = pd.to_datetime(args.asof).floor("15min")
        else:
            asof = pd.to_datetime(idx["t_dec"].max())

        sub = idx[idx["t_dec"] == asof]
        if len(sub) == 0:
            raise ValueError(f"asof {asof} not found in eligible idx. Use one of idx['t_dec'] <= cutoff.")
        tmp_idx = sub.copy()
        tmp_idx["split"] = "test"
        tmp_ds = RtDataset(
            df=df, idx=tmp_idx, colmap=colmap,
            hist_cols=hist_cols,
            exog_wide=exog_wide,
            future_vars=future_vars,
            rtbase_wide=rtbase_wide,
            use_future_da=args.use_future_da,
            scaler_hist=scaler_hist,
            scaler_future=scaler_future,
            scaler_rt_delta=scaler_rt_delta,
            split="test",
            drop_invalid_samples=False,
            hist_ratio_min=0.0,
            fut_ratio_min=0.0,
        )
        b = tmp_ds[0]
        x_hist = b["x_hist"].unsqueeze(0).to(device)
        x_time = b["x_time"].unsqueeze(0).to(device)
        x_fut  = b["x_fut"].unsqueeze(0).to(device)
        rt_base = b["rt_base"].cpu().numpy()[None, :]

        with torch.no_grad():
            delta_n = model(x_hist, x_time, x_fut).cpu().numpy()
        delta = delta_n * scaler_rt_delta[1] + scaler_rt_delta[0]
        yhat = rt_base + delta

        ts = [asof + pd.Timedelta(minutes=STEP_MINUTES * h) for h in range(1, HORIZON + 1)]
        out = pd.DataFrame({"t": ts, "rt_pred": yhat[0].astype(float)})
        out_csv = args.out_csv if args.out_csv else os.path.join(
            d_eval, f"RT_PRED_{args.run_id}_{asof.strftime('%Y%m%d_%H%M')}.csv"
        )
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[PRED] saved: {out_csv}", flush=True)
        return

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PatchTSTResidual(
        c_hist=len(hist_cols),
        c_fut=len(future_vars) + (1 if args.use_future_da else 0),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        patch_len=args.patch_len,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_path = os.path.join(d_ckpt, "best.pt")
    patience = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for b in dl_train:
            x_hist = b["x_hist"].to(device)
            x_time = b["x_time"].to(device)
            x_fut  = b["x_fut"].to(device)
            y      = b["y"].to(device)

            yhat = model(x_hist, x_time, x_fut)
            loss = masked_huber(yhat, y, delta=2.0)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        va_losses = []
        with torch.no_grad():
            for b in dl_val:
                x_hist = b["x_hist"].to(device)
                x_time = b["x_time"].to(device)
                x_fut  = b["x_fut"].to(device)
                y      = b["y"].to(device)
                yhat = model(x_hist, x_time, x_fut)
                loss = masked_huber(yhat, y, delta=2.0)
                va_losses.append(float(loss.detach().cpu().item()))

        tr_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
        va_loss = float(np.mean(va_losses)) if va_losses else 0.0
        print(f"[EPOCH {ep:03d}] train_loss={tr_loss:.6f} val_loss={va_loss:.6f}", flush=True)

        if va_loss + 1e-8 < best_val:
            best_val = va_loss
            patience = 0
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "scaler_hist": {"mean": scaler_hist.mean.tolist(), "std": scaler_hist.std.tolist()},
                "scaler_future": scaler_future,
                "scaler_rt_delta": scaler_rt_delta,
                "hist_cols": hist_cols,
                "future_vars": future_vars,
                "use_future_da": args.use_future_da,
            }, best_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[EARLYSTOP] patience={args.patience} reached. best_val={best_val:.6f}", flush=True)
                break

    ck = torch.load(best_path, map_location=device)
    model.load_state_dict(ck["model"])
    print(f"[CKPT] loaded: {best_path} best_val={best_val:.6f}", flush=True)

    rows = []
    for split_name, dl in [("val", dl_val), ("test", dl_test)]:
        y_true_n, y_pred_n, rt_base = predict_all(model, dl, device)

        delta_true = y_true_n * scaler_rt_delta[1] + scaler_rt_delta[0]
        delta_pred = y_pred_n * scaler_rt_delta[1] + scaler_rt_delta[0]
        y_true = rt_base + delta_true
        y_pred = rt_base + delta_pred

        mae_by_h, rmse_by_h = [], []
        for h in range(HORIZON):
            mae_h, rmse_h = masked_np_metrics(y_true[:, h], y_pred[:, h])
            mae_by_h.append(mae_h)
            rmse_by_h.append(rmse_h)

        all_mae = float(np.nanmean(mae_by_h))
        all_rmse = float(np.nanmean(rmse_by_h))
        rows.append({
            "tag": args.run_id, "split": split_name, "target": "rt_price",
            "all_MAE": all_mae, "all_RMSE": all_rmse,
            "h24_MAE": float(mae_by_h[-1]), "h24_RMSE": float(rmse_by_h[-1]),
        })

        plt.figure()
        plt.plot(np.arange(1, HORIZON + 1), mae_by_h)
        plt.xlabel("horizon_step (1..24)")
        plt.ylabel("MAE")
        plt.title(f"{args.run_id} | {split_name} | RT | MAE_by_h")
        plt.savefig(os.path.join(d_eval, f"{args.run_id}_{split_name}_mae_by_h.png"), dpi=160, bbox_inches="tight")
        plt.close()

        # --------- 新增：test 真实 vs 预测曲线图（默认不开，需要参数 --plot_test_curve）---------
        if split_name == "test" and args.plot_test_curve:
            idx_test_plot = ds_test.idx.reset_index(drop=True)[["t_dec", "date_tgt_max"]].copy()
            plot_test_true_pred_curve(
                run_id=args.run_id,
                d_eval=d_eval,
                idx_split=idx_test_plot,
                y_true=y_true,
                y_pred=y_pred,
                prefer_last_days=args.plot_last_days,
                use_h=args.plot_test_h,
            )

    summary = pd.DataFrame(rows)
    all_path = os.path.join(d_eval, f"ALL_SUMMARY_{args.run_id}.csv")
    summary.to_csv(all_path, index=False, encoding="utf-8-sig")
    print("[EVAL] saved:", all_path, flush=True)

    meta = {
        "run_id": args.run_id,
        "data_path": os.path.abspath(args.data_path),
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "step_minutes": STEP_MINUTES,
        "cutoff": {
            "mode": args.cutoff_mode,
            "required_cols": required,
            "last_valid_time_by_col": last_valid_map,
            "missing_or_allnan": missing_map,
            "cutoff_time": str(cutoff_time),
            "filter_rule": "keep samples with tgt_max_time <= cutoff_time",
        },
        "split_rule": {
            "by": "date(tgt_max_time)",
            "val_days": args.val_days,
            "test_days": args.test_days,
            "require_full_day": args.require_full_day,
            "split_info": split_info,
        },
        "exog": {
            "future_vars": future_vars,
            "driver_method": args.exog_driver_method,
            "fallback_direct": args.exog_fallback_direct,
            "wide_path": os.path.abspath(exog_path),
        },
        "rt_baseline": {
            "rt_method": args.rt_method,
            "use_future_da": args.use_future_da,
            "scaler_rt_delta": {"mu": scaler_rt_delta[0], "sd": scaler_rt_delta[1]},
        },
        "model": {
            "patch_len": args.patch_len,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
        },
        "optim": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "device": str(device),
        },
        "outputs": {
            "best_ckpt": os.path.abspath(best_path),
            "eval_summary": os.path.abspath(all_path),
            "eval_dir": os.path.abspath(d_eval),
        }
    }
    meta_path = os.path.join(d_meta, f"meta_{args.run_id}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[META] saved:", meta_path, flush=True)
    print("[DONE] one-file RT pipeline finished.", flush=True)


if __name__ == "__main__":
    main()
