# exog_pipeline_rt6h_v2.py
# -*- coding: utf-8 -*-
"""
Exogenous forecasting pipeline (RT 6h = 24 steps ahead, 15min) for Shandong dataset.

Production constraint:
- Does NOT use any '*预测值' columns for training/inference output.
- '*预测值' columns, if present, are evaluated ONLY as a baseline for comparison.

Core upgrades in this version:
- Feature-availability-aware cutoff (critical): cutoff_time=min(last_valid of required columns),
  keep only samples with tgt_max_time <= cutoff_time.
- Strict split by date(tgt_max_time), optionally keep only full days (96 points/day).
- Driver (load/pv/line) method options (per variable):
    * direct: direct multi-step regression
    * lag1d: target-aligned lag1d baseline ONLY, y_hat(t,h)=y(t+h-96), no model training
    * lag1d_residual: target-aligned lag1d baseline + residual learning
        y_base(t,h) = y(t+h-96),  r(t,h)=y(t+h)-y_base(t,h),  y_hat=y_base+r_hat
    * pv_day_residual: (for pv only) lag1d baseline + residual ONLY on "daylight & base>eps"
        - Train residual model only on gated points
        - Inference: if gated -> y_base + r_hat else y_base
  This reduces over-correction for PV at h24 and night slots.

Modeling:
- load/pv/line: HistGradientBoostingRegressor per horizon if method requires training
- ru/rd: prior(month/weekend/slot) + conditional residual model using predicted drivers

Outputs:
  outdir/
    index/rt_seq24_index_{run_id}.parquet
    pred/exog_pred_rt6h_{run_id}_wide.parquet
    pred_long/exog_pred_rt6h_{run_id}_long_{var}.parquet (one file per var)
    meta/exog_pred_rt6h_{run_id}_meta.json
    eval/*_summary.csv + plots
    eval/all_metrics_{run_id}.csv   (CONSOLIDATED metrics, recommended)
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# -----------------------------
# Task constants
# -----------------------------
STEP_MINUTES = 15
HORIZON = 24          # 6h
LOOKBACK = 192        # 48h
DAY_STEPS = 96        # 24h / 15min = 96

# Prior bucketing for reserves
USE_MONTH_IN_PRIOR = True
PRIOR_STAT = "median"  # median is robust for spiky reserves

# HGBT params (defaults; can be overridden by CLI --max_iter)
HGB_PARAMS = dict(
    max_depth=6,
    learning_rate=0.05,
    max_iter=500,
    l2_regularization=0.1,
    random_state=42,
)

# Print frequency for horizon loops (reduce IO overhead)
PRINT_EVERY_H = 4

# Defaults
DEFAULT_OUTDIR = "outputs_exog_rt6h"
DEFAULT_DATA_PATH = "山东-全年-带时间点.xlsx"


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def dt_floor(ts: pd.Series, minutes: int = 15) -> pd.Series:
    return ts.dt.floor(f"{minutes}min")

def get_slot_id(ts: pd.Series) -> pd.Series:
    return (ts.dt.hour * (60 // STEP_MINUTES) + (ts.dt.minute // STEP_MINUTES)).astype(np.int16)

def safe_to_parquet(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        csv_path = path.replace(".parquet", ".csv")
        print(f"[WARN] Parquet failed ({e}); fallback to CSV: {csv_path}", flush=True)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

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

def add_time_features_for_tgt_time(tgt_time: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"tgt_time": tgt_time})
    d["tgt_hour"] = d["tgt_time"].dt.hour.astype(np.int16)
    d["tgt_minute"] = d["tgt_time"].dt.minute.astype(np.int16)
    d["tgt_weekday"] = d["tgt_time"].dt.weekday.astype(np.int16)
    d["tgt_is_weekend"] = (d["tgt_weekday"] >= 5).astype(np.int8)
    d["tgt_month"] = d["tgt_time"].dt.month.astype(np.int16)
    d["tgt_dayofyear"] = d["tgt_time"].dt.dayofyear.astype(np.int16)

    hod = d["tgt_hour"].astype(float) + d["tgt_minute"].astype(float) / 60.0
    d["tgt_sin_hod"] = np.sin(2 * np.pi * hod / 24.0)
    d["tgt_cos_hod"] = np.cos(2 * np.pi * hod / 24.0)

    doy = d["tgt_dayofyear"].astype(float)
    d["tgt_sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    d["tgt_cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    d["tgt_slot"] = get_slot_id(d["tgt_time"])
    return d.drop(columns=["tgt_time"])

def check_15min_continuity(t: pd.Series) -> None:
    dt_diff = t.diff().dropna()
    if not (dt_diff == pd.Timedelta(minutes=STEP_MINUTES)).all():
        bad = dt_diff[dt_diff != pd.Timedelta(minutes=STEP_MINUTES)]
        raise ValueError(
            f"Time is not strictly {STEP_MINUTES}min continuous. Found {len(bad)} irregular intervals.\n"
            f"Fix by reindexing to 15min grid before running this script."
        )

def should_print_h(h: int, horizon: int) -> bool:
    return (h == 1) or (h == horizon) or (h % PRINT_EVERY_H == 0)

def last_valid_timestamp(s: pd.Series) -> Optional[pd.Timestamp]:
    """Return last timestamp index where s is finite (non-NaN, non-Inf)."""
    if s is None:
        return None
    a = pd.to_numeric(s, errors="coerce").to_numpy()
    good = np.isfinite(a)
    if not good.any():
        return None
    return pd.Timestamp(s.index[np.where(good)[0][-1]])

def compute_feature_cutoff_time(
    df: pd.DataFrame,
    time_col: str,
    required_cols: List[str],
) -> Tuple[pd.Timestamp, Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    """
    Compute cutoff_time as min(last_valid_time per required col).
    Returns:
      cutoff_time
      last_valid_time_map (iso strings)
      missing_or_allnan_map (col -> reason)
    """
    missing = {}
    last_map: Dict[str, Optional[str]] = {}

    df2 = df[[time_col] + [c for c in required_cols if c in df.columns]].copy()
    df2 = df2.set_index(time_col)

    times = []
    for c in required_cols:
        if c not in df.columns:
            missing[c] = "missing_col"
            last_map[c] = None
            continue
        lv = last_valid_timestamp(df2[c])
        if lv is None:
            missing[c] = "all_nan_or_nonfinite"
            last_map[c] = None
            continue
        last_map[c] = lv.isoformat(sep=" ")
        times.append(lv)

    if len(times) == 0:
        raise ValueError(
            f"[CUTOFF] No valid required columns to compute cutoff.\n"
            f"Required cols: {required_cols}\n"
            f"Missing/AllNaN: {missing}"
        )

    cutoff = min(times)
    return cutoff, last_map, missing


# -----------------------------
# Column detection & data load
# -----------------------------
def detect_colmap(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
    col_time = infer_col(df, ["datetime", "时间", "日期时间"])
    col_rt   = infer_col(df, ["实时出清电价", "实时电价", "RT", "实时价格"])
    col_da   = infer_col(df, ["日前出清电价", "日前电价", "DA", "日前价格"])

    col_load = infer_col(df, ["系统负荷实际值"])
    col_pv   = infer_col(df, ["光伏实际值"])
    col_line = infer_col(df, ["联络线实际值"])
    col_ru   = infer_col(df, ["上旋备用实际值"])
    col_rd   = infer_col(df, ["下旋备用实际值"])

    missing = []
    for k, v in [("time", col_time), ("rt", col_rt), ("da", col_da),
                 ("load", col_load), ("pv", col_pv), ("line", col_line), ("ru", col_ru), ("rd", col_rd)]:
        if v is None:
            missing.append(k)
    if missing:
        raise ValueError(
            f"Required columns missing: {missing}\n"
            f"Available columns (first 80): {df.columns.tolist()[:80]}"
        )

    colmap = dict(
        time=col_time, rt=col_rt, da=col_da,
        load=col_load, pv=col_pv, line=col_line, ru=col_ru, rd=col_rd
    )

    compare_pred_cols = dict(
        load_pred=infer_col(df, ["系统负荷预测值"]),
        pv_pred=infer_col(df, ["光伏预测值"]),
        line_pred=infer_col(df, ["联络线预测值"]),
        ru_pred=infer_col(df, ["上旋备用预测值"]),
        rd_pred=infer_col(df, ["下旋备用预测值"]),
    )
    return colmap, compare_pred_cols

def load_data(data_path: str) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Optional[str]]]:
    df = pd.read_excel(data_path, engine="openpyxl")
    colmap, compare_pred_cols = detect_colmap(df)

    df[colmap["time"]] = pd.to_datetime(df[colmap["time"]], errors="coerce")
    df = df.dropna(subset=[colmap["time"]]).copy()
    df[colmap["time"]] = dt_floor(df[colmap["time"]], STEP_MINUTES)

    df = df.sort_values(colmap["time"]).drop_duplicates(colmap["time"], keep="last").reset_index(drop=True)

    check_15min_continuity(df[colmap["time"]])

    need_cols = [colmap[k] for k in ["rt", "da", "load", "pv", "line", "ru", "rd"]]
    nan_stats = df[need_cols].isna().sum().sort_values(ascending=False)
    print("[NaN] required cols NaN count:\n", nan_stats, flush=True)

    return df, colmap, compare_pred_cols


# -----------------------------
# Index construction (t_dec, tgt_max_time)
# -----------------------------
def make_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    n = len(df)
    valid_i = np.arange(LOOKBACK - 1, n - HORIZON, dtype=np.int64)
    t_dec = df[time_col].iloc[valid_i].reset_index(drop=True)

    tgt_max_time = t_dec + pd.to_timedelta(HORIZON * STEP_MINUTES, unit="m")
    date_tgt_max = pd.to_datetime(tgt_max_time.dt.date)

    idx = pd.DataFrame({
        "t_dec": t_dec,
        "tgt_max_time": tgt_max_time,
        "date_tgt_max": date_tgt_max,
    })
    return idx

def split_by_tgtmax_day(
    idx: pd.DataFrame,
    val_days: int,
    test_days: int,
    require_full_day: bool,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    day_counts = idx.groupby("date_tgt_max").size().sort_index()
    if require_full_day:
        eligible_days = day_counts[day_counts == 96].index
    else:
        eligible_days = day_counts.index

    eligible_days = pd.Series(eligible_days).sort_values().reset_index(drop=True)
    if len(eligible_days) < (val_days + test_days + 1):
        raise ValueError(
            f"[SPLIT] Not enough eligible days after cutoff/full-day filter.\n"
            f"eligible_days={len(eligible_days)}, need>={val_days + test_days + 1}\n"
            f"day_counts_tail:\n{day_counts.tail(14)}"
        )

    test_start = eligible_days.iloc[-test_days]
    val_start = eligible_days.iloc[-(test_days + val_days)]
    last_eligible_day = eligible_days.iloc[-1]

    idx = idx.copy()
    idx["split"] = "train"
    idx.loc[idx["date_tgt_max"] >= val_start, "split"] = "val"
    idx.loc[idx["date_tgt_max"] >= test_start, "split"] = "test"

    if require_full_day:
        keep = idx["date_tgt_max"].isin(set(eligible_days))
        idx = idx[keep].reset_index(drop=True)

    info = {
        "val_start": str(pd.Timestamp(val_start).date()),
        "test_start": str(pd.Timestamp(test_start).date()),
        "last_eligible_day": str(pd.Timestamp(last_eligible_day).date()),
        "require_full_day": str(require_full_day),
    }
    return idx, info


# -----------------------------
# Vectorized rolling feature engineering
# -----------------------------
def rolling_features_for_series(s: pd.Series, prefix: str) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    out[f"{prefix}_val"] = s

    def add_window(win: int, name: str):
        r = s.rolling(win, min_periods=win)
        out[f"{prefix}_{name}_mean"] = r.mean()
        out[f"{prefix}_{name}_std"]  = r.std(ddof=0)
        out[f"{prefix}_{name}_min"]  = r.min()
        out[f"{prefix}_{name}_max"]  = r.max()
        out[f"{prefix}_{name}_trend"] = s - s.shift(win - 1)

    add_window(LOOKBACK, "48h")
    add_window(96, "24h")
    add_window(24, "6h")
    add_window(8, "2h")

    # decision-time same-slot lags (history features)
    out[f"{prefix}_lag1d_same_slot"] = s.shift(96)
    out[f"{prefix}_lag2d_same_slot"] = s.shift(192)
    return out

def build_history_feature_matrix(
    df: pd.DataFrame,
    idx: pd.DataFrame,
    colmap: Dict[str, str],
    hist_vars: List[str],
    include_da_rt: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    time_col = colmap["time"]
    pos = pd.Series(np.arange(len(df), dtype=np.int64), index=df[time_col]).to_dict()
    dec_pos = idx["t_dec"].map(pos).astype(np.int64).values

    feat_series: Dict[str, pd.Series] = {}

    for v in hist_vars:
        s = pd.to_numeric(df[colmap[v]], errors="coerce").astype(float)
        feat_series.update(rolling_features_for_series(s, prefix=v))

    if include_da_rt:
        for v in ["da", "rt"]:
            s = pd.to_numeric(df[colmap[v]], errors="coerce").astype(float)
            feat_series.update(rolling_features_for_series(s, prefix=v))

    feature_names = list(feat_series.keys())
    cols = [feat_series[name].iloc[dec_pos].to_numpy(dtype=np.float32) for name in feature_names]
    X = np.stack(cols, axis=1).astype(np.float32, copy=False)
    return X, feature_names

def precompute_time_features(idx: pd.DataFrame) -> Tuple[Dict[int, np.ndarray], Dict[int, pd.Series], List[str]]:
    t_dec = idx["t_dec"]
    time_feat: Dict[int, np.ndarray] = {}
    tgt_time: Dict[int, pd.Series] = {}

    tmp_names = add_time_features_for_tgt_time(t_dec + pd.to_timedelta(1 * STEP_MINUTES, unit="m")).columns.tolist()

    for h in range(1, HORIZON + 1):
        tt = t_dec + pd.to_timedelta(h * STEP_MINUTES, unit="m")
        feat_df = add_time_features_for_tgt_time(tt)
        time_feat[h] = feat_df.to_numpy(dtype=np.float32)
        tgt_time[h] = tt
    return time_feat, tgt_time, tmp_names


# -----------------------------
# Model containers
# -----------------------------
@dataclass
class DriverModels:
    vars: List[str]
    base_feature_names: List[str]
    time_feature_names: List[str]
    driver_method_map: Dict[str, str]   # var -> method
    models: Dict[str, Dict[int, Optional[HistGradientBoostingRegressor]]]  # var -> h -> model or None


# -----------------------------
# Target extraction helpers
# -----------------------------
def _get_dec_pos(df: pd.DataFrame, idx: pd.DataFrame, time_col: str) -> np.ndarray:
    pos = pd.Series(np.arange(len(df), dtype=np.int64), index=df[time_col]).to_dict()
    return idx["t_dec"].map(pos).astype(np.int64).values

def extract_future_y(df: pd.DataFrame, idx: pd.DataFrame, colmap: Dict[str, str], y_key: str, h: int) -> np.ndarray:
    time_col = colmap["time"]
    dec_pos = _get_dec_pos(df, idx, time_col)
    series = pd.to_numeric(df[colmap[y_key]], errors="coerce").to_numpy(dtype=float)
    return series[dec_pos + h]

def extract_target_lag(df: pd.DataFrame, idx: pd.DataFrame, colmap: Dict[str, str], y_key: str, h: int, lag_steps: int) -> np.ndarray:
    """
    target-aligned lag baseline:
      y_base(t,h) = y(t+h-lag_steps)
    """
    time_col = colmap["time"]
    dec_pos = _get_dec_pos(df, idx, time_col)
    series = pd.to_numeric(df[colmap[y_key]], errors="coerce").to_numpy(dtype=float)
    tgt_pos = dec_pos + h
    base_pos = tgt_pos - lag_steps

    y_base = np.full(len(tgt_pos), np.nan, dtype=float)
    good = (base_pos >= 0) & (base_pos < len(series))
    y_base[good] = series[base_pos[good]]
    return y_base


# -----------------------------
# Driver methods parsing
# -----------------------------
def parse_driver_methods(s: str) -> Dict[str, str]:
    """
    Example: "load=lag1d_residual,pv=pv_day_residual,line=lag1d"
    """
    out: Dict[str, str] = {}
    if not s:
        return out
    parts = [x.strip() for x in s.split(",") if x.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"--driver_methods invalid token: {p}. Use var=method.")
        var, method = [x.strip() for x in p.split("=", 1)]
        out[var] = method
    return out

def validate_driver_methods(driver_method_map: Dict[str, str], driver_vars: List[str]) -> None:
    allowed = {"direct", "lag1d", "lag1d_residual", "pv_day_residual"}
    for v in driver_vars:
        if v not in driver_method_map:
            raise ValueError(f"driver_methods missing var={v}. Please specify for all drivers: {driver_vars}")
        m = driver_method_map[v]
        if m not in allowed:
            raise ValueError(f"Unknown method '{m}' for var={v}. Allowed={sorted(list(allowed))}")
        if m == "pv_day_residual" and v != "pv":
            raise ValueError("pv_day_residual is only allowed for pv.")


# -----------------------------
# Driver training & prediction (load/pv/line) with per-var methods
# -----------------------------
def train_driver_models(
    df: pd.DataFrame,
    idx: pd.DataFrame,
    colmap: Dict[str, str],
    driver_vars: List[str],
    outdir: str,
    run_id: str,
    hgb_params: Dict,
    driver_method_map: Dict[str, str],
    pv_day_start: int,
    pv_day_end: int,
    pv_gate_eps: float,
) -> DriverModels:
    ensure_dir(outdir)

    print(f"  [FEAT] Building history features for drivers={driver_vars}...", flush=True)
    X_base, base_names = build_history_feature_matrix(df, idx, colmap, hist_vars=driver_vars, include_da_rt=True)
    print(f"  [FEAT] Driver history features shape: {X_base.shape}", flush=True)

    print(f"  [FEAT] Precomputing time features...", flush=True)
    time_feat, tgt_time_by_h, time_names = precompute_time_features(idx)
    time_dim = time_feat[1].shape[1]
    print(f"  [FEAT] Time features done (dim={time_dim}).", flush=True)

    train_mask = (idx["split"].values == "train")

    n_samples, base_dim = X_base.shape
    X_buf = np.empty((n_samples, base_dim + time_dim), dtype=np.float32)
    X_buf[:, :base_dim] = X_base

    models: Dict[str, Dict[int, Optional[HistGradientBoostingRegressor]]] = {v: {} for v in driver_vars}

    print("[TRAIN][DRIVER] method map:", driver_method_map, flush=True)
    for v in driver_vars:
        method = driver_method_map[v]
        print(f"[TRAIN][DRIVER] var={v} method={method}", flush=True)

        # If lag1d only -> no training
        if method == "lag1d":
            for h in range(1, HORIZON + 1):
                models[v][h] = None
            continue

        for h in range(1, HORIZON + 1):
            X_buf[:, base_dim:] = time_feat[h]
            y_true = extract_future_y(df, idx, colmap, y_key=v, h=h)

            if method == "direct":
                y_target = y_true

                mask = train_mask & np.isfinite(y_target)

            elif method in {"lag1d_residual", "pv_day_residual"}:
                y_base = extract_target_lag(df, idx, colmap, y_key=v, h=h, lag_steps=DAY_STEPS)
                y_target = y_true - y_base

                mask = train_mask & np.isfinite(y_target)

                if method == "pv_day_residual":
                    # Gate by daylight & base magnitude (causal: uses tgt_time & lag1d base only)
                    tt = tgt_time_by_h[h]
                    hour = tt.dt.hour.to_numpy()
                    is_day = (hour >= pv_day_start) & (hour <= pv_day_end)
                    base_ok = np.isfinite(y_base) & (y_base > pv_gate_eps)
                    gate = is_day & base_ok
                    mask = mask & gate

            else:
                raise ValueError(f"Unknown driver method={method}")

            n_use = int(mask.sum())
            if should_print_h(h, HORIZON):
                print(f"  [TRAIN][DRIVER] h={h:02d}/{HORIZON} usable_train={n_use}/{int(train_mask.sum())}", flush=True)

            if n_use == 0:
                raise ValueError(
                    f"[TRAIN][DRIVER] var={v} h={h}: no usable training samples.\n"
                    f"Likely gating too strict (pv_day_residual) or target all NaN."
                )

            m = HistGradientBoostingRegressor(**hgb_params)
            m.fit(X_buf[mask], y_target[mask])
            models[v][h] = m

    meta = {
        "run_id": run_id,
        "stage": "drivers_multistep",
        "vars": driver_vars,
        "driver_method_map": driver_method_map,
        "pv_day_residual": {
            "day_start_hour": pv_day_start,
            "day_end_hour": pv_day_end,
            "gate_eps": pv_gate_eps,
            "note": "Train/apply residual only when (daylight & lag1d_base>eps). Others use lag1d_base only.",
        },
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "step_minutes": STEP_MINUTES,
        "day_steps": DAY_STEPS,
        "model": "HistGradientBoostingRegressor",
        "model_params": hgb_params,
        "base_feature_names": base_names,
        "time_feature_names": time_names,
        "note": "pure; targets filtered by finite on train split; no '*预测值' used.",
    }
    with open(os.path.join(outdir, f"drivers_{run_id}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return DriverModels(
        vars=driver_vars,
        base_feature_names=base_names,
        time_feature_names=time_names,
        driver_method_map=driver_method_map,
        models=models
    )

def predict_driver_multistep(
    df: pd.DataFrame,
    idx: pd.DataFrame,
    colmap: Dict[str, str],
    driver_models: DriverModels,
    pv_day_start: int,
    pv_day_end: int,
    pv_gate_eps: float,
) -> Dict[str, np.ndarray]:
    X_base, _ = build_history_feature_matrix(df, idx, colmap, hist_vars=driver_models.vars, include_da_rt=True)
    time_feat, tgt_time_by_h, _ = precompute_time_features(idx)

    n_samples, base_dim = X_base.shape
    time_dim = time_feat[1].shape[1]
    X_buf = np.empty((n_samples, base_dim + time_dim), dtype=np.float32)
    X_buf[:, :base_dim] = X_base

    pred: Dict[str, np.ndarray] = {}
    for v in driver_models.vars:
        method = driver_models.driver_method_map[v]
        yhat = np.full((n_samples, HORIZON), np.nan, dtype=np.float32)

        for h in range(1, HORIZON + 1):
            # baseline always available for lag-based methods
            if method in {"lag1d", "lag1d_residual", "pv_day_residual"}:
                y_base = extract_target_lag(df, idx, colmap, y_key=v, h=h, lag_steps=DAY_STEPS).astype(np.float32, copy=False)
            else:
                y_base = None

            if method == "lag1d":
                yhat[:, h - 1] = y_base
                continue

            # model-based
            X_buf[:, base_dim:] = time_feat[h]
            m = driver_models.models[v][h]
            if m is None:
                # safety fallback
                yhat[:, h - 1] = y_base if y_base is not None else np.nan
                continue

            out = m.predict(X_buf).astype(np.float32, copy=False)

            if method == "direct":
                yhat[:, h - 1] = out

            elif method == "lag1d_residual":
                # y_hat = y_base + residual_hat, but if base NaN -> keep NaN
                ok = np.isfinite(y_base)
                tmp = np.full(n_samples, np.nan, dtype=np.float32)
                tmp[ok] = y_base[ok] + out[ok]
                yhat[:, h - 1] = tmp

            elif method == "pv_day_residual":
                # Gate at inference using tgt_time and lag1d base only
                tt = tgt_time_by_h[h]
                hour = tt.dt.hour.to_numpy()
                is_day = (hour >= pv_day_start) & (hour <= pv_day_end)
                base_ok = np.isfinite(y_base) & (y_base > pv_gate_eps)
                gate = is_day & base_ok

                tmp = np.full(n_samples, np.nan, dtype=np.float32)
                # default: use base
                tmp[np.isfinite(y_base)] = y_base[np.isfinite(y_base)]
                # gated: apply correction
                tmp[gate] = y_base[gate] + out[gate]
                yhat[:, h - 1] = tmp

            else:
                raise ValueError(f"Unknown driver method={method}")

        pred[v] = yhat.astype(float)
    return pred


# -----------------------------
# Reserves: prior + conditional residual
# -----------------------------
def build_prior_map(
    df: pd.DataFrame,
    idx: pd.DataFrame,
    colmap: Dict[str, str],
    reserve_var: str,
) -> Tuple[Dict[Tuple[int, int, int], float], float]:
    train_idx = idx[idx["split"] == "train"].copy()
    if len(train_idx) == 0:
        raise ValueError("No train samples to build prior.")

    rows = []
    for h in range(1, HORIZON + 1):
        tgt_time = train_idx["t_dec"] + pd.to_timedelta(h * STEP_MINUTES, unit="m")
        y_true = extract_future_y(df, train_idx, colmap, y_key=reserve_var, h=h)
        good = np.isfinite(y_true)
        if good.sum() == 0:
            continue

        tt = tgt_time.iloc[good].reset_index(drop=True)
        yy = y_true[good]

        slot = get_slot_id(tt).to_numpy()
        is_weekend = (tt.dt.weekday >= 5).astype(np.int8).to_numpy()
        if USE_MONTH_IN_PRIOR:
            month = tt.dt.month.astype(np.int16).to_numpy()
            key = list(zip(month.tolist(), is_weekend.tolist(), slot.tolist()))
        else:
            key = list(zip(is_weekend.tolist(), slot.tolist()))

        rows.append(pd.DataFrame({"key": key, "y": yy}))

    if len(rows) == 0:
        raise ValueError(f"Prior build failed: all y_true are NaN for reserve_var={reserve_var}")

    d = pd.concat(rows, ignore_index=True)

    if PRIOR_STAT == "median":
        prior_series = d.groupby("key")["y"].median()
        global_prior = float(d["y"].median())
    else:
        prior_series = d.groupby("key")["y"].mean()
        global_prior = float(d["y"].median())

    prior_map = prior_series.to_dict()
    return prior_map, global_prior

def lookup_prior(
    tgt_time: pd.Series,
    prior_map: Dict,
    global_prior: float,
) -> np.ndarray:
    slot = get_slot_id(tgt_time).to_numpy()
    is_weekend = (tgt_time.dt.weekday >= 5).astype(np.int8).to_numpy()
    if USE_MONTH_IN_PRIOR:
        month = tgt_time.dt.month.astype(np.int16).to_numpy()
        keys = list(zip(month.tolist(), is_weekend.tolist(), slot.tolist()))
    else:
        keys = list(zip(is_weekend.tolist(), slot.tolist()))

    out = np.empty(len(tgt_time), dtype=float)
    for i, k in enumerate(keys):
        out[i] = float(prior_map.get(k, global_prior))
    return out

def train_reserve_models(
    df: pd.DataFrame,
    idx: pd.DataFrame,
    colmap: Dict[str, str],
    driver_pred: Dict[str, np.ndarray],
    reserve_vars: List[str],
    outdir: str,
    run_id: str,
    hgb_params: Dict,
) -> Tuple[Dict[str, Dict[int, HistGradientBoostingRegressor]], Dict[str, Tuple[Dict, float]]]:
    ensure_dir(outdir)

    print(f"  [FEAT] Building history features for reserves...", flush=True)
    X_base, base_names = build_history_feature_matrix(
        df, idx, colmap,
        hist_vars=["load", "pv", "line", "ru", "rd"],
        include_da_rt=True
    )
    print(f"  [FEAT] Reserve history features shape: {X_base.shape}", flush=True)

    print(f"  [FEAT] Precomputing time features...", flush=True)
    time_feat, tgt_time_by_h, time_names = precompute_time_features(idx)
    time_dim = time_feat[1].shape[1]
    print(f"  [FEAT] Time features done (dim={time_dim}).", flush=True)

    train_mask = (idx["split"].values == "train")

    priors: Dict[str, Tuple[Dict, float]] = {}
    for v in reserve_vars:
        print(f"    [PRIOR] Building prior for {v}...", flush=True)
        priors[v] = build_prior_map(df, idx, colmap, reserve_var=v)
        print(f"    [PRIOR] {v} done (prior_map size: {len(priors[v][0])})", flush=True)

    n_samples, base_dim = X_base.shape
    X_buf = np.empty((n_samples, base_dim + time_dim + 3), dtype=np.float32)
    X_buf[:, :base_dim] = X_base

    models: Dict[str, Dict[int, HistGradientBoostingRegressor]] = {v: {} for v in reserve_vars}

    for v in reserve_vars:
        prior_map, global_prior = priors[v]
        print(f"[TRAIN][RESERVE] var={v} (prior+cond residual)", flush=True)
        for h in range(1, HORIZON + 1):
            X_buf[:, base_dim:base_dim + time_dim] = time_feat[h]
            X_buf[:, base_dim + time_dim + 0] = driver_pred["load"][:, h - 1].astype(np.float32, copy=False)
            X_buf[:, base_dim + time_dim + 1] = driver_pred["pv"][:, h - 1].astype(np.float32, copy=False)
            X_buf[:, base_dim + time_dim + 2] = driver_pred["line"][:, h - 1].astype(np.float32, copy=False)

            y_true = extract_future_y(df, idx, colmap, y_key=v, h=h)
            prior = lookup_prior(tgt_time_by_h[h], prior_map, global_prior)
            y_res = (y_true - prior).astype(float)

            mask = train_mask & np.isfinite(y_res)
            n_use = int(mask.sum())

            if should_print_h(h, HORIZON):
                print(f"  [TRAIN][RESERVE] h={h:02d}/{HORIZON} usable_train={n_use}/{int(train_mask.sum())}", flush=True)

            if n_use == 0:
                raise ValueError(f"[TRAIN][RESERVE] var={v} h={h}: no usable training samples (y_res all NaN/Inf in train).")

            m = HistGradientBoostingRegressor(**hgb_params)
            m.fit(X_buf[mask], y_res[mask])
            models[v][h] = m

    meta = {
        "run_id": run_id,
        "stage": "reserves_prior_plus_cond",
        "reserve_vars": reserve_vars,
        "prior_bucket": {
            "use_month": USE_MONTH_IN_PRIOR,
            "stat": PRIOR_STAT,
            "keys": ["month", "is_weekend", "slot"] if USE_MONTH_IN_PRIOR else ["is_weekend", "slot"],
        },
        "cond_driver_features": ["load_hat", "pv_hat", "line_hat"],
        "base_feature_names": base_names,
        "time_feature_names": time_names,
        "model": "HistGradientBoostingRegressor",
        "model_params": hgb_params,
        "note": "y filtered by finite on train split; drivers are self-predicted; no '*预测值' used.",
    }
    with open(os.path.join(outdir, f"reserves_{run_id}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return models, priors

def predict_reserves(
    df: pd.DataFrame,
    idx: pd.DataFrame,
    colmap: Dict[str, str],
    driver_pred: Dict[str, np.ndarray],
    reserve_models: Dict[str, Dict[int, HistGradientBoostingRegressor]],
    priors: Dict[str, Tuple[Dict, float]],
    reserve_vars: List[str],
) -> Dict[str, np.ndarray]:
    X_base, _ = build_history_feature_matrix(
        df, idx, colmap,
        hist_vars=["load", "pv", "line", "ru", "rd"],
        include_da_rt=True
    )
    time_feat, tgt_time_by_h, _ = precompute_time_features(idx)

    n_samples, base_dim = X_base.shape
    time_dim = time_feat[1].shape[1]
    X_buf = np.empty((n_samples, base_dim + time_dim + 3), dtype=np.float32)
    X_buf[:, :base_dim] = X_base

    pred: Dict[str, np.ndarray] = {}
    for v in reserve_vars:
        prior_map, global_prior = priors[v]
        yhat = np.full((n_samples, HORIZON), np.nan, dtype=np.float32)

        for h in range(1, HORIZON + 1):
            prior = lookup_prior(tgt_time_by_h[h], prior_map, global_prior).astype(np.float32, copy=False)

            X_buf[:, base_dim:base_dim + time_dim] = time_feat[h]
            X_buf[:, base_dim + time_dim + 0] = driver_pred["load"][:, h - 1].astype(np.float32, copy=False)
            X_buf[:, base_dim + time_dim + 1] = driver_pred["pv"][:, h - 1].astype(np.float32, copy=False)
            X_buf[:, base_dim + time_dim + 2] = driver_pred["line"][:, h - 1].astype(np.float32, copy=False)

            res_hat = reserve_models[v][h].predict(X_buf).astype(np.float32, copy=False)
            yhat[:, h - 1] = prior + res_hat

        pred[v] = yhat.astype(float)
    return pred


# -----------------------------
# Baseline builders (comparison only / diagnostics)
# -----------------------------
def build_use_pred_baseline(
    df: pd.DataFrame,
    idx: pd.DataFrame,
    colmap: Dict[str, str],
    compare_pred_cols: Dict[str, Optional[str]],
) -> Dict[str, np.ndarray]:
    time_col = colmap["time"]
    dec_pos = _get_dec_pos(df, idx, time_col)

    mapping = {
        "load": compare_pred_cols.get("load_pred"),
        "pv": compare_pred_cols.get("pv_pred"),
        "line": compare_pred_cols.get("line_pred"),
        "ru": compare_pred_cols.get("ru_pred"),
        "rd": compare_pred_cols.get("rd_pred"),
    }

    out: Dict[str, np.ndarray] = {}
    n = len(idx)
    for v, c in mapping.items():
        if c is None:
            continue
        series = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        yhat = np.full((n, HORIZON), np.nan, dtype=float)
        for h in range(1, HORIZON + 1):
            yhat[:, h - 1] = series[dec_pos + h]
        out[v] = yhat
    return out

def build_target_lag1d_baseline(
    df: pd.DataFrame,
    idx: pd.DataFrame,
    colmap: Dict[str, str],
    vars_list: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    n = len(idx)
    for v in vars_list:
        yhat = np.full((n, HORIZON), np.nan, dtype=float)
        for h in range(1, HORIZON + 1):
            yhat[:, h - 1] = extract_target_lag(df, idx, colmap, y_key=v, h=h, lag_steps=DAY_STEPS)
        out[v] = yhat
    return out


# -----------------------------
# Evaluation & plotting (works on wide arrays)
# -----------------------------
def evaluate_wide(
    df: pd.DataFrame,
    idx: pd.DataFrame,
    colmap: Dict[str, str],
    pred_wide: Dict[str, np.ndarray],
    outdir: str,
    tag: str,
) -> pd.DataFrame:
    ensure_dir(outdir)

    time_col = colmap["time"]
    dec_pos_all = _get_dec_pos(df, idx, time_col)

    rows = []
    for split in ["val", "test"]:
        mask_split = (idx["split"].values == split)
        if mask_split.sum() == 0:
            continue
        dec_pos = dec_pos_all[mask_split]

        for v, yhat_all in pred_wide.items():
            if v not in colmap:
                continue

            mae_by_h = []
            rmse_by_h = []
            valid_by_h = []
            y_col = pd.to_numeric(df[colmap[v]], errors="coerce").to_numpy(dtype=float)

            for h in range(1, HORIZON + 1):
                y_true = y_col[dec_pos + h]
                y_hat = yhat_all[mask_split, h - 1]

                m = np.isfinite(y_true) & np.isfinite(y_hat)
                valid_by_h.append(int(m.sum()))
                if m.sum() == 0:
                    mae_by_h.append(np.nan)
                    rmse_by_h.append(np.nan)
                else:
                    mae_by_h.append(mean_absolute_error(y_true[m], y_hat[m]))
                    rmse_by_h.append(rmse(y_true[m], y_hat[m]))

            valid_all = int(np.sum(valid_by_h))
            valid_h24 = int(valid_by_h[-1])

            mae_valid = [x for x in mae_by_h if np.isfinite(x)]
            rmse_valid = [x for x in rmse_by_h if np.isfinite(x)]
            all_mae = float(np.mean(mae_valid)) if len(mae_valid) > 0 else np.nan
            all_rmse = float(np.mean(rmse_valid)) if len(rmse_valid) > 0 else np.nan
            h24_mae = float(mae_by_h[-1]) if valid_h24 > 0 and np.isfinite(mae_by_h[-1]) else np.nan
            h24_rmse = float(rmse_by_h[-1]) if valid_h24 > 0 and np.isfinite(rmse_by_h[-1]) else np.nan

            rows.append({
                "tag": tag,
                "split": split,
                "var": v,
                "all_MAE": all_mae,
                "all_RMSE": all_rmse,
                "h24_MAE": h24_mae,
                "h24_RMSE": h24_rmse,
                "valid_points_all": valid_all,
                "valid_points_h24": valid_h24,
                "no_valid_points": int(valid_all == 0),
            })

            # MAE_by_h plot only if there is some valid data
            if valid_all > 0:
                plt.figure()
                plt.plot(np.arange(1, HORIZON + 1), mae_by_h)
                plt.xlabel("horizon_step (1..24)")
                plt.ylabel("MAE")
                plt.title(f"{tag} | {split} | {v} | MAE_by_h")
                plt.savefig(os.path.join(outdir, f"{tag}_{split}_{v}_MAE_by_h.png"), dpi=160, bbox_inches="tight")
                plt.close()

            # last7d plot for test, h=24 only
            if split == "test" and valid_h24 > 0:
                idx_s = idx[mask_split].copy()
                days = pd.Series(idx_s["date_tgt_max"].unique()).sort_values()
                last_days = set(days.iloc[-7:].tolist()) if len(days) >= 7 else set(days.tolist())
                mask_last = mask_split & idx["date_tgt_max"].isin(last_days).values

                if mask_last.sum() > 0:
                    dec_pos_last = dec_pos_all[mask_last]
                    t_dec_last = idx.loc[mask_last, "t_dec"].values

                    y_true = y_col[dec_pos_last + HORIZON]
                    y_hat = yhat_all[mask_last, HORIZON - 1]
                    m = np.isfinite(y_true) & np.isfinite(y_hat)

                    if m.sum() > 0:
                        plt.figure()
                        plt.plot(t_dec_last[m], y_true[m], label="true")
                        plt.plot(t_dec_last[m], y_hat[m], label="pred")
                        plt.legend()
                        plt.xlabel("t_dec")
                        plt.ylabel(v)
                        plt.title(f"{tag} | test | {v} | h24 last7d (by tgt_max days)")
                        plt.savefig(os.path.join(outdir, f"{tag}_test_{v}_h24_last7d.png"), dpi=160, bbox_inches="tight")
                        plt.close()

    res = pd.DataFrame(rows)
    out_path = os.path.join(outdir, f"{tag}_summary.csv")
    res.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("[EVAL] saved:", out_path, flush=True)
    return res


# -----------------------------
# Output writers
# -----------------------------
def save_pred_wide(
    idx: pd.DataFrame,
    pred_wide: Dict[str, np.ndarray],
    out_path: str,
    run_id: str,
) -> None:
    cols = {
        "t_dec": idx["t_dec"].values,
        "split": idx["split"].values,
    }
    for v, arr in pred_wide.items():
        for h in range(1, HORIZON + 1):
            cols[f"{v}_h{h:02d}"] = arr[:, h - 1].astype(float)
    cols["model_version"] = [run_id] * len(idx)
    base = pd.DataFrame(cols)
    safe_to_parquet(base, out_path)

def save_pred_long_per_var(
    idx: pd.DataFrame,
    pred_wide: Dict[str, np.ndarray],
    outdir: str,
    run_id: str,
) -> List[str]:
    ensure_dir(outdir)
    t_dec = idx["t_dec"].values
    split = idx["split"].values

    files = []
    for v, arr in pred_wide.items():
        n = len(idx)
        t_dec_rep = np.repeat(t_dec, HORIZON)
        split_rep = np.repeat(split, HORIZON)
        h_rep = np.tile(np.arange(1, HORIZON + 1, dtype=np.int16), n)
        tgt_time = pd.to_datetime(t_dec_rep) + pd.to_timedelta(h_rep * STEP_MINUTES, unit="m")
        y_hat = arr.reshape(-1).astype(float)

        dfv = pd.DataFrame({
            "t_dec": pd.to_datetime(t_dec_rep),
            "h": h_rep,
            "tgt_time": tgt_time,
            "var": v,
            "y_hat": y_hat,
            "split": split_rep,
            "method": "pure_all",
            "model_version": run_id,
            "is_valid": np.isfinite(y_hat).astype(np.int8),
        })
        dfv.loc[dfv["is_valid"] == 0, "y_hat"] = np.nan

        path = os.path.join(outdir, f"exog_pred_rt6h_{run_id}_long_{v}.parquet")
        safe_to_parquet(dfv, path)
        files.append(path)
    return files


# -----------------------------
# Pipeline orchestration
# -----------------------------
def run_pipeline(
    data_path: str,
    outdir: str,
    run_id: str,
    val_days: int,
    test_days: int,
    max_iter: int,
    require_full_day: bool,
    cutoff_mode: str,
    driver_methods: str,
    pv_day_start: int,
    pv_day_end: int,
    pv_gate_eps: float,
) -> None:
    ensure_dir(outdir)

    hgb_params = dict(HGB_PARAMS)
    hgb_params["max_iter"] = int(max_iter)

    print(f"[START] Loading data from: {data_path}", flush=True)
    df, colmap, compare_pred_cols = load_data(data_path)

    print("[DATA] shape:", df.shape, flush=True)
    print("[COLMAP]", colmap, flush=True)
    print("[COMPARE_PRED_COLS]", compare_pred_cols, flush=True)

    time_col = colmap["time"]
    data_time_min = df[time_col].min()
    data_time_max = df[time_col].max()
    print(f"[TIME] min={data_time_min} max={data_time_max}", flush=True)

    # --- Feature-availability-aware cutoff ---
    if cutoff_mode == "rt_chain":
        required_cols = [
            colmap["rt"], colmap["da"],
            colmap["load"], colmap["pv"], colmap["line"],
            colmap["ru"], colmap["rd"],
        ]
    elif cutoff_mode == "exog_only":
        required_cols = [
            colmap["load"], colmap["pv"], colmap["line"],
            colmap["ru"], colmap["rd"],
        ]
    else:
        raise ValueError(f"Unknown cutoff_mode={cutoff_mode}. Use rt_chain or exog_only.")

    cutoff_time, last_valid_map, missing_map = compute_feature_cutoff_time(
        df=df,
        time_col=time_col,
        required_cols=required_cols,
    )
    print("[CUTOFF] mode=", cutoff_mode, flush=True)
    print("[CUTOFF] required_cols:", required_cols, flush=True)
    print("[CUTOFF] last_valid_time_by_col:", flush=True)
    for c in required_cols:
        print(f"  - {c}: {last_valid_map.get(c)}", flush=True)
    if missing_map:
        print("[CUTOFF] missing/allnan cols:", missing_map, flush=True)
    print(f"[CUTOFF] cutoff_time = {cutoff_time} (min over required cols)", flush=True)

    # --- Build raw index ---
    print("[INDEX] Building raw index...", flush=True)
    idx = make_index(df, time_col=time_col)
    print("[INDEX] raw samples:", len(idx), flush=True)

    # --- Apply cutoff: keep only tgt_max_time <= cutoff_time ---
    before = len(idx)
    idx = idx[idx["tgt_max_time"] <= cutoff_time].reset_index(drop=True)
    after = len(idx)
    print(f"[INDEX] after cutoff filter: {after} (dropped {before - after})", flush=True)

    if len(idx) == 0:
        raise ValueError("[INDEX] No samples remain after cutoff filter. Check required cols cutoff or data integrity.")

    # --- Split ---
    idx, split_info = split_by_tgtmax_day(
        idx=idx,
        val_days=val_days,
        test_days=test_days,
        require_full_day=require_full_day,
    )

    print("[SPLIT] require_full_day=", require_full_day, flush=True)
    print("[SPLIT] counts:\n", idx["split"].value_counts(), flush=True)
    print("[SPLIT] val date range:",
          idx.loc[idx["split"] == "val", "date_tgt_max"].min(), "~", idx.loc[idx["split"] == "val", "date_tgt_max"].max(), flush=True)
    print("[SPLIT] test date range:",
          idx.loc[idx["split"] == "test", "date_tgt_max"].min(), "~", idx.loc[idx["split"] == "test", "date_tgt_max"].max(), flush=True)
    print("[SPLIT] info:", split_info, flush=True)

    d_index = os.path.join(outdir, "index")
    d_models = os.path.join(outdir, "models")
    d_pred = os.path.join(outdir, "pred")
    d_pred_long = os.path.join(outdir, "pred_long")
    d_meta = os.path.join(outdir, "meta")
    d_eval = os.path.join(outdir, "eval")
    for d in [d_index, d_models, d_pred, d_pred_long, d_meta, d_eval]:
        ensure_dir(d)

    index_path = os.path.join(d_index, f"rt_seq24_index_{run_id}.parquet")
    safe_to_parquet(idx, index_path)
    print("[INDEX] saved:", index_path, flush=True)

    # --- Driver methods map ---
    driver_vars = ["load", "pv", "line"]
    driver_method_map = parse_driver_methods(driver_methods)
    validate_driver_methods(driver_method_map, driver_vars)

    # Stage-1: drivers
    print("\n[STAGE-1] Training driver models (load, pv, line)...", flush=True)
    driver_models = train_driver_models(
        df=df, idx=idx, colmap=colmap, driver_vars=driver_vars,
        outdir=d_models, run_id=run_id,
        hgb_params=hgb_params,
        driver_method_map=driver_method_map,
        pv_day_start=pv_day_start,
        pv_day_end=pv_day_end,
        pv_gate_eps=pv_gate_eps,
    )
    print("[STAGE-1] Predicting drivers...", flush=True)
    driver_pred = predict_driver_multistep(
        df=df, idx=idx, colmap=colmap, driver_models=driver_models,
        pv_day_start=pv_day_start, pv_day_end=pv_day_end, pv_gate_eps=pv_gate_eps
    )
    print("[STAGE-1] Done.", flush=True)

    # Stage-2: reserves
    print("\n[STAGE-2] Training reserve models (ru, rd)...", flush=True)
    reserve_vars = ["ru", "rd"]
    reserve_models, priors = train_reserve_models(df, idx, colmap, driver_pred, reserve_vars, d_models, run_id, hgb_params)
    print("[STAGE-2] Predicting reserves...", flush=True)
    reserve_pred = predict_reserves(df, idx, colmap, driver_pred, reserve_models, priors, reserve_vars)
    print("[STAGE-2] Done.", flush=True)

    # Merge preds
    pred_wide: Dict[str, np.ndarray] = {}
    pred_wide.update(driver_pred)
    pred_wide.update(reserve_pred)

    # Save wide + long
    print("\n[OUTPUT] Saving predictions...", flush=True)
    wide_path = os.path.join(d_pred, f"exog_pred_rt6h_{run_id}_wide.parquet")
    save_pred_wide(idx, pred_wide, wide_path, run_id)
    print("[PRED] saved wide:", wide_path, flush=True)

    print(f"[PRED] long rows per var ~= n_samples*{HORIZON} = {len(idx)}*{HORIZON} = {len(idx)*HORIZON}", flush=True)
    long_files = save_pred_long_per_var(idx, pred_wide, d_pred_long, run_id)
    print("[PRED] saved long per var:", len(long_files), "files", flush=True)

    # Evaluate (collect all into one consolidated file)
    print("\n[EVAL] Evaluating predictions (pure)...", flush=True)
    all_metrics = []
    all_metrics.append(evaluate_wide(df, idx, colmap, pred_wide, d_eval, tag=f"pure_all_{run_id}"))

    print("\n[EVAL] Evaluating target-lag1d baseline (drivers)...", flush=True)
    lag1d_base = build_target_lag1d_baseline(df, idx, colmap, vars_list=["load", "pv", "line"])
    all_metrics.append(evaluate_wide(df, idx, colmap, lag1d_base, d_eval, tag=f"target_lag1d_baseline_{run_id}"))

    use_pred = build_use_pred_baseline(df, idx, colmap, compare_pred_cols)
    if len(use_pred) > 0:
        print("\n[EVAL] Evaluating use_pred baseline (comparison only)...", flush=True)
        all_metrics.append(evaluate_wide(df, idx, colmap, use_pred, d_eval, tag=f"use_pred_baseline_{run_id}"))
    else:
        print("[INFO] no '*预测值' columns found for baseline evaluation.", flush=True)

    # Consolidated metrics file
    all_df = pd.concat(all_metrics, ignore_index=True) if len(all_metrics) > 0 else pd.DataFrame()
    all_path = os.path.join(d_eval, f"all_metrics_{run_id}.csv")
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    print("[EVAL] consolidated saved:", all_path, flush=True)

    # meta
    meta = {
        "run_id": run_id,
        "data_path": os.path.abspath(data_path),
        "time_range": {"min": str(data_time_min), "max": str(data_time_max)},
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "step_minutes": STEP_MINUTES,
        "cutoff": {
            "mode": cutoff_mode,
            "required_cols": required_cols,
            "last_valid_time_by_col": last_valid_map,
            "missing_or_allnan": missing_map,
            "cutoff_time": str(cutoff_time),
            "filter_rule": "keep samples with tgt_max_time <= cutoff_time",
        },
        "split_rule": {
            "by": "date(tgt_max_time)",
            "val_days": val_days,
            "test_days": test_days,
            "require_full_day": require_full_day,
            "split_info": split_info,
        },
        "driver_method_map": driver_method_map,
        "pv_day_residual": {
            "day_start_hour": pv_day_start,
            "day_end_hour": pv_day_end,
            "gate_eps": pv_gate_eps,
        },
        "vars": ["load", "pv", "line", "ru", "rd"],
        "production_chain_uses_pred_cols": False,
        "optional_compare_pred_cols": compare_pred_cols,
        "hgb_params": hgb_params,
        "outputs": {
            "index": os.path.abspath(index_path),
            "pred_wide": os.path.abspath(wide_path),
            "pred_long_files": [os.path.abspath(p) for p in long_files],
            "eval_dir": os.path.abspath(d_eval),
            "eval_consolidated": os.path.abspath(all_path),
        },
        "notes": [
            "Cutoff is feature-availability-aware (min last-valid of required columns).",
            "Training filters NaN/Inf target on train split per var/horizon.",
            "HGBT can handle NaN in X; we do not impute X.",
            "pv_day_residual uses only tgt_time + lag1d_base to gate residual learning/application (no forecast columns).",
            "Evaluation includes valid_points counts; metrics are NaN if no valid points.",
        ],
    }
    meta_path = os.path.join(d_meta, f"exog_pred_rt6h_{run_id}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[META] saved:", meta_path, flush=True)

    print("\n[DONE] pipeline finished.", flush=True)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    ap.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    ap.add_argument("--run_id", type=str, default="pure_all_v2")
    ap.add_argument("--val_days", type=int, default=5)
    ap.add_argument("--test_days", type=int, default=6)
    ap.add_argument("--max_iter", type=int, default=HGB_PARAMS["max_iter"], help="override HGBT max_iter (default=500)")

    ap.add_argument("--require_full_day", action="store_true",
                    help="keep only days with 96 samples/day (recommended for clean val/test)")
    ap.add_argument("--cutoff_mode", type=str, default="rt_chain", choices=["rt_chain", "exog_only"],
                    help="rt_chain: cutoff by RT+DA+5 exogs (recommended); exog_only: cutoff by 5 exogs only.")

    ap.add_argument(
        "--driver_methods", type=str, required=True,
        help="Per-var driver methods, e.g. 'load=lag1d_residual,pv=pv_day_residual,line=lag1d'. "
             "Allowed methods: direct, lag1d, lag1d_residual, pv_day_residual (pv only)."
    )

    ap.add_argument("--pv_day_start", type=int, default=6, help="pv_day_residual: daylight start hour (inclusive)")
    ap.add_argument("--pv_day_end", type=int, default=18, help="pv_day_residual: daylight end hour (inclusive)")
    ap.add_argument("--pv_gate_eps", type=float, default=1.0,
                    help="pv_day_residual: apply residual only when lag1d_base > eps")

    args = ap.parse_args()

    run_pipeline(
        data_path=args.data_path.strip(),
        outdir=args.outdir,
        run_id=args.run_id,
        val_days=args.val_days,
        test_days=args.test_days,
        max_iter=args.max_iter,
        require_full_day=args.require_full_day,
        cutoff_mode=args.cutoff_mode,
        driver_methods=args.driver_methods,
        pv_day_start=args.pv_day_start,
        pv_day_end=args.pv_day_end,
        pv_gate_eps=args.pv_gate_eps,
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] User interrupted the script.", flush=True)
        raise
    except Exception:
        print(f"\n[ERROR] Script failed with error:", flush=True)
        import traceback
        traceback.print_exc()
        raise
