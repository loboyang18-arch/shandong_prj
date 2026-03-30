# -*- coding: utf-8 -*-
"""
日前电价预测 Baseline v3（分时段分模型，参考 tenghe1 v3.2）
=====================================================

核心改进点（相对于 v2 单模型）：
- 三段分时段建模：0-8、8-16、16-24 分别训练一个模型
- 增加 D-1 同时刻的实际日前价格特征：dminus1_same_time_price
- 增加周周期特征：dminus7_da_mean（若缺失则回退为 D-1 截止过去 7 日滚动均值）
- 晚高峰（16-24）对高价样本做温和加权（训练集分位数阈值以上）

注意：
- 本脚本仍保持“仅使用 D-1 09:00 及之前可得信息”的约束（价格与实时价格部分）。
- 外生变量默认优先使用“预测值”列；若缺失可回退到“实际值”（会提示风险）。
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ===== 中文显示设置 =====
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


STEP_MINUTES = 15
DAY_STEPS = 96
CUTOFF_TIME = time(9, 0)

DEFAULT_DATA_PATH = "山东-全年-带时间点.xlsx"
DEFAULT_OUTDIR = "outputs_da_price"
DEFAULT_RUN_ID = "da_baseline_v3_segmented"


@dataclass
class ColMap:
    time: str
    rt: str
    da: str
    da_fcst: Optional[str] = None
    da_price_fcst: Optional[str] = None

    load: Optional[str] = None
    pv: Optional[str] = None
    line: Optional[str] = None
    ru: Optional[str] = None
    rd: Optional[str] = None

    load_fcst: Optional[str] = None
    pv_fcst: Optional[str] = None
    line_fcst: Optional[str] = None
    ru_fcst: Optional[str] = None
    rd_fcst: Optional[str] = None


def infer_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = df.columns.tolist()
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        hits = [x for x in cols if c in str(x)]
        if len(hits) == 1:
            return hits[0]
    return None


def detect_colmap(df: pd.DataFrame) -> ColMap:
    time_col = infer_col(df, ["datetime", "时间", "日期时间"])
    rt_col = infer_col(df, ["实时出清电价", "实时电价", "RT", "实时价格", "实际实时价格"])
    da_col = infer_col(df, ["日前出清电价", "日前电价", "DA", "日前价格", "实际日前价格"])

    if time_col is None or rt_col is None or da_col is None:
        raise ValueError("未找到必需列：时间列、实时电价列、实际日前电价列。请检查表头。")

    da_fcst = infer_col(df, ["日前电价预测值", "日前价格预测值", "日前电价预测", "日前价格预测", "DA预测", "日前预测电价"])
    da_price_fcst = infer_col(df, ["日前价格预测"])
    if da_price_fcst == da_fcst:
        da_price_fcst = None

    load = infer_col(df, ["系统负荷实际值", "系统负荷"])
    pv = infer_col(df, ["光伏实际值", "光伏"])
    line = infer_col(df, ["联络线实际值", "联络线"])
    ru = infer_col(df, ["上旋备用实际值", "上旋备用"])
    rd = infer_col(df, ["下旋备用实际值", "下旋备用"])

    load_fcst = infer_col(df, ["系统负荷预测值", "负荷预测值", "负荷预测"])
    pv_fcst = infer_col(df, ["光伏预测值", "光伏预测"])
    line_fcst = infer_col(df, ["联络线预测值", "联络线预测"])
    ru_fcst = infer_col(df, ["上旋备用预测值", "上旋备用预测"])
    rd_fcst = infer_col(df, ["下旋备用预测值", "下旋备用预测"])

    return ColMap(
        time=time_col,
        rt=rt_col,
        da=da_col,
        da_fcst=da_fcst,
        da_price_fcst=da_price_fcst,
        load=load,
        pv=pv,
        line=line,
        ru=ru,
        rd=rd,
        load_fcst=load_fcst,
        pv_fcst=pv_fcst,
        line_fcst=line_fcst,
        ru_fcst=ru_fcst,
        rd_fcst=rd_fcst,
    )


def load_excel(data_path: str) -> Tuple[pd.DataFrame, ColMap]:
    df = pd.read_excel(data_path, engine="openpyxl", sheet_name=0)
    colmap = detect_colmap(df)
    df[colmap.time] = pd.to_datetime(df[colmap.time], errors="coerce")
    df = df.dropna(subset=[colmap.time]).copy()
    df[colmap.time] = df[colmap.time].dt.floor(f"{STEP_MINUTES}min")
    df = df.sort_values(colmap.time).drop_duplicates(colmap.time, keep="last").reset_index(drop=True)
    return df, colmap


def add_time_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    df["hour"] = df[dt_col].dt.hour
    df["minute"] = df[dt_col].dt.minute
    df["weekday"] = df[dt_col].dt.weekday
    df["month"] = df[dt_col].dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def safe_to_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def safe_window_stats(series: pd.Series, window_hours: float) -> Tuple[float, float]:
    if series.empty:
        return np.nan, np.nan
    points = int(window_hours * 60 / STEP_MINUTES)
    if points <= 0:
        return np.nan, np.nan
    recent = series.iloc[-points:]
    return float(recent.mean()), float(recent.std(ddof=0))


def pick_exog(name: str, forecast_col: Optional[str], actual_col: Optional[str], df: pd.DataFrame) -> Optional[str]:
    if forecast_col and (forecast_col in df.columns):
        return forecast_col
    if actual_col and (actual_col in df.columns):
        return actual_col
    return None


def build_samples(df: pd.DataFrame, colmap: ColMap) -> Tuple[pd.DataFrame, np.ndarray, List[str], pd.DataFrame]:
    df = df.copy()
    df = add_time_features(df, colmap.time)
    df["date"] = df[colmap.time].dt.date

    df[colmap.da] = safe_to_float(df[colmap.da])
    df[colmap.rt] = safe_to_float(df[colmap.rt])
    if colmap.da_fcst and colmap.da_fcst in df.columns:
        df[colmap.da_fcst] = safe_to_float(df[colmap.da_fcst])
    if colmap.da_price_fcst and colmap.da_price_fcst in df.columns:
        df[colmap.da_price_fcst] = safe_to_float(df[colmap.da_price_fcst])

    exog_defs = [
        ("load", colmap.load_fcst, colmap.load),
        ("pv", colmap.pv_fcst, colmap.pv),
        ("line", colmap.line_fcst, colmap.line),
        ("ru", colmap.ru_fcst, colmap.ru),
        ("rd", colmap.rd_fcst, colmap.rd),
    ]

    exog_cols: List[Tuple[str, str]] = []
    exog_used: Dict[str, str] = {}
    for name, fcst, act in exog_defs:
        chosen = pick_exog(name, fcst, act, df)
        if chosen:
            df[chosen] = safe_to_float(df[chosen])
            exog_cols.append((name, chosen))
            exog_used[name] = chosen

    if exog_used:
        used_str = ", ".join([f"{k}={v}" for k, v in exog_used.items()])
        print(f"[外生变量] 使用列（优先预测值）: {used_str}")

    # 按日期汇总的日均日前价格
    daily_da_mean = df.groupby("date")[colmap.da].mean().sort_index()
    daily_da_mean_3d = daily_da_mean.rolling(window=3, min_periods=1).mean()
    daily_da_mean_7d = daily_da_mean.rolling(window=7, min_periods=1).mean()

    # 为 dminus1_same_time_price 做映射：datetime -> 当时刻的日前价格
    da_map = pd.Series(df[colmap.da].values, index=pd.to_datetime(df[colmap.time])).to_dict()

    df_rt = df[[colmap.time, "date", colmap.rt]].copy()

    feature_rows = []
    target_list = []
    meta_rows = []

    all_dates = sorted(df["date"].unique())
    for d in all_dates:
        d_prev = d - timedelta(days=1)
        if d_prev not in daily_da_mean.index:
            continue

        mask_d = df["date"] == d
        df_d = df.loc[mask_d].copy()
        if df_d.empty:
            continue

        mask_prev = df["date"] == d_prev
        df_prev = df.loc[mask_prev].copy()
        if df_prev.empty:
            continue

        da_prev = df_prev[colmap.da]

        # D-1 全日统计
        dminus1_da_mean = float(np.nanmean(da_prev))
        dminus1_da_max = float(np.nanmax(da_prev))
        dminus1_da_min = float(np.nanmin(da_prev))
        dminus1_da_std = float(pd.Series(da_prev).std(ddof=0))

        # D-1 四时段统计
        def block_stats(sub: pd.DataFrame) -> Tuple[float, float]:
            if sub.empty:
                return np.nan, np.nan
            s = sub[colmap.da]
            return float(np.nanmean(s)), float(pd.Series(s).std(ddof=0))

        b0_mean, b0_std = block_stats(df_prev[(df_prev["hour"] >= 0) & (df_prev["hour"] < 6)])
        b1_mean, b1_std = block_stats(df_prev[(df_prev["hour"] >= 6) & (df_prev["hour"] < 12)])
        b2_mean, b2_std = block_stats(df_prev[(df_prev["hour"] >= 12) & (df_prev["hour"] < 18)])
        b3_mean, b3_std = block_stats(df_prev[(df_prev["hour"] >= 18) & (df_prev["hour"] < 24)])

        # 实时价格：截至 D-1 09:00
        cutoff_dt = datetime.combine(d_prev, CUTOFF_TIME)
        rt_hist = df_rt[df_rt[colmap.time] <= cutoff_dt][colmap.rt]
        rt_last = float(rt_hist.iloc[-1]) if len(rt_hist) else np.nan
        rt_mean_24, rt_std_24 = safe_window_stats(rt_hist, 24.0)
        rt_mean_6, _ = safe_window_stats(rt_hist, 6.0)
        rt_trend_6 = float(rt_last - rt_mean_6) if (np.isfinite(rt_last) and np.isfinite(rt_mean_6)) else np.nan

        # 多日滚动（日均价）
        da_mean_prev_1d = float(daily_da_mean.get(d_prev, np.nan))
        da_mean_prev_3d = float(daily_da_mean_3d.get(d_prev, np.nan))
        da_mean_prev_7d = float(daily_da_mean_7d.get(d_prev, np.nan))

        # 周周期：D-7 日均价（若缺失回退）
        d_minus7 = d - timedelta(days=7)
        dminus7_da_mean = daily_da_mean.get(d_minus7, np.nan)
        if not np.isfinite(dminus7_da_mean):
            dminus7_da_mean = da_mean_prev_7d
        dminus7_da_mean = float(dminus7_da_mean)

        for _, row in df_d.iterrows():
            feat = {}
            feat["hour"] = row["hour"]
            feat["minute"] = row["minute"]
            feat["weekday"] = row["weekday"]
            feat["month"] = row["month"]
            feat["is_weekend"] = row["is_weekend"]
            feat["hour_sin"] = row["hour_sin"]
            feat["hour_cos"] = row["hour_cos"]
            feat["weekday_sin"] = row["weekday_sin"]
            feat["weekday_cos"] = row["weekday_cos"]
            feat["month_sin"] = row["month_sin"]
            feat["month_cos"] = row["month_cos"]

            feat["dminus1_da_mean"] = dminus1_da_mean
            feat["dminus1_da_max"] = dminus1_da_max
            feat["dminus1_da_min"] = dminus1_da_min
            feat["dminus1_da_std"] = dminus1_da_std
            feat["dminus1_block0_6_mean"] = b0_mean
            feat["dminus1_block0_6_std"] = b0_std
            feat["dminus1_block6_12_mean"] = b1_mean
            feat["dminus1_block6_12_std"] = b1_std
            feat["dminus1_block12_18_mean"] = b2_mean
            feat["dminus1_block12_18_std"] = b2_std
            feat["dminus1_block18_24_mean"] = b3_mean
            feat["dminus1_block18_24_std"] = b3_std

            feat["da_mean_prev_1d"] = da_mean_prev_1d
            feat["da_mean_prev_3d"] = da_mean_prev_3d
            feat["da_mean_prev_7d"] = da_mean_prev_7d
            feat["dminus7_da_mean"] = dminus7_da_mean

            # D-1 同时刻价格
            dt_target = pd.to_datetime(row[colmap.time])
            dt_dminus1 = dt_target - timedelta(days=1)
            feat["dminus1_same_time_price"] = float(da_map.get(dt_dminus1, np.nan))

            feat["rt_last_before_cutoff"] = rt_last
            feat["rt_mean_24h_before_cutoff"] = rt_mean_24
            feat["rt_std_24h_before_cutoff"] = rt_std_24
            feat["rt_trend_6h_before_cutoff"] = rt_trend_6

            for name, col in exog_cols:
                feat[f"exog_{name}"] = row[col]

            feature_rows.append(feat)
            target_list.append(row[colmap.da])
            meta_rows.append({
                "datetime": dt_target,
                "da_fcst": float(row[colmap.da_fcst]) if (colmap.da_fcst and colmap.da_fcst in df.columns) else np.nan,
                "da_price_fcst": float(row[colmap.da_price_fcst]) if (colmap.da_price_fcst and colmap.da_price_fcst in df.columns) else np.nan,
            })

    df_model = pd.DataFrame(feature_rows)
    y = np.array(target_list, dtype=float)
    df_meta = pd.DataFrame(meta_rows)

    valid = np.isfinite(y)
    df_model = df_model.loc[valid].reset_index(drop=True)
    df_meta = df_meta.loc[valid].reset_index(drop=True)
    y = y[valid]

    feature_cols = df_model.columns.tolist()
    return df_model, y, feature_cols, df_meta


def _score(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"[{name}] 平均绝对误差={mae:.2f}  均方根误差={rmse:.2f}")
    return mae, rmse


def train_and_evaluate(
    df_model: pd.DataFrame,
    y: np.ndarray,
    feature_cols: List[str],
    df_meta: pd.DataFrame,
    val_days: int,
    test_days: int,
    outdir: str,
    run_id: str,
    high_price_quantile: float,
    high_price_weight: float,
):
    os.makedirs(outdir, exist_ok=True)
    d_eval = os.path.join(outdir, run_id, "eval")
    os.makedirs(d_eval, exist_ok=True)

    n_samples = len(df_model)
    approx_days = n_samples // DAY_STEPS

    n_test = test_days * DAY_STEPS
    n_val = val_days * DAY_STEPS
    if n_samples < (n_test + n_val + DAY_STEPS):
        raise ValueError("数据量不足，无法按天切分进行评估。")

    n_train = n_samples - n_val - n_test
    X_train = df_model.iloc[:n_train][feature_cols].values
    y_train = y[:n_train]
    X_val = df_model.iloc[n_train:n_train + n_val][feature_cols].values
    y_val = y[n_train:n_train + n_val]
    X_test = df_model.iloc[n_train + n_val:][feature_cols].values
    y_test = y[n_train + n_val:]

    df_train = df_model.iloc[:n_train].reset_index(drop=True)
    df_val = df_model.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df_model.iloc[n_train + n_val:].reset_index(drop=True)
    meta_test = df_meta.iloc[n_train + n_val:].reset_index(drop=True)

    print(f"样本总数: {n_samples}（约 {approx_days} 天）")
    print(f"训练数据集: {len(y_train)}（{len(y_train)//DAY_STEPS} 天）  验证数据集: {len(y_val)}（{len(y_val)//DAY_STEPS} 天）  测试数据集: {len(y_test)}（{len(y_test)//DAY_STEPS} 天）")

    segs = [
        (0, 8, "分段0_0到8"),
        (8, 16, "分段1_8到16"),
        (16, 24, "分段2_16到24"),
    ]

    models: Dict[str, lgb.LGBMRegressor] = {}

    # 训练三个分段模型
    for start_h, end_h, seg_name in segs:
        m_train = (df_train["hour"] >= start_h) & (df_train["hour"] < end_h)
        m_val = (df_val["hour"] >= start_h) & (df_val["hour"] < end_h)

        Xtr = df_train.loc[m_train, feature_cols].values
        ytr = y_train[m_train.to_numpy()]
        Xva = df_val.loc[m_val, feature_cols].values
        yva = y_val[m_val.to_numpy()]

        sample_weight = None
        if start_h == 16:
            # 晚高峰段高价加权：仅在训练样本内部计算阈值
            if len(ytr) > 0:
                thr = float(np.quantile(ytr, high_price_quantile))
                w = np.ones(len(ytr), dtype=float)
                w[ytr >= thr] = high_price_weight
                sample_weight = w
                print(f"[高价加权] 分段2（16到24）：高价阈值分位数={high_price_quantile}，阈值={thr:.2f}，高价样本权重={high_price_weight}")

        model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=96,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="regression_l2",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)] if len(yva) else None,
            eval_metric="l2",
            sample_weight=sample_weight,
        )
        models[seg_name] = model

    def predict_by_segment(df_split: pd.DataFrame) -> np.ndarray:
        y_pred = np.full(len(df_split), np.nan, dtype=float)
        for start_h, end_h, seg_name in segs:
            m = (df_split["hour"] >= start_h) & (df_split["hour"] < end_h)
            if not m.any():
                continue
            Xs = df_split.loc[m, feature_cols].values
            y_pred[m.to_numpy()] = models[seg_name].predict(Xs)
        return y_pred

    y_pred_train = predict_by_segment(df_train)
    y_pred_val = predict_by_segment(df_val)
    y_pred_test = predict_by_segment(df_test)

    print("\n================ 日前电价预测 Baseline v3（分时段分模型）评估 ================")
    _score("训练数据集", y_train, y_pred_train)
    _score("验证数据集", y_val, y_pred_val)
    test_mae, test_rmse = _score("测试数据集", y_test, y_pred_test)

    # 与原始文件中的两列预测对比（若存在）
    cmp_rows = [
        {"模型": "baseline_v3_segmented", "数据集": "测试数据集", "平均绝对误差": float(test_mae), "均方根误差": float(test_rmse), "样本数": int(len(y_test))},
    ]

    def add_excel_forecast(col: str, show: str):
        if col not in meta_test.columns:
            return
        arr = meta_test[col].to_numpy(dtype=float)
        m = np.isfinite(arr) & np.isfinite(y_test)
        if not m.any():
            return
        mae = float(mean_absolute_error(y_test[m], arr[m]))
        rmse = float(np.sqrt(mean_squared_error(y_test[m], arr[m])))
        print(f"[{show}] 平均绝对误差={mae:.2f}  均方根误差={rmse:.2f}  （样本数={int(m.sum())}）")
        # 同一批点上的模型误差
        mae_m = float(mean_absolute_error(y_test[m], y_pred_test[m]))
        rmse_m = float(np.sqrt(mean_squared_error(y_test[m], y_pred_test[m])))
        cmp_rows.append({"模型": "baseline_v3_segmented", "数据集": f"测试数据集（与{show}同一批点）", "平均绝对误差": mae_m, "均方根误差": rmse_m, "样本数": int(m.sum())})
        cmp_rows.append({"模型": show, "数据集": "测试数据集", "平均绝对误差": mae, "均方根误差": rmse, "样本数": int(m.sum())})

    add_excel_forecast("da_fcst", "原始文件_日前电价预测值")
    add_excel_forecast("da_price_fcst", "原始文件_日前价格预测")

    cmp = pd.DataFrame(cmp_rows)
    cmp_path = os.path.join(d_eval, f"{run_id}_compare_with_excel_forecasts.csv")
    cmp.to_csv(cmp_path, index=False, encoding="utf-8-sig")
    print(f"[对比结果] 已保存: {cmp_path}")

    # 保存曲线图
    df_plot = pd.DataFrame({
        "datetime": pd.to_datetime(meta_test["datetime"]),
        "真实值": y_test,
        "模型预测值": y_pred_test,
        "原始文件_日前电价预测值": meta_test["da_fcst"].to_numpy(dtype=float) if "da_fcst" in meta_test.columns else np.nan,
        "原始文件_日前价格预测": meta_test["da_price_fcst"].to_numpy(dtype=float) if "da_price_fcst" in meta_test.columns else np.nan,
    }).sort_values("datetime").reset_index(drop=True)

    plt.figure(figsize=(16, 4))
    plt.plot(df_plot["datetime"], df_plot["真实值"], label="真实值", linewidth=1.5)
    plt.plot(df_plot["datetime"], df_plot["模型预测值"], label="模型预测值（分时段分模型）", linewidth=1.2)
    if "原始文件_日前电价预测值" in df_plot.columns:
        plt.plot(df_plot["datetime"], df_plot["原始文件_日前电价预测值"], label="原始文件预测值（日前电价预测值）", linewidth=1.1)
    if "原始文件_日前价格预测" in df_plot.columns:
        plt.plot(df_plot["datetime"], df_plot["原始文件_日前价格预测"], label="原始文件预测值（日前价格预测）", linewidth=1.1)
    plt.title("测试数据集：真实值、分时段分模型预测值、原始文件预测值对比")
    plt.xlabel("时间")
    plt.ylabel("价格")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(d_eval, f"{run_id}_test_overview_with_excel.png")
    plt.savefig(fig_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[曲线图] 已保存: {fig_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    ap.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    ap.add_argument("--run_id", type=str, default=DEFAULT_RUN_ID)
    ap.add_argument("--val_days", type=int, default=30, help="验证数据集天数（默认30天）")
    ap.add_argument("--test_days", type=int, default=30, help="测试数据集天数（默认30天）")
    ap.add_argument("--high_price_quantile", type=float, default=0.85, help="晚高峰高价样本分位数阈值（默认0.85）")
    ap.add_argument("--high_price_weight", type=float, default=1.5, help="晚高峰高价样本权重（默认1.5）")
    args = ap.parse_args()

    print("=" * 80)
    print("日前电价预测 Baseline v3（分时段分模型，严格 D-1 09:00 决策）")
    print("=" * 80)

    df, colmap = load_excel(args.data_path)
    print(f"[数据] 形状={df.shape} 时间范围={df[colmap.time].min()} ~ {df[colmap.time].max()}")

    print("[特征] 构建样本...")
    df_model, y, feature_cols, df_meta = build_samples(df, colmap)
    print(f"[特征] 特征数量={len(feature_cols)}")

    train_and_evaluate(
        df_model=df_model,
        y=y,
        feature_cols=feature_cols,
        df_meta=df_meta,
        val_days=args.val_days,
        test_days=args.test_days,
        outdir=args.outdir,
        run_id=args.run_id,
        high_price_quantile=args.high_price_quantile,
        high_price_weight=args.high_price_weight,
    )


if __name__ == "__main__":
    main()

