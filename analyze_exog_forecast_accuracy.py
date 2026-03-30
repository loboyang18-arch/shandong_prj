# -*- coding: utf-8 -*-
"""
分析原始数据中“外生变量预测值”的精度（逐变量对比 预测值 vs 实际值）

输出：
- outputs_exog_accuracy/exog_forecast_accuracy.csv
- outputs_exog_accuracy/plots/*.png
"""

import os
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 中文显示设置（避免图标题/图例乱码）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


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


def detect_datetime_col(df: pd.DataFrame) -> str:
    dt = infer_col(df, ["datetime", "时间", "日期时间"])
    if dt is None:
        raise ValueError("未找到时间列（尝试匹配：datetime/时间/日期时间）。")
    return dt


def normalize_base_name(col: str) -> Optional[Tuple[str, str]]:
    """
    返回 (base_name, kind)，kind in {"actual", "forecast"}。
    """
    s = str(col)
    if "实际值" in s:
        return s.replace("实际值", "").strip(), "actual"
    if "预测值" in s:
        return s.replace("预测值", "").strip(), "forecast"
    # 兼容少量变体
    if s.endswith("实际"):
        return s[: -len("实际")].strip(), "actual"
    if s.endswith("预测"):
        return s[: -len("预测")].strip(), "forecast"
    return None


def safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {
            "有效样本数": 0,
            "平均绝对误差": np.nan,
            "均方根误差": np.nan,
            "平均偏差(预测-实际)": np.nan,
            "相关系数": np.nan,
            "平均绝对百分比误差": np.nan,
        }

    yt = y_true[mask]
    yp = y_pred[mask]
    err = yp - yt

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    if yt.size >= 2 and np.std(yt) > 0 and np.std(yp) > 0:
        corr = float(np.corrcoef(yt, yp)[0, 1])
    else:
        corr = np.nan

    denom = np.where(np.abs(yt) > 1e-9, np.abs(yt), np.nan)
    mape = float(np.nanmean(np.abs(err) / denom) * 100.0)

    return {
        "有效样本数": int(mask.sum()),
        "平均绝对误差": mae,
        "均方根误差": rmse,
        "平均偏差(预测-实际)": bias,
        "相关系数": corr,
        "平均绝对百分比误差": mape,
    }


def sanitize_filename(name: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if ch in bad else ch for ch in name)
    out = out.replace(" ", "_").strip("_")
    return out[:120] if len(out) > 120 else out


def plot_curves(
    df: pd.DataFrame,
    dt_col: str,
    base: str,
    actual_col: str,
    forecast_col: str,
    outdir: str,
    last_days: int,
):
    d = df[[dt_col, actual_col, forecast_col]].copy()
    d[dt_col] = pd.to_datetime(d[dt_col], errors="coerce")
    d = d.dropna(subset=[dt_col]).sort_values(dt_col)
    d[actual_col] = pd.to_numeric(d[actual_col], errors="coerce")
    d[forecast_col] = pd.to_numeric(d[forecast_col], errors="coerce")

    os.makedirs(outdir, exist_ok=True)
    stem = sanitize_filename(base)

    # 全量时间序列
    plt.figure(figsize=(16, 4))
    plt.plot(d[dt_col], d[actual_col], label="实际值", linewidth=1.2)
    plt.plot(d[dt_col], d[forecast_col], label="预测值", linewidth=1.2)
    plt.title(f"{base}：实际值与预测值对比（全量时间）")
    plt.xlabel("时间")
    plt.ylabel("数值")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{stem}_全量时间.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 最近若干天放大（尽量选择“实际值与预测值都有数据”的时间窗口，避免出现只有一条线）
    if last_days and last_days > 0 and not d.empty:
        m_actual = np.isfinite(d[actual_col].to_numpy(dtype=float, na_value=np.nan))
        m_forecast = np.isfinite(d[forecast_col].to_numpy(dtype=float, na_value=np.nan))
        m_both = m_actual & m_forecast

        # 优先选两者都有效的最后时间点；若不存在，则尽量选两者各自最后有效时间点的较早者
        if m_both.any():
            end_dt = d.loc[m_both, dt_col].max()
        elif m_actual.any() and m_forecast.any():
            end_dt = min(d.loc[m_actual, dt_col].max(), d.loc[m_forecast, dt_col].max())
        else:
            end_dt = d[dt_col].max()

        start_dt = end_dt - pd.Timedelta(days=int(last_days))
        dz = d[(d[dt_col] >= start_dt) & (d[dt_col] <= end_dt)]
        if not dz.empty:
            plt.figure(figsize=(16, 4))
            plt.plot(dz[dt_col], dz[actual_col], label="实际值", linewidth=1.4, linestyle="-", alpha=0.95, zorder=2)
            plt.plot(dz[dt_col], dz[forecast_col], label="预测值", linewidth=1.4, linestyle="--", alpha=0.85, zorder=3)
            plt.title(f"{base}：实际值与预测值对比（最近{int(last_days)}天，尽量取有重叠数据的窗口）")
            plt.xlabel("时间")
            plt.ylabel("数值")
            plt.legend()
            plt.grid(alpha=0.3)
            # 如果该窗口内实际值完全缺失，给出醒目标注
            if not np.isfinite(pd.to_numeric(dz[actual_col], errors='coerce')).any():
                plt.text(0.01, 0.02, "提示：该时间窗口内“实际值”为空（原始数据缺失）", transform=plt.gca().transAxes)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{stem}_最近{int(last_days)}天.png"), dpi=200, bbox_inches="tight")
            plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="山东-全年-带时间点.xlsx")
    ap.add_argument("--outdir", type=str, default="outputs_exog_accuracy")
    ap.add_argument("--plot_last_days", type=int, default=14, help="同时输出最近多少天的放大曲线（默认14天；设为0表示不输出）")
    args = ap.parse_args()

    df = pd.read_excel(args.data_path, engine="openpyxl")
    dt_col = detect_datetime_col(df)
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

    # 识别“实际值/预测值”列对
    pairs: Dict[str, Dict[str, str]] = {}
    for c in df.columns:
        norm = normalize_base_name(str(c))
        if norm is None:
            continue
        base, kind = norm
        pairs.setdefault(base, {})[kind] = str(c)

    bases = sorted([b for b, d in pairs.items() if ("actual" in d and "forecast" in d)])
    if not bases:
        raise ValueError("未找到任何同时包含“实际值”和“预测值”的变量列对。")

    rows = []
    plot_dir = os.path.join(args.outdir, "plots")
    for base in bases:
        actual_col = pairs[base]["actual"]
        forecast_col = pairs[base]["forecast"]
        y_true = safe_numeric(df[actual_col])
        y_pred = safe_numeric(df[forecast_col])
        m = compute_metrics(y_true, y_pred)
        rows.append({
            "变量": base,
            "实际值列": actual_col,
            "预测值列": forecast_col,
            **m,
        })
        plot_curves(
            df=df,
            dt_col=dt_col,
            base=base,
            actual_col=actual_col,
            forecast_col=forecast_col,
            outdir=plot_dir,
            last_days=args.plot_last_days,
        )

    out = pd.DataFrame(rows).sort_values(["均方根误差", "平均绝对误差"], ascending=[True, True])
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "exog_forecast_accuracy.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"已完成外生变量预测精度评估，共 {len(out)} 个变量。")
    print(f"结果已保存：{out_path}")
    print(f"对比曲线图片已保存：{plot_dir}")


if __name__ == "__main__":
    main()

