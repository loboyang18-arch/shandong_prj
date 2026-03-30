# audit_shandong_excel.py
# -*- coding: utf-8 -*-
"""
Audit script for Shandong Excel dataset:
- Column inventory
- Time axis continuity (15min grid)
- Per-column coverage: first/last finite time, NaN ratio, non-finite ratio
- Missing blocks (continuous NaN runs) optional
- Daily coverage report

Usage:
  python -u audit_shandong_excel.py --data_path "山东-全年-带时间点.xlsx" --outdir "audit_outputs" --save_missing_blocks
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


STEP_MINUTES = 15


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_time_col(df: pd.DataFrame) -> str:
    candidates = ["datetime", "时间", "日期时间", "Datetime", "DATE_TIME", "date_time"]
    cols = df.columns.tolist()
    for c in candidates:
        if c in cols:
            return c
    # fallback: find a column containing "time" or "日期"
    for c in cols:
        lc = str(c).lower()
        if "time" in lc or "日期" in str(c) or "时间" in str(c):
            return c
    raise ValueError(f"Cannot infer time column. Available columns head: {cols[:60]}")


def read_excel_any_sheet(path: str, sheet: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Robust Excel loader:
    - If sheet is provided: load that sheet.
    - Else: load the first sheet.
    - If pandas returns dict (sheet_name=None), automatically pick the first key.
    """
    if sheet is not None:
        df = pd.read_excel(path, engine="openpyxl", sheet_name=sheet)
        if isinstance(df, dict):
            # Shouldn't happen with explicit sheet, but keep safe
            first = list(df.keys())[0]
            return df[first], first
        return df, str(sheet)

    # Default: first sheet (sheet_name=0) -> DataFrame
    df = pd.read_excel(path, engine="openpyxl", sheet_name=0)
    if isinstance(df, dict):
        first = list(df.keys())[0]
        return df[first], first
    return df, "0(first)"


def dt_floor(ts: pd.Series, minutes: int = 15) -> pd.Series:
    return ts.dt.floor(f"{minutes}min")


def is_finite_array(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a)


def find_missing_blocks(time: np.ndarray, finite_mask: np.ndarray, min_len: int = 4) -> List[Dict]:
    """
    finite_mask=True means OK; False means missing/non-finite.
    Return continuous missing blocks with length >= min_len.
    """
    bad = ~finite_mask
    if bad.sum() == 0:
        return []

    blocks = []
    n = len(bad)
    i = 0
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < n and bad[j]:
            j += 1
        length = j - i
        if length >= min_len:
            blocks.append({
                "start_time": str(pd.Timestamp(time[i])),
                "end_time": str(pd.Timestamp(time[j - 1])),
                "length": int(length),
            })
        i = j
    return blocks


# -----------------------------
# Main audit
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="audit_outputs")
    ap.add_argument("--sheet", type=str, default=None, help="Excel sheet name (optional). Default: first sheet.")
    ap.add_argument("--save_missing_blocks", action="store_true", help="save missing blocks for each column")
    ap.add_argument("--min_block_len", type=int, default=4, help="min length (points) of missing block to report")
    ap.add_argument("--time_col", type=str, default=None, help="override time column name (optional)")
    ap.add_argument("--strict_15min", action="store_true", help="raise error if not strict 15min continuous")
    args = ap.parse_args()

    data_path = args.data_path.strip()
    outdir = args.outdir
    ensure_dir(outdir)

    print(f"[START] audit_shandong_excel.py", flush=True)
    print(f"[ARGS] data_path={data_path}", flush=True)
    print(f"[ARGS] outdir={outdir} sheet={args.sheet}", flush=True)

    df, sheet_used = read_excel_any_sheet(data_path, sheet=args.sheet)
    print(f"[LOAD] sheet={sheet_used} shape={df.shape}", flush=True)

    # time col
    time_col = args.time_col if args.time_col else infer_time_col(df)
    print(f"[TIME] time_col={time_col}", flush=True)

    # parse time and floor
    t_raw = pd.to_datetime(df[time_col], errors="coerce")
    n_time_ok = int(t_raw.notna().sum())
    df = df.copy()
    df[time_col] = t_raw
    df = df.dropna(subset=[time_col]).copy()
    df[time_col] = dt_floor(df[time_col], STEP_MINUTES)

    # sort & dedup
    df = df.sort_values(time_col).reset_index(drop=True)
    n_raw = int(len(df))
    n_dups = int(df.duplicated(time_col).sum())
    if n_dups > 0:
        df = df.drop_duplicates(time_col, keep="last").reset_index(drop=True)
    n_after = int(len(df))

    t = df[time_col]
    tmin = pd.Timestamp(t.min())
    tmax = pd.Timestamp(t.max())

    # build full expected grid
    full_grid = pd.date_range(tmin, tmax, freq=f"{STEP_MINUTES}min")
    s_full = pd.Series(full_grid)
    s_have = pd.Series(t.values)

    have_set = set(s_have.astype("datetime64[ns]").tolist())
    full_set = set(s_full.astype("datetime64[ns]").tolist())

    missing_ts = sorted(list(full_set - have_set))
    extra_ts = sorted(list(have_set - full_set))

    # gap check
    dt_diff = t.diff().dropna()
    gap_mask = dt_diff > pd.Timedelta(minutes=STEP_MINUTES)
    n_gaps = int(gap_mask.sum())

    time_axis_report = {
        "time_col": time_col,
        "sheet_used": sheet_used,
        "n_rows_raw_after_dropna_time": n_raw,
        "n_time_parse_ok_in_original": n_time_ok,
        "n_time_duplicates_after_floor": n_dups,
        "time_min": str(tmin),
        "time_max": str(tmax),
        "n_rows_after_dedup": n_after,
        "expected_full_grid_points": int(len(full_grid)),
        "missing_timestamps_count": int(len(missing_ts)),
        "extra_timestamps_count": int(len(extra_ts)),
        "n_gaps_gt_15min": n_gaps,
        "missing_timestamps_sample_head": [str(pd.Timestamp(x)) for x in missing_ts[:10]],
        "missing_timestamps_sample_tail": [str(pd.Timestamp(x)) for x in missing_ts[-10:]],
    }

    with open(os.path.join(outdir, "time_axis_report.json"), "w", encoding="utf-8") as f:
        json.dump(time_axis_report, f, ensure_ascii=False, indent=2)
    print(f"[OUT] time_axis_report.json saved.", flush=True)

    if args.strict_15min and (len(missing_ts) > 0 or n_gaps > 0):
        raise ValueError(f"[STRICT] time axis not strict 15min: missing={len(missing_ts)} gaps={n_gaps}")

    # per-column summary
    cols = [c for c in df.columns if c != time_col]
    summary_rows = []
    all_blocks = []

    # daily coverage base
    df_date = df[time_col].dt.date.astype(str)
    daily_rows = []

    print(f"[COL] auditing {len(cols)} columns...", flush=True)

    time_np = df[time_col].to_numpy(dtype="datetime64[ns]")

    for c in cols:
        s = df[c]

        # numeric coercion for coverage
        a = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        finite = is_finite_array(a)
        finite_cnt = int(finite.sum())
        nan_cnt = int(np.isnan(a).sum())
        nonfinite_cnt = int((~np.isfinite(a) & ~np.isnan(a)).sum())  # inf/-inf
        total = int(len(a))
        finite_ratio = float(finite_cnt / total) if total > 0 else 0.0

        first_valid = None
        last_valid = None
        if finite_cnt > 0:
            first_valid = str(pd.Timestamp(time_np[np.where(finite)[0][0]]))
            last_valid = str(pd.Timestamp(time_np[np.where(finite)[0][-1]]))

        # leading/trailing missing
        lead_nan = int(np.argmax(finite)) if finite_cnt > 0 else total
        trail_nan = int(np.argmax(finite[::-1])) if finite_cnt > 0 else total

        # longest missing run
        blocks = find_missing_blocks(time_np, finite, min_len=args.min_block_len)
        longest_block = max([b["length"] for b in blocks], default=0)
        n_blocks = len(blocks)

        # stats on finite values
        if finite_cnt > 0:
            vv = a[finite]
            vmin = float(np.nanmin(vv))
            vmax = float(np.nanmax(vv))
            vmean = float(np.nanmean(vv))
            vstd = float(np.nanstd(vv))
        else:
            vmin = vmax = vmean = vstd = np.nan

        summary_rows.append({
            "col": str(c),
            "dtype": str(df[c].dtype),
            "total": total,
            "finite_cnt": finite_cnt,
            "finite_ratio": finite_ratio,
            "nan_cnt": nan_cnt,
            "nonfinite_cnt": nonfinite_cnt,
            "first_valid_time": first_valid,
            "last_valid_time": last_valid,
            "leading_missing_points": lead_nan,
            "trailing_missing_points": trail_nan,
            "missing_blocks_cnt_ge_minlen": n_blocks,
            "longest_missing_block_len": int(longest_block),
            "min": vmin,
            "max": vmax,
            "mean": vmean,
            "std": vstd,
        })

        if args.save_missing_blocks and n_blocks > 0:
            for b in blocks:
                all_blocks.append({
                    "col": str(c),
                    **b
                })

        # daily coverage for this column
        # (finite_ratio per day)
        tmp = pd.DataFrame({"date": df_date, "finite": finite.astype(np.int8)})
        d = tmp.groupby("date")["finite"].mean().reset_index()
        d["col"] = str(c)
        daily_rows.append(d)

    per_col_summary = pd.DataFrame(summary_rows).sort_values(
        by=["finite_ratio", "col"], ascending=[True, True]
    ).reset_index(drop=True)

    per_col_path = os.path.join(outdir, "per_column_summary.csv")
    per_col_summary.to_csv(per_col_path, index=False, encoding="utf-8-sig")
    print(f"[OUT] per_column_summary.csv saved: {per_col_path}", flush=True)

    # daily coverage overall
    daily_all = pd.concat(daily_rows, ignore_index=True)  # columns: date, finite, col
    # pivot to wide for easier reading
    daily_wide = daily_all.pivot_table(index="date", columns="col", values="finite", aggfunc="mean")
    daily_wide = daily_wide.sort_index()

    daily_overall = pd.DataFrame({
        "date": daily_wide.index,
        "min_valid_ratio_across_cols": daily_wide.min(axis=1).values,
        "mean_valid_ratio_across_cols": daily_wide.mean(axis=1).values,
        "cols_with_any_missing": (daily_wide < 1.0).sum(axis=1).values,
        "cols_all_missing": (daily_wide == 0.0).sum(axis=1).values,
    })

    daily_overall_path = os.path.join(outdir, "daily_coverage_overall.csv")
    daily_overall.to_csv(daily_overall_path, index=False, encoding="utf-8-sig")
    print(f"[OUT] daily_coverage_overall.csv saved: {daily_overall_path}", flush=True)

    if args.save_missing_blocks:
        blocks_path = os.path.join(outdir, "missing_blocks.csv")
        pd.DataFrame(all_blocks).to_csv(blocks_path, index=False, encoding="utf-8-sig")
        print(f"[OUT] missing_blocks.csv saved: {blocks_path}", flush=True)

    # quick console summary
    print("\n[SUMMARY] time axis:", flush=True)
    print(json.dumps(time_axis_report, ensure_ascii=False, indent=2), flush=True)

    print("\n[SUMMARY] worst columns by finite_ratio (top 15):", flush=True)
    show = per_col_summary[["col", "finite_ratio", "first_valid_time", "last_valid_time", "longest_missing_block_len"]].head(15)
    print(show.to_string(index=False), flush=True)

    print("\n[DONE] audit finished.", flush=True)


if __name__ == "__main__":
    main()
