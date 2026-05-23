#!/usr/bin/env python3
"""Rebuild the return-scale calibration of the ODE mu candidates.

Background
----------
The handoff package `ode_team_handoff_20260523/02_mu_inputs/final_mu_inputs_calibrated.csv`
contained `mu_calibrated_horizon` / `mu_calibrated_daily` columns, but the
generating script was missing from the repo (reproducibility gap, audit Finding 3).
This script restores it.

Method (reproduces the documented manifest method, plus a leakage fix)
----------------------------------------------------------------------
For each `input_name` independently:
  - Pooled expanding OLS  `future_return ~ 1 + mu_signal`  over all asset-date
    training rows strictly before the signal date.
  - Minimum training history: `MIN_TRAIN_DATES` (252) distinct signal dates.
  - Predict `mu_calibrated_horizon` for the current date; `mu_calibrated_daily`
    is that value divided by the horizon.

Leakage fix vs the original (audit Finding 2/3)
-----------------------------------------------
`future_return` at date s is a horizon-day forward return, realised only at
index(s) + horizon. The original calibration used "all dates before the signal
date", which silently includes the last `horizon` dates whose forward window has
NOT closed yet. This rebuild adds a horizon embargo: a training date s is usable
for calibrating date t only if index(s) + horizon <= index(t).

Set EMBARGO = False to reproduce the original (un-embargoed) numbers exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
HANDOFF = ROOT / "ode_inputs_cnn" / "ode_team_handoff_20260523" / "02_mu_inputs"
LONG_PATH = HANDOFF / "final_mu_inputs_long.csv"
OUT_PATH = HANDOFF / "final_mu_inputs_calibrated.csv"
SUMMARY_PATH = HANDOFF / "mu_calibration_summary.csv"

HORIZON = 20
MIN_TRAIN_DATES = 252
EMBARGO = True  # horizon-embargo on training rows; False = reproduce original


def calibrate_one_input(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding pooled OLS calibration for a single input_name."""
    df = df.sort_values(["date", "asset"]).reset_index(drop=True)
    unique_dates = sorted(df["date"].unique())
    date_index = {d: i for i, d in enumerate(unique_dates)}

    horizon_col = np.full(len(df), np.nan)
    rows_by_date = {d: g for d, g in df.groupby("date")}

    for i, date in enumerate(unique_dates):
        # training dates: strictly before `date`, and (if EMBARGO) forward-window closed
        last_train_idx = (i - HORIZON) if EMBARGO else (i - 1)
        if last_train_idx < 0:
            continue
        train_dates = unique_dates[: last_train_idx + 1]
        if len(train_dates) < MIN_TRAIN_DATES:
            continue

        train = df[df["date"].isin(train_dates)]
        x = train["mu_signal"].to_numpy(dtype=float)
        y = train["future_return"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2 or np.std(x[ok]) < 1e-12:
            continue

        # OLS: y = a + b x
        b, a = np.polyfit(x[ok], y[ok], 1)

        cur = rows_by_date[date]
        pred = a + b * cur["mu_signal"].to_numpy(dtype=float)
        horizon_col[cur.index.to_numpy()] = pred

    out = df.copy()
    out["mu_calibrated_horizon"] = horizon_col
    out["mu_calibrated_daily"] = horizon_col / HORIZON
    out["calibration_min_train_dates"] = MIN_TRAIN_DATES
    return out


def main() -> None:
    print(f"Loading {LONG_PATH} ...")
    long = pd.read_csv(LONG_PATH, parse_dates=["date"])
    print(f"  rows={len(long)}  inputs={sorted(long['input_name'].unique())}")
    print(f"  embargo={EMBARGO}  horizon={HORIZON}  min_train_dates={MIN_TRAIN_DATES}")

    parts = [calibrate_one_input(g) for _, g in long.groupby("input_name")]
    calibrated = pd.concat(parts, ignore_index=True).sort_values(
        ["input_name", "date", "asset"]
    ).reset_index(drop=True)
    calibrated.to_csv(OUT_PATH, index=False)
    print(f"  wrote {OUT_PATH}  ({len(calibrated)} rows)")

    # summary per input
    summary_rows = []
    for name, g in calibrated.groupby("input_name"):
        cal = g.dropna(subset=["mu_calibrated_horizon"])
        if cal.empty:
            continue
        rc = (
            cal.groupby("date")
            .apply(lambda x: x["mu_calibrated_daily"].rank().corr(x["future_return"].rank()))
            .dropna()
            .mean()
        )
        rmse = float(np.sqrt(np.mean((cal["mu_calibrated_horizon"] - cal["future_return"]) ** 2)))
        summary_rows.append({
            "input_name": name,
            "calibrated_dates": cal["date"].nunique(),
            "start": str(cal["date"].min().date()),
            "daily_rank_corr": round(float(rc), 4),
            "rmse": round(rmse, 4),
            "mean_horizon_mu": round(float(cal["mu_calibrated_horizon"].mean()), 5),
            "std_horizon_mu": round(float(cal["mu_calibrated_horizon"].std()), 5),
            "embargo": EMBARGO,
        })
    summary = pd.DataFrame(summary_rows).sort_values("daily_rank_corr", ascending=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"  wrote {SUMMARY_PATH}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
