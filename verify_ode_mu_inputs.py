#!/usr/bin/env python3
"""Integrity checks for the curated ODE mu handoff files.

The checks are intentionally dependency-light so they can run even before the
research environment is rebuilt.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).parent
HANDOFF = ROOT / "outputs" / "ode_handoff"
MU_DIR = HANDOFF / "02_mu_inputs"
SIGMA_DIR = HANDOFF / "03_sigma_returns"

SELECTED = MU_DIR / "selected_mu_input.csv"
CALIBRATED = MU_DIR / "selected_mu_input_calibrated.csv"
PERFORMANCE = MU_DIR / "input_performance_summary.csv"
RETURNS = SIGMA_DIR / "returns_for_ode.csv"
SIGMA_SHRUNK = SIGMA_DIR / "sigma_shrunk_wide.csv"

CRITICAL_SELECTED_COLUMNS = [
    "date",
    "asset",
    "input_name",
    "mu_signal",
    "mu_raw_score",
    "mu_rank",
    "mu_centered_rank",
    "future_return",
    "lookback",
    "horizon",
]

REQUIRED_INPUTS = {
    "mu_rank_baseline",
    "mu_image_factor_rank",
    "mu_image_factor_strict_rank",
    "mu_sharpe_baseline",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def is_missing(value: str) -> bool:
    return value == "" or value.lower() in {"nan", "none", "null"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_selected(rows: list[dict[str, str]]) -> dict:
    require(rows, "selected_mu_input.csv is empty")
    header = rows[0].keys()
    for col in CRITICAL_SELECTED_COLUMNS:
        require(col in header, f"missing selected column: {col}")

    key_counts = Counter((row["date"], row["asset"]) for row in rows)
    duplicates = [key for key, count in key_counts.items() if count > 1]
    require(not duplicates, f"duplicate date-asset rows: {duplicates[:5]}")

    missing = defaultdict(int)
    for row in rows:
        for col in CRITICAL_SELECTED_COLUMNS:
            if is_missing(row[col]):
                missing[col] += 1
    require(not missing, f"missing critical selected values: {dict(missing)}")

    dates = sorted({row["date"] for row in rows})
    assets = sorted({row["asset"] for row in rows})
    counts_by_date = Counter(row["date"] for row in rows)
    require(
        min(counts_by_date.values()) == max(counts_by_date.values()) == len(assets),
        "selected file is not a balanced date x asset panel",
    )

    centered_violations = 0
    rank_violations = 0
    by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_date[row["date"]].append(row)
    for date, group in by_date.items():
        signal_sum = sum(float(row["mu_signal"]) for row in group)
        if abs(signal_sum) > 1e-10:
            centered_violations += 1
        for row in group:
            rank = float(row["mu_rank"])
            if not 0.0 < rank <= 1.0:
                rank_violations += 1
    require(centered_violations == 0, f"mu_signal is not centered on {centered_violations} dates")
    require(rank_violations == 0, f"mu_rank outside (0, 1] on {rank_violations} rows")

    return {
        "rows": len(rows),
        "dates": len(dates),
        "assets": assets,
        "start": dates[0],
        "end": dates[-1],
    }


def check_calibrated(rows: list[dict[str, str]], selected_keys: set[tuple[str, str]]) -> dict:
    require(rows, "selected_mu_input_calibrated.csv is empty")
    for col in ["mu_calibrated_daily", "mu_calibrated_horizon"]:
        require(col in rows[0], f"missing calibrated column: {col}")

    cal_keys = {(row["date"], row["asset"]) for row in rows}
    require(cal_keys == selected_keys, "calibrated grid does not match selected mu grid")

    by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_date[row["date"]].append(row)
    dates = sorted(by_date)
    nonnull_dates = [
        date
        for date in dates
        if all(not is_missing(row["mu_calibrated_daily"]) for row in by_date[date])
    ]
    require(nonnull_dates, "no non-null calibrated dates")
    first_nonnull = nonnull_dates[0]
    for date in dates[dates.index(first_nonnull):]:
        require(
            all(not is_missing(row["mu_calibrated_daily"]) for row in by_date[date]),
            f"calibrated values have a non-leading gap at {date}",
        )

    return {
        "nonnull_rows": sum(
            1 for row in rows if not is_missing(row["mu_calibrated_daily"])
        ),
        "nonnull_dates": len(nonnull_dates),
        "first_nonnull_date": first_nonnull,
    }


def check_alignment(selected_rows: list[dict[str, str]]) -> dict:
    returns = read_csv(RETURNS)
    sigma = read_csv(SIGMA_SHRUNK)

    assets = sorted({row["asset"] for row in selected_rows})
    selected_keys = {(row["date"], row["asset"]) for row in selected_rows}
    return_keys = {(row["date"], asset) for row in returns for asset in assets}
    sigma_dates = {row["date"] for row in sigma}
    selected_dates = {row["date"] for row in selected_rows}

    require(selected_keys == return_keys, "selected mu grid does not match returns_for_ode grid")
    require(selected_dates == sigma_dates, "selected mu dates do not match sigma_shrunk_wide dates")

    return {
        "returns_keys": len(return_keys),
        "sigma_dates": len(sigma_dates),
    }


def check_performance() -> dict:
    rows = read_csv(PERFORMANCE)
    names = {row["input_name"] for row in rows}
    require(REQUIRED_INPUTS <= names, f"missing required performance rows: {sorted(REQUIRED_INPUTS - names)}")
    recommended = [row for row in rows if row.get("is_recommended") == "True"]
    require(len(recommended) == 1, "expected exactly one recommended input")
    return {
        "rows": len(rows),
        "recommended": recommended[0]["input_name"],
        "recommended_rank_corr": recommended[0]["rank_corr"],
        "recommended_top_k_sharpe": recommended[0]["top_k_sharpe"],
    }


def main() -> None:
    selected_rows = read_csv(SELECTED)
    selected_summary = check_selected(selected_rows)
    selected_keys = {(row["date"], row["asset"]) for row in selected_rows}
    calibrated_summary = check_calibrated(read_csv(CALIBRATED), selected_keys)
    alignment_summary = check_alignment(selected_rows)
    performance_summary = check_performance()

    summary = {
        "status": "PASS",
        "selected_mu_input": selected_summary,
        "calibrated_mu_input": calibrated_summary,
        "alignment": alignment_summary,
        "performance": performance_summary,
        "caveat": "This verifies file integrity and grid alignment, not statistical significance.",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
