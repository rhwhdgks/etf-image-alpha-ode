#!/usr/bin/env python3
"""Additional Sigma(t) variants for the ODE handoff.

The handoff already ships:
- sigma_wide.csv         (60-day rolling sample covariance, plain)
- sigma_shrunk_wide.csv  (Ledoit-Wolf shrunk per-window)

This script adds two more, so the ODE team can A/B test the choice of Sigma:
- sigma_ewma_wide.csv         RiskMetrics-style exponentially weighted (lambda=0.94)
- sigma_multiwindow_wide.csv  blend of 60d and 120d sample covariances (0.5 / 0.5)

All variants share the same date grid (2880 dates) and the same wide schema.
Per-date condition-number stats are compared in sigma_variants_report.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import REQUIRED_PRICE_FIELDS, load_etf_csv, restrict_common_valid_sample

ROOT = Path(__file__).parent
HANDOFF = ROOT / "ode_inputs_cnn" / "ode_team_handoff_20260523" / "03_sigma_returns"
DATA_PATH = ROOT / "etfdata.csv"

ASSETS = sorted([
    "alternative", "corp_bond_ig", "developed_equity", "emerging_equity",
    "korea_equity", "short_treasury", "treasury_7_10y",
])

EWMA_LAMBDA = 0.94      # RiskMetrics default
EWMA_TRUNCATE = 252     # truncate past contributions beyond 1 year
WINDOW_SHORT = 60
WINDOW_LONG = 120
BLEND_W_SHORT = 0.5


def _log_returns() -> pd.DataFrame:
    _, long_panel, _, _, _ = load_etf_csv(str(DATA_PATH))
    common_panel, _ = restrict_common_valid_sample(
        long_panel, selected_assets=ASSETS, required_fields=REQUIRED_PRICE_FIELDS)
    close = (
        common_panel.pivot(index="date", columns="asset", values="close")
        .sort_index()[ASSETS]
    )
    close.index = pd.to_datetime(close.index).normalize()
    return np.log(close / close.shift(1)).dropna()


def _cond(matrix: np.ndarray) -> float:
    eigs = np.linalg.eigvalsh(matrix)
    return float(eigs.max() / max(eigs.min(), 1e-15))


def ewma_cov(window: np.ndarray, lam: float) -> np.ndarray:
    """Weighted sample covariance with exponential weights (latest day = highest)."""
    n = len(window)
    weights = np.array([(1 - lam) * lam ** (n - 1 - t) for t in range(n)])
    weights = weights / weights.sum()
    mean = (window * weights[:, None]).sum(axis=0)
    dev = window - mean
    return (dev.T * weights) @ dev


def to_wide_row(date, sigma: np.ndarray, sigma_cols, cov_pairs) -> dict:
    row = {"date": date}
    for i, a in enumerate(ASSETS):
        row[f"{a}_sigma_ii"] = float(sigma[i, i])
    for (a, b) in cov_pairs:
        i, j = ASSETS.index(a), ASSETS.index(b)
        row[f"{a}_{b}_cov"] = float(sigma[i, j])
    return row


def to_long_rows(date, sigma: np.ndarray, cov_pairs) -> list[dict]:
    rows = []
    for i, a in enumerate(ASSETS):
        rows.append({"date": date, "asset_i": a, "asset_j": a,
                     "sigma_value": float(sigma[i, i]), "entry_type": "variance"})
    for (a, b) in cov_pairs:
        i, j = ASSETS.index(a), ASSETS.index(b)
        v = float(sigma[i, j])
        rows.append({"date": date, "asset_i": a, "asset_j": b,
                     "sigma_value": v, "entry_type": "covariance"})
        rows.append({"date": date, "asset_i": b, "asset_j": a,
                     "sigma_value": v, "entry_type": "covariance"})
    return rows


def main() -> None:
    log_ret = _log_returns()
    print(f"log returns: {log_ret.shape}")
    sample_wide = pd.read_csv(HANDOFF / "sigma_wide.csv", parse_dates=["date"])
    target_dates = sorted(sample_wide["date"].unique())

    sigma_cols = [f"{a}_sigma_ii" for a in ASSETS]
    cov_pairs = [(a, b) for ai, a in enumerate(ASSETS) for b in ASSETS[ai + 1:]]
    cov_cols = [f"{a}_{b}_cov" for a, b in cov_pairs]

    ewma_wide_rows, ewma_long_rows, ewma_diag = [], [], []
    mw_wide_rows, mw_long_rows, mw_diag = [], [], []

    ret_index = log_ret.index
    for date in target_dates:
        pos = ret_index.searchsorted(date)
        if pos >= len(ret_index) or ret_index[pos] != date:
            raise ValueError(f"date {date} missing")

        # EWMA: trailing EWMA_TRUNCATE days
        ewma_start = max(0, pos - EWMA_TRUNCATE + 1)
        ewma_window = log_ret.iloc[ewma_start : pos + 1][ASSETS].to_numpy(dtype=float)
        ewma_sigma = ewma_cov(ewma_window, EWMA_LAMBDA)
        ewma_wide_rows.append(to_wide_row(date, ewma_sigma, sigma_cols, cov_pairs))
        ewma_long_rows.extend(to_long_rows(date, ewma_sigma, cov_pairs))
        ewma_diag.append({"date": date, "cond": _cond(ewma_sigma),
                          "min_eig": float(np.linalg.eigvalsh(ewma_sigma).min())})

        # Multi-window blend: 60d + 120d sample cov
        if pos < WINDOW_LONG - 1:
            # not enough history -> reuse 60d only
            short = log_ret.iloc[pos - WINDOW_SHORT + 1 : pos + 1][ASSETS].to_numpy(dtype=float)
            mw_sigma = np.cov(short, rowvar=False)
        else:
            short = log_ret.iloc[pos - WINDOW_SHORT + 1 : pos + 1][ASSETS].to_numpy(dtype=float)
            long_ = log_ret.iloc[pos - WINDOW_LONG + 1 : pos + 1][ASSETS].to_numpy(dtype=float)
            mw_sigma = BLEND_W_SHORT * np.cov(short, rowvar=False) + \
                       (1 - BLEND_W_SHORT) * np.cov(long_, rowvar=False)
        mw_wide_rows.append(to_wide_row(date, mw_sigma, sigma_cols, cov_pairs))
        mw_long_rows.extend(to_long_rows(date, mw_sigma, cov_pairs))
        mw_diag.append({"date": date, "cond": _cond(mw_sigma),
                        "min_eig": float(np.linalg.eigvalsh(mw_sigma).min())})

    for name, wide_rows, long_rows in [
        ("ewma", ewma_wide_rows, ewma_long_rows),
        ("multiwindow", mw_wide_rows, mw_long_rows),
    ]:
        wide_df = pd.DataFrame(wide_rows)[["date"] + sigma_cols + cov_cols]
        long_df = pd.DataFrame(long_rows).sort_values(["date", "asset_i", "asset_j"]).reset_index(drop=True)
        wide_df["date"] = pd.to_datetime(wide_df["date"]).dt.strftime("%Y-%m-%d")
        long_df["date"] = pd.to_datetime(long_df["date"]).dt.strftime("%Y-%m-%d")
        wide_df.to_csv(HANDOFF / f"sigma_{name}_wide.csv", index=False)
        long_df.to_csv(HANDOFF / f"sigma_{name}_long.csv", index=False)
        print(f"wrote sigma_{name}_wide.csv ({len(wide_df)} rows), sigma_{name}_long.csv ({len(long_df)} rows)")

    # ---- comparison report ----
    # Recompute sample + shrunk cond from existing wide files for a single table
    def reload_conds(wide_path: Path) -> pd.Series:
        df = pd.read_csv(wide_path, parse_dates=["date"])
        cnds = []
        for _, r in df.iterrows():
            S = np.zeros((7, 7))
            for i, a in enumerate(ASSETS):
                S[i, i] = r[f"{a}_sigma_ii"]
            for a, b in cov_pairs:
                i, j = ASSETS.index(a), ASSETS.index(b)
                S[i, j] = S[j, i] = r[f"{a}_{b}_cov"]
            cnds.append(_cond(S))
        return pd.Series(cnds, name=wide_path.stem)

    cs_sample = reload_conds(HANDOFF / "sigma_wide.csv")
    cs_shrunk = reload_conds(HANDOFF / "sigma_shrunk_wide.csv")
    cs_ewma = pd.Series([d["cond"] for d in ewma_diag], name="ewma")
    cs_mw = pd.Series([d["cond"] for d in mw_diag], name="multiwindow")

    rows = []
    for name, s in [("sample (60d)", cs_sample), ("shrunk (LW)", cs_shrunk),
                    ("ewma (lambda=0.94)", cs_ewma), ("multiwindow (60d+120d)", cs_mw)]:
        rows.append({
            "variant": name,
            "median": int(s.median()),
            "p95": int(s.quantile(0.95)),
            "max": int(s.max()),
            "dates_cond_gt_1e4": int((s > 1e4).sum()),
        })
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(HANDOFF / "sigma_variants_diagnostics.csv", index=False)
    print()
    print(cmp_df.to_string(index=False))

    md = [
        "# Sigma Variants — condition-number comparison",
        "",
        "Four Sigma(t) variants are available in this handoff:",
        "",
        "| variant | file | description |",
        "|---|---|---|",
        "| sample | `sigma_wide.csv` | plain 60-day rolling sample covariance |",
        "| shrunk | `sigma_shrunk_wide.csv` | Ledoit-Wolf shrinkage per window |",
        f"| ewma | `sigma_ewma_wide.csv` | exponentially weighted (lambda={EWMA_LAMBDA}, truncated at {EWMA_TRUNCATE}) |",
        f"| multiwindow | `sigma_multiwindow_wide.csv` | blend of {WINDOW_SHORT}-day and {WINDOW_LONG}-day sample covs (0.5 / 0.5) |",
        "",
        "## Condition number distribution",
        "",
        "| variant | median | p95 | max | dates cond > 1e4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['variant']} | {r['median']} | {r['p95']} | {r['max']} | {r['dates_cond_gt_1e4']} |")
    md += [
        "",
        "## Interpretation",
        "",
        "- **shrunk** is the most numerically stable choice (smallest condition number).",
        "  Recommended default when the constant-gamma ODE relies on `Sigma^-1 mu`.",
        "- **ewma** gives the recent regime more weight (effective lookback ~1/(1-lambda) ~ 17 days).",
        "  More adaptive to vol regime changes but noisier than rolling.",
        "- **multiwindow** averages a fast 60-day estimate with a slow 120-day estimate;",
        "  trades a small lag for less noise than 60-day alone.",
        "- **sample** kept for A/B comparison with the original handoff.",
    ]
    (HANDOFF / "sigma_variants_report.md").write_text("\n".join(md), encoding="utf-8")
    print()
    print(f"wrote sigma_variants_report.md")


if __name__ == "__main__":
    main()
