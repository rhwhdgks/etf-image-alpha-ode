#!/usr/bin/env python3
"""Ledoit-Wolf shrinkage Sigma(t) for the ODE handoff package.

Background (audit Gap 1)
------------------------
The handoff `03_sigma_returns/sigma_wide.csv` is a plain 60-day rolling sample
covariance. Its condition number has a median ~2,400 but a p95 ~10,600 and a max
~18,400. With a constant risk-aversion ODE, the equilibrium weight is
`w* = (alpha / 2 gamma_0) * Sigma^-1 * mu`, so Sigma enters through its inverse and
high-condition-number dates can produce unstable / explosive weights.

This script adds a shrunk Sigma(t) alongside the sample Sigma(t). Ledoit-Wolf
(2004) shrinks the sample covariance toward a scaled-identity target with an
analytically optimal intensity, which lowers the condition number and stabilises
the inverse without introducing look-ahead (each date uses only its own trailing
60-day window).

Outputs (into 03_sigma_returns/)
--------------------------------
- sigma_shrunk_wide.csv   : same schema as sigma_wide.csv
- sigma_shrunk_long.csv   : same schema as sigma_long.csv
- sigma_shrinkage_report.md : per-date shrinkage intensity + condition-number comparison
The original sample-covariance files are left untouched; the ODE team can compare.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.data.loader import REQUIRED_PRICE_FIELDS, load_etf_csv, restrict_common_valid_sample

ROOT = Path(__file__).parent
HANDOFF = ROOT / "ode_inputs_cnn" / "ode_team_handoff_20260523" / "03_sigma_returns"
DATA_PATH = ROOT / "etfdata.csv"
SIGMA_WINDOW = 60

ASSETS = [
    "alternative", "corp_bond_ig", "developed_equity", "emerging_equity",
    "korea_equity", "short_treasury", "treasury_7_10y",
]


def _log_returns() -> pd.DataFrame:
    _, long_panel, _, _, _ = load_etf_csv(str(DATA_PATH))
    common_panel, _ = restrict_common_valid_sample(
        long_panel, selected_assets=ASSETS, required_fields=REQUIRED_PRICE_FIELDS
    )
    close = (
        common_panel.pivot(index="date", columns="asset", values="close")
        .sort_index()[sorted(ASSETS)]
    )
    close.index = pd.to_datetime(close.index).normalize()
    return np.log(close / close.shift(1)).dropna()


def _cond(matrix: np.ndarray) -> float:
    eigs = np.linalg.eigvalsh(matrix)
    lo = max(eigs.min(), 1e-15)
    return float(eigs.max() / lo)


def main() -> None:
    assets = sorted(ASSETS)
    log_ret = _log_returns()
    print(f"log returns: {log_ret.shape[0]} dates, {log_ret.shape[1]} assets")

    # target date list = existing sample-Sigma dates
    sample_wide = pd.read_csv(HANDOFF / "sigma_wide.csv", parse_dates=["date"])
    target_dates = sorted(sample_wide["date"].unique())
    sample_wide = sample_wide.set_index("date")

    sigma_cols = [f"{a}_sigma_ii" for a in assets]
    cov_pairs = [(a, b) for ai, a in enumerate(assets) for b in assets[ai + 1:]]
    cov_cols = [f"{a}_{b}_cov" for a, b in cov_pairs]

    wide_rows, long_rows, report_rows = [], [], []
    ret_index = log_ret.index

    for date in target_dates:
        pos = ret_index.searchsorted(date)
        if pos >= len(ret_index) or ret_index[pos] != date:
            raise ValueError(f"date {date} not in log-return index")
        if pos < SIGMA_WINDOW - 1:
            raise ValueError(f"date {date} has < {SIGMA_WINDOW} window history")
        window = log_ret.iloc[pos - SIGMA_WINDOW + 1 : pos + 1][assets].to_numpy(dtype=float)

        lw = LedoitWolf().fit(window)
        shrunk = lw.covariance_
        alpha = float(lw.shrinkage_)
        sample_cov = np.cov(window, rowvar=False)

        # wide row
        wide = {"date": date}
        for i, a in enumerate(assets):
            wide[f"{a}_sigma_ii"] = float(shrunk[i, i])
        for (a, b) in cov_pairs:
            i, j = assets.index(a), assets.index(b)
            wide[f"{a}_{b}_cov"] = float(shrunk[i, j])
        wide_rows.append(wide)

        # long rows (full 7x7)
        for i, a in enumerate(assets):
            long_rows.append({"date": date, "asset_i": a, "asset_j": a,
                              "sigma_value": float(shrunk[i, i]), "entry_type": "variance"})
        for (a, b) in cov_pairs:
            i, j = assets.index(a), assets.index(b)
            v = float(shrunk[i, j])
            long_rows.append({"date": date, "asset_i": a, "asset_j": b,
                              "sigma_value": v, "entry_type": "covariance"})
            long_rows.append({"date": date, "asset_i": b, "asset_j": a,
                              "sigma_value": v, "entry_type": "covariance"})

        report_rows.append({
            "date": date,
            "shrinkage_alpha": alpha,
            "cond_sample": _cond(sample_cov),
            "cond_shrunk": _cond(shrunk),
        })

    wide_df = pd.DataFrame(wide_rows)[["date"] + sigma_cols + cov_cols]
    long_df = pd.DataFrame(long_rows).sort_values(["date", "asset_i", "asset_j"]).reset_index(drop=True)
    report = pd.DataFrame(report_rows)

    # sanity: PSD on every date
    bad_psd = 0
    for date in target_dates[:50] + target_dates[-50:]:
        sub = long_df[long_df["date"] == date]
        M = np.zeros((7, 7))
        for _, r in sub.iterrows():
            M[assets.index(r["asset_i"]), assets.index(r["asset_j"])] = r["sigma_value"]
        if np.linalg.eigvalsh(M).min() < -1e-12:
            bad_psd += 1
    print(f"PSD check (100 sampled dates): {bad_psd} non-PSD")

    for df in (wide_df, long_df):
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    wide_df.to_csv(HANDOFF / "sigma_shrunk_wide.csv", index=False)
    long_df.to_csv(HANDOFF / "sigma_shrunk_long.csv", index=False)
    print(f"wrote sigma_shrunk_wide.csv ({len(wide_df)} rows), sigma_shrunk_long.csv ({len(long_df)} rows)")

    # condition-number comparison
    cs, csh = report["cond_sample"], report["cond_shrunk"]
    md = [
        "# Sigma Ledoit-Wolf Shrinkage Report",
        "",
        "## Method",
        "",
        f"- Per date: 60-day trailing log-return window -> Ledoit-Wolf shrinkage (sklearn).",
        "- Shrinkage target: scaled identity; intensity alpha estimated analytically per window.",
        "- No look-ahead: each date uses only its own trailing window.",
        "",
        "## Shrinkage intensity",
        "",
        f"- alpha mean {report['shrinkage_alpha'].mean():.3f}, "
        f"median {report['shrinkage_alpha'].median():.3f}, "
        f"min {report['shrinkage_alpha'].min():.3f}, max {report['shrinkage_alpha'].max():.3f}",
        "",
        "## Condition number: sample vs shrunk",
        "",
        "| stat | sample Sigma | shrunk Sigma |",
        "|---|---:|---:|",
        f"| median | {cs.median():.0f} | {csh.median():.0f} |",
        f"| p95 | {cs.quantile(0.95):.0f} | {csh.quantile(0.95):.0f} |",
        f"| max | {cs.max():.0f} | {csh.max():.0f} |",
        f"| dates cond > 1e4 | {(cs > 1e4).sum()} | {(csh > 1e4).sum()} |",
        "",
        "## Interpretation",
        "",
        "- Shrinkage lowers the condition number, so `Sigma^-1` is more stable for the",
        "  constant-gamma ODE equilibrium `w* = (alpha/2 gamma_0) Sigma^-1 mu`.",
        "- Use `sigma_shrunk_wide.csv` as the recommended Sigma input; `sigma_wide.csv`",
        "  (sample covariance) is kept for comparison / ablation.",
        "- Shrinkage trades a small bias for a large variance reduction in the inverse;",
        "  the ODE team can A/B test both Sigma versions.",
    ]
    (HANDOFF / "sigma_shrinkage_report.md").write_text("\n".join(md), encoding="utf-8")
    report.assign(date=report["date"].astype(str)).to_csv(
        HANDOFF / "sigma_shrinkage_diagnostics.csv", index=False)
    print(f"wrote sigma_shrinkage_report.md + sigma_shrinkage_diagnostics.csv")
    print()
    print("\n".join(md[12:21]))


if __name__ == "__main__":
    main()
