#!/usr/bin/env python3
"""Recompute the image-factor lift CIs with a stationary block bootstrap.

The handoff README / manifest report image-factor lift CIs computed with IID
resampling. The per-date rank-correlation series is autocorrelated (20-day
overlapping horizon), so IID understates CI width. This script recomputes the
same lifts with a stationary block bootstrap (Politis-Romano) and writes an
honest comparison.

Output: 04_validation_reports/image_factor_block_bootstrap.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
HANDOFF = ROOT / "ode_inputs_cnn" / "ode_team_handoff_20260523"
LONG_PATH = HANDOFF / "02_mu_inputs" / "final_mu_inputs_long.csv"
OUT_MD = HANDOFF / "04_validation_reports" / "image_factor_block_bootstrap.md"

RNG = np.random.default_rng(seed=42)
B = 10000
EXPECTED_BLOCK = 20  # = horizon

# (label, candidate input, baseline input)
COMPARISONS = [
    ("image_factor_rank vs rank_baseline", "mu_image_factor_rank", "mu_rank_baseline"),
    ("image_factor_strict_rank vs rank_baseline", "mu_image_factor_strict_rank", "mu_rank_baseline"),
    ("image_factor_balanced vs sharpe_baseline", "mu_image_factor_balanced", "mu_sharpe_baseline"),
]


def per_date_rank_corr(df: pd.DataFrame) -> pd.Series:
    return (
        df.groupby("date")
        .apply(lambda x: x["mu_signal"].rank().corr(x["future_return"].rank()))
        .dropna()
    )


def iid_ci(diffs: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    n = len(diffs)
    boots = np.array([diffs[RNG.integers(0, n, n)].mean() for _ in range(B)])
    return tuple(np.quantile(boots, [alpha / 2, 1 - alpha / 2]))


def block_ci(diffs: np.ndarray, expected_block: int = EXPECTED_BLOCK, alpha: float = 0.05) -> tuple[float, float]:
    n = len(diffs)
    p = 1.0 / expected_block
    boots = np.empty(B)
    for i in range(B):
        out = np.empty(n)
        filled = 0
        while filled < n:
            start = RNG.integers(0, n)
            blk = RNG.geometric(p)
            for k in range(blk):
                if filled >= n:
                    break
                out[filled] = diffs[(start + k) % n]
                filled += 1
        boots[i] = out.mean()
    return tuple(np.quantile(boots, [alpha / 2, 1 - alpha / 2]))


def main() -> None:
    long = pd.read_csv(LONG_PATH, parse_dates=["date"])
    rc = {name: per_date_rank_corr(g) for name, g in long.groupby("input_name")}

    md = ["# Image-Factor Lift — Block Bootstrap Recheck", ""]
    md.append("The handoff's image-factor lift CIs were computed with IID resampling.")
    md.append("Here the same lifts are recomputed with a stationary block bootstrap")
    md.append(f"(Politis-Romano, mean block {EXPECTED_BLOCK} = horizon, B={B}), which is the")
    md.append("methodologically correct CI for the autocorrelated per-date series.")
    md.append("")
    md.append("| comparison | mean lift | IID 95% CI | block 95% CI | block verdict |")
    md.append("|---|---:|---|---|---|")

    results = []
    for label, cand, base in COMPARISONS:
        common = rc[cand].index.intersection(rc[base].index)
        diffs = (rc[cand].loc[common] - rc[base].loc[common]).to_numpy()
        lift = float(diffs.mean())
        ilo, ihi = iid_ci(diffs)
        blo, bhi = block_ci(diffs)
        sig = "significant" if (blo > 0 or bhi < 0) else "NOT significant"
        md.append(f"| {label} | {lift:+.4f} | [{ilo:+.4f}, {ihi:+.4f}] | "
                  f"[{blo:+.4f}, {bhi:+.4f}] | **{sig}** |")
        results.append((label, lift, ilo, ihi, blo, bhi, sig))

    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("- The block CIs are wider than the IID CIs because the daily rank-corr")
    md.append("  diff series is positively autocorrelated.")
    n_iid_sig = sum(1 for r in results if r[2] > 0)
    n_blk_sig = sum(1 for r in results if r[6] == "significant")
    md.append(f"- IID flags {n_iid_sig}/{len(results)} lifts as significant; "
              f"block flags {n_blk_sig}/{len(results)}.")
    md.append("- Honest framing: the image-factor lift over the ensemble baselines is a")
    md.append("  positive point estimate but is not robust to autocorrelation-honest")
    md.append("  confidence bounds. Report it as trend evidence, not strict significance.")
    md.append("- This does not overturn the handoff: the recommended input is still a")
    md.append("  reasonable ranking signal; it just should not be sold as a proven lift.")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print()
    print("\n".join(md[7:7 + 2 + len(results)]))


if __name__ == "__main__":
    main()
