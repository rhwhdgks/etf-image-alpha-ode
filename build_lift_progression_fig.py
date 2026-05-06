#!/usr/bin/env python3
"""3단계 lift progression figure: 단일 best → CNN+logistic mix → 4-family mix.

Two grouped bars per stage: rank corr (left axis) and top-k Sharpe (right axis).
Reads numbers from comparison_with_baselines.csv.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent
OUT = ROOT / "ode_inputs_cnn"
FIG = OUT / "figures"
sns.set_theme(style="whitegrid", context="talk")


STAGES = [
    ("Single best\n(cnn_2d_residual_small)", "cnn_2d_residual_small"),
    ("Phase 3 mix\n(logistic + 1D + 2D)", "ensemble_best"),
    ("Phase 4 mix\n(+ cnnlstm, 4-family)", "ensemble_4family"),
]


def main() -> None:
    df = pd.read_csv(OUT / "comparison_with_baselines.csv").set_index("model_name")
    rank_corr = [df.loc[m, "future_return_rank_correlation"] for _, m in STAGES]
    sharpe = [df.loc[m, "top_k_sharpe"] for _, m in STAGES]
    labels = [s for s, _ in STAGES]

    x = np.arange(len(STAGES))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6.5))
    ax2 = ax1.twinx()
    sns.set_style("whitegrid")

    bars1 = ax1.bar(x - width / 2, rank_corr, width, color="#2980b9", label="OOS rank corr", edgecolor="black")
    bars2 = ax2.bar(x + width / 2, sharpe, width, color="#e67e22", label="Top-k Sharpe", edgecolor="black")

    for b, v in zip(bars1, rank_corr):
        ax1.annotate(f"{v:+.4f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=12, fontweight="bold", color="#1f4e6e")
    for b, v in zip(bars2, sharpe):
        ax2.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=12, fontweight="bold", color="#a65000")

    # delta annotations — both on ax1 (consistent coords) at separate y positions
    rc_lifts = [rank_corr[i] - rank_corr[i - 1] for i in range(1, len(rank_corr))]
    sh_lifts = [sharpe[i] - sharpe[i - 1] for i in range(1, len(sharpe))]
    rc_y = max(rank_corr) * 1.30
    sh_y = max(rank_corr) * 1.18
    for i, (rcd, shd) in enumerate(zip(rc_lifts, sh_lifts), start=1):
        mid = (x[i - 1] + x[i]) / 2
        rc_color = "#16a085" if rcd > 0 else "#c0392b"
        sh_color = "#16a085" if shd > 0 else "#c0392b"
        ax1.annotate(f"Δrank {rcd:+.4f}", (mid, rc_y),
                     ha="center", fontsize=12, fontweight="bold", color=rc_color)
        ax1.annotate(f"ΔSharpe {shd:+.3f}", (mid, sh_y),
                     ha="center", fontsize=12, fontweight="bold", color=sh_color)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_ylabel("OOS rank correlation", color="#2980b9", fontsize=14)
    ax2.set_ylabel("Top-k Sharpe (annualized)", color="#e67e22", fontsize=14)
    ax1.tick_params(axis="y", colors="#2980b9")
    ax2.tick_params(axis="y", colors="#e67e22")
    ax1.set_ylim(0, max(rank_corr) * 1.45)
    ax2.set_ylim(0, max(sharpe) * 1.45)

    fig.suptitle("Lift progression — single → 3-family → 4-family ensemble", fontsize=16, y=1.00)
    fig.tight_layout()
    out_path = FIG / "11_3stage_lift_progression.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
