#!/usr/bin/env python3
"""Paired bootstrap CI for ensemble lift comparisons.

Compares:
  (1) ensemble_best (3-member, Phase 3) vs ensemble_4family (4-member, Phase 4)
  (2) cnn_2d_residual_small (single CNN best) vs lstm_image_scale (single LSTM best)

For each pair we compute the per-date rank correlation series, take the daily
diff (paired), and bootstrap the mean diff with B=10000 resamples.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent
OUT = ROOT / "ode_inputs_cnn"
FIG = OUT / "figures"
sns.set_theme(style="whitegrid", context="talk")
RNG = np.random.default_rng(seed=42)
B = 10000


def per_date_rank_corr(df: pd.DataFrame, signal_col: str = "signal_value") -> pd.Series:
    g = df.groupby("date").apply(
        lambda x: x[signal_col].rank().corr(x["future_return"].rank())
    )
    return g.dropna()


def load_ensemble_signal(members: list[tuple[str, str]], aggregation: str = "rank_mean") -> pd.DataFrame:
    frames = []
    for model, src in members:
        df = pd.read_csv(ROOT / src / "walkforward_predictions.csv")
        df["date"] = pd.to_datetime(df["date"], format="mixed").dt.normalize()
        df = df[df["model_name"] == model][["date", "asset", "signal_value", "future_return"]].copy()
        df["model"] = model
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    if aggregation == "rank_mean":
        long["signal_value"] = long.groupby(["date", "model"])["signal_value"].rank(pct=True)
    ens = (
        long.groupby(["date", "asset"], as_index=False)
        .agg(signal_value=("signal_value", "mean"), future_return=("future_return", "first"))
    )
    return ens


def load_single_model(model: str, src: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / src / "walkforward_predictions.csv")
    df["date"] = pd.to_datetime(df["date"], format="mixed").dt.normalize()
    return df[df["model_name"] == model][["date", "asset", "signal_value", "future_return"]].copy()


def paired_bootstrap_ci(diffs: np.ndarray, b: int = B, alpha: float = 0.05) -> tuple[float, float, float]:
    n = len(diffs)
    boots = np.empty(b)
    for i in range(b):
        idx = RNG.integers(0, n, size=n)
        boots[i] = diffs[idx].mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(diffs.mean()), float(lo), float(hi), boots


def compare(name_a: str, series_a: pd.Series, name_b: str, series_b: pd.Series) -> dict:
    common = series_a.index.intersection(series_b.index)
    a = series_a.loc[common].to_numpy()
    b = series_b.loc[common].to_numpy()
    diffs = b - a
    mean_diff, lo, hi, boots = paired_bootstrap_ci(diffs)
    return {
        "name_a": name_a,
        "name_b": name_b,
        "n_dates": len(diffs),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": mean_diff,
        "ci_low": lo,
        "ci_high": hi,
        "significant": (lo > 0) or (hi < 0),
        "boots": boots,
    }


ENSEMBLE_BEST = [
    ("logistic_image_scale", "outputs_walkforward_4model"),
    ("cnn_1d_cumulative_scale", "outputs_walkforward_1dcnn_extra"),
    ("cnn_2d_residual_small", "outputs_walkforward_2d_phase2"),
]
ENSEMBLE_4FAMILY = ENSEMBLE_BEST + [("cnnlstm_image_scale", "cnnlstm")]


def main() -> None:
    print("Loading ensemble signals ...")
    ens3 = load_ensemble_signal(ENSEMBLE_BEST, "rank_mean")
    ens4 = load_ensemble_signal(ENSEMBLE_4FAMILY, "rank_mean")
    rc3 = per_date_rank_corr(ens3)
    rc4 = per_date_rank_corr(ens4)

    print("Loading single-model signals ...")
    cnn = load_single_model("cnn_2d_residual_small", "outputs_walkforward_2d_phase2")
    cnnlstm = load_single_model("cnnlstm_image_scale", "cnnlstm")
    log_img = load_single_model("logistic_image_scale", "outputs_walkforward_4model")
    log_cum = load_single_model("logistic_cumulative_scale", "outputs_walkforward_4model")
    rc_cnn = per_date_rank_corr(cnn)
    rc_cnnlstm = per_date_rank_corr(cnnlstm)
    rc_log_img = per_date_rank_corr(log_img)
    rc_log_cum = per_date_rank_corr(log_cum)

    pair1 = compare("ensemble_best (3)", rc3, "ensemble_4family (4)", rc4)
    pair2 = compare("cnn_2d_residual_small", rc_cnn, "cnnlstm_image_scale", rc_cnnlstm)
    pair3 = compare("logistic_image_scale", rc_log_img, "ensemble_best (3)", rc3)
    pair4 = compare("logistic_image_scale", rc_log_img, "ensemble_4family (4)", rc4)
    pair5 = compare("logistic_cumulative_scale", rc_log_cum, "ensemble_4family (4)", rc4)

    # Markdown report
    md = ["# Bootstrap Significance Test\n"]
    md.append(f"Paired bootstrap (B={B}, 95% CI) on per-date rank correlation series.\n")
    md.append("Positive `mean_diff` = B is better than A on average.\n")

    pairs = [
        ("Pair 1 — ensemble lift (Phase 3 → Phase 4)", pair1),
        ("Pair 2 — single-model best (CNN vs LSTM)", pair2),
        ("Pair 3 — logistic_image vs ensemble_best (Phase 3 lift)", pair3),
        ("Pair 4 — logistic_image vs ensemble_4family (Phase 4 lift)", pair4),
        ("Pair 5 — logistic_cumulative (worst) vs ensemble_4family", pair5),
    ]
    for label, p in pairs:
        md.append(f"## {label}\n")
        md.append(f"- A: `{p['name_a']}` — mean rank corr {p['mean_a']:.4f}")
        md.append(f"- B: `{p['name_b']}` — mean rank corr {p['mean_b']:.4f}")
        md.append(f"- N dates: {p['n_dates']}")
        md.append(f"- mean diff (B − A): **{p['mean_diff']:+.4f}**")
        md.append(f"- 95% CI: [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]")
        verdict = "**significant** (CI excludes 0)" if p["significant"] else "**NOT significant** (CI includes 0)"
        md.append(f"- Verdict: {verdict}\n")

    out_md = OUT / "significance_test.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote {out_md}")
    for line in md:
        print(line)

    # Figure: focus on the 3 key comparisons (drop redundant pair2, pair5)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_specs = [
        (pair3, "logistic_image  →  ensemble_best", "Phase 3 lift (vs baseline)", True),
        (pair4, "logistic_image  →  ensemble_4family", "Phase 4 lift (vs baseline)", True),
        (pair1, "ensemble_best  →  ensemble_4family", "Phase 3 vs 4 (within-ensemble)", False),
    ]
    for ax, (p, title, subtitle, is_sig) in zip(axes, plot_specs):
        # histogram bar color reflects significance
        bar_color = "#5DADE2" if is_sig else "#D5DBDB"
        ax.hist(p["boots"], bins=80, color=bar_color, edgecolor="black", alpha=0.85)
        # axes lines
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
        ax.axvline(p["mean_diff"], color="#c0392b", linewidth=3,
                   label=f"obs Δ = {p['mean_diff']:+.4f}")
        ax.axvline(p["ci_low"], color="#16a085", linestyle=":", linewidth=2.5,
                   label=f"95% CI low = {p['ci_low']:+.4f}")
        ax.axvline(p["ci_high"], color="#16a085", linestyle=":", linewidth=2.5,
                   label=f"95% CI high = {p['ci_high']:+.4f}")
        # title with verdict badge
        verdict_text = "✓ SIGNIFICANT" if is_sig else "borderline"
        verdict_color = "#16a085" if is_sig else "#c0392b"
        ax.set_title(f"{title}\n{subtitle}", fontsize=14, fontweight="bold")
        ax.text(0.02, 0.95, verdict_text, transform=ax.transAxes,
                fontsize=14, fontweight="bold", color=verdict_color,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=verdict_color, linewidth=2))
        ax.set_xlabel("paired Δ rank correlation  (B − A)", fontsize=12)
        ax.set_ylabel("bootstrap frequency", fontsize=11)
        ax.tick_params(labelsize=11)
        ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Paired bootstrap CI (B={B}) — does the lift survive statistical scrutiny?",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_png = FIG / "12_bootstrap_ci.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
