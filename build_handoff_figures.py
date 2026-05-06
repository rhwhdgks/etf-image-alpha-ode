#!/usr/bin/env python3
"""Generate presentation-ready figures for the CNN hand-off package.

Outputs land in ode_inputs_cnn/figures/ and are embedded from
ode_inputs_cnn/HANDOFF_SUMMARY.md.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent / "ode_inputs_cnn"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

ASSETS = [
    "alternative",
    "corp_bond_ig",
    "developed_equity",
    "emerging_equity",
    "korea_equity",
    "short_treasury",
    "treasury_7_10y",
]

MODEL_DIRS = [
    "cnn_1d_cumulative_scale",
    "cnn_1d_image_scale",
    "cnn_1d_multiscale_image_scale",
    "cnn_1d_dilated_image_scale",
    "cnn_1d_attention_image_scale",
    "cnn_2d_rendered_images",
    "cnn_2d_residual_images",
    "cnn_2d_residual_small",
    "lstm_image_scale",
    "lstm_cumulative_scale",
    "cnnlstm_image_scale",
    "cnnlstm_cumulative_scale",
    "ensemble_top3",
    "ensemble_best",
    "ensemble_4family",
]

sns.set_theme(style="whitegrid", context="talk")
PALETTE = sns.color_palette("tab20", n_colors=len(MODEL_DIRS))
MODEL_COLOR = {m: PALETTE[i] for i, m in enumerate(MODEL_DIRS)}
ENSEMBLE_COLOR = "#c0392b"


def save(fig, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT.parent)}")


def fig_model_comparison(comparison: pd.DataFrame) -> None:
    order = comparison.sort_values("oos_rank_corr_mean", ascending=True)
    colors = [ENSEMBLE_COLOR if m == "ensemble_top3" else "#4c72b0" for m in order["model_name"]]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    axes[0].barh(order["model_name"], order["oos_rank_corr_mean"], color=colors, edgecolor="black", linewidth=0.5)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("OOS Rank Correlation (mean)")
    axes[0].set_title("Spearman rank corr vs future 20-day return")

    axes[1].barh(order["model_name"], order["oos_pct_positive"] * 100, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].axvline(50, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Positive-day pct (%)")
    axes[1].set_title("Share of folds/dates with positive rank corr")

    fig.suptitle("CNN Walk-Forward OOS — Model Comparison", fontsize=18, y=1.02)
    save(fig, "01_model_comparison.png")


def fig_mu_timeseries(ensemble_bundle: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(ASSETS), 1, figsize=(14, 2.1 * len(ASSETS)), sharex=True)
    for ax, asset in zip(axes, ASSETS):
        col = f"{asset}_mu"
        series = ensemble_bundle[["date", col]].dropna()
        ax.plot(series["date"], series[col], color=MODEL_COLOR["ensemble_top3"], linewidth=0.9)
        ax.axhline(0, color="grey", linewidth=0.6, alpha=0.5)
        ax.set_ylabel(asset, rotation=0, labelpad=70, ha="right", fontsize=11)
        ax.set_ylim(-0.003, 0.003)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Ensemble $\\hat\\mu_t$ (daily return scale) per asset", fontsize=16, y=0.995)
    save(fig, "02_mu_timeseries_ensemble.png")


def fig_mu_distribution(ensemble_mu: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.violinplot(
        data=ensemble_mu,
        x="asset",
        y="mu_hat_daily",
        order=ASSETS,
        ax=ax,
        inner="quartile",
        color="#4c72b0",
    )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylim(-0.003, 0.003)
    ax.set_xlabel("")
    ax.set_ylabel("$\\hat\\mu_t$ (daily return scale)")
    ax.set_title("Ensemble calibrated $\\hat\\mu$ distribution per asset")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    save(fig, "03_mu_distribution_ensemble.png")


def fig_sigma_condition(ensemble_bundle: pd.DataFrame) -> None:
    records = []
    n = len(ASSETS)
    for _, row in ensemble_bundle.iterrows():
        try:
            sigma = np.zeros((n, n))
            for i, a in enumerate(ASSETS):
                sigma[i, i] = row[f"{a}_sigma_ii"]
            for i, a in enumerate(ASSETS):
                for j, b in enumerate(ASSETS):
                    if i >= j:
                        continue
                    key = f"{a}_{b}_cov"
                    if key not in ensemble_bundle.columns:
                        key = f"{b}_{a}_cov"
                    sigma[i, j] = sigma[j, i] = row[key]
            if np.any(np.isnan(sigma)):
                continue
            w = np.linalg.eigvalsh(sigma)
            if w.min() <= 0:
                continue
            records.append({"date": row["date"], "cond": w.max() / w.min()})
        except Exception:
            continue
    cond_df = pd.DataFrame(records).sort_values("date")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cond_df["date"], cond_df["cond"], color="#34495e", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_ylabel("cond(Σ)  (log scale)")
    ax.set_title(f"Rolling 60-day covariance condition number (median = {cond_df['cond'].median():.0f})")
    save(fig, "04_sigma_condition_number.png")


def fig_model_correlation(all_raw: pd.DataFrame) -> None:
    """Correlation matrix including logistic baselines (loaded separately from
    walkforward CSVs) since logistic models have no per-model bundle dir."""
    base_models = [m for m in MODEL_DIRS if not m.startswith("ensemble_")]

    wide_cnn = (
        all_raw.pivot_table(index=["date", "asset"], columns="model", values="mu_raw_score")
        .reindex(columns=base_models)
    )

    LOGISTIC_SRC = {
        "logistic_image_scale": "outputs_walkforward_4model",
        "logistic_cumulative_scale": "outputs_walkforward_4model",
    }
    log_frames = []
    repo_root = ROOT.parent
    for m, d in LOGISTIC_SRC.items():
        df = pd.read_csv(repo_root / d / "walkforward_predictions.csv")
        df["date"] = pd.to_datetime(df["date"], format="mixed").dt.normalize()
        df = df[df["model_name"] == m][["date", "asset", "signal_value"]].copy()
        df = df.rename(columns={"signal_value": m}).set_index(["date", "asset"])
        log_frames.append(df)
    wide_log = pd.concat(log_frames, axis=1)
    wide = pd.concat([wide_cnn, wide_log], axis=1).dropna()
    corr = wide.corr()

    label_map = {
        "logistic_image_scale":          "logistic_image",
        "logistic_cumulative_scale":     "logistic_cumulative",
        "cnn_1d_cumulative_scale":       "1d_cumulative",
        "cnn_1d_image_scale":            "1d_image",
        "cnn_1d_multiscale_image_scale": "1d_multiscale",
        "cnn_1d_dilated_image_scale":    "1d_dilated",
        "cnn_1d_attention_image_scale":  "1d_attention",
        "cnn_2d_rendered_images":        "2d_rendered",
        "cnn_2d_residual_images":        "2d_residual",
        "cnn_2d_residual_small":         "2d_residual_small ★",
        "lstm_image_scale":              "lstm_image",
        "lstm_cumulative_scale":         "lstm_cumulative",
        "cnnlstm_image_scale":           "cnnlstm_image ★",
        "cnnlstm_cumulative_scale":      "cnnlstm_cumulative",
    }
    # Order: logistic first, then CNN 1D, CNN 2D, LSTM, hybrid
    ordered = [
        "logistic_image_scale", "logistic_cumulative_scale",
        "cnn_1d_image_scale", "cnn_1d_cumulative_scale",
        "cnn_1d_multiscale_image_scale", "cnn_1d_dilated_image_scale", "cnn_1d_attention_image_scale",
        "cnn_2d_rendered_images", "cnn_2d_residual_images", "cnn_2d_residual_small",
        "lstm_image_scale", "lstm_cumulative_scale",
        "cnnlstm_image_scale", "cnnlstm_cumulative_scale",
    ]
    corr = corr.reindex(index=ordered, columns=ordered)
    pretty = [label_map.get(m, m) for m in ordered]
    corr.index = pretty
    corr.columns = pretty

    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Pearson ρ (raw signal)", "shrink": 0.8},
        square=True,
    )
    ax.set_title("Model raw-signal correlation matrix (★ = ensemble_4family member)", fontsize=15, pad=14)
    ax.tick_params(axis="x", labelsize=11, rotation=40)
    ax.tick_params(axis="y", labelsize=11, rotation=0)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    save(fig, "05_model_raw_correlation.png")


def fig_rolling_rank_corr(all_preds: pd.DataFrame) -> None:
    def _daily_rank_corr(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return np.nan
        return group["mu_raw_score"].rank().corr(group["future_return"].rank(), method="pearson")

    fig, ax = plt.subplots(figsize=(14, 6))
    for model in MODEL_DIRS:
        sub = all_preds[all_preds["model"] == model]
        if sub.empty:
            continue
        daily = sub.groupby("date").apply(_daily_rank_corr, include_groups=False).dropna()
        if daily.empty:
            continue
        rolled = daily.rolling(window=60, min_periods=30).mean()
        color = ENSEMBLE_COLOR if model == "ensemble_top3" else MODEL_COLOR[model]
        lw = 2.0 if model == "ensemble_top3" else 1.0
        alpha = 1.0 if model == "ensemble_top3" else 0.55
        ax.plot(rolled.index, rolled.values, label=model, color=color, linewidth=lw, alpha=alpha)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("60-day rolling rank corr")
    ax.set_xlabel("Date")
    ax.set_title("Signal quality stability over time")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    save(fig, "06_rolling_rank_corr.png")


def fig_coverage(ensemble_bundle: pd.DataFrame) -> None:
    cols = [f"{a}_mu" for a in ASSETS]
    bundle = ensemble_bundle[["date"] + cols].copy()
    avail = (~bundle[cols].isna()).astype(int).T
    avail.index = ASSETS

    fig, ax = plt.subplots(figsize=(14, 4))
    dates = pd.to_datetime(bundle["date"])
    ax.imshow(
        avail.values,
        aspect="auto",
        cmap="Greens",
        interpolation="nearest",
        extent=(dates.min().toordinal(), dates.max().toordinal(), len(ASSETS), 0),
    )
    ax.set_yticks(np.arange(len(ASSETS)) + 0.5)
    ax.set_yticklabels(ASSETS)
    tick_dates = pd.date_range(dates.min(), dates.max(), periods=8)
    ax.set_xticks([d.toordinal() for d in tick_dates])
    ax.set_xticklabels([d.strftime("%Y-%m") for d in tick_dates])
    ax.set_title("μ coverage (green = available) — ensemble bundle")
    save(fig, "07_coverage_heatmap.png")


def fig_risk_timeseries(ensemble_bundle: pd.DataFrame) -> None:
    cols = [f"{a}_risk" for a in ASSETS]
    data = ensemble_bundle[["date"] + cols].dropna()

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, asset in enumerate(ASSETS):
        ax.plot(data["date"], data[f"{asset}_risk"], label=asset, color=PALETTE[i], linewidth=0.8, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("risk z-score")
    ax.set_xlabel("Date")
    ax.set_title("Cross-sectional risk z-score per asset (ensemble bundle)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    save(fig, "08_risk_zscore_timeseries.png")


def load_all_raw() -> pd.DataFrame:
    frames = []
    for model in MODEL_DIRS:
        path = ROOT / model / "mu_daily.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["date", "asset", "mu_raw_score", "future_return"])
        df["model"] = model
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    print("Loading inputs ...")
    comparison = pd.read_csv(ROOT / "comparison.csv")
    ensemble_bundle = pd.read_csv(ROOT / "ensemble_top3" / "ode_bundle.csv", parse_dates=["date"])
    ensemble_mu = pd.read_csv(ROOT / "ensemble_top3" / "mu_daily.csv", parse_dates=["date"])
    all_raw = load_all_raw()
    all_raw["date"] = pd.to_datetime(all_raw["date"])

    print("Building figures ...")
    fig_model_comparison(comparison)
    fig_mu_timeseries(ensemble_bundle)
    fig_mu_distribution(ensemble_mu)
    fig_sigma_condition(ensemble_bundle)
    fig_model_correlation(all_raw)
    fig_rolling_rank_corr(all_raw)
    fig_coverage(ensemble_bundle)
    fig_risk_timeseries(ensemble_bundle)

    print(f"\nAll figures saved under {FIG_DIR}/")


if __name__ == "__main__":
    main()
