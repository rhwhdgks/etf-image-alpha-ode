#!/usr/bin/env python3
"""Build the `ensemble_4family/` bundle — Phase-4 winner with CNN+LSTM hybrid.

Selected via Phase-4 ensemble_search.csv (mode=rank, k=4) after the teammate's
improved CNN+LSTM hybrid (`cnnlstm_image_scale`) replaced the original hybrid:
    logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small
    + cnnlstm_image_scale
  → OOS rank corr 0.0671 (#1), top-k Sharpe 0.541 (vs ensemble_best Sharpe 0.630).

Note: legacy ensemble_best (3-member, Sharpe 0.630) remains the recommended
default for Sharpe-prioritized usage. ensemble_4family is the rank-corr
prioritized alternative with full family diversity (logistic + 1D CNN + 2D CNN
+ LSTM).

Aggregation is cross-sectional percentile rank per (date, model), averaged
across members.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from collect_cnn_ode_signals import build_per_model_ode
from make_ode_inputs import compute_rolling_sigma

ROOT = Path(__file__).parent
OUT = ROOT / "ode_inputs_cnn"

MEMBERS = {
    "logistic_image_scale":     "outputs_walkforward_4model",
    "cnn_1d_cumulative_scale":  "outputs_walkforward_1dcnn_extra",
    "cnn_2d_residual_small":    "outputs_walkforward_2d_phase2",
    "cnnlstm_image_scale":      "cnnlstm",
}
BUNDLE_NAME = "ensemble_4family"
RISK_PATH = ROOT / "outputs_walkforward_risk" / "walkforward_predictions.csv"
RISK_MODEL = "logistic_cumulative_scale"
DATA_PATH = ROOT / "etfdata.csv"
SIGMA_WINDOW = 60
HORIZON = 20


def load_members() -> pd.DataFrame:
    frames = []
    for model, d in MEMBERS.items():
        df = pd.read_csv(ROOT / d / "walkforward_predictions.csv")
        df["date"] = pd.to_datetime(df["date"], format="mixed").dt.normalize()
        df = df[df["model_name"] == model].copy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_rank_ensemble(members_df: pd.DataFrame) -> pd.DataFrame:
    members_df = members_df.copy()
    members_df["signal_rank"] = (
        members_df.groupby(["date", "model_name"])["signal_value"].rank(pct=True)
    )
    grouped = (
        members_df.groupby(["date", "asset"], as_index=False)
        .agg(
            signal_value=("signal_rank", "mean"),
            future_return=("future_return", "first"),
            target=("target", "first"),
            fold=("fold", "first"),
        )
    )
    grouped["model_name"] = BUNDLE_NAME
    grouped["selection_score"] = grouped["signal_value"]
    grouped["confidence"] = grouped["signal_value"]
    return grouped.sort_values(["date", "asset"]).reset_index(drop=True)


def main() -> None:
    print(f"Loading {len(MEMBERS)} members ...")
    members_df = load_members()
    print(f"  rows={len(members_df)} unique dates={members_df['date'].nunique()}")

    print("Building rank-averaged ensemble signal ...")
    ensemble_df = build_rank_ensemble(members_df)
    print(f"  ensemble rows: {len(ensemble_df)}")

    print("Loading risk signal ...")
    risk_raw = pd.read_csv(RISK_PATH, parse_dates=["date"])
    risk_df = risk_raw[risk_raw["model_name"] == RISK_MODEL].copy()

    assets = sorted(ensemble_df["asset"].unique())
    print(f"Computing rolling Sigma (window={SIGMA_WINDOW}) ...")
    sigma_df = compute_rolling_sigma(str(DATA_PATH), assets, window=SIGMA_WINDOW)

    print(f"Packaging {BUNDLE_NAME}/ ...")
    quality = build_per_model_ode(
        model_df=ensemble_df,
        model_name=BUNDLE_NAME,
        risk_df=risk_df,
        sigma_df=sigma_df,
        horizon=HORIZON,
        sigma_window=SIGMA_WINDOW,
        output_dir=OUT,
        risk_model_name=RISK_MODEL,
    )

    cfg_path = OUT / BUNDLE_NAME / "ode_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["ensemble_members"] = list(MEMBERS.keys())
    cfg["aggregation"] = "cross_sectional_percentile_rank_mean"
    cfg["selection_rationale"] = (
        "Phase-4 winner from ensemble_search.csv (rank corr sort, k=4): "
        "logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small "
        "+ cnnlstm_image_scale (teammate's improved CNN+LSTM hybrid). "
        "OOS rank corr 0.0671 (#1), top-k Sharpe 0.541. Both metrics improved "
        "vs the previous lstm_image_scale variant (0.0653 / 0.452). Note that "
        "ensemble_best (3-member, no LSTM) still has the highest Sharpe (0.630). "
        "Use ensemble_4family when point-prediction quality matters more; "
        "ensemble_best for portfolio Sharpe priority."
    )
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print("\n=== Quality ===")
    for k, v in quality.items():
        print(f"  {k}: {v}")
    print(f"\nBundle written to: {OUT/BUNDLE_NAME}/")


if __name__ == "__main__":
    main()
