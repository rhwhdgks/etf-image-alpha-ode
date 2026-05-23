#!/usr/bin/env python3
"""Package extra reference mu(t) candidates for the ODE handoff (Tier A).

These are baseline / single-model / classic-factor signals reshaped into the
same schema as final_mu_inputs_long.csv. No retraining: every model source is
an existing walk-forward prediction (same fold generator, so leakage-clean and
directly comparable). momentum and equal-weight are computed here.

Purpose: give the ODE team a wide spectrum (raw floor -> best ensemble) so they
can measure how much each signal layer actually contributes inside the ODE.

Outputs (into 02_mu_inputs/):
  extra_mu_candidates_long.csv   (8 inputs x 2880 dates x 7 assets)
  extra_mu_candidates_wide.csv
  extra_mu_candidates_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
HANDOFF = ROOT / "ode_inputs_cnn" / "ode_team_handoff_20260523" / "02_mu_inputs"
LONG_PATH = HANDOFF / "final_mu_inputs_long.csv"
WF = ROOT / "archive" / "model_exploration" / "walkforward_outputs"
RETURNS_PATH = ROOT / "ode_inputs_cnn" / "returns_daily.csv"

LOOKBACK, HORIZON = 60, 20

# candidate -> (source_csv, model_name, role)
MODEL_SOURCES = {
    "mu_logistic_cumulative": (
        WF / "outputs_walkforward_4model" / "walkforward_predictions.csv",
        "logistic_cumulative_scale", "raw floor baseline: no image, no deep model"),
    "mu_logistic_image": (
        WF / "outputs_walkforward_4model" / "walkforward_predictions.csv",
        "logistic_image_scale", "linear + image baseline"),
    "mu_cnn_2d_residual_small": (
        WF / "outputs_walkforward_2d_phase2" / "walkforward_predictions.csv",
        "cnn_2d_residual_small", "best single CNN (Phase 2 rehab)"),
    "mu_cnn_1d_cumulative": (
        WF / "outputs_walkforward_1dcnn_extra" / "walkforward_predictions.csv",
        "cnn_1d_cumulative_scale", "high-Sharpe single CNN (no image)"),
    "mu_lstm_image": (
        ROOT / "lstm" / "walkforward_predictions.csv",
        "lstm_image_scale", "best single model overall (LSTM family)"),
    "mu_cnnlstm_image": (
        ROOT / "cnnlstm" / "walkforward_predictions.csv",
        "cnnlstm_image_scale", "best single CNN+LSTM hybrid"),
}


def _norm(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="mixed").dt.normalize()


def _cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Add mu_rank / mu_centered_rank / mu_zscore / mu_signal per date."""
    df = df.copy()
    g = df.groupby("date")["mu_raw_score"]
    df["mu_rank"] = g.rank(pct=True)
    rank_mean = df.groupby("date")["mu_rank"].transform("mean")
    df["mu_centered_rank"] = df["mu_rank"] - rank_mean
    mean = g.transform("mean")
    std = g.transform("std")
    df["mu_zscore"] = np.where(std > 1e-12, (df["mu_raw_score"] - mean) / std, 0.0)
    df["mu_signal"] = df["mu_centered_rank"]
    df["mu_signal_type"] = "cross_sectional_centered_rank"
    return df


def main() -> None:
    # canonical date-asset-future_return grid from the handoff
    base = pd.read_csv(LONG_PATH, parse_dates=["date"])
    base = base[base["input_name"] == "mu_rank_baseline"][["date", "asset", "future_return"]].copy()
    base["date"] = _norm(base["date"])
    grid_dates = sorted(base["date"].unique())
    assets = sorted(base["asset"].unique())
    print(f"canonical grid: {len(grid_dates)} dates x {len(assets)} assets = {len(base)} rows")

    candidates = []

    # --- model-sourced candidates ---
    for name, (path, model, role) in MODEL_SOURCES.items():
        wf = pd.read_csv(path)
        wf["date"] = _norm(wf["date"])
        wf = wf[wf["model_name"] == model][["date", "asset", "signal_value"]].copy()
        wf = wf.rename(columns={"signal_value": "mu_raw_score"})
        merged = base.merge(wf, on=["date", "asset"], how="left")
        miss = merged["mu_raw_score"].isna().sum()
        if miss:
            raise ValueError(f"{name}: {miss} rows missing raw score after merge")
        merged["input_name"] = name
        merged["source_model_name"] = model
        merged["input_role"] = role
        merged["source_file"] = str(path.relative_to(ROOT))
        candidates.append(merged)

    # --- momentum: 60-day trailing cumulative log return ---
    ret = pd.read_csv(RETURNS_PATH, parse_dates=["date"])
    ret["date"] = _norm(ret["date"])
    ret = ret.sort_values("date").reset_index(drop=True)
    mom_wide = ret.set_index("date")[assets].rolling(LOOKBACK).sum()
    mom_long = mom_wide.reset_index().melt(id_vars="date", var_name="asset", value_name="mu_raw_score")
    mom = base.merge(mom_long, on=["date", "asset"], how="left")
    if mom["mu_raw_score"].isna().any():
        raise ValueError("momentum: NaN raw score (insufficient warmup)")
    mom["input_name"] = "mu_momentum_60d"
    mom["source_model_name"] = "momentum_60d_cumret"
    mom["input_role"] = "classic factor baseline: 60-day trailing cumulative return"
    mom["source_file"] = str(RETURNS_PATH.relative_to(ROOT))
    candidates.append(mom)

    # --- equal-weight: null signal (constant) ---
    eq = base.copy()
    eq["mu_raw_score"] = 0.0
    eq["input_name"] = "mu_equal_weight"
    eq["source_model_name"] = "equal_weight_null"
    eq["input_role"] = "null floor: constant signal, ODE reduces to Sigma-only / equal weight"
    eq["source_file"] = "n/a"
    candidates.append(eq)

    # --- assemble + cross-sectional transforms ---
    long = pd.concat(candidates, ignore_index=True)
    long = _cross_sectional(long)
    long["is_recommended"] = False
    long["lookback"] = LOOKBACK
    long["horizon"] = HORIZON
    long["chart_variant"] = "n/a"
    long["strict_window_ma"] = False

    col_order = [
        "date", "asset", "input_name", "source_model_name", "mu_signal", "mu_signal_type",
        "mu_raw_score", "mu_rank", "mu_centered_rank", "mu_zscore", "future_return",
        "is_recommended", "input_role", "lookback", "horizon", "chart_variant",
        "strict_window_ma", "source_file",
    ]
    long = long[col_order].sort_values(["input_name", "date", "asset"]).reset_index(drop=True)

    long_export = long.copy()
    long_export["date"] = long_export["date"].dt.strftime("%Y-%m-%d")
    long_export.to_csv(HANDOFF / "extra_mu_candidates_long.csv", index=False)

    wide = long.pivot_table(index=["date", "asset"], columns="input_name", values="mu_signal")
    wide.columns = [f"mu_signal_{c}" for c in wide.columns]
    wide = wide.reset_index()
    fr = long[["date", "asset", "future_return"]].drop_duplicates(["date", "asset"])
    wide = wide.merge(fr, on=["date", "asset"])
    wide_export = wide.copy()
    wide_export["date"] = wide_export["date"].dt.strftime("%Y-%m-%d")
    wide_export.to_csv(HANDOFF / "extra_mu_candidates_wide.csv", index=False)

    # --- summary: recompute rank corr + top-k Sharpe as sanity check ---
    import math
    rows = []
    for name, g in long.groupby("input_name"):
        per_date = g.groupby("date").apply(
            lambda x: x["mu_signal"].rank().corr(x["future_return"].rank()))
        rc = float(per_date.dropna().mean())
        # top-2 Sharpe, rebal every horizon
        dts = sorted(g["date"].unique())[::HORIZON]
        rets = []
        for d in dts:
            fr_d = g[g["date"] == d].sort_values("mu_signal", ascending=False)
            if len(fr_d) >= 2:
                rets.append(float(fr_d.head(2)["future_return"].mean()))
        r = np.array(rets)
        sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(252.0 / HORIZON)) if len(r) > 1 and r.std() > 0 else float("nan")
        rows.append({"input_name": name, "rank_corr": round(rc, 4),
                     "top_k_sharpe": round(sharpe, 4), "n_dates": g["date"].nunique()})
    summary = pd.DataFrame(rows).sort_values("rank_corr", ascending=False)
    summary.to_csv(HANDOFF / "extra_mu_candidates_summary.csv", index=False)

    print(f"wrote extra_mu_candidates_long.csv ({len(long)} rows), wide, summary")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
