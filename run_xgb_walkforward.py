#!/usr/bin/env python3
"""Walk-forward OOS evaluation for the XGBoost 5th model family.

Adds a gradient-boosted-tree family to the existing CNN / logistic / LSTM /
CNN+LSTM families. Trees have a different inductive bias (non-smooth, automatic
feature interaction), so they are expected to be weakly correlated with the
deep-learning families — useful for ensemble diversification.

Uses the identical walk-forward fold grid as every other model, so predictions
are leakage-clean and directly comparable / poolable into the ensemble search.

Output: xgb/walkforward_predictions.csv  (+ comparison + best)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import PipelineConfig
from src.data.loader import OPTIONAL_FIELDS, REQUIRED_PRICE_FIELDS, load_etf_csv, restrict_common_valid_sample
from src.eval.metrics import compute_prediction_metrics, mean_rank_correlation, top_k_backtest
from src.features.labels import get_target_metadata
from src.pipeline import build_samples, set_global_seed, _prediction_frame
from src.walkforward import generate_walkforward_folds

XGB_MODELS = ["xgb_image_scale", "xgb_cumulative_scale"]


def build_xgb_feature_sets(bundle, enabled: list[str] | None) -> dict[str, np.ndarray]:
    n = len(bundle.metadata)
    sets = {
        "xgb_image_scale": bundle.image_sequences.reshape(n, -1),
        "xgb_cumulative_scale": bundle.cumulative_sequences.reshape(n, -1),
    }
    if enabled:
        sets = {k: v for k, v in sets.items() if k in set(enabled)}
    if not sets:
        raise ValueError("no XGBoost model selected")
    return sets


def fit_predict_xgb_repeated(train_x, train_y, val_x, val_y, test_x, config) -> np.ndarray:
    repeats = max(1, int(config.cnn_repeats))
    all_scores = []
    for repeat_idx in range(repeats):
        set_global_seed(config.seed + repeat_idx)
        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_lambda=2.0,
            min_child_weight=5,
            random_state=config.seed + repeat_idx,
            n_jobs=-1,
            early_stopping_rounds=40,
        )
        model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
        all_scores.append(model.predict(test_x).astype(float))
    return np.vstack(all_scores).mean(axis=0)


def run_xgb_walkforward(bundle, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = bundle.metadata
    unique_dates = sorted(metadata["date"].drop_duplicates().tolist())
    folds = generate_walkforward_folds(unique_dates, config)
    if not folds:
        raise ValueError("not enough dates for walk-forward")

    feature_sets = build_xgb_feature_sets(bundle, config.enabled_models or XGB_MODELS)
    target_meta = get_target_metadata(config.label_mode, config.target_name)
    selection_sign = float(target_meta["selection_sign"])
    target = metadata["target"].to_numpy(dtype=np.float32)

    all_predictions = []
    for fold_idx, fold in enumerate(folds):
        train_mask = metadata["date"].isin(fold["train_dates"]).to_numpy()
        val_mask = metadata["date"].isin(fold["val_dates"]).to_numpy()
        test_mask = metadata["date"].isin(fold["test_dates"]).to_numpy()
        print(f"fold {fold_idx + 1}/{len(folds)}: "
              f"train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}", flush=True)

        for model_name, features in feature_sets.items():
            scores = fit_predict_xgb_repeated(
                features[train_mask], target[train_mask],
                features[val_mask], target[val_mask],
                features[test_mask], config,
            )
            pred = _prediction_frame(metadata[test_mask], model_name, scores, None, selection_sign)
            pred["fold"] = fold_idx
            all_predictions.append(pred)

    oos_df = pd.concat(all_predictions, ignore_index=True)
    rows = []
    for model_name, group in oos_df.groupby("model_name"):
        metrics = compute_prediction_metrics(group, config.label_mode)
        metrics["target_rank_correlation"] = mean_rank_correlation(group, "target", "signal_value")
        metrics["future_return_rank_correlation"] = mean_rank_correlation(group, "future_return", "selection_score")
        metrics.update(top_k_backtest(group, config.horizon, config.top_k, score_column="selection_score"))
        metrics["model_name"] = model_name
        metrics["n_folds"] = int(group["fold"].nunique())
        rows.append(metrics)
    comparison_df = pd.DataFrame(rows).sort_values("future_return_rank_correlation", ascending=False).reset_index(drop=True)
    return oos_df, comparison_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="etfdata.csv")
    parser.add_argument("--output-dir", default="xgb")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--models", nargs="*", default=XGB_MODELS)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    config = PipelineConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        lookback=args.lookback,
        horizon=args.horizon,
        label_mode="regression",
        target_name="future_return",
        enabled_models=args.models,
        cnn_repeats=args.repeats,
    )

    set_global_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    raw_df, long_panel, assets, fields, coverage_df = load_etf_csv(config.data_path)
    required = REQUIRED_PRICE_FIELDS.copy()
    if config.include_volume and "volume" in fields:
        required.extend(OPTIONAL_FIELDS)
    common_panel, _ = restrict_common_valid_sample(
        long_panel, selected_assets=config.selected_assets or assets, required_fields=required)
    bundle, _ = build_samples(common_panel, config)

    oos_df, comparison_df = run_xgb_walkforward(bundle, config)
    out = Path(config.output_dir)
    oos_df.to_csv(out / "walkforward_predictions.csv", index=False)
    comparison_df.to_csv(out / "walkforward_comparison.csv", index=False)
    best = comparison_df.iloc[0]["model_name"]
    oos_df[oos_df["model_name"] == best].to_csv(out / "walkforward_best_predictions.csv", index=False)

    print()
    print(comparison_df[["model_name", "future_return_rank_correlation", "top_k_sharpe", "n_folds"]].to_string(index=False))
    print(f"\nsaved: {out}/walkforward_predictions.csv")
    print(f"best: {best}")


if __name__ == "__main__":
    main()
