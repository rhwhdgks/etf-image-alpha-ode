from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.data.loader import OPTIONAL_FIELDS, REQUIRED_PRICE_FIELDS, build_dataset_summary, load_etf_csv, restrict_common_valid_sample
from src.eval.metrics import compute_prediction_metrics, mean_rank_correlation, top_k_backtest
from src.features.labels import get_target_metadata
from src.models.cnn import fit_torch_model, predict_torch_model
from src.models.lstm import build_lstm_model
from src.pipeline import build_samples, set_global_seed, _prediction_frame
from src.walkforward import generate_walkforward_folds


LSTM_MODELS = [
    "lstm_image_scale",
    "lstm_cumulative_scale",
    "cnn_lstm_image_scale",
    "cnn_lstm_cumulative_scale",
]


def build_lstm_feature_sets(bundle, n: int, enabled_models: list[str] | None = None) -> dict[str, np.ndarray]:
    all_sets = {
        "lstm_image_scale": np.transpose(bundle.image_sequences, (0, 2, 1)),
        "lstm_cumulative_scale": np.transpose(bundle.cumulative_sequences, (0, 2, 1)),
        "cnn_lstm_image_scale": np.transpose(bundle.image_sequences, (0, 2, 1)),
        "cnn_lstm_cumulative_scale": np.transpose(bundle.cumulative_sequences, (0, 2, 1)),
    }
    if enabled_models:
        chosen = set(enabled_models)
        all_sets = {k: v for k, v in all_sets.items() if k in chosen}
    if not all_sets:
        raise ValueError("no LSTM model selected")
    return all_sets


def fit_predict_lstm_repeated(model_name, train_x, train_y, val_x, val_y, test_x, config):
    repeats = max(1, int(config.cnn_repeats))
    all_scores = []
    for repeat_idx in range(repeats):
        set_global_seed(config.seed + repeat_idx)
        model = fit_torch_model(
            build_lstm_model(model_name, tuple(train_x.shape[1:])),
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            label_mode=config.label_mode,
            epochs=config.cnn_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            patience=config.patience,
            device=config.device,
        )
        scores, _ = predict_torch_model(model, test_x, config.label_mode, batch_size=config.batch_size)
        all_scores.append(scores.astype(float))
    scores = np.vstack(all_scores).mean(axis=0)
    confidence = np.abs(scores - 0.5) * 2.0 if config.label_mode == "classification" else None
    return scores, confidence


def run_lstm_walkforward(bundle, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = bundle.metadata
    unique_dates = sorted(metadata["date"].drop_duplicates().tolist())
    folds = generate_walkforward_folds(unique_dates, config)
    if not folds:
        raise ValueError("not enough dates for walk-forward")

    feature_sets = build_lstm_feature_sets(bundle, len(metadata), config.enabled_models or LSTM_MODELS)
    target_meta = get_target_metadata(config.label_mode, config.target_name)
    selection_sign = float(target_meta["selection_sign"])
    target = metadata["target"].to_numpy(dtype=np.float32)

    all_predictions = []
    for fold_idx, fold in enumerate(folds):
        train_mask = metadata["date"].isin(fold["train_dates"]).to_numpy()
        val_mask = metadata["date"].isin(fold["val_dates"]).to_numpy()
        test_mask = metadata["date"].isin(fold["test_dates"]).to_numpy()
        print(f"fold {fold_idx + 1}/{len(folds)}: train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}")

        for model_name, features in feature_sets.items():
            scores, confidence = fit_predict_lstm_repeated(
                model_name,
                features[train_mask], target[train_mask],
                features[val_mask], target[val_mask],
                features[test_mask],
                config,
            )
            pred = _prediction_frame(metadata[test_mask], model_name, scores, confidence, selection_sign)
            pred["fold"] = fold_idx
            all_predictions.append(pred)

    oos_df = pd.concat(all_predictions, ignore_index=True)
    rows = []
    for model_name, group in oos_df.groupby("model_name"):
        group = group.copy()
        metrics = compute_prediction_metrics(group, config.label_mode)
        if config.label_mode == "classification":
            metrics["rank_correlation"] = mean_rank_correlation(group, "future_return", "selection_score")
        else:
            metrics["target_rank_correlation"] = mean_rank_correlation(group, "target", "signal_value")
            metrics["future_return_rank_correlation"] = mean_rank_correlation(group, "future_return", "selection_score")
        metrics.update(top_k_backtest(group, config.horizon, config.top_k, score_column="selection_score"))
        metrics["model_name"] = model_name
        metrics["n_folds"] = int(group["fold"].nunique())
        rows.append(metrics)

    comparison_df = pd.DataFrame(rows).sort_values(
        [c for c in ["top_k_sharpe", "future_return_rank_correlation", "rank_correlation", "rmse"] if c in pd.DataFrame(rows).columns],
        ascending=[False, False, False, True][:len([c for c in ["top_k_sharpe", "future_return_rank_correlation", "rank_correlation", "rmse"] if c in pd.DataFrame(rows).columns])],
        na_position="last",
    ).reset_index(drop=True)
    return oos_df, comparison_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="03_data/etfdata.csv")
    parser.add_argument("--output-dir", default="results/lstm_walkforward")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--label-mode", choices=["classification", "regression"], default="regression")
    parser.add_argument("--target-name", default="future_return")
    parser.add_argument("--models", nargs="*", default=LSTM_MODELS)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = PipelineConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        lookback=args.lookback,
        horizon=args.horizon,
        label_mode=args.label_mode,
        target_name=args.target_name,
        enabled_models=args.models,
        cnn_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device=args.device,
    )

    set_global_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    raw_df, long_panel, assets, fields, coverage_df = load_etf_csv(config.data_path)
    required = REQUIRED_PRICE_FIELDS.copy()
    if config.include_volume and "volume" in fields:
        required.extend(OPTIONAL_FIELDS)
    common_panel, _ = restrict_common_valid_sample(long_panel, selected_assets=config.selected_assets or assets, required_fields=required)
    bundle, _ = build_samples(common_panel, config)

    oos_df, comparison_df = run_lstm_walkforward(bundle, config)
    pred_path = Path(config.output_dir) / "lstm_walkforward_predictions.csv"
    comp_path = Path(config.output_dir) / "lstm_walkforward_comparison.csv"
    best_path = Path(config.output_dir) / "lstm_best_predictions.csv"

    oos_df.to_csv(pred_path, index=False)
    comparison_df.to_csv(comp_path, index=False)
    best_model = comparison_df.iloc[0]["model_name"]
    oos_df[oos_df["model_name"] == best_model].to_csv(best_path, index=False)

    print(comparison_df)
    print(f"saved: {pred_path}")
    print(f"best model: {best_model}")
    print(f"best predictions: {best_path}")


if __name__ == "__main__":
    main()
