#!/usr/bin/env python3
r"""Walk-forward runner for CNNLSTM models.

Assumption
----------
You already created this file:

    src/models/cnnlstm.py

and it defines:

    class CNNLSTMRegressor(nn.Module):
        ...

Expected input shape for CNNLSTMRegressor:

    (channels, lookback)

This script intentionally does NOT define the CNNLSTM model again. It imports the
model from src/models/cnnlstm.py and only handles data loading, walk-forward
splitting, training, prediction export, and metric reporting.

Example
-------
python run_cnnlstm_walkforward.py \
  --data-path etfdata.csv \
  --output-dir outputs_walkforward_cnnlstm \
  --lookback 60 \
  --horizon 20 \
  --cnn-epochs 8 \
  --cnn-repeats 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.data.loader import (
    OPTIONAL_FIELDS,
    REQUIRED_PRICE_FIELDS,
    load_etf_csv,
    restrict_common_valid_sample,
)
from src.eval.metrics import compute_prediction_metrics, mean_rank_correlation, top_k_backtest
from src.features.labels import get_target_metadata
from src.models.cnn import fit_torch_model, predict_torch_model
from src.models.cnnlstm import CNNLSTMRegressor
from src.pipeline import (
    SampleBundle,
    _json_default,
    _markdown_table,
    _prediction_frame,
    _to_records,
    build_samples,
    set_global_seed,
)
from src.walkforward import generate_walkforward_folds


_DEFAULT_MODELS = [
    "cnn_lstm_image_scale",
    "cnn_lstm_cumulative_scale",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk-forward OOS evaluation for CNNLSTM models")
    p.add_argument("--data-path", default="etfdata.csv")
    p.add_argument("--output-dir", default="outputs_walkforward_cnnlstm")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--label-mode", default="regression", choices=["regression", "classification"])
    p.add_argument("--target-name", default="future_return")
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of CNNLSTM models. Default: cnn_lstm_image_scale cnn_lstm_cumulative_scale",
    )
    p.add_argument("--wf-min-train-days", type=int, default=500)
    p.add_argument("--wf-val-days", type=int, default=60)
    p.add_argument("--wf-test-days", type=int, default=60)
    p.add_argument("--cnn-epochs", type=int, default=8)
    p.add_argument("--cnn-repeats", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    p.add_argument(
        "--no-volume",
        action="store_true",
        help="Exclude volume channel even if volume exists in data.",
    )
    p.add_argument(
        "--no-moving-average",
        action="store_true",
        help="Exclude moving-average channel.",
    )
    return p


def _build_cnnlstm_feature_sets(
    bundle: SampleBundle,
    n: int,
    enabled_models: list[str] | None,
) -> dict[str, np.ndarray]:
    """Build CNNLSTM inputs in (n, channels, lookback) shape.

    The existing build_samples() creates sequences in (n, lookback, channels).
    CNNLSTMRegressor expects (batch, channels, lookback), so we transpose.
    """
    all_sets = {
        "cnn_lstm_image_scale": np.transpose(bundle.image_sequences, (0, 2, 1)),
        "cnn_lstm_cumulative_scale": np.transpose(bundle.cumulative_sequences, (0, 2, 1)),
    }
    if enabled_models:
        unknown = sorted(set(enabled_models) - set(all_sets))
        if unknown:
            raise ValueError(f"Unknown CNNLSTM model(s): {unknown}. Available: {sorted(all_sets)}")
        return {k: v for k, v in all_sets.items() if k in set(enabled_models)}
    return all_sets


def _fit_predict_cnnlstm_repeated(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Train CNNLSTM multiple times and average predictions.

    This mirrors src.pipeline._fit_predict_cnn_repeated(), but imports the model
    from src.models.cnnlstm instead of src.models.cnn.
    """
    if model_name not in {"cnn_lstm_image_scale", "cnn_lstm_cumulative_scale"}:
        raise ValueError(f"unsupported CNNLSTM model name: {model_name}")

    repeats = max(1, int(config.cnn_repeats))
    all_scores = []

    for repeat_idx in range(repeats):
        set_global_seed(config.seed + repeat_idx)
        model = CNNLSTMRegressor(input_shape=tuple(train_x.shape[1:]))
        model = fit_torch_model(
            model,
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
        scores, _ = predict_torch_model(
            model,
            test_x,
            config.label_mode,
            batch_size=config.batch_size,
        )
        all_scores.append(scores.astype(float))

    averaged_scores = np.vstack(all_scores).mean(axis=0)

    if config.label_mode == "classification":
        confidence = np.abs(averaged_scores - 0.5) * 2.0
        return averaged_scores, confidence
    return averaged_scores, None


def _fit_predict_fold(
    bundle: SampleBundle,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    feature_sets: dict[str, np.ndarray],
    config: PipelineConfig,
) -> pd.DataFrame:
    metadata = bundle.metadata
    target_metadata = get_target_metadata(config.label_mode, config.target_name)
    selection_sign = float(target_metadata["selection_sign"])
    target = metadata["target"].to_numpy(dtype=np.float32)

    predictions = []
    for model_name, features in feature_sets.items():
        train_x = features[train_mask]
        val_x = features[val_mask]
        test_x = features[test_mask]
        train_y = target[train_mask]
        val_y = target[val_mask]

        scores, confidence = _fit_predict_cnnlstm_repeated(
            model_name=model_name,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            test_x=test_x,
            config=config,
        )

        pred_frame = _prediction_frame(metadata[test_mask], model_name, scores, confidence, selection_sign)
        predictions.append(pred_frame)

    return pd.concat(predictions, ignore_index=True)


def run_cnnlstm_walkforward(bundle: SampleBundle, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-window walk-forward evaluation for CNNLSTM only."""
    metadata = bundle.metadata
    unique_dates = sorted(metadata["date"].drop_duplicates().tolist())
    folds = generate_walkforward_folds(unique_dates, config)

    if not folds:
        total_needed = config.wf_min_train_days + config.wf_val_days + config.wf_test_days
        raise ValueError(
            f"not enough dates for walk-forward: need at least {total_needed}, got {len(unique_dates)}"
        )

    print(f"Walk-forward CNNLSTM: {len(folds)} folds over {len(unique_dates)} total dates")

    n = len(metadata)
    feature_sets = _build_cnnlstm_feature_sets(bundle, n, config.enabled_models or None)

    all_predictions = []
    for fold_idx, fold in enumerate(folds):
        train_mask = metadata["date"].isin(fold["train_dates"]).to_numpy()
        val_mask = metadata["date"].isin(fold["val_dates"]).to_numpy()
        test_mask = metadata["date"].isin(fold["test_dates"]).to_numpy()

        if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        print(
            f"  fold {fold_idx + 1:2d}/{len(folds)}: "
            f"train={train_mask.sum():5d}  val={val_mask.sum():4d}  test={test_mask.sum():4d}  "
            f"test_end={fold['test_end_date'].strftime('%Y-%m-%d')}"
        )

        fold_preds = _fit_predict_fold(
            bundle=bundle,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            feature_sets=feature_sets,
            config=config,
        )
        fold_preds["fold"] = fold_idx
        all_predictions.append(fold_preds)

    oos_df = pd.concat(all_predictions, ignore_index=True)

    comparison_rows = []
    for model_name, group in oos_df.groupby("model_name"):
        group = group.copy()
        metrics = compute_prediction_metrics(group, config.label_mode)
        if config.label_mode == "classification":
            metrics["rank_correlation"] = mean_rank_correlation(group, "future_return", "selection_score")
        else:
            metrics["target_rank_correlation"] = mean_rank_correlation(group, "target", "signal_value")
            metrics["future_return_rank_correlation"] = mean_rank_correlation(
                group,
                "future_return",
                "selection_score",
            )
        metrics.update(top_k_backtest(group, config.horizon, config.top_k, score_column="selection_score"))
        metrics["model_name"] = model_name
        metrics["n_folds"] = int(group["fold"].nunique())
        comparison_rows.append(metrics)

    comparison_df = pd.DataFrame(comparison_rows)
    sort_preferences = [
        ("top_k_sharpe", False),
        ("future_return_rank_correlation", False),
        ("rank_correlation", False),
        ("target_rank_correlation", False),
        ("roc_auc", False),
        ("rmse", True),
    ]
    sort_cols = [col for col, _ in sort_preferences if col in comparison_df.columns]
    ascending = [asc for col, asc in sort_preferences if col in comparison_df.columns]
    if sort_cols:
        comparison_df = comparison_df.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)

    return oos_df, comparison_df


def main() -> None:
    args = build_parser().parse_args()
    enabled_models = args.models if args.models else _DEFAULT_MODELS

    config = PipelineConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        lookback=args.lookback,
        horizon=args.horizon,
        label_mode=args.label_mode,
        target_name=args.target_name,
        enabled_models=enabled_models,
        include_volume=not args.no_volume,
        include_moving_average=not args.no_moving_average,
        wf_min_train_days=args.wf_min_train_days,
        wf_val_days=args.wf_val_days,
        wf_test_days=args.wf_test_days,
        cnn_epochs=args.cnn_epochs,
        cnn_repeats=args.cnn_repeats,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        weight_decay=args.weight_decay,
        top_k=args.top_k,
        seed=args.seed,
        device=args.device,
        sample_preview_count=0,
    )

    set_global_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {config.data_path} ...")
    _, long_panel, assets, fields, _ = load_etf_csv(config.data_path)

    required_fields = REQUIRED_PRICE_FIELDS.copy()
    if config.include_volume:
        if "volume" in fields:
            required_fields.extend(OPTIONAL_FIELDS)
        else:
            print("Volume field not found. Falling back to include_volume=False.")
            config.include_volume = False

    common_panel, _ = restrict_common_valid_sample(
        long_panel,
        selected_assets=assets,
        required_fields=required_fields,
    )
    if common_panel.empty:
        raise ValueError("no common valid sample after filtering")

    print(f"Building samples  (lookback={config.lookback}, horizon={config.horizon}) ...")
    bundle, prep = build_samples(common_panel, config)
    print(f"  total samples: {prep['n_samples']}  shape: {prep['sequence_shape']}")
    print(f"  channels: {', '.join(bundle.sequence_channels)}")

    print(
        f"\nRunning CNNLSTM walk-forward OOS "
        f"(min_train={config.wf_min_train_days}, val={config.wf_val_days}, test={config.wf_test_days}) ..."
    )
    oos_df, comparison_df = run_cnnlstm_walkforward(bundle, config)

    oos_export = oos_df.copy()
    oos_export["date"] = pd.to_datetime(oos_export["date"]).dt.strftime("%Y-%m-%d")
    oos_export.to_csv(output_dir / "walkforward_predictions.csv", index=False)
    comparison_df.to_csv(output_dir / "walkforward_comparison.csv", index=False)

    report_lines = [
        "# CNNLSTM Walk-Forward OOS Evaluation",
        "",
        "## Configuration",
        f"- lookback={config.lookback}, horizon={config.horizon}",
        f"- label_mode={config.label_mode}, target={config.target_name}",
        f"- wf_min_train_days={config.wf_min_train_days}  wf_val_days={config.wf_val_days}  wf_test_days={config.wf_test_days}",
        f"- cnn_epochs={config.cnn_epochs}, cnn_repeats={config.cnn_repeats}",
        f"- models: {', '.join(enabled_models)}",
        f"- total OOS predictions: {len(oos_df)}",
        "",
        "## Model Comparison (Aggregated OOS)",
        _markdown_table(comparison_df),
        "",
        "## Ensemble Handoff",
        "- Add this output directory to `build_extended_ensemble.py` or a copied ensemble script.",
        "- Main file to merge: `walkforward_predictions.csv`.",
        "- Required model names are stored in the `model_name` column.",
    ]
    (output_dir / "walkforward_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    payload = {
        "config": config.to_dict(),
        "prep_summary": {k: v for k, v in prep.items() if k != "sample_previews"},
        "comparison": _to_records(comparison_df),
    }
    (output_dir / "walkforward_results.json").write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print("\n=== CNNLSTM Walk-Forward Results ===")
    display_cols = [
        c
        for c in [
            "model_name",
            "n_folds",
            "top_k_sharpe",
            "future_return_rank_correlation",
            "target_rank_correlation",
            "rmse",
            "mae",
            "top_k_cumulative_return",
        ]
        if c in comparison_df.columns
    ]
    print(comparison_df[display_cols].to_string(index=False))
    print(f"\nOutputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
