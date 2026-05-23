#!/usr/bin/env python3
"""Full CNN pretraining experiment using auxiliary ETF chart images.

Main idea:
  1. Pretrain `cnn_2d_residual_small` on extra ETF OHLCV images.
  2. Fine-tune the pretrained CNN on the original 7 ETF walk-forward folds.
  3. Evaluate only on the original 7 ETF universe.

The extra ETF data is never added to the final ODE portfolio universe.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import PipelineConfig
from src.data.loader import (
    OPTIONAL_FIELDS,
    REQUIRED_PRICE_FIELDS,
    load_etf_csv,
    restrict_common_valid_sample,
)
from src.eval.metrics import compute_prediction_metrics, mean_rank_correlation, top_k_backtest
from src.features.labels import compute_auxiliary_targets, get_target_metadata, resolve_target
from src.images.chart_renderer import render_jiang_chart
from src.models.cnn import build_cnn_model, fit_torch_model, predict_torch_model
from src.pipeline import SampleBundle, build_samples, set_global_seed
from src.walkforward import generate_walkforward_folds


ROOT = Path(__file__).parent
OUT_DIR = ROOT / "experiments" / "cnn_extra_pretraining_per_fold"
EXTRA_LONG = ROOT / "data" / "extra_market" / "extra_etf_ohlcv_long.csv"
BASELINE_PREDICTIONS = ROOT / "outputs_walkforward_2d_phase2" / "walkforward_predictions.csv"
BASELINE_MU_DAILY = ROOT / "ode_inputs_cnn" / "cnn_2d_residual_small" / "mu_daily.csv"
MODEL_NAME = "cnn_2d_residual_small_extra_pretrained"
BASELINE_MODEL_NAME = "cnn_2d_residual_small"


def _date_norm(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed").dt.normalize()


def _load_main_bundle(config: PipelineConfig) -> tuple[SampleBundle, dict]:
    _, long_panel, assets, fields, _ = load_etf_csv(config.data_path)
    required_fields = REQUIRED_PRICE_FIELDS.copy()
    if config.include_volume and "volume" in fields:
        required_fields.extend(OPTIONAL_FIELDS)
    elif config.include_volume:
        config.include_volume = False

    common_panel, common_summary = restrict_common_valid_sample(
        long_panel,
        selected_assets=assets,
        required_fields=required_fields,
    )
    bundle, prep = build_samples(common_panel, config)
    prep["common_summary"] = common_summary
    return bundle, prep


def _build_chart_only_bundle(panel: pd.DataFrame, config: PipelineConfig) -> tuple[SampleBundle, dict]:
    """Build only metadata and rendered chart images for memory-efficient pretraining."""
    chart_images = []
    metadata_rows = []
    asset_frames = {}
    for asset, asset_frame in panel.groupby("asset"):
        ordered = asset_frame.sort_values("date").reset_index(drop=True).copy()
        if config.include_moving_average and not config.strict_window_ma:
            ordered["ma"] = (
                ordered["close"]
                .rolling(window=config.resolved_ma_window, min_periods=config.resolved_ma_window)
                .mean()
            )
        asset_frames[asset] = ordered

    for asset, asset_frame in asset_frames.items():
        last_signal_index = len(asset_frame) - config.horizon - 1
        min_signal_index = config.lookback - 1
        if config.include_moving_average and config.strict_window_ma:
            # Match the original full-series MA warmup grid without using
            # pre-window prices in the rendered MA values.
            min_signal_index = config.lookback + config.resolved_ma_window - 2
        for idx in range(min_signal_index, last_signal_index + 1):
            window = asset_frame.iloc[idx - config.lookback + 1 : idx + 1].copy()
            future_slice = asset_frame.iloc[idx + 1 : idx + config.horizon + 1].copy()
            if config.include_moving_average and config.strict_window_ma:
                window["ma"] = (
                    window["close"]
                    .rolling(window=config.resolved_ma_window, min_periods=1)
                    .mean()
                )
            if window[REQUIRED_PRICE_FIELDS].isna().any().any():
                continue
            if config.include_volume and window["volume"].isna().any():
                continue
            if config.include_moving_average and window["ma"].isna().any():
                continue

            auxiliary_targets = compute_auxiliary_targets(
                current_close=float(window["close"].iloc[-1]),
                future_close_path=future_slice["close"].to_numpy(dtype=float),
            )
            chart_images.append(
                render_jiang_chart(
                    window,
                    image_height=config.image_height,
                    include_moving_average=config.include_moving_average,
                    include_volume=config.include_volume,
                    chart_variant=config.chart_variant,
                )
            )
            metadata_rows.append(
                {
                    "date": pd.Timestamp(window["date"].iloc[-1]),
                    "asset": asset,
                    "target": resolve_target(config.label_mode, config.target_name, auxiliary_targets),
                    **auxiliary_targets,
                }
            )

    metadata = pd.DataFrame(metadata_rows)
    order = metadata.sort_values(["date", "asset"]).index.to_numpy()
    metadata = metadata.loc[order].reset_index(drop=True)
    empty = np.empty((len(metadata), 0), dtype=np.float32)
    bundle = SampleBundle(
        metadata=metadata,
        cumulative_sequences=empty,
        image_sequences=empty,
        chart_images=np.stack(chart_images).astype(np.uint8)[order],
        sequence_channels=[],
    )
    prep = {
        "lookback": config.lookback,
        "horizon": config.horizon,
        "label_mode": config.label_mode,
        "target_name": config.target_name,
        "include_moving_average": config.include_moving_average,
        "include_volume": config.include_volume,
        "chart_variant": config.chart_variant,
        "strict_window_ma": config.strict_window_ma,
        "n_samples": int(len(metadata)),
        "chart_image_shape": list(bundle.chart_images.shape[1:]),
        "target_mean": float(metadata["target"].mean()),
        "target_std": float(metadata["target"].std(ddof=0)),
        "target_min": float(metadata["target"].min()),
        "target_max": float(metadata["target"].max()),
    }
    return bundle, prep


def _load_extra_bundle(config: PipelineConfig, pretrain_cutoff: pd.Timestamp) -> tuple[SampleBundle, dict]:
    extra = pd.read_csv(EXTRA_LONG, parse_dates=["date"])
    extra["date"] = _date_norm(extra["date"])
    extra = extra.rename(columns={"ticker": "asset"})
    extra = extra.loc[extra["date"] <= pretrain_cutoff].copy()
    extra = extra.sort_values(["date", "asset"]).reset_index(drop=True)
    if extra.empty:
        raise ValueError(f"no extra ETF rows before cutoff={pretrain_cutoff.date()}")

    bundle, prep = _build_chart_only_bundle(extra, config)
    prep["pretrain_cutoff"] = pretrain_cutoff.isoformat()
    prep["n_extra_assets"] = int(extra["asset"].nunique())
    prep["extra_raw_rows"] = int(len(extra))
    prep["extra_raw_start"] = extra["date"].min().isoformat()
    prep["extra_raw_end"] = extra["date"].max().isoformat()
    return bundle, prep


def _load_extra_panel(pretrain_cutoff: pd.Timestamp) -> pd.DataFrame:
    extra = pd.read_csv(EXTRA_LONG, parse_dates=["date"])
    extra["date"] = _date_norm(extra["date"])
    extra = extra.rename(columns={"ticker": "asset"})
    extra = extra.loc[extra["date"] <= pretrain_cutoff].copy()
    extra = extra.sort_values(["date", "asset"]).reset_index(drop=True)
    if extra.empty:
        raise ValueError(f"no extra ETF rows before cutoff={pretrain_cutoff.date()}")
    return extra


def _chronological_train_val_masks(metadata: pd.DataFrame, val_days: int) -> tuple[np.ndarray, np.ndarray]:
    dates = sorted(metadata["date"].drop_duplicates().tolist())
    if len(dates) <= val_days:
        raise ValueError("not enough extra ETF dates for pretrain validation split")
    train_dates = set(dates[:-val_days])
    val_dates = set(dates[-val_days:])
    return (
        metadata["date"].isin(train_dates).to_numpy(),
        metadata["date"].isin(val_dates).to_numpy(),
    )


class _ChartImageDataset(Dataset):
    """Keep chart images as uint8 and convert to float only per mini-batch."""

    def __init__(self, images: np.ndarray, targets: np.ndarray, mask: np.ndarray):
        self.images = images
        self.targets = targets.astype(np.float32, copy=False)
        self.indices = np.flatnonzero(mask)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        image = torch.from_numpy(self.images[idx]).unsqueeze(0).float().div(255.0)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return image, target


def _clear_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fit_torch_model_from_loaders(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    label_mode: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: str,
) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss() if label_mode == "classification" else nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_x)
                losses.append(float(criterion(predictions, batch_y).item()))

        val_loss = float(np.mean(losses)) if losses else float("inf")
        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model.cpu()


def _pretrain_model(extra_bundle: SampleBundle, config: PipelineConfig, args: argparse.Namespace) -> torch.nn.Module:
    target = extra_bundle.metadata["target"].to_numpy(dtype=np.float32)
    train_mask, val_mask = _chronological_train_val_masks(extra_bundle.metadata, args.pretrain_val_days)

    print(
        f"Pretraining on extra ETFs: train={train_mask.sum()} val={val_mask.sum()} "
        f"image_shape={(1, *extra_bundle.chart_images.shape[1:])}",
        flush=True,
    )
    train_loader = DataLoader(
        _ChartImageDataset(extra_bundle.chart_images, target, train_mask),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        _ChartImageDataset(extra_bundle.chart_images, target, val_mask),
        batch_size=args.batch_size,
        shuffle=False,
    )
    model = _fit_torch_model_from_loaders(
        build_cnn_model(BASELINE_MODEL_NAME, (1, config.image_height, config.image_width)),
        train_loader=train_loader,
        val_loader=val_loader,
        label_mode=config.label_mode,
        epochs=args.pretrain_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.pretrain_patience,
        device=args.device,
    )
    del train_loader, val_loader
    _clear_torch_memory()
    return model


def _pretrain_model_for_cutoff(
    extra_panel: pd.DataFrame,
    cutoff: pd.Timestamp,
    config: PipelineConfig,
    args: argparse.Namespace,
    pretrain_path: Path,
) -> torch.nn.Module:
    if args.resume and pretrain_path.exists():
        model = build_cnn_model(BASELINE_MODEL_NAME, (1, config.image_height, config.image_width))
        model.load_state_dict(torch.load(pretrain_path, map_location="cpu"))
        return model

    panel = extra_panel.loc[extra_panel["date"] <= cutoff].copy()
    if panel.empty:
        raise ValueError(f"no extra ETF data before fold cutoff={cutoff.date()}")
    bundle, prep = _build_chart_only_bundle(panel, config)
    print(
        f"    pretrain cutoff={cutoff.strftime('%Y-%m-%d')} "
        f"extra_samples={prep['n_samples']} extra_assets={panel['asset'].nunique()}",
        flush=True,
    )
    model = _pretrain_model(bundle, config, args)
    pretrain_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), pretrain_path)
    del bundle, panel
    _clear_torch_memory()
    return model


def _prediction_frame(
    metadata: pd.DataFrame,
    scores: np.ndarray,
    confidence: np.ndarray | None,
    selection_sign: float,
) -> pd.DataFrame:
    frame = metadata.copy()
    frame["model_name"] = MODEL_NAME
    frame["signal_value"] = scores.astype(float)
    frame["confidence"] = confidence.astype(float) if confidence is not None else np.nan
    frame["selection_score"] = frame["signal_value"] * float(selection_sign)
    return frame


def _fine_tune_walkforward(
    main_bundle: SampleBundle,
    pretrained_model: torch.nn.Module,
    config: PipelineConfig,
    args: argparse.Namespace,
    out_dir: Path,
) -> pd.DataFrame:
    metadata = main_bundle.metadata.copy()
    dates = sorted(metadata["date"].drop_duplicates().tolist())
    folds = generate_walkforward_folds(dates, config)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    if not folds:
        raise ValueError("no walk-forward folds available")

    features = main_bundle.chart_images[:, None, :, :].astype(np.float32) / 255.0
    target = metadata["target"].to_numpy(dtype=np.float32)
    selection_sign = float(get_target_metadata(config.label_mode, config.target_name)["selection_sign"])
    pretrained_state = pretrained_model.state_dict()
    fold_dir = out_dir / "fold_predictions"
    fold_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    print(f"Fine-tuning on original 7 ETF universe: folds={len(folds)}", flush=True)
    for fold_idx, fold in enumerate(folds):
        fold_path = fold_dir / f"fold_{fold_idx:03d}.csv"
        if args.resume and fold_path.exists():
            cached = pd.read_csv(fold_path, parse_dates=["date"])
            cached["date"] = _date_norm(cached["date"])
            all_predictions.append(cached)
            print(f"  fold {fold_idx + 1:2d}/{len(folds)}: cached -> {fold_path}", flush=True)
            continue

        train_mask = metadata["date"].isin(fold["train_dates"]).to_numpy()
        val_mask = metadata["date"].isin(fold["val_dates"]).to_numpy()
        test_mask = metadata["date"].isin(fold["test_dates"]).to_numpy()

        train_val_dates = metadata.loc[train_mask | val_mask, "date"]
        test_dates = metadata.loc[test_mask, "date"]
        assert train_val_dates.max() < test_dates.min(), "walk-forward leakage: train/val overlaps test"

        print(
            f"  fold {fold_idx + 1:2d}/{len(folds)}: "
            f"train={train_mask.sum():5d} val={val_mask.sum():4d} test={test_mask.sum():4d} "
            f"test={test_dates.min().strftime('%Y-%m-%d')}..{test_dates.max().strftime('%Y-%m-%d')}",
            flush=True,
        )
        set_global_seed(config.seed + fold_idx)
        model = build_cnn_model(BASELINE_MODEL_NAME, tuple(features.shape[1:]))
        model.load_state_dict(pretrained_state)
        model = fit_torch_model(
            model,
            train_x=features[train_mask],
            train_y=target[train_mask],
            val_x=features[val_mask],
            val_y=target[val_mask],
            label_mode=config.label_mode,
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            learning_rate=args.finetune_learning_rate,
            weight_decay=args.weight_decay,
            patience=args.finetune_patience,
            device=args.device,
        )
        scores, confidence = predict_torch_model(model, features[test_mask], config.label_mode, args.batch_size)
        pred = _prediction_frame(metadata.loc[test_mask], scores, confidence, selection_sign)
        pred["fold"] = fold_idx
        fold_export = pred.copy()
        fold_export["date"] = fold_export["date"].dt.strftime("%Y-%m-%d")
        fold_export.to_csv(fold_path, index=False)
        all_predictions.append(pred)

        partial = pd.concat(all_predictions, ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
        partial_export = partial.copy()
        partial_export["date"] = partial_export["date"].dt.strftime("%Y-%m-%d")
        partial_export.to_csv(out_dir / "walkforward_predictions_partial.csv", index=False)
        del model, scores, confidence, fold_export, partial, partial_export
        _clear_torch_memory()

    return pd.concat(all_predictions, ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)


def _fine_tune_walkforward_per_fold_pretrain(
    main_bundle: SampleBundle,
    extra_panel: pd.DataFrame,
    config: PipelineConfig,
    args: argparse.Namespace,
    out_dir: Path,
) -> pd.DataFrame:
    metadata = main_bundle.metadata.copy()
    dates = sorted(metadata["date"].drop_duplicates().tolist())
    folds = generate_walkforward_folds(dates, config)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    if not folds:
        raise ValueError("no walk-forward folds available")

    features = main_bundle.chart_images[:, None, :, :].astype(np.float32) / 255.0
    target = metadata["target"].to_numpy(dtype=np.float32)
    selection_sign = float(get_target_metadata(config.label_mode, config.target_name)["selection_sign"])
    fold_dir = out_dir / "fold_predictions"
    pretrain_dir = out_dir / "pretrain_by_fold"
    fold_dir.mkdir(parents=True, exist_ok=True)
    pretrain_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    print(f"Per-fold pretraining + fine-tuning: folds={len(folds)}", flush=True)
    for fold_idx, fold in enumerate(folds):
        fold_path = fold_dir / f"fold_{fold_idx:03d}.csv"
        if args.resume and fold_path.exists():
            cached = pd.read_csv(fold_path, parse_dates=["date"])
            cached["date"] = _date_norm(cached["date"])
            all_predictions.append(cached)
            print(f"  fold {fold_idx + 1:2d}/{len(folds)}: cached -> {fold_path}", flush=True)
            continue

        train_mask = metadata["date"].isin(fold["train_dates"]).to_numpy()
        val_mask = metadata["date"].isin(fold["val_dates"]).to_numpy()
        test_mask = metadata["date"].isin(fold["test_dates"]).to_numpy()
        train_val_dates = metadata.loc[train_mask | val_mask, "date"]
        test_dates = metadata.loc[test_mask, "date"]
        assert train_val_dates.max() < test_dates.min(), "walk-forward leakage: train/val overlaps test"

        cutoff = pd.Timestamp(train_val_dates.max())
        print(
            f"  fold {fold_idx + 1:2d}/{len(folds)}: "
            f"train={train_mask.sum():5d} val={val_mask.sum():4d} test={test_mask.sum():4d} "
            f"pretrain_cutoff={cutoff.strftime('%Y-%m-%d')} "
            f"test={test_dates.min().strftime('%Y-%m-%d')}..{test_dates.max().strftime('%Y-%m-%d')}",
            flush=True,
        )
        set_global_seed(config.seed + fold_idx)
        pretrain_path = pretrain_dir / f"fold_{fold_idx:03d}_pretrained.pt"
        pretrained_model = _pretrain_model_for_cutoff(extra_panel, cutoff, config, args, pretrain_path)
        model = build_cnn_model(BASELINE_MODEL_NAME, tuple(features.shape[1:]))
        model.load_state_dict(pretrained_model.state_dict())
        del pretrained_model
        _clear_torch_memory()
        model = fit_torch_model(
            model,
            train_x=features[train_mask],
            train_y=target[train_mask],
            val_x=features[val_mask],
            val_y=target[val_mask],
            label_mode=config.label_mode,
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            learning_rate=args.finetune_learning_rate,
            weight_decay=args.weight_decay,
            patience=args.finetune_patience,
            device=args.device,
        )
        scores, confidence = predict_torch_model(model, features[test_mask], config.label_mode, args.batch_size)
        pred = _prediction_frame(metadata.loc[test_mask], scores, confidence, selection_sign)
        pred["fold"] = fold_idx
        pred["pretrain_cutoff"] = cutoff
        fold_export = pred.copy()
        fold_export["date"] = fold_export["date"].dt.strftime("%Y-%m-%d")
        fold_export["pretrain_cutoff"] = pd.to_datetime(fold_export["pretrain_cutoff"]).dt.strftime("%Y-%m-%d")
        fold_export.to_csv(fold_path, index=False)
        all_predictions.append(pred)

        partial = pd.concat(all_predictions, ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
        partial_export = partial.copy()
        partial_export["date"] = partial_export["date"].dt.strftime("%Y-%m-%d")
        if "pretrain_cutoff" in partial_export.columns:
            partial_export["pretrain_cutoff"] = pd.to_datetime(partial_export["pretrain_cutoff"]).dt.strftime("%Y-%m-%d")
        partial_export.to_csv(out_dir / "walkforward_predictions_partial.csv", index=False)
        del model, scores, confidence, fold_export, partial, partial_export
        _clear_torch_memory()

    return pd.concat(all_predictions, ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)


def _evaluate(predictions: pd.DataFrame, config: PipelineConfig, model_name: str) -> dict:
    frame = predictions.copy()
    metrics = compute_prediction_metrics(frame, config.label_mode)
    metrics["future_return_rank_correlation"] = mean_rank_correlation(
        frame,
        target_column="future_return",
        score_column="selection_score",
    )
    metrics.update(top_k_backtest(frame, config.horizon, config.top_k, score_column="selection_score"))
    metrics["model_name"] = model_name
    metrics["n_dates"] = int(frame["date"].nunique())
    metrics["n_rows"] = int(len(frame))
    metrics["n_folds"] = int(frame["fold"].nunique()) if "fold" in frame.columns else float("nan")
    return metrics


def _load_baseline_on_same_grid(predictions: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, dict | None]:
    if BASELINE_PREDICTIONS.exists():
        base = pd.read_csv(BASELINE_PREDICTIONS, parse_dates=["date"])
        base["date"] = _date_norm(base["date"])
        base = base.loc[base["model_name"] == BASELINE_MODEL_NAME].copy()
    elif BASELINE_MU_DAILY.exists():
        base = pd.read_csv(BASELINE_MU_DAILY, parse_dates=["date"])
        base["date"] = _date_norm(base["date"])
        base["model_name"] = BASELINE_MODEL_NAME
        base["signal_value"] = base["mu_raw_score"].astype(float)
        base["selection_score"] = base["signal_value"]
        base["target"] = base["future_return"].astype(float)
        base["confidence"] = np.nan
    else:
        return pd.DataFrame(), None

    grid = predictions[["date", "asset"]].drop_duplicates()
    base = grid.merge(base, on=["date", "asset"], how="inner")
    if base.empty:
        return base, None
    return base, _evaluate(base, config, BASELINE_MODEL_NAME)


def _write_report(
    out_dir: Path,
    args: argparse.Namespace,
    main_prep: dict,
    extra_prep: dict,
    comparison: pd.DataFrame,
) -> None:
    best = comparison.sort_values(["future_return_rank_correlation", "top_k_sharpe"], ascending=False).iloc[0]
    lines = [
        "# Extra ETF CNN Pretraining Experiment",
        "",
        "## Purpose",
        "",
        "This experiment checks whether extra ETF chart images improve the CNN feature extractor while keeping the final ODE universe fixed at the original 7 ETF assets.",
        "",
        "## Data",
        "",
        f"- Extra ETF data: `{EXTRA_LONG.relative_to(ROOT)}`",
        f"- Extra raw period used for pretraining: {extra_prep['extra_raw_start']} to {extra_prep['extra_raw_end']}",
        f"- Extra assets: {extra_prep['n_extra_assets']}",
        f"- Extra pretraining samples: {extra_prep['n_samples']}",
        f"- Main 7 ETF samples: {main_prep['n_samples']}",
        f"- Lookback / horizon: {args.lookback} / {args.horizon}",
        f"- Chart variant: {args.chart_variant}",
        "",
        "## Leakage Control",
        "",
        f"- Extra ETF pretraining cutoff: `{extra_prep['pretrain_cutoff']}`.",
        "- This cutoff is before the first original 7 ETF OOS test date.",
        "- Fine-tuning uses walk-forward train/validation windows only.",
        "- Each fold asserts `max(train/val date) < min(test date)`.",
        "",
        "## Training",
        "",
        f"- Pretrain epochs / patience: {args.pretrain_epochs} / {args.pretrain_patience}",
        f"- Fine-tune epochs / patience: {args.finetune_epochs} / {args.finetune_patience}",
        f"- Batch size: {args.batch_size}",
        f"- Device: {args.device}",
        "",
        "## Comparison",
        "",
        "| model | rank corr | Sharpe | cumulative return | hit rate | RMSE | n dates |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| `{row['model_name']}` | {row['future_return_rank_correlation']:.4f} | "
            f"{row['top_k_sharpe']:.4f} | {row['top_k_cumulative_return']:.4f} | "
            f"{row['top_k_hit_rate']:.4f} | {row.get('rmse', float('nan')):.4f} | "
            f"{int(row['n_dates'])} |"
        )
    lines.extend(
        [
            "",
            "## Preliminary Interpretation",
            "",
            f"Best model by rank corr: `{best['model_name']}`.",
            "",
            "If the pretrained CNN improves rank correlation, the extra ETF images are useful for representation learning. If not, the result should be reported as evidence that extra-domain ETF pretraining did not improve the 7-asset OOS signal under this protocol.",
        ]
    )
    (out_dir / "extra_pretraining_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full extra ETF CNN pretraining experiment")
    parser.add_argument("--data-path", default="etfdata.csv")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--chart-variant", default="ohlc_ma_volume")
    parser.add_argument("--wf-min-train-days", type=int, default=500)
    parser.add_argument("--wf-val-days", type=int, default=60)
    parser.add_argument("--wf-test-days", type=int, default=60)
    parser.add_argument("--pretrain-val-days", type=int, default=120)
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--pretrain-patience", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--finetune-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-folds", type=int, default=None, help="debug only; omit for full run")
    parser.add_argument("--pretrain-mode", choices=["fixed", "per_fold"], default="per_fold")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="ignore saved checkpoints")
    parser.add_argument(
        "--strict-window-ma",
        action="store_true",
        help="compute MA inside each rendered lookback window only, avoiding pre-window history",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        data_path=args.data_path,
        output_dir=str(out_dir),
        lookback=args.lookback,
        horizon=args.horizon,
        label_mode="regression",
        target_name="future_return",
        enabled_models=[BASELINE_MODEL_NAME],
        chart_variant=args.chart_variant,
        image_height=64,
        wf_min_train_days=args.wf_min_train_days,
        wf_val_days=args.wf_val_days,
        wf_test_days=args.wf_test_days,
        cnn_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        learning_rate=args.finetune_learning_rate,
        weight_decay=args.weight_decay,
        patience=args.finetune_patience,
        top_k=args.top_k,
        seed=args.seed,
        strict_window_ma=args.strict_window_ma,
        device=args.device,
        sample_preview_count=0,
    )
    set_global_seed(args.seed)

    main_bundle, main_prep = _load_main_bundle(config)
    main_dates = sorted(main_bundle.metadata["date"].drop_duplicates().tolist())
    folds = generate_walkforward_folds(main_dates, config)
    if not folds:
        raise ValueError("no main walk-forward folds available")
    first_test_date = min(folds[0]["test_dates"])
    pretrain_cutoff = pd.Timestamp(first_test_date) - pd.Timedelta(days=1)

    print(f"Main samples: {main_prep['n_samples']} dates={len(main_dates)}", flush=True)
    print(f"First OOS test date: {pd.Timestamp(first_test_date).date()}", flush=True)
    print(f"Extra ETF pretrain cutoff: {pretrain_cutoff.date()}", flush=True)

    extra_config = PipelineConfig(**{**config.to_dict(), "output_dir": str(out_dir / "extra_pretrain")})
    if args.pretrain_mode == "fixed":
        extra_bundle, extra_prep = _load_extra_bundle(extra_config, pretrain_cutoff)
        pretrain_path = out_dir / "pretrained_cnn_2d_residual_small.pt"
        if args.resume and pretrain_path.exists():
            print(f"Loading cached pretrained model: {pretrain_path}", flush=True)
            features_shape = (1, config.image_height, config.image_width)
            pretrained_model = build_cnn_model(BASELINE_MODEL_NAME, features_shape)
            pretrained_model.load_state_dict(torch.load(pretrain_path, map_location="cpu"))
        else:
            pretrained_model = _pretrain_model(extra_bundle, config, args)
            torch.save(pretrained_model.state_dict(), pretrain_path)
        predictions = _fine_tune_walkforward(main_bundle, pretrained_model, config, args, out_dir)
    else:
        last_cutoff = max(max(fold["val_dates"]) for fold in folds)
        extra_panel = _load_extra_panel(pd.Timestamp(last_cutoff))
        extra_prep = {
            "pretrain_cutoff": "per-fold dynamic",
            "n_extra_assets": int(extra_panel["asset"].nunique()),
            "extra_raw_rows": int(len(extra_panel)),
            "extra_raw_start": extra_panel["date"].min().isoformat(),
            "extra_raw_end": extra_panel["date"].max().isoformat(),
            "n_samples": "per-fold dynamic",
        }
        predictions = _fine_tune_walkforward_per_fold_pretrain(
            main_bundle=main_bundle,
            extra_panel=extra_panel,
            config=config,
            args=args,
            out_dir=out_dir,
        )
    export_predictions = predictions.copy()
    export_predictions["date"] = export_predictions["date"].dt.strftime("%Y-%m-%d")
    export_predictions.to_csv(out_dir / "walkforward_predictions.csv", index=False)

    rows = [_evaluate(predictions, config, MODEL_NAME)]
    baseline_df, baseline_metrics = _load_baseline_on_same_grid(predictions, config)
    if baseline_metrics is not None:
        rows.append(baseline_metrics)
        baseline_export = baseline_df.copy()
        baseline_export["date"] = baseline_export["date"].dt.strftime("%Y-%m-%d")
        baseline_export.to_csv(out_dir / "baseline_same_grid_predictions.csv", index=False)

    comparison = pd.DataFrame(rows).sort_values(["future_return_rank_correlation", "top_k_sharpe"], ascending=False)
    comparison.to_csv(out_dir / "comparison.csv", index=False)

    metadata = {
        "model_name": MODEL_NAME,
        "baseline_model_name": BASELINE_MODEL_NAME,
        "extra_data": str(EXTRA_LONG.relative_to(ROOT)),
        "baseline_predictions": str(BASELINE_PREDICTIONS.relative_to(ROOT)),
        "output_dir": str(out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir),
        "first_oos_test_date": str(pd.Timestamp(first_test_date).date()),
        "pretrain_cutoff": str(pretrain_cutoff.date()),
        "pretrain_mode": args.pretrain_mode,
        "config": config.to_dict(),
        "args": vars(args),
        "main_prep": {k: v for k, v in main_prep.items() if k != "sample_previews"},
        "extra_prep": {k: v for k, v in extra_prep.items() if k != "sample_previews"},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    _write_report(out_dir, args, main_prep, extra_prep, comparison)

    print("\n=== Extra ETF CNN Pretraining Comparison ===", flush=True)
    display = [
        "model_name",
        "future_return_rank_correlation",
        "top_k_sharpe",
        "top_k_cumulative_return",
        "top_k_hit_rate",
        "rmse",
        "n_dates",
    ]
    print(comparison[display].to_string(index=False), flush=True)
    print(f"\nOutputs saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
