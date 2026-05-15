#!/usr/bin/env python3
"""Build Jiang-style image factors and test their incremental value.

Outputs are written under:
  ode_inputs_cnn/image_factor_extension/

The script intentionally reuses the existing walk-forward folds and target
construction so that image-factor tests line up with the current CNN/LSTM
handoff package.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import PipelineConfig
from src.data.loader import OPTIONAL_FIELDS, REQUIRED_PRICE_FIELDS, load_etf_csv, restrict_common_valid_sample
from src.eval.metrics import mean_rank_correlation, top_k_backtest
from src.models.cnn import build_cnn_model, extract_torch_features, fit_torch_model, predict_torch_model
from src.pipeline import build_samples, set_global_seed
from src.walkforward import generate_walkforward_folds


ROOT = Path(__file__).parent
DEFAULT_OUT = ROOT / "ode_inputs_cnn" / "image_factor_extension"
ASSET_COLS = [
    "alternative",
    "corp_bond_ig",
    "developed_equity",
    "emerging_equity",
    "korea_equity",
    "short_treasury",
    "treasury_7_10y",
]
FACTOR_COLUMNS = ["image_score", "image_factor_pc1", "image_factor_pc2", "image_factor_pc3"]
CONTROL_COLUMNS = ["ret_pc_loading_1", "ret_pc_loading_2", "ret_pc_loading_3"]
ENSEMBLE_SOURCES = {
    "ensemble_best": ROOT / "ode_inputs_cnn" / "ensemble_best" / "mu_daily.csv",
    "ensemble_4family": ROOT / "ode_inputs_cnn" / "ensemble_4family" / "mu_daily.csv",
}


def _markdown_table(df: pd.DataFrame, float_precision: int = 4) -> str:
    headers = list(df.columns)

    def fmt(value) -> str:
        if isinstance(value, float):
            if np.isnan(value):
                return "nan"
            return f"{value:.{float_precision}f}"
        return str(value)

    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(fmt(row[col]) for col in headers) + " |")
    return "\n".join(rows)


def _date_norm(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed").dt.normalize()


def _load_sample_bundle(config: PipelineConfig):
    _, long_panel, assets, fields, _ = load_etf_csv(config.data_path)
    required_fields = REQUIRED_PRICE_FIELDS.copy()
    if config.include_volume:
        if "volume" in fields:
            required_fields.extend(OPTIONAL_FIELDS)
        else:
            config.include_volume = False

    common_panel, _ = restrict_common_valid_sample(
        long_panel,
        selected_assets=assets,
        required_fields=required_fields,
    )
    if common_panel.empty:
        raise ValueError("no common valid sample after filtering")
    return build_samples(common_panel, config)


def _orient_component(values: np.ndarray, target: np.ndarray) -> float:
    if np.std(values) <= 1e-12 or np.std(target) <= 1e-12:
        return 1.0
    corr = np.corrcoef(values, target)[0, 1]
    if np.isfinite(corr) and corr < 0:
        return -1.0
    return 1.0


def build_image_factor_panel(
    config: PipelineConfig,
    max_folds: int | None = None,
    fold_dates: list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    set_global_seed(config.seed)
    bundle, prep = _load_sample_bundle(config)
    metadata = bundle.metadata.copy()
    unique_dates = sorted(fold_dates if fold_dates is not None else metadata["date"].drop_duplicates().tolist())
    folds = generate_walkforward_folds(unique_dates, config)
    if max_folds is not None:
        folds = folds[:max_folds]
    if not folds:
        raise ValueError("no walk-forward folds available")

    features = bundle.chart_images[:, None, :, :].astype(np.float32) / 255.0
    target = metadata["target"].to_numpy(dtype=np.float32)
    rows = []

    print(
        f"Image factor extraction: samples={prep['n_samples']} image_shape={features.shape[1:]} folds={len(folds)}",
        flush=True,
    )

    for fold_idx, fold in enumerate(folds):
        train_mask = metadata["date"].isin(fold["train_dates"]).to_numpy()
        val_mask = metadata["date"].isin(fold["val_dates"]).to_numpy()
        test_mask = metadata["date"].isin(fold["test_dates"]).to_numpy()
        train_val_mask = train_mask | val_mask

        train_val_dates = metadata.loc[train_val_mask, "date"]
        test_dates = metadata.loc[test_mask, "date"]
        assert train_val_dates.max() < test_dates.min(), "walk-forward leakage: train/val overlaps test"

        print(
            f"  fold {fold_idx + 1:2d}/{len(folds)}: "
            f"train={train_mask.sum():5d} val={val_mask.sum():4d} test={test_mask.sum():4d} "
            f"test_start={test_dates.min().strftime('%Y-%m-%d')}",
            flush=True,
        )

        model = fit_torch_model(
            build_cnn_model(config.enabled_models[0], tuple(features.shape[1:])),
            train_x=features[train_mask],
            train_y=target[train_mask],
            val_x=features[val_mask],
            val_y=target[val_mask],
            label_mode=config.label_mode,
            epochs=config.cnn_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            patience=config.patience,
            device=config.device,
        )

        test_scores, _ = predict_torch_model(model, features[test_mask], config.label_mode, config.batch_size)
        train_val_fc = extract_torch_features(model, features[train_val_mask], config.batch_size)
        test_fc = extract_torch_features(model, features[test_mask], config.batch_size)

        scaler = StandardScaler()
        pca = PCA(n_components=3, random_state=config.seed)
        train_val_pcs = pca.fit_transform(scaler.fit_transform(train_val_fc))
        test_pcs = pca.transform(scaler.transform(test_fc))

        y_train_val = metadata.loc[train_val_mask, "future_return"].to_numpy(dtype=float)
        signs = np.array([
            _orient_component(train_val_pcs[:, comp_idx], y_train_val)
            for comp_idx in range(3)
        ])
        test_pcs = test_pcs * signs

        fold_meta = metadata.loc[test_mask, ["date", "asset", "target", "future_return"]].copy()
        fold_meta["fold"] = fold_idx
        fold_meta["image_score"] = test_scores.astype(float)
        for comp_idx in range(3):
            fold_meta[f"image_factor_pc{comp_idx + 1}"] = test_pcs[:, comp_idx].astype(float)
        rows.append(fold_meta)

    panel = pd.concat(rows, ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
    if panel.duplicated(["date", "asset"]).any():
        raise ValueError("image_factor_panel has duplicate date-asset rows")
    if panel[FACTOR_COLUMNS].isna().any().any():
        raise ValueError("image_factor_panel contains NaN factor values")
    return panel


def build_common_pca_controls(
    signal_dates: list[pd.Timestamp],
    returns_path: Path,
    window: int,
    n_components: int = 3,
) -> pd.DataFrame:
    returns = pd.read_csv(returns_path, parse_dates=["date"])
    returns["date"] = _date_norm(returns["date"])
    returns = returns.sort_values("date").set_index("date")
    assets = [col for col in ASSET_COLS if col in returns.columns]
    if len(assets) < n_components:
        raise ValueError(f"need at least {n_components} assets for PCA controls")

    rows = []
    for date in sorted(pd.to_datetime(signal_dates).normalize().unique()):
        hist = returns.loc[returns.index <= date, assets].tail(window)
        if len(hist) < window:
            continue
        assert hist.index.max() <= date, "rolling PCA leakage: future return used"

        pca = PCA(n_components=n_components)
        pca.fit(hist.to_numpy(dtype=float))
        loadings = pca.components_.T
        for comp_idx in range(n_components):
            if np.nansum(loadings[:, comp_idx]) < 0:
                loadings[:, comp_idx] *= -1.0
        for asset_idx, asset in enumerate(assets):
            row = {"date": date, "asset": asset}
            for comp_idx in range(n_components):
                row[f"ret_pc_loading_{comp_idx + 1}"] = float(loadings[asset_idx, comp_idx])
            rows.append(row)

    controls = pd.DataFrame(rows)
    if controls.empty:
        raise ValueError("rolling PCA controls are empty")
    if controls[CONTROL_COLUMNS].isna().any().any():
        raise ValueError("common_pca_controls contains NaN values")
    return controls.sort_values(["date", "asset"]).reset_index(drop=True)


def _ols_cluster(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> dict:
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y = y[mask]
    x = x[mask]
    clusters = clusters[mask]
    n, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    sse = float((resid ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 1e-12 else float("nan")

    meat = np.zeros((k, k), dtype=float)
    unique_clusters = pd.Series(clusters).drop_duplicates().to_numpy()
    for cluster in unique_clusters:
        idx = clusters == cluster
        xu = x[idx].T @ resid[idx]
        meat += np.outer(xu, xu)

    g = len(unique_clusters)
    cov = xtx_inv @ meat @ xtx_inv
    if g > 1 and n > k:
        cov *= (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t_stat = beta / se
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=max(g - 1, 1)))
    return {
        "beta": beta,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "r2": r2,
        "n_obs": int(n),
        "n_clusters": int(g),
    }


def _daily_rank_corr(df: pd.DataFrame, score_col: str, target_col: str = "future_return") -> float:
    values = []
    for _, group in df.groupby("date"):
        if group[score_col].nunique() < 2 or group[target_col].nunique() < 2:
            continue
        rho, _ = spearmanr(group[score_col], group[target_col])
        if np.isfinite(rho):
            values.append(float(rho))
    return float(np.mean(values)) if values else float("nan")


def run_factor_significance(panel: pd.DataFrame, controls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = panel.copy()
    controls = controls.copy()
    panel["date"] = _date_norm(panel["date"])
    controls["date"] = _date_norm(controls["date"])
    merged = panel.merge(controls, on=["date", "asset"], how="inner")
    if len(merged) != len(panel):
        missing = len(panel) - len(merged)
        print(f"  warning: dropped {missing} rows without rolling PCA controls")
    if merged.empty:
        raise ValueError("no rows left after merging image factors and PCA controls")

    y = merged["future_return"].to_numpy(dtype=float)
    clusters = merged["date"].astype(str).to_numpy()
    x_base = np.column_stack([np.ones(len(merged)), merged[CONTROL_COLUMNS].to_numpy(dtype=float)])
    base = _ols_cluster(y, x_base, clusters)

    rows = []
    for factor in FACTOR_COLUMNS:
        x_aug = np.column_stack([x_base, merged[factor].to_numpy(dtype=float)])
        aug = _ols_cluster(y, x_aug, clusters)
        factor_idx = x_aug.shape[1] - 1
        rows.append(
            {
                "factor_name": factor,
                "coefficient": float(aug["beta"][factor_idx]),
                "std_error_cluster_date": float(aug["se"][factor_idx]),
                "t_stat": float(aug["t_stat"][factor_idx]),
                "p_value": float(aug["p_value"][factor_idx]),
                "base_r2": float(base["r2"]),
                "augmented_r2": float(aug["r2"]),
                "delta_r2": float(aug["r2"] - base["r2"]),
                "daily_rank_correlation": _daily_rank_corr(merged, factor),
                "n_obs": int(aug["n_obs"]),
                "n_dates": int(aug["n_clusters"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["p_value", "delta_r2"], ascending=[True, False]), merged


def _prediction_metrics(df: pd.DataFrame, score_col: str, horizon: int, top_k: int) -> dict:
    pred = df[["date", "asset", "future_return"]].copy()
    pred["signal_value"] = df[score_col].astype(float).to_numpy()
    pred["selection_score"] = pred["signal_value"]
    metrics = {
        "rank_corr": mean_rank_correlation(pred, "future_return", "selection_score"),
    }
    metrics.update(top_k_backtest(pred, horizon, top_k, "selection_score"))
    return metrics


def _per_date_rank_corr_series(df: pd.DataFrame, score_col: str) -> pd.Series:
    rows = []
    for date, group in df.groupby("date"):
        if group[score_col].nunique() < 2 or group["future_return"].nunique() < 2:
            continue
        rho, _ = spearmanr(group[score_col], group["future_return"])
        if np.isfinite(rho):
            rows.append((date, float(rho)))
    return pd.Series(dict(rows)).sort_index()


def _paired_bootstrap(
    base_series: pd.Series,
    candidate_series: pd.Series,
    samples: int,
    seed: int,
) -> dict:
    aligned = pd.concat([base_series.rename("base"), candidate_series.rename("candidate")], axis=1).dropna()
    if aligned.empty:
        return {"rank_corr_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    diffs = aligned["candidate"].to_numpy() - aligned["base"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for idx in range(samples):
        sample_idx = rng.integers(0, len(diffs), size=len(diffs))
        draws[idx] = float(diffs[sample_idx].mean())
    return {
        "rank_corr_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def _load_ensemble_source(name: str, panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["date"] = _date_norm(panel["date"])
    source = pd.read_csv(ENSEMBLE_SOURCES[name], parse_dates=["date"])
    source["date"] = _date_norm(source["date"])
    source = source[["date", "asset", "mu_raw_score"]].rename(columns={"mu_raw_score": "base_signal"})
    merged = panel[["date", "asset", "future_return"] + FACTOR_COLUMNS].merge(
        source,
        on=["date", "asset"],
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"no overlap with {name} ensemble source")
    return merged.sort_values(["date", "asset"]).reset_index(drop=True)


def run_ensemble_extension(
    panel: pd.DataFrame,
    horizon: int,
    top_k: int,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for base_name in ["ensemble_best", "ensemble_4family"]:
        base_df = _load_ensemble_source(base_name, panel)
        base_df["base_rank"] = base_df.groupby("date")["base_signal"].rank(pct=True)
        base_metrics = _prediction_metrics(base_df, "base_signal", horizon, top_k)
        base_series = _per_date_rank_corr_series(base_df, "base_signal")
        rows.append(
            {
                "candidate_name": base_name,
                "base_ensemble": base_name,
                "added_factor": "",
                **base_metrics,
                "factor_vs_base_pearson": float("nan"),
                "factor_vs_base_rank_corr_mean": float("nan"),
                "rank_corr_diff_vs_base": 0.0,
                "bootstrap_ci_low": 0.0,
                "bootstrap_ci_high": 0.0,
                "n_dates": int(base_df["date"].nunique()),
            }
        )

        for factor in ["image_score", "image_factor_pc1"]:
            candidate = base_df.copy()
            candidate["factor_rank"] = candidate.groupby("date")[factor].rank(pct=True)
            candidate["candidate_signal"] = 0.5 * (candidate["base_rank"] + candidate["factor_rank"])
            metrics = _prediction_metrics(candidate, "candidate_signal", horizon, top_k)
            candidate_series = _per_date_rank_corr_series(candidate, "candidate_signal")
            boot = _paired_bootstrap(base_series, candidate_series, bootstrap_samples, seed)

            factor_rank_corrs = []
            for _, group in candidate.groupby("date"):
                if group["base_signal"].nunique() < 2 or group[factor].nunique() < 2:
                    continue
                rho, _ = spearmanr(group["base_signal"], group[factor])
                if np.isfinite(rho):
                    factor_rank_corrs.append(float(rho))

            rows.append(
                {
                    "candidate_name": f"{base_name}+{factor}",
                    "base_ensemble": base_name,
                    "added_factor": factor,
                    **metrics,
                    "factor_vs_base_pearson": float(candidate["base_signal"].corr(candidate[factor])),
                    "factor_vs_base_rank_corr_mean": float(np.mean(factor_rank_corrs)) if factor_rank_corrs else float("nan"),
                    "rank_corr_diff_vs_base": boot["rank_corr_diff"],
                    "bootstrap_ci_low": boot["ci_low"],
                    "bootstrap_ci_high": boot["ci_high"],
                    "n_dates": int(candidate["date"].nunique()),
                }
            )

    return pd.DataFrame(rows).sort_values(["rank_corr", "top_k_sharpe"], ascending=[False, False])


def export_signal_panels(
    out_dir: Path,
    image_panel: pd.DataFrame,
    model_prefix: str = "cnn_2d_residual_small",
) -> None:
    panel = image_panel.copy()
    panel["date"] = _date_norm(panel["date"])

    factor_rows = []
    for factor in FACTOR_COLUMNS:
        signal = panel[["date", "asset", "future_return", "fold"]].copy()
        signal["model_name"] = f"{model_prefix}_{factor}"
        signal["signal_value"] = panel[factor].astype(float).to_numpy()
        signal["confidence"] = np.nan
        factor_rows.append(signal)
    factor_signals = pd.concat(factor_rows, ignore_index=True)
    factor_signals = factor_signals[
        ["date", "asset", "model_name", "signal_value", "confidence", "future_return", "fold"]
    ].sort_values(["date", "asset", "model_name"])
    factor_signals["date"] = factor_signals["date"].dt.strftime("%Y-%m-%d")
    factor_signals.to_csv(out_dir / "image_factor_signals.csv", index=False)

    mu_rows = []
    for base_name in ["ensemble_best", "ensemble_4family"]:
        base_df = _load_ensemble_source(base_name, panel)
        base_df["base_rank"] = base_df.groupby("date")["base_signal"].rank(pct=True)

        base_signal = base_df[["date", "asset", "future_return"]].copy()
        base_signal["model_name"] = base_name
        base_signal["signal_value"] = base_df["base_signal"].astype(float).to_numpy()
        base_signal["confidence"] = np.nan
        mu_rows.append(base_signal)

        for factor in ["image_score", "image_factor_pc1"]:
            candidate = base_df[["date", "asset", "future_return"]].copy()
            factor_rank = base_df.groupby("date")[factor].rank(pct=True)
            candidate["model_name"] = f"{base_name}+{factor}"
            candidate["signal_value"] = 0.5 * (base_df["base_rank"] + factor_rank)
            candidate["confidence"] = np.nan
            mu_rows.append(candidate)

    mu_signals = pd.concat(mu_rows, ignore_index=True)
    mu_signals = mu_signals[
        ["date", "asset", "model_name", "signal_value", "confidence", "future_return"]
    ].sort_values(["date", "asset", "model_name"])
    mu_signals["date"] = mu_signals["date"].dt.strftime("%Y-%m-%d")
    mu_signals.to_csv(out_dir / "ode_mu_candidate_signals.csv", index=False)


def build_report(
    out_dir: Path,
    config: PipelineConfig,
    image_panel: pd.DataFrame,
    controls: pd.DataFrame,
    significance: pd.DataFrame,
    ensemble_results: pd.DataFrame,
) -> str:
    best_factor = significance.sort_values(["p_value", "delta_r2"], ascending=[True, False]).iloc[0]
    best_ensemble = ensemble_results.sort_values(["rank_corr", "top_k_sharpe"], ascending=[False, False]).iloc[0]

    lines = [
        "# Jiang-Style Image Factor Extension",
        "",
        "## Summary",
        f"- Extractor: `{config.enabled_models[0]}`",
        f"- Lookback/horizon: `{config.lookback}/{config.horizon}`",
        f"- OOS rows: `{len(image_panel)}` across `{image_panel['date'].nunique()}` dates",
        f"- Rolling PCA controls: `{len(controls)}` asset-date rows",
        f"- Best factor by p-value: `{best_factor['factor_name']}` "
        f"(t={best_factor['t_stat']:.3f}, p={best_factor['p_value']:.4f}, "
        f"delta R2={best_factor['delta_r2']:.6f})",
        f"- Best ensemble candidate by rank corr: `{best_ensemble['candidate_name']}` "
        f"(rank corr={best_ensemble['rank_corr']:.4f}, Sharpe={best_ensemble['top_k_sharpe']:.4f})",
        "",
        "## Image Factor Significance",
        _markdown_table(significance, float_precision=5),
        "",
        "## Ensemble Search",
        _markdown_table(ensemble_results, float_precision=5),
        "",
        "## Interpretation Rules",
        "- If a factor has p < 0.05, call it statistically significant after rolling PCA controls.",
        "- If p includes 0 but delta R2 and ensemble rank-corr lift are positive, frame it as mechanism/trend evidence.",
        "- If ensemble CI includes 0, do not claim strict improvement over the baseline ensemble.",
        "",
        "## Output Files",
        f"- `{out_dir / 'image_factor_panel.csv'}`",
        f"- `{out_dir / 'common_pca_controls.csv'}`",
        f"- `{out_dir / 'image_factor_significance.csv'}`",
        f"- `{out_dir / 'ensemble_image_factor_search.csv'}`",
        f"- `{out_dir / 'image_factor_signals.csv'}`",
        f"- `{out_dir / 'ode_mu_candidate_signals.csv'}`",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Jiang-style image factor extension.")
    p.add_argument("--data-path", default="etfdata.csv")
    p.add_argument("--returns-path", default="ode_inputs_cnn/returns_daily.csv")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--image-height", type=int, default=64)
    p.add_argument("--wf-min-train-days", type=int, default=500)
    p.add_argument("--wf-val-days", type=int, default=60)
    p.add_argument("--wf-test-days", type=int, default=60)
    p.add_argument("--cnn-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--rolling-pca-window", type=int, default=252)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-folds", type=int, default=None, help="debug/smoke-test option")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        data_path=args.data_path,
        output_dir=str(out_dir),
        lookback=args.lookback,
        horizon=args.horizon,
        label_mode="regression",
        target_name="future_return",
        enabled_models=["cnn_2d_residual_small"],
        image_height=args.image_height,
        wf_min_train_days=args.wf_min_train_days,
        wf_val_days=args.wf_val_days,
        wf_test_days=args.wf_test_days,
        cnn_epochs=args.cnn_epochs,
        patience=args.patience,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        top_k=args.top_k,
        seed=args.seed,
        sample_preview_count=0,
    )

    image_panel = build_image_factor_panel(config, max_folds=args.max_folds)
    image_export = image_panel.copy()
    image_export["date"] = image_export["date"].dt.strftime("%Y-%m-%d")
    image_export.to_csv(out_dir / "image_factor_panel.csv", index=False)

    controls = build_common_pca_controls(
        signal_dates=image_panel["date"].tolist(),
        returns_path=Path(args.returns_path),
        window=args.rolling_pca_window,
    )
    controls_export = controls.copy()
    controls_export["date"] = controls_export["date"].dt.strftime("%Y-%m-%d")
    controls_export.to_csv(out_dir / "common_pca_controls.csv", index=False)

    significance, merged_regression_panel = run_factor_significance(image_panel, controls)
    significance.to_csv(out_dir / "image_factor_significance.csv", index=False)

    ensemble_results = run_ensemble_extension(
        panel=merged_regression_panel,
        horizon=args.horizon,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    ensemble_results.to_csv(out_dir / "ensemble_image_factor_search.csv", index=False)
    export_signal_panels(out_dir, image_panel)

    report = build_report(out_dir, config, image_panel, controls, significance, ensemble_results)
    (out_dir / "image_factor_report.md").write_text(report, encoding="utf-8")
    (out_dir / "image_factor_config.json").write_text(
        json.dumps({**config.to_dict(), "rolling_pca_window": args.rolling_pca_window}, indent=2),
        encoding="utf-8",
    )

    print("\n=== Image Factor Extension ===")
    print(significance.to_string(index=False))
    print("\n=== Ensemble Extension ===")
    print(ensemble_results.to_string(index=False))
    print(f"\nOutputs saved to {out_dir}")


if __name__ == "__main__":
    main()
