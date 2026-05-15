#!/usr/bin/env python3
"""Run image-component ablations for the Jiang-style ETF image factor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_image_factor_extension as ext
from src.config import PipelineConfig
from src.images.chart_renderer import (
    CHART_VARIANTS,
    chart_variant_uses_moving_average,
    chart_variant_uses_volume,
)


DEFAULT_OUT = ext.ROOT / "ode_inputs_cnn" / "image_factor_ablation"
ABLATION_VARIANTS = [
    "close_only",
    "ohlc_full",
    "ohlc_ma",
    "ohlc_volume",
    "ohlc_ma_volume",
    "high_low_range",
]
VARIANT_OUTPUT_FILES = [
    "image_factor_panel.csv",
    "common_pca_controls.csv",
    "image_factor_significance.csv",
    "ensemble_image_factor_search.csv",
    "image_factor_signals.csv",
    "ode_mu_candidate_signals.csv",
    "image_factor_report.md",
    "image_factor_config.json",
]


def _markdown_table(df: pd.DataFrame, float_precision: int = 5) -> str:
    if df.empty:
        return "_No rows._"
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


def _variant_config(args: argparse.Namespace, variant: str, output_dir: Path) -> PipelineConfig:
    return PipelineConfig(
        data_path=args.data_path,
        output_dir=str(output_dir),
        lookback=args.lookback,
        horizon=args.horizon,
        label_mode="regression",
        target_name="future_return",
        enabled_models=["cnn_2d_residual_small"],
        include_moving_average=chart_variant_uses_moving_average(variant),
        include_volume=chart_variant_uses_volume(variant),
        chart_variant=variant,
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


def _reference_fold_dates(args: argparse.Namespace) -> list[pd.Timestamp]:
    ref_config = _variant_config(args, "ohlc_ma_volume", Path(args.output_dir) / "_reference_grid")
    bundle, _ = ext._load_sample_bundle(ref_config)
    return sorted(bundle.metadata["date"].drop_duplicates().tolist())


def _export_date_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def _grid_key(panel: pd.DataFrame) -> pd.DataFrame:
    key = panel[["date", "asset"]].copy()
    key["date"] = pd.to_datetime(key["date"]).dt.normalize()
    return key.sort_values(["date", "asset"]).reset_index(drop=True)


def _candidate_row(df: pd.DataFrame, candidate_name: str) -> pd.Series | None:
    rows = df.loc[df["candidate_name"] == candidate_name]
    if rows.empty:
        return None
    return rows.iloc[0]


def _variant_outputs_complete(variant_dir: Path) -> bool:
    return all((variant_dir / file_name).exists() for file_name in VARIANT_OUTPUT_FILES)


def _load_completed_variant(variant: str, variant_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_panel = pd.read_csv(variant_dir / "image_factor_panel.csv", parse_dates=["date"])
    significance = pd.read_csv(variant_dir / "image_factor_significance.csv")
    ensemble = pd.read_csv(variant_dir / "ensemble_image_factor_search.csv")
    significance.insert(0, "chart_variant", variant)
    ensemble.insert(0, "chart_variant", variant)
    summary = pd.DataFrame([_summary_row(variant, significance, ensemble)])
    return summary, significance, ensemble, _grid_key(image_panel)


def _summary_row(variant: str, significance: pd.DataFrame, ensemble: pd.DataFrame) -> dict:
    best_factor = significance.sort_values(["p_value", "delta_r2"], ascending=[True, False]).iloc[0]
    best_ensemble = ensemble.sort_values(["rank_corr", "top_k_sharpe"], ascending=[False, False]).iloc[0]
    best_pc1 = _candidate_row(ensemble, "ensemble_best+image_factor_pc1")
    best_score = _candidate_row(ensemble, "ensemble_best+image_score")
    four_pc1 = _candidate_row(ensemble, "ensemble_4family+image_factor_pc1")
    four_score = _candidate_row(ensemble, "ensemble_4family+image_score")
    return {
        "chart_variant": variant,
        "best_factor": best_factor["factor_name"],
        "best_factor_t_stat": float(best_factor["t_stat"]),
        "best_factor_p_value": float(best_factor["p_value"]),
        "best_factor_delta_r2": float(best_factor["delta_r2"]),
        "best_factor_daily_rank_corr": float(best_factor["daily_rank_correlation"]),
        "best_ensemble_candidate": best_ensemble["candidate_name"],
        "best_ensemble_rank_corr": float(best_ensemble["rank_corr"]),
        "best_ensemble_sharpe": float(best_ensemble["top_k_sharpe"]),
        "ensemble_best_pc1_rank_corr_diff": float(best_pc1["rank_corr_diff_vs_base"]) if best_pc1 is not None else np.nan,
        "ensemble_best_score_rank_corr_diff": float(best_score["rank_corr_diff_vs_base"]) if best_score is not None else np.nan,
        "ensemble_4family_pc1_rank_corr_diff": float(four_pc1["rank_corr_diff_vs_base"]) if four_pc1 is not None else np.nan,
        "ensemble_4family_score_rank_corr_diff": float(four_score["rank_corr_diff_vs_base"]) if four_score is not None else np.nan,
    }


def _build_variant_report(
    variant: str,
    out_dir: Path,
    config: PipelineConfig,
    image_panel: pd.DataFrame,
    controls: pd.DataFrame,
    significance: pd.DataFrame,
    ensemble: pd.DataFrame,
) -> str:
    base = ext.build_report(out_dir, config, image_panel, controls, significance, ensemble)
    base = base.replace("# Jiang-Style Image Factor Extension", f"# Image Factor Ablation - {variant}", 1)
    insert = (
        f"- Chart variant: `{variant}`\n"
        f"- Rendered MA: `{config.include_moving_average}`\n"
        f"- Rendered volume: `{config.include_volume}`"
    )
    return base.replace("## Summary\n", f"## Summary\n{insert}\n", 1)


def _build_ablation_report(
    out_dir: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    ensemble: pd.DataFrame,
) -> str:
    best_sig = summary.sort_values(["best_factor_p_value", "best_factor_delta_r2"], ascending=[True, False]).iloc[0]
    best_ens = summary.sort_values(["best_ensemble_rank_corr", "best_ensemble_sharpe"], ascending=[False, False]).iloc[0]
    compact_cols = [
        "chart_variant",
        "best_factor",
        "best_factor_t_stat",
        "best_factor_p_value",
        "best_factor_delta_r2",
        "best_factor_daily_rank_corr",
        "best_ensemble_candidate",
        "best_ensemble_rank_corr",
        "best_ensemble_sharpe",
    ]
    lines = [
        "# Image Factor Ablation Report",
        "",
        "## Scope",
        "- Optimization is excluded in this run.",
        "- Model architecture is fixed to `cnn_2d_residual_small`; only chart rendering components vary.",
        f"- Lookback/horizon: `{args.lookback}/{args.horizon}`",
        f"- CNN epochs/patience: `{args.cnn_epochs}/{args.patience}`",
        "",
        "## Main Result",
        f"- Best PCA-control factor by p-value: `{best_sig['chart_variant']}::{best_sig['best_factor']}` "
        f"(t={best_sig['best_factor_t_stat']:.3f}, p={best_sig['best_factor_p_value']:.4f}, "
        f"delta R2={best_sig['best_factor_delta_r2']:.6f})",
        f"- Best ensemble row by rank corr: `{best_ens['chart_variant']}::{best_ens['best_ensemble_candidate']}` "
        f"(rank corr={best_ens['best_ensemble_rank_corr']:.4f}, Sharpe={best_ens['best_ensemble_sharpe']:.4f})",
        "",
        "## Variant Summary",
        _markdown_table(summary[compact_cols], float_precision=5),
        "",
        "## All Significance Tests",
        _markdown_table(
            significance[
                [
                    "chart_variant",
                    "factor_name",
                    "t_stat",
                    "p_value",
                    "delta_r2",
                    "daily_rank_correlation",
                    "n_obs",
                    "n_dates",
                ]
            ],
            float_precision=5,
        ),
        "",
        "## Ensemble Additions",
        _markdown_table(
            ensemble[
                [
                    "chart_variant",
                    "candidate_name",
                    "rank_corr",
                    "top_k_sharpe",
                    "rank_corr_diff_vs_base",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "factor_vs_base_rank_corr_mean",
                ]
            ],
            float_precision=5,
        ),
        "",
        "## Interpretation",
        "- If a variant with MA improves significance, the trend-line component is the likely information source.",
        "- If a variant with volume improves significance, volume-path information is contributing.",
        "- If `high_low_range` wins, intrawindow volatility/range shape matters more than open-close ticks.",
        "- If `close_only` wins, most usable information is in the price path itself, not richer candle details.",
        "",
        "## Output Files",
        f"- `{out_dir / 'ablation_summary.csv'}`",
        f"- `{out_dir / 'ablation_significance.csv'}`",
        f"- `{out_dir / 'ablation_ensemble_search.csv'}`",
    ]
    return "\n".join(lines) + "\n"


def run_variant(
    args: argparse.Namespace,
    variant: str,
    reference_dates: list[pd.Timestamp],
    base_grid: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variant_dir = Path(args.output_dir) / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    config = _variant_config(args, variant, variant_dir)

    print(f"\n=== Image factor ablation: {variant} ===", flush=True)
    image_panel = ext.build_image_factor_panel(config, max_folds=args.max_folds, fold_dates=reference_dates)
    grid = _grid_key(image_panel)
    if base_grid is not None and not grid.equals(base_grid):
        raise ValueError(f"{variant} does not match the reference date-asset OOS grid")

    _export_date_csv(image_panel, variant_dir / "image_factor_panel.csv")
    controls = ext.build_common_pca_controls(
        signal_dates=image_panel["date"].tolist(),
        returns_path=Path(args.returns_path),
        window=args.rolling_pca_window,
    )
    _export_date_csv(controls, variant_dir / "common_pca_controls.csv")

    significance, merged = ext.run_factor_significance(image_panel, controls)
    significance.to_csv(variant_dir / "image_factor_significance.csv", index=False)

    ensemble = ext.run_ensemble_extension(
        panel=merged,
        horizon=args.horizon,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    ensemble.to_csv(variant_dir / "ensemble_image_factor_search.csv", index=False)
    ext.export_signal_panels(
        variant_dir,
        image_panel,
        model_prefix=f"cnn_2d_residual_small_{variant}",
    )

    report = _build_variant_report(variant, variant_dir, config, image_panel, controls, significance, ensemble)
    (variant_dir / "image_factor_report.md").write_text(report, encoding="utf-8")
    (variant_dir / "image_factor_config.json").write_text(
        json.dumps({**config.to_dict(), "rolling_pca_window": args.rolling_pca_window}, indent=2),
        encoding="utf-8",
    )

    significance = significance.copy()
    significance.insert(0, "chart_variant", variant)
    ensemble = ensemble.copy()
    ensemble.insert(0, "chart_variant", variant)
    summary = pd.DataFrame([_summary_row(variant, significance, ensemble)])
    return summary, significance, ensemble, grid


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Jiang-style image factor ablations.")
    p.add_argument("--data-path", default="etfdata.csv")
    p.add_argument("--returns-path", default="ode_inputs_cnn/returns_daily.csv")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--variants", nargs="*", default=ABLATION_VARIANTS)
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
    p.add_argument("--force", action="store_true", help="recompute variants even if completed outputs exist")
    return p


def main() -> None:
    args = build_parser().parse_args()
    invalid = sorted(set(args.variants) - CHART_VARIANTS)
    if invalid:
        raise ValueError(f"unsupported variants: {invalid}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_dates = _reference_fold_dates(args)

    summaries = []
    significances = []
    ensembles = []
    base_grid = None
    for variant in args.variants:
        variant_dir = out_dir / variant
        if not args.force and _variant_outputs_complete(variant_dir):
            print(f"\n=== Image factor ablation: {variant} already complete; loading outputs ===", flush=True)
            summary, significance, ensemble, grid = _load_completed_variant(variant, variant_dir)
        else:
            summary, significance, ensemble, grid = run_variant(args, variant, reference_dates, base_grid)
        if base_grid is None:
            base_grid = grid
        elif not grid.equals(base_grid):
            raise ValueError(f"{variant} does not match the reference date-asset OOS grid")
        summaries.append(summary)
        significances.append(significance)
        ensembles.append(ensemble)

    summary_df = pd.concat(summaries, ignore_index=True)
    significance_df = pd.concat(significances, ignore_index=True)
    ensemble_df = pd.concat(ensembles, ignore_index=True)
    summary_df.to_csv(out_dir / "ablation_summary.csv", index=False)
    significance_df.to_csv(out_dir / "ablation_significance.csv", index=False)
    ensemble_df.to_csv(out_dir / "ablation_ensemble_search.csv", index=False)
    report = _build_ablation_report(out_dir, args, summary_df, significance_df, ensemble_df)
    (out_dir / "image_factor_ablation_report.md").write_text(report, encoding="utf-8")

    print("\n=== Ablation Summary ===", flush=True)
    print(summary_df.to_string(index=False))
    print(f"\nOutputs saved to {out_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted by Ctrl+C. Re-run the same command to resume completed variants, "
            "or add --force to recompute from scratch.",
            flush=True,
        )
        raise SystemExit(130)
