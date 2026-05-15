#!/usr/bin/env python3
"""Fast robustness screen for the selected image-factor specification.

This script keeps the chart variant fixed at `ohlc_ma_volume` and checks
whether the image-factor significance direction survives simple
lookback/horizon changes. It intentionally focuses on factor significance,
not ODE optimization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_image_factor_extension as ext
from src.config import PipelineConfig


ROOT = Path(__file__).parent
DEFAULT_OUT = ROOT / "ode_inputs_cnn" / "image_factor_robustness"
CHART_VARIANT = "ohlc_ma_volume"
DEFAULT_CONFIGS = ["20:20", "60:20", "60:60"]


def _parse_window_config(value: str) -> tuple[int, int]:
    try:
        lookback, horizon = value.split(":", 1)
        return int(lookback), int(horizon)
    except Exception as exc:
        raise argparse.ArgumentTypeError("window config must look like LOOKBACK:HORIZON") from exc


def _date_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def _markdown_table(df: pd.DataFrame, digits: int = 5) -> str:
    headers = list(df.columns)

    def fmt(value) -> str:
        if isinstance(value, float):
            if np.isnan(value):
                return "nan"
            return f"{value:.{digits}f}"
        return str(value)

    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(fmt(row[col]) for col in headers) + " |")
    return "\n".join(rows)


def _config(args: argparse.Namespace, lookback: int, horizon: int, out_dir: Path) -> PipelineConfig:
    return PipelineConfig(
        data_path=args.data_path,
        output_dir=str(out_dir),
        lookback=lookback,
        horizon=horizon,
        label_mode="regression",
        target_name="future_return",
        enabled_models=["cnn_2d_residual_small"],
        include_moving_average=True,
        include_volume=True,
        chart_variant=CHART_VARIANT,
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


def run_one(args: argparse.Namespace, lookback: int, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tag = f"lookback_{lookback}_horizon_{horizon}"
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _config(args, lookback, horizon, out_dir)

    print(f"\n=== Robustness: {tag} ===", flush=True)
    image_panel = ext.build_image_factor_panel(config, max_folds=args.max_folds)
    _date_csv(image_panel, out_dir / "image_factor_panel.csv")

    controls = ext.build_common_pca_controls(
        signal_dates=image_panel["date"].tolist(),
        returns_path=Path(args.returns_path),
        window=args.rolling_pca_window,
    )
    _date_csv(controls, out_dir / "common_pca_controls.csv")

    significance, merged = ext.run_factor_significance(image_panel, controls)
    significance.to_csv(out_dir / "image_factor_significance.csv", index=False)

    best_by_p = significance.sort_values(["p_value", "delta_r2"], ascending=[True, False]).iloc[0]
    best_by_rank = significance.sort_values("daily_rank_correlation", ascending=False).iloc[0]
    summary = pd.DataFrame(
        [
            {
                "window_tag": tag,
                "lookback": lookback,
                "horizon": horizon,
                "chart_variant": CHART_VARIANT,
                "n_rows": len(image_panel),
                "n_dates": image_panel["date"].nunique(),
                "n_folds": image_panel["fold"].nunique(),
                "best_p_factor": best_by_p["factor_name"],
                "best_p_t_stat": float(best_by_p["t_stat"]),
                "best_p_value": float(best_by_p["p_value"]),
                "best_p_delta_r2": float(best_by_p["delta_r2"]),
                "best_p_daily_rank_corr": float(best_by_p["daily_rank_correlation"]),
                "best_rank_factor": best_by_rank["factor_name"],
                "best_daily_rank_corr": float(best_by_rank["daily_rank_correlation"]),
                "best_rank_p_value": float(best_by_rank["p_value"]),
                "best_rank_delta_r2": float(best_by_rank["delta_r2"]),
            }
        ]
    )
    summary.to_csv(out_dir / "robustness_summary.csv", index=False)
    (out_dir / "config.json").write_text(
        json.dumps({**config.to_dict(), "rolling_pca_window": args.rolling_pca_window}, indent=2),
        encoding="utf-8",
    )
    return summary, significance.assign(window_tag=tag, lookback=lookback, horizon=horizon)


def build_report(args: argparse.Namespace, summary: pd.DataFrame, significance: pd.DataFrame) -> str:
    lines = [
        "# Image Factor Robustness Screen",
        "",
        "## Scope",
        "",
        f"- Chart variant fixed to `{CHART_VARIANT}`.",
        "- Model fixed to `cnn_2d_residual_small`.",
        "- This is a robustness screen for image-factor significance, not an ODE backtest.",
        f"- Max folds: `{args.max_folds}`",
        f"- CNN epochs / patience: `{args.cnn_epochs}/{args.patience}`",
        "",
        "## Summary",
        "",
        _markdown_table(summary, digits=5),
        "",
        "## Interpretation",
        "",
        "- If the same factor direction remains positive across windows, the final `60/20` choice is more robust.",
        "- If p-values or rank correlations vary strongly, treat the final input as horizon/window-specific.",
        "- Full confirmation requires rerunning with `--max-folds` unset and `--cnn-epochs 30 --patience 5`.",
        "",
        "## All Significance Rows",
        "",
        _markdown_table(
            significance[
                [
                    "window_tag",
                    "factor_name",
                    "t_stat",
                    "p_value",
                    "delta_r2",
                    "daily_rank_correlation",
                    "n_obs",
                    "n_dates",
                ]
            ],
            digits=5,
        ),
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a fast image-factor robustness screen.")
    p.add_argument("--data-path", default="etfdata.csv")
    p.add_argument("--returns-path", default="ode_inputs_cnn/returns_daily.csv")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS, help="Window configs as LOOKBACK:HORIZON")
    p.add_argument("--image-height", type=int, default=64)
    p.add_argument("--wf-min-train-days", type=int, default=500)
    p.add_argument("--wf-val-days", type=int, default=60)
    p.add_argument("--wf-test-days", type=int, default=60)
    p.add_argument("--cnn-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--rolling-pca-window", type=int, default=252)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-folds", type=int, default=6)
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    sigs = []
    for value in args.configs:
        lookback, horizon = _parse_window_config(value)
        summary, significance = run_one(args, lookback, horizon)
        summaries.append(summary)
        sigs.append(significance)

    summary_df = pd.concat(summaries, ignore_index=True)
    sig_df = pd.concat(sigs, ignore_index=True)
    summary_df.to_csv(out_dir / "robustness_summary.csv", index=False)
    sig_df.to_csv(out_dir / "robustness_significance.csv", index=False)
    report = build_report(args, summary_df, sig_df)
    (out_dir / "image_factor_robustness_report.md").write_text(report, encoding="utf-8")

    print("\n=== Robustness Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nOutputs saved to {out_dir}")


if __name__ == "__main__":
    main()
