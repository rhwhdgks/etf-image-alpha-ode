#!/usr/bin/env python3
"""Collect final ODE mu(t) candidate inputs in one handoff folder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
SOURCE_SIGNALS = ROOT / "ode_inputs_cnn" / "image_factor_ablation" / "ohlc_ma_volume" / "ode_mu_candidate_signals.csv"
SOURCE_PERFORMANCE = ROOT / "ode_inputs_cnn" / "image_factor_ablation" / "ablation_ensemble_search.csv"
OUT_DIR = ROOT / "ode_inputs_cnn" / "final_ode_mu_inputs"
LOOKBACK = 60
HORIZON = 20
CHART_VARIANT = "ohlc_ma_volume"
RECOMMENDED_INPUT = "mu_image_factor_rank"

INPUTS = {
    "mu_sharpe_baseline": {
        "source_model_name": "ensemble_best",
        "role": "Sharpe-stable baseline from prior ensemble sprint",
    },
    "mu_rank_baseline": {
        "source_model_name": "ensemble_4family",
        "role": "Rank-correlation baseline before image-factor ablation add-on",
    },
    "mu_image_factor_rank": {
        "source_model_name": "ensemble_4family+image_factor_pc1",
        "role": "Recommended ranking-improved input using full OHLC+MA+Volume image factor PC1",
    },
    "mu_image_factor_balanced": {
        "source_model_name": "ensemble_best+image_factor_pc1",
        "role": "Secondary image-factor input with higher Sharpe than the 4-family add-on",
    },
}


def _date_norm(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed").dt.normalize()


def _zscore(group: pd.Series) -> pd.Series:
    std = group.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(np.zeros(len(group), dtype=float), index=group.index)
    return (group - group.mean()) / std


def build_long_inputs() -> pd.DataFrame:
    source = pd.read_csv(SOURCE_SIGNALS, parse_dates=["date"])
    source["date"] = _date_norm(source["date"])

    frames = []
    for input_name, meta in INPUTS.items():
        model_name = meta["source_model_name"]
        frame = source.loc[source["model_name"] == model_name].copy()
        if frame.empty:
            raise ValueError(f"missing source model in {SOURCE_SIGNALS}: {model_name}")
        frame["input_name"] = input_name
        frame["source_model_name"] = model_name
        frame["input_role"] = meta["role"]
        frame["is_recommended"] = input_name == RECOMMENDED_INPUT
        frames.append(frame)

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.rename(columns={"signal_value": "mu_raw_score"})
    long_df["mu_rank"] = (
        long_df.groupby(["date", "input_name"])["mu_raw_score"]
        .rank(method="average", pct=True)
        .astype(float)
    )
    long_df["mu_centered_rank"] = (
        long_df["mu_rank"]
        - long_df.groupby(["date", "input_name"])["mu_rank"].transform("mean")
    )
    long_df["mu_zscore"] = (
        long_df.groupby(["date", "input_name"])["mu_raw_score"]
        .transform(_zscore)
        .astype(float)
    )
    long_df["mu_signal"] = long_df["mu_centered_rank"]
    long_df["mu_signal_type"] = "cross_sectional_centered_rank"
    long_df["lookback"] = LOOKBACK
    long_df["horizon"] = HORIZON
    long_df["chart_variant"] = CHART_VARIANT
    long_df["source_file"] = str(SOURCE_SIGNALS.relative_to(ROOT))

    columns = [
        "date",
        "asset",
        "input_name",
        "source_model_name",
        "mu_signal",
        "mu_signal_type",
        "mu_raw_score",
        "mu_rank",
        "mu_centered_rank",
        "mu_zscore",
        "future_return",
        "is_recommended",
        "input_role",
        "lookback",
        "horizon",
        "chart_variant",
        "source_file",
    ]
    long_df = long_df[columns].sort_values(["date", "asset", "input_name"]).reset_index(drop=True)
    if long_df.duplicated(["date", "asset", "input_name"]).any():
        raise ValueError("duplicate date-asset-input rows in final ODE mu inputs")
    if long_df[["mu_signal", "mu_raw_score", "mu_rank", "mu_centered_rank", "mu_zscore"]].isna().any().any():
        raise ValueError("NaN values found in final ODE mu inputs")
    return long_df


def build_wide_inputs(long_df: pd.DataFrame) -> pd.DataFrame:
    base = long_df[["date", "asset", "future_return"]].drop_duplicates(["date", "asset"])
    for value_col in ["mu_signal", "mu_raw_score", "mu_zscore"]:
        wide = long_df.pivot(index=["date", "asset"], columns="input_name", values=value_col)
        wide.columns = [f"{value_col}_{col}" for col in wide.columns]
        wide = wide.reset_index()
        base = base.merge(wide, on=["date", "asset"], how="left")
    return base.sort_values(["date", "asset"]).reset_index(drop=True)


def build_performance_summary() -> pd.DataFrame:
    perf = pd.read_csv(SOURCE_PERFORMANCE)
    perf = perf.loc[perf["chart_variant"] == CHART_VARIANT].copy()
    rows = []
    for input_name, meta in INPUTS.items():
        model_name = meta["source_model_name"]
        row = perf.loc[perf["candidate_name"] == model_name]
        if row.empty:
            raise ValueError(f"missing performance row for {model_name}")
        row = row.iloc[0].to_dict()
        row["input_name"] = input_name
        row["source_model_name"] = model_name
        row["is_recommended"] = input_name == RECOMMENDED_INPUT
        row["input_role"] = meta["role"]
        rows.append(row)

    columns = [
        "input_name",
        "source_model_name",
        "is_recommended",
        "rank_corr",
        "top_k_sharpe",
        "top_k_cumulative_return",
        "top_k_hit_rate",
        "turnover",
        "rank_corr_diff_vs_base",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "factor_vs_base_rank_corr_mean",
        "input_role",
    ]
    return pd.DataFrame(rows)[columns].sort_values(["is_recommended", "rank_corr"], ascending=[False, False])


def write_readme(out_dir: Path, performance: pd.DataFrame, long_df: pd.DataFrame) -> None:
    recommended = performance.loc[performance["is_recommended"]].iloc[0]
    lines = [
        "# Final ODE Mu Inputs",
        "",
        "## Purpose",
        "",
        "This folder collects the final `mu(t)` candidate inputs from the image-factor sprint.",
        "Optimization and ODE solver integration are intentionally excluded here.",
        "",
        "## Recommended Input",
        "",
        f"- Recommended input: `{recommended['input_name']}`",
        f"- Source model: `{recommended['source_model_name']}`",
        "- Signal type: `cross_sectional_centered_rank`",
        f"- Lookback / horizon: `{LOOKBACK}/{HORIZON}`",
        f"- Chart variant: `{CHART_VARIANT}`",
        f"- OOS rows per input: `{long_df['date'].nunique() * long_df['asset'].nunique()}`",
        f"- OOS dates: `{long_df['date'].nunique()}`",
        "",
        "The recommended input is designed as an expected-return ranking signal, not as a standalone trading rule.",
        "",
        "## Performance Snapshot",
        "",
        "| input_name | source_model_name | rank_corr | Sharpe | rank_corr_lift | bootstrap_CI |",
        "|---|---|---:|---:|---:|---|",
    ]
    for _, row in performance.iterrows():
        ci = f"[{row['bootstrap_ci_low']:.5f}, {row['bootstrap_ci_high']:.5f}]"
        lines.append(
            f"| `{row['input_name']}` | `{row['source_model_name']}` | "
            f"{row['rank_corr']:.4f} | {row['top_k_sharpe']:.4f} | "
            f"{row['rank_corr_diff_vs_base']:.4f} | {ci} |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "| file | use |",
            "|---|---|",
            "| `final_mu_inputs_long.csv` | Main handoff file. One row per `date-asset-input`. |",
            "| `selected_mu_input.csv` | Recommended input only: `mu_image_factor_rank`. |",
            "| `final_mu_inputs_wide.csv` | Convenience wide table with all candidate inputs. |",
            "| `input_performance_summary.csv` | Rank-corr, Sharpe, bootstrap CI summary. |",
            "| `manifest.json` | Machine-readable metadata. |",
            "",
            "## Column Guide",
            "",
            "- `mu_signal`: primary ODE input, equal to date-wise centered cross-sectional rank.",
            "- `mu_raw_score`: original ensemble/image-factor score.",
            "- `mu_rank`: date-wise percentile rank of `mu_raw_score`.",
            "- `mu_centered_rank`: `mu_rank` demeaned within each date and input.",
            "- `mu_zscore`: date-wise z-score of `mu_raw_score`.",
            "- `future_return`: realized 20-day target return, included for evaluation only. Do not use as an input.",
            "",
            "## Interpretation",
            "",
            "`mu_image_factor_rank` improves ranking quality versus `ensemble_4family` "
            "but lowers simple top-k Sharpe. Use it as an ODE expected-return ranking input and compare "
            "against `mu_sharpe_baseline` in the optimizer.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df = build_long_inputs()
    wide_df = build_wide_inputs(long_df)
    selected = long_df.loc[long_df["input_name"] == RECOMMENDED_INPUT].copy()
    performance = build_performance_summary()

    export_long = long_df.copy()
    export_wide = wide_df.copy()
    export_selected = selected.copy()
    for frame in [export_long, export_wide, export_selected]:
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")

    export_long.to_csv(OUT_DIR / "final_mu_inputs_long.csv", index=False)
    export_wide.to_csv(OUT_DIR / "final_mu_inputs_wide.csv", index=False)
    export_selected.to_csv(OUT_DIR / "selected_mu_input.csv", index=False)
    performance.to_csv(OUT_DIR / "input_performance_summary.csv", index=False)
    write_readme(OUT_DIR, performance, long_df)

    manifest = {
        "recommended_input": RECOMMENDED_INPUT,
        "source_signals": str(SOURCE_SIGNALS.relative_to(ROOT)),
        "source_performance": str(SOURCE_PERFORMANCE.relative_to(ROOT)),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "chart_variant": CHART_VARIANT,
        "signal_type": "cross_sectional_centered_rank",
        "inputs": INPUTS,
        "n_rows_long": int(len(long_df)),
        "n_rows_selected": int(len(selected)),
        "n_dates": int(long_df["date"].nunique()),
        "n_assets": int(long_df["asset"].nunique()),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Final ODE mu inputs saved to {OUT_DIR}")
    print(performance.to_string(index=False))


if __name__ == "__main__":
    main()
