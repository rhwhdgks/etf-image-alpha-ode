#!/usr/bin/env python3
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd


try:
    ROOT = Path(__file__).parent
except NameError:
    ROOT = Path.cwd()

OUT = ROOT / "ode_inputs_cnn"
OUT.mkdir(parents=True, exist_ok=True)

import pandas as pd

path = ROOT / "outputs_walkforward_cnnlstm" / "walkforward_predictions.csv"
df = pd.read_csv(path)

df["model_name"] = df["model_name"].replace({
    "cnn_lstm_image_scale": "cnnlstm_image_scale",
    "cnn_lstm_cumulative_scale": "cnnlstm_cumulative_scale",
})

df.to_csv(path, index=False)

SOURCES = {
    "logistic_cumulative_scale": "outputs_walkforward_base/walkforward_predictions.csv",
    "logistic_image_scale": "outputs_walkforward_base/walkforward_predictions.csv",
    "cnn_1d_image_scale": "outputs_walkforward_base/walkforward_predictions.csv",
    "cnn_1d_attention_image_scale": "outputs_walkforward_base/walkforward_predictions.csv",
    "cnn_1d_dilated_image_scale": "outputs_walkforward_base/walkforward_predictions.csv",

    "cnn_lstm_image_scale": "results/lstm_walkforward/lstm_best_predictions.csv",

    "cnnlstm_image_scale": "outputs_walkforward_cnnlstm/walkforward_predictions.csv",
    "cnnlstm_cumulative_scale": "outputs_walkforward_cnnlstm/walkforward_predictions.csv",
}

HORIZON = 20
TOP_K = 2

REQUIRED_COLS = {
    "date",
    "asset",
    "signal_value",
    "future_return",
    "model_name",
}


def resolve_prediction_path(src: str) -> Path:
    path = ROOT / src
    if path.suffix.lower() == ".csv":
        return path
    return path / "walkforward_predictions.csv"


def read_model_predictions(model_name: str, src: str) -> pd.DataFrame:
    path = resolve_prediction_path(src)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file for {model_name}: {path}\n"
            f"현재 ROOT = {ROOT}"
        )

    df = pd.read_csv(path)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df[df["model_name"] == model_name][
        ["date", "asset", "signal_value", "future_return"]
    ].copy()

    if df.empty:
        available = pd.read_csv(path, usecols=["model_name"])["model_name"].unique()
        raise ValueError(
            f"No rows found for model_name='{model_name}' in {path}\n"
            f"Available model_name values: {available}"
        )

    # 날짜 형식 통일: 'YYYY-MM-DD'와 'YYYY-MM-DD HH:MM:SS'가 섞여도 날짜만 남김
    df["date"] = pd.to_datetime(df["date"], format="mixed").dt.normalize()

    df["model_name"] = model_name
    return df


def load_long() -> pd.DataFrame:
    frames = []

    for model_name, src in SOURCES.items():
        sub = read_model_predictions(model_name, src)
        print(
            f"  loaded {model_name:32s} "
            f"rows={len(sub):6d} dates={sub['date'].nunique():4d}"
        )
        frames.append(sub)

    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"], format="mixed").dt.normalize()
    return long


def pivot_wide(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = long.pivot_table(
        index=["date", "asset"],
        columns="model_name",
        values="signal_value",
    )

    rank = (
        long.assign(
            rk=long.groupby(["date", "model_name"])["signal_value"].rank(pct=True)
        )
        .pivot_table(
            index=["date", "asset"],
            columns="model_name",
            values="rk",
        )
    )

    future = long.groupby(["date", "asset"])["future_return"].first()

    raw["future_return"] = future
    rank["future_return"] = future

    return raw.reset_index(), rank.reset_index()


def eval_combo(df: pd.DataFrame, models: list[str]) -> dict:
    signal = df[models].mean(axis=1)

    tmp = df[["date", "asset", "future_return"]].copy()
    tmp["signal"] = signal.values

    # 날짜별 Spearman rank correlation 평균
    rank_corrs = []

    for _, g in tmp.groupby("date"):
        if len(g) < 2:
            continue

        corr = g["signal"].rank().corr(g["future_return"].rank())

        if pd.notna(corr):
            rank_corrs.append(corr)

    rank_corr = float(np.mean(rank_corrs)) if rank_corrs else np.nan

    # top-k portfolio Sharpe
    rebalance_dates = sorted(tmp["date"].unique())[::HORIZON]

    rets = []
    for d in rebalance_dates:
        frame = tmp[tmp["date"] == d].sort_values(
            "signal",
            ascending=False,
        )

        if len(frame) < TOP_K:
            continue

        rets.append(float(frame.head(TOP_K)["future_return"].mean()))

    r = np.array(rets, dtype=float)

    if len(r) < 2 or r.std(ddof=1) == 0:
        sharpe = np.nan
    else:
        sharpe = float(
            r.mean() / r.std(ddof=1) * math.sqrt(252.0 / HORIZON)
        )

    cumulative_return = float(np.prod(1 + r) - 1) if len(r) else np.nan
    hit_rate = float((r > 0).mean()) if len(r) else np.nan

    return {
        "rank_corr": rank_corr,
        "top_k_sharpe": sharpe,
        "top_k_cum": cumulative_return,
        "top_k_hit": hit_rate,
    }


def search(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    models = list(SOURCES.keys())
    rows = []

    for k in range(1, 5):
        for combo in itertools.combinations(models, k):
            result = eval_combo(df, list(combo))

            rows.append(
                {
                    "mode": mode,
                    "k": k,
                    "members": " + ".join(combo),
                    **result,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    print(f"ROOT = {ROOT}")
    print("Loading predictions ...")
    long = load_long()

    print("Pivoting predictions ...")
    raw_wide, rank_wide = pivot_wide(long)

    print(f"  raw_wide shape: {raw_wide.shape}")
    print(f"  rank_wide shape: {rank_wide.shape}")

    print("Searching ensemble combinations: raw mean ...")
    raw_results = search(raw_wide, "raw")

    print("Searching ensemble combinations: rank mean ...")
    rank_results = search(rank_wide, "rank")

    full = pd.concat([raw_results, rank_results], ignore_index=True)

    search_csv = OUT / "ensemble_search.csv"
    full.to_csv(search_csv, index=False)
    print(f"wrote {search_csv} ({len(full)} rows)")

    lines = ["# Ensemble search — top-20 per aggregation mode", ""]

    for mode in ["raw", "rank"]:
        sub_rank = (
            full[full["mode"] == mode]
            .sort_values("rank_corr", ascending=False)
            .head(20)
        )

        lines.append(f"## Mode: {mode} — sorted by OOS rank corr")
        lines.append("")
        lines.append(
            sub_rank[
                [
                    "k",
                    "members",
                    "rank_corr",
                    "top_k_sharpe",
                    "top_k_cum",
                    "top_k_hit",
                ]
            ].to_markdown(index=False, floatfmt=".4f")
        )
        lines.append("")

        sub_sharpe = (
            full[full["mode"] == mode]
            .sort_values("top_k_sharpe", ascending=False)
            .head(20)
        )

        lines.append(f"## Mode: {mode} — sorted by top-k Sharpe")
        lines.append("")
        lines.append(
            sub_sharpe[
                [
                    "k",
                    "members",
                    "rank_corr",
                    "top_k_sharpe",
                    "top_k_cum",
                    "top_k_hit",
                ]
            ].to_markdown(index=False, floatfmt=".4f")
        )
        lines.append("")

    top_md = OUT / "ensemble_search_top.md"
    top_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {top_md}")

    print("\n=== TOP-5 by rank corr ===")
    print(
        full.sort_values("rank_corr", ascending=False)
        .head(5)[["mode", "k", "members", "rank_corr", "top_k_sharpe"]]
        .to_string(index=False)
    )

    print("\n=== TOP-5 by Sharpe ===")
    print(
        full.sort_values("top_k_sharpe", ascending=False)
        .head(5)[["mode", "k", "members", "rank_corr", "top_k_sharpe"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()