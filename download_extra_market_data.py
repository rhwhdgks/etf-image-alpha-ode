#!/usr/bin/env python3
"""Download extra ETF OHLCV data for CNN image pretraining.

The final ODE universe remains the original 7 ETF assets. This file only
builds an auxiliary market-image dataset with a matched historical period.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).parent
OUT_DIR = ROOT / "data" / "extra_market"
DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE",
    "EFA", "EEM", "EWJ", "EWY", "FXI", "VGK",
    "SHY", "IEF", "TLT", "LQD", "HYG",
    "GLD", "SLV", "USO", "VNQ", "DBC",
]
FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def _normalize_download(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("download returned an empty frame")
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("expected yfinance multi-index columns for multiple tickers")

    frames = []
    for ticker in tickers:
        if ticker not in raw.columns.get_level_values(0):
            continue
        frame = raw[ticker].copy()
        keep = [field for field in FIELDS if field in frame.columns]
        if len(keep) < len(FIELDS):
            continue
        frame = frame[keep].rename(columns=str.lower)
        frame["ticker"] = ticker
        frame["date"] = pd.to_datetime(frame.index).tz_localize(None).normalize()
        frames.append(frame.reset_index(drop=True))

    if not frames:
        raise ValueError("no ticker had complete OHLCV fields")

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.dropna(subset=["open", "high", "low", "close", "volume"])
    long_df = long_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    if long_df.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate date-ticker rows after download")
    return long_df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def _build_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for field in ["close", "high", "low", "open", "volume"]:
        wide = long_df.pivot(index="date", columns="ticker", values=field)
        wide.columns = [f"{field}_{col.lower()}" for col in wide.columns]
        pieces.append(wide)
    return pd.concat(pieces, axis=1).sort_index().reset_index()


def _summary(long_df: pd.DataFrame, args: argparse.Namespace) -> dict:
    coverage = (
        long_df.groupby("ticker")
        .agg(
            start=("date", "min"),
            end=("date", "max"),
            rows=("date", "size"),
            missing_close=("close", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    return {
        "purpose": "auxiliary ETF OHLCV data for CNN image pretraining/robustness",
        "final_ode_universe": "unchanged original 7 ETF assets",
        "source": "Yahoo Finance via yfinance",
        "auto_adjust": True,
        "requested_start": args.start,
        "requested_end": args.end,
        "n_requested_tickers": len(args.tickers),
        "n_downloaded_tickers": int(coverage["ticker"].nunique()),
        "n_rows_long": int(len(long_df)),
        "date_start": str(long_df["date"].min().date()),
        "date_end": str(long_df["date"].max().date()),
        "coverage": [
            {
                "ticker": row.ticker,
                "start": str(row.start.date()),
                "end": str(row.end.date()),
                "rows": int(row.rows),
                "missing_close": int(row.missing_close),
            }
            for row in coverage.itertuples(index=False)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download extra ETF OHLCV data")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2026-03-10")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = yf.download(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    long_df = _normalize_download(raw, args.tickers)
    wide_df = _build_wide(long_df)
    summary = _summary(long_df, args)

    long_export = long_df.copy()
    wide_export = wide_df.copy()
    long_export["date"] = long_export["date"].dt.strftime("%Y-%m-%d")
    wide_export["date"] = wide_export["date"].dt.strftime("%Y-%m-%d")

    long_export.to_csv(out_dir / "extra_etf_ohlcv_long.csv", index=False)
    wide_export.to_csv(out_dir / "extra_etf_ohlcv_wide.csv", index=False)
    (out_dir / "extra_etf_ohlcv_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"saved: {out_dir}")
    print(f"tickers: {summary['n_downloaded_tickers']}/{summary['n_requested_tickers']}")
    print(f"rows: {summary['n_rows_long']}")
    print(f"dates: {summary['date_start']} -> {summary['date_end']}")


if __name__ == "__main__":
    main()
