#!/usr/bin/env python3
"""Collect Sigma(t) inputs for the final ODE handoff folder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
OUT_DIR = ROOT / "ode_inputs_cnn" / "final_ode_mu_inputs"
SOURCE_BUNDLE = ROOT / "ode_inputs_cnn" / "ensemble_best" / "ode_bundle.csv"
SOURCE_RETURNS = ROOT / "ode_inputs_cnn" / "returns_daily.csv"
SELECTED_MU = OUT_DIR / "selected_mu_input.csv"
SIGMA_WINDOW = 60


def _date_norm(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed").dt.normalize()


def _assets_from_bundle(bundle: pd.DataFrame) -> list[str]:
    return sorted(col[:-9] for col in bundle.columns if col.endswith("_sigma_ii"))


def _cov_columns(bundle: pd.DataFrame) -> list[str]:
    return [col for col in bundle.columns if col.endswith("_cov")]


def build_sigma_files() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = pd.read_csv(SOURCE_BUNDLE, parse_dates=["date"])
    selected_mu = pd.read_csv(SELECTED_MU, usecols=["date"], parse_dates=["date"])
    bundle["date"] = _date_norm(bundle["date"])
    selected_dates = set(_date_norm(selected_mu["date"]))
    bundle = bundle.loc[bundle["date"].isin(selected_dates)].copy().sort_values("date")
    if bundle.empty:
        raise ValueError("no Sigma rows overlap selected mu input dates")

    assets = _assets_from_bundle(bundle)
    sigma_cols = [f"{asset}_sigma_ii" for asset in assets]
    cov_cols = _cov_columns(bundle)
    keep_cols = ["date"] + sigma_cols + cov_cols
    sigma_mask = bundle[sigma_cols + cov_cols].notna().all(axis=1)
    sigma_wide = bundle.loc[sigma_mask, keep_cols].copy()
    sigma_wide = sigma_wide.drop_duplicates(["date"]).sort_values("date").reset_index(drop=True)
    if sigma_wide[sigma_cols + cov_cols].isna().any().any():
        raise ValueError("NaN values found in Sigma wide matrix")
    if set(sigma_wide["date"]) != selected_dates:
        missing = sorted(selected_dates - set(sigma_wide["date"]))
        raise ValueError(f"Sigma dates do not fully cover selected mu dates; missing={len(missing)}")

    long_rows = []
    for _, row in sigma_wide.iterrows():
        date = row["date"]
        for asset in assets:
            long_rows.append(
                {
                    "date": date,
                    "asset_i": asset,
                    "asset_j": asset,
                    "sigma_value": float(row[f"{asset}_sigma_ii"]),
                    "entry_type": "variance",
                }
            )
        for col in cov_cols:
            # Asset names contain underscores; find the split by known asset names.
            matches = [(a, b) for a in assets for b in assets if a != b and f"{a}_{b}_cov" == col]
            if not matches:
                raise ValueError(f"cannot parse covariance column: {col}")
            asset_i, asset_j = matches[0]
            value = float(row[col])
            long_rows.append(
                {
                    "date": date,
                    "asset_i": asset_i,
                    "asset_j": asset_j,
                    "sigma_value": value,
                    "entry_type": "covariance",
                }
            )
            long_rows.append(
                {
                    "date": date,
                    "asset_i": asset_j,
                    "asset_j": asset_i,
                    "sigma_value": value,
                    "entry_type": "covariance",
                }
            )

    sigma_long = pd.DataFrame(long_rows).sort_values(["date", "asset_i", "asset_j"]).reset_index(drop=True)
    if sigma_long.duplicated(["date", "asset_i", "asset_j"]).any():
        raise ValueError("duplicate Sigma long entries")
    if sigma_long["sigma_value"].isna().any():
        raise ValueError("NaN values found in Sigma long matrix")

    export_wide = sigma_wide.copy()
    export_long = sigma_long.copy()
    export_wide["date"] = export_wide["date"].dt.strftime("%Y-%m-%d")
    export_long["date"] = export_long["date"].dt.strftime("%Y-%m-%d")
    export_wide.to_csv(OUT_DIR / "sigma_wide.csv", index=False)
    export_long.to_csv(OUT_DIR / "sigma_long.csv", index=False)

    returns = pd.read_csv(SOURCE_RETURNS, parse_dates=["date"])
    returns["date"] = _date_norm(returns["date"])
    returns = returns.loc[returns["date"].isin(selected_dates)].sort_values("date")
    returns_export = returns.copy()
    returns_export["date"] = returns_export["date"].dt.strftime("%Y-%m-%d")
    returns_export.to_csv(OUT_DIR / "returns_for_ode.csv", index=False)

    diag_values = sigma_wide[sigma_cols].to_numpy(dtype=float)
    metadata = {
        "source_bundle": str(SOURCE_BUNDLE.relative_to(ROOT)),
        "source_returns": str(SOURCE_RETURNS.relative_to(ROOT)),
        "sigma_window": SIGMA_WINDOW,
        "scale": "daily log-return covariance",
        "date_start": str(sigma_wide["date"].min().date()),
        "date_end": str(sigma_wide["date"].max().date()),
        "n_dates": int(sigma_wide["date"].nunique()),
        "n_assets": int(len(assets)),
        "assets": assets,
        "wide_file": "sigma_wide.csv",
        "long_file": "sigma_long.csv",
        "returns_file": "returns_for_ode.csv",
        "min_variance": float(np.nanmin(diag_values)),
        "max_variance": float(np.nanmax(diag_values)),
    }
    (OUT_DIR / "sigma_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    metadata = build_sigma_files()
    print("Final ODE Sigma inputs saved")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
