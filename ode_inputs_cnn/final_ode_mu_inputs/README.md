# Final ODE Mu Inputs

## Purpose

This folder collects the final `mu(t)` candidate inputs from the image-factor sprint.

The scope is limited to signal construction and handoff. ODE solver integration, portfolio weight optimization, and fund-level policy constraints are not included here.

## Folder Location

Repository root:

```text
/home/jonghan/findalpha/delta/sprint2
```

Final handoff folder:

```text
ode_inputs_cnn/final_ode_mu_inputs/
```

Main recommended file:

```text
ode_inputs_cnn/final_ode_mu_inputs/selected_mu_input.csv
```

## Recommended Input

| item | value |
|---|---|
| Recommended input | `mu_image_factor_rank` |
| Source model | `ensemble_4family+image_factor_pc1` |
| Signal type | `cross_sectional_centered_rank` |
| Chart variant | `ohlc_ma_volume` |
| Lookback / horizon | `60 / 20` |
| OOS rows per input | `20,160` |
| OOS dates | `2,880` |
| Assets | `7` |

The recommended input is an expected-return ranking signal. It should be treated as a `mu(t)` candidate, not as a standalone trading rule.

## Why This Input

The original no-image baseline is weak. Adding path representation improves the ranking signal.

| signal / model | interpretation | rank corr | Sharpe | cumulative return | hit rate |
|---|---|---:|---:|---:|---:|
| `logistic_cumulative_scale` | ETF raw/cumulative-return baseline, no image, no CNN/LSTM | -0.0072 | 0.0764 | -0.0512 | 0.5556 |
| Equal-weight ETF buy-and-hold | no model, no signal, 7-asset daily equal weight over OOS period | n/a | 0.2086 | 0.2186 | 0.5257 |
| `logistic_image_scale` | image-style scaling but simple logistic model | 0.0392 | 0.3850 | 0.6569 | 0.5903 |
| `ensemble_4family` | rank-corr baseline before image-factor add-on | 0.0674 | 0.5430 | 1.2395 | 0.6042 |
| `ensemble_4family+image_factor_pc1` | recommended final image-factor ranking input | 0.0798 | 0.3155 | 0.5517 | 0.5694 |

Interpretation:

- The no-image raw ETF baseline has almost no cross-sectional prediction ability.
- Image-style representation improves rank correlation even with a simple logistic model.
- The final image-factor ensemble improves ranking quality over `ensemble_4family`.
- The final image-factor input lowers simple top-k Sharpe, so it should be tested inside the ODE optimizer rather than treated as a direct trading strategy.

## Candidate Inputs In This Folder

| input_name | source_model_name | role |
|---|---|---|
| `mu_image_factor_rank` | `ensemble_4family+image_factor_pc1` | Recommended ranking-improved `mu(t)` input |
| `mu_image_factor_balanced` | `ensemble_best+image_factor_pc1` | Secondary image-factor input with smaller Sharpe trade-off |
| `mu_rank_baseline` | `ensemble_4family` | Rank-correlation baseline before image-factor add-on |
| `mu_sharpe_baseline` | `ensemble_best` | Sharpe-stable baseline from previous ensemble sprint |

## Performance Snapshot

| input_name | source_model_name | rank corr | Sharpe | rank corr lift | bootstrap CI |
|---|---|---:|---:|---:|---|
| `mu_image_factor_rank` | `ensemble_4family+image_factor_pc1` | 0.0798 | 0.3155 | 0.0124 | [0.00001, 0.02505] |
| `mu_image_factor_balanced` | `ensemble_best+image_factor_pc1` | 0.0731 | 0.5109 | 0.0125 | [0.00057, 0.02448] |
| `mu_rank_baseline` | `ensemble_4family` | 0.0674 | 0.5430 | 0.0000 | baseline |
| `mu_sharpe_baseline` | `ensemble_best` | 0.0606 | 0.6425 | 0.0000 | baseline |

## Files

| file | use |
|---|---|
| `selected_mu_input.csv` | Recommended input only: `mu_image_factor_rank`. Start here for ODE integration. |
| `final_mu_inputs_long.csv` | Main multi-candidate handoff file. One row per `date-asset-input`. |
| `final_mu_inputs_wide.csv` | Convenience wide table with all candidate inputs on each `date-asset` row. |
| `sigma_wide.csv` | Daily `Sigma(t)` in wide covariance format, aligned to the selected mu dates. |
| `sigma_long.csv` | Daily `Sigma(t)` in long matrix-entry format: one row per `date-asset_i-asset_j`. |
| `returns_for_ode.csv` | Realized daily asset returns aligned to the selected mu/Sigma dates. |
| `input_performance_summary.csv` | Rank-corr, Sharpe, cumulative return, hit rate, turnover, bootstrap CI. |
| `handoff_summary_ko.md` | Korean handoff summary for the team. |
| `manifest.json` | Machine-readable metadata. |
| `sigma_manifest.json` | Machine-readable Sigma metadata. |

## Sigma(t) Inputs

Sigma is already available in this folder.

| file | shape | meaning |
|---|---:|---|
| `sigma_wide.csv` | 2,880 x 29 | One row per date. Contains 7 variances and 21 off-diagonal covariances. |
| `sigma_long.csv` | 141,120 x 5 | One row per matrix entry. Contains full 7 x 7 covariance matrix per date. |
| `returns_for_ode.csv` | 2,880 x 8 | Daily realized returns aligned to `mu(t)` and `Sigma(t)`. |

Sigma details:

- Source: `ode_inputs_cnn/ensemble_best/ode_bundle.csv`
- Scale: daily log-return covariance
- Rolling window: 60 trading days
- Date range: 2014-09-24 to 2026-03-09
- Assets: 7
- Missing values: none after alignment

The Sigma estimates are not model-specific to the image factor. They are rolling historical covariance estimates aligned to the final ODE mu input dates.

## Related Evidence Files

| purpose | path |
|---|---|
| Phase 1 image-factor significance report | `ode_inputs_cnn/image_factor_extension/image_factor_report.md` |
| Phase 1 image-factor significance CSV | `ode_inputs_cnn/image_factor_extension/image_factor_significance.csv` |
| Phase 2 ablation final report | `ode_inputs_cnn/image_factor_phase12_final_report.md` |
| Phase 2 ablation report | `ode_inputs_cnn/image_factor_ablation/image_factor_ablation_report.md` |
| Phase 2 ablation summary | `ode_inputs_cnn/image_factor_ablation/ablation_summary.csv` |
| Phase 2 ablation significance | `ode_inputs_cnn/image_factor_ablation/ablation_significance.csv` |
| Phase 2 ensemble add-on search | `ode_inputs_cnn/image_factor_ablation/ablation_ensemble_search.csv` |
| Original model comparison with no-image baseline | `ode_inputs_cnn/comparison_with_baselines.csv` |

## Column Guide

| column | meaning |
|---|---|
| `date` | Signal date. |
| `asset` | ETF asset name. |
| `input_name` | ODE input candidate name. |
| `source_model_name` | Source signal/model used to create this input. |
| `mu_signal` | Primary ODE input. Date-wise centered cross-sectional rank. |
| `mu_signal_type` | Scaling method. Current value: `cross_sectional_centered_rank`. |
| `mu_raw_score` | Original ensemble/image-factor score before ODE handoff scaling. |
| `mu_rank` | Date-wise percentile rank of `mu_raw_score`. |
| `mu_centered_rank` | `mu_rank` demeaned within each date and input. |
| `mu_zscore` | Date-wise z-score of `mu_raw_score`. |
| `future_return` | Realized 20-day return. Evaluation only. Do not use this as an ODE input. |
| `is_recommended` | Whether the row belongs to the recommended input. |
| `lookback` | Historical lookback window used to create the signal. |
| `horizon` | Prediction horizon. |
| `chart_variant` | Image rendering variant behind the image factor. |

## How To Use In ODE

Recommended first experiment:

1. Read `selected_mu_input.csv`.
2. Pivot by `date` and `asset`.
3. Use `mu_signal` as the expected-return ranking input `mu(t)`.
4. Read `sigma_wide.csv` or `sigma_long.csv` for `Sigma(t)`.
5. Compare against `mu_sharpe_baseline` and `mu_rank_baseline` from `final_mu_inputs_long.csv`.

Minimal pandas sketch:

```python
import pandas as pd

mu = pd.read_csv("ode_inputs_cnn/final_ode_mu_inputs/selected_mu_input.csv", parse_dates=["date"])
mu_wide = mu.pivot(index="date", columns="asset", values="mu_signal")

sigma = pd.read_csv("ode_inputs_cnn/final_ode_mu_inputs/sigma_wide.csv", parse_dates=["date"])
returns = pd.read_csv("ode_inputs_cnn/final_ode_mu_inputs/returns_for_ode.csv", parse_dates=["date"])
```

## Interpretation Caveat

`mu_image_factor_rank` improves cross-sectional ranking quality versus `ensemble_4family`, but it does not improve simple top-k Sharpe. The correct interpretation is:

> The image factor adds useful expected-return ranking information. Whether that information improves final portfolio performance should be tested inside the ODE optimizer with risk control.
