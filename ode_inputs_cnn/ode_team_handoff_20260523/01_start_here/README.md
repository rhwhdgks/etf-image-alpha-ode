# ODE Team Input README

## Purpose

This handoff package collects the final `mu(t)` candidate inputs, `Sigma(t)`, and realized returns from the image-factor sprint.
Optimization and ODE solver integration are intentionally excluded here.

## Recommended Input

- Recommended input: `mu_image_factor_rank`
- Source model: `ensemble_4family+image_factor_pc1`
- Signal type: `cross_sectional_centered_rank`
- Lookback / horizon: `60/20`
- Chart variant: `ohlc_ma_volume`
- OOS rows per input: `20160`
- OOS dates: `2880`

The recommended input is designed as an expected-return ranking signal, not as a standalone trading rule.

## Performance Snapshot

| input_name | source_model_name | strict_MA | rank_corr | Sharpe | rank_corr_lift | bootstrap_CI |
|---|---|---|---:|---:|---:|---|
| `mu_image_factor_rank` | `ensemble_4family+image_factor_pc1` | `False` | 0.0798 | 0.3155 | 0.0124 | [0.00001, 0.02505] |
| `mu_image_factor_strict_rank` | `ensemble_4family+image_factor_pc1` | `True` | 0.0771 | 0.6053 | 0.0097 | [-0.00290, 0.02195] |
| `mu_image_factor_balanced` | `ensemble_best+image_factor_pc1` | `False` | 0.0731 | 0.5109 | 0.0125 | [0.00057, 0.02448] |
| `mu_rank_baseline` | `ensemble_4family` | `False` | 0.0674 | 0.5430 | 0.0000 | [0.00000, 0.00000] |
| `mu_sharpe_baseline` | `ensemble_best` | `False` | 0.0606 | 0.6425 | 0.0000 | [0.00000, 0.00000] |

## Files

| file | use |
|---|---|
| `02_mu_inputs/final_mu_inputs_long.csv` | Main handoff file. One row per `date-asset-input`. |
| `02_mu_inputs/selected_mu_input.csv` | Recommended input only: `mu_image_factor_rank`. |
| `02_mu_inputs/final_mu_inputs_wide.csv` | Convenience wide table with all candidate inputs. |
| `02_mu_inputs/final_mu_inputs_calibrated.csv` | Expanding-window return-scale calibration of each `mu_signal`. |
| `02_mu_inputs/input_performance_summary.csv` | Rank-corr, Sharpe, bootstrap CI summary. |
| `02_mu_inputs/main_vs_strict_mu_comparison.csv` | Direct comparison of baseline, main image factor, and strict-window MA image factor. |
| `02_mu_inputs/mu_calibration_summary.csv` | Calibration coverage, RMSE, and calibrated rank-corr summary. |
| `03_sigma_returns/sigma_wide.csv` | Daily rolling covariance matrix in wide format. |
| `03_sigma_returns/sigma_long.csv` | Daily full covariance matrix in long format. |
| `03_sigma_returns/returns_for_ode.csv` | Realized daily returns aligned to the signal dates. |
| `01_start_here/manifest.json` | Machine-readable package metadata. |

## Column Guide

- `mu_signal`: primary ODE input, equal to date-wise centered cross-sectional rank.
- `mu_raw_score`: original ensemble/image-factor score.
- `mu_rank`: date-wise percentile rank of `mu_raw_score`.
- `mu_centered_rank`: `mu_rank` demeaned within each date and input.
- `mu_zscore`: date-wise z-score of `mu_raw_score`.
- `future_return`: realized 20-day target return, included for evaluation only. Do not use as an input.
- `mu_calibrated_horizon`: expanding-window calibrated expected 20-day return; available in `final_mu_inputs_calibrated.csv`.
- `mu_calibrated_daily`: calibrated expected daily return, equal to horizon value divided by 20.

## Interpretation

`mu_image_factor_rank` improves ranking quality versus `ensemble_4family` but lowers simple top-k Sharpe. Use it as an ODE expected-return ranking input and compare against `mu_sharpe_baseline` in the optimizer. `mu_image_factor_strict_rank` is included as a conservative robustness candidate where the MA line uses only within-window prices.

If the ODE implementation requires return-scale expected returns rather than ranks, use `final_mu_inputs_calibrated.csv`. Treat calibrated values as a secondary input because calibration reduces unit mismatch but adds estimation noise.

## Minimal Loading Example

```python
import pandas as pd

root = "ode_inputs_cnn/ode_team_handoff_20260523"

mu = pd.read_csv(f"{root}/02_mu_inputs/selected_mu_input.csv", parse_dates=["date"])
mu_wide = mu.pivot(index="date", columns="asset", values="mu_signal")

sigma = pd.read_csv(f"{root}/03_sigma_returns/sigma_wide.csv", parse_dates=["date"])
returns = pd.read_csv(f"{root}/03_sigma_returns/returns_for_ode.csv", parse_dates=["date"])
```

`future_return` is included only for validation. Do not use it as an optimizer input.
