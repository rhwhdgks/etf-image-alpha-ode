# Return-Scale Mu Calibration

## Purpose

The original `mu_signal` is a cross-sectional centered-rank signal. This file adds a return-scale candidate using expanding-window calibration.

## Method

- For each input independently, fit `future_return ~ 1 + mu_signal` using only dates before the signal date.
- Minimum training history: `252` signal dates.
- Output columns: `mu_calibrated_horizon` and `mu_calibrated_daily`.
- This is leakage-controlled but should be treated as a calibration layer, not as a new model.

## Summary

| input | dates | start | daily rank corr | RMSE | mean horizon mu | std horizon mu |
|---|---:|---|---:|---:|---:|---:|
| `mu_image_factor_strict_rank` | 2628 | 2015-09-24 | 0.0639 | 0.0460 | 0.00177 | 0.00321 |
| `mu_image_factor_rank` | 2628 | 2015-09-24 | 0.0575 | 0.0461 | 0.00177 | 0.00335 |
| `mu_image_factor_balanced` | 2628 | 2015-09-24 | 0.0517 | 0.0461 | 0.00177 | 0.00302 |
| `mu_rank_baseline` | 2628 | 2015-09-24 | 0.0440 | 0.0461 | 0.00177 | 0.00286 |
| `mu_sharpe_baseline` | 2628 | 2015-09-24 | 0.0270 | 0.0461 | 0.00177 | 0.00257 |

## Interpretation

- Use `mu_signal` as the robust ranking input.
- Use `mu_calibrated_horizon` only if the ODE implementation requires return-scale expected returns.
- Calibration reduces the unit mismatch with mean-variance notation, but adds estimation noise.
