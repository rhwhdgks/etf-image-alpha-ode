# Return-Scale Mu Calibration

## Purpose

The original `mu_signal` is a cross-sectional centered-rank signal. This file adds a return-scale candidate using expanding-window calibration.

## Method

- For each input independently, fit pooled OLS `future_return ~ 1 + mu_signal` over all
  asset-date training rows before the signal date.
- Minimum training history: `252` signal dates.
- **Horizon embargo**: `future_return` at date s is a 20-day forward return realised only
  at index(s) + horizon. A training date s is used to calibrate date t only if
  `index(s) + horizon <= index(t)`, so no not-yet-realised forward return enters the fit.
- Output columns: `mu_calibrated_horizon` and `mu_calibrated_daily`.
- This is leakage-controlled but should be treated as a calibration layer, not a new model.
- Reproducing script: `build_mu_calibration.py` (repo root). Set `EMBARGO = False` there
  to reproduce the earlier un-embargoed numbers.

## Summary (horizon-embargo applied)

| input | dates | start | daily rank corr | RMSE | mean horizon mu | std horizon mu |
|---|---:|---|---:|---:|---:|---:|
| `mu_image_factor_strict_rank` | 2609 | 2015-10-21 | 0.0610 | 0.0462 | 0.00176 | 0.00321 |
| `mu_image_factor_rank` | 2609 | 2015-10-21 | 0.0542 | 0.0463 | 0.00176 | 0.00335 |
| `mu_image_factor_balanced` | 2609 | 2015-10-21 | 0.0481 | 0.0463 | 0.00176 | 0.00303 |
| `mu_rank_baseline` | 2609 | 2015-10-21 | 0.0421 | 0.0463 | 0.00176 | 0.00286 |
| `mu_sharpe_baseline` | 2609 | 2015-10-21 | 0.0165 | 0.0463 | 0.00176 | 0.00258 |

For reference, the earlier un-embargoed calibration (first date 2015-09-24, 2628 dates)
reported slightly higher daily rank corr (e.g. strict 0.0639, rank 0.0575). The drop is
expected: removing the not-yet-realised forward returns from the OLS training set removes
a small look-ahead advantage.

## Interpretation

- Use `mu_signal` as the robust ranking input.
- Use `mu_calibrated_horizon` only if the ODE implementation requires return-scale expected returns.
- Calibration reduces the unit mismatch with mean-variance notation, but adds estimation noise.
- The relative ordering of inputs by calibrated rank corr is unchanged by the embargo fix:
  image-factor inputs still rank above the baselines.
