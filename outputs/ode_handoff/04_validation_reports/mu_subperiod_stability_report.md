# Mu Input Subperiod Stability

## Purpose

This check tests whether the image-factor mu improvement is concentrated in one market regime or remains directionally useful across subperiods.

## Main Inputs

- `mu_rank_baseline`: 4-family ensemble before image-factor add-on.
- `mu_image_factor_rank`: main full-series MA image-factor input.
- `mu_image_factor_strict_rank`: conservative strict-window MA image-factor input.

## Period Results

| period | input | dates | rank corr | rank corr lift | Sharpe | Sharpe lift | hit rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `pre_covid_2014_2019` | `mu_rank_baseline` | 1327 | 0.0619 | 0.0000 | 0.3298 | 0.0000 | 0.5821 |
| `pre_covid_2014_2019` | `mu_image_factor_rank` | 1327 | 0.0828 | 0.0209 | 0.0837 | -0.2462 | 0.5075 |
| `pre_covid_2014_2019` | `mu_image_factor_strict_rank` | 1327 | 0.1074 | 0.0455 | 0.3380 | 0.0082 | 0.5672 |
| `covid_inflation_2020_2022` | `mu_rank_baseline` | 756 | 0.0754 | 0.0000 | 0.1794 | 0.0000 | 0.5263 |
| `covid_inflation_2020_2022` | `mu_image_factor_rank` | 756 | 0.0606 | -0.0148 | -0.0947 | -0.2740 | 0.4737 |
| `covid_inflation_2020_2022` | `mu_image_factor_strict_rank` | 756 | 0.0072 | -0.0683 | -0.2015 | -0.3809 | 0.5526 |
| `recent_2023_2026` | `mu_rank_baseline` | 797 | 0.0688 | 0.0000 | 0.9491 | 0.0000 | 0.5000 |
| `recent_2023_2026` | `mu_image_factor_rank` | 797 | 0.0929 | 0.0241 | 1.0478 | 0.0987 | 0.5250 |
| `recent_2023_2026` | `mu_image_factor_strict_rank` | 797 | 0.0929 | 0.0240 | 0.9076 | -0.0415 | 0.5250 |
| `full_2014_2026` | `mu_rank_baseline` | 2880 | 0.0674 | 0.0000 | 0.5430 | 0.0000 | 0.6042 |
| `full_2014_2026` | `mu_image_factor_rank` | 2880 | 0.0798 | 0.0124 | 0.3155 | -0.2275 | 0.5694 |
| `full_2014_2026` | `mu_image_factor_strict_rank` | 2880 | 0.0771 | 0.0097 | 0.6053 | 0.0623 | 0.5833 |

## Interpretation

- Main image-factor rank-corr lift is positive in `3/4` reported periods.
- Strict-window image-factor rank-corr lift is positive in `3/4` reported periods.
- If a subperiod has negative lift, report the image factor as regime-sensitive rather than universally dominant.
