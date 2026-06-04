# Strict Window MA Robustness Result

## Question
Does the MA-based image factor survive when the moving-average line is computed only from prices inside each 60-day rendered window?

## Setup
- Original MA: MA was computed on the full past time series before slicing the 60-day image window.
- Strict window MA: MA is recomputed inside each 60-day window only, while keeping the same OOS date-asset grid as the original experiment.
- Model/folds/target remain unchanged: `cnn_2d_residual_small`, 48 walk-forward folds, `60/20`, future return regression.

## Main Comparison
| spec | factor/candidate | rank corr | Sharpe | rank corr lift | bootstrap CI | p-value | delta R2 |
|---|---|---:|---:|---:|---|---:|---:|
| `original_full_series_ma` | `ensemble_4family+image_factor_pc1` | 0.0798 | 0.3155 | 0.0124 | [0.00001, 0.02505] | 0.0004 | 0.001586 |
| `strict_window_ma` | `ensemble_4family+image_factor_pc1` | 0.0771 | 0.6053 | 0.0097 | [-0.00290, 0.02195] | 0.8712 | 0.000002 |

## Interpretation
- Strict window MA weakens standalone PCA-control significance: `image_factor_pc1` is not statistically significant by the panel OLS p-value.
- However, strict `image_factor_pc1` still improves ensemble ranking: `ensemble_4family` rank corr rises from 0.0674 to 0.0771.
- The strict CI for `ensemble_4family+image_factor_pc1` includes zero, so phrase it as robustness/trend evidence, not strict statistical proof.
- The stronger original result may partly reflect the extra historical smoothing embedded in full-series MA, but the ranking contribution does not disappear when MA is restricted to the image window.
- For final reporting, keep the original full chart as the main signal and use strict window MA as a conservative robustness check.
