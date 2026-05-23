# Extra ETF Per-Fold Pretraining Result Note

## Status
- Full per-fold run completed: `48/48` folds.
- Extra ETF pretraining used only data before each fold test period.
- Main 7-ETF prediction universe was unchanged.

## Important Comparison Note
- Raw `comparison.csv` is not directly apples-to-apples because baseline coverage is 1320 dates while extra-pretrained coverage is 2880 dates.
- `same_grid_comparison.csv` restricts both models to the shared baseline grid for the fair direct comparison.

## Same-Grid Result
| model | dates | rank corr | Sharpe | cumulative return | hit rate | RMSE | turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_same_grid` | 1320 | 0.0311 | 0.1238 | 0.0390 | 0.5606 | 0.0354 | 0.5538 |
| `extra_pretrained_same_grid` | 1320 | -0.0752 | -0.0277 | -0.0394 | 0.5606 | 0.0358 | 0.5923 |
| `extra_pretrained_full_grid` | 2880 | -0.0581 | 0.1003 | 0.0050 | 0.5556 | 0.0465 | 0.6049 |

## Interpretation
- Same-grid rank corr change: -0.1063.
- Same-grid Sharpe change: -0.1515.
- Under this protocol, extra ETF supervised pretraining did not improve the original 7-ETF CNN signal.
- Treat this as a robustness/negative result, not as the recommended ODE input.
- Keep using the existing image-factor ensemble input unless a different pretraining objective is tested.
