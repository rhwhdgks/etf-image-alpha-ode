# Extra ETF CNN Pretraining Experiment

## Purpose

This experiment checks whether extra ETF chart images improve the CNN feature extractor while keeping the final ODE universe fixed at the original 7 ETF assets.

## Data

- Extra ETF data: `data/extra_market/extra_etf_ohlcv_long.csv`
- Extra raw period used for pretraining: 2005-01-03T00:00:00 to 2025-12-09T00:00:00
- Extra assets: 30
- Extra pretraining samples: per-fold dynamic
- Main 7 ETF samples: 24101
- Lookback / horizon: 60 / 20
- Chart variant: ohlc_ma_volume

## Leakage Control

- Extra ETF pretraining cutoff: `per-fold dynamic`.
- This cutoff is before the first original 7 ETF OOS test date.
- Fine-tuning uses walk-forward train/validation windows only.
- Each fold asserts `max(train/val date) < min(test date)`.

## Training

- Pretrain epochs / patience: 8 / 3
- Fine-tune epochs / patience: 30 / 5
- Batch size: 128
- Device: cpu

## Comparison

| model | rank corr | Sharpe | cumulative return | hit rate | RMSE | n dates |
|---|---:|---:|---:|---:|---:|---:|
| `cnn_2d_residual_small` | 0.0311 | 0.1238 | 0.0390 | 0.5606 | 0.0354 | 1320 |
| `cnn_2d_residual_small_extra_pretrained` | -0.0581 | 0.1003 | 0.0050 | 0.5556 | 0.0465 | 2880 |

## Preliminary Interpretation

Best model by rank corr: `cnn_2d_residual_small`.

If the pretrained CNN improves rank correlation, the extra ETF images are useful for representation learning. If not, the result should be reported as evidence that extra-domain ETF pretraining did not improve the 7-asset OOS signal under this protocol.
