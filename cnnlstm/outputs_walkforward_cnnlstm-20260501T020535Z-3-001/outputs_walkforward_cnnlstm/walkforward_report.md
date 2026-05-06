# CNNLSTM Walk-Forward OOS Evaluation

## Configuration
- lookback=60, horizon=20
- label_mode=regression, target=future_return
- wf_min_train_days=500  wf_val_days=60  wf_test_days=60
- cnn_epochs=8, cnn_repeats=1
- models: cnn_lstm_image_scale, cnn_lstm_cumulative_scale
- total OOS predictions: 40320

## Model Comparison (Aggregated OOS)
| rmse | mae | target_rank_correlation | future_return_rank_correlation | top_k_cumulative_return | top_k_sharpe | top_k_hit_rate | top_bottom_spread_mean | turnover | n_rebalances | model_name | n_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0460 | 0.0303 | 0.0448 | 0.0448 | 0.8998 | 0.4337 | 0.5764 | 0.0029 | 0.5210 | 144.0000 | cnn_lstm_image_scale | 48 |
| 0.0459 | 0.0300 | -0.0125 | -0.0125 | 0.5788 | 0.3610 | 0.5625 | 0.0002 | 0.6608 | 144.0000 | cnn_lstm_cumulative_scale | 48 |

## Ensemble Handoff
- Add this output directory to `build_extended_ensemble.py` or a copied ensemble script.
- Main file to merge: `walkforward_predictions.csv`.
- Required model names are stored in the `model_name` column.
