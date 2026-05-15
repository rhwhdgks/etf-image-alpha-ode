# Image Factor Robustness Screen

## Scope

- Chart variant fixed to `ohlc_ma_volume`.
- Model fixed to `cnn_2d_residual_small`.
- Fast screen only: 6 folds, 5 epochs, patience 2.
- Purpose: check whether image-factor information survives simple lookback/horizon changes.

## Summary

| window_tag | lookback | horizon | n_rows | n_dates | n_folds | best_p_factor | best_p_t_stat | best_p_value | best_p_delta_r2 | best_p_daily_rank_corr | best_rank_factor | best_daily_rank_corr | best_rank_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lookback_20_horizon_20 | 20 | 20 | 2520 | 360 | 6 | image_factor_pc2 | -2.97820 | 0.00310 | 0.00404 | -0.04921 | image_factor_pc1 | 0.09881 | 0.07109 |
| lookback_60_horizon_20 | 60 | 20 | 2520 | 360 | 6 | image_factor_pc3 | 4.99704 | 0.00000 | 0.01733 | 0.01121 | image_score | 0.12242 | 0.07346 |
| lookback_60_horizon_60 | 60 | 60 | 2520 | 360 | 6 | image_factor_pc2 | 7.02480 | 0.00000 | 0.03744 | 0.00565 | image_score | 0.08135 | 0.01470 |

## Readout

- `20/20`: best p-value factor is significant, but the strongest rank-corr factor is only marginal. Shorter lookback changes the factor direction.
- `60/20`: significant factors remain, but the fast-screen rank-corr direction is noisy relative to the full 48-fold run.
- `60/60`: significant factor strength is strongest in this fast screen, and `image_score` has the best daily rank correlation.
- Conclusion: image factor information is not only a single-window artifact, but the exact PCA component is window/horizon sensitive.

## Decision

Keep `60/20` as the main ODE handoff setting because it matches the existing CNN/LSTM ensemble and final ablation grid. Use this robustness screen only as supporting evidence, not as a replacement for the full 48-fold result.

## All Significance Rows

| window_tag | factor_name | t_stat | p_value | delta_r2 | daily_rank_correlation | n_obs | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lookback_20_horizon_20 | image_factor_pc2 | -2.97820 | 0.00310 | 0.00404 | -0.04921 | 2520 | 360 |
| lookback_20_horizon_20 | image_factor_pc1 | 1.81024 | 0.07109 | 0.00163 | 0.09881 | 2520 | 360 |
| lookback_20_horizon_20 | image_score | -1.23396 | 0.21803 | 0.00119 | 0.03532 | 2520 | 360 |
| lookback_20_horizon_20 | image_factor_pc3 | -1.02874 | 0.30429 | 0.00070 | -0.01796 | 2520 | 360 |
| lookback_60_horizon_20 | image_factor_pc3 | 4.99704 | 0.00000 | 0.01733 | 0.01121 | 2520 | 360 |
| lookback_60_horizon_20 | image_factor_pc1 | -3.52401 | 0.00048 | 0.01043 | -0.10794 | 2520 | 360 |
| lookback_60_horizon_20 | image_score | -1.79521 | 0.07346 | 0.00309 | 0.12242 | 2520 | 360 |
| lookback_60_horizon_20 | image_factor_pc2 | -0.83345 | 0.40515 | 0.00092 | -0.03234 | 2520 | 360 |
| lookback_60_horizon_60 | image_factor_pc2 | 7.02480 | 0.00000 | 0.03744 | 0.00565 | 2520 | 360 |
| lookback_60_horizon_60 | image_factor_pc3 | -4.49755 | 0.00001 | 0.00861 | -0.03601 | 2520 | 360 |
| lookback_60_horizon_60 | image_score | -2.45143 | 0.01470 | 0.00582 | 0.08135 | 2520 | 360 |
| lookback_60_horizon_60 | image_factor_pc1 | 0.63449 | 0.52616 | 0.00033 | 0.08115 | 2520 | 360 |
