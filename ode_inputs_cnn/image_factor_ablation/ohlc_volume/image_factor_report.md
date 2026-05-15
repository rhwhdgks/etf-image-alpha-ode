# Image Factor Ablation - ohlc_volume

## Summary
- Chart variant: `ohlc_volume`
- Rendered MA: `False`
- Rendered volume: `True`
- Extractor: `cnn_2d_residual_small`
- Lookback/horizon: `60/20`
- OOS rows: `20160` across `2880` dates
- Rolling PCA controls: `20160` asset-date rows
- Best factor by p-value: `image_factor_pc1` (t=-6.537, p=0.0000, delta R2=0.003877)
- Best ensemble candidate by rank corr: `ensemble_4family` (rank corr=0.0674, Sharpe=0.5430)

## Image Factor Significance
| factor_name | coefficient | std_error_cluster_date | t_stat | p_value | base_r2 | augmented_r2 | delta_r2 | daily_rank_correlation | n_obs | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_factor_pc1 | -0.00131 | 0.00020 | -6.53683 | 0.00000 | 0.00421 | 0.00808 | 0.00388 | 0.01984 | 20160 | 2880 |
| image_score | -0.25841 | 0.06718 | -3.84653 | 0.00012 | 0.00421 | 0.00589 | 0.00169 | -0.01391 | 20160 | 2880 |
| image_factor_pc3 | 0.00083 | 0.00034 | 2.45203 | 0.01426 | 0.00421 | 0.00494 | 0.00073 | 0.00278 | 20160 | 2880 |
| image_factor_pc2 | -0.00049 | 0.00029 | -1.64823 | 0.09941 | 0.00421 | 0.00456 | 0.00035 | 0.03045 | 20160 | 2880 |

## Ensemble Search
| candidate_name | base_ensemble | added_factor | rank_corr | top_k_cumulative_return | top_k_sharpe | top_k_hit_rate | top_bottom_spread_mean | turnover | n_rebalances | factor_vs_base_pearson | factor_vs_base_rank_corr_mean | rank_corr_diff_vs_base | bootstrap_ci_low | bootstrap_ci_high | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ensemble_4family | ensemble_4family |  | 0.06738 | 1.23950 | 0.54300 | 0.60417 | 0.00591 | 0.60140 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_best | ensemble_best |  | 0.06055 | 1.50864 | 0.64255 | 0.58333 | 0.00618 | 0.66434 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_4family+image_factor_pc1 | ensemble_4family | image_factor_pc1 | 0.05799 | 1.03702 | 0.50597 | 0.62500 | 0.00415 | 0.55594 | 144.00000 | 0.26668 | 0.27783 | -0.00939 | -0.02094 | 0.00232 | 2880 |
| ensemble_best+image_factor_pc1 | ensemble_best | image_factor_pc1 | 0.05528 | 0.97642 | 0.48819 | 0.59722 | 0.00357 | 0.59091 | 144.00000 | 0.22652 | 0.22288 | -0.00542 | -0.01719 | 0.00665 | 2880 |
| ensemble_4family+image_score | ensemble_4family | image_score | 0.03735 | 0.38911 | 0.26244 | 0.59028 | 0.00091 | 0.59091 | 144.00000 | 0.13945 | 0.25816 | -0.03003 | -0.04193 | -0.01783 | 2880 |
| ensemble_best+image_score | ensemble_best | image_score | 0.03423 | 0.72147 | 0.37760 | 0.58333 | 0.00131 | 0.63287 | 144.00000 | 0.13139 | 0.23616 | -0.02632 | -0.03834 | -0.01431 | 2880 |

## Interpretation Rules
- If a factor has p < 0.05, call it statistically significant after rolling PCA controls.
- If p includes 0 but delta R2 and ensemble rank-corr lift are positive, frame it as mechanism/trend evidence.
- If ensemble CI includes 0, do not claim strict improvement over the baseline ensemble.

## Output Files
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_volume/image_factor_panel.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_volume/common_pca_controls.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_volume/image_factor_significance.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_volume/ensemble_image_factor_search.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_volume/image_factor_signals.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_volume/ode_mu_candidate_signals.csv`
