# Image Factor Ablation - ohlc_ma

## Summary
- Chart variant: `ohlc_ma`
- Rendered MA: `True`
- Rendered volume: `False`
- Extractor: `cnn_2d_residual_small`
- Lookback/horizon: `60/20`
- OOS rows: `20160` across `2880` dates
- Rolling PCA controls: `20160` asset-date rows
- Best factor by p-value: `image_score` (t=7.232, p=0.0000, delta R2=0.008839)
- Best ensemble candidate by rank corr: `ensemble_4family` (rank corr=0.0674, Sharpe=0.5430)

## Image Factor Significance
| factor_name | coefficient | std_error_cluster_date | t_stat | p_value | base_r2 | augmented_r2 | delta_r2 | daily_rank_correlation | n_obs | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_score | 0.51261 | 0.07088 | 7.23197 | 0.00000 | 0.00421 | 0.01305 | 0.00884 | 0.01837 | 20160 | 2880 |
| image_factor_pc2 | 0.00161 | 0.00031 | 5.14581 | 0.00000 | 0.00421 | 0.00862 | 0.00441 | 0.00651 | 20160 | 2880 |
| image_factor_pc1 | 0.00069 | 0.00030 | 2.31334 | 0.02077 | 0.00421 | 0.00548 | 0.00127 | 0.02490 | 20160 | 2880 |
| image_factor_pc3 | 0.00057 | 0.00029 | 1.99447 | 0.04620 | 0.00421 | 0.00459 | 0.00038 | 0.02648 | 20160 | 2880 |

## Ensemble Search
| candidate_name | base_ensemble | added_factor | rank_corr | top_k_cumulative_return | top_k_sharpe | top_k_hit_rate | top_bottom_spread_mean | turnover | n_rebalances | factor_vs_base_pearson | factor_vs_base_rank_corr_mean | rank_corr_diff_vs_base | bootstrap_ci_low | bootstrap_ci_high | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ensemble_4family | ensemble_4family |  | 0.06738 | 1.23950 | 0.54300 | 0.60417 | 0.00591 | 0.60140 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_4family+image_factor_pc1 | ensemble_4family | image_factor_pc1 | 0.06245 | 0.69755 | 0.35701 | 0.57639 | 0.00489 | 0.54895 | 144.00000 | 0.22553 | 0.23270 | -0.00493 | -0.01742 | 0.00722 | 2880 |
| ensemble_best | ensemble_best |  | 0.06055 | 1.50864 | 0.64255 | 0.58333 | 0.00618 | 0.66434 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_best+image_factor_pc1 | ensemble_best | image_factor_pc1 | 0.05733 | 1.20544 | 0.50938 | 0.55556 | 0.00777 | 0.57692 | 144.00000 | 0.23180 | 0.24575 | -0.00322 | -0.01539 | 0.00849 | 2880 |
| ensemble_4family+image_score | ensemble_4family | image_score | 0.05645 | 0.80385 | 0.39948 | 0.58333 | 0.00659 | 0.60839 | 144.00000 | 0.15793 | 0.20536 | -0.01093 | -0.02300 | 0.00109 | 2880 |
| ensemble_best+image_score | ensemble_best | image_score | 0.05101 | 0.93171 | 0.44044 | 0.55556 | 0.00632 | 0.64336 | 144.00000 | 0.17880 | 0.24159 | -0.00954 | -0.02152 | 0.00229 | 2880 |

## Interpretation Rules
- If a factor has p < 0.05, call it statistically significant after rolling PCA controls.
- If p includes 0 but delta R2 and ensemble rank-corr lift are positive, frame it as mechanism/trend evidence.
- If ensemble CI includes 0, do not claim strict improvement over the baseline ensemble.

## Output Files
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_ma/image_factor_panel.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_ma/common_pca_controls.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_ma/image_factor_significance.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_ma/ensemble_image_factor_search.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_ma/image_factor_signals.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_ma/ode_mu_candidate_signals.csv`
