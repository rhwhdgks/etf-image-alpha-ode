# Jiang-Style Image Factor Extension

## Summary
- Extractor: `cnn_2d_residual_small`
- Lookback/horizon: `60/20`
- Strict window MA: `True`
- OOS rows: `20160` across `2880` dates
- Rolling PCA controls: `20160` asset-date rows
- Best factor by p-value: `image_factor_pc3` (t=1.557, p=0.1196, delta R2=0.000407)
- Best ensemble candidate by rank corr: `ensemble_4family+image_factor_pc1` (rank corr=0.0771, Sharpe=0.6053)

## Image Factor Significance
| factor_name | coefficient | std_error_cluster_date | t_stat | p_value | base_r2 | augmented_r2 | delta_r2 | daily_rank_correlation | n_obs | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_factor_pc3 | 0.00060 | 0.00038 | 1.55707 | 0.11956 | 0.00421 | 0.00461 | 0.00041 | -0.02791 | 20160 | 2880 |
| image_score | -0.12480 | 0.09117 | -1.36894 | 0.17113 | 0.00421 | 0.00445 | 0.00025 | 0.00867 | 20160 | 2880 |
| image_factor_pc2 | -0.00019 | 0.00030 | -0.64047 | 0.52192 | 0.00421 | 0.00427 | 0.00006 | 0.04839 | 20160 | 2880 |
| image_factor_pc1 | 0.00003 | 0.00020 | 0.16213 | 0.87122 | 0.00421 | 0.00421 | 0.00000 | 0.06872 | 20160 | 2880 |

## Ensemble Search
| candidate_name | base_ensemble | added_factor | rank_corr | top_k_cumulative_return | top_k_sharpe | top_k_hit_rate | top_bottom_spread_mean | turnover | n_rebalances | factor_vs_base_pearson | factor_vs_base_rank_corr_mean | rank_corr_diff_vs_base | bootstrap_ci_low | bootstrap_ci_high | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ensemble_4family+image_factor_pc1 | ensemble_4family | image_factor_pc1 | 0.07708 | 1.43240 | 0.60529 | 0.58333 | 0.00950 | 0.55245 | 144.00000 | 0.23194 | 0.28103 | 0.00969 | -0.00290 | 0.02195 | 2880 |
| ensemble_best+image_factor_pc1 | ensemble_best | image_factor_pc1 | 0.07525 | 1.43882 | 0.60909 | 0.56250 | 0.00795 | 0.58042 | 144.00000 | 0.23045 | 0.27174 | 0.01469 | 0.00209 | 0.02712 | 2880 |
| ensemble_4family | ensemble_4family |  | 0.06738 | 1.23950 | 0.54300 | 0.60417 | 0.00591 | 0.60140 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_best | ensemble_best |  | 0.06055 | 1.50864 | 0.64255 | 0.58333 | 0.00618 | 0.66434 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_4family+image_score | ensemble_4family | image_score | 0.04991 | 1.72148 | 0.67405 | 0.59028 | 0.00625 | 0.59091 | 144.00000 | 0.14066 | 0.28850 | -0.01747 | -0.02942 | -0.00541 | 2880 |
| ensemble_best+image_score | ensemble_best | image_score | 0.04756 | 1.39368 | 0.55840 | 0.56250 | 0.00547 | 0.58392 | 144.00000 | 0.12900 | 0.27515 | -0.01300 | -0.02473 | -0.00123 | 2880 |

## Interpretation Rules
- If a factor has p < 0.05, call it statistically significant after rolling PCA controls.
- If p includes 0 but delta R2 and ensemble rank-corr lift are positive, frame it as mechanism/trend evidence.
- If ensemble CI includes 0, do not claim strict improvement over the baseline ensemble.

## Output Files
- `ode_inputs_cnn/image_factor_strict_window_ma/image_factor_panel.csv`
- `ode_inputs_cnn/image_factor_strict_window_ma/common_pca_controls.csv`
- `ode_inputs_cnn/image_factor_strict_window_ma/image_factor_significance.csv`
- `ode_inputs_cnn/image_factor_strict_window_ma/ensemble_image_factor_search.csv`
- `ode_inputs_cnn/image_factor_strict_window_ma/image_factor_signals.csv`
- `ode_inputs_cnn/image_factor_strict_window_ma/ode_mu_candidate_signals.csv`
