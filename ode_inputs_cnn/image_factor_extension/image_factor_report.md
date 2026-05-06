# Jiang-Style Image Factor Extension

## Summary
- Extractor: `cnn_2d_residual_small`
- Lookback/horizon: `60/20`
- OOS rows: `20160` across `2880` dates
- Rolling PCA controls: `20160` asset-date rows
- Best factor by p-value: `image_score` (t=4.612, p=0.0000, delta R2=0.002257)
- Best ensemble candidate by rank corr: `ensemble_4family` (rank corr=0.0674, Sharpe=0.5430)

## Image Factor Significance
| factor_name | coefficient | std_error_cluster_date | t_stat | p_value | base_r2 | augmented_r2 | delta_r2 | daily_rank_correlation | n_obs | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_score | 0.22594 | 0.04899 | 4.61167 | 0.00000 | 0.00421 | 0.00646 | 0.00226 | 0.01214 | 20160 | 2880 |
| image_factor_pc3 | -0.00068 | 0.00031 | -2.22444 | 0.02620 | 0.00421 | 0.00469 | 0.00048 | -0.01364 | 20160 | 2880 |
| image_factor_pc1 | 0.00027 | 0.00021 | 1.33770 | 0.18110 | 0.00421 | 0.00441 | 0.00020 | 0.04064 | 20160 | 2880 |
| image_factor_pc2 | -0.00004 | 0.00027 | -0.13506 | 0.89258 | 0.00421 | 0.00421 | 0.00000 | 0.01301 | 20160 | 2880 |

## Ensemble Search
| candidate_name | base_ensemble | added_factor | rank_corr | top_k_cumulative_return | top_k_sharpe | top_k_hit_rate | top_bottom_spread_mean | turnover | n_rebalances | factor_vs_base_pearson | factor_vs_base_rank_corr_mean | rank_corr_diff_vs_base | bootstrap_ci_low | bootstrap_ci_high | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ensemble_4family | ensemble_4family | nan | 0.06738 | 1.23950 | 0.54300 | 0.60417 | 0.00591 | 0.60140 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_4family+image_factor_pc1 | ensemble_4family | image_factor_pc1 | 0.06722 | 0.58367 | 0.32451 | 0.59722 | 0.00290 | 0.53497 | 144.00000 | 0.30861 | 0.31114 | -0.00016 | -0.01058 | 0.01037 | 2880 |
| ensemble_best+image_factor_pc1 | ensemble_best | image_factor_pc1 | 0.06547 | 0.92690 | 0.44224 | 0.59028 | 0.00507 | 0.56993 | 144.00000 | 0.30408 | 0.32435 | 0.00492 | -0.00533 | 0.01538 | 2880 |
| ensemble_best | ensemble_best | nan | 0.06055 | 1.50864 | 0.64255 | 0.58333 | 0.00618 | 0.66434 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_4family+image_score | ensemble_4family | image_score | 0.05255 | 0.36359 | 0.25301 | 0.59722 | 0.00394 | 0.59790 | 144.00000 | 0.18488 | 0.34293 | -0.01483 | -0.02536 | -0.00421 | 2880 |
| ensemble_best+image_score | ensemble_best | image_score | 0.04864 | 0.74651 | 0.39173 | 0.59028 | 0.00441 | 0.65035 | 144.00000 | 0.20871 | 0.34505 | -0.01192 | -0.02258 | -0.00158 | 2880 |

## Interpretation Rules
- If a factor has p < 0.05, call it statistically significant after rolling PCA controls.
- If p includes 0 but delta R2 and ensemble rank-corr lift are positive, frame it as mechanism/trend evidence.
- If ensemble CI includes 0, do not claim strict improvement over the baseline ensemble.

## Output Files
- `ode_inputs_cnn/image_factor_extension/image_factor_panel.csv`
- `ode_inputs_cnn/image_factor_extension/common_pca_controls.csv`
- `ode_inputs_cnn/image_factor_extension/image_factor_significance.csv`
- `ode_inputs_cnn/image_factor_extension/ensemble_image_factor_search.csv`
- `ode_inputs_cnn/image_factor_extension/image_factor_signals.csv`
- `ode_inputs_cnn/image_factor_extension/ode_mu_candidate_signals.csv`
