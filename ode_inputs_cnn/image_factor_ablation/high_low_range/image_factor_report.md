# Image Factor Ablation - high_low_range

## Summary
- Chart variant: `high_low_range`
- Rendered MA: `False`
- Rendered volume: `False`
- Extractor: `cnn_2d_residual_small`
- Lookback/horizon: `60/20`
- OOS rows: `20160` across `2880` dates
- Rolling PCA controls: `20160` asset-date rows
- Best factor by p-value: `image_factor_pc3` (t=6.273, p=0.0000, delta R2=0.003972)
- Best ensemble candidate by rank corr: `ensemble_4family` (rank corr=0.0674, Sharpe=0.5430)

## Image Factor Significance
| factor_name | coefficient | std_error_cluster_date | t_stat | p_value | base_r2 | augmented_r2 | delta_r2 | daily_rank_correlation | n_obs | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_factor_pc3 | 0.00177 | 0.00028 | 6.27259 | 0.00000 | 0.00421 | 0.00818 | 0.00397 | 0.04288 | 20160 | 2880 |
| image_factor_pc1 | -0.00031 | 0.00030 | -1.04217 | 0.29742 | 0.00421 | 0.00442 | 0.00021 | 0.01146 | 20160 | 2880 |
| image_score | -0.08466 | 0.08387 | -1.00936 | 0.31289 | 0.00421 | 0.00441 | 0.00020 | 0.00930 | 20160 | 2880 |
| image_factor_pc2 | 0.00018 | 0.00029 | 0.64291 | 0.52033 | 0.00421 | 0.00427 | 0.00006 | 0.00601 | 20160 | 2880 |

## Ensemble Search
| candidate_name | base_ensemble | added_factor | rank_corr | top_k_cumulative_return | top_k_sharpe | top_k_hit_rate | top_bottom_spread_mean | turnover | n_rebalances | factor_vs_base_pearson | factor_vs_base_rank_corr_mean | rank_corr_diff_vs_base | bootstrap_ci_low | bootstrap_ci_high | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ensemble_4family | ensemble_4family |  | 0.06738 | 1.23950 | 0.54300 | 0.60417 | 0.00591 | 0.60140 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_best | ensemble_best |  | 0.06055 | 1.50864 | 0.64255 | 0.58333 | 0.00618 | 0.66434 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_4family+image_score | ensemble_4family | image_score | 0.05164 | 0.78501 | 0.38112 | 0.59722 | 0.00373 | 0.64685 | 144.00000 | 0.08135 | 0.13918 | -0.01574 | -0.02911 | -0.00283 | 2880 |
| ensemble_4family+image_factor_pc1 | ensemble_4family | image_factor_pc1 | 0.04714 | 0.81085 | 0.39340 | 0.58333 | 0.00501 | 0.59091 | 144.00000 | 0.11105 | 0.12228 | -0.02001 | -0.03384 | -0.00666 | 2880 |
| ensemble_best+image_score | ensemble_best | image_score | 0.04361 | 0.86506 | 0.40812 | 0.59028 | 0.00299 | 0.63986 | 144.00000 | 0.06660 | 0.12698 | -0.01694 | -0.03024 | -0.00380 | 2880 |
| ensemble_best+image_factor_pc1 | ensemble_best | image_factor_pc1 | 0.03961 | 0.59494 | 0.33894 | 0.55556 | 0.00342 | 0.63986 | 144.00000 | 0.11184 | 0.11384 | -0.02094 | -0.03435 | -0.00767 | 2880 |

## Interpretation Rules
- If a factor has p < 0.05, call it statistically significant after rolling PCA controls.
- If p includes 0 but delta R2 and ensemble rank-corr lift are positive, frame it as mechanism/trend evidence.
- If ensemble CI includes 0, do not claim strict improvement over the baseline ensemble.

## Output Files
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/high_low_range/image_factor_panel.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/high_low_range/common_pca_controls.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/high_low_range/image_factor_significance.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/high_low_range/ensemble_image_factor_search.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/high_low_range/image_factor_signals.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/high_low_range/ode_mu_candidate_signals.csv`
