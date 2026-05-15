# Image Factor Ablation - close_only

## Summary
- Chart variant: `close_only`
- Rendered MA: `False`
- Rendered volume: `False`
- Extractor: `cnn_2d_residual_small`
- Lookback/horizon: `60/20`
- OOS rows: `20160` across `2880` dates
- Rolling PCA controls: `20160` asset-date rows
- Best factor by p-value: `image_factor_pc2` (t=6.859, p=0.0000, delta R2=0.004431)
- Best ensemble candidate by rank corr: `ensemble_4family` (rank corr=0.0674, Sharpe=0.5430)

## Image Factor Significance
| factor_name | coefficient | std_error_cluster_date | t_stat | p_value | base_r2 | augmented_r2 | delta_r2 | daily_rank_correlation | n_obs | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_factor_pc2 | 0.00150 | 0.00022 | 6.85920 | 0.00000 | 0.00421 | 0.00864 | 0.00443 | 0.01957 | 20160 | 2880 |
| image_factor_pc1 | -0.00128 | 0.00026 | -4.81914 | 0.00000 | 0.00421 | 0.00790 | 0.00370 | -0.00574 | 20160 | 2880 |
| image_score | 0.23095 | 0.05574 | 4.14306 | 0.00004 | 0.00421 | 0.00614 | 0.00194 | -0.00062 | 20160 | 2880 |
| image_factor_pc3 | -0.00052 | 0.00030 | -1.72279 | 0.08503 | 0.00421 | 0.00450 | 0.00029 | -0.00362 | 20160 | 2880 |

## Ensemble Search
| candidate_name | base_ensemble | added_factor | rank_corr | top_k_cumulative_return | top_k_sharpe | top_k_hit_rate | top_bottom_spread_mean | turnover | n_rebalances | factor_vs_base_pearson | factor_vs_base_rank_corr_mean | rank_corr_diff_vs_base | bootstrap_ci_low | bootstrap_ci_high | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ensemble_4family | ensemble_4family |  | 0.06738 | 1.23950 | 0.54300 | 0.60417 | 0.00591 | 0.60140 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_best | ensemble_best |  | 0.06055 | 1.50864 | 0.64255 | 0.58333 | 0.00618 | 0.66434 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_4family+image_score | ensemble_4family | image_score | 0.04515 | 0.97091 | 0.43517 | 0.63889 | 0.00430 | 0.62937 | 144.00000 | 0.07296 | 0.14540 | -0.02223 | -0.03490 | -0.00932 | 2880 |
| ensemble_4family+image_factor_pc1 | ensemble_4family | image_factor_pc1 | 0.04177 | 1.40794 | 0.59397 | 0.58333 | 0.00675 | 0.60490 | 144.00000 | 0.08910 | 0.10143 | -0.02561 | -0.03931 | -0.01200 | 2880 |
| ensemble_best+image_score | ensemble_best | image_score | 0.03904 | 1.04620 | 0.47095 | 0.59028 | 0.00515 | 0.62238 | 144.00000 | 0.07452 | 0.13849 | -0.02132 | -0.03408 | -0.00874 | 2880 |
| ensemble_best+image_factor_pc1 | ensemble_best | image_factor_pc1 | 0.03630 | 0.78343 | 0.38714 | 0.59028 | 0.00495 | 0.63636 | 144.00000 | 0.06920 | 0.08789 | -0.02426 | -0.03866 | -0.01024 | 2880 |

## Interpretation Rules
- If a factor has p < 0.05, call it statistically significant after rolling PCA controls.
- If p includes 0 but delta R2 and ensemble rank-corr lift are positive, frame it as mechanism/trend evidence.
- If ensemble CI includes 0, do not claim strict improvement over the baseline ensemble.

## Output Files
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/close_only/image_factor_panel.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/close_only/common_pca_controls.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/close_only/image_factor_significance.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/close_only/ensemble_image_factor_search.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/close_only/image_factor_signals.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/close_only/ode_mu_candidate_signals.csv`
