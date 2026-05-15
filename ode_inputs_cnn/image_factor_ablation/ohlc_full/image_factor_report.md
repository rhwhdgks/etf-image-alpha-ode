# Image Factor Ablation - ohlc_full

## Summary
- Chart variant: `ohlc_full`
- Rendered MA: `False`
- Rendered volume: `False`
- Extractor: `cnn_2d_residual_small`
- Lookback/horizon: `60/20`
- OOS rows: `20160` across `2880` dates
- Rolling PCA controls: `20160` asset-date rows
- Best factor by p-value: `image_factor_pc2` (t=2.848, p=0.0044, delta R2=0.001391)
- Best ensemble candidate by rank corr: `ensemble_4family` (rank corr=0.0674, Sharpe=0.5430)

## Image Factor Significance
| factor_name | coefficient | std_error_cluster_date | t_stat | p_value | base_r2 | augmented_r2 | delta_r2 | daily_rank_correlation | n_obs | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_factor_pc2 | 0.00091 | 0.00032 | 2.84807 | 0.00443 | 0.00421 | 0.00560 | 0.00139 | 0.02016 | 20160 | 2880 |
| image_factor_pc3 | -0.00116 | 0.00050 | -2.30814 | 0.02106 | 0.00421 | 0.00577 | 0.00157 | 0.00988 | 20160 | 2880 |
| image_score | 0.18225 | 0.08129 | 2.24206 | 0.02503 | 0.00421 | 0.00511 | 0.00090 | 0.02036 | 20160 | 2880 |
| image_factor_pc1 | -0.00007 | 0.00032 | -0.23447 | 0.81463 | 0.00421 | 0.00422 | 0.00001 | 0.01731 | 20160 | 2880 |

## Ensemble Search
| candidate_name | base_ensemble | added_factor | rank_corr | top_k_cumulative_return | top_k_sharpe | top_k_hit_rate | top_bottom_spread_mean | turnover | n_rebalances | factor_vs_base_pearson | factor_vs_base_rank_corr_mean | rank_corr_diff_vs_base | bootstrap_ci_low | bootstrap_ci_high | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ensemble_4family | ensemble_4family |  | 0.06738 | 1.23950 | 0.54300 | 0.60417 | 0.00591 | 0.60140 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_best | ensemble_best |  | 0.06055 | 1.50864 | 0.64255 | 0.58333 | 0.00618 | 0.66434 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_4family+image_score | ensemble_4family | image_score | 0.05508 | 0.84590 | 0.42418 | 0.64583 | 0.00489 | 0.63986 | 144.00000 | 0.12009 | 0.14188 | -0.01230 | -0.02517 | 0.00062 | 2880 |
| ensemble_4family+image_factor_pc1 | ensemble_4family | image_factor_pc1 | 0.05265 | 1.67592 | 0.64740 | 0.60417 | 0.00473 | 0.61189 | 144.00000 | 0.13966 | 0.15561 | -0.01473 | -0.02799 | -0.00160 | 2880 |
| ensemble_best+image_score | ensemble_best | image_score | 0.05157 | 1.14391 | 0.49784 | 0.64583 | 0.00551 | 0.66434 | 144.00000 | 0.11840 | 0.14530 | -0.00915 | -0.02171 | 0.00349 | 2880 |
| ensemble_best+image_factor_pc1 | ensemble_best | image_factor_pc1 | 0.04734 | 1.75934 | 0.68845 | 0.60417 | 0.00589 | 0.61888 | 144.00000 | 0.14982 | 0.16119 | -0.01321 | -0.02640 | -0.00004 | 2880 |

## Interpretation Rules
- If a factor has p < 0.05, call it statistically significant after rolling PCA controls.
- If p includes 0 but delta R2 and ensemble rank-corr lift are positive, frame it as mechanism/trend evidence.
- If ensemble CI includes 0, do not claim strict improvement over the baseline ensemble.

## Output Files
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_full/image_factor_panel.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_full/common_pca_controls.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_full/image_factor_significance.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_full/ensemble_image_factor_search.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_full/image_factor_signals.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_full/ode_mu_candidate_signals.csv`
