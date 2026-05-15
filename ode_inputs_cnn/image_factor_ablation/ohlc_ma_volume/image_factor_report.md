# Image Factor Ablation - ohlc_ma_volume

## Summary
- Chart variant: `ohlc_ma_volume`
- Rendered MA: `True`
- Rendered volume: `True`
- Extractor: `cnn_2d_residual_small`
- Lookback/horizon: `60/20`
- OOS rows: `20160` across `2880` dates
- Rolling PCA controls: `20160` asset-date rows
- Best factor by p-value: `image_factor_pc3` (t=4.259, p=0.0000, delta R2=0.003180)
- Best ensemble candidate by rank corr: `ensemble_4family+image_factor_pc1` (rank corr=0.0798, Sharpe=0.3155)

## Image Factor Significance
| factor_name | coefficient | std_error_cluster_date | t_stat | p_value | base_r2 | augmented_r2 | delta_r2 | daily_rank_correlation | n_obs | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_factor_pc3 | 0.00177 | 0.00041 | 4.25927 | 0.00002 | 0.00421 | 0.00739 | 0.00318 | -0.03635 | 20160 | 2880 |
| image_factor_pc2 | -0.00094 | 0.00025 | -3.81771 | 0.00014 | 0.00421 | 0.00556 | 0.00135 | -0.02884 | 20160 | 2880 |
| image_factor_pc1 | 0.00082 | 0.00023 | 3.55110 | 0.00039 | 0.00421 | 0.00579 | 0.00159 | 0.05921 | 20160 | 2880 |
| image_score | 0.15042 | 0.04506 | 3.33847 | 0.00085 | 0.00421 | 0.00538 | 0.00117 | 0.02851 | 20160 | 2880 |

## Ensemble Search
| candidate_name | base_ensemble | added_factor | rank_corr | top_k_cumulative_return | top_k_sharpe | top_k_hit_rate | top_bottom_spread_mean | turnover | n_rebalances | factor_vs_base_pearson | factor_vs_base_rank_corr_mean | rank_corr_diff_vs_base | bootstrap_ci_low | bootstrap_ci_high | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ensemble_4family+image_factor_pc1 | ensemble_4family | image_factor_pc1 | 0.07978 | 0.55169 | 0.31550 | 0.56944 | 0.00484 | 0.54895 | 144.00000 | 0.21958 | 0.21930 | 0.01239 | 0.00001 | 0.02505 | 2880 |
| ensemble_best+image_factor_pc1 | ensemble_best | image_factor_pc1 | 0.07309 | 1.16359 | 0.51094 | 0.56944 | 0.00899 | 0.58741 | 144.00000 | 0.24548 | 0.24508 | 0.01254 | 0.00057 | 0.02448 | 2880 |
| ensemble_4family | ensemble_4family |  | 0.06738 | 1.23950 | 0.54300 | 0.60417 | 0.00591 | 0.60140 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_4family+image_score | ensemble_4family | image_score | 0.06065 | 0.70733 | 0.36370 | 0.56944 | 0.00443 | 0.57343 | 144.00000 | 0.19761 | 0.27944 | -0.00704 | -0.01854 | 0.00429 | 2880 |
| ensemble_best | ensemble_best |  | 0.06055 | 1.50864 | 0.64255 | 0.58333 | 0.00618 | 0.66434 | 144.00000 | nan | nan | 0.00000 | 0.00000 | 0.00000 | 2880 |
| ensemble_best+image_score | ensemble_best | image_score | 0.05428 | 0.65585 | 0.35927 | 0.52083 | 0.00446 | 0.62238 | 144.00000 | 0.22931 | 0.31540 | -0.00628 | -0.01722 | 0.00481 | 2880 |

## Interpretation Rules
- If a factor has p < 0.05, call it statistically significant after rolling PCA controls.
- If p includes 0 but delta R2 and ensemble rank-corr lift are positive, frame it as mechanism/trend evidence.
- If ensemble CI includes 0, do not claim strict improvement over the baseline ensemble.

## Output Files
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_ma_volume/image_factor_panel.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_ma_volume/common_pca_controls.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_ma_volume/image_factor_significance.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_ma_volume/ensemble_image_factor_search.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_ma_volume/image_factor_signals.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ohlc_ma_volume/ode_mu_candidate_signals.csv`
