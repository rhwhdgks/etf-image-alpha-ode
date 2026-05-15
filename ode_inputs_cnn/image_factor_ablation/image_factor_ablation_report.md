# Image Factor Ablation Report

## Scope
- Optimization is excluded in this run.
- Model architecture is fixed to `cnn_2d_residual_small`; only chart rendering components vary.
- Lookback/horizon: `60/20`
- CNN epochs/patience: `30/5`

## Main Result
- Best PCA-control factor by p-value: `ohlc_ma::image_score` (t=7.232, p=0.0000, delta R2=0.008839)
- Best ensemble row by rank corr: `ohlc_ma_volume::ensemble_4family+image_factor_pc1` (rank corr=0.0798, Sharpe=0.3155)

## Variant Summary
| chart_variant | best_factor | best_factor_t_stat | best_factor_p_value | best_factor_delta_r2 | best_factor_daily_rank_corr | best_ensemble_candidate | best_ensemble_rank_corr | best_ensemble_sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| close_only | image_factor_pc2 | 6.85920 | 0.00000 | 0.00443 | 0.01957 | ensemble_4family | 0.06738 | 0.54300 |
| ohlc_full | image_factor_pc2 | 2.84807 | 0.00443 | 0.00139 | 0.02016 | ensemble_4family | 0.06738 | 0.54300 |
| ohlc_ma | image_score | 7.23197 | 0.00000 | 0.00884 | 0.01837 | ensemble_4family | 0.06738 | 0.54300 |
| ohlc_volume | image_factor_pc1 | -6.53683 | 0.00000 | 0.00388 | 0.01984 | ensemble_4family | 0.06738 | 0.54300 |
| ohlc_ma_volume | image_factor_pc3 | 4.25927 | 0.00002 | 0.00318 | -0.03635 | ensemble_4family+image_factor_pc1 | 0.07978 | 0.31550 |
| high_low_range | image_factor_pc3 | 6.27259 | 0.00000 | 0.00397 | 0.04288 | ensemble_4family | 0.06738 | 0.54300 |

## All Significance Tests
| chart_variant | factor_name | t_stat | p_value | delta_r2 | daily_rank_correlation | n_obs | n_dates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| close_only | image_factor_pc2 | 6.85920 | 0.00000 | 0.00443 | 0.01957 | 20160 | 2880 |
| close_only | image_factor_pc1 | -4.81914 | 0.00000 | 0.00370 | -0.00574 | 20160 | 2880 |
| close_only | image_score | 4.14306 | 0.00004 | 0.00194 | -0.00062 | 20160 | 2880 |
| close_only | image_factor_pc3 | -1.72279 | 0.08503 | 0.00029 | -0.00362 | 20160 | 2880 |
| ohlc_full | image_factor_pc2 | 2.84807 | 0.00443 | 0.00139 | 0.02016 | 20160 | 2880 |
| ohlc_full | image_factor_pc3 | -2.30814 | 0.02106 | 0.00157 | 0.00988 | 20160 | 2880 |
| ohlc_full | image_score | 2.24206 | 0.02503 | 0.00090 | 0.02036 | 20160 | 2880 |
| ohlc_full | image_factor_pc1 | -0.23447 | 0.81463 | 0.00001 | 0.01731 | 20160 | 2880 |
| ohlc_ma | image_score | 7.23197 | 0.00000 | 0.00884 | 0.01837 | 20160 | 2880 |
| ohlc_ma | image_factor_pc2 | 5.14581 | 0.00000 | 0.00441 | 0.00651 | 20160 | 2880 |
| ohlc_ma | image_factor_pc1 | 2.31334 | 0.02077 | 0.00127 | 0.02490 | 20160 | 2880 |
| ohlc_ma | image_factor_pc3 | 1.99447 | 0.04620 | 0.00038 | 0.02648 | 20160 | 2880 |
| ohlc_volume | image_factor_pc1 | -6.53683 | 0.00000 | 0.00388 | 0.01984 | 20160 | 2880 |
| ohlc_volume | image_score | -3.84653 | 0.00012 | 0.00169 | -0.01391 | 20160 | 2880 |
| ohlc_volume | image_factor_pc3 | 2.45203 | 0.01426 | 0.00073 | 0.00278 | 20160 | 2880 |
| ohlc_volume | image_factor_pc2 | -1.64823 | 0.09941 | 0.00035 | 0.03045 | 20160 | 2880 |
| ohlc_ma_volume | image_factor_pc3 | 4.25927 | 0.00002 | 0.00318 | -0.03635 | 20160 | 2880 |
| ohlc_ma_volume | image_factor_pc2 | -3.81771 | 0.00014 | 0.00135 | -0.02884 | 20160 | 2880 |
| ohlc_ma_volume | image_factor_pc1 | 3.55110 | 0.00039 | 0.00159 | 0.05921 | 20160 | 2880 |
| ohlc_ma_volume | image_score | 3.33847 | 0.00085 | 0.00117 | 0.02851 | 20160 | 2880 |
| high_low_range | image_factor_pc3 | 6.27259 | 0.00000 | 0.00397 | 0.04288 | 20160 | 2880 |
| high_low_range | image_factor_pc1 | -1.04217 | 0.29742 | 0.00021 | 0.01146 | 20160 | 2880 |
| high_low_range | image_score | -1.00936 | 0.31289 | 0.00020 | 0.00930 | 20160 | 2880 |
| high_low_range | image_factor_pc2 | 0.64291 | 0.52033 | 0.00006 | 0.00601 | 20160 | 2880 |

## Ensemble Additions
| chart_variant | candidate_name | rank_corr | top_k_sharpe | rank_corr_diff_vs_base | bootstrap_ci_low | bootstrap_ci_high | factor_vs_base_rank_corr_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| close_only | ensemble_4family | 0.06738 | 0.54300 | 0.00000 | 0.00000 | 0.00000 | nan |
| close_only | ensemble_best | 0.06055 | 0.64255 | 0.00000 | 0.00000 | 0.00000 | nan |
| close_only | ensemble_4family+image_score | 0.04515 | 0.43517 | -0.02223 | -0.03490 | -0.00932 | 0.14540 |
| close_only | ensemble_4family+image_factor_pc1 | 0.04177 | 0.59397 | -0.02561 | -0.03931 | -0.01200 | 0.10143 |
| close_only | ensemble_best+image_score | 0.03904 | 0.47095 | -0.02132 | -0.03408 | -0.00874 | 0.13849 |
| close_only | ensemble_best+image_factor_pc1 | 0.03630 | 0.38714 | -0.02426 | -0.03866 | -0.01024 | 0.08789 |
| ohlc_full | ensemble_4family | 0.06738 | 0.54300 | 0.00000 | 0.00000 | 0.00000 | nan |
| ohlc_full | ensemble_best | 0.06055 | 0.64255 | 0.00000 | 0.00000 | 0.00000 | nan |
| ohlc_full | ensemble_4family+image_score | 0.05508 | 0.42418 | -0.01230 | -0.02517 | 0.00062 | 0.14188 |
| ohlc_full | ensemble_4family+image_factor_pc1 | 0.05265 | 0.64740 | -0.01473 | -0.02799 | -0.00160 | 0.15561 |
| ohlc_full | ensemble_best+image_score | 0.05157 | 0.49784 | -0.00915 | -0.02171 | 0.00349 | 0.14530 |
| ohlc_full | ensemble_best+image_factor_pc1 | 0.04734 | 0.68845 | -0.01321 | -0.02640 | -0.00004 | 0.16119 |
| ohlc_ma | ensemble_4family | 0.06738 | 0.54300 | 0.00000 | 0.00000 | 0.00000 | nan |
| ohlc_ma | ensemble_4family+image_factor_pc1 | 0.06245 | 0.35701 | -0.00493 | -0.01742 | 0.00722 | 0.23270 |
| ohlc_ma | ensemble_best | 0.06055 | 0.64255 | 0.00000 | 0.00000 | 0.00000 | nan |
| ohlc_ma | ensemble_best+image_factor_pc1 | 0.05733 | 0.50938 | -0.00322 | -0.01539 | 0.00849 | 0.24575 |
| ohlc_ma | ensemble_4family+image_score | 0.05645 | 0.39948 | -0.01093 | -0.02300 | 0.00109 | 0.20536 |
| ohlc_ma | ensemble_best+image_score | 0.05101 | 0.44044 | -0.00954 | -0.02152 | 0.00229 | 0.24159 |
| ohlc_volume | ensemble_4family | 0.06738 | 0.54300 | 0.00000 | 0.00000 | 0.00000 | nan |
| ohlc_volume | ensemble_best | 0.06055 | 0.64255 | 0.00000 | 0.00000 | 0.00000 | nan |
| ohlc_volume | ensemble_4family+image_factor_pc1 | 0.05799 | 0.50597 | -0.00939 | -0.02094 | 0.00232 | 0.27783 |
| ohlc_volume | ensemble_best+image_factor_pc1 | 0.05528 | 0.48819 | -0.00542 | -0.01719 | 0.00665 | 0.22288 |
| ohlc_volume | ensemble_4family+image_score | 0.03735 | 0.26244 | -0.03003 | -0.04193 | -0.01783 | 0.25816 |
| ohlc_volume | ensemble_best+image_score | 0.03423 | 0.37760 | -0.02632 | -0.03834 | -0.01431 | 0.23616 |
| ohlc_ma_volume | ensemble_4family+image_factor_pc1 | 0.07978 | 0.31550 | 0.01239 | 0.00001 | 0.02505 | 0.21930 |
| ohlc_ma_volume | ensemble_best+image_factor_pc1 | 0.07309 | 0.51094 | 0.01254 | 0.00057 | 0.02448 | 0.24508 |
| ohlc_ma_volume | ensemble_4family | 0.06738 | 0.54300 | 0.00000 | 0.00000 | 0.00000 | nan |
| ohlc_ma_volume | ensemble_4family+image_score | 0.06065 | 0.36370 | -0.00704 | -0.01854 | 0.00429 | 0.27944 |
| ohlc_ma_volume | ensemble_best | 0.06055 | 0.64255 | 0.00000 | 0.00000 | 0.00000 | nan |
| ohlc_ma_volume | ensemble_best+image_score | 0.05428 | 0.35927 | -0.00628 | -0.01722 | 0.00481 | 0.31540 |
| high_low_range | ensemble_4family | 0.06738 | 0.54300 | 0.00000 | 0.00000 | 0.00000 | nan |
| high_low_range | ensemble_best | 0.06055 | 0.64255 | 0.00000 | 0.00000 | 0.00000 | nan |
| high_low_range | ensemble_4family+image_score | 0.05164 | 0.38112 | -0.01574 | -0.02911 | -0.00283 | 0.13918 |
| high_low_range | ensemble_4family+image_factor_pc1 | 0.04714 | 0.39340 | -0.02001 | -0.03384 | -0.00666 | 0.12228 |
| high_low_range | ensemble_best+image_score | 0.04361 | 0.40812 | -0.01694 | -0.03024 | -0.00380 | 0.12698 |
| high_low_range | ensemble_best+image_factor_pc1 | 0.03961 | 0.33894 | -0.02094 | -0.03435 | -0.00767 | 0.11384 |

## Interpretation
- If a variant with MA improves significance, the trend-line component is the likely information source.
- If a variant with volume improves significance, volume-path information is contributing.
- If `high_low_range` wins, intrawindow volatility/range shape matters more than open-close ticks.
- If `close_only` wins, most usable information is in the price path itself, not richer candle details.

## Output Files
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ablation_summary.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ablation_significance.csv`
- `/home/jonghan/findalpha/delta/sprint2/ode_inputs_cnn/image_factor_ablation/ablation_ensemble_search.csv`
