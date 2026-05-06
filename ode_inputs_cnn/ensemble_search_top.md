# Ensemble search — top-20 per aggregation mode

## Mode: raw (sorted by OOS rank corr)

|   k | members                                                                                              |   rank_corr |   top_k_sharpe |   top_k_cum |   top_k_hit |
|----:|:-----------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|------------:|
|   4 | cnn_1d_cumulative_scale + cnn_2d_residual_images + cnn_2d_residual_small + cnnlstm_image_scale       |      0.0614 |         0.2855 |      0.4560 |      0.6111 |
|   3 | cnn_1d_cumulative_scale + cnn_2d_residual_small + lstm_image_scale                                   |      0.0591 |         0.2315 |      0.3039 |      0.6181 |
|   4 | cnn_1d_cumulative_scale + cnn_2d_residual_images + cnn_2d_residual_small + lstm_image_scale          |      0.0581 |         0.3057 |      0.5191 |      0.6181 |
|   2 | cnn_2d_residual_small + cnnlstm_image_scale                                                          |      0.0579 |         0.3486 |      0.6359 |      0.5972 |
|   3 | cnn_2d_residual_small + lstm_image_scale + cnnlstm_image_scale                                       |      0.0578 |         0.2597 |      0.3870 |      0.5903 |
|   4 | cnn_1d_cumulative_scale + cnn_2d_residual_small + lstm_image_scale + cnnlstm_image_scale             |      0.0574 |         0.2952 |      0.4887 |      0.6042 |
|   3 | logistic_image_scale + cnn_2d_residual_small + lstm_image_scale                                      |      0.0572 |         0.3643 |      0.7126 |      0.6111 |
|   2 | cnn_2d_residual_small + lstm_image_scale                                                             |      0.0568 |         0.1980 |      0.2296 |      0.5694 |
|   3 | cnn_1d_cumulative_scale + cnn_2d_residual_small + cnnlstm_image_scale                                |      0.0565 |         0.3702 |      0.7021 |      0.5972 |
|   3 | cnn_2d_residual_images + cnn_2d_residual_small + cnnlstm_image_scale                                 |      0.0561 |         0.3456 |      0.6290 |      0.6042 |
|   4 | logistic_image_scale + cnn_2d_residual_small + lstm_image_scale + lstm_cumulative_scale              |      0.0561 |         0.4482 |      1.0008 |      0.5903 |
|   4 | logistic_image_scale + cnn_2d_residual_images + cnn_2d_residual_small + lstm_image_scale             |      0.0554 |         0.3154 |      0.5448 |      0.5972 |
|   4 | cnn_2d_residual_images + cnn_2d_residual_small + lstm_image_scale + cnnlstm_image_scale              |      0.0548 |         0.3969 |      0.8035 |      0.6181 |
|   2 | lstm_image_scale + cnnlstm_image_scale                                                               |      0.0545 |         0.2783 |      0.4454 |      0.5625 |
|   3 | logistic_image_scale + cnn_2d_residual_small + cnnlstm_image_scale                                   |      0.0545 |         0.3040 |      0.5179 |      0.6111 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small + lstm_image_scale            |      0.0542 |         0.3587 |      0.6909 |      0.5972 |
|   4 | logistic_image_scale + cnn_2d_residual_small + lstm_image_scale + cnnlstm_image_scale                |      0.0541 |         0.3468 |      0.6513 |      0.6042 |
|   4 | cnn_2d_residual_images + cnn_2d_residual_small + lstm_cumulative_scale + cnnlstm_image_scale         |      0.0541 |         0.3423 |      0.6244 |      0.5972 |
|   4 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale + lstm_image_scale |      0.0540 |         0.3395 |      0.6160 |      0.5833 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small + cnnlstm_image_scale         |      0.0538 |         0.3289 |      0.5876 |      0.6042 |

## Mode: raw (sorted by top-k Sharpe)

|   k | members                                                                                                  |   rank_corr |   top_k_sharpe |   top_k_cum |   top_k_hit |
|----:|:---------------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|------------:|
|   2 | lstm_image_scale + cnnlstm_cumulative_scale                                                              |      0.0254 |         0.6987 |      1.8669 |      0.5972 |
|   4 | cnn_1d_cumulative_scale + lstm_image_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale              |      0.0259 |         0.6482 |      1.6907 |      0.6111 |
|   3 | cnn_1d_cumulative_scale + lstm_image_scale + cnnlstm_cumulative_scale                                    |      0.0288 |         0.6411 |      1.6354 |      0.5903 |
|   3 | cnn_1d_cumulative_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale                                 |      0.0167 |         0.6369 |      1.6302 |      0.5972 |
|   4 | logistic_image_scale + lstm_image_scale + lstm_cumulative_scale + cnnlstm_cumulative_scale               |      0.0401 |         0.6189 |      1.5142 |      0.5972 |
|   2 | cnnlstm_image_scale + cnnlstm_cumulative_scale                                                           |      0.0144 |         0.6110 |      1.4791 |      0.5972 |
|   3 | lstm_image_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale                                        |      0.0286 |         0.6046 |      1.4760 |      0.5903 |
|   3 | logistic_image_scale + lstm_image_scale + cnnlstm_cumulative_scale                                       |      0.0403 |         0.5854 |      1.3712 |      0.5972 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + lstm_image_scale + cnnlstm_cumulative_scale             |      0.0450 |         0.5776 |      1.3630 |      0.5694 |
|   4 | logistic_image_scale + cnn_1d_multiscale_image_scale + lstm_image_scale + cnnlstm_cumulative_scale       |      0.0354 |         0.5691 |      1.2963 |      0.5903 |
|   3 | cnn_1d_attention_image_scale + lstm_cumulative_scale + cnnlstm_cumulative_scale                          |      0.0018 |         0.5653 |      1.4447 |      0.6181 |
|   3 | logistic_image_scale + cnn_2d_residual_small + cnnlstm_cumulative_scale                                  |      0.0397 |         0.5444 |      1.2161 |      0.6042 |
|   3 | logistic_image_scale + lstm_cumulative_scale + cnnlstm_cumulative_scale                                  |      0.0268 |         0.5428 |      1.2669 |      0.5903 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale          |      0.0398 |         0.5359 |      1.1961 |      0.5972 |
|   3 | cnn_1d_cumulative_scale + cnn_1d_multiscale_image_scale + cnnlstm_cumulative_scale                       |      0.0062 |         0.5342 |      1.0096 |      0.5764 |
|   4 | cnn_1d_cumulative_scale + cnn_1d_multiscale_image_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale |      0.0223 |         0.5329 |      1.1501 |      0.5694 |
|   4 | cnn_1d_cumulative_scale + lstm_cumulative_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale         |      0.0110 |         0.5328 |      1.2630 |      0.5694 |
|   2 | cnn_1d_multiscale_image_scale + cnnlstm_cumulative_scale                                                 |      0.0031 |         0.5303 |      1.0829 |      0.5764 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small + cnnlstm_cumulative_scale        |      0.0414 |         0.5292 |      1.1651 |      0.5833 |
|   3 | lstm_cumulative_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale                                   |      0.0093 |         0.5289 |      1.1465 |      0.5764 |

## Mode: rank (sorted by OOS rank corr)

|   k | members                                                                                              |   rank_corr |   top_k_sharpe |   top_k_cum |   top_k_hit |
|----:|:-----------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|------------:|
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small + cnnlstm_image_scale         |      0.0671 |         0.5406 |      1.2323 |      0.6111 |
|   4 | cnn_1d_cumulative_scale + cnn_2d_residual_small + lstm_image_scale + cnnlstm_image_scale             |      0.0670 |         0.2901 |      0.4837 |      0.5903 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small + lstm_image_scale            |      0.0653 |         0.4519 |      0.9813 |      0.5903 |
|   3 | cnn_1d_cumulative_scale + cnn_2d_residual_small + lstm_image_scale                                   |      0.0628 |         0.3602 |      0.6804 |      0.5694 |
|   4 | logistic_image_scale + cnn_2d_residual_small + lstm_image_scale + cnnlstm_image_scale                |      0.0621 |         0.2950 |      0.4847 |      0.5903 |
|   3 | cnn_2d_residual_small + lstm_image_scale + cnnlstm_image_scale                                       |      0.0617 |         0.2150 |      0.2633 |      0.5625 |
|   3 | cnn_1d_cumulative_scale + cnn_2d_residual_small + cnnlstm_image_scale                                |      0.0614 |         0.6821 |      1.6648 |      0.5903 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small                               |      0.0614 |         0.6302 |      1.4609 |      0.5764 |
|   3 | logistic_image_scale + cnn_2d_residual_small + cnnlstm_image_scale                                   |      0.0602 |         0.4699 |      0.9551 |      0.5972 |
|   4 | cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_small + lstm_image_scale      |      0.0598 |         0.2758 |      0.4293 |      0.5278 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + lstm_image_scale + cnnlstm_image_scale              |      0.0597 |         0.3214 |      0.5483 |      0.5556 |
|   4 | cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_small + cnnlstm_image_scale   |      0.0592 |         0.2219 |      0.2832 |      0.5625 |
|   4 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_small + lstm_image_scale         |      0.0592 |         0.2430 |      0.3399 |      0.5764 |
|   4 | cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small + cnnlstm_image_scale |      0.0589 |         0.3395 |      0.6204 |      0.5764 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + lstm_image_scale       |      0.0584 |         0.3465 |      0.6734 |      0.5833 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_small  |      0.0582 |         0.3940 |      0.7966 |      0.5625 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + lstm_image_scale                                    |      0.0580 |         0.5651 |      1.2063 |      0.5972 |
|   4 | cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small + lstm_image_scale    |      0.0579 |         0.2800 |      0.4301 |      0.5694 |
|   2 | cnn_2d_residual_small + cnnlstm_image_scale                                                          |      0.0579 |         0.5335 |      1.0938 |      0.6250 |
|   4 | cnn_1d_multiscale_image_scale + cnn_2d_residual_small + lstm_image_scale + cnnlstm_image_scale       |      0.0579 |         0.2755 |      0.4308 |      0.5972 |

## Mode: rank (sorted by top-k Sharpe)

|   k | members                                                                                           |   rank_corr |   top_k_sharpe |   top_k_cum |   top_k_hit |
|----:|:--------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|------------:|
|   3 | cnn_1d_cumulative_scale + cnn_2d_residual_small + cnnlstm_image_scale                             |      0.0614 |         0.6821 |      1.6648 |      0.5903 |
|   2 | cnn_1d_cumulative_scale + cnnlstm_image_scale                                                     |      0.0467 |         0.6489 |      1.6063 |      0.5903 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small                            |      0.0614 |         0.6302 |      1.4609 |      0.5764 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + lstm_image_scale + cnnlstm_cumulative_scale      |      0.0399 |         0.5865 |      1.4494 |      0.5764 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnnlstm_image_scale                              |      0.0576 |         0.5746 |      1.3074 |      0.5903 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_multiscale_image_scale                    |      0.0348 |         0.5660 |      1.2744 |      0.5903 |
|   3 | cnn_1d_cumulative_scale + cnn_2d_residual_small + cnnlstm_cumulative_scale                        |      0.0210 |         0.5655 |      1.1122 |      0.5972 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + lstm_image_scale                                 |      0.0580 |         0.5651 |      1.2063 |      0.5972 |
|   4 | cnn_1d_cumulative_scale + cnn_2d_residual_images + cnnlstm_image_scale + cnnlstm_cumulative_scale |      0.0236 |         0.5621 |      1.2266 |      0.5972 |
|   2 | logistic_image_scale + cnn_1d_attention_image_scale                                               |      0.0334 |         0.5564 |      1.1311 |      0.6111 |
|   4 | logistic_image_scale + cnn_2d_rendered_images + cnnlstm_image_scale + cnnlstm_cumulative_scale    |      0.0186 |         0.5518 |      1.1267 |      0.5833 |
|   3 | logistic_image_scale + lstm_cumulative_scale + cnnlstm_image_scale                                |      0.0275 |         0.5494 |      1.2757 |      0.5556 |
|   3 | cnn_1d_cumulative_scale + lstm_image_scale + cnnlstm_image_scale                                  |      0.0577 |         0.5409 |      1.2912 |      0.5833 |
|   4 | cnn_1d_cumulative_scale + lstm_cumulative_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale  |      0.0136 |         0.5407 |      1.2372 |      0.5694 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small + cnnlstm_image_scale      |      0.0671 |         0.5406 |      1.2323 |      0.6111 |
|   4 | cnn_1d_cumulative_scale + cnn_2d_residual_small + cnnlstm_image_scale + cnnlstm_cumulative_scale  |      0.0390 |         0.5356 |      1.1792 |      0.6111 |
|   2 | cnn_2d_residual_small + cnnlstm_image_scale                                                       |      0.0579 |         0.5335 |      1.0938 |      0.6250 |
|   3 | cnn_1d_cumulative_scale + lstm_image_scale + cnnlstm_cumulative_scale                             |      0.0301 |         0.5318 |      1.0682 |      0.5764 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small + cnnlstm_cumulative_scale |      0.0381 |         0.5302 |      1.1556 |      0.5833 |
|   3 | cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnnlstm_image_scale                      |      0.0538 |         0.5282 |      1.1844 |      0.5833 |
