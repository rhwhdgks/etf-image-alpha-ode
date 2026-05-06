# Ensemble search — top-20 per aggregation mode

## Mode: raw — sorted by OOS rank corr

|   k | members                                                                                                |   rank_corr |   top_k_sharpe |   top_k_cum |   top_k_hit |
|----:|:-------------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|------------:|
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_attention_image_scale + cnn_lstm_image_scale        |   0.049566  |      0.219018  |   0.279039  |    0.590278 |
|   3 | logistic_image_scale + cnn_1d_image_scale + cnn_lstm_image_scale                                       |   0.04938   |      0.174311  |   0.175356  |    0.541667 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_attention_image_scale + cnnlstm_image_scale         |   0.0493056 |      0.22649   |   0.296879  |    0.590278 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_dilated_image_scale + cnn_lstm_image_scale          |   0.0491939 |      0.171394  |   0.168581  |    0.555556 |
|   2 | logistic_image_scale + cnnlstm_image_scale                                                             |   0.0489583 |      0.45337   |   0.893377  |    0.583333 |
|   3 | logistic_image_scale + cnn_lstm_image_scale + cnnlstm_image_scale                                      |   0.0487971 |      0.414788  |   0.791715  |    0.597222 |
|   3 | logistic_image_scale + cnn_1d_image_scale + cnnlstm_image_scale                                        |   0.0481399 |      0.198646  |   0.229315  |    0.555556 |
|   3 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_attention_image_scale                               |   0.0481151 |      0.299277  |   0.481905  |    0.555556 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_lstm_image_scale + cnnlstm_image_scale                 |   0.0480779 |      0.228418  |   0.305923  |    0.5625   |
|   2 | logistic_image_scale + cnn_1d_image_scale                                                              |   0.0475942 |      0.212503  |   0.258968  |    0.541667 |
|   3 | logistic_image_scale + cnn_1d_attention_image_scale + cnnlstm_image_scale                              |   0.0474578 |      0.230047  |   0.307982  |    0.583333 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_dilated_image_scale + cnnlstm_image_scale           |   0.0471726 |      0.17314   |   0.17251   |    0.555556 |
|   4 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_lstm_image_scale + cnnlstm_image_scale         |   0.0470858 |      0.261234  |   0.393594  |    0.576389 |
|   2 | logistic_image_scale + cnn_lstm_image_scale                                                            |   0.0456101 |      0.495521  |   1.04654   |    0.590278 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_attention_image_scale + cnnlstm_cumulative_scale    |   0.0451389 |      0.289507  |   0.465429  |    0.5625   |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_lstm_image_scale + cnnlstm_cumulative_scale            |   0.0450025 |      0.446732  |   0.851538  |    0.569444 |
|   1 | cnnlstm_image_scale                                                                                    |   0.0448289 |      0.433729  |   0.899802  |    0.576389 |
|   3 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_dilated_image_scale                                 |   0.0447297 |      0.260427  |   0.381412  |    0.548611 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_dilated_image_scale + cnnlstm_image_scale |   0.0446181 |      0.163236  |   0.151064  |    0.569444 |
|   3 | cnn_1d_image_scale + cnn_1d_attention_image_scale + cnnlstm_image_scale                                |   0.0445933 |      0.0774432 |  -0.0333793 |    0.555556 |

## Mode: raw — sorted by top-k Sharpe

|   k | members                                                                                              |   rank_corr |   top_k_sharpe |   top_k_cum |   top_k_hit |
|----:|:-----------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|------------:|
|   2 | cnn_lstm_image_scale + cnnlstm_cumulative_scale                                                      |   0.0119792 |       0.637125 |    1.58577  |    0.597222 |
|   2 | cnnlstm_image_scale + cnnlstm_cumulative_scale                                                       |   0.0144345 |       0.611041 |    1.47912  |    0.597222 |
|   4 | logistic_image_scale + cnn_lstm_image_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale         |   0.0385541 |       0.574033 |    1.38044  |    0.618056 |
|   3 | cnn_lstm_image_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale                                |   0.0251488 |       0.561369 |    1.29     |    0.611111 |
|   3 | logistic_image_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale                                |   0.0412698 |       0.513498 |    1.16287  |    0.583333 |
|   2 | logistic_image_scale + cnn_lstm_image_scale                                                          |   0.0456101 |       0.495521 |    1.04654  |    0.590278 |
|   3 | logistic_image_scale + cnn_lstm_image_scale + cnnlstm_cumulative_scale                               |   0.0389261 |       0.493495 |    1.07558  |    0.597222 |
|   2 | logistic_image_scale + cnnlstm_image_scale                                                           |   0.0489583 |       0.45337  |    0.893377 |    0.583333 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_lstm_image_scale + cnnlstm_cumulative_scale          |   0.0450025 |       0.446732 |    0.851538 |    0.569444 |
|   1 | cnnlstm_image_scale                                                                                  |   0.0448289 |       0.433729 |    0.899802 |    0.576389 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale           |   0.043316  |       0.432022 |    0.810576 |    0.576389 |
|   1 | logistic_image_scale                                                                                 |   0.0424851 |       0.422578 |    0.764666 |    0.604167 |
|   3 | logistic_image_scale + cnn_lstm_image_scale + cnnlstm_image_scale                                    |   0.0487971 |       0.414788 |    0.791715 |    0.597222 |
|   3 | cnn_1d_image_scale + cnn_1d_dilated_image_scale + cnnlstm_cumulative_scale                           |   0.013182  |       0.411121 |    0.841285 |    0.611111 |
|   3 | logistic_image_scale + cnn_1d_image_scale + cnnlstm_cumulative_scale                                 |   0.0416543 |       0.395466 |    0.7084   |    0.541667 |
|   3 | logistic_cumulative_scale + logistic_image_scale + cnn_1d_dilated_image_scale                        |   0.0351935 |       0.380106 |    0.793573 |    0.597222 |
|   2 | logistic_image_scale + cnnlstm_cumulative_scale                                                      |   0.0374504 |       0.373668 |    0.674352 |    0.590278 |
|   4 | logistic_cumulative_scale + logistic_image_scale + cnn_1d_dilated_image_scale + cnn_lstm_image_scale |   0.0358507 |       0.370675 |    0.761417 |    0.590278 |
|   4 | logistic_cumulative_scale + logistic_image_scale + cnn_1d_dilated_image_scale + cnnlstm_image_scale  |   0.0368304 |       0.36499  |    0.740991 |    0.597222 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale |   0.0439856 |       0.363769 |    0.69789  |    0.597222 |

## Mode: rank — sorted by OOS rank corr

|   k | members                                                                                              |   rank_corr |   top_k_sharpe |   top_k_cum |   top_k_hit |
|----:|:-----------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|------------:|
|   2 | logistic_image_scale + cnnlstm_image_scale                                                           |   0.0552471 |       0.441215 |   0.889259  |    0.604167 |
|   3 | logistic_image_scale + cnn_1d_image_scale + cnn_lstm_image_scale                                     |   0.0521321 |       0.273496 |   0.427194  |    0.583333 |
|   2 | logistic_image_scale + cnn_lstm_image_scale                                                          |   0.0505979 |       0.331029 |   0.611437  |    0.604167 |
|   3 | logistic_image_scale + cnn_1d_attention_image_scale + cnnlstm_image_scale                            |   0.0501795 |       0.496697 |   1.03592   |    0.569444 |
|   3 | logistic_image_scale + cnn_1d_image_scale + cnnlstm_image_scale                                      |   0.0501041 |       0.171761 |   0.161093  |    0.5625   |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_lstm_image_scale + cnnlstm_image_scale     |   0.0499755 |       0.198231 |   0.225817  |    0.597222 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_attention_image_scale + cnn_lstm_image_scale      |   0.0497185 |       0.261235 |   0.387601  |    0.5625   |
|   3 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_lstm_image_scale                           |   0.048534  |       0.33212  |   0.596442  |    0.590278 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_attention_image_scale + cnnlstm_image_scale       |   0.0484711 |       0.188987 |   0.200834  |    0.569444 |
|   4 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_lstm_image_scale + cnnlstm_image_scale       |   0.0470721 |       0.325305 |   0.582089  |    0.590278 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_lstm_image_scale + cnnlstm_image_scale               |   0.0459199 |       0.214658 |   0.267427  |    0.597222 |
|   3 | logistic_image_scale + cnn_lstm_image_scale + cnnlstm_image_scale                                    |   0.0452963 |       0.407187 |   0.783043  |    0.604167 |
|   4 | cnn_1d_image_scale + cnn_1d_attention_image_scale + cnn_lstm_image_scale + cnnlstm_image_scale       |   0.0449538 |       0.131795 |   0.0722739 |    0.555556 |
|   1 | cnnlstm_image_scale                                                                                  |   0.0448289 |       0.433729 |   0.899802  |    0.576389 |
|   3 | cnn_1d_attention_image_scale + cnn_lstm_image_scale + cnnlstm_image_scale                            |   0.0447848 |       0.267049 |   0.410433  |    0.576389 |
|   4 | logistic_cumulative_scale + logistic_image_scale + cnn_lstm_image_scale + cnnlstm_image_scale        |   0.043121  |       0.302422 |   0.529104  |    0.618056 |
|   1 | logistic_image_scale                                                                                 |   0.0424851 |       0.422578 |   0.764666  |    0.604167 |
|   3 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_lstm_image_scale                             |   0.0418716 |       0.356692 |   0.676125  |    0.597222 |
|   4 | logistic_cumulative_scale + logistic_image_scale + cnn_1d_dilated_image_scale + cnn_lstm_image_scale |   0.0416675 |       0.309322 |   0.550402  |    0.604167 |
|   4 | logistic_cumulative_scale + cnn_1d_image_scale + cnn_1d_attention_image_scale + cnnlstm_image_scale  |   0.0414918 |       0.192022 |   0.218352  |    0.583333 |

## Mode: rank — sorted by top-k Sharpe

|   k | members                                                                                              |   rank_corr |   top_k_sharpe |   top_k_cum |   top_k_hit |
|----:|:-----------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|------------:|
|   3 | logistic_image_scale + cnn_lstm_image_scale + cnnlstm_cumulative_scale                               |   0.031899  |       0.702713 |    1.95125  |    0.597222 |
|   3 | logistic_image_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale                                |   0.0337624 |       0.586636 |    1.37852  |    0.576389 |
|   3 | logistic_image_scale + cnn_1d_attention_image_scale + cnnlstm_image_scale                            |   0.0501795 |       0.496697 |    1.03592  |    0.569444 |
|   4 | logistic_cumulative_scale + cnn_lstm_image_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale    |   0.0239507 |       0.489667 |    1.09876  |    0.597222 |
|   2 | logistic_image_scale + cnnlstm_cumulative_scale                                                      |   0.0188957 |       0.489203 |    1.01501  |    0.583333 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale           |   0.0343345 |       0.479389 |    0.968828 |    0.590278 |
|   4 | logistic_image_scale + cnn_1d_dilated_image_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale   |   0.0257152 |       0.475958 |    0.972677 |    0.569444 |
|   3 | logistic_image_scale + cnn_1d_dilated_image_scale + cnnlstm_cumulative_scale                         |   0.0140205 |       0.4592   |    0.945478 |    0.569444 |
|   2 | logistic_image_scale + cnnlstm_image_scale                                                           |   0.0552471 |       0.441215 |    0.889259 |    0.604167 |
|   1 | cnnlstm_image_scale                                                                                  |   0.0448289 |       0.433729 |    0.899802 |    0.576389 |
|   4 | logistic_image_scale + cnn_lstm_image_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale         |   0.0360303 |       0.433693 |    0.875444 |    0.555556 |
|   2 | cnnlstm_image_scale + cnnlstm_cumulative_scale                                                       |   0.0217685 |       0.431845 |    0.802846 |    0.597222 |
|   2 | cnn_lstm_image_scale + cnnlstm_cumulative_scale                                                      |   0.019696  |       0.430081 |    0.885909 |    0.611111 |
|   4 | logistic_cumulative_scale + logistic_image_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale    |   0.0276664 |       0.427313 |    0.918679 |    0.583333 |
|   1 | logistic_image_scale                                                                                 |   0.0424851 |       0.422578 |    0.764666 |    0.604167 |
|   3 | logistic_image_scale + cnn_1d_attention_image_scale + cnnlstm_cumulative_scale                       |   0.0275109 |       0.421765 |    0.793901 |    0.597222 |
|   2 | cnn_lstm_image_scale + cnnlstm_image_scale                                                           |   0.0386333 |       0.420219 |    0.849323 |    0.590278 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale |   0.0346485 |       0.414097 |    0.790079 |    0.590278 |
|   3 | logistic_cumulative_scale + cnnlstm_image_scale + cnnlstm_cumulative_scale                           |   0.0166507 |       0.408883 |    0.832079 |    0.576389 |
|   3 | logistic_image_scale + cnn_lstm_image_scale + cnnlstm_image_scale                                    |   0.0452963 |       0.407187 |    0.783043 |    0.604167 |
