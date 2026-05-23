# Image Factor Multiple-Testing Check

## Purpose

This check addresses data-snooping concerns from testing multiple chart variants and image-factor candidates.

## Test Family

- Tests included: `28` = 6 ablation variants x 4 factors plus strict-window MA 4 factors.
- Corrections: Bonferroni and Benjamini-Hochberg FDR across the full test family.

## Summary

- Raw p < 0.05: `18` tests.
- Bonferroni p < 0.05: `12` tests.
- FDR-BH p < 0.05: `17` tests.

## Top Corrected Results

| test | t-stat | raw p | Bonferroni p | FDR p | delta R2 | daily rank corr |
|---|---:|---:|---:|---:|---:|---:|
| `ohlc_ma::image_score` | 7.232 | 6.07e-13 | 1.7e-11 | 1.7e-11 | 0.008839 | 0.0184 |
| `close_only::image_factor_pc2` | 6.859 | 8.44e-12 | 2.36e-10 | 1.18e-10 | 0.004431 | 0.0196 |
| `ohlc_volume::image_factor_pc1` | -6.537 | 7.41e-11 | 2.07e-09 | 6.91e-10 | 0.003877 | 0.0198 |
| `high_low_range::image_factor_pc3` | 6.273 | 4.08e-10 | 1.14e-08 | 2.86e-09 | 0.003972 | 0.0429 |
| `ohlc_ma::image_factor_pc2` | 5.146 | 2.84e-07 | 7.96e-06 | 1.59e-06 | 0.004413 | 0.0065 |
| `close_only::image_factor_pc1` | -4.819 | 1.52e-06 | 4.25e-05 | 7.08e-06 | 0.003697 | -0.0057 |
| `ohlc_ma_volume::image_factor_pc3` | 4.259 | 2.12e-05 | 0.000593 | 8.47e-05 | 0.003180 | -0.0363 |
| `close_only::image_score` | 4.143 | 3.53e-05 | 0.000987 | 0.000123 | 0.001936 | -0.0006 |
| `ohlc_volume::image_score` | -3.847 | 0.000122 | 0.00343 | 0.000381 | 0.001686 | -0.0139 |
| `ohlc_ma_volume::image_factor_pc2` | -3.818 | 0.000138 | 0.00385 | 0.000385 | 0.001352 | -0.0288 |
| `ohlc_ma_volume::image_factor_pc1` | 3.551 | 0.00039 | 0.0109 | 0.000992 | 0.001586 | 0.0592 |
| `ohlc_ma_volume::image_score` | 3.338 | 0.000853 | 0.0239 | 0.00199 | 0.001174 | 0.0285 |

## Interpretation

- The strongest original ablation results remain significant after conservative multiple-testing correction.
- Strict-window MA standalone OLS factors do not survive correction, so strict MA should be framed as ensemble robustness rather than standalone factor significance.
- This supports a cautious claim: image-factor information is not only a single cherry-picked p-value, but inference should still emphasize OOS ranking and robustness rather than raw significance alone.
