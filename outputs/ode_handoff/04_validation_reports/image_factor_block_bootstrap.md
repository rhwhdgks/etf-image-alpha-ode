# Image-Factor Lift — Block Bootstrap Recheck

The handoff's image-factor lift CIs were computed with IID resampling.
Here the same lifts are recomputed with a stationary block bootstrap
(Politis-Romano, mean block 20 = horizon, B=10000), which is the
methodologically correct CI for the autocorrelated per-date series.

| comparison | mean lift | IID 95% CI | block 95% CI | block verdict |
|---|---:|---|---|---|
| image_factor_rank vs rank_baseline | +0.0124 | [+0.0000, +0.0250] | [-0.0267, +0.0507] | **NOT significant** |
| image_factor_strict_rank vs rank_baseline | +0.0097 | [-0.0029, +0.0223] | [-0.0270, +0.0462] | **NOT significant** |
| image_factor_balanced vs sharpe_baseline | +0.0125 | [+0.0005, +0.0245] | [-0.0250, +0.0500] | **NOT significant** |

## Interpretation

- The block CIs are wider than the IID CIs because the daily rank-corr
  diff series is positively autocorrelated.
- IID flags 2/3 lifts as significant; block flags 0/3.
- Honest framing: the image-factor lift over the ensemble baselines is a
  positive point estimate but is not robust to autocorrelation-honest
  confidence bounds. Report it as trend evidence, not strict significance.
- This does not overturn the handoff: the recommended input is still a
  reasonable ranking signal; it just should not be sold as a proven lift.