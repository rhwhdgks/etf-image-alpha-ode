# Bootstrap Significance Test

Paired bootstrap (B=10000, 95% CI) on per-date rank correlation series.

Two methods reported:

- **IID**: classic resampling — assumes daily diffs are uncorrelated.
- **Block**: stationary block bootstrap (Politis-Romano, mean block 20) — accounts for autocorrelation from the 20-day overlapping return horizon. This is the honest CI; it is usually wider.

Positive `mean_diff` = B is better than A on average.

## Pair 1 — ensemble lift (Phase 3 → Phase 4)

- A: `ensemble_best (3)` — mean rank corr 0.0606
- B: `ensemble_4family (4)` — mean rank corr 0.0674
- N dates: 2880
- mean diff (B − A): **+0.0068**
- IID 95% CI: [-0.0024, +0.0157] → **NOT significant**
- Block 95% CI: [-0.0153, +0.0300] → **NOT significant**  ← honest

## Pair 2 — single-model best (CNN vs LSTM)

- A: `cnn_2d_residual_small` — mean rank corr 0.0426
- B: `cnnlstm_image_scale` — mean rank corr 0.0448
- N dates: 2880
- mean diff (B − A): **+0.0022**
- IID 95% CI: [-0.0202, +0.0249] → **NOT significant**
- Block 95% CI: [-0.0611, +0.0686] → **NOT significant**  ← honest

## Pair 3 — logistic_image vs ensemble_best (Phase 3 lift)

- A: `logistic_image_scale` — mean rank corr 0.0392
- B: `ensemble_best (3)` — mean rank corr 0.0606
- N dates: 2880
- mean diff (B − A): **+0.0213**
- IID 95% CI: [+0.0072, +0.0353] → **significant**
- Block 95% CI: [-0.0148, +0.0585] → **NOT significant**  ← honest

## Pair 4 — logistic_image vs ensemble_4family (Phase 4 lift)

- A: `logistic_image_scale` — mean rank corr 0.0392
- B: `ensemble_4family (4)` — mean rank corr 0.0674
- N dates: 2880
- mean diff (B − A): **+0.0281**
- IID 95% CI: [+0.0128, +0.0435] → **significant**
- Block 95% CI: [-0.0119, +0.0680] → **NOT significant**  ← honest

## Pair 5 — logistic_cumulative (worst) vs ensemble_4family

- A: `logistic_cumulative_scale` — mean rank corr -0.0072
- B: `ensemble_4family (4)` — mean rank corr 0.0674
- N dates: 2880
- mean diff (B − A): **+0.0746**
- IID 95% CI: [+0.0532, +0.0964] → **significant**
- Block 95% CI: [+0.0126, +0.1375] → **significant**  ← honest

## Interpretation

The block bootstrap is the methodologically correct choice here: the per-date rank-correlation series is autocorrelated because the 20-day forward-return horizon makes adjacent days overlap. IID resampling ignores this and produces CIs that are too narrow.

Honest (block-bootstrap) conclusions:

- **ensemble vs raw floor** (`logistic_cumulative`, no image / no model): lift +0.075, block CI excludes 0 → **significant**.
- **ensemble vs image baseline** (`logistic_image`): lift +0.021 / +0.028, block CI **includes 0** → not statistically significant. The IID CI flagged these as significant, but that was an autocorrelation artifact.
- **within-ensemble** (Phase 3 → Phase 4): not significant under either method.

This re-confirms the honest 'same-tier' framing: the statistically robust lift comes from the **image transformation itself** (raw floor → image baseline); stacking CNN/LSTM/ensemble on top adds a positive point estimate that stays inside autocorrelation-honest confidence bounds.
