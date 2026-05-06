# Bootstrap Significance Test

Paired bootstrap (B=10000, 95% CI) on per-date rank correlation series.

Positive `mean_diff` = B is better than A on average.

## Pair 1 — ensemble lift (Phase 3 → Phase 4)

- A: `ensemble_best (3)` — mean rank corr 0.0606
- B: `ensemble_4family (4)` — mean rank corr 0.0674
- N dates: 2880
- mean diff (B − A): **+0.0068**
- 95% CI: [-0.0024, +0.0157]
- Verdict: **NOT significant** (CI includes 0)

## Pair 2 — single-model best (CNN vs LSTM)

- A: `cnn_2d_residual_small` — mean rank corr 0.0426
- B: `cnnlstm_image_scale` — mean rank corr 0.0448
- N dates: 2880
- mean diff (B − A): **+0.0022**
- 95% CI: [-0.0201, +0.0245]
- Verdict: **NOT significant** (CI includes 0)
