# Sigma Ledoit-Wolf Shrinkage Report

## Method

- Per date: 60-day trailing log-return window -> Ledoit-Wolf shrinkage (sklearn).
- Shrinkage target: scaled identity; intensity alpha estimated analytically per window.
- No look-ahead: each date uses only its own trailing window.

## Shrinkage intensity

- alpha mean 0.080, median 0.073, min 0.027, max 0.328

## Condition number: sample vs shrunk

| stat | sample Sigma | shrunk Sigma |
|---|---:|---:|
| median | 2409 | 64 |
| p95 | 10644 | 142 |
| max | 18444 | 206 |
| dates cond > 1e4 | 176 | 0 |

## Interpretation

- Shrinkage lowers the condition number, so `Sigma^-1` is more stable for the
  constant-gamma ODE equilibrium `w* = (alpha/2 gamma_0) Sigma^-1 mu`.
- Use `sigma_shrunk_wide.csv` as the recommended Sigma input; `sigma_wide.csv`
  (sample covariance) is kept for comparison / ablation.
- Shrinkage trades a small bias for a large variance reduction in the inverse;
  the ODE team can A/B test both Sigma versions.