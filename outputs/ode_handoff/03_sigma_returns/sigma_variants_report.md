# Sigma Variants — condition-number comparison

Four Sigma(t) variants are available in this handoff:

| variant | file | description |
|---|---|---|
| sample | `sigma_wide.csv` | plain 60-day rolling sample covariance |
| shrunk | `sigma_shrunk_wide.csv` | Ledoit-Wolf shrinkage per window |
| ewma | `sigma_ewma_wide.csv` | exponentially weighted (lambda=0.94, truncated at 252) |
| multiwindow | `sigma_multiwindow_wide.csv` | blend of 60-day and 120-day sample covs (0.5 / 0.5) |

## Condition number distribution

| variant | median | p95 | max | dates cond > 1e4 |
|---|---:|---:|---:|---:|
| sample (60d) | 2409 | 10643 | 18443 | 176 |
| shrunk (LW) | 64 | 142 | 205 | 0 |
| ewma (lambda=0.94) | 2588 | 10901 | 29436 | 220 |
| multiwindow (60d+120d) | 2227 | 9884 | 15025 | 138 |

## Interpretation

- **shrunk** is the most numerically stable choice (smallest condition number).
  Recommended default when the constant-gamma ODE relies on `Sigma^-1 mu`.
- **ewma** gives the recent regime more weight (effective lookback ~1/(1-lambda) ~ 17 days).
  More adaptive to vol regime changes but noisier than rolling.
- **multiwindow** averages a fast 60-day estimate with a slow 120-day estimate;
  trades a small lag for less noise than 60-day alone.
- **sample** kept for A/B comparison with the original handoff.