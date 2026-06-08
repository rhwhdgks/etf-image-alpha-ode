# ODE Team Handoff Package

This folder is the curated ODE-input package from the ETF image-factor sprint.

It contains expected-return-style `mu(t)` candidates, covariance `Sigma(t)` variants, realized returns, and validation reports. The package is designed for downstream ODE portfolio optimization experiments, not for production market-making execution.

## Recommended Inputs

| Input | File | Column / schema |
|---|---|---|
| Rank-scale `mu(t)` | `02_mu_inputs/selected_mu_input.csv` | `mu_signal` |
| Return-scale `mu(t)` | `02_mu_inputs/selected_mu_input_calibrated.csv` | `mu_calibrated_daily` |
| Default `Sigma(t)` | `03_sigma_returns/sigma_shrunk_wide.csv` | 7 variances + 21 covariances |
| Comparison `Sigma(t)` | `03_sigma_returns/sigma_wide.csv` | 7 variances + 21 covariances |
| Realized returns | `03_sigma_returns/returns_for_ode.csv` | 7 asset return columns |

Default experiment:

- `mu`: start with `selected_mu_input.csv` / `mu_signal` for rank-based allocation tests.
- `mu` return-scale alternative: test `selected_mu_input_calibrated.csv` / `mu_calibrated_daily` if the ODE solver requires expected returns in daily return units.
- `Sigma`: start with `sigma_shrunk_wide.csv`.
- Benchmark `Sigma`: compare against `sigma_wide.csv`.

## Main Performance Snapshot

| Candidate | Rank Corr | Top-k Sharpe | Role |
|---|---:|---:|---|
| `mu_rank_baseline` | 0.0674 | 0.5430 | 4-family baseline before image-factor add-on |
| `mu_image_factor_rank` | 0.0798 | 0.3155 | main image-factor ranking input |
| `mu_image_factor_strict_rank` | 0.0771 | 0.6053 | strict-window MA robustness input |
| `mu_sharpe_baseline` | 0.0606 | 0.6425 | Sharpe-priority comparison |

Important interpretation:

- `mu_image_factor_rank` is the recommended primary signal because it has the highest cross-sectional rank correlation.
- `mu_image_factor_strict_rank` is a conservative robustness candidate because the moving-average line is recomputed only inside each 60-day image window.
- `mu_sharpe_baseline` is useful as a comparison because simple top-k Sharpe is higher, even though rank quality is lower.
- These metrics are signal diagnostics, not final portfolio PnL. The ODE optimizer should compare weight paths, realized returns, turnover, drawdown, and sensitivity to `Sigma`.

## Files Kept In This Package

| File | Purpose |
|---|---|
| `01_start_here/` | Korean handoff summary and machine-readable manifest |
| `02_mu_inputs/selected_mu_input.csv` | recommended `mu_image_factor_rank` only |
| `02_mu_inputs/selected_mu_input_calibrated.csv` | calibrated recommended input only |
| `02_mu_inputs/final_mu_inputs_wide.csv` | 5 main `mu(t)` candidates in compact wide format |
| `02_mu_inputs/extra_mu_candidates_wide.csv` | 8 reference/baseline candidates in compact wide format |
| `02_mu_inputs/input_performance_summary.csv` | main candidate metrics |
| `03_sigma_returns/sigma_shrunk_wide.csv` | Ledoit-Wolf shrunk covariance, recommended |
| `03_sigma_returns/sigma_wide.csv` | 60-day sample covariance, comparison baseline |
| `03_sigma_returns/sigma_ewma_wide.csv` | EWMA covariance variant |
| `03_sigma_returns/sigma_multiwindow_wide.csv` | multi-window covariance variant |
| `03_sigma_returns/returns_for_ode.csv` | realized ETF returns |
| `04_validation_reports/` | validation and robustness reports |
| `04_validation_reports/mu_input_validation_audit_ko.md` | Korean audit for `mu(t)` leakage, overfit/underfit, calibration, and remaining paper risks |
| `05_source_notes/` | source notes for image factor and negative pretraining result |
| `06_mu_submission_validation/` | locked-candidate and fold-boundary purged validation workspace |

Full long-format CSVs, model checkpoints, fold-level predictions, and paper PDFs are excluded from the GitHub version.

## Minimal Loading Example

```python
from pathlib import Path
import pandas as pd

root = Path("outputs/ode_handoff")

mu = pd.read_csv(root / "02_mu_inputs/selected_mu_input.csv", parse_dates=["date"])
mu_wide = mu.pivot(index="date", columns="asset", values="mu_signal")

mu_cal = pd.read_csv(root / "02_mu_inputs/selected_mu_input_calibrated.csv", parse_dates=["date"])
mu_cal_wide = mu_cal.pivot(index="date", columns="asset", values="mu_calibrated_daily")

sigma = pd.read_csv(root / "03_sigma_returns/sigma_shrunk_wide.csv", parse_dates=["date"])
returns = pd.read_csv(root / "03_sigma_returns/returns_for_ode.csv", parse_dates=["date"])
```

## Process Integrity Notes

- `future_return` is an evaluation target only. Do not use it as an optimizer input.
- Signal dates are walk-forward out-of-sample dates.
- Rolling PCA controls and covariance estimates use trailing windows only.
- The 20-day prediction horizon creates overlapping labels, so block-bootstrap caveats should be used for significance claims.
- `06_mu_submission_validation/` drops the first 20 OOS dates of every 60-day test fold as a boundary-purge sensitivity check. The image-factor rank signal remains directionally positive, but block-bootstrap CIs still include zero.
- The code now supports purged retraining via `--wf-embargo-days 20`; a 1-fold/1-epoch smoke test passed under `06_mu_submission_validation/purged_retraining_smoke/`.
- Extra ETF supervised pretraining was tested and excluded because it underperformed the clean 7-ETF setup.

## ODE Integration Checklist

- Confirm asset order before building matrix inputs.
- Choose whether the ODE solver expects rank-scale `mu_signal` or return-scale `mu_calibrated_daily`.
- Start with `sigma_shrunk_wide.csv` for numerical stability.
- Keep `sigma_wide.csv`, `sigma_ewma_wide.csv`, and `sigma_multiwindow_wide.csv` for sensitivity tests.
- Exclude `future_return` from optimizer inputs and reserve it for realized-performance evaluation.
