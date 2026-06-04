# Outputs

This folder contains the curated, GitHub-safe subset of final research artifacts.

The full experimental dump was intentionally removed. The retained files are enough to inspect the ETF alpha signals, review the covariance/risk handoff, and run downstream ODE optimizer prototypes without carrying every fold prediction or model checkpoint.

## Main Package

`ode_handoff/`

Contains:

- selected rank-scale `mu(t)` input
- selected return-scale calibrated `mu(t)` input
- compact wide-format `mu(t)` candidate tables
- Ledoit-Wolf shrunk covariance `Sigma(t)`
- comparison covariance variants
- realized returns and close prices
- validation reports and source notes

## Recommended Files

| File | Use |
|---|---|
| `ode_handoff/02_mu_inputs/selected_mu_input.csv` | primary rank-scale `mu(t)` input |
| `ode_handoff/02_mu_inputs/selected_mu_input_calibrated.csv` | return-scale calibrated version of the primary `mu(t)` |
| `ode_handoff/02_mu_inputs/final_mu_inputs_wide.csv` | 5 main `mu(t)` candidates for A/B tests |
| `ode_handoff/03_sigma_returns/sigma_shrunk_wide.csv` | default shrunk `Sigma(t)` for ODE |
| `ode_handoff/03_sigma_returns/sigma_wide.csv` | plain sample covariance comparison |
| `ode_handoff/03_sigma_returns/returns_for_ode.csv` | realized returns for backtesting |
| `ode_handoff/04_validation_reports/process_integrity_report.md` | leakage / bootstrap / reproducibility audit |
| `ode_handoff/03_sigma_returns/sigma_shrinkage_report.md` | covariance shrinkage diagnostics |

## Why Shrunk Sigma Is Included

ODE and mean-variance-style optimizers are sensitive to `Sigma(t)^-1`. The 60-day sample covariance had median condition number 2409 and 176 dates above `1e4`; Ledoit-Wolf shrinkage reduced the median to 64 and removed all dates above `1e4`. For optimizer testing, `sigma_shrunk_wide.csv` should be the default and `sigma_wide.csv` should be treated as an ablation.

## Excluded From GitHub

- full per-fold predictions
- long-format covariance matrices
- model checkpoints
- raw paper PDFs
- old presentation exports
- local virtual environments

The excluded files are not required to understand or consume the final ODE handoff package.
