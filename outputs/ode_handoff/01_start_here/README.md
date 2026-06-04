# ODE Handoff Quick Start

This is a curated GitHub package, not the full internal experiment archive.

## Start Here

1. Read `../README_ODE_TEAM.md`.
2. Load `../02_mu_inputs/selected_mu_input.csv`.
3. Pivot `mu_signal` into a `date x asset` matrix.
4. Load `../03_sigma_returns/sigma_shrunk_wide.csv` for the recommended covariance input.
5. Load `../03_sigma_returns/returns_for_ode.csv` for realized return evaluation.

## Main Files

| File | Purpose |
|---|---|
| `../02_mu_inputs/selected_mu_input.csv` | recommended rank-scale `mu(t)` |
| `../02_mu_inputs/selected_mu_input_calibrated.csv` | return-scale calibrated `mu(t)` |
| `../02_mu_inputs/final_mu_inputs_wide.csv` | compact table of 5 main candidates |
| `../03_sigma_returns/sigma_shrunk_wide.csv` | recommended `Sigma(t)` |
| `../03_sigma_returns/sigma_wide.csv` | sample covariance comparison |
| `../04_validation_reports/process_integrity_report.md` | audit / caveats |

## Warning

`future_return` is included for validation only. It must not be used as an ODE optimizer input.
