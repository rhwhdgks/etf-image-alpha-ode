# Reproducibility

This document summarizes how to reproduce the public portfolio version of the project.

The original full internal artifact dump is intentionally excluded. Running the builders will regenerate large outputs under `ode_inputs_cnn/`, which is ignored by Git.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommended Python version: 3.12.x.

## Deterministic Settings

- Default seed: 42
- CNN / image-factor training uses `src.pipeline.set_global_seed`.
- Bootstrap utilities use fixed NumPy RNG seeds.
- OLS calibration and covariance shrinkage scripts are deterministic.

## Rebuild Map

| Step | Command |
|---|---|
| walk-forward model evaluation | `python run_walkforward.py --lookback 60 --horizon 20` |
| image factor extraction | `python build_image_factor_extension.py` |
| image component ablation | `python build_image_factor_ablation.py` |
| image robustness checks | `python build_image_factor_robustness.py` |
| final `mu(t)` package | `python build_final_ode_mu_inputs.py` |
| final sample covariance package | `python build_final_ode_sigma_inputs.py` |
| return-scale mu calibration | `python build_mu_calibration.py` |
| covariance shrinkage | `python build_sigma_shrunk.py` |
| covariance variants | `python build_sigma_variants.py` |
| block bootstrap robustness | `python build_image_factor_block_ci.py` |

## Public Artifacts

The GitHub version keeps a compact, reviewable subset under `outputs/ode_handoff/`.

| File | Purpose |
|---|---|
| `02_mu_inputs/selected_mu_input.csv` | recommended rank-scale `mu(t)` |
| `02_mu_inputs/selected_mu_input_calibrated.csv` | return-scale calibrated `mu(t)` |
| `02_mu_inputs/final_mu_inputs_wide.csv` | compact 5-candidate `mu(t)` table |
| `03_sigma_returns/sigma_shrunk_wide.csv` | recommended covariance input |
| `03_sigma_returns/sigma_wide.csv` | sample covariance comparison |
| `03_sigma_returns/returns_for_ode.csv` | realized returns |

## Limitations To Report

- The target uses overlapping 20-day returns, so autocorrelation-aware bootstrap is necessary.
- Image-factor gains are not uniformly strong across all regimes.
- `Sigma(t)` is a historical covariance estimate, not a CNN prediction.
- Negative transfer was observed when simply pretraining on 30 extra ETFs.
