# Reproducibility

This file describes the curated GitHub package. The full internal experiment archive is not included.

## Environment

- Python: 3.12.x
- Dependencies: repo-root `requirements.txt`
- Default seed: 42

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main Rebuild Commands

| Target | Command |
|---|---|
| walk-forward baseline predictions | `python run_walkforward.py --lookback 60 --horizon 20` |
| image factor extension | `python build_image_factor_extension.py` |
| 6-way image factor ablation | `python build_image_factor_ablation.py` |
| robustness checks | `python build_image_factor_robustness.py` |
| final `mu(t)` files | `python build_final_ode_mu_inputs.py` |
| final sample covariance files | `python build_final_ode_sigma_inputs.py` |
| calibrated selected `mu(t)` | `python build_mu_calibration.py` |
| shrunk covariance | `python build_sigma_shrunk.py` |
| EWMA / multi-window covariance variants | `python build_sigma_variants.py` |
| block bootstrap note | `python build_image_factor_block_ci.py` |

Most scripts write full generated artifacts to `ode_inputs_cnn/`, which is ignored in the public repo. The curated review files are copied into `outputs/ode_handoff/`.

## Curated Files

This package keeps compact wide-format inputs and selected recommended signals:

- `02_mu_inputs/selected_mu_input.csv`
- `02_mu_inputs/selected_mu_input_calibrated.csv`
- `02_mu_inputs/final_mu_inputs_wide.csv`
- `03_sigma_returns/sigma_shrunk_wide.csv`
- `03_sigma_returns/sigma_wide.csv`
- `03_sigma_returns/returns_for_ode.csv`

Full fold-level predictions, model checkpoints, and long-format covariance matrices are excluded.

## Known Limitations

- The 20-day forward target induces autocorrelation; block bootstrap results should be treated as more honest than IID bootstrap intervals.
- `future_return` is included only for validation and must not be used as an optimizer input.
- Image-factor improvement is regime-sensitive and weaker in 2020-2022.
- Extra ETF supervised pretraining was tested and performed worse; it is documented as a negative robustness result.
