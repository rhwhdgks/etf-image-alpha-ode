# Reproducibility

This document is the single reference for reproducing every artifact in the
handoff package from the repo. Pair it with `01_start_here/manifest.json` (full
file inventory) and `04_validation_reports/process_integrity_report.md` (audit
summary).

## 1. Environment

- Python: **3.12.3**
- Dependencies pinned in repo-root `requirements.txt`
- Bootstrap:

  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```

Key packages (pinned):
`numpy 2.4.4` · `pandas 3.0.2` · `scipy 1.17.1` · `scikit-learn 1.8.0` ·
`torch 2.11.0` · `xgboost 3.2.0` · `matplotlib 3.10.8` · `seaborn 0.13.2` ·
`pillow 12.2.0`.

## 2. Seed conventions

Every stochastic step uses a fixed seed:

| script | seeding mechanism |
|---|---|
| CNN / image-factor training | `src.pipeline.set_global_seed(config.seed)` — seeds Python `random`, NumPy, PyTorch (CPU+CUDA) |
| `run_xgb_walkforward.py` | `set_global_seed(config.seed)` plus `XGBRegressor(random_state=config.seed + repeat_idx)` |
| `bootstrap_significance.py`, `build_image_factor_block_ci.py` | `np.random.default_rng(seed=42)` at module load |
| `build_mu_calibration.py`, `build_sigma_shrunk.py`, `build_sigma_variants.py`, `build_extra_mu_candidates.py` | deterministic (OLS / Ledoit-Wolf / weighted sums; no RNG) |

Default `config.seed = 42`.

## 3. End-to-end build chain

```
etfdata.csv  ──►  walk-forward outputs (archive/model_exploration/walkforward_outputs/)
                     │
                     ├─► build_image_factor_extension.py / _ablation.py / _robustness.py
                     │      └─► ode_inputs_cnn/image_factor_*  ──┐
                     │                                            │
                     ├─► collect_cnn_ode_signals.py               │
                     │      └─► ode_inputs_cnn/{model}/ode_bundle.csv
                     │                                            │
                     ├─► build_ensemble_best.py / build_ensemble_4family.py
                     │      └─► ensemble_best/ / ensemble_4family/
                     │                                            │
                     ├─► build_final_ode_mu_inputs.py             │
                     │      └─► ode_inputs_cnn/final_ode_mu_inputs/  (used by build_final_ode_sigma_inputs.py)
                     │                                            │
                     ├─► build_final_ode_sigma_inputs.py          │
                     │                                            ▼
                     │             handoff: 02_mu_inputs/{final_*}, 03_sigma_returns/{sigma_wide,long,returns_for_ode}
                     │
                     ├─► build_mu_calibration.py        ─► handoff: 02_mu_inputs/final_mu_inputs_calibrated.csv + mu_calibration_summary.csv
                     ├─► build_extra_mu_candidates.py   ─► handoff: 02_mu_inputs/extra_mu_candidates_*
                     ├─► build_sigma_shrunk.py          ─► handoff: 03_sigma_returns/sigma_shrunk_* + sigma_shrinkage_report.md
                     ├─► build_sigma_variants.py        ─► handoff: 03_sigma_returns/sigma_ewma_*, sigma_multiwindow_*, sigma_variants_report.md
                     ├─► bootstrap_significance.py      ─► handoff: significance_test.md (-> mirrored to ode_inputs_cnn/)
                     └─► build_image_factor_block_ci.py ─► handoff: 04_validation_reports/image_factor_block_bootstrap.md
```

## 4. Reproduce a specific artifact

| target | command |
|---|---|
| `02_mu_inputs/final_mu_inputs_calibrated.csv` | `python build_mu_calibration.py` |
| `02_mu_inputs/extra_mu_candidates_*` | `python build_extra_mu_candidates.py` |
| `03_sigma_returns/sigma_shrunk_*` + shrinkage report | `python build_sigma_shrunk.py` |
| `03_sigma_returns/sigma_{ewma,multiwindow}_*` + variants report | `python build_sigma_variants.py` |
| `04_validation_reports/image_factor_block_bootstrap.md` | `python build_image_factor_block_ci.py` |
| `ode_inputs_cnn/significance_test.md` + `figures/12_bootstrap_ci.png` | `python bootstrap_significance.py` |

The image-factor candidate inputs themselves (`final_mu_inputs_long.csv`, etc.)
are produced by the earlier image-factor sprint scripts under
`build_image_factor_*.py` + `build_final_ode_mu_inputs.py` — see
`05_source_notes/image_factor_phase12_final_report.md` for the upstream pipeline.

## 5. Determinism checks

- Re-running `bootstrap_significance.py` and `build_image_factor_block_ci.py`
  must produce bit-identical CSVs / numbers, because they fix the RNG seed.
- Deterministic scripts (`build_mu_calibration.py`, `build_sigma_shrunk.py`,
  `build_sigma_variants.py`, `build_extra_mu_candidates.py`) likewise reproduce
  bit-identically.
- The CNN / image-factor training pipeline reproduces up to PyTorch's
  documented CPU determinism (same OS / BLAS); minor floating-point drift
  across hardware can shift rank-corr by ~1e-4. Aggregate metrics in the
  handoff are robust to this.

## 6. Known limitations (do not pretend away)

See `manifest.json -> known_limitations` and
`04_validation_reports/process_integrity_report.md`:

- Walk-forward folds have no horizon embargo (documented, lift-symmetric).
- Original calibration script was missing; restored and re-run with horizon
  embargo (file replaced, manifest updated).
- IID bootstrap CIs in earlier reports overstate significance; honest block
  bootstrap CIs are provided alongside.
- 5th model family (XGBoost on raw input) was tested and excluded as a
  negative result; see `process_integrity_report.md` section 4b.
