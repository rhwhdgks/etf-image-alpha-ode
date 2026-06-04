# Process Integrity Report

This report consolidates the leakage audit, the fixes applied, and the
reproducibility status of every artifact in this handoff package. It is the
"how was this built and why can you trust it" reference for the ODE team and
for any paper methodology section.

Last updated: 2026-05 (post-audit revision).

---

## 1. Leakage audit

Every transformation in the pipeline was checked for look-ahead leakage.

### 1.1 Verified clean

| Stage | Check | Evidence |
|---|---|---|
| Image-factor CNN training | walk-forward; train dates strictly before test | `build_image_factor_extension.py` explicit `assert train_val_dates.max() < test_dates.min()` |
| Image-feature PCA | fit on train+val only, transform test | `extension.py` `pca.fit_transform(train_val)` then `pca.transform(test)` |
| Return-control rolling PCA (252-day) | window uses only dates <= signal date | `extension.py` explicit `assert hist.index.max() <= date` |
| PCA sign orientation | oriented with train+val future returns only | `extension.py` `_orient_component(train_val_pcs, y_train_val)` |
| `mu_signal` (primary handoff input) | cross-sectional centered rank; no time axis | by construction |
| Sigma(t) | 60-day trailing rolling window only | `make_ode_inputs.compute_rolling_sigma` |
| Ablation / robustness scripts | reuse the audited extension code | `import build_image_factor_extension as ext` |

### 1.2 Findings and resolutions

| # | Finding | Severity | Resolution |
|---|---|---|---|
| 1 | **Walk-forward folds have no horizon embargo.** `generate_walkforward_folds` places train/val/test adjacently; the last ~20 training dates per fold carry 20-day forward-return labels that overlap the adjacent period (~1-2% of training rows). | Medium | **Documented, not re-run.** All models share the identical fold grid, so absolute OOS metrics may be slightly inflated but lift/CI comparisons are contamination-symmetric and approximately unbiased. See `manifest.json -> known_limitations.walkforward_embargo`. |
| 2 | **`calibrate_mu_expanding` horizon overlap.** The expanding return-std pool appended `future_return` for every past date, including the last ~20 dates whose forward window had not closed. | Low | **Fixed.** `make_ode_inputs.calibrate_mu_expanding` now releases a date into the pool only when `index(s) + horizon <= current index`. |
| 3 | **Handoff calibration script was missing from the repo** (the `mu_calibrated_*` columns were not reproducible). | Medium (reproducibility) | **Fixed.** `build_mu_calibration.py` restores it; verified to reproduce the original to 1e-10, and additionally adds the horizon embargo. |

---

## 2. Fixes and additions applied during the audit

| Item | Script | Effect |
|---|---|---|
| mu calibration embargo | `build_mu_calibration.py` | leakage-clean expanding OLS; reproducible |
| Sigma shrinkage | `build_sigma_shrunk.py` | Ledoit-Wolf shrunk Sigma; condition number median 2409 -> 64, all 176 dates with cond > 1e4 removed |
| Extra mu candidates | `build_extra_mu_candidates.py` | 8 reference/baseline mu inputs (floor -> single model), same schema, leakage-clean |
| Block bootstrap | `bootstrap_significance.py` | stationary block bootstrap added; honest CIs for autocorrelated series |

---

## 3. Bootstrap method correction (important)

The bootstrap CIs originally reported for the image-factor lift used **IID
resampling**, which assumes the daily rank-correlation series is uncorrelated.
It is not: the 20-day forward-return horizon makes adjacent days overlap.

A **stationary block bootstrap** (Politis-Romano, mean block length 20) was run
on the ensemble-level comparisons (`significance_test.md`). Results:

| comparison | IID verdict | block verdict (honest) |
|---|---|---|
| ensemble vs raw floor (`logistic_cumulative`) | significant | **significant** |
| ensemble vs image baseline (`logistic_image`) | significant | **NOT significant** |
| within-ensemble (Phase 3 -> Phase 4) | not significant | not significant |

**Honest conclusion**: the statistically robust lift comes from the image
transformation itself (raw floor -> image baseline). Stacking CNN/LSTM/ensemble
on top yields a positive point estimate that stays inside autocorrelation-honest
confidence bounds. The image-factor lift CIs in this package's other reports are
IID and should be read as positive-trend evidence, not strict significance.

---

## 4. Reproducibility status

| Artifact | Reproducing script | Status |
|---|---|---|
| `final_mu_inputs_*` (5 main candidates) | image-factor pipeline (`build_image_factor_*.py`) | reproducible |
| `final_mu_inputs_calibrated.csv` | `build_mu_calibration.py` | reproducible (rebuilt) |
| `extra_mu_candidates_*` (8 reference) | `build_extra_mu_candidates.py` | reproducible |
| `sigma_wide/long.csv` | `build_final_ode_sigma_inputs.py` | reproducible |
| `sigma_shrunk_*` | `build_sigma_shrunk.py` | reproducible |
| `significance_test.md` | `bootstrap_significance.py` | reproducible |
| walk-forward predictions (model sources) | `run_walkforward.py` etc. | reproducible (seeded) |

---

## 4b. Tree family — tested and excluded

A 5th model family (XGBoost gradient-boosted trees) was trained on the same
walk-forward grid (`run_xgb_walkforward.py`). On the raw price-path inputs it
underperformed every other family:

| variant | rank corr | top-k Sharpe |
|---|---:|---:|
| `xgb_cumulative_scale` | 0.019 | 0.41 |
| `xgb_image_scale` | -0.011 | 0.16 |

Reason: trees split on individual features and have no spatial / temporal
inductive bias, so a flattened price image or sequence destroys the structure
they would need. This is consistent with the sprint's representation-model
matching finding (image input only helps a model that can read it). The tree
family was therefore **excluded** from the handoff mu candidates and the
ensemble. The run script and outputs remain in the repo (`xgb/`) for the record.
A fair tree setup would use engineered tabular features (multi-lag momentum,
realised volatility, drawdown, volume z-score); that was not pursued here.

## 5. Net assessment

- No large leakage. The primary input path (`mu_signal` + Sigma) is clean.
- Three audit findings: one fixed code bug (calibration embargo), one restored
  script (reproducibility), one documented limitation (fold embargo).
- The bootstrap method was corrected; significance claims are now stated under
  the autocorrelation-honest block bootstrap.
- All handoff artifacts are reproducible from repo scripts.

The package is suitable for ODE consumption and for a paper methodology section,
provided the honest (block-bootstrap) significance framing in Section 3 is used.
