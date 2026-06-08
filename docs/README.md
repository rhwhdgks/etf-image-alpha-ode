# Documentation Index

This folder contains the research notes, validation reports, and figures that support the public portfolio version of the ETF image-factor project.

## Suggested Review Path

1. Start with `reports/paper_draft_ko.md` for the full Korean research narrative.
2. Read `reports/process_integrity_report.md` to check leakage, bootstrap, and reproducibility limitations.
3. Read `reports/image_factor_phase12_final_report.md` for the Jiang-style image-factor extension and 6-way chart ablation.
4. Read `reports/mu_input_validation_audit_ko.md`, `reports/mu_calibration_report.md`, and `../outputs/ode_handoff/06_mu_submission_validation/mu_submission_validation_report.md` to understand whether the prepared `mu(t)` inputs are defensible for ODE use.
5. Use the figures in `figures/` for presentation or interview explanation.

## Core Reports

| File | Purpose |
|---|---|
| `reports/paper_draft_ko.md` | paper-style Korean research narrative |
| `reports/cnn_lstm_research_story.md` | project story from CNN/LSTM modeling to ensemble construction |
| `reports/reproducibility.md` | reproducibility and build-chain notes |
| `reports/process_integrity_report.md` | leakage, bootstrap, and process audit |
| `reports/mu_input_validation_audit_ko.md` | ODE용 `mu(t)` 입력의 overfit/underfit/leakage 감사 |
| `reports/image_factor_phase12_final_report.md` | image-factor extraction and chart-component ablation |
| `reports/multiple_testing_report.md` | multiple-testing correction summary |
| `reports/image_factor_block_bootstrap.md` | autocorrelation-aware bootstrap note |
| `reports/mu_subperiod_stability_report.md` | regime/subperiod stability |
| `reports/mu_calibration_report.md` | rank-to-return-scale `mu(t)` calibration |
| `reports/extra_pretraining_result_note.md` | negative result from 30-ETF supervised pretraining |
| `reports/strict_window_ma_result_note.md` | strict-window moving-average robustness check |

## ODE Handoff Validation Workspace

| Folder | Purpose |
|---|---|
| `../outputs/ode_handoff/06_mu_submission_validation/` | locked-candidate metrics, fold-boundary purged sensitivity, block-bootstrap lift, calibration sanity, purged retraining smoke |

## Figures

| File | Purpose |
|---|---|
| `figures/jiang_sample_image.png` | example price-chart image input |
| `figures/model_comparison.png` | model comparison summary |
| `figures/ablation_image_vs_cnn.png` | image transformation vs CNN ablation |
| `figures/model_correlation_matrix.png` | model signal correlation structure |
| `figures/2d_cnn_loss_curves.png` | 2D CNN undertraining diagnosis |
| `figures/ensemble_lift_progression.png` | ensemble lift progression |
| `figures/bootstrap_ci.png` | bootstrap CI visualization |

## Literature Map

The source paper PDFs are not kept in the public repository. See `literature/README.md` for the reference map:

- Jiang-style price images.
- Image-based asset pricing / factor interpretation.
- ODE-based dynamic mean-variance optimization.

## Positioning Note

The documents should be read as research and risk-input evidence. They do not describe a production LP/MM execution stack; execution, quoting, venue routing, fill modeling, and inventory constraints are outside this repository.
