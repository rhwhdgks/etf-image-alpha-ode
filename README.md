# Image-Based ETF Alpha Signals for ODE Portfolio Inputs

Research and engineering project for building cross-sectional ETF alpha signals from price-chart images, then packaging those signals with covariance estimates for a downstream ODE-based dynamic portfolio optimizer.

This repository is positioned as a finance research / risk-input project for LP/MM-adjacent roles. It is not a market-making execution engine: there is no quoting logic, fill simulator, venue routing, or inventory controller. The focus is the layer before that: signal construction, risk/covariance handoff, and process integrity.

## What This Project Demonstrates

- Cross-sectional ETF alpha research using 7 asset-class ETFs.
- Jiang-style OHLCV chart rendering and CNN image-factor extraction.
- Signal comparison across logistic, 1D CNN, 2D CNN, LSTM, and CNN+LSTM hybrid model families.
- Image-factor validation after rolling PCA common-factor controls.
- ODE-ready handoff files for expected-return-style `mu(t)`, covariance `Sigma(t)`, and realized returns `R(t)`.
- Covariance shrinkage diagnostics showing why sample covariance can be unstable before optimization.
- Leakage checks, block-bootstrap caveats, multiple-testing notes, and negative-result documentation.

## Research Scope

| Item | Setting |
|---|---|
| Universe | 7 ETF asset classes |
| Target | 20-day forward return ranking |
| Main lookback | 60 trading days |
| Core representation | OHLCV price chart image |
| Main image-factor model | 2D residual CNN feature / score |
| Evaluation | walk-forward out-of-sample |
| Downstream use | ODE portfolio optimizer input package |

The project asks whether price-path image information can improve ETF ranking signals that later become optimizer inputs. It does not claim production trading profitability.

## Key Results

| Signal | Interpretation | Rank Corr | Top-k Sharpe |
|---|---|---:|---:|
| `logistic_cumulative_scale` | raw no-image floor | -0.0072 | 0.0764 |
| `logistic_image_scale` | linear model with image-style scaling | 0.0392 | 0.3850 |
| `ensemble_4family` | logistic + CNN + LSTM/CNNLSTM baseline | 0.0674 | 0.5430 |
| `mu_image_factor_rank` | image-factor-enhanced ODE `mu(t)` candidate | 0.0798 | 0.3155 |
| `mu_image_factor_strict_rank` | strict-window MA robustness candidate | 0.0771 | 0.6053 |

Interpretation:

- The biggest robust lift comes from image-style representation versus the raw no-image baseline.
- CNN/LSTM models did not simply dominate simpler models; the useful gain came from combining low-correlation model families.
- The image-factor add-on improved cross-sectional rank quality, but standalone top-k Sharpe was not always higher. That is why the final deliverable is an optimizer input package rather than a standalone trading rule.
- Ledoit-Wolf shrinkage is recommended for `Sigma(t)` because it reduced the median covariance condition number from 2409 to 64 and removed all dates with condition number above `1e4`.

## Repository Layout

```text
.
├── src/                         # data, features, image rendering, models, evaluation
├── docs/                        # research reports, validation notes, selected figures
├── outputs/ode_handoff/          # curated GitHub-safe ODE input package
├── etfdata.csv                  # 7-ETF OHLCV source data
├── run_walkforward.py           # walk-forward model evaluation entry point
├── build_image_factor_*.py      # image factor extraction / ablation / robustness
├── build_final_ode_*.py         # final mu and Sigma handoff builders
├── build_mu_calibration.py      # expanding rank-to-return-scale mu calibration
├── build_sigma_*.py             # covariance shrinkage and variant builders
└── requirements.txt
```

Large intermediate fold predictions, model checkpoints, source paper PDFs, old presentation exports, and local virtual environments were intentionally removed from the public-facing project structure.

## Main Output Files

| File | Purpose |
|---|---|
| `outputs/ode_handoff/02_mu_inputs/selected_mu_input.csv` | recommended rank-scale `mu(t)` candidate |
| `outputs/ode_handoff/02_mu_inputs/selected_mu_input_calibrated.csv` | same signal with expanding return-scale calibration |
| `outputs/ode_handoff/02_mu_inputs/final_mu_inputs_wide.csv` | 5 main `mu(t)` candidates in compact wide format |
| `outputs/ode_handoff/02_mu_inputs/input_performance_summary.csv` | rank correlation, Sharpe, hit rate, turnover summary |
| `outputs/ode_handoff/03_sigma_returns/sigma_shrunk_wide.csv` | recommended Ledoit-Wolf shrunk covariance input |
| `outputs/ode_handoff/03_sigma_returns/sigma_wide.csv` | plain 60-day rolling sample covariance for comparison |
| `outputs/ode_handoff/03_sigma_returns/returns_for_ode.csv` | realized returns for downstream backtests |
| `docs/reports/process_integrity_report.md` | leakage, bootstrap, and reproducibility audit |
| `docs/reports/paper_draft_ko.md` | Korean paper-style research draft |

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Baseline walk-forward model run
python run_walkforward.py --lookback 60 --horizon 20

# Image factor ablation
python build_image_factor_ablation.py

# Final ODE input builders
python build_final_ode_mu_inputs.py
python build_final_ode_sigma_inputs.py
python build_mu_calibration.py
python build_sigma_shrunk.py
python build_sigma_variants.py
```

Full deep-learning walk-forward runs are compute-heavy. For review, start with `docs/README.md` and the curated CSVs under `outputs/ode_handoff/`.

## How This Connects To ODE

The downstream ODE optimizer needs three time-indexed inputs:

- `mu(t)`: expected-return-style signal by date and asset.
- `Sigma(t)`: covariance matrix by date.
- `R(t)`: realized returns for backtest and evaluation.

This repository provides all three:

- Use `mu_signal` in `selected_mu_input.csv` for rank-scale ODE experiments.
- Use `mu_calibrated_daily` in `selected_mu_input_calibrated.csv` when the ODE implementation needs daily return-scale expected returns.
- Use `sigma_shrunk_wide.csv` as the default covariance input and compare it against `sigma_wide.csv`.

## Process Integrity

- Walk-forward training uses chronological splits.
- Image-feature PCA is fit on train/validation data and only transformed on test data.
- Rolling PCA controls and covariance estimates use trailing windows only.
- `future_return` columns are evaluation targets and must not be used as optimizer inputs.
- Overlapping 20-day targets make IID bootstrap too optimistic, so block-bootstrap caveats are documented.
- Extra ETF supervised pretraining was a negative result and is kept as robustness evidence, not as a recommended input.

This project is for research and portfolio-engineering demonstration only. It is not investment advice.
