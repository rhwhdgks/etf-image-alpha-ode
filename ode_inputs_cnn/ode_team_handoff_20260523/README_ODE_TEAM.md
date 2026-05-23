# ODE Team Handoff Package

## 목적

이 패키지는 이미지 팩터/CNN-LSTM 실험에서 만든 ODE 포트폴리오 최적화용 입력을 정리한 전달본입니다.

Optimization/ODE solver 실행 결과는 포함하지 않습니다. 여기에는 ODE 코드가 바로 읽을 수 있는 `mu(t)`, `Sigma(t)`, realized return, 검증 보고서만 포함합니다.

## 최신 상태

| 항목 | 값 |
|---|---|
| 패키지 기준일 | 2026-05-19 |
| 메인 추천 `mu(t)` | `mu_image_factor_rank` |
| 보수적 robustness `mu(t)` | `mu_image_factor_strict_rank` |
| 기본 `Sigma(t)` | **`sigma_shrunk_wide.csv`** — Ledoit-Wolf shrunk (cond median 2409→64) |
| 비교용 `Sigma(t)` | `sigma_wide.csv` — 60-day rolling sample covariance (A/B test 용) |
| OOS 기간 | 2014-09-24 ~ 2026-03-09 |
| OOS dates | 2,880 |
| 자산 수 | 7 |
| `mu` 후보 수 | 5 |
| 주의 | `future_return`은 검증용 target이며 ODE input으로 사용 금지 |

## 먼저 볼 파일
1. `01_start_here/handoff_summary_ko.md`
2. `01_start_here/README.md`
3. `02_mu_inputs/main_vs_strict_mu_comparison.csv`
4. `02_mu_inputs/input_performance_summary.csv`

## 추천 실험 순서

1. Baseline: `mu_rank_baseline`의 `mu_signal`
2. Main image factor: `mu_image_factor_rank`의 `mu_signal`
3. Conservative image factor: `mu_image_factor_strict_rank`의 `mu_signal`
4. Return-scale ODE input: `mu_image_factor_strict_rank`의 `mu_calibrated_daily`
5. Sigma input: **`03_sigma_returns/sigma_shrunk_wide.csv`** (권장) — 비교용으로 `sigma_wide.csv` 도 같이 돌려보기
6. Sigma A/B: shrunk vs sample 두 버전으로 같은 ODE 돌려 weight trajectory 안정성 비교

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `01_start_here/` | ODE 팀용 설명서, manifest |
| `02_mu_inputs/` | mu 후보 입력 파일 |
| `03_sigma_returns/` | Sigma(t), realized returns |
| `04_validation_reports/` | multiple testing, subperiod, calibration 검증 |
| `05_source_notes/` | image factor, strict MA, extra ETF negative result 메모 |

## 핵심 입력 파일

| 파일 | 용도 |
|---|---|
| `02_mu_inputs/final_mu_inputs_long.csv` | 5개 메인 mu 후보 (image-factor 연구), long format |
| `02_mu_inputs/final_mu_inputs_wide.csv` | 5개 메인 mu 후보, wide format |
| `02_mu_inputs/final_mu_inputs_calibrated.csv` | return-scale calibrated mu |
| `02_mu_inputs/selected_mu_input.csv` | 현재 메인 추천 input 하나 |
| `02_mu_inputs/extra_mu_candidates_long.csv` | **8개 reference/baseline mu 후보** (floor~single model), long |
| `02_mu_inputs/extra_mu_candidates_wide.csv` | 8개 reference 후보, wide format |
| `03_sigma_returns/sigma_shrunk_wide.csv` | **권장 Sigma** — Ledoit-Wolf shrunk |
| `03_sigma_returns/sigma_wide.csv` | 비교용 Sigma — 60일 rolling sample covariance |
| `03_sigma_returns/sigma_long.csv` | 날짜별 full covariance matrix, long format |
| `03_sigma_returns/returns_for_ode.csv` | realized daily log returns |
| `03_sigma_returns/prices_for_ode.csv` | daily close prices (transaction-cost / turnover 시뮬레이션용) |

## 핵심 성능 요약

| input | source | rank corr | Sharpe | 해석 |
|---|---|---:|---:|---|
| `mu_rank_baseline` | `ensemble_4family` | 0.0674 | 0.5430 | image factor 추가 전 기준점 |
| `mu_image_factor_rank` | `ensemble_4family+image_factor_pc1` | 0.0798 | 0.3155 | rank corr 최고, 메인 추천 |
| `mu_image_factor_strict_rank` | strict-window `ensemble_4family+image_factor_pc1` | 0.0771 | 0.6053 | MA leakage 우려를 줄인 보수적 후보 |
| `mu_sharpe_baseline` | `ensemble_best` | 0.0606 | 0.6425 | Sharpe 안정형 비교군 |

해석은 명확히 제한합니다. `mu_image_factor_rank`는 ranking 성능을 올리지만 단순 top-k Sharpe는 낮습니다. 따라서 이 신호는 독립 trading rule이라기보다 ODE optimizer 안에서 `mu(t)` 후보로 비교해야 합니다.

## Mu 후보 요약

### 메인 후보 5개 (image-factor 연구, `final_mu_inputs_long.csv`)

| input | 의미 |
|---|---|
| `mu_rank_baseline` | image factor 추가 전 4-family ensemble baseline |
| `mu_image_factor_rank` | full OHLC+MA+volume image factor 추가, rank corr 최고 |
| `mu_image_factor_strict_rank` | MA를 60일 window 내부에서만 계산한 보수적 후보 |
| `mu_image_factor_balanced` | image factor 추가, Sharpe trade-off 비교 후보 |
| `mu_sharpe_baseline` | Sharpe 안정형 baseline |

### Reference 후보 8개 (`extra_mu_candidates_long.csv`)

floor ~ single model 스펙트럼. 재학습 없이 기존 walk-forward 예측을 동일 schema 로 패키징
(동일 fold grid, leakage-clean). ODE 안에서 "각 신호 layer 가 실제로 얼마나 기여하나" 측정용.

| input | rank corr | Sharpe | 역할 |
|---|---:|---:|---|
| `mu_lstm_image` | 0.0506 | 0.185 | 단일 모델 전체 1위 |
| `mu_cnnlstm_image` | 0.0448 | 0.434 | 단일 CNN+LSTM hybrid 1위 |
| `mu_cnn_2d_residual_small` | 0.0426 | 0.208 | 단일 CNN 1위 |
| `mu_logistic_image` | 0.0392 | 0.385 | 선형 + image baseline |
| `mu_cnn_1d_cumulative` | 0.0271 | 0.521 | high-Sharpe 단일 CNN |
| `mu_momentum_60d` | 0.0245 | 0.180 | classic 60일 모멘텀 factor |
| `mu_logistic_cumulative` | −0.0072 | 0.076 | raw floor (no image, no deep model) |
| `mu_equal_weight` | n/a | n/a | null floor (constant signal) |

## ODE 입력 스케일

| 컬럼 | 설명 |
|---|---|
| `mu_signal` | 기본 추천. 날짜별 centered cross-sectional rank |
| `mu_zscore` | 날짜별 z-score |
| `mu_calibrated_horizon` | expanding calibration 기반 20일 기대수익률 |
| `mu_calibrated_daily` | `mu_calibrated_horizon / 20`, ODE 수식상 가장 자연스러운 daily return-scale input |

기본 실험은 `mu_signal`로 시작하고, ODE 수식에서 return 단위가 필요하면 `mu_calibrated_daily`를 사용하세요.

## 검증 보고서

| 파일 | 의미 |
|---|---|
| `04_validation_reports/multiple_testing_report.md` | 28개 image-factor test에 대한 Bonferroni/FDR 보정 |
| `04_validation_reports/mu_subperiod_stability_report.md` | 2014-2019, 2020-2022, 2023-2026 regime별 안정성 |
| `04_validation_reports/mu_calibration_report.md` | rank signal을 return-scale `mu`로 보정한 결과 |
| `05_source_notes/strict_window_ma_result_note.md` | MA선을 각 60일 window 내부에서만 계산한 robustness 결과 |
| `05_source_notes/extra_pretraining_result_note.md` | 30개 ETF per-fold pretraining negative result |

## 주의

- `future_return`은 평가용 realized target입니다. ODE input으로 사용하면 안 됩니다.
- `Sigma(t)`는 이미지 모델이 예측한 값이 아니라 과거 수익률 기반 rolling covariance입니다.
- image factor는 2020-2022 구간에서 약해지는 regime sensitivity가 있습니다. `04_validation_reports/mu_subperiod_stability_report.md`를 확인하세요.
- 30개 extra ETF supervised pretraining은 성능이 악화되었습니다. 이 결과는 `05_source_notes/extra_pretraining_result_note.md`에 정리되어 있습니다.
