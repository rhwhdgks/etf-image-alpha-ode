# ODE 입력 전달 요약

이 폴더는 GitHub 포트폴리오용으로 정리한 ODE 입력 패키지다. 전체 실험 덤프가 아니라, ODE팀이 바로 확인할 수 있는 핵심 `mu(t)`, `Sigma(t)`, realized return, 검증 보고서만 남겼다.

## 최종 추천

| 항목 | 내용 |
|---|---|
| 추천 `mu(t)` | `mu_image_factor_rank` |
| 파일 | `../02_mu_inputs/selected_mu_input.csv` |
| 기본 컬럼 | `mu_signal` |
| return-scale 대안 | `../02_mu_inputs/selected_mu_input_calibrated.csv`의 `mu_calibrated_daily` |
| 추천 `Sigma(t)` | `../03_sigma_returns/sigma_shrunk_wide.csv` |
| realized return | `../03_sigma_returns/returns_for_ode.csv` |

## 성능 요약

| 후보 | rank corr | Sharpe | 역할 |
|---|---:|---:|---|
| `mu_rank_baseline` | 0.0674 | 0.5430 | image factor 이전 baseline |
| `mu_image_factor_rank` | 0.0798 | 0.3155 | main image-factor ranking input |
| `mu_image_factor_strict_rank` | 0.0771 | 0.6053 | strict-window MA robustness input |
| `mu_sharpe_baseline` | 0.0606 | 0.6425 | Sharpe-priority 비교군 |

해석은 조심해야 한다. `mu_image_factor_rank`는 rank corr를 올리지만 단순 top-k Sharpe는 낮다. 따라서 단독 매매전략이 아니라 ODE optimizer 안에서 `mu(t)` 후보로 비교하는 것이 맞다.

## 남긴 파일

| 파일 | 용도 |
|---|---|
| `../02_mu_inputs/selected_mu_input.csv` | 추천 `mu(t)` 한 개 |
| `../02_mu_inputs/selected_mu_input_calibrated.csv` | 추천 `mu(t)`의 return-scale calibration |
| `../02_mu_inputs/final_mu_inputs_wide.csv` | 메인 후보 5개 wide format |
| `../02_mu_inputs/extra_mu_candidates_wide.csv` | reference/baseline 후보 8개 wide format |
| `../02_mu_inputs/input_performance_summary.csv` | 후보별 성능 |
| `../03_sigma_returns/sigma_shrunk_wide.csv` | Ledoit-Wolf shrinkage covariance |
| `../03_sigma_returns/sigma_wide.csv` | 60일 sample covariance |
| `../03_sigma_returns/returns_for_ode.csv` | realized returns |
| `../04_validation_reports/process_integrity_report.md` | leakage / bootstrap / 한계 감사 |

## ODE팀 사용 순서

1. `selected_mu_input.csv`를 읽는다.
2. `date x asset` 형태로 pivot한다.
3. `mu_signal`을 rank-scale `mu(t)`로 사용한다.
4. return 단위가 필요하면 `selected_mu_input_calibrated.csv`의 `mu_calibrated_daily`를 사용한다.
5. `sigma_shrunk_wide.csv`를 기본 `Sigma(t)`로 사용한다.
6. `sigma_wide.csv`를 비교군으로 같이 돌린다.

## 주의

- `future_return`은 검증용 target이다. ODE input으로 쓰면 안 된다.
- `Sigma(t)`는 CNN이 예측한 값이 아니라 과거 수익률 기반 covariance estimate다.
- image factor는 2020-2022 구간에서 약해지는 regime sensitivity가 있다.
- 30개 extra ETF supervised pretraining은 성능이 악화된 negative result다.
