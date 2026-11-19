# ETF 차트 이미지 수익률 순위 예측

7개 ETF의 직전 60거래일 가격·거래량을 차트 이미지로 만들어, 20거래일 뒤 ETF 간 수익률 순위를 예측했습니다. 차트 이미지가 숫자 자료만 쓸 때보다 순위 예측에 도움이 되는지 비교하고, 결과를 포트폴리오 최적화(ODE) 모형에 넣을 수익 신호 `mu(t)`, 공분산 `Sigma(t)`, 실현 수익률 `R(t)`로 정리했습니다.

모든 성과 수치는 과거 자료 기반 연구 결과이며 거래비용을 반영한 실거래 성과가 아닙니다.

## 설정

| 항목 | 내용 |
|---|---|
| 대상 | 7개 ETF |
| 입력 | 직전 60거래일 시가·고가·저가·종가·거래량 |
| 예측 대상 | 20거래일 뒤 ETF 간 수익률 순위 |
| 주요 모델 | 차트 이미지 2D CNN, 통계·시계열 비교 모델 |
| 평가 | 시간 순서로 학습·검증·평가 기간을 나눈 walk-forward |
| 산출물 | ODE 모형용 `mu(t)`, `Sigma(t)`, `R(t)` |

특징값과 공분산은 각 시점 이전 자료만으로 계산했습니다. `future_return`은 평가에만 쓰고 모델 입력에는 쓰지 않습니다.

## 결과

순위 상관은 예측 순위와 실제 순위의 일치도, 상위 ETF 선택 지표는 예측 상위 ETF를 고르는 규칙의 연구용 샤프지수입니다.

| 신호 | 설명 | 순위 상관 | 상위 ETF 선택 지표 |
|---|---|---:|---:|
| `logistic_cumulative_scale` | 이미지를 쓰지 않은 비교 모델 | -0.0072 | 0.0764 |
| `logistic_image_scale` | 이미지용 값 크기 조정을 적용한 선형 모델 | 0.0392 | 0.3850 |
| `ensemble_4family` | 통계·이미지·시계열 모델 결합 | 0.0674 | 0.5430 |
| `mu_image_factor_rank` | 이미지 특징을 추가한 순위 신호 | 0.0798 | 0.3155 |
| `mu_image_factor_strict_rank` | 이동평균 계산을 더 보수적으로 제한한 신호 | 0.0771 | 0.6053 |

- 이미지용 값 크기 조정 방식에서 비교 모델보다 순위 예측이 개선됨
- 단일 이미지·시계열 모델이 항상 최선은 아니었고, 여러 모델을 결합했을 때 더 나았음
- 이미지 특징을 추가하면 순위 상관은 올랐지만 상위 ETF 선택 지표가 항상 함께 오르지는 않음
- `mu(t)`는 상대 순위 신호이며, 기대수익률 크기로 쓰려면 별도 보정이 필요함

## 차트 구성 비교

차트에 넣는 정보에 따라 결과가 달라지는지 6가지 이미지를 비교했습니다.

| 차트 | 포함 정보 |
|---|---|
| `close_only` | 종가 |
| `ohlc_full` | 시가·고가·저가·종가 |
| `ohlc_ma` | OHLC + 이동평균 |
| `ohlc_volume` | OHLC + 거래량 |
| `ohlc_ma_volume` | OHLC + 이동평균 + 거래량 |
| `high_low_range` | 고가·저가 범위 강조 |

## 공분산 안정화

60일 표본 공분산은 일부 날짜에 불안정해서 Ledoit-Wolf 축소 공분산을 기본값으로 썼습니다.

- 조건수 중앙값: 보정 전 2409 → 보정 후 64
- 조건수가 `1e4`를 넘는 날짜는 제외

## ODE 모형 전달 자료

| 파일 | 내용 |
|---|---|
| `outputs/ode_handoff/02_mu_inputs/selected_mu_input.csv` | 추천 순위 기반 `mu(t)` |
| `outputs/ode_handoff/02_mu_inputs/selected_mu_input_calibrated.csv` | 같은 신호를 확장 윈도우로 수익률 단위 보정한 버전 |
| `outputs/ode_handoff/02_mu_inputs/final_mu_inputs_wide.csv` | 주요 `mu(t)` 후보 5개 (wide) |
| `outputs/ode_handoff/02_mu_inputs/input_performance_summary.csv` | 순위 상관, 샤프, 적중률, 회전율 요약 |
| `outputs/ode_handoff/03_sigma_returns/sigma_shrunk_wide.csv` | Ledoit-Wolf 축소 공분산 (기본) |
| `outputs/ode_handoff/03_sigma_returns/sigma_wide.csv` | 60일 표본 공분산 (비교용) |
| `outputs/ode_handoff/03_sigma_returns/returns_for_ode.csv` | 백테스트용 실현 수익률 |

순위 기반 실험에는 `selected_mu_input.csv`의 `mu_signal`, 일간 기대수익 단위가 필요하면 `selected_mu_input_calibrated.csv`의 `mu_calibrated_daily`를 씁니다.

## 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_walkforward.py --lookback 60 --horizon 20   # walk-forward 모델 평가
python build_image_factor_ablation.py                  # 이미지 특징 ablation

python build_final_ode_mu_inputs.py                    # ODE 입력 생성
python build_final_ode_sigma_inputs.py
python build_mu_calibration.py
python build_sigma_shrunk.py
python build_sigma_variants.py
```

전체 walk-forward 재학습은 오래 걸립니다. 결과만 보려면 [docs/README.md](docs/README.md)와 `outputs/ode_handoff/`의 요약 CSV를 먼저 보면 됩니다.

## 구조

```text
src/                      데이터, 특징, 이미지 렌더링, 모델, 평가
docs/                     연구 보고서, 검증 노트, 그림
outputs/ode_handoff/      ODE 전달 자료
etfdata.csv               7개 ETF OHLCV 원자료
run_walkforward.py        walk-forward 평가
build_image_factor_*.py   이미지 특징 추출·ablation·강건성
build_final_ode_*.py      최종 mu·Sigma 생성
build_mu_calibration.py   순위 신호 → 수익률 단위 보정
build_sigma_*.py          공분산 축소·변형
```

중간 산출물, fold별 예측, 모델 체크포인트, 논문 PDF, 발표 자료는 공개 저장소에서 제외했습니다.

## 검증과 한계

- 시간 순서 기간 분리, 이미지 특징 축소에는 학습·검증 기간 자료만 사용
- 이미지 특징 효과는 bootstrap과 다중검정 보정으로 점검. 일부 개선은 통계적으로 확정된 성과라기보다 추가 연구 대상에 가까움
- 20거래일 수익률을 예측하므로 인접 평가 구간이 서로 겹침
- 추가 ETF 자료로 사전학습한 결과는 도움이 되지 않았고, 그 결과도 기록함
- 실제 운용에는 거래비용, 매매 빈도, 주문 규모, 시장 국면 변화의 영향을 따로 검증해야 함

## 참고 문서

| 문서 | 내용 |
|---|---|
| [docs/README.md](docs/README.md) | 문서 안내 |
| [docs/reports/process_integrity_report.md](docs/reports/process_integrity_report.md) | 누수, bootstrap, 재현성 점검 |
| [docs/reports/paper_draft_ko.md](docs/reports/paper_draft_ko.md) | 논문 형식 연구 초안 |
| [outputs/ode_handoff/01_start_here/README.md](outputs/ode_handoff/01_start_here/README.md) | ODE 전달 자료 시작 문서 |
| [outputs/ode_handoff/06_mu_submission_validation/](outputs/ode_handoff/06_mu_submission_validation) | 후보 고정, fold 경계 purge 검증 |
