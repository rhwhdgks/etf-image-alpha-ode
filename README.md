# ETF 차트 이미지의 수익률 순위 예측 연구

7개 ETF의 최근 60거래일 가격·거래량을 차트 이미지로 만들고, 20거래일 뒤 어느 ETF의 수익률이 상대적으로 높을지 연구했습니다. 예측 결과는 포트폴리오 분석에 쓸 수 있는 **수익률 순위 신호**와 **ETF 간 위험 자료**로 정리했습니다.

모델을 만들 때 미래 자료를 미리 보지 않도록 기간을 나눠 검증했습니다. 아래 성과 수치는 과거 자료를 이용한 연구 결과이며, 실제 거래 수익은 아닙니다.

## 무엇을 알아봤나

핵심 질문은 **가격 차트를 이미지로 보면 숫자 자료만 볼 때보다 ETF 간 수익률 순위를 더 잘 예측할 수 있는가?**입니다. 이를 위해 여러 모델을 비교하고, 결과를 포트폴리오 최적화 연구에 전달할 자료로 만들었습니다.

포트폴리오 분석에는 날짜별 수익 신호 `mu(t)`, ETF 간 수익률이 함께 움직이는 정도인 공분산 `Sigma(t)`, 평가에 쓰는 실제 수익률 `R(t)`가 필요합니다. 이 저장소에는 이 세 자료를 만드는 코드와 결과가 있습니다. `mu(t)`의 순위 신호를 예상 수익률 수치로 쓰려면 별도 보정이 필요합니다.

## 이 저장소에서 한 일

- 가격·거래량 자료를 차트 이미지로 만들고, 통계 모델과 이미지·시계열 모델의 예측을 비교
- 시장 전체가 함께 움직이는 영향을 고려해 이미지 정보가 추가로 도움이 되는지 확인
- 과거 자료로 학습하고 이후 기간에서 평가해 미래 정보가 섞이는지 점검
- ETF 간 위험 추정치가 불안정한 경우를 찾아 보정
- 수익 신호 후보, 위험 자료, 실제 수익률을 포트폴리오 연구팀에 전달할 형식으로 정리

## 연구 범위

| 항목 | 설정 |
|---|---|
| 분석 대상 | 7개 ETF |
| 예측 대상 | 20거래일 뒤 ETF 간 수익률 순위 |
| 입력 자료 | 직전 60거래일의 시가·고가·저가·종가·거래량 |
| 주요 모델 | 차트 이미지를 읽는 2D CNN |
| 평가 방식 | 과거 기간으로 학습하고 이후 기간에서 반복 평가 |
| 결과물 | 포트폴리오 연구에 사용할 수익 신호·위험 자료 |

## 전체 파이프라인

```text
ETF 가격·거래량 자료
  -> 직전 60일 차트 이미지 생성
  -> 여러 예측 모델 학습
  -> 모델별 수익률 순위 신호 생성
  -> 학습에 쓰지 않은 이후 기간에서 평가
  -> 순위 신호를 예상 수익률 단위로 별도 보정
  -> 위험 자료·실제 수익률과 함께 포트폴리오 연구팀에 전달
```

모델이 미래 데이터를 미리 보지 않도록 시간 순서대로 학습·검증·평가 기간을 나눴습니다. 특징값과 위험 추정치를 만들 때도 해당 시점 이전의 자료만 사용하도록 설계했습니다.

## 핵심 결과

아래 `순위 상관`은 예측한 ETF 순위와 실제 순위가 얼마나 일치했는지 보여줍니다. `상위 ETF 선택 지표`는 예측 상위 ETF를 고르는 규칙의 연구용 샤프지수입니다. 거래비용을 반영한 실거래 성과로 해석하면 안 됩니다.

| 신호 | 설명 | 순위 상관 | 상위 ETF 선택 지표 |
|---|---|---:|---:|
| `logistic_cumulative_scale` | 차트 이미지를 쓰지 않은 비교 모델 | -0.0072 | 0.0764 |
| `logistic_image_scale` | 이미지에 쓰는 값의 크기 조정 방식을 적용한 선형 모델 | 0.0392 | 0.3850 |
| `ensemble_4family` | 여러 통계·이미지·시계열 모델을 합친 결과 | 0.0674 | 0.5430 |
| `mu_image_factor_rank` | 이미지 특징을 추가한 포트폴리오 연구용 순위 신호 | 0.0798 | 0.3155 |
| `mu_image_factor_strict_rank` | 차트의 이동평균 계산을 더 보수적으로 제한한 신호 | 0.0771 | 0.6053 |

표의 `mu(t)`는 ETF 간 상대 순위를 나타냅니다. 실제 예상 수익률의 크기로 사용하려면 별도 보정이 필요합니다.

해석은 다음과 같습니다.

- 이미지를 만들 때 쓰는 값의 크기 조정 방식에서 비교 모델보다 뚜렷한 순위 예측 개선이 있었습니다.
- 복잡한 이미지·시계열 모델 하나가 항상 가장 좋았던 것은 아닙니다. 서로 다른 모델의 예측을 합쳤을 때 도움이 됐습니다.
- 이미지 특징을 추가하면 순위 상관은 좋아졌지만, 상위 ETF 선택 지표가 항상 함께 좋아지지는 않았습니다.
- 결과물은 바로 거래에 쓰는 규칙이 아니라 포트폴리오 분석에 넣을 자료입니다.

## 차트의 어떤 정보가 도움이 됐나

차트에 어떤 요소를 넣느냐에 따라 결과가 달라지는지 알아보려고 6가지 이미지를 비교했습니다.

| 차트 종류 | 포함한 정보 |
|---|---|
| `close_only` | 종가의 움직임 |
| `ohlc_full` | 시가·고가·저가·종가 |
| `ohlc_ma` | 시가·고가·저가·종가와 이동평균 |
| `ohlc_volume` | 시가·고가·저가·종가와 거래량 |
| `ohlc_ma_volume` | 시가·고가·저가·종가, 이동평균, 거래량 |
| `high_low_range` | 고가와 저가의 차이를 강조한 형태 |

이 비교는 차트의 어떤 정보가 예측에 영향을 주는지 확인하기 위한 것입니다. 이동평균이나 거래량을 넣었을 때 결과가 달라지는지도 살펴봤습니다.

## 포트폴리오 연구팀에 전달한 자료

포트폴리오를 날짜별로 조정하는 ODE 모형에는 세 가지 자료가 필요합니다.

- `mu(t)`: 날짜별·ETF별 수익률 순위 신호와 보정된 예상 수익률 후보
- `Sigma(t)`: 날짜별 ETF 수익률이 함께 움직이는 정도를 나타내는 위험 자료
- `R(t)`: 과거 평가에 쓰는 실제 수익률

이 레포는 세 가지를 모두 제공합니다.

| 파일 | 용도 |
|---|---|
| `outputs/ode_handoff/02_mu_inputs/selected_mu_input.csv` | 추천 rank-scale `mu(t)` 후보 |
| `outputs/ode_handoff/02_mu_inputs/selected_mu_input_calibrated.csv` | 동일 시그널의 expanding return-scale calibration 버전 |
| `outputs/ode_handoff/02_mu_inputs/final_mu_inputs_wide.csv` | 5개 메인 `mu(t)` 후보의 wide format |
| `outputs/ode_handoff/02_mu_inputs/input_performance_summary.csv` | rank corr, Sharpe, hit rate, turnover 요약 |
| `outputs/ode_handoff/03_sigma_returns/sigma_shrunk_wide.csv` | 추천 Ledoit-Wolf shrunk covariance input |
| `outputs/ode_handoff/03_sigma_returns/sigma_wide.csv` | 비교용 60일 rolling sample covariance |
| `outputs/ode_handoff/03_sigma_returns/returns_for_ode.csv` | downstream backtest용 realized return |

사용 권장안은 다음과 같습니다.

- rank 기반 ODE 실험에는 `selected_mu_input.csv`의 `mu_signal`을 사용합니다.
- ODE 구현이 daily return 단위의 기대수익을 요구하면 `selected_mu_input_calibrated.csv`의 `mu_calibrated_daily`를 사용합니다.
- covariance input은 `sigma_shrunk_wide.csv`를 기본으로 쓰고, `sigma_wide.csv`는 비교용으로 사용합니다.

## 위험 자료의 안정성

직전 60일 자료만으로 ETF 간 위험을 계산하면 일부 날짜에는 값이 불안정해집니다. 이를 보정한 `Ledoit-Wolf` 방식의 공분산을 기본 후보로 제시했습니다.

주요 진단 결과:

- 보정 전 안정성 지표의 중앙값: 2409
- 보정 후 안정성 지표의 중앙값: 64
- 안정성 지표가 `1e4`를 넘는 불안정한 날짜는 제외

이 지표는 낮을수록 계산이 안정적입니다. 그래서 단순 계산 결과보다 보정한 `sigma_shrunk_wide.csv`를 기본 위험 자료로 전달했습니다.

## 레포 구조

```text
.
├── src/                         # 데이터, feature, 이미지 렌더링, 모델, 평가 코드
├── docs/                        # 연구 보고서, 검증 노트, 핵심 figure
├── outputs/ode_handoff/          # GitHub용으로 정리한 ODE 입력 패키지
├── etfdata.csv                  # 7개 ETF OHLCV 원자료
├── run_walkforward.py           # walk-forward 모델 평가 entry point
├── build_image_factor_*.py      # image factor 추출 / ablation / robustness
├── build_final_ode_*.py         # 최종 mu 및 Sigma handoff 생성
├── build_mu_calibration.py      # rank signal을 return-scale mu로 보정
├── build_sigma_*.py             # covariance shrinkage 및 variant 생성
└── requirements.txt
```

대용량 중간 산출물, fold별 예측 파일, model checkpoint, 원문 논문 PDF, 발표 자료, local virtual environment는 공개 레포에서 제외했습니다.

## 빠른 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 기본 walk-forward 모델 실행
python run_walkforward.py --lookback 60 --horizon 20

# image factor ablation
python build_image_factor_ablation.py

# 최종 ODE 입력 생성
python build_final_ode_mu_inputs.py
python build_final_ode_sigma_inputs.py
python build_mu_calibration.py
python build_sigma_shrunk.py
python build_sigma_variants.py
```

전체 모델을 과거부터 순서대로 다시 학습하면 시간이 오래 걸릴 수 있습니다. 결과를 먼저 살펴보려면 `docs/README.md`와 `outputs/ode_handoff/`의 요약 CSV부터 보면 됩니다.

## 검증과 한계

이 프로젝트는 결과를 과장하지 않기 위해 다음 검증 항목을 별도로 문서화했습니다.

- 시간 순서를 지킨 학습·검증·평가 기간 분리
- 이미지 특징을 줄이는 계산에는 학습·검증 기간의 자료만 사용
- ETF의 공통 움직임과 위험을 추정할 때 해당 시점 이전 자료만 사용
- 이미지 특징의 효과를 반복 표본 추출과 다중 검정 보정으로 점검
- 추가 ETF 자료로 미리 학습했지만 도움이 되지 않은 결과도 기록

주의할 점도 있습니다.

- 20거래일 뒤 수익률을 예측하므로 인접한 평가 기간의 결과가 서로 겹칩니다.
- 이미지 특징의 일부 개선은 통계적으로 확실한 성과보다 추가 연구할 만한 단서에 가깝습니다.
- `future_return`은 예측 결과를 평가할 때만 쓰는 미래 수익률입니다. 예측 모델의 입력으로 쓰면 안 됩니다.
- 이 저장소는 투자 조언이 아닙니다. 실제 운용 전에는 거래비용, 매매 빈도, 주문 규모, 시장 상황 변화의 영향을 별도로 검증해야 합니다.

## 참고 문서

| 문서 | 내용 |
|---|---|
| `docs/README.md` | 문서 전체 안내 |
| `docs/reports/process_integrity_report.md` | leakage, bootstrap, reproducibility 감사 |
| `docs/reports/paper_draft_ko.md` | 논문형 한국어 연구 초안 |
| `outputs/ode_handoff/01_start_here/README.md` | ODE 팀 전달용 시작 문서 |
| `outputs/ode_handoff/06_mu_submission_validation/` | 후보 고정 및 fold-boundary purge 보완 검증 |
| `outputs/ode_handoff/01_start_here/manifest.json` | 전달 파일 manifest |
