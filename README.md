# 이미지 기반 ETF 알파 시그널과 ODE 포트폴리오 입력 패키지

ETF 가격 차트를 이미지로 변환해 cross-sectional 알파 시그널을 만들고, 이를 ODE 기반 동적 포트폴리오 최적화에 필요한 `mu(t)`와 `Sigma(t)` 입력으로 정리한 금융 리서치/엔지니어링 프로젝트입니다.

이 레포는 LP/MM 직무와 인접한 리서치 및 리스크 입력 프로젝트로 포지셔닝했습니다. 실제 market-making execution engine은 아닙니다. 즉, 호가 제출, 체결 시뮬레이터, 거래소 라우팅, inventory controller는 포함하지 않습니다. 대신 그 이전 단계인 시그널 생성, covariance/risk handoff, 데이터 누수 점검, 재현성 관리에 집중했습니다.

## 프로젝트가 보여주는 역량

- 7개 ETF 자산군에 대한 cross-sectional 알파 시그널 리서치
- Jiang-style OHLCV 차트 이미지 생성 및 CNN image factor 추출
- logistic, 1D CNN, 2D CNN, LSTM, CNN+LSTM hybrid 모델군 비교
- rolling PCA 공통요인 통제 이후 image factor의 추가 설명력 검정
- ODE optimizer가 바로 사용할 수 있는 `mu(t)`, `Sigma(t)`, realized return 패키징
- sample covariance가 불안정한 이유와 Ledoit-Wolf shrinkage 적용 근거 제시
- leakage check, block bootstrap, multiple-testing, negative result까지 포함한 process integrity 관리

## 연구 범위

| 항목 | 설정 |
|---|---|
| 투자 universe | 7개 ETF asset class |
| 예측 대상 | 20일 후 forward return ranking |
| 기본 lookback | 60 trading days |
| 핵심 표현 방식 | OHLCV 가격 차트 이미지 |
| 메인 image-factor 모델 | 2D residual CNN feature / score |
| 평가 방식 | walk-forward out-of-sample |
| downstream 용도 | ODE 포트폴리오 optimizer 입력 패키지 |

핵심 질문은 “가격 경로를 이미지로 표현했을 때 ETF 간 상대 기대수익 ranking에 정보가 추가되는가?”입니다. 이 프로젝트는 production trading 수익성을 주장하지 않고, optimizer에 들어가기 전 단계의 alpha/risk input을 만드는 데 초점을 둡니다.

## 핵심 결과

| Signal | 해석 | Rank Corr | Top-k Sharpe |
|---|---|---:|---:|
| `logistic_cumulative_scale` | 이미지 없는 raw baseline | -0.0072 | 0.0764 |
| `logistic_image_scale` | image-style scaling을 쓴 선형 baseline | 0.0392 | 0.3850 |
| `ensemble_4family` | logistic + CNN + LSTM/CNNLSTM baseline | 0.0674 | 0.5430 |
| `mu_image_factor_rank` | image factor를 추가한 ODE용 `mu(t)` 후보 | 0.0798 | 0.3155 |
| `mu_image_factor_strict_rank` | MA 계산을 더 보수적으로 제한한 robustness 후보 | 0.0771 | 0.6053 |

해석:

- 가장 뚜렷한 개선은 raw no-image baseline 대비 image-style representation에서 나왔습니다.
- CNN/LSTM이 단순 모델을 압도한 것이 아니라, 상관이 낮은 model family를 섞을 때 ensemble 효과가 나타났습니다.
- image-factor add-on은 cross-sectional rank quality를 개선했지만, 단순 top-k Sharpe가 항상 좋아지지는 않았습니다. 그래서 최종 산출물은 standalone trading rule이 아니라 optimizer input package입니다.
- `Sigma(t)`에는 Ledoit-Wolf shrinkage를 기본 추천합니다. 60일 sample covariance의 median condition number가 2409였는데, shrinkage 적용 후 64로 낮아졌고 condition number가 `1e4`를 넘는 날짜가 제거되었습니다.

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

## 주요 산출물

| 파일 | 용도 |
|---|---|
| `outputs/ode_handoff/02_mu_inputs/selected_mu_input.csv` | 추천 rank-scale `mu(t)` 후보 |
| `outputs/ode_handoff/02_mu_inputs/selected_mu_input_calibrated.csv` | 동일 시그널의 expanding return-scale calibration 버전 |
| `outputs/ode_handoff/02_mu_inputs/final_mu_inputs_wide.csv` | 5개 메인 `mu(t)` 후보의 wide format |
| `outputs/ode_handoff/02_mu_inputs/input_performance_summary.csv` | rank corr, Sharpe, hit rate, turnover 요약 |
| `outputs/ode_handoff/03_sigma_returns/sigma_shrunk_wide.csv` | 추천 Ledoit-Wolf shrunk covariance input |
| `outputs/ode_handoff/03_sigma_returns/sigma_wide.csv` | 비교용 60일 rolling sample covariance |
| `outputs/ode_handoff/03_sigma_returns/returns_for_ode.csv` | downstream backtest용 realized return |
| `docs/reports/process_integrity_report.md` | leakage, bootstrap, reproducibility 감사 보고서 |
| `docs/reports/paper_draft_ko.md` | 논문형 한국어 연구 초안 |

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

전체 deep-learning walk-forward 실행은 시간이 오래 걸릴 수 있습니다. 리뷰 목적이라면 `docs/README.md`와 `outputs/ode_handoff/`의 curated CSV부터 보는 것을 권장합니다.

## ODE와의 연결

downstream ODE optimizer는 세 가지 time-indexed input이 필요합니다.

- `mu(t)`: 날짜별, 자산별 기대수익 스타일 시그널
- `Sigma(t)`: 날짜별 covariance matrix
- `R(t)`: backtest 및 평가용 realized return

이 레포는 세 가지를 모두 제공합니다.

- rank-scale 실험에는 `selected_mu_input.csv`의 `mu_signal`을 사용합니다.
- ODE 구현이 daily return 단위의 기대수익을 요구하면 `selected_mu_input_calibrated.csv`의 `mu_calibrated_daily`를 사용합니다.
- covariance input은 `sigma_shrunk_wide.csv`를 기본으로 쓰고, `sigma_wide.csv`와 비교합니다.

## Process Integrity

- walk-forward 학습은 시간 순서를 유지한 chronological split을 사용합니다.
- image-feature PCA는 train/validation 구간에만 fit하고 test 구간에는 transform만 적용합니다.
- rolling PCA control과 covariance estimate는 trailing window만 사용합니다.
- `future_return` 컬럼은 평가용 target이며 optimizer input으로 사용하면 안 됩니다.
- 20일 forward target은 overlapping label을 만들기 때문에 IID bootstrap이 과도하게 낙관적일 수 있어 block-bootstrap caveat를 문서화했습니다.
- 30개 extra ETF supervised pretraining은 성능이 악화된 negative result였고, 추천 input이 아니라 robustness evidence로 남겼습니다.

이 프로젝트는 금융 리서치 및 포트폴리오 엔지니어링 역량을 보여주기 위한 목적이며, 투자 조언이 아닙니다.
