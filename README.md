# 이미지 기반 ETF 알파 시그널과 ODE 포트폴리오 입력 패키지

ETF 가격 차트를 이미지로 변환해 cross-sectional 알파 시그널을 만들고, 이를 ODE 기반 동적 포트폴리오 최적화에 필요한 `mu(t)`와 `Sigma(t)` 입력으로 정리한 금융 리서치/엔지니어링 프로젝트입니다.

이 레포는 LP/MM 직무와 인접한 리서치 역량을 보여주기 위한 포트폴리오 프로젝트입니다. 실제 market-making execution engine은 아니며, 호가 제출, 체결 시뮬레이터, 거래소 라우팅, inventory controller는 포함하지 않습니다. 대신 그 이전 단계인 alpha signal 생성, risk input 구성, 데이터 누수 점검, 재현성 관리에 집중했습니다.

## 프로젝트 목적

ODE 기반 동적 포트폴리오 최적화에서는 날짜별 기대수익 `mu(t)`, 공분산 `Sigma(t)`, 실현수익률 `R(t)`가 필요합니다. 이 프로젝트의 목표는 ETF 가격 경로에서 추출한 이미지 기반 정보를 활용해, ODE optimizer에 전달할 수 있는 `mu(t)` 후보와 안정적인 `Sigma(t)` 입력을 만드는 것입니다.

핵심 질문은 다음과 같습니다.

> 가격 경로를 차트 이미지로 표현하면 ETF 간 상대 기대수익 ranking에 추가 정보가 생기는가?

이 프로젝트는 production trading 수익성을 주장하지 않습니다. 목적은 optimizer에 들어가기 전 단계의 alpha/risk input을 연구하고, 그 과정이 leakage와 overfitting 측면에서 방어 가능하도록 정리하는 것입니다.

## 프로젝트가 보여주는 역량

- 7개 ETF 자산군에 대한 cross-sectional 알파 시그널 리서치
- Jiang-style OHLCV 차트 이미지 생성 및 CNN image factor 추출
- logistic, 1D CNN, 2D CNN, LSTM, CNN+LSTM hybrid 모델군 비교
- rolling PCA 공통요인 통제 이후 image factor의 추가 설명력 검정
- ODE optimizer가 바로 사용할 수 있는 `mu(t)`, `Sigma(t)`, realized return 패키징
- sample covariance의 불안정성을 진단하고 Ledoit-Wolf shrinkage 적용
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

## 전체 파이프라인

```text
ETF OHLCV data
  -> 60일 가격 차트 이미지 생성
  -> CNN / LSTM / logistic 모델 학습
  -> image factor 및 ensemble signal 추출
  -> walk-forward OOS 평가
  -> rank-scale mu(t) 후보 생성
  -> return-scale calibration
  -> Sigma(t), R(t)와 함께 ODE 팀에 전달
```

중요한 점은 모델이 미래 데이터를 보지 않도록 시간 순서를 유지했다는 것입니다. 학습, 검증, 테스트는 walk-forward 방식으로 분리했고, feature PCA와 covariance estimate는 각 시점 이전 데이터만 사용하도록 설계했습니다.

## 핵심 결과

| Signal | 해석 | Rank Corr | Top-k Sharpe |
|---|---|---:|---:|
| `logistic_cumulative_scale` | 이미지 없는 raw baseline | -0.0072 | 0.0764 |
| `logistic_image_scale` | image-style scaling을 쓴 선형 baseline | 0.0392 | 0.3850 |
| `ensemble_4family` | logistic + CNN + LSTM/CNNLSTM baseline | 0.0674 | 0.5430 |
| `mu_image_factor_rank` | image factor를 추가한 ODE용 `mu(t)` 후보 | 0.0798 | 0.3155 |
| `mu_image_factor_strict_rank` | MA 계산을 더 보수적으로 제한한 robustness 후보 | 0.0771 | 0.6053 |

해석은 다음과 같습니다.

- raw no-image baseline 대비 image-style representation에서 가장 뚜렷한 개선이 나왔습니다.
- CNN/LSTM이 단순 모델을 일방적으로 압도한 것이 아니라, 상관이 낮은 model family를 섞을 때 ensemble 효과가 커졌습니다.
- image-factor add-on은 cross-sectional rank quality를 개선했지만, top-k Sharpe가 항상 함께 좋아지지는 않았습니다.
- 따라서 최종 산출물은 standalone trading rule이 아니라 ODE optimizer에 넣기 위한 input package로 해석해야 합니다.

## Image Factor Ablation

가격 이미지를 구성하는 요소별 정보성을 보기 위해 6개 이미지 variant를 비교했습니다.

| Variant | 의미 |
|---|---|
| `close_only` | 종가 path만 사용 |
| `ohlc_full` | OHLC 캔들만 사용 |
| `ohlc_ma` | OHLC + moving average |
| `ohlc_volume` | OHLC + volume bar |
| `ohlc_ma_volume` | OHLC + MA + volume, 기본 Jiang-style |
| `high_low_range` | high-low range block 강조 |

이 ablation의 목적은 단순 성능 경쟁이 아니라, 이미지 안의 어떤 구성 요소가 path information을 만드는지 확인하는 것입니다. MA나 volume이 포함될 때 유의성이 달라진다면, 단순 종가 수준이 아니라 가격 경로의 구조가 시그널에 반영된다는 해석이 가능합니다.

## ODE 입력 패키지

downstream ODE optimizer는 세 가지 time-indexed input을 필요로 합니다.

- `mu(t)`: 날짜별, 자산별 기대수익 스타일 시그널
- `Sigma(t)`: 날짜별 covariance matrix
- `R(t)`: backtest 및 평가용 realized return

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

## Sigma 입력

단순 60일 sample covariance는 ETF 수가 적어도 기간별로 불안정할 수 있습니다. 이 프로젝트에서는 Ledoit-Wolf shrinkage covariance를 기본 추천합니다.

주요 진단 결과:

- 60일 sample covariance의 median condition number: 2409
- shrinkage 적용 후 median condition number: 64
- condition number가 `1e4`를 넘는 불안정 날짜 제거

따라서 ODE 팀에는 raw sample covariance보다 `sigma_shrunk_wide.csv`를 기본 `Sigma(t)` 입력으로 전달하는 것이 더 방어 가능합니다.

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

전체 deep-learning walk-forward 실행은 시간이 오래 걸릴 수 있습니다. 리뷰 목적이라면 `docs/README.md`와 `outputs/ode_handoff/`의 curated CSV부터 보는 것을 권장합니다.

## 검증과 한계

이 프로젝트는 결과를 과장하지 않기 위해 다음 검증 항목을 별도로 문서화했습니다.

- walk-forward chronological split
- train/validation 구간에만 fit한 image-feature PCA
- trailing window 기반 rolling PCA control과 covariance estimate
- block bootstrap 기반 image-factor 유의성 점검
- multiple-testing correction
- extra ETF supervised pretraining의 negative result 기록

주의할 점도 있습니다.

- 20일 forward target은 overlapping label을 만들기 때문에 IID 가정이 약합니다.
- 일부 image-factor lift는 strict statistical significance보다 mechanism evidence에 가깝습니다.
- `future_return` 컬럼은 평가용 target이며 ODE 입력으로 사용하면 안 됩니다.
- 이 레포는 투자 조언이 아니며, 실제 운용 전에는 별도의 거래비용, 리밸런싱, capacity, regime robustness 검증이 필요합니다.

## 참고 문서

| 문서 | 내용 |
|---|---|
| `docs/README.md` | 문서 전체 안내 |
| `docs/reports/process_integrity_report.md` | leakage, bootstrap, reproducibility 감사 |
| `docs/reports/paper_draft_ko.md` | 논문형 한국어 연구 초안 |
| `outputs/ode_handoff/01_start_here/README.md` | ODE 팀 전달용 시작 문서 |
| `outputs/ode_handoff/06_mu_submission_validation/` | 후보 고정 및 fold-boundary purge 보완 검증 |
| `outputs/ode_handoff/01_start_here/manifest.json` | 전달 파일 manifest |

## 한 줄 요약

이 프로젝트는 ETF 가격 이미지를 활용해 cross-sectional `mu(t)` 후보를 만들고, shrinkage covariance와 함께 ODE 기반 포트폴리오 최적화에 전달할 수 있도록 정리한 금융 리서치 프로젝트입니다.
