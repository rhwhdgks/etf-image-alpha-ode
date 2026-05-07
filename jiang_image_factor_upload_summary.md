# Jiang-Style Image Factor 확장 정리

## 한 줄 요약

Jiang 논문의 아이디어처럼 ETF 가격 경로를 이미지로 바꾸고, 2D CNN의 마지막 FC feature를 `image factor`로 추출했다. 그 결과, ETF 공통요인인 rolling PCA factor를 통제한 뒤에도 `image_score`가 통계적으로 유의했다. 다만 portfolio ensemble에 바로 섞었을 때 성능 개선은 제한적이어서, 현재 결론은 "이미지 경로 정보는 존재하지만, ODE 입력으로는 검증용 보조 신호로 쓰는 것이 안전하다"이다.

## 1. Jiang 그림이 의미하는 것

사용자가 보낸 Jiang 그림은 주가 데이터를 숫자 시계열로만 보지 않고, 일정 기간의 OHLC 경로를 작은 이미지로 바꿔 CNN에 넣는 구조를 보여준다.

예를 들어 5일, 20일, 60일 가격 흐름을 각각 이미지로 만들고, CNN이 그 이미지 안의 추세, 반전, 변동성, 꼬리 모양 같은 시각적 패턴을 학습한다. 마지막에는 FC layer와 softmax 또는 output layer를 거쳐 미래 수익 방향이나 수익률을 예측한다.

중요한 점은 CNN의 최종 예측값만 보는 것이 아니라, output 직전의 FC representation을 가격 경로의 압축된 이미지 특성으로 볼 수 있다는 것이다. 이번 확장은 바로 이 부분을 ETF 연구에 적용한 것이다.

## 2. 기존 CNN/LSTM 연구와의 연결

기존 연구는 모델의 prediction score와 ensemble 성능 중심이었다.

기존 핵심 결론은 다음과 같았다.

- CNN, Logistic, LSTM, CNN-LSTM 단독 모델은 대부분 비슷한 성능 구간에 있었다.
- 하지만 서로 상관이 낮은 model family를 섞으면 rank correlation의 천장이 올라갔다.
- `ensemble_best`는 Sharpe 기준으로 강했고, `ensemble_4family`는 rank correlation 기준으로 강했다.

이번 Jiang-style 확장은 여기에 질문을 하나 더 추가한다.

"CNN이 만든 예측값이 아니라, CNN 내부 feature 자체가 가격 경로에서 나온 유의미한 image factor인가?"

즉, 기존 연구가 "어떤 모델이 예측을 잘하나"였다면, 이번 확장은 "이미지로 변환된 가격 경로가 독립적인 factor 정보를 갖고 있나"를 검정한 것이다.

## 3. 이번에 구현한 방법

메인 모델은 `cnn_2d_residual_small`로 고정했다. 이유는 Jiang 그림처럼 실제 2D chart image를 CNN에 넣는 구조와 가장 가깝고, 기존 Phase 2 실험에서 2D CNN을 재조정했을 때 단일 CNN rank correlation이 개선됐기 때문이다.

초기에는 1D CNN, dilated CNN, attention CNN, 2D CNN 등 여러 변형을 실험했지만, 최종 정리에서는 CNN을 두 역할로 압축했다.

- `cnn_1d_cumulative_scale`: 기존 `ensemble_best`에 들어간 비교용 1D CNN
- `cnn_2d_residual_small`: Jiang-style image factor를 추출하는 메인 2D CNN

따라서 발표/제출에서는 CNN 변형 전체를 나열하지 않고, "탐색 후 1D CNN 대표와 2D image-factor extractor만 남겼다"는 방식으로 설명하면 된다.

설정은 다음과 같다.

| 항목 | 값 |
|---|---:|
| Lookback window | 60 |
| Prediction horizon | 20 |
| Model | `cnn_2d_residual_small` |
| Epochs | 30 |
| Patience | 5 |
| Weight decay | 5e-4 |
| Dropout | 0.2 |
| Walk-forward folds | 48 |
| OOS rows | 20,160 |
| OOS dates | 2,880 |
| Assets | 7 ETF assets |

구현 흐름은 다음과 같다.

1. ETF OHLCV 데이터를 공통 valid sample로 맞춘다.
2. 60일 가격 경로를 Jiang-style 2D chart image로 만든다.
3. 각 walk-forward fold마다 과거 train/validation 구간으로 CNN을 학습한다.
4. Test 구간에서는 미래 정보를 보지 않고 OOS image score와 FC feature를 추출한다.
5. Train+validation FC feature에만 `StandardScaler + PCA(3)`를 fit한다.
6. Test FC feature를 PCA 공간으로 변환해 `image_factor_pc1~3`을 만든다.
7. Rolling 252일 ETF return PCA loading을 공통요인 control로 만든다.
8. `future_return ~ PCA controls + image factor` 형태의 pooled panel OLS를 date-clustered SE로 검정한다.

## 4. Image factor 유의성 결과

핵심 결과는 다음과 같다.

| Factor | t-stat | p-value | Delta R2 | Daily rank corr |
|---|---:|---:|---:|---:|
| `image_score` | 4.612 | 0.000004 | 0.002257 | 0.0121 |
| `image_factor_pc3` | -2.224 | 0.0262 | 0.000479 | -0.0136 |
| `image_factor_pc1` | 1.338 | 0.1811 | 0.000199 | 0.0406 |
| `image_factor_pc2` | -0.135 | 0.8926 | 0.000002 | 0.0130 |

해석은 다음과 같다.

- `image_score`는 rolling PCA 공통요인을 통제한 뒤에도 통계적으로 유의했다.
- 따라서 ETF 가격 경로를 이미지로 바꿔 CNN에 넣는 방식이 단순 노이즈만 만든 것은 아니다.
- 다만 daily rank correlation은 크지 않다. 즉, 유의성은 있지만 cross-sectional ranking power는 강하지 않다.
- `image_factor_pc1`은 p-value는 유의하지 않지만 daily rank correlation이 가장 높아, portfolio ranking 신호 후보로는 따로 볼 가치가 있다.

## 5. Ensemble에 추가했을 때의 결과

기존 ensemble에 image factor를 추가해 보았다. aggregation은 기존 연구와 맞춰 날짜별 cross-sectional percentile rank 평균을 사용했다.

| Candidate | Rank corr | Sharpe | 해석 |
|---|---:|---:|---|
| `ensemble_4family` | 0.06738 | 0.5430 | Rank corr 기준 기존 최고 |
| `ensemble_4family + image_factor_pc1` | 0.06722 | 0.3245 | 거의 동일하지만 Sharpe 하락 |
| `ensemble_best + image_factor_pc1` | 0.06547 | 0.4422 | Rank corr는 개선, Sharpe 하락 |
| `ensemble_best` | 0.06055 | 0.6425 | Sharpe 기준 기존 최고 |
| `ensemble_4family + image_score` | 0.05255 | 0.2530 | 성능 하락 |
| `ensemble_best + image_score` | 0.04864 | 0.3917 | 성능 하락 |

`ensemble_best + image_factor_pc1`은 rank correlation을 `0.06055 -> 0.06547`로 올렸다. 하지만 paired bootstrap CI가 `[-0.00533, 0.01538]`로 0을 포함했기 때문에 통계적으로 확실한 개선이라고 말할 수는 없다.

반면 `image_score`는 회귀 검정에서는 유의했지만, ensemble에 단순 rank 평균으로 섞었을 때는 성능이 떨어졌다. 이 부분이 중요하다. "factor로 유의하다"와 "portfolio score로 바로 좋다"는 같은 말이 아니다.

## 6. 최종 해석

이번 결과는 과장하면 안 된다.

잘못된 표현은 다음과 같다.

- "Jiang image factor가 기존 ensemble을 확실히 이겼다."
- "2D CNN image factor만으로 ODE 입력을 대체할 수 있다."
- "이미지 팩터를 넣으면 portfolio 성능이 무조건 좋아진다."

현재 데이터에서 더 정확한 표현은 다음이다.

"Jiang-style image transformation으로 만든 CNN score는 ETF rolling PCA 공통요인을 통제한 뒤에도 유의했다. 이는 가격 경로 이미지에 독립적인 정보가 있음을 시사한다. 다만 이 정보를 기존 ensemble에 단순히 더했을 때 portfolio 성능 개선은 제한적이므로, image factor는 ODE의 기본 mu 입력이라기보다 보조 신호 또는 ablation candidate로 사용하는 것이 적절하다."

## 7. ODE와의 연결

ODE portfolio optimizer는 나중에 자산별 기대수익 `mu(t)`, 공분산 `Sigma(t)`, 위험회피 또는 risk control 신호를 입력으로 받는다.

이번 image factor 결과는 다음 방식으로 연결할 수 있다.

| 용도 | 추천 신호 | 이유 |
|---|---|---|
| 기본 `mu(t)` 입력 | `ensemble_best` | Sharpe 기준 가장 안정적 |
| Rank 중심 비교 입력 | `ensemble_4family` | cross-sectional rank corr 최고 |
| Image factor ablation | `ensemble_best + image_factor_pc1` | rank corr 개선 추세는 있으나 CI는 불확실 |
| 보조 path signal | `image_score`, `image_factor_pc1~3` | PCA control 이후 path 정보 검정에 사용 가능 |

따라서 ODE 단계에서는 `ensemble_best`를 default mu로 두고, `ensemble_best + image_factor_pc1`을 비교 실험으로 넣는 것이 가장 안전하다. 이 비교를 통해 image path 정보가 실제 portfolio weight trajectory를 얼마나 바꾸는지 확인할 수 있다.

## 8. 발표/업로드용 결론 문장

이번 확장은 Jiang-style CNN을 단순 예측 모델이 아니라 ETF 가격 경로에서 추출한 image factor extractor로 재해석한 것이다. 60일 price image를 2D CNN에 넣고 output 직전 FC feature를 PCA factor로 변환한 뒤, rolling return PCA 공통요인을 통제해 검정했다. 결과적으로 `image_score`는 유의했지만, 이를 기존 ensemble에 단순 추가했을 때 portfolio 성능 개선은 제한적이었다. 따라서 현재 결론은 "이미지 경로 정보는 존재하지만, ODE 입력에서는 기본 mu를 대체하기보다 보조 신호와 ablation candidate로 쓰는 것이 적절하다"이다.

## 9. 관련 산출물

- `build_image_factor_extension.py`
- `src/models/cnn.py`
- `ode_inputs_cnn/image_factor_extension/image_factor_report.md`
- `ode_inputs_cnn/image_factor_extension/image_factor_significance.csv`
- `ode_inputs_cnn/image_factor_extension/ensemble_image_factor_search.csv`
- `ode_inputs_cnn/image_factor_extension/image_factor_signals.csv`
- `ode_inputs_cnn/image_factor_extension/ode_mu_candidate_signals.csv`
