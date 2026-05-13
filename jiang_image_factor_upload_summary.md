# ETF `mu(t)` 예측을 위한 CNN/Ensemble 연구 정리

## 0. 전체 요약

우리 팀의 최종 목표는 ETF 자산에 대해 ODE 기반 동적 포트폴리오 최적화를 하는 것이다. 이때 ODE optimizer에는 자산별 기대수익률 `mu(t)`가 필요한데, 내 역할은 CNN과 image-based signal을 이용해 이 `mu(t)` 후보를 만드는 것이다.

처음에는 여러 CNN 변형을 단일 모델로 비교했다. 하지만 단일 CNN 하나가 baseline을 압도하는 구조는 아니었다. 오히려 중요한 결과는 **서로 다른 model family의 신호를 ensemble로 섞을 때 성능의 천장이 올라간다**는 점이었다.

그래서 연구 흐름은 다음과 같이 정리된다.

1. 선행연구를 바탕으로 가격 경로를 이미지로 변환한다.
2. Logistic, 1D CNN, 2D CNN, LSTM, CNN-LSTM을 같은 OOS grid에서 비교한다.
3. 단일 모델보다 family 간 상관구조가 더 중요하다는 점을 확인한다.
4. `ensemble_best`와 `ensemble_4family`를 ODE용 `mu(t)` 후보로 만든다.
5. Jiang-style 2D CNN 내부 feature를 `image factor`로 추출해 추가 검정한다.

최종 결론은 다음과 같다. **ODE 기본 `mu(t)` 후보는 아직 `ensemble_best`가 가장 안전하고, Jiang-style image factor는 독립적인 가격 경로 정보가 있음을 보여주는 보조 신호와 ablation candidate로 쓰는 것이 적절하다.**

## 1. 연구의 출발점

ODE 포트폴리오 최적화는 매 시점마다 각 ETF에 얼마를 투자할지 결정하는 문제다. 이때 필요한 입력은 크게 세 가지다.

| 입력 | 의미 | 이번 연구와의 관계 |
|---|---|---|
| `mu(t)` | 각 자산의 기대수익률 | CNN/ensemble 신호로 만들 후보 |
| `Sigma(t)` | 자산 간 공분산 | 과거 수익률로 계산 가능 |
| risk/control signal | 위험회피 조절 신호 | image factor가 보조 후보가 될 수 있음 |

`Sigma(t)`는 과거 수익률로 비교적 직접 계산할 수 있지만, `mu(t)`는 미래 수익률과 관련된 예측 신호가 필요하다. 그래서 이번 sprint에서는 ETF 가격 경로를 이미지로 바꿔 CNN에 넣고, 이 신호가 `mu(t)` 후보로 의미가 있는지 확인했다.

## 2. 선행연구에서 가져온 아이디어

이번 연구는 완전히 새 모델을 임의로 만든 것이 아니라, 세 가지 선행연구 흐름을 ETF 데이터에 맞게 연결한 것이다.

| 선행연구 흐름 | 가져온 아이디어 | 이번 연구에서의 적용 |
|---|---|---|
| Jiang-style price image | 가격 경로를 이미지로 변환해 CNN에 입력 | ETF OHLC 60일 window를 2D chart image로 변환 |
| Image-based asset pricing / DV-style 해석 | 모델 출력을 단순 예측값이 아니라 characteristic/factor로 해석 | CNN 내부 FC feature를 image factor로 추출 |
| ODE portfolio optimization | `mu(t)`, `Sigma(t)`, risk control을 이용해 동적 비중 결정 | ensemble score를 `mu(t)` 후보로 저장 |

### 2.1 Jiang-style price image

사용자가 보낸 Jiang 그림은 주가 데이터를 숫자 시계열로만 보지 않고, 일정 기간의 OHLC 경로를 작은 이미지로 바꿔 CNN에 넣는 구조를 보여준다.

예를 들어 5일, 20일, 60일 가격 흐름을 각각 이미지로 만들고, CNN이 그 이미지 안의 추세, 반전, 변동성, 꼬리 모양 같은 시각적 패턴을 학습한다. 마지막에는 FC layer와 softmax 또는 output layer를 거쳐 미래 수익 방향이나 수익률을 예측한다.

중요한 점은 CNN의 최종 예측값만 보는 것이 아니라, output 직전의 FC representation을 가격 경로의 압축된 이미지 특성으로 볼 수 있다는 것이다. 이번 확장은 바로 이 부분을 ETF 연구에 적용한 것이다.

### 2.2 Image-based asset pricing 관점

Image-based asset pricing 계열 연구의 핵심은 이미지 모델의 출력이 단순한 black-box prediction score가 아니라, 가격 경로에서 추출된 characteristic 또는 factor로 해석될 수 있다는 점이다.

이번 연구에서는 이 관점을 사용해 `cnn_2d_residual_small`의 output 직전 64차원 FC feature를 뽑고, 이를 PCA로 압축해 `image_factor_pc1~3`을 만들었다. 즉 CNN을 단순 예측 모델이 아니라 **가격 경로 factor extractor**로 사용했다.

### 2.3 ODE portfolio optimization 관점

ODE optimizer는 최종적으로 `mu(t)`와 `Sigma(t)`를 받아 portfolio weight path를 계산한다. 따라서 내 CNN 연구의 산출물은 단순 accuracy 표가 아니라, 나중에 ODE가 바로 읽을 수 있는 asset-date별 signal panel이어야 한다.

그래서 최종 산출물은 `date`, `asset`, `model_name`, `signal_value` 형태로 저장했고, `ensemble_best`, `ensemble_4family`, image factor 후보를 모두 ODE 입력 후보로 넘길 수 있게 만들었다.

## 3. 전체 연구 흐름: 단일 모델에서 ensemble까지

기존 연구는 모델의 prediction score와 ensemble 성능 중심이었다.

기존 핵심 결론은 다음과 같았다.

- CNN, Logistic, LSTM, CNN-LSTM 단독 모델은 대부분 비슷한 성능 구간에 있었다.
- 하지만 서로 상관이 낮은 model family를 섞으면 rank correlation의 천장이 올라갔다.
- `ensemble_best`는 Sharpe 기준으로 강했고, `ensemble_4family`는 rank correlation 기준으로 강했다.

이번 Jiang-style 확장은 여기에 질문을 하나 더 추가한다.

"CNN이 만든 예측값이 아니라, CNN 내부 feature 자체가 가격 경로에서 나온 유의미한 image factor인가?"

즉, 기존 연구가 "어떤 모델이 예측을 잘하나"였다면, 이번 확장은 "이미지로 변환된 가격 경로가 독립적인 factor 정보를 갖고 있나"를 검정한 것이다.

전체 진행은 다음 순서였다.

| 단계 | 내용 | 핵심 결과 |
|---|---|---|
| Baseline | Logistic + cumulative/image scaling | 단순 모델도 생각보다 강함 |
| CNN 탐색 | 1D CNN, attention, dilated, 2D CNN 등 비교 | 단일 CNN이 압도적으로 이긴 것은 아님 |
| 2D CNN 재조정 | `cnn_2d_residual_small`로 capacity/regularization 조정 | 2D CNN이 image extractor로 쓸 수 있는 수준으로 개선 |
| Mixed-family ensemble | Logistic + 1D CNN + 2D CNN 조합 | `ensemble_best`가 Sharpe 기준 강함 |
| LSTM/CNNLSTM 합류 | 시간 순서 정보를 보는 family 추가 | `ensemble_4family`가 rank corr 기준 최고 |
| Image factor 확장 | 2D CNN 내부 feature를 factor로 검정 | `image_score`가 PCA controls 이후에도 유의 |

이 과정에서 가장 중요한 교훈은 **단일 CNN 구조를 계속 바꾸는 것보다 서로 다른 family의 신호를 조합하는 것이 더 큰 개선을 만든다**는 점이었다.

## 4. CNN 모델을 줄인 기준

초기에는 CNN 모델이 많았다. 1D CNN, multiscale CNN, dilated CNN, attention CNN, 2D rendered image CNN, 2D residual CNN 등을 실험했다.

하지만 최종 제출에서는 모델을 많이 보여주는 것이 오히려 메시지를 흐린다. 그래서 CNN을 다음 두 역할만 남겼다.

| 남긴 모델 | 역할 | 이유 |
|---|---|---|
| `cnn_1d_cumulative_scale` | 기존 ensemble의 CNN member | `ensemble_best`에 실제로 기여 |
| `cnn_2d_residual_small` | Jiang-style image factor extractor | 실제 2D chart image를 쓰는 구조와 가장 가까움 |

나머지 CNN 변형과 중간 walk-forward 결과는 `archive/model_exploration/`으로 옮겼다. 삭제한 것은 아니고, 필요할 때만 확인하는 탐색 기록으로 분리한 것이다.

## 5. Jiang-style image factor 챕터: 구현 방법

메인 모델은 `cnn_2d_residual_small`로 고정했다. 이유는 Jiang 그림처럼 실제 2D chart image를 CNN에 넣는 구조와 가장 가깝고, 기존 Phase 2 실험에서 2D CNN을 재조정했을 때 단일 CNN rank correlation이 개선됐기 때문이다.

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

## 6. Ensemble을 만들게 된 이유와 최종 후보

처음에는 CNN 모델 자체의 성능을 높이는 것이 목표처럼 보였다. 하지만 실험 결과, 대부분의 단일 모델은 비슷한 성능 구간에 몰려 있었다. 그래서 질문이 바뀌었다.

```text
어떤 단일 모델이 제일 좋은가?
```

보다 중요한 질문은 다음이었다.

```text
서로 다른 정보를 보는 모델을 어떻게 섞으면 ODE용 mu 후보가 더 안정적인가?
```

모델 간 score correlation을 확인했을 때, 같은 CNN family끼리는 서로 비슷하게 움직이는 경향이 있었다. 반면 Logistic, 1D CNN, 2D CNN, CNN-LSTM처럼 입력 표현과 구조가 다른 family끼리는 상관이 낮았다. 이 낮은 상관이 ensemble의 핵심 근거가 됐다.

최종적으로 중요한 ensemble 후보는 두 개다.

| 후보 | 구성 | 역할 |
|---|---|---|
| `ensemble_best` | Logistic + 1D CNN + 2D CNN | Sharpe 기준 ODE 기본 `mu(t)` 후보 |
| `ensemble_4family` | Logistic + 1D CNN + 2D CNN + CNN-LSTM | rank correlation 기준 비교 후보 |

실험상 `ensemble_best`는 portfolio Sharpe 기준으로 가장 안정적이었고, `ensemble_4family`는 cross-sectional rank correlation 기준으로 가장 높았다. 따라서 ODE 단계에서는 하나만 고정하기보다 두 후보를 같이 넘겨서 portfolio trajectory 차이를 비교하는 것이 좋다.

## 7. Image factor 유의성 결과

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

## 8. Image factor를 ensemble에 추가했을 때의 결과

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

## 9. 최종 해석

이번 결과는 과장하면 안 된다.

잘못된 표현은 다음과 같다.

- "Jiang image factor가 기존 ensemble을 확실히 이겼다."
- "2D CNN image factor만으로 ODE 입력을 대체할 수 있다."
- "이미지 팩터를 넣으면 portfolio 성능이 무조건 좋아진다."

현재 데이터에서 더 정확한 표현은 다음이다.

"Jiang-style image transformation으로 만든 CNN score는 ETF rolling PCA 공통요인을 통제한 뒤에도 유의했다. 이는 가격 경로 이미지에 독립적인 정보가 있음을 시사한다. 다만 이 정보를 기존 ensemble에 단순히 더했을 때 portfolio 성능 개선은 제한적이므로, image factor는 ODE의 기본 mu 입력이라기보다 보조 신호 또는 ablation candidate로 사용하는 것이 적절하다."

## 10. ODE와의 연결

ODE portfolio optimizer는 나중에 자산별 기대수익 `mu(t)`, 공분산 `Sigma(t)`, 위험회피 또는 risk control 신호를 입력으로 받는다.

이번 image factor 결과는 다음 방식으로 연결할 수 있다.

| 용도 | 추천 신호 | 이유 |
|---|---|---|
| 기본 `mu(t)` 입력 | `ensemble_best` | Sharpe 기준 가장 안정적 |
| Rank 중심 비교 입력 | `ensemble_4family` | cross-sectional rank corr 최고 |
| Image factor ablation | `ensemble_best + image_factor_pc1` | rank corr 개선 추세는 있으나 CI는 불확실 |
| 보조 path signal | `image_score`, `image_factor_pc1~3` | PCA control 이후 path 정보 검정에 사용 가능 |

따라서 ODE 단계에서는 `ensemble_best`를 default mu로 두고, `ensemble_best + image_factor_pc1`을 비교 실험으로 넣는 것이 가장 안전하다. 이 비교를 통해 image path 정보가 실제 portfolio weight trajectory를 얼마나 바꾸는지 확인할 수 있다.

## 11. 발표/업로드용 결론 문장

이번 연구는 ETF ODE optimizer에 넣을 `mu(t)` 후보를 만들기 위해 CNN 계열 모델과 ensemble을 비교한 작업이다. 단일 CNN 하나가 압도적으로 이긴 것이 아니라, Logistic, CNN, LSTM/CNN-LSTM처럼 서로 다른 family를 조합할 때 신호의 안정성이 좋아졌다. 그중 Jiang-style image factor 챕터에서는 60일 price image를 2D CNN에 넣고 output 직전 FC feature를 PCA factor로 변환한 뒤, rolling return PCA 공통요인을 통제해 검정했다. 결과적으로 `image_score`는 유의했지만, 이를 기존 ensemble에 단순 추가했을 때 portfolio 성능 개선은 제한적이었다. 따라서 현재 결론은 `ensemble_best`를 ODE 기본 `mu(t)` 후보로 사용하고, image factor는 가격 경로 정보의 존재를 보여주는 보조 신호와 ablation candidate로 사용하는 것이다.

## 12. 최종 폴더 구조

최종 제출/발표에서 볼 핵심 폴더는 다음과 같다.

| 폴더 | 의미 |
|---|---|
| `ode_inputs_cnn/ensemble_best` | ODE 기본 `mu(t)` 후보 |
| `ode_inputs_cnn/ensemble_4family` | rank correlation 기준 비교 후보 |
| `ode_inputs_cnn/cnn_1d_cumulative_scale` | 최종 1D CNN 대표 |
| `ode_inputs_cnn/cnn_2d_residual_small` | 최종 2D CNN image factor extractor |
| `ode_inputs_cnn/image_factor_extension` | image factor 검정 및 ODE 후보 신호 |
| `archive/model_exploration` | 최종에서 제외한 CNN 탐색 기록 |

## 13. 관련 산출물

- `build_image_factor_extension.py`
- `src/models/cnn.py`
- `ode_inputs_cnn/image_factor_extension/image_factor_report.md`
- `ode_inputs_cnn/image_factor_extension/image_factor_significance.csv`
- `ode_inputs_cnn/image_factor_extension/ensemble_image_factor_search.csv`
- `ode_inputs_cnn/image_factor_extension/image_factor_signals.csv`
- `ode_inputs_cnn/image_factor_extension/ode_mu_candidate_signals.csv`

## 14. 앞으로 할 일

현재 sprint2 폴더는 **이미지 팩터가 실제로 의미 있는지 확인한 실험 단계**로 보면 된다. 앞으로의 연구는 크게 두 축으로 이어진다.

### 14.1 Image Factor 확장

원래 주식 데이터라면 CAPM, FF3, FF4 같은 factor model에 image factor를 추가해서 유의성을 확인하는 방식이 자연스럽다. 하지만 이번 데이터는 개별 주식이 아니라 7개 ETF/지수성 자산이다. 이 구조에서는 FF factor를 그대로 적용하기 어렵다.

그래서 현재 sprint2에서는 대체 방법으로 rolling PCA control을 사용했다.

```text
ETF return에서 공통요인 PCA 추출
    -> PC loading 1~3을 control로 사용
    -> 여기에 CNN image feature 추가
    -> image factor가 유의한지 확인
```

이 검정에서 유효성이 있으면, 단순히 과거 수익률 평균이 아니라 **가격 path 정보 자체가 의미 있는 신호**라고 해석할 수 있다.

앞으로 더 해야 할 일은 다음과 같다.

- 이미지 생성 방식 자체를 다양화한다.
- ETF 구조에 맞는 factor-model-style 검정을 더 정교화한다.
- CAPM/FF3/FF4를 직접 쓰기 어려운 이유를 명확히 설명한다.
- PCA common factor 통제 후에도 image factor가 유의한지 반복 검증한다.
- 유의한 image factor를 바탕으로 `mu(t)` 시계열을 더 안정적으로 추출한다.

이미지 생성 방식의 ablation 후보는 다음과 같다.

| 후보 | 목적 |
|---|---|
| MA선 포함 vs 제외 | 추세선 정보가 추가 설명력을 갖는지 확인 |
| 거래량 포함 vs 제외 | volume path가 가격 path 외 정보를 주는지 확인 |
| OHLC 전체 사용 | 캔들 구조의 고가/저가/시가/종가 정보 반영 |
| close-only 이미지 | 단순 종가 path만으로도 충분한지 비교 |
| high-low range 강조 | 변동성/꼬리 정보가 factor로 유효한지 확인 |

이 ablation의 목적은 단순히 CNN 입력을 늘리는 것이 아니라, **과거 가격 path의 어떤 시각적 요소가 실제로 정보성을 갖는지 분해하는 것**이다.

### 14.2 Optimization 확장

Image Factor 단계에서 만든 시계열 예측 신호는 이후 ODE portfolio optimization의 입력으로 들어간다.

Optimization 단계의 핵심은 다음과 같다.

| 입력 | 생성 방식 |
|---|---|
| `mu(t)` | CNN/ensemble/image factor 기반 예측 신호 |
| `Sigma(t)` | 과거 ETF return으로 rolling covariance 추정 |
| risk aversion | 초기에는 임의의 고정값 또는 단순 schedule 사용 |
| asset class constraint | 실제 기금 운용처럼 자산 class별 제약 적용 가능 |

이후에는 portfolio 성과를 비교해야 한다.

비교 대상은 다음과 같다.

- 단순 6:4 포트폴리오
- Mean-variance portfolio
- ODE dynamic portfolio
- `ensemble_best` 기반 ODE
- `ensemble_4family` 기반 ODE
- image factor를 추가한 ODE ablation

따라서 전체 연구의 최종 흐름은 다음과 같다.

```text
가격 path image
    -> CNN / image factor
    -> mu(t) 시계열 추출
    -> Sigma(t), risk aversion과 결합
    -> ODE portfolio optimization
    -> 6:4, Mean-variance와 성과 비교
```

즉 sprint2는 최종 optimization 자체가 아니라, 그 전에 필요한 **이미지 기반 `mu(t)` 후보를 만들고 검증한 단계**다.
