# Paper Draft — 이미지 기반 ETF μ 시그널 sprint 결과 정리

이미지 기반 ETF μ-시그널 sprint 의 paper section 초안. 정직한 통계 framing 까지
포함. `REPRODUCIBILITY.md` 의 방법론과 짝으로 읽으면 됨.

---

## Abstract

본 연구는 2014–2026 기간의 7개 ETF universe 에서 cross-sectional 기대수익률
예측 문제를 다룬다. OOS 2,880일에 대해 동일한 walk-forward fold 위에서
5개 모델 family — logistic baseline, Jiang 스타일 chart image 기반 1D/2D CNN,
시퀀스 기반 LSTM, CNN+LSTM hybrid, gradient-boosted tree — 를 비교한다.
세 가지 일관된 결과를 얻었다.

**(i)** autocorrelation-honest stationary block bootstrap 으로 본 유일한
통계적으로 robust 한 lift 는 특정 deep model 의 기여가 아니라 **image
representation 자체**에서 나온다.

**(ii)** image-enabled tier 안에서 추가적인 lift 는 **상관이 낮은 model
family 의 ensemble** 에서 발생한다 (logistic + CNN + LSTM 의 혼합). 우리가
선택한 ensemble 의 멤버는 **단독 성능 1위가 아닌 모델** 을 포함한다.

**(iii)** Naive 한 scale-up — 30개 추가 ETF supervised pretraining, raw
chart image 를 트리 모델에 투입 — 은 오히려 성능을 악화시키며, **representation
–task matching 이 transfer 의 필요조건**임을 보인다.

Block bootstrap CI 는 tier-내 ensemble lift 에서 0 을 포함하므로, 우리는 그
부분을 strict statistical proof 가 아니라 diversification mechanism 과
일관된 positive trend evidence 로 framing 한다.

---

## 1. 문제 설정

7개 ETF (`alternative`, `corp_bond_ig`, `developed_equity`, `emerging_equity`,
`korea_equity`, `short_treasury`, `treasury_7_10y`) 에 대해 20-day forward
log-return 을 예측한다. 대상 기간: 2014-09-24 ~ 2026-03-09 (OOS 2,880일).
입력은 자산별 60일 rolling window. 다운스트림은 **constant risk aversion**
(γ(t) = γ₀) 의 ODE-based dynamic mean-variance 최적화이며, 이 setup 에서
최적해는 `w* ∝ Σ(t)⁻¹ μ(t)` 이므로 input 은 (μ(t), Σ(t), realised returns R)
이다.

평가는 날짜별 Spearman rank correlation 의 평균과 top-k=2 portfolio Sharpe
(20일마다 rebalance, 연환산) 를 사용한다. 통계적 추론은 **paired stationary
block bootstrap** (Politis–Romano, mean block length 20 = horizon) 을 적용
한다. 일별 rank-correlation 시계열은 20일 horizon overlap 때문에 자기상관을
가지므로, 자기상관을 무시하는 IID resampling 의 CI 는 너무 좁다.

## 2. 핵심 결과 세 가지

### 2.1 Representation > Model (Pillar 1)

2×2 ablation 으로 image transformation 효과와 model class 효과를 분리한다.
같은 OOS grid, 같은 logistic head 에서:

|              | numerical input | image input  |
|--------------|----------------:|-------------:|
| **logistic** |        −0.0072  |  **+0.0392** |
| **CNN**      |        +0.0271  |     +0.0280  |

logistic 머리를 고정한 채 numerical → image 로 입력을 바꾸면 rank corr 가
+0.046 상승한다. 반면 image 입력을 고정한 채 logistic → CNN 으로 모델을
바꾸면 추가 lift 는 거의 0 이다.

raw OHLCV baseline (`logistic_cumulative_scale`, no-image floor) 대비
ensemble lift 는 block bootstrap **+0.075, 95% CI [+0.013, +0.137]**
(0 배제) 로 **이 sprint 의 유일한 statistically robust claim** 이다.
image-enabled tier 안에서의 lift 는 점추정으로는 양수이지만 block CI 가
0 을 포함한다 (§3 참조).

### 2.2 Ensemble lift = correlation-structure lift (Pillar 2)

image-enabled tier 의 단일 모델 rank corr 는 모두 0.04 – 0.05 구간에 몰려
있어 통계적으로 구분 불가하다. 체계적 추가 lift 는 **상관이 낮은 family 끼리의
rank-mean ensemble** 에서 발생한다.

| Phase | 멤버 | Rank corr | Top-k Sharpe |
|-------|------|----------:|-------------:|
| 1     | CNN-only top-3 | 0.0375 | 0.275 |
| 3     | logistic + 1D-CNN + 2D-CNN | 0.0606 | **0.6425** |
| 4     | logistic + 1D-CNN + 2D-CNN + CNN+LSTM | **0.0674** | 0.5430 |

raw-signal correlation 의 구조:

| 쌍 | ρ |
|---|---:|
| CNN-CNN (같은 family 내) | 0.5 ~ 0.6 |
| CNN-LSTM | 0.03 ~ 0.18 |
| **cnnlstm × logistic_image** | **−0.17** |

특히 주목할 점: 우리가 선택한 4-family ensemble 은 **단독 rank corr 1위인
`lstm_image_scale` (0.051) 을 포함하지 않고**, **단독 4위인
`cnnlstm_image_scale` (0.045) 을 포함**한다. 결정 요인은 후자의 logistic
member 와의 **음의 상관 (−0.17)** 이며, 이는 전자에는 없다. 즉
**ensemble lift 는 단일 모델 강도가 아니라 신호 간 상관 구조에 의해 결정**
된다는 직접적 empirical 증거다.

### 2.3 Scale 은 free lunch 가 아니다 — representation–task matching (Pillar 3)

두 가지 scale-up 시도 모두 **negative transfer** 를 보였다.

- **30-ETF supervised pretraining.**
  CNN backbone 을 target 7개 외 30개 추가 ETF (다른 sector / region /
  asset class) 에서 supervised pretrain 후 7-ETF 로 fine-tune. 동일 grid
  비교에서 rank corr 가 baseline **+0.0311 → −0.0752** 로, Sharpe 도
  0.124 → −0.028 로 크게 악화. 원인은 task / distribution mismatch 로
  추정된다: pretraining 목표는 이질적 universe 위의 자산별 수익 예측인 데
  반해, 다운스트림 task 는 고정된 좁은 basket 위의 cross-sectional ranking
  이다. 모델 representation 이 basket-specific dynamic 에서 멀어지면서
  성능이 떨어진다. 이는 robustness negative result 로 기록되며, 주 파이프
  라인을 부정하지 않지만 "데이터가 많을수록 무조건 낫다" 식 주장을 차단
  한다.

- **Gradient-boosted trees on raw price image.**
  트리는 개별 feature 분기에 기반하며 공간·시간적 inductive bias 가 없다.
  flatten 된 Jiang 차트 픽셀을 XGBoost 에 그대로 넣으면 rank corr 가
  −0.011 (`xgb_image_scale`) 로 나오고, 누적수익 입력 변형 (`xgb_cumulative_scale`)
  도 +0.019 에 그쳐 모든 family 중 최하위다. 트리는 핸드오프에서 제외하며,
  이는 **representation – model 의 inductive bias 일치** 가 필요하다는
  negative result 이다. image representation 은 그것을 읽어낼 수 있는
  모델과 짝일 때만 유효하다.

이 두 결과는 Pillar 1 을 한 단계 더 sharp 하게 만든다: "데이터 / 모델
capacity 가 더 많아서" 가 아니라 "올바른 representation 과 그것을 활용
가능한 모델의 짝" 이 lift 의 원천이다.

## 3. 정직한 통계 framing

위 §2 의 headline 숫자는 점추정과 ensemble-search 의 winner 를 사용한다.
둘 다 autocorrelation 과 post-selection bias 의 위험에 노출되어 있다.
IID 와 stationary block bootstrap CI 를 모두 보고하되, block CI 를
honest verdict 로 채택한다.

| 비교 | mean diff | IID 95% CI | **block 95% CI (honest)** | verdict |
|---|---:|---|---|---|
| ensemble_4family vs `logistic_cumulative` (raw floor) | +0.075 | [+0.053, +0.096] | **[+0.013, +0.138]** | ✅ significant |
| ensemble_best vs `logistic_image` (image baseline) | +0.021 | [+0.007, +0.035] | **[−0.015, +0.059]** | not significant |
| ensemble_4family vs `logistic_image` | +0.028 | [+0.013, +0.043] | **[−0.012, +0.068]** | not significant |
| Phase 3 → Phase 4 (within-ensemble) | +0.007 | [−0.002, +0.016] | **[−0.015, +0.030]** | not significant |
| image_factor_pc1 marginal lift on ensembles | +0.010 ~ +0.013 | [≈0, +0.025] | **[−0.027, +0.051]** | not significant |

두 가지 보정이 중요하다.

1. block bootstrap CI 는 IID 대비 약 3배 넓다. 20일 forward return horizon
   이 일별 시계열에 양의 자기상관을 유발하기 때문이다. 이전 IID 단계에서
   "significant" 로 잡혔던 ensemble-vs-image-baseline 과 image_factor
   marginal lift 들은 자기상관 artifact 였다.

2. Phase 3 / Phase 4 winner 는 2,940개 ensemble-search 조합 중에서
   선택되었다. block CI 도 이 post-selection 을 따로 보정하지는 않는다.
   Hansen 의 Superior Predictive Ability test 같은 정식 multiple-testing
   보정을 추가하면 CI 가 더 넓어지지만, 정성적 결론 (strict significance
   부족) 은 바뀌지 않는다.

따라서: **이 sprint 의 strict 한 통계적 claim 은 raw OHLCV floor 대비
이미지 변환 lift 하나** 다. image-enabled tier 안에서의 lift 는 점추정으로
양수이고 Pillar 2 의 mechanism 과 일관되나, honest CI 하에서는 strict
statistical proof 가 아니다.

## 4. 다운스트림 ODE 백테스트로 넘기는 질문들

위 통계만으로는 풀리지 않고 ODE 백테스트에서 empirical 하게 답해야 하는 질문
세 가지:

1. `logistic_image` 대비 ensemble 의 점추정 양의 lift 가 simplex constraint
   하의 ODE 에서 realised PnL 개선으로 이어지는가?
2. `ensemble_4family` (Phase 4, rank-prio) 가 `ensemble_best`
   (Phase 3, Sharpe-prio) 보다 realised Sharpe 가 높은가?
3. Ledoit–Wolf shrinkage 된 Σ 가 sample covariance 의 high-condition-number
   윈도우에서 발생하는 weight 폭발을 실제로 제거하는가?

핸드오프는 이 비교가 가능하도록 13개 μ candidate (이미지 팩터 연구
candidate 5개 + reference / baseline candidate 8개; raw floor ~ best single
model 전 스펙트럼) 와 4개 Σ 변형 (sample, Ledoit–Wolf shrunk, EWMA,
multi-window blend) 을 동일 날짜 grid 위에 제공한다.

## 5. 재현성

모든 산출물은 repo 스크립트로 재현 가능하다 (`REPRODUCIBILITY.md` 참조).
데이터 무결성은 독립 재계산으로 검증되었고, walk-forward fold 는 모든
family 가 동일한 deterministic generator 를 공유하며, 모든 stochastic
단계는 fixed seed 를 쓴다. 알려진 한계 세 가지가 명시되어 있다: (a)
walk-forward fold 의 horizon embargo 부재 (~1–2 % 학습 행 오염, 모든
모델에 symmetric); (b) 원본 calibration 스크립트 부재를 horizon-embargo
수정과 함께 복원; (c) image-factor 보고서들의 IID bootstrap CI 와 honest
block CI 를 병기.

## 6. 한계와 향후 작업

- 2,880 OOS 일수는 우리가 보고자 하는 CI 폭에 비해 짧다. 더 긴 historical
  데이터가 가용해지면 block CI 가 좁아져 현재 positive trend 가 formal
  significance 로 전환될 가능성이 있다.
- 임시방편적 "2,940 조합 중 winner" framing 을 Hansen SPA / Romano-Wolf
  step-down 으로 대체하면 post-selection 을 정식으로 보정할 수 있다.
- 엔지니어링된 tabular feature (다중 lag 모멘텀, realised volatility,
  drawdown, volume z-score) 를 사용하면 tree family 에 공정한 representation
  을 제공할 수 있고, 현재 negative XGBoost 결과를 5번째 family 의 긍정적
  기여로 전환할 가능성이 있다.
- 학습 / val / test 사이에 `horizon` 길이의 purge gap 을 삽입하고 walk-
  forward 전체를 재실행하면 embargo 한계가 해소된다.
- 30-ETF universe 위의 self-supervised pretraining (next-step prediction 또는
  contrastive objective) 후 target 7 로 fine-tune 하는 방식은 현재의
  supervised pretraining negative transfer 를 진짜 gain 으로 바꿀 가능성이
  있다.

## 7. 결론

본 sprint 의 empirical 결론은 일관적이다. **가격 경로의 이미지 representation
은 raw OHLCV 입력 대비 statistically robust 한 cross-sectional 수익
예측 lift 의 원천이다. image-enabled tier 안에서는 model architecture 의
한계수익이 빠르게 체감하며, 남은 lift 는 상관이 낮은 model family 끼리의
ensemble 에서 발생한다.** Naive 한 scale-up — 더 넓은 pretraining
universe 나 raw image 위의 tree 모델 — 은 성능을 개선하지 않고 악화시킨다.
representation lift 는 strict 한 통계적 significance 가 입증되었으며,
tier-내 ensemble lift 는 점추정으로 양수이되 autocorrelation-honest
confidence interval 안에 머문다. 후자의 실증 검증은 다운스트림 ODE
백테스트로 넘긴다.
