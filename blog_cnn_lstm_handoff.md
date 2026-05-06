# 주가 차트를 이미지로 바꿔서 CNN에 넣어봤다 — 그런데 베이스라인과 같은 티어였다

> 7개 ETF 자산의 μ 예측에 CNN 7종 + LSTM/hybrid 4종을 붙였다. 결과를 한 줄로 요약하면: **단독 성능은 모두 비슷한 티어, 하지만 상관이 낮은 family 끼리 섞을 때마다 천장이 올라갔다**. 이 글은 "**이겼다/졌다**"가 아니라 "**상관구조를 봐야 보인다**"는 교훈, 그리고 그 thesis 가 LSTM 합류 후에도 일관되는지의 검증 기록.
>
> **최종 결과**:
> - Phase 3 — `ensemble_best` (logistic + 1D CNN + 2D CNN): rank corr **0.0606** / Sharpe **0.643**
> - Phase 4 — `ensemble_4family` (+ cnnlstm hybrid): rank corr **0.0674** / Sharpe 0.543 (rank corr 천장 갱신, Sharpe trade-off)
> - Bootstrap CI [−0.002, +0.016] 0 살짝 포함 → borderline NOT significant
> - 단 cnnlstm vs logistic ρ = **−0.17 (음의 상관)** 으로 mechanism evidence 강하게 일관

---

## 1. 시작 — 뭘 만들려고 했나

요즘 한 팀에서 **ODE(미분방정식) 기반 동적 포트폴리오 최적화**를 다루고 있다. 쉽게 말하면:

> "매일매일 시장이 변하는데, 7개 자산(주식·채권·대체투자 등)에 **얼마씩 넣을지를 수식으로 풀어서 정한다**"

이 수식엔 세 가지 입력이 필요하다.

- **μ(mu)**: 각 자산이 앞으로 얼마나 오를지 (기대수익)
- **Σ(sigma)**: 자산들이 같이 움직이는 정도 (공분산)
- **R**: 실제 일별 수익률

Σ·R은 과거 데이터로 기계적으로 나온다. 어려운 건 **μ** — 미래를 맞춰야 하니 완벽은 불가. 다만 **랭킹(어떤 게 더 오를 것 같은지)**만 잘 맞아도 포트폴리오엔 도움이 된다.

나는 이 팀에서 **μ 예측 시그널을 CNN으로 만드는 파트**를 맡았다.

---

## 2. Jiang-style 이미지 변환 — 주가를 그림으로

보통 주가 예측 모델은 숫자 시퀀스를 그대로 넣는다. `[100, 102, 101, 103, ...]` 같은 식.

그런데 2016년 Jiang이라는 연구자가 이런 제안을 했다:

> "**사람 트레이더도 차트를 보고 판단한다**. 그럼 모델도 차트 이미지를 보게 하자"

그래서 60일치 OHLCV(시가·고가·저가·종가·거래량) 데이터를 **정규화된 2D 이미지**로 변환한다. 가로축은 시간, 세로축은 가격 레벨, 픽셀 값은 캔들 모양·거래량 강도 등으로 채운다.

```
숫자 sequence  →  이미지  →  CNN (컴퓨터 비전 모델)  →  μ 예측값
```

CNN은 원래 고양이·강아지 사진 구분용으로 유명한 모델이다. 그걸 주가 차트에 붙이는 게 Jiang-style의 아이디어.

---

## 3. 실험 설계 — CNN 7종 + 로지스틱 베이스라인 2종

"어떤 CNN 구조가 이 문제에 제일 잘 맞을까?"를 보려 7가지 변형을 준비했다. 핵심 설계 결정 하나 — **같은 평가 그리드에 로지스틱 회귀 2종도 올려놨다**. 단순 베이스라인이 얼마나 나오는지 모르면, CNN 내부 순위는 의미가 없다.

| # | 모델 | 입력 | 계열 |
|---|---|---|---|
| 0 | `logistic_cumulative_scale` | 누적 수익률 | 베이스라인 |
| 0 | `logistic_image_scale` | Jiang 이미지 | 베이스라인 (image) |
| 1 | `cnn_1d_image_scale` | 이미지 (flat) | CNN 1D |
| 2 | `cnn_1d_multiscale_image_scale` | 이미지 | CNN 1D |
| 3 | `cnn_1d_dilated_image_scale` | 이미지 | CNN 1D |
| 4 | `cnn_1d_attention_image_scale` | 이미지 + SE | CNN 1D |
| 5 | `cnn_1d_cumulative_scale` | 누적 수익률 | CNN (no image) |
| 6 | `cnn_2d_rendered_images` | 2D 캔들 이미지 | CNN 2D |
| 7 | `cnn_2d_residual_images` | 2D + ResNet | CNN 2D |

모두 **walk-forward out-of-sample**으로 평가. 풀어서 설명하면:

> "2015년까지 학습 → 2016년 예측 → 2016년까지 학습 → 2017년 예측..." 식으로 **미래를 본 적 없는 상태에서만 예측하게** 한 것. 금융 모델링의 가장 흔한 함정이 leakage라 이걸 엄격히 막았다.

총 **24 fold × 7년 OOS**, 평가 날짜 2,880일.

---

## 4. 결과 — "이겼다/졌다"가 아니라 "같은 티어에서 서로 다른 강점"

![Model comparison](ode_inputs_cnn/figures/01_model_comparison.png)

숫자로 요약하면:

| 지표 1등 | 모델 | 값 |
|---|---|---|
| Rank correlation | `logistic_image_scale` | **0.0392** |
| Top-k Sharpe | `cnn_1d_cumulative_scale` | **0.521** |

읽는 법:
- **Rank corr (점예측 품질)**: 단순 로지스틱 + 이미지가 최고 CNN(0.028)보다 살짝 앞섬 (0.039 vs 0.028) — 단, 격차는 **noise 수준**
- **Top-k Sharpe (포트폴리오 품질)**: CNN이 확실히 앞섬 (0.52 vs 0.39)
- **두 metric이 서로 다른 챔피언을 가리킴** → "어떤 모델이 이긴다"의 answer는 metric에 따라 달라짐

자세히 보면 **대부분의 모델이 rank corr 0.02~0.04 티어**에 몰려 있다. 최하위(`logistic_cumulative` −0.007)를 제외하면 개별 격차는 대체로 noise 수준. "**CNN이 logistic을 이긴다**"고 말할 수 있는 데이터는 아니었다.

### 2×2 ablation으로 기여 분해

![Ablation](ode_inputs_cnn/figures/09_ablation_image_vs_cnn.png)

| | No image | Image |
|---|---|---|
| **Logistic** | −0.0072 | **+0.0392** |
| **CNN** | +0.0271 | +0.0280 |

- **이미지 변환 효과** (logistic 기준): −0.007 → +0.039 = **+0.046 lift** 🚀 (핵심 기여자)
- **CNN 효과** (no-image 기준): −0.007 → +0.027 = +0.034
- **이미지 위에 CNN 얹는 효과**: +0.039 → +0.028 = **−0.011 (오히려 감소)**

해석을 정직하게 말하면:

1. **Rank corr lift의 대부분은 "이미지 변환"에서 나온다** — Jiang-style의 공이 크다
2. **CNN은 이미지를 쓰든 안 쓰든 rank corr가 비슷**하고, 이미지를 잘 쓰는 건 오히려 로지스틱 쪽
3. **Sharpe 관점에서는 CNN이 앞섬** — 점예측 정확도와 포트폴리오 구성 품질이 분리되는 케이스

**결론**: 단독 성능 줄세우기만으로는 "CNN이 베이스라인보다 확실히 낫다"고 말할 수 없다. 하지만 **두 family가 서로 다른 강점을 가진다는 게 보임**. 그럼 질문은 바뀐다 — "**어떤 모델이 최고냐**"가 아니라 "**이 다른 강점들을 어떻게 합쳐서 쓸 것이냐**".

---

## 5. 진짜 레버는 "상관 구조"에 있다

모델 간 raw score 상관을 찍어보자:

![Model correlation](ode_inputs_cnn/figures/05_model_raw_correlation.png)

- **CNN 1D끼리**: ρ ≈ 0.5~0.6 (비슷한 정보를 다른 방식으로 표현)
- **CNN 2D와 1D**: ρ ≈ 0.25~0.28 (더 독립적)
- **CNN과 logistic**: ρ ≈ 0.2~0.3 (가장 독립적)

앙상블 이론의 핵심: **신호 자체의 강도보다 "신호 간 상관 구조"가 결과를 지배한다.** 상관 낮은 두 신호를 섞으면 variance가 줄어서 Sharpe·rank corr 모두 좋아진다.

### CNN-only 앙상블의 천장 확인

CNN top-3 raw-mean 앙상블:
- Rank corr: 0.032
- Sharpe: 0.374

→ 단일 CNN보다는 낫지만, **CNN끼리만 섞어서는 `logistic_image_scale` 0.039도 못 넘는다**. CNN 가문 안에선 이미 서로 너무 닮아서 천장이 있다.

여기서 교훈: "**더 좋은 CNN 설계로 logistic 이기기**"가 아니라, "**상관 낮은 family를 멤버로 초대**"하는 쪽이 레버가 크다.

---

## 6. Mixed-family 앙상블 (Phase 1) — 1차 천장 돌파

9개 모델 전체에서 size 2~4 조합을 전수 탐색 (raw 평균 · cross-sectional percentile rank 평균 두 방식).

**Winner v1**: `logistic_image_scale` + `cnn_1d_attention_image_scale` + `cnn_1d_cumulative_scale` (rank 평균)

| 지표 | 값 | 비교 |
|---|---|---|
| OOS rank corr | **0.0422** | 단독 logistic_image (0.039) 최초 돌파 |
| Top-k Sharpe | **0.503** | CNN 단독 best (0.52)에 근접 |
| 두 metric 동시 상위권 | ✓ | 이 시점 유일 |

처음으로 **rank corr 0.039 천장을 넘고, Sharpe도 상위권**. 세 멤버가 각기 다른 축 — logistic의 선형성 + CNN의 이미지 처리 + CNN의 시퀀스 처리 — 을 잡아서 나온 시너지.

---

## 7. 2D CNN 재조정 (Phase 2) + 재탐색 (Phase 3) — 2차 천장 돌파

§3 표에서 2D CNN 두 개가 rank corr 0.007 근처로 거의 바닥이었다. 단순히 "데이터 부족이라 overfit"이라 단정하고 포기하기 전에, 학습곡선부터 찍어봤다:

![2D CNN loss curves](ode_inputs_cnn/figures/10_2d_cnn_loss_curves.png)

- **2D residual**: epoch **18**에서 최적 val loss. 기본 설정은 **8 epoch + patience 2** — 학습 끝나기도 전에 조기 종료되고 있었음
- **1D dilated**: epoch 5에서 최적 → 1D는 8 epoch로 충분
- **val loss가 계속 내려가는 모양** → overfit이 아니라 **undertrain**

즉 원인은 "데이터 부족"도 "2D가 이 문제에 안 맞음"도 아니라 **"capacity는 큰데 학습 시간은 짧다"**. Phase 2로 2D만 재훈련 (30 epoch + patience 5):

| 변형 | params | 설정 | rank corr | Sharpe |
|---|---|---|---|---|
| 원본 `cnn_2d_residual_images` | 60K | 8 ep, wd 1e-4 | 0.007 | 0.10 |
| `cnn_2d_residual_wd` (wd만 강화) | 60K | 30 ep, wd 5e-4 | 0.006 | 0.25 |
| ★ `cnn_2d_residual_small` | **23K** | 30 ep, wd 5e-4, dropout 0.2 | **0.043** | 0.21 |

**핵심**: capacity(1/3 축소) + strong wd + dropout을 **모두** 걸어야 효과. wd만 강화는 오히려 망가짐. 이 재조정판 `cnn_2d_residual_small`이 **단일 CNN 중 rank corr 1위**로 튀어올라 — logistic_image (0.039)와 겨우 같은 티어에 진입.

Phase 3로 재조정 2D를 포함해서 ensemble을 재탐색 → **winner v2**: `logistic_image` + `cnn_1d_cumulative` + `cnn_2d_residual_small` (rank 평균)

| 지표 | v1 | v2 | 변화 |
|---|---|---|---|
| OOS rank corr | 0.042 | **0.061** | +45% |
| Top-k Sharpe | 0.503 | **0.643** | +28% |
| 두 metric **동시 1위** | ✓ | ✓ | 여전히 유일 |

세 멤버가 **3개 다른 family** — logistic (선형) + 1D CNN (no-image 시퀀스) + 2D CNN (이미지). 상관이 최소화되면서 synergy가 최대화된 케이스. 즉 **§5에서 말한 "상관 구조가 레버"** 가설이 한 번 더 실증된 셈.

---

## 8. 그래서 다음엔 LSTM — 왜 같은 함정에 안 빠지는가

지금까지 궤적을 보면, **이 연구의 진짜 레버는 "신모델이 이기는 것"이 아니라 "족(族, family) 다양화"**였다. 그럼 다음 다양화 카드는 뭘까?

힌트 하나:

> **Jiang-style 이미지는 시간을 "공간"으로 바꿔 넣는다.** CNN은 2D 패턴을 학습하지만, **"이 시점 다음에 저 시점이 온다"는 명시적 순서 정보는 흐려진다.**

LSTM(Long Short-Term Memory)은 시퀀스 전용으로 태어난 모델이다. 핵심 특징:

- **명시적 시간 순서 처리**: t 시점 정보를 t+1 시점으로 "전달"하는 게이트 구조
- **메모리 게이트**: 오래된 정보 중 "기억할 것"과 "잊을 것"을 학습으로 고름
- **regime 변화에 민감**: 패턴이 바뀌는 구간을 포착하기 유리

CNN이 "사진 한 장을 보고 판단하는 사람"이라면, LSTM은 "**연속 장면을 이어보며 맥락을 쌓는 사람**"이다.

### 기대하는 것 (단독 성능이 아니라 앙상블 기여)

1. **CNN과 정보 축이 다르다** → 상관이 낮을 것 (CNN-CNN 0.5~0.6 vs CNN-LSTM 예상 < 0.3)
2. **상관이 낮으면 앙상블 폭발력이 커진다** — §6·§7에서 두 번 검증된 패턴. LSTM은 현 3-family에 **4번째 축**으로 합류
3. **발표 스토리가 완성된다**: "이미지(CNN = 공간) + 시퀀스(LSTM = 시간) + 선형(logistic = 평균장)의 **n-way 상보성**"

### 실패할 수도 있는 지점

- LSTM도 결국 주가 시계열 특유의 노이즈엔 약할 수 있다
- 학습 데이터가 적으면 overfitting 위험 (CNN보다 데이터 탐식) — 2D CNN rehab과 같은 hyperparameter 신경 필요
- `logistic_cumulative`(시퀀스+단순모델)가 이미 실패(−0.007)한 점을 고려하면, 시퀀스 입력 자체가 어려운 과제일 수도

**중요**: 목표는 **"LSTM 단독이 CNN을 이기는 것"이 아니다**. 0.02~0.03만 내도 **상관이 낮으면 ensemble에 충분히 기여**. 이 framing을 놓치지 말 것.

---

## 9. Phase 4 — LSTM family 합류 후 검증 결과

LSTM 팀원이 4개 모델을 합류시켰다: 순수 LSTM 2종 (`lstm_*`) + CNN+LSTM hybrid 2종 (`cnnlstm_*`, 팀원 v2 작품).

### 단일 LSTM family 성능 — 예측한 것보다 잘 나왔다

| 모델 | rank corr | Sharpe |
|---|---|---|
| **`lstm_image_scale`** | **0.0506** | 0.185 |
| `cnnlstm_image_scale` (팀원 작품) | 0.0448 | **0.434** |
| `cnnlstm_cumulative_scale` | −0.0125 | 0.361 |
| `lstm_cumulative_scale` | −0.0167 | 0.297 |

- **`lstm_image_scale` 이 단독 1위** (0.0506) — 단일 CNN best (0.043) 보다 점추정 우위
- **`cnnlstm_image_scale` 이 Sharpe 1위** (0.434) — rank 는 4위지만 portfolio quality 가 LSTM family 중 최고. 점예측 ≠ portfolio 분리 케이스
- 개별 격차는 noise 범위 — 0.043 ↔ 0.045 ↔ 0.051 차이는 CI 안

### 14모델 ensemble 재탐색 (1470 조합 × 2 = 2940)

| 우선 metric | Winner | rank corr | Sharpe |
|---|---|---|---|
| rank corr | **logistic_image + cnn_1d_cumulative + cnn_2d_residual_small + cnnlstm_image** (k=4) | **0.0671** | 0.541 |
| (이전 winner, lstm_image 버전) | logistic + 1D + 2D + lstm_image | 0.0653 | 0.452 |
| Sharpe | `logistic_image + cnn_1d_cumulative + cnn_2d_residual_small` (k=3, Phase 3 winner) | 0.0614 | **0.6302** |

**§9 의 두 metric 동시 1위 패턴은 여전히 깨졌다**. Phase 4 합류 후 (cnnlstm 버전):
- rank corr +0.0068 (3-family → 4-family, 이전 lstm 버전 +0.0027 의 2.5배)
- Sharpe −0.099 (이전 lstm 버전 −0.178 대비 trade-off 작아짐)

### "단독 1위가 ensemble winner 에 못 들어간다" 의 정확한 사례

`lstm_image_scale` 가 단독 rank corr 1위 (0.0506) 인데, ensemble winner top-1 에 못 들어가고 단독 4위 `cnnlstm_image_scale` (0.0448) 가 들어간다. 이유는 ρ 매트릭스에 있다:

| | logistic | 1D | 2D | cnnlstm | lstm |
|---|---|---|---|---|---|
| logistic_image | 1.00 | 0.03 | 0.16 | **−0.17** | −0.16 |
| 1D cumulative | 0.03 | 1.00 | 0.09 | 0.32 | 0.18 |
| 2D residual_small | 0.16 | 0.09 | 1.00 | 0.10 | 0.13 |
| **cnnlstm_image** | **−0.17** | 0.32 | 0.10 | 1.00 | 0.63 |
| lstm_image | −0.16 | 0.18 | 0.13 | 0.63 | 1.00 |

- **`cnnlstm_image` vs `logistic`: ρ = −0.17 (음의 상관!)** — 가장 독립적
- `cnnlstm` vs `cnn_2d`: 0.10 (낮음)
- `cnnlstm` vs `lstm_image`: 0.63 (같은 LSTM family, 둘 다 winner 에 못 들어가는 이유)

**§7 thesis 의 정확한 실증**: 신호 강도보다 **"멤버 간 상관"** 이 ensemble 결과를 지배. cnnlstm 이 단독 4위지만 logistic 과 음의 상관 (−0.17) 이라 합류 효과가 lstm_image (logistic 과 ρ −0.16, 비슷하지만 cnnlstm 이 1D/2D 와의 상관까지 종합하면 더 독립적) 보다 큼.

### Bootstrap CI — 정직성 점검

paired bootstrap (B=10000) 으로 fold 별 rank corr 시계열 비교:

| Pair | mean diff | 95% CI | Verdict |
|---|---|---|---|
| 4-family (cnnlstm) − 3-family | **+0.0068** | [−0.0024, +0.0157] | **borderline NOT significant** |
| cnnlstm best − CNN best (단일) | +0.0022 | [−0.0201, +0.0245] | NOT significant |

이전 (lstm 버전) lift +0.0027 / CI [−0.007, +0.012] 대비 점추정 2.5배 + CI lower bound 가 0 에 거의 닿음 (−0.0024). **여전히 strict significance 는 부족하지만 trend evidence 가 강해졌다**.

### 3단계 lift 진행

![3-stage lift](ode_inputs_cnn/figures/11_3stage_lift_progression.png)

Phase 1 (CNN-only) **0.038** → Phase 3 (+ logistic) **0.061** → Phase 4 (+ cnnlstm) **0.067**. 단조 증가, Phase 3 → 4 의 step 이 이전보다 커짐.

![Bootstrap CI](ode_inputs_cnn/figures/12_bootstrap_ci.png)

### 정직한 framing

"family 다양화가 lift 를 만든다" 는 thesis 가 한 번 더 검증되었지만, **여전히 borderline NOT significant**. 다음 세 줄 사이의 차이를 정확히 잡아야 한다:

- ❌ "Phase 4 lift +11% — cnnlstm 이 천장을 뚫었다" (overclaim)
- ❌ "Phase 4 도 noise — LSTM 무의미" (underclaim)
- ✅ "점추정 +0.007 (이전 +0.003 의 2배), CI lower bound 가 0 에 거의 닿음, ρ = −0.17 (음의 상관) 이 mechanism 강하게 지지 — strict statistical proof 까지는 부족하지만 trend evidence 가 일관되게 강해지고 있다"

**production 정책**: trade-off 를 회피하지 말고 두 번들 다 제공.
- `ensemble_best` (Sharpe 우선) — 실무 default
- `ensemble_4family` (rank-corr 우선) — thesis 검증용 + cnnlstm 합류 lift

ODE 팀이 두 입력으로 weight trajectory 차이를 비교하면 "rank corr 의 미세 lift 가 portfolio level 에서 어떻게 펼쳐지는가" 라는 새 ablation 도 가능.

---

## 10. 정리

### ✅ 얻은 것
- 13 모델 (7 CNN + 2 logistic + 2 LSTM + 2 CNN+LSTM hybrid) OOS 성능 맵 — 대부분 같은 티어라는 정직한 관찰
- 2×2 ablation 으로 lift 기여 분해 (이미지 변환이 rank corr 주 기여, CNN/LSTM 은 Sharpe 기여)
- CNN-only ensemble 의 한계 규명 (0.038 — logistic_image 못 넘음)
- Mixed-family ensemble 로 1차 돌파 (Phase 1 → 3, 0.038 → 0.061)
- 2D CNN underperform 원인 규명 — overfit 이 아니라 undertrain + overparam
- 재조정 2D 포함 재탐색으로 Phase 3 천장 0.061 (Sharpe 0.643 동시 1위)
- cnnlstm hybrid 합류 Phase 4 — rank 0.067 (점추정 +0.007), CI lower bound −0.002 (0 에 거의 닿음)
- "단독 1위 ≠ ensemble winner" 의 정확한 사례 발견 (lstm 0.051 단독 1위 vs cnnlstm 0.045 단독 4위 → ensemble winner 는 cnnlstm)
- Bootstrap significance 테스트 인프라 구축 — 다음 sprint 도 재사용 가능
- ODE 스프린트가 바로 쓸 수 있는 두 production 번들: `ensemble_best` (Sharpe-prio) + `ensemble_4family` (rank-prio)

### 🎯 다음
- ODE solver 실제 통합 + realized PnL backtest (두 ensemble 번들로 비교)
- 5번째 family (tree / transformer) 탐색 — 같은 mechanism 검증 가능한지
- γ(t) 시변 위험회피 신호 통합

### 🧠 네 줄 교훈

> **① "새 모델이 베이스라인을 이긴다"는 프레임을 경계하자.** 단일 metric으로 줄세우면 잘못된 교훈이 나오기 쉽다. 이번 경우 CNN·logistic·LSTM 은 **같은 티어에서 서로 다른 강점**을 가졌고, "이겼다/졌다"의 질문은 이 상보 구조를 덮는다. 대신 "**어떤 축의 정보를 잡느냐**"로 프레임을 바꾸면 그림이 보인다.
>
> **② 앙상블의 진짜 레버는 "상관 구조".** 같은 family 끼리는 ρ ≈ 0.5~0.6 으로 닮아서 천장이 있고, family 섞을 때 진짜 lift 가 나온다. Phase 1 → 3 → 4 로 rank corr 단조 증가 (0.038 → 0.061 → 0.067), 그리고 **단독 1위 LSTM 이 ensemble winner 에서 빠지고 단독 4위 cnnlstm 이 들어가는 패턴** — cnnlstm vs logistic ρ = −0.17 (음의 상관!) 때문. 개별 모델 성능보다 ρ 매트릭스를 먼저 보자.
>
> **③ "underperform"의 원인을 단정짓지 말고 학습곡선부터 찍어라.** 2D CNN 을 "데이터 부족"으로 포기할 뻔했지만, 학습곡선이 말해준 건 **단지 일찍 멈췄다**는 것. Capacity 줄이고 학습 시간 늘리니 단일 CNN 1위로 올라왔고, ensemble 천장도 같이 뚫렸다. "모델이 나쁘다"와 "프로토콜이 나쁘다"는 전혀 다른 문제.
>
> **④ Lift 가 보여도 CI 까지 보자.** Phase 4 lift 의 95% CI lower bound 가 0 에 거의 닿지만 살짝 미만 — strict statistical proof 부족. 그러나 점추정 일관 단조 증가 + ρ = −0.17 음의 상관 + Sharpe trade-off 가 모두 mechanism 과 일치. 결론은 "lift 가 있다" 가 아니라 **"mechanism evidence 가 일관되게 쌓인다"** — 둘은 다른 주장.

---

## 부록: 용어 정리

- **OOS (Out-of-Sample)**: 모델이 학습에 쓰지 않은 기간에 대한 예측
- **Walk-forward**: 시간 순서대로 학습→예측→학습→예측을 반복하는 평가 방식
- **Rank correlation (Spearman)**: 예측 랭킹과 실제 랭킹의 일치도 (−1 ~ +1)
- **Top-k Sharpe**: 예측 상위 k개 자산에 투자한 전략의 위험조정 수익률
- **ODE**: 미분방정식. 여기선 포트폴리오 비중의 시간 변화를 수식으로 풀기 위한 도구
- **μ, Σ**: 각각 기대수익과 공분산 — 포트폴리오 이론의 두 핵심 입력
- **Family (계열)**: 학습 방식·입력 표현이 본질적으로 다른 모델 그룹. 예: 선형 (logistic) vs 합성곱 (CNN) vs 순환 (LSTM). 같은 family 안의 모델끼리는 상관이 높아 앙상블 효과가 제한적.

---

*스프린트 저장소의 실제 숫자·figure·코드는 `ode_inputs_cnn/HANDOFF_SUMMARY.md`에서 확인 가능.*
