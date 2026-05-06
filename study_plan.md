# Sprint2 발표 준비 — 공부 리스트

CNN→LSTM 핸드오프 발표 전에 알고 있어야 할 개념들. 우선순위순으로 정리.
각 항목은 **무엇·왜·핵심수식·검색키워드** 형식.

---

## A. 수학·통계 (★ 반드시 — 질문 핵심)

### A-1. Spearman rank correlation
- **무엇**: 두 변수의 **순위(rank)** 간 Pearson 상관. 값이 −1 ~ +1.
- **왜**: 우리는 μ의 절대값이 아니라 "어떤 자산이 더 오를지"의 **랭킹**만 맞으면 됨. 또 금융 데이터는 outlier가 많아서 절대값 기반 Pearson은 noise에 휘둘림.
- **수식**:
  ```
  ρ_s = 1 − (6 Σd_i²) / (n(n²−1)),   d_i = rank(x_i) − rank(y_i)
  ```
- **발표용 한줄**: "랭킹만 맞으면 포트폴리오엔 충분하니, rank corr이 우리 1차 metric이다."
- **검색**: Spearman vs Pearson, rank correlation finance

### A-2. Sharpe ratio + annualization
- **무엇**: 위험조정 수익률. 평균 수익을 변동성으로 나눔.
- **수식**:
  ```
  Sharpe = mean(r) / std(r) × √(periods_per_year / horizon)
  우리 케이스: √(252 / 20) ≈ √12.6 ≈ 3.55
  ```
- **왜 √(252/20)인가**: 수익률이 i.i.d.라 가정하면 분산은 시간에 비례, 표준편차는 √시간에 비례. 연환산 = (단위기간 Sharpe) × √(1년에 몇 단위기간 들어가는지).
- **발표용 한줄**: "rank corr는 점예측 품질, Sharpe는 포트폴리오 구성 품질 — 둘이 다른 답을 줄 수 있다."

### A-3. Cross-sectional z-score / percentile rank
- **무엇**: 한 **날짜 t** 안에서 7개 자산의 점수를 정규화. 시간축은 안 씀.
- **수식**:
  ```
  z(t,i) = (s(t,i) − mean_t) / std_t        (z-score)
  rank(t,i) = percentile_t(s(t,i))           (rank 변환)
  ```
- **왜 leakage가 안 생기나**: 미래 시점 정보가 z(t,i) 계산에 안 들어감. 같은 날짜 안에서만 비교.
- **왜 ensemble에서 rank-mean이 raw-mean보다 천장이 높았나**: 모델별 score scale이 달라도 percentile은 [0,1]로 통일됨 → 한 모델이 ensemble을 압도하지 못함.
- **검색**: cross-sectional normalization, percentile ranking factor model

### A-4. Log return
- **수식**: `r_t = log(P_t / P_{t-1}) = log P_t − log P_{t-1}`
- **왜 log?**:
  1. **Additive**: 누적 = 단순 합. `r_{0→T} = Σ r_t`
  2. **Symmetric**: +50%↔−50%이 비대칭이 아님 (log scale에서)
  3. **Normality 근사**: 작은 수익률에서 simple return ≈ log return
- **발표용**: "논문이 log return 스케일을 요구해서 우리도 그렇게 맞췄다."

### A-5. Mean-Variance Optimization (Markowitz)
- **목적함수**: `max_w  wᵀμ − (γ/2) wᵀΣw`
  - `w`: 포트폴리오 비중 (Σw=1)
  - `μ`: 기대수익 벡터
  - `Σ`: 공분산
  - `γ`: 위험회피 계수
- **1차 조건 (constraint 없는)**: `w* = (1/γ) Σ⁻¹ μ`
- **핵심 직관**: μ의 절대크기가 아니라 **μ의 랭킹과 Σ⁻¹과의 곱**이 weight를 정함. 그래서 rank corr이 의미 있는 metric.
- **검색**: Markowitz portfolio, mean-variance, quadratic utility

### A-6. Covariance condition number & shrinkage
- **무엇**: `cond(Σ) = λ_max / λ_min`. 클수록 `Σ⁻¹` 계산이 불안정.
- **우리 데이터**: median ≈ 2715 (10³~10⁴ 권장 범위 내, 가끔 spike).
- **shrinkage 직관 (Ledoit-Wolf)**: `Σ_shrunk = α·Σ_sample + (1−α)·target` — sample이 noisy하니 정형화된 target과 섞음.
- **왜 ODE solver가 신경 써야 하나**: condition 큰 시점에 `Σ⁻¹μ` 직접 풀면 weight가 폭발. 우리 번들에선 이 시점 markers만 제공하고 shrinkage는 ODE 팀 몫.

### A-7. ★ 앙상블 bias-variance 분해 (★★ 발표 핵심)
- **수식**: n개 모델의 평균 예측 분산 =
  ```
  Var(평균) = (1/n²) Σ_i Σ_j Cov(model_i, model_j)
            = (1/n²) [n·σ² + n(n−1)·ρ·σ²]
            = σ² [1/n + (n−1)/n · ρ]
            ≈ ρ·σ²    (n→∞)
  ```
- **메시지**:
  - ρ=0이면 분산은 1/n로 줄어듦 → ensemble 효과 최대
  - ρ=1이면 분산 그대로 → ensemble 무의미
  - **상관 ρ가 ensemble 천장을 결정한다** ← 이게 §7 "상관이 레버"의 수학적 근거
- **발표용 한줄**: "CNN-CNN ρ≈0.5 vs CNN-logistic ρ≈0.25. 같은 family 안에선 천장 명확하고, family 섞을 때만 분산 감소가 의미 있다."
- **검색**: bias-variance ensemble, diversity in ensembles, Krogh-Vedelsby

---

## B. 머신러닝 모델 (★★ LSTM 합류로 더 중요)

### B-1. LSTM 게이트 수식
입력 `x_t`, 이전 hidden `h_{t-1}`, cell state `c_{t-1}` → 새 `h_t, c_t`:
```
i_t = σ(W_i·[h_{t-1}, x_t] + b_i)        ← input gate (얼마나 받아들일지)
f_t = σ(W_f·[h_{t-1}, x_t] + b_f)        ← forget gate (이전 기억 얼마나 잊을지)
o_t = σ(W_o·[h_{t-1}, x_t] + b_o)        ← output gate (얼마나 출력할지)
c̃_t = tanh(W_c·[h_{t-1}, x_t] + b_c)     ← 새 정보 후보
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t          ← 셀 상태 업데이트
h_t = o_t ⊙ tanh(c_t)                    ← hidden state
```
- **왜 RNN보다 좋은가**: forget gate가 1에 가까우면 grad가 거의 그대로 흐름 → vanishing gradient 완화
- **CNN과의 차이 (발표용)**: CNN은 "사진 한 장 보고 판단", LSTM은 "연속 장면을 이어보며 맥락 누적"
- **검색**: LSTM Hochreiter Schmidhuber, gated RNN, Christopher Olah LSTM blog

### B-2. CNN 기본 + 우리가 쓴 변형들
1. **Convolution**: 커널이 입력 위를 sliding하며 dot product. parameter sharing.
2. **Receptive field**: 출력 한 픽셀이 입력의 몇 픽셀을 보는가. 깊이 쌓을수록 커짐.
3. **Pooling**: max/avg, downsampling
4. **Dilated convolution**: 커널 사이를 띄움 → params 그대로, RF 키움. **`cnn_1d_dilated`** 가 이것.
5. **Residual / ResNet**: `y = F(x) + x` skip connection. **`cnn_2d_residual`**. 깊은 망의 학습 안정화.
6. **SE attention (Squeeze-Excitation)**: 채널별 가중치를 학습 (어떤 feature map이 중요한지). **`cnn_1d_attention`**.
7. **Multi-scale**: 여러 kernel size를 병렬로 적용. **`cnn_1d_multiscale`**.
- **발표용 한줄**: "각 변형은 receptive field·feature 가중·skip connection 같은 inductive bias를 다르게 거는 시도였고, 단독 성능은 비슷했지만 상관 구조엔 차이를 만들었다."

### B-3. CNN+LSTM hybrid (LSTM 팀원 작품)
- **구조**: CNN이 local pattern 추출 → LSTM이 시간순 시퀀스 처리
- **왜?**: 단순 LSTM은 long sequence에서 local 패턴 캐치가 약함. CNN이 먼저 단기 패턴을 압축해주면 LSTM이 더 잘 학습.
- **이번 결과 해석**: image 입력에선 cnn_lstm이 단순 lstm보다 Sharpe 높음 (0.32 vs 0.19), cumulative 입력에선 둘 다 nonpositive rank corr → 입력 표현 효과가 모델 효과보다 큼 (이미지 ablation 결론과 일치)

### B-4. Regularization 3종 세트 (Phase 2 rehab의 핵심)
1. **Dropout**: 학습 중 뉴런 일부를 무작위 끔. ensemble 근사 효과.
2. **Weight decay (L2)**: loss에 `λ·||w||²` 추가. 큰 weight 패널티.
3. **Early stopping (patience)**: val loss가 N epoch 안 줄면 멈춤.
- **발표용 핵심**: "wd만 강화는 망가졌고 (0.006), 셋 다 (capacity 1/3 + dropout 0.2 + wd 5e-4 + patience 5) 걸어야 0.043으로 살아남." → 단일 약 처방이 아니라 **regularization은 carbon set**으로 작동.

### B-5. Overfit vs Undertrain — loss curve로 진단
| 증상 | train loss | val loss | 처방 |
|---|---|---|---|
| **Overfit** | ↓ 계속 | 어느 시점부터 ↑ | regularization 강화, capacity↓ |
| **Undertrain** | ↓ 계속 | ↓ 계속 (멈췄으면 더 내려갈 것) | epoch↑, patience↑ |
| **Underfit** | 처음부터 안 내려감 | 처음부터 안 내려감 | capacity↑, lr 점검 |
- **우리 2D residual 사례**: val loss가 epoch 18에서 minimum인데 8 epoch에서 끊김 → **undertrain**. 데이터 부족 아님.
- **발표용 한줄**: "성능 안 나오면 모델 탓하기 전에 loss curve부터 본다."

### B-6. Walk-forward vs k-fold CV
- **k-fold가 시계열에 잘못된 이유**: 미래로 학습 → 과거 예측의 가능성 (look-ahead leakage)
- **Walk-forward (expanding/rolling)**: 항상 train_max_date < test_min_date 보장
- **우리 설정**: 24 fold (CNN), 48 fold (LSTM은 더 잘게). 매 fold가 1년 OOS.
- **검색**: time-series cross-validation, walk-forward, anchored CV

### B-7. (보너스) Adam optimizer 직관
- SGD에 1차·2차 모멘텀 추가. 각 파라미터별 learning rate 조정.
- **왜 default로 쓰나**: 튜닝 적어도 잘 됨. 우리 lr=1e-3, wd=1e-4가 표준.

---

## C. 도메인 / 금융

### C-1. OHLCV의 의미
- **Open**: 시가, **High**: 고가, **Low**: 저가, **Close**: 종가, **Volume**: 거래량
- **왜 다섯 개를 다 쓰나**: 종가만 쓰면 정보 손실. 캔들 모양(open-high-low-close 관계)이 단기 패턴 단서.

### C-2. Jiang 2016 이미지 변환
- **참조**: Jiang, Kelly, Xiu (2023, *Review of Financial Studies*) "(Re-)Imag(in)ing Price Trends" — 원래 출처. 우리 코드의 "Jiang-style"이 이걸 가리킴.
- **변환 방식**: 60일치 OHLCV를 **흑백 2D 이미지**로. 가로 = 시간, 세로 = 가격레벨, 픽셀 = 캔들 모양·이동평균선·거래량 막대.
- **왜 효과가 있나** (이번 ablation 결과): rank corr lift의 대부분이 이미지 변환에서 나옴. CNN이 추가하는 건 portfolio level Sharpe.
- **발표 시 보여줄 것**: `lstm/sample_images/` 의 PNG 한 장 — "이런 이미지를 입력으로 줍니다"
- **검색**: Jiang Kelly Xiu CNN price chart

### C-3. ODE 기반 동적 포트폴리오 (final consumer)
논문: **An ODE-Based Dynamic Mean-Variance Portfolio Optimisation with Time-Varying Risk Aversion** (`05_paper/`)
- **핵심 ODE**: `dw/dt = f(μ(t), Σ(t), γ(t))` — weight가 시간에 따라 어떻게 변하는지를 미분방정식으로
- **Time-varying γ(t)**: `γ(t) = γ_0 · exp(−λ t)` 같은 스케줄 — 초반엔 보수적, 후반엔 공격적
- **Simplex projection**: `Σwᵢ = 1, wᵢ ≥ 0` constraint. 매 step마다 ODE 해를 simplex 위로 사영.
- **Euler vs RK4 solver**: 우리는 그냥 numerical ODE solver 표준 — Euler는 1차 (간단·빠름), RK4는 4차 (정확·느림)
- **발표 한줄**: "정적 Markowitz를 시간 미분으로 풀면 ODE가 되고, γ를 시간변수로 두면 동적 위험회피가 된다."

### C-4. 7 자산 universe — 한 줄씩
| 자산 | 정체 | 특징 |
|---|---|---|
| `developed_equity` | 선진국 주식 (S&P 500 류) | 고변동, 고기대수익 |
| `emerging_equity` | 신흥국 주식 | 더 고변동, 환율 리스크 |
| `korea_equity` | 한국 주식 (KOSPI) | 별도 분리, EM과 약간 상관 |
| `treasury_7_10y` | 7~10년 국채 | 듀레이션 중간, 안전자산 |
| `short_treasury` | 단기 국채 | 거의 현금 등가 |
| `corp_bond_ig` | 투자등급 회사채 | 국채+크레딧 스프레드 |
| `alternative` | 대안투자 (REITs/원자재 류) | 주식·채권과 상관 낮음 |
- **왜 이 7개**: 자산군 다양성 — 주식 3 + 채권 3 + 대안 1. 분산투자 stress test로 적합.

### C-5. Top-k 백테스트 규약
- 매 horizon(=20영업일)마다 모델 신호 상위 k(=2) 자산을 동가중 매수, horizon 동안 hold, 다음 rebalance 때 갱신.
- **Hit rate**: rebalance 횟수 중 평균 수익률>0 비율.
- **Turnover**: 인접 두 rebalance 사이 자산 교체 비율.
- **왜 top-2**: 7자산 중 상위 2개 = 약 30% 상위. 너무 적지도 많지도 않음.

---

## D. 방법론 / Narrative (★ 질문 받기 좋은 부분)

### D-1. Post-selection bias
- **문제**: 770(+ LSTM 합류 시 1500+) 조합에서 "OOS 1위" 뽑으면, 그 조합의 OOS 성능은 실제 일반화보다 부풀려짐 (multiple comparison 문제).
- **완화 방법**:
  1. **별도 holdout 기간** 두기 (우리 안 함)
  2. **Top-k 평균** vs Top-1 — robustness 체크
  3. **Bootstrapped CI** — fold별 결과로 신뢰구간
- **발표 솔직 framing**: "0.061은 점추정. 실제 deploy 시는 0.04-0.06 정도로 보면 안전."
- **검색**: data snooping, post-selection inference, multiple testing

### D-2. 점예측 vs 포트폴리오 metric 분리
- **rank corr**: 평균적인 예측 정확도 — 모든 날짜·자산 동등 가중
- **Sharpe**: top-k 자산만 선택, 나머지 정확도는 무관
- **왜 둘이 다른 답을 줄 수 있나**: top 영역 정확도가 평균보다 더 좋거나 나쁠 수 있음 (heavy-tail)
- **이번 결과**: cnn_1d_cumulative는 rank 0.027인데 top-k Sharpe 0.52 → "랭킹은 그저 그런데 상위만 잘 맞춤"
- **발표 한줄**: "ODE solver는 결국 weight를 푸는 거니까 두 metric 다 본다."

### D-3. Family diversity 논리
- **No-Free-Lunch theorem**: 모든 데이터에 최강인 모델은 없다 → 다른 inductive bias를 가진 모델은 다른 오차를 만든다.
- **Family 정의**: 학습 메커니즘이 본질적으로 다른 그룹.
  - Linear (logistic) — 평균장 / convex
  - Conv (CNN) — local + spatial invariance
  - Recurrent (LSTM) — sequential / memory
  - Tree (이번엔 안 씀) — non-smooth / interaction
- **왜 family 다양화가 model-within-family 다양화보다 효과적**: family 간 구조적 차이가 ρ를 0에 가깝게 만듦. 같은 family 안에선 hyperparameter만 달라도 결국 비슷.
- **검색**: ensemble diversity measures, structural vs hyperparameter diversity

### D-4. 2×2 Ablation 의 한계
- 우리 ablation: 4 cell (logistic/CNN × image/no-image), 각 cell rank corr 비교
- **암묵 가정**: image 효과와 model 효과가 **independent (additive)** — interaction = 0
- **현실**: interaction이 있을 수 있음 (Image × CNN cell이 +0.028로 logistic+image 0.039보다 낮은 이유 = 음의 interaction 가능성)
- **더 엄밀하게 하려면**: ANOVA-style, 또는 더 많은 cell + factorial design
- **발표 시 솔직**: "additive 가정 하에 분해. interaction은 정확히 측정 안 했다."

---

## E. 빨리 외울 숫자 (cheat sheet)

| 항목 | 값 |
|---|---|
| 자산 수 | 7 |
| Lookback | 60 영업일 |
| Horizon | 20 영업일 |
| Σ rolling window | 60 영업일 |
| Walk-forward fold (CNN) | 24 |
| Walk-forward fold (LSTM) | 48 |
| OOS 평가 일수 | 약 2,880일 |
| 데이터 기간 | 2014-09-24 ~ 2026-04 (μ 유효) |
| Top-k | 2 |
| ensemble_best rank corr | 0.0606 |
| ensemble_best Sharpe | 0.643 |
| ensemble_4family rank corr (★ Phase 4) | **0.0674** |
| ensemble_4family Sharpe | 0.543 |
| 단일 CNN 최고 (cnn_2d_residual_small) | 0.043 |
| 단일 logistic 최고 (logistic_image) | 0.039 |
| 단일 LSTM 최고 (lstm_image_scale) | 0.051 |
| 단일 hybrid 최고 (cnnlstm_image_scale) | 0.045 (Sharpe 0.434) |
| 2D residual params (원본 / small) | 60K / 23K |
| Σ condition number median | 2,715 |
| ★ cnnlstm vs logistic ρ | **−0.17 (음의 상관)** |
| Phase 4 lift bootstrap CI | [−0.002, +0.016] borderline |

---

## F. 발표 전 작업 체크리스트

- [ ] **LSTM ensemble 재탐색** — `02_code/build_extended_ensemble.py` SOURCES 에 4 LSTM 추가, 1500+ 조합 다시 돌리기
- [ ] **상관 매트릭스 갱신** — figure 05 LSTM 4개 포함 버전. CNN-LSTM, logistic-LSTM ρ 확인
- [ ] **3단계 lift 그래프 슬라이드 1장**: "단일 best (0.043) → CNN+logistic mix (0.061) → +LSTM (?)"
- [ ] **샘플 이미지 1장**: `lstm/sample_images/alternative_*.png` — "입력은 이런 60일 차트"
- [ ] **2D loss curve figure (figures/10)** — "왜 Phase 2 rehab을 했는가"
- [ ] **ablation heatmap (figures/09)** — "이미지 효과 vs CNN 효과"

---

## G. 예상 Q&A (각각 30초 답변 준비)

1. **"왜 rank corr 0.06 정도밖에 안 나오나?"** → 일별 OOS 점예측은 noise가 지배. 0.06이면 t-stat 의미있음. 포트폴리오 관점에선 충분.
2. **"ensemble이 정말 lift인가, 아니면 post-selection인가?"** → 둘 다 일부. Phase 4 lift +0.0068 의 95% CI 가 [−0.002, +0.016] 으로 borderline (lower bound 가 0 에 거의 닿음). single metric 으로는 strict significance 부족하지만, mechanism evidence (점추정 단조 증가, ρ = −0.17 음의 상관, Sharpe trade-off) 가 일관.
3. **"왜 logistic이 CNN과 비슷하게 나오나? CNN의 의미가 뭔가?"** → 점예측은 비슷하나 (a) Sharpe는 CNN이 우위 (b) ensemble 멤버로서 CNN의 상관이 logistic과 다름 = 다른 정보 축.
4. **"2D CNN이 처음에 망친 이유?"** → undertrain. 8 epoch + patience 2로 끊김. 학습곡선 확인 후 30 epoch + patience 5로 재훈련 + capacity 축소 + dropout = 4.3배 향상.
5. **"LSTM이 CNN을 대체하나?"** → 아니. 단독 성능은 같은 티어. **4번째 family 로서 ensemble 다양성 기여**가 목표. Phase 4 결과: 단독 1위 `lstm_image` (0.051) 가 아니라 단독 4위 `cnnlstm_image` (0.045) 가 ensemble winner 에 들어감 — cnnlstm 이 logistic 과 ρ = −0.17 (음의 상관) 으로 더 독립적이기 때문. **"단독 성능 ≠ ensemble 기여"** 의 정확한 사례.
6. **"왜 walk-forward인가? k-fold 안 되나?"** → 시계열에 k-fold 쓰면 미래로 학습해서 과거 예측 = look-ahead leakage. walk-forward는 항상 train_date < test_date 보장.
7. **"Σ가 같은 값 (모델 무관)인 이유?"** → Σ는 실제 수익률에서 rolling으로 계산, 모델과 무관. 모델별로 달라지는 건 μ와 risk_score.
8. **"horizon 20일은 어떻게 정했나?"** → 논문 규약 + 월 1회 rebalance가 실무적으로 합리적. sensitivity는 안 했음 (한계).
9. **"ODE solver는 누가 푸나?"** → 다음 sprint. 우리는 입력 (μ, Σ, R, risk)을 daily 정렬해서 넘겨주는 데까지.

---

## H. 한 줄 슬로건 (발표 전체 메시지)

> **"단일 모델 줄세우기로는 안 보이는 것 — 상관 구조를 보면 lift가 보인다."**
>
> CNN 7종 + logistic 2종 + LSTM 4종을 같은 그리드에서 평가했고, 단독 최강은 모델별로 비슷한 티어에 머물렀지만, **family 간 상관이 낮은 조합을 섞을 때마다 천장이 올라갔다** (0.039 → 0.042 → 0.061 → ?). LSTM은 4번째 family로서 이 패턴이 한 번 더 작동하는지를 검증한다.
