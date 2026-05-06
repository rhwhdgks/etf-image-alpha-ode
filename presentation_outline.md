# Sprint 2 발표 PPT — 슬라이드별 내용 정리

> 이 문서를 보면서 [FIND_A_템플릿.pptx](FIND_A_템플릿.pptx) 에 직접 슬라이드를 채우면 됩니다.
> 각 슬라이드: **제목 / 본문 bullets / figure / 발표 멘트 (≈60초) / 예상 Q&A** 순.
> figure 경로는 [ode_inputs_cnn/figures/](ode_inputs_cnn/figures/) 기준.

전체 13 슬라이드 (cover 1 + 내용 12), 발표 시간 ≈ 15-18분.

---

## Slide 0 — Cover (표지)

**제목**: Sprint 2 — ETF μ Signal Pipeline
**부제**: CNN + LSTM ensemble · ODE 동적 포트폴리오 입력 · 발표자/날짜

> 멘트 (15초): "ODE 기반 동적 포트폴리오의 입력 시그널을 CNN과 LSTM 으로 만든 sprint2 결과 발표입니다. 핵심 thesis 는 '단일 모델 줄세우기보다 family 다양화가 진짜 레버다' 입니다."

---

## Slide 1 — ODE 기반 동적 포트폴리오, 무엇을 푸는가

**섹션 헤더**: 1. 무엇을 푸는가
**제목**: ODE 기반 동적 포트폴리오 — μ 시그널 생성

**본문 (4 bullets)**:
- 7개 ETF 자산에 매일 얼마씩 넣을지를 미분방정식으로 결정
- `dw/dt = f(μ(t), Σ(t), γ(t))` — 우리 역할은 **μ(t) 시그널 생성**
- Σ·R 은 과거 데이터로 자동 계산, μ 는 미래 예측 → **어려운 부분**
- 랭킹 (어떤 자산이 더 오를지) 만 잘 맞아도 포트폴리오엔 충분

**figure**: 없음 (수식 텍스트로 충분)

> 멘트 (60초): "다음 sprint 가 ODE 미분방정식으로 weight trajectory 를 풀 거고, 거기에 들어갈 입력 셋이 μ, Σ, R 입니다. Σ·R 은 과거 수익률에서 기계적으로 나오는데 μ 는 미래 기대수익이라 예측이 필요해요. 다행히 절대값까지 맞출 필요는 없고 자산간 랭킹만 잘 맞아도 Markowitz 가 작동합니다. 그래서 1차 metric 으로 Spearman rank correlation 을 봤습니다."

**Q&A 대비**: ODE solver 가 정확히 뭔지 → "Markowitz 에 시간 미분 추가, simplex projection 으로 weight 합 = 1 보장"

---

## Slide 2 — 입력 표현: Jiang-style 이미지

**섹션 헤더**: 2. 입력 — Jiang-style 이미지 변환
**제목**: 60일 OHLCV → 2D 흑백 이미지 → CNN

**본문 (3 bullets)**:
- "사람 트레이더가 차트 보고 판단하듯 모델도 이미지로 본다" (Jiang-Kelly-Xiu, RFS 2023)
- 캔들 모양 + 이동평균선 + 거래량 막대 → 픽셀 인코딩
- 왼쪽: alternative 자산 60일 입력 이미지 샘플

**figure**: [lstm/sample_images/alternative_20120702.png](lstm/sample_images/alternative_20120702.png) (왼쪽 절반, 큼직하게)

**레이아웃**: 왼쪽 50% 이미지, 오른쪽 50% bullets

> 멘트 (60초): "보통 주가 예측은 숫자 시퀀스를 그대로 넣는데, 2023년 RFS 논문에서 차트 이미지로 변환해서 CNN 에 넣는 방식이 효과적이라는 게 보고됐습니다. 60일 OHLCV 를 흑백 이미지로 만드는 거예요. 가로축은 시간, 세로축은 가격, 캔들 모양과 거래량 막대가 픽셀로 들어갑니다. 우리도 이 방식을 그대로 채택했고, **결과적으로 이 이미지 변환 자체가 큰 lift 를 줬다**는 게 뒤에 ablation 에서 나옵니다."

---

## Slide 3 — 실험 설계

**섹션 헤더**: 3. 실험 설계
**제목**: 13 모델 walk-forward OOS — 24~48 folds × 2,880일

**본문 (4 bullets)**:
- **8 CNN** (1D 4종 + 2D 3종 + Phase 2 rehab 1종) + **2 logistic** + **2 LSTM** + **2 CNN+LSTM hybrid**
- Walk-forward expanding: `train_max_date < test_min_date` 항상 보장 → **leakage 차단**
- Lookback 60일, horizon 20일 (월 1회 rebalance 가정)
- 평가 metric: **Spearman rank corr** (점예측) + **top-2 Sharpe** (포트폴리오)

**figure**: 없음 (실험 셋업 표 형태)

**보조 표 (수동으로 그려도 됨)**:
| Family | 모델 수 | 비고 |
|---|---|---|
| Logistic baseline | 2 | image / cumulative |
| CNN 1D | 4 | image / dilated / attention / multiscale |
| CNN 2D | 3 | rendered / residual / **Phase 2 rehab** |
| LSTM | 2 | image / cumulative |
| CNN+LSTM hybrid | 2 | image / cumulative (팀원 작품) |

> 멘트 (45초): "총 13개 모델을 같은 grid에 올렸습니다. CNN 8종, logistic 2종, LSTM 4종 (순수 LSTM 2 + CNN+LSTM 하이브리드 2). 모두 walk-forward — 매 fold 의 학습 종료일이 테스트 시작일보다 항상 빠르게 잡았어요. Lookback 60일에 horizon 20일이라 월 1회 rebalance 시나리오. 두 metric 을 다 봤는데 rank corr 는 점예측 정확도, top-2 Sharpe 는 실제 포트폴리오 품질입니다."

**Q&A 대비**: walk-forward vs k-fold → "k-fold 는 미래로 학습해서 과거 예측 가능 — leakage. walk-forward 는 시간 단방향만"

---

## Slide 4 — 결과: 단독은 모두 같은 티어

**섹션 헤더**: 4. 결과 — 단독은 모두 같은 티어
**제목**: rank corr 1위 ensemble_4family · Sharpe 1위 ensemble_best

**본문 (3 bullets)**:
- 단독 model 격차 (CNN 0.043 ↔ logistic 0.039 ↔ LSTM 0.051) 는 **noise 범위**
- ★★ `ensemble_4family` rank corr **0.0674** / Sharpe 0.543
- ★ `ensemble_best` rank corr 0.0606 / Sharpe **0.643**

**figure**: [figures/01_model_comparison.png](ode_inputs_cnn/figures/01_model_comparison.png) (큰 figure, 슬라이드 60% 차지)

**레이아웃**: 위쪽 bullets (3줄), 아래쪽 figure 큰 가로 바형

> 멘트 (75초): "왼쪽이 rank correlation, 오른쪽이 top-k Sharpe 막대 그래프입니다. 색깔이 family 입니다. 핵심 관찰 세 개. 첫째, 단일 모델끼리 줄세우면 격차가 noise 범위예요 — 단독 1위가 LSTM image 인데 0.051 이고, CNN best 가 0.043, logistic best 가 0.039. 점추정 차이는 작아요. 둘째, **rank corr 1위와 Sharpe 1위가 다른 ensemble** 입니다. ensemble_4family 가 점예측 1위, ensemble_best 가 portfolio 1위. 셋째, 두 metric 으로 줄세우면 답이 달라진다는 게 'CNN이 logistic 을 이긴다' 식의 단순 framing 의 한계를 보여줍니다."

**Q&A 대비**: rank corr 0.06 이 의미 있는 수준이냐 → "absolute 로는 작은 신호. 단 portfolio level 에선 충분, 그리고 ensemble 자체가 noise 줄임"

---

## Slide 5 — Ablation: 이미지 효과 vs 모델 효과

**섹션 헤더**: 5. Ablation — 이미지 효과 vs 모델 효과
**제목**: rank corr lift 의 대부분은 **이미지 변환**에서

**본문 (3 bullets)**:
- Logistic + 이미지: −0.007 → +0.039 = **+0.046 lift** ← Jiang의 공
- CNN + 이미지: +0.027 → +0.028 = **≈0 lift** ← 이미지 위에 CNN 얹는 효과 미미
- → rank corr 만 보면 CNN 이 logistic 을 이긴다고 단정 못함. **Sharpe 는 별개** (CNN 우위)

**figure**: [figures/09_ablation_image_vs_cnn.png](ode_inputs_cnn/figures/09_ablation_image_vs_cnn.png) (오른쪽 절반)

**레이아웃**: 왼쪽 bullets, 오른쪽 ablation heatmap

> 멘트 (60초): "2×2 ablation 입니다. 입력 (이미지/no image) × 모델 (logistic/CNN). 정직하게 분해하면 rank corr lift 의 대부분이 이미지 변환에서 나옵니다 — logistic 에 이미지 입력 주면 0.046 올라가요. 반면 같은 이미지 입력에 CNN 을 얹어도 추가 lift 는 거의 없습니다. 이게 'CNN 이 베이스라인을 이긴다' framing 을 의심하게 만든 출발점이에요. 단 portfolio Sharpe 는 CNN 이 우위 — 점예측 정확도와 포트폴리오 구성 품질이 분리되는 케이스입니다."

**Q&A 대비**: 그럼 CNN 의 의미가 뭐냐 → "단독 우위는 모호하지만 **ensemble 다양성 기여** 가 다음 슬라이드의 thesis"

---

## Slide 6 — ★ 진짜 레버: 상관 구조

**섹션 헤더**: 6. ★ 진짜 레버 — 상관 구조
**제목**: ρ 가 ensemble 천장을 결정한다

**본문 (4 bullets)**:
- CNN-CNN ρ ≈ 0.5~0.6 (같은 family, 닮음) → CNN-only 앙상블 천장 명확
- LSTM-CNN ρ ≈ 0.03~0.18 ← **가장 독립적인 family**
- ★ **`cnnlstm_image` vs `logistic_image`: ρ = −0.17 (음의 상관!)** ← 분산 감소 효과 최대
- 수식: `Var(평균) ≈ ρ·σ²` (n→∞) — 신호 강도보다 ρ 가 결과를 지배

**figure**: [figures/05_model_raw_correlation.png](ode_inputs_cnn/figures/05_model_raw_correlation.png) (오른쪽)

**레이아웃**: 왼쪽 bullets + 작은 ρ 매트릭스 표, 오른쪽 heatmap

**보조 표 (Phase 4 멤버 5×5)**:
|  | logistic | 1D | 2D | cnnlstm | lstm |
|---|---|---|---|---|---|
| logistic | 1.00 | 0.03 | 0.16 | **−0.17** | −0.16 |
| 1D | 0.03 | 1.00 | 0.09 | 0.32 | 0.18 |
| 2D | 0.16 | 0.09 | 1.00 | 0.10 | 0.13 |
| cnnlstm | **−0.17** | 0.32 | 0.10 | 1.00 | 0.63 |
| lstm | −0.16 | 0.18 | 0.13 | 0.63 | 1.00 |

> 멘트 (90초, 핵심 슬라이드): "이게 발표의 thesis 핵심입니다. 앙상블 이론에서 n개 모델을 평균한 분산은 σ² 곱하기 \[1/n + (n-1)/n × ρ\] 인데, n 이 커지면 ρ × σ² 로 수렴합니다. 즉 **신호 자체의 강도보다 신호 간 상관 ρ 가 ensemble 천장을 결정**해요. 매트릭스를 보면 CNN 끼리는 ρ 0.5~0.6 으로 닮았고, LSTM 과 CNN 은 0.03~0.18 로 가장 독립적입니다. 그리고 가장 흥미로운 게 **cnnlstm_image 와 logistic_image 가 ρ = −0.17, 음의 상관** 이에요. 이게 다음 슬라이드에서 단독 4위 모델이 ensemble winner 에 들어가는 이유가 됩니다."

**Q&A 대비**: 왜 음의 상관이 좋은가 → "분산 감소 효과는 ρ 가 작을수록 큼. 0 보다 작으면 더 좋음 — anti-correlated signals are 좋음"

---

## Slide 7 — Phase 2: 2D CNN underperform 진단

**섹션 헤더**: 7. Phase 2 — 2D CNN underperform 진단
**제목**: 원인은 데이터 부족 아니라 **undertrain + overparam**

**본문 (3 bullets)**:
- 원본 cnn_2d_residual: 8 epoch + patience 2 → val loss 가 **epoch 18 에 minimum** 인데 일찍 끊김
- Rehab: capacity 60K→23K + dropout 0.2 + wd 5e-4 + patience 5 → rank corr **0.007 → 0.043 (4.3배)**
- 교훈: "underperform" 의 원인을 단정짓지 말고 **학습곡선부터 찍어라**

**figure**: [figures/10_2d_cnn_loss_curves.png](ode_inputs_cnn/figures/10_2d_cnn_loss_curves.png) (왼쪽 큰 영역)

**레이아웃**: 왼쪽 60% figure, 오른쪽 40% bullets

> 멘트 (75초): "처음에 2D CNN 이 rank corr 0.007 로 거의 바닥이었어요. 그냥 '데이터 부족이라 overfit 한다' 결론 내고 포기할 수 있었지만, **학습곡선부터 찍어봤습니다**. 보면 train loss 와 val loss 가 둘 다 epoch 18까지 계속 내려가는데 우리가 8 epoch + patience 2 로 일찍 끊고 있었어요. 즉 overfit 이 아니라 **undertrain** 이었습니다. 그래서 capacity 를 1/3 로 줄이고 (60K→23K params), dropout 0.2, weight decay 5e-4, patience 5 로 늘려서 재학습했더니 rank corr 0.043 으로 4.3배 점프 — CNN family 1위가 됐습니다. 교훈: '모델이 나쁘다' 와 '프로토콜이 나쁘다' 는 전혀 다른 문제입니다."

**Q&A 대비**: 왜 capacity 를 줄였는가 → "wd 만 강화는 효과 없음 (0.006). capacity + dropout + wd + patience 를 **모두** 걸어야 효과. 단일 약 처방이 아님"

---

## Slide 8 — Phase 4: cnnlstm hybrid 합류

**섹션 헤더**: 8. Phase 4 — cnnlstm hybrid 합류
**제목**: ★ 단독 1위 LSTM 이 ensemble winner 에서 빠지는 패턴

**본문 (4 bullets)**:
- 단독 LSTM 1위: `lstm_image_scale` (rank **0.0506**)
- 단독 hybrid 1위: `cnnlstm_image_scale` (rank 0.0448, **Sharpe 0.434 — LSTM family 1위**)
- ensemble_4family winner = `logistic + 1D + 2D + cnnlstm_image` (NOT lstm_image)
- 이유: cnnlstm vs logistic ρ = **−0.17 (음의 상관)**, lstm vs cnnlstm ρ = 0.63 (둘 다 못 들어감)

**figure**: 없음 (텍스트 + 작은 winner 비교 표)

**보조 표 (winner 비교)**:
| Winner | rank corr | Sharpe |
|---|---|---|
| ★★ logistic + 1D + 2D + **cnnlstm_image** (k=4) | **0.0671** | 0.541 |
| logistic + 1D + 2D + **lstm_image** (이전) | 0.0653 | 0.452 |
| ★ logistic + 1D + 2D (k=3, Phase 3 winner) | 0.0614 | **0.6302** |

> 멘트 (75초): "여기가 Phase 4 의 핵심 발견입니다. 직관적으로는 단독 1위인 lstm_image_scale (0.051) 이 ensemble winner 에 들어가야 할 것 같죠. 그런데 실제로는 **단독 4위 cnnlstm_image (0.045) 가 winner 에 들어갑니다**. 왜? 슬라이드 6 의 ρ 매트릭스 때문이에요. cnnlstm 이 logistic 과 ρ = −0.17 로 음의 상관이라 ensemble 합류 시 분산 감소 효과가 더 큽니다. 또 cnnlstm 과 lstm 은 같은 LSTM family 라 ρ = 0.63 으로 서로 닮았는데, 이건 **둘 다 ensemble 에 못 들어가는 이유** 입니다 — 둘 중 하나만 골라야 하는데 logistic 과 더 독립적인 cnnlstm 이 선택돼요. 이게 '단독 성능 ≠ ensemble 기여' 의 정확한 사례입니다."

**Q&A 대비**: 그럼 lstm_image 는 쓸모없는가 → "단독 모델로는 LSTM family 의 capability 증명. ensemble 에선 자리가 cnnlstm 에 양보된 것 — family 다양화 정책의 결과"

---

## Slide 9 — 3단계 lift progression

**섹션 헤더**: 9. 3단계 lift progression
**제목**: rank corr 단조 증가: 0.038 → 0.061 → 0.067

**본문 (3 bullets)**:
- Phase 1 (CNN-only top-3): rank **0.0375** / Sharpe 0.275
- Phase 3 (+ logistic, 3-family): **0.0606** / Sharpe **0.643** ← 두 metric 동시 1위
- Phase 4 (+ cnnlstm, 4-family): **0.0674** / Sharpe 0.543 ← rank 천장 갱신, Sharpe trade-off

**figure**: [figures/11_3stage_lift_progression.png](ode_inputs_cnn/figures/11_3stage_lift_progression.png) (큰 figure)

**레이아웃**: 위 bullets 3줄, 아래 figure 큰 가로형

> 멘트 (60초): "단계별 lift 진행입니다. Phase 1 은 CNN 만으로 ensemble 했더니 rank 0.038 — logistic 단독 (0.039) 도 못 넘었어요. Phase 3 에서 logistic 을 mixed family 로 합류시키니까 0.061 로 천장 돌파, Sharpe 도 0.643 으로 두 metric 동시 1위. Phase 4 에서 cnnlstm hybrid 까지 합류시키니까 rank 가 0.067 로 더 올라갔지만 Sharpe 는 0.543 으로 trade-off 가 발생했어요. 점추정으로는 단조 증가지만 마지막 step 의 statistical significance 는 다음 슬라이드에서 짚겠습니다."

---

## Slide 10 — Significance: Bootstrap CI

**섹션 헤더**: 10. Significance of Improvements
**제목**: Ensemble lift 는 baseline 대비 statistically significant ✅

**본문 (4 bullets, 표로)**:

| 비교 | mean diff | 95% CI | Verdict |
|---|---|---|---|
| `logistic_image` → `ensemble_best` | **+0.021** | **[+0.008, +0.035]** | ✅ **significant** |
| `logistic_image` → `ensemble_4family` | **+0.028** | **[+0.013, +0.043]** | ✅ **significant** |
| `logistic_cumulative` → `ensemble_4family` | **+0.075** | **[+0.052, +0.096]** | ✅ strongly significant |
| `ensemble_best` → `ensemble_4family` | +0.007 | [−0.002, +0.016] | borderline |

**figure**: [figures/12_bootstrap_ci.png](ode_inputs_cnn/figures/12_bootstrap_ci.png) (5 panel grid)

> 멘트 (90초): "10000회 paired bootstrap 으로 5가지 비교를 했습니다. **첫째 핵심 — Ensemble vs realistic baseline (logistic_image)**: 두 ensemble 다 lift CI 가 0 을 명확히 배제 — **statistically significant**. ensemble_best 는 +0.021 (CI [+0.008, +0.035]), ensemble_4family 는 +0.028 (CI [+0.013, +0.043]). 즉 'ensemble 이 baseline 대비 정말 좋은가' 에 대한 답은 **YES, statistically proven**. **둘째 — no-image baseline (logistic_cumulative) 와 비교**하면 lift 는 +0.075 로 더 크고 CI 도 한참 0 위쪽 — 이미지 변환 + ensemble 의 누적 효과를 보여줍니다. **셋째 — ensemble 끼리의 미세 차이 (Phase 3 vs 4)** 만 borderline (lift +0.007, CI lower bound 가 0 살짝 미만). 즉 **'ensemble 이 baseline 보다 낫다' 는 statistical proof 통과**, **'어느 ensemble 이 더 낫나' 는 ODE 백테스트로 답**."

**Q&A 대비**:
- "결국 noise 인가?" → "ensemble vs baseline 은 noise 아님 (CI 0 배제). ensemble 끼리 미세 차이만 borderline."
- "어느 ensemble 을 default 로?" → "Sharpe 우선이면 ensemble_best, rank corr 우선이면 ensemble_4family. ODE 백테스트에서 어느 게 더 좋은지 검증 예정."

---

## Slide 11 — 4줄 교훈

**섹션 헤더**: 11. 정리
**제목**: 4줄 교훈

**본문 (4 bullets, 굵게)**:
- **① "단일 모델 줄세우기" 프레임 경계** — 같은 티어에서 서로 다른 강점을 본다
- **② 앙상블의 진짜 레버는 상관 구조** — Phase 1→3→4 단조 증가의 mechanism. 단독 1위 ≠ ensemble winner 사례 (cnnlstm)
- **③ Underperform 원인을 단정짓지 말고 학습곡선부터 찍어라** — 2D CNN rehab
- **④ Lift 정직성 — Ensemble vs baseline 은 통계적 lift 검증, ensemble 내부 미세 차이는 ODE 백테스트로 답**

**figure**: 없음 (텍스트만, 큼직하게)

> 멘트 (60초): "발표 핵심 메시지 4줄입니다. 첫째, 단일 metric 줄세우기는 잘못된 교훈을 만들기 쉽다 — 같은 티어에서 서로 다른 강점을 가지는 모델은 줄세우면 노이즈에 휘둘립니다. 둘째, ensemble 의 진짜 레버는 신호 강도가 아니라 상관 구조 — cnnlstm 이 단독 4위인데 ensemble winner 에 들어가는 게 결정적 사례. 셋째, 모델이 못한다고 단정하기 전에 학습곡선부터 본다 — 2D CNN 이 'undertrain' 이었지 'capacity 부족' 이 아니었어요. 넷째, lift 가 보여도 CI 까지 본다 — 우리 Phase 4 lift 가 borderline 인 만큼 'lift 있다' 와 'mechanism evidence 일관' 은 다른 주장입니다."

---

## Slide 12 — 다음: ODE 통합 + 확장

**섹션 헤더**: 12. 다음 — ODE 통합 + 확장
**제목**: 두 production 번들 + 후속 실험

**본문 (3 bullets)**:
- **두 번들 제공**: `ensemble_best` (Sharpe-prio default, 0.643) + `ensemble_4family` (rank-prio, +cnnlstm)
- **ODE 팀 다음 작업**: 두 입력으로 weight trajectory 비교 → "점예측 미세 lift 가 portfolio level 어떻게 펼쳐지나" 새 ablation
- **확장 후보**: 5번째 family (tree / transformer), γ(t) 시변 위험회피, Σ shrinkage

**figure**: 없음

> 멘트 (45초): "다음 sprint 에 넘기는 게 두 production 번들입니다. ensemble_best 가 Sharpe 우선 default (0.643), ensemble_4family 가 rank corr 우선 (cnnlstm 포함). ODE 팀이 두 입력으로 weight trajectory 를 만들어서 비교하면 '점예측 미세 lift 가 portfolio level 에서 어떻게 펼쳐지는가' 라는 새 ablation 이 가능합니다. 확장 후보로는 5번째 family — tree 모델이나 transformer — 추가, γ(t) 시변 위험회피 통합, Σ shrinkage 정도가 있습니다. 이상입니다."

---

## 발표 시 주의사항

### 시간 분배 (총 ≈18분)
| 슬라이드 | 시간 | 누적 |
|---|---|---|
| 0 (cover) | 0:15 | 0:15 |
| 1-3 (배경/설계) | 3:00 | 3:15 |
| 4-5 (단일 결과 + ablation) | 2:15 | 5:30 |
| 6 (★ 상관 구조 — 핵심) | 1:30 | 7:00 |
| 7 (Phase 2 rehab) | 1:15 | 8:15 |
| 8-9 (Phase 4 + lift) | 2:15 | 10:30 |
| 10 (★ Bootstrap CI — 정직성) | 1:15 | 11:45 |
| 11-12 (교훈 + 다음) | 1:45 | 13:30 |
| 여유/Q&A 도입 | 1:30 | 15:00 |
| **Q&A** | 5-10분 | |

### 강조 포인트 (★ 별 표시)
- Slide 6 (상관 구조): thesis 의 핵심. 여기서 청중이 "아 그래서 ensemble 이 그렇게 작동하는구나" 가 들어와야 함
- Slide 10 (Bootstrap): 정직성으로 신뢰 확보. "lift 있다" 단정 안 하는 게 오히려 강함
- Slide 8 (단독 1위 ≠ winner): mechanism 의 가장 직관적인 사례

### Q&A 대비 (자세한 답변은 [study_plan.md](study_plan.md) §G 참조)
1. rank corr 0.06 이 의미 있나? → 절대값 작지만 portfolio 관점은 충분
2. ensemble lift 가 post-selection bias 아닌가? → 일부 그렇다, CI 가 borderline 인 게 그 증거
3. 왜 logistic 이 CNN 과 비슷한가? → 점예측 비슷, Sharpe 다름 + ensemble 멤버로 다른 정보 축
4. 2D CNN 이 처음 망친 이유? → undertrain + overparam (overfit 아님)
5. LSTM 이 CNN 대체? → 아니. 4번째 family 로서 ensemble 다양성 기여 + 단독 ≠ ensemble 기여
6. walk-forward vs k-fold? → leakage 차단
7. Σ 가 모델 무관한 이유? → 실제 수익률에서 rolling 으로만 계산
8. horizon 20 일은 어떻게 정했나? → 논문 규약 + 월 1회 rebalance
9. ODE solver 는 누가? → 다음 sprint

### 슬라이드 디자인 팁
- 본문 폰트 사이즈: bullets 14-15pt, 제목 22-24pt, 섹션 헤더 16pt
- figure 는 슬라이드의 50-65% 차지하게 — 작게 넣지 말 것
- 색상은 템플릿 그대로 유지 (Pretendard, dark blue/red 계열)
- 슬라이드별 본문 줄 수 4 이하 — 더 넣지 말 것
- 숫자는 굵게: **0.0674**, **+0.0068** 등

### 만약 시간 부족 (15분 발표)
줄여도 되는 슬라이드:
- Slide 7 (Phase 2 rehab) — 1줄로 압축, "내부 디테일이라 패스" 라 말하고 넘기기
- Slide 12 (다음) — 30초로 압축

압축 못 하는 슬라이드: **6, 8, 10** (thesis 의 3대 기둥)
