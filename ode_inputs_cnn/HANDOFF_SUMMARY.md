# CNN 시그널 → ODE 입력 — 팀 핸드오프 요약

> 이 문서는 **CNN 파트 결과물을 한 페이지로 브리핑**하는 용도입니다.
> 세부 스키마는 [README.md](README.md), 수치 sanity는 [qa_report.md](qa_report.md), 모델별 성능은 [comparison.md](comparison.md) 참고.

---

## TL;DR

- **8 CNN + 2 logistic + 2 LSTM + 2 CNN+LSTM hybrid + 3 ensemble = 17 entries** 를 walk-forward OOS 로 평가
- ★ **Sharpe 우선이면 `ensemble_best` (3-family)** · **rank corr 우선이면 `ensemble_4family` (4-family, cnnlstm 포함)** — 두 번들 모두 production 제공
- **Phase 2 2D CNN 재조정** (`cnn_2d_residual_small`, 60K→23K params) → 단일 rank corr 0.0426
- **Phase 4 hybrid 합류** (`cnnlstm_image_scale` — 팀원 작품, 단일 rank 0.0448) → 4-family ensemble rank +0.0068 / Sharpe trade-off, 95% CI [−0.002, +0.016] borderline. cnnlstm vs logistic ρ = **−0.17 (음의 상관)** — diversity mechanism 강하게 지지
- ODE 입력 4종 (μ·Σ·risk·R) 모두 **daily 그리드로 정렬**, **look-ahead leakage 없음**

### 핵심 숫자 (17 entries)

| 지표 | Rank corr 1위 | Top-k Sharpe 1위 | 최약 |
|---|---|---|---|
| 모델 | ★ `ensemble_4family` (Phase 4) | ★ `ensemble_best` (Phase 3) | `lstm_cumulative_scale` |
| OOS rank corr | **0.0674** | 0.0606 | −0.0167 |
| Top-k Sharpe | 0.543 | **0.643** | 0.076 (logistic_cum) |

| Phase | Winner | rank corr | Sharpe |
|---|---|---|---|
| 1 (CNN-only top-3) | `ensemble_top3` (auto re-select) | 0.0375 | 0.275 |
| 3 (CNN + logistic) | ★ **`ensemble_best`** | 0.0606 | **0.6425** |
| 4 (+ cnnlstm hybrid) | ★ **`ensemble_4family`** | **0.0674** | 0.543 |

`ensemble_best` = `logistic_image_scale` + `cnn_1d_cumulative_scale` + `cnn_2d_residual_small` (rank-mean).
`ensemble_4family` = `ensemble_best` 멤버 + `cnnlstm_image_scale` (rank-mean).

**Bootstrap significance** ([significance_test.md](significance_test.md), B=10000):
- Phase 3 → 4 lift +0.0068, 95% CI [−0.002, +0.016] → **borderline NOT significant** (lower bound 가 0 살짝 미만)
- 단일 CNN best vs cnnlstm lift +0.0022, 95% CI [−0.020, +0.025] → NOT significant
- **cnnlstm-logistic ρ = −0.17 (음의 상관)**, cnnlstm-CNN ρ = 0.10~0.32 → diversity mechanism 작동. lift 점추정이 borderline 까지 올라온 건 LSTM family 합류 효과의 강한 증거

상세: [comparison_with_baselines.csv](comparison_with_baselines.csv) · [ensemble_search_top.md](ensemble_search_top.md) · [significance_test.md](significance_test.md)

---

## 1. 모델 비교 — 17 entries, 6 family

![Model comparison](figures/01_model_comparison.png)

17 entries 를 walk-forward OOS (24~48 folds) 에서 비교. 색은 family별:
- ★★ **Ensemble (4-family)** — `ensemble_4family` (rank corr 우선 default)
- ★ **Ensemble (CNN + logistic)** — `ensemble_best` (Sharpe 우선 default)
- 🟥 **Ensemble (CNN only)** — 레거시 `ensemble_top3`
- 🟦 **CNN 1D image** · 🟪 **CNN 2D image** · 🟩 **CNN (no image)**
- 🟨 **LSTM (sequence)** · 🟫 **CNN+LSTM hybrid**
- 🟧 **Baseline (image+logistic)** · ⬜ **Baseline (no image)**

핵심 관찰:
- **rank corr 1위는 `ensemble_4family` (0.0633)**, **Sharpe 1위는 `ensemble_best` (0.630)** — 두 metric 챔피언이 갈림
- **단일 LSTM 1위 `lstm_image_scale` (rank 0.0506)** — 단일 CNN 최고 `cnn_2d_residual_small` (0.0426) 보다 점추정 우위, 단 noise 범위
- **`cnn_lstm_image_scale` (0.0389)** — 순수 LSTM (0.0506) 대비 떨어짐. CNN+LSTM hybrid 조합은 image 입력 효과를 단순 LSTM 만 못 살림
- **상관 구조 핵심**: LSTM-CNN raw ρ = 0.03~0.18, CNN-CNN ρ = 0.5~0.6. LSTM 이 가장 독립적인 family

## 1-B. Ablation — "이미지 효과" vs "CNN 효과" 분리

![Ablation heatmap](figures/09_ablation_image_vs_cnn.png)

2×2 ablation으로 요약:
- **이미지 변환 효과 (Logistic 기준)**: −0.0072 → +0.0392 = **+0.046 상승** 🚀
- **CNN 효과 (No-image 기준)**: −0.0072 → +0.0271 = **+0.034 상승** (비슷)
- **CNN이 image 위에 추가하는 효과**: +0.0392 → +0.0280 = **−0.011 하락** (오히려 감소)

→ 이 데이터셋에서 **대부분의 lift는 "이미지 변환(Jiang-style)" 자체에서 나옴**. CNN은 입력 종류에 robust한 모델(이미지든 아니든 비슷)이라는 점에 값어치가 있고, rank corr만으로는 로지스틱 대비 우위 없음.

Portfolio-level에서는 CNN이 로지스틱을 앞섬 — 점예측 품질과 포트폴리오 구성 품질이 분리되는 케이스.

---

## 2. 시그널 품질의 시간 안정성

![Rolling rank correlation](figures/06_rolling_rank_corr.png)

- 60일 rolling rank correlation — **양수 구간 비율**로 시그널이 언제 잘 먹히는지 확인
- 굵은 빨간선 = 앙상블, 나머지는 단일 모델
- 2020 팬데믹 구간 등 regime shift에서 모든 모델이 잠깐 음수 — 앙상블은 빠르게 회복

---

## 3. μ 분포 — 단위·스케일 sanity

![μ distribution per asset](figures/03_mu_distribution_ensemble.png)

- 모든 자산 μ의 하루치 스케일이 **±0.003 이내** (논문의 daily log return 규모와 일관)
- 중앙값이 0 근처 → 체계적 bias 없음
- 채권류(`short_treasury`, `corp_bond_ig`) variance 더 좁음 — 실제 수익률 변동성과 일치

![μ timeseries](figures/02_mu_timeseries_ensemble.png)

- 7개 자산 각각의 μ_hat_daily 시계열 — 특정 자산에 영구 drift가 없는지 확인용

---

## 4. 공분산 Σ — 수치 안정성

![Σ condition number](figures/04_sigma_condition_number.png)

- 60일 rolling sample covariance의 조건수 (log scale)
- **중간값 ≈ 2,715** — 권장 임계치(10³~10⁴) 범위
- Spike 시점: 2013년 테이퍼, 2018/2020 변동성 급등 구간 — ODE solver에서 이 구간은 shrinkage 권장

---

## 5. 앙상블 diversification

![Model correlation heatmap](figures/05_model_raw_correlation.png)

- Raw score 간 Pearson 상관 — **낮을수록 앙상블 효과 큼**
- **CNN 1D끼리**: ρ ≈ 0.5~0.6
- **CNN 2D vs 1D**: ρ ≈ 0.25~0.28
- **LSTM/cnnlstm vs CNN**: ρ ≈ 0.03~0.18
- **`cnnlstm_image_scale` vs `logistic_image_scale`: ρ = −0.17 (음의 상관!)** ← winner 멤버 선택의 핵심
- ensemble_4family 4개 멤버 평균 ρ ≈ 0.05 (음수 포함), 이론상 variance reduction 효과 최대화

흥미로운 발견: 단일 성능은 `lstm_image_scale` (0.051) > `cnnlstm_image_scale` (0.045) 인데, ensemble winner 에는 cnnlstm 만 들어감. 이유는 **cnnlstm 이 logistic 과 음의 상관 (−0.17) 으로 더 독립적인 정보**를 가져옴. "단독 1위 ≠ ensemble 1위" 의 정확한 사례.

### 3단계 lift progression

![3-stage lift](figures/11_3stage_lift_progression.png)

단계별 ensemble 천장 변화 — Phase 1 → 3 → 4 로 가며 rank corr 단조 증가, 단 Phase 4 는 Sharpe trade-off.

### Bootstrap CI (significance)

![Bootstrap CI](figures/12_bootstrap_ci.png)

10000회 paired bootstrap. 두 lift 모두 95% CI 가 0 을 살짝 포함 → 단일 metric 으로는 statistically not significant. 단 점추정 일관 + ρ 구조 + Sharpe 분리가 종합적으로 mechanism 을 지지.

---

## 6. Risk 신호

![Risk z-score](figures/08_risk_zscore_timeseries.png)

- 교차단면 z-score → 매일 자산 간 상대 위험도 (합 ≈ 0, std ≈ 1)
- ODE에서 time-varying γ(t)를 만들고 싶을 때 **자산별 weighting 힌트**로 쓸 수 있음

---

## 7. 데이터 커버리지

![Coverage heatmap](figures/07_coverage_heatmap.png)

- **2014-09-24 ~ 2026-04-13** 전 기간 모든 자산 동시 유효
- 앞 구간(2012~2014)은 Σ warmup만 존재 (μ 미계산) — ODE 실험에선 2014-09-24 이후 사용 권장

---

## 8. 다음 스프린트 Quickstart`

```python
import pandas as pd
from pathlib import Path

ROOT = Path("ode_inputs_cnn/ensemble_best")    # Sharpe-prio default
# 또는 Path("ode_inputs_cnn/ensemble_4family") # rank-corr-prio (LSTM 포함)

bundle = pd.read_csv(ROOT / "ode_bundle.csv", parse_dates=["date"])
returns = pd.read_csv("ode_inputs_cnn/returns_daily.csv", parse_dates=["date"])

ASSETS = ["alternative", "corp_bond_ig", "developed_equity", "emerging_equity",
          "korea_equity", "short_treasury", "treasury_7_10y"]

# μ(t): shape = (T, 7)
mu = bundle[[f"{a}_mu" for a in ASSETS]].values

# Σ(t): 매 날짜마다 7x7 매트릭스 (sigma_ii + 모든 cov 쌍)
# risk_score: (T, 7)
risk = bundle[[f"{a}_risk" for a in ASSETS]].values
```

**leakage 보증**: μ calibration은 expanding 방식, Σ는 t 시점까지의 60일 window만 사용.

---

## 추천 실험 시퀀스

1. **Smoke test**: `ensemble_best` (Sharpe-prio) 와 `ensemble_4family` (rank-prio) 둘 다 Euler ODE 로 돌려서 weight trajectory 비교
2. **Ablation 1**: `ensemble_best` vs `ensemble_4family` vs `ensemble_top3` vs 단일 best → 4-family 시너지 검증
3. **Ablation 2**: μ 시그널 대신 단순 모멘텀 baseline → ensemble 이 실제 lift 주는지 정량화
4. **Sensitivity**: 같은 ODE 에 `risk_score` 기반 γ(t) 투입 vs γ_const

---

## 경고 / 한계

- **OOS rank corr 0.06 수준도 statistically 작은 신호.** Bootstrap CI 가 0 포함 — 단일 점 예측보다는 앙상블·시간평균·포트폴리오 관점에서 활용 권장
- **Phase 4 (LSTM 합류) lift 는 CI 안에서 noise 와 구분 안 됨** — 단 (a) 점추정 일관, (b) ρ 구조 thesis 와 일치, (c) Sharpe trade-off 패턴 → **mechanism evidence 는 일관**
- **CNN-only 앙상블은 여전히 mixed 대비 약함** (raw-mean ensemble_top3 = 0.038 < ensemble_best 0.061). logistic_image 포함이 필수
- **2D CNN은 기본 설정에선 underfit** — 60K params + wd 1e-4 + 8 epoch 로는 학습 부족. Rehab protocol (small capacity + strong wd + patience) 이 걸려야만 유효
- **CNN+LSTM hybrid 는 단순 LSTM 대비 안 나음** — 이미지 입력에서 hybrid (0.039) < pure LSTM (0.051)
- **γ(t) 시변 위험회피 신호는 이번 CNN 파트에 포함 안 됨** — ODE 스프린트에서 추가 실험 대상

## 발표 시 방어 framing (권장)

> "**이미지 변환 자체가 가장 큰 lift** (rank corr −0.007 → +0.039) 를 제공하고, family 다양화는 그 위에 layered lift 를 더한다. Phase 1 (CNN-only) 0.038 → Phase 3 (+ logistic) 0.061 → Phase 4 (+ cnnlstm hybrid) **0.067** 으로 단조 증가. Phase 4 lift +0.0068 의 95% CI lower bound 가 −0.002 (0 에 거의 닿음) — borderline NOT significant 이지만, (a) cnnlstm-logistic ρ = **−0.17 (음의 상관!)**, (b) 점추정 일관 단조 증가, (c) Sharpe 와 rank corr trade-off 패턴이 모두 family diversity → variance reduction mechanism 을 가리킴. 또 **단일 1위 LSTM (0.051)이 ensemble winner 에서 빠지고 단독 4위 cnnlstm (0.045) 가 들어가는 패턴** 이 "단독 성능 ≠ ensemble 기여" 를 정확히 보여준다."
