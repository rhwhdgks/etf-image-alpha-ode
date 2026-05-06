# sprint2-etf-cnn

7개 ETF 자산의 μ(기대수익) 시그널을 CNN으로 생성해 ODE 기반 동적 포트폴리오 최적화 스프린트에 넘겨주는 핸드오프 패키지.

## 요약

- 60일 OHLCV를 Jiang-style 이미지로 변환 → **CNN 8종 + LSTM 4종 + logistic 2종 = 14 entries** 를 walk-forward OOS 평가
- 평가: 24~48 folds × 7년 OOS × 2,880일
- 결과물: `ode_inputs_cnn/` 아래 14 모델별 μ·Σ·risk·R 번들 + 3종 앙상블 (top3 / best / 4family) + QA + 베이스라인 비교 + bootstrap significance
- **Phase 3** (CNN + logistic mix) `ensemble_best` rank corr **0.0606**, Sharpe **0.630**
- **Phase 4** (CNN+LSTM hybrid 합류, `cnnlstm_image_scale`) `ensemble_4family` rank corr **0.0674** / Sharpe **0.543** — 점추정 +0.0068 향상, 95% CI [−0.002, +0.016] 0 살짝 포함 (borderline NOT significant). cnnlstm vs logistic ρ = **−0.17 (음의 상관)** 으로 분산 감소 효과 큼 — **상관 구조 thesis 강하게 지지**
- 자세한 해설은 [blog_cnn_lstm_handoff.md](blog_cnn_lstm_handoff.md)

## 디렉토리

```
sprint2-etf-cnn/
├── src/                             모델·데이터·평가 코드
├── etfdata.csv                      원본 ETF 일별 OHLCV
├── requirements.txt
│
├── run_walkforward.py               walk-forward OOS 실행 entrypoint
├── make_ode_inputs.py               μ calibration, rolling Σ, ODE 번들 생성
├── collect_cnn_ode_signals.py       CNN 예측 → 모델별 번들 디렉토리 생성
├── build_handoff_package.py         README/QA/ensemble 생성 (1-shot)
├── build_handoff_figures.py         발표용 figure 8종 생성
├── build_baseline_comparison.py     CNN vs logistic 베이스라인 비교 + ablation
│
├── outputs_walkforward_4model/      walk-forward 원본 예측 (CNN 4 + 2 logistic)
├── outputs_walkforward_mu/          1D CNN μ 실험
├── outputs_walkforward_1dcnn_extra/ attention/dilated/multiscale/cumulative
├── outputs_walkforward_2d_residual/ 2D CNN + ResNet
├── outputs_walkforward_2d_phase2/   Phase 2 rehab — cnn_2d_residual_small/wd
├── outputs_walkforward_2d_fix/      Phase 1 2D diagnostic (30ep + patience 5)
├── outputs_walkforward_risk/        risk signal용 walk-forward
├── lstm/                            LSTM 4종 walk-forward 결과 (팀원 산출물)
│
├── build_extended_ensemble.py       14모델 × 1470 조합 ensemble search
├── build_ensemble_best.py           Phase 3 winner (3-멤버) 번들 빌드
├── build_ensemble_4family.py        Phase 4 winner (4-멤버, LSTM 포함) 번들 빌드
├── build_lift_progression_fig.py    3단계 lift figure
├── bootstrap_significance.py        paired bootstrap CI 검정
│
├── ode_inputs_cnn/                  ★ 최종 핸드오프 패키지
│   ├── README.md
│   ├── HANDOFF_SUMMARY.md           팀 1페이지 브리핑
│   ├── qa_report.md / qa_report.json
│   ├── comparison.csv / comparison.md
│   ├── comparison_with_baselines.csv
│   ├── ensemble_search.csv          770→2940 조합 전수 결과
│   ├── significance_test.md         bootstrap CI 검정 결과
│   ├── returns_daily.csv / prices_daily.csv
│   ├── figures/                     발표용 PNG 12종
│   ├── ensemble_best/               ★ Phase 3 default (Sharpe-prio)
│   ├── ensemble_4family/            ★ Phase 4 alt (rank corr-prio, LSTM 포함)
│   ├── ensemble_top3/               레거시 CNN-only 앙상블
│   └── {model_name}/                12 모델별 mu_daily / ode_bundle / ode_config
│
└── blog_cnn_lstm_handoff.md         CNN → LSTM 로드맵 블로그 글
```

## 빠른 재현

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. walk-forward 실행 (결과 → outputs_walkforward_*/)
python run_walkforward.py --lookback 60 --horizon 20

# 2. CNN 예측 → ODE 입력 번들
python collect_cnn_ode_signals.py \
  --pred-paths outputs_walkforward_4model/walkforward_predictions.csv \
               outputs_walkforward_1dcnn_extra/walkforward_predictions.csv \
               outputs_walkforward_2d_residual/walkforward_predictions.csv \
  --risk-path outputs_walkforward_risk/walkforward_predictions.csv \
  --output-dir ode_inputs_cnn

# 3. 핸드오프 패키지 finalize (README/QA/ensemble)
python build_handoff_package.py

# 4. 발표용 figure
python build_handoff_figures.py

# 5. 베이스라인 비교 + ablation
python build_baseline_comparison.py
```

## 다음 스프린트가 쓰는 방법

```python
import pandas as pd
from pathlib import Path

ROOT = Path("ode_inputs_cnn/ensemble_best")  # 권장; ensemble_top3는 레거시 CNN-only

bundle = pd.read_csv(ROOT / "ode_bundle.csv", parse_dates=["date"])
returns = pd.read_csv("ode_inputs_cnn/returns_daily.csv", parse_dates=["date"])

ASSETS = ["alternative", "corp_bond_ig", "developed_equity", "emerging_equity",
          "korea_equity", "short_treasury", "treasury_7_10y"]

mu = bundle[[f"{a}_mu" for a in ASSETS]].values        # (T, 7)
risk = bundle[[f"{a}_risk" for a in ASSETS]].values    # (T, 7)
```

Leakage 보증: μ calibration은 expanding 방식, Σ는 t 시점까지의 60일 window만 사용.

## 핵심 결과

| 지표 | 1위 | 값 | 추천 default |
|---|---|---|---|
| Top-k Sharpe (portfolio quality) | ★ `ensemble_best` (3-family) | **0.630** | ✓ Sharpe-prio |
| OOS rank corr (signal quality) | ★ `ensemble_4family` (4-family, +LSTM) | **0.0633** | ✓ rank-prio |
| 최고 단일 LSTM | `lstm_image_scale` | rank 0.0506 | — |
| 최고 단일 CNN | `cnn_2d_residual_small` (Phase 2 rehab) | rank 0.0426 | — |

| Phase | 변화 | rank corr | Sharpe | 비고 |
|---|---|---|---|---|
| 1 | CNN-only top-3 ensemble | 0.0375 | 0.275 | logistic_image (0.039) 천장 못 넘음 |
| 3 | + logistic_image (mixed family) | 0.0606 | **0.630** | 두 metric 동시 1위 |
| 4 | + cnnlstm_image (4-family hybrid) | **0.0674** | 0.543 | rank +0.007 / Sharpe trade-off (작아짐) |

**Bootstrap CI**: Phase 3 → Phase 4 lift +0.0068 의 95% CI [−0.002, +0.016] → borderline NOT significant. 단 cnnlstm vs logistic ρ = −0.17 (음의 상관) — family 다양화 thesis 의 mechanism 강하게 지지.

→ 다음 단계: ODE 본체 통합 + (선택) 5번째 family — [blog 글](blog_cnn_lstm_handoff.md) 참고.

## 참고

- 논문: An ODE-Based Dynamic Mean-Variance Portfolio Optimisation with Time-Varying Risk Aversion
- 이미지 변환: Jiang et al. 2016 (60일 OHLCV → 2D 이미지)
