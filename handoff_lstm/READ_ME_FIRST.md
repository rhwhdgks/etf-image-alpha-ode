# LSTM 팀원 핸드오프 — 여기부터 읽으세요

CNN 파트가 끝난 시점의 sprint2 스냅샷입니다. LSTM을 네 번째 family로 붙여
`ensemble_best`를 갱신하는 것이 목표.

## 폴더 구조

```
handoff_lstm/
├── READ_ME_FIRST.md           ← 지금 이 파일
├── 01_docs/
│   └── SPRINT2_HANDOFF.md     ★ 통합 문서 (Phase 4 갱신판)
├── 02_code/                   walk-forward + 앙상블 + bootstrap 파이프라인
├── 03_data/                   원본 ETF + 스키마 레퍼런스 예측 CSV
├── 04_reference_bundle/       현 결과물 (ensemble_best + ensemble_4family + 12 figures + significance)
│   ├── ensemble_best/         Phase 3 winner (Sharpe-prio default)
│   ├── ensemble_4family/      Phase 4 winner (rank-prio, LSTM 포함)
│   ├── ensemble_top3/         레거시 CNN-only 앙상블
│   ├── figures/               PNG 12종 (lift / bootstrap 포함)
│   ├── ensemble_search.csv    2940 조합 전수
│   └── significance_test.md   bootstrap CI 검정
└── 05_paper/                  ODE 논문 원문
```

## 읽는 순서 (30분)

**`01_docs/SPRINT2_HANDOFF.md` 한 문서로 통합**되어 있음. 순서대로 §1 TL;DR → §10 LSTM 합류 절차 → §11 번들 스키마 → §16 세 줄 교훈까지 읽으면 충분.

급하면 §1 (TL;DR) + §10 (LSTM 파트) + §11 (consumer 코드)만 봐도 작업 시작 가능.

## 환경 세팅

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r 02_code/requirements.txt
```

## LSTM 합류 절차 (실무)

1. **스키마 맞추기**: LSTM 예측 CSV를 `03_data/sample_walkforward_predictions.csv`
   와 동일한 컬럼(`date, asset, signal_value, future_return, model_name, ...`)
   으로 저장
2. **같은 fold로 학습**: `02_code/src/walkforward.py` 재사용 (fold 경계 일관성 필수)
3. **앙상블 탐색**:
   `02_code/build_extended_ensemble.py`의 `SOURCES` dict에 LSTM 항목 한 줄 추가
   → 다시 돌리면 기존 10개 + LSTM = 11개 × {raw, rank} 조합을 전부 탐색
4. **Best 번들 재생성**:
   새 best 조합을 `02_code/build_ensemble_best.py`의 `MEMBERS`에 넣고 실행
   → ODE 팀에 넘길 `mu_daily.csv / ode_bundle.csv` 갱신

## 비교 baseline (Phase 4 갱신)

현재 1위 (두 production 번들):

| 우선 metric | model | OOS rank corr | Top-k Sharpe |
|---|---|---|---|
| Sharpe | **`ensemble_best`** (3-family, Phase 3) | 0.0606 | **0.643** |
| Rank corr | **`ensemble_4family`** (+cnnlstm hybrid, Phase 4) | **0.0674** | 0.543 |
| 단일 LSTM 1위 | `lstm_image_scale` | 0.0506 | — |
| 단일 hybrid 1위 | `cnnlstm_image_scale` (Sharpe 0.434) | 0.0448 | — |
| 단일 CNN 1위 | `cnn_2d_residual_small` (Phase 2 rehab) | 0.0426 | — |
| 단일 logistic 1위 | `logistic_image_scale` | 0.0392 | — |

**핵심 발견**: 단독 1위 `lstm_image_scale` 이 ensemble winner 에서 빠지고 단독 4위 `cnnlstm_image_scale` 이 들어감 — cnnlstm vs logistic ρ = **−0.17 (음의 상관!)** 으로 더 독립적. Phase 3 → 4 lift +0.0068, 95% CI [−0.002, +0.016] borderline NOT significant 단 trend evidence 강해짐.

## 질문이 있으면

- ODE 입력 스키마 / calibration 규칙: `01_docs/02_bundle_README.md`
- Leakage 보증 / Σ 계산: 같은 문서 §6
- Phase 2/3 왜 2D CNN만 재조정했는지: `01_docs/blog_cnn_lstm_handoff.md` §6-C, §7
- 앙상블 탐색 770 조합 결과 raw: `04_reference_bundle/ensemble_search.csv`
