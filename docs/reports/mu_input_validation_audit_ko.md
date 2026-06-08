# ODE용 `mu(t)` 입력 검증 감사

이 문서는 ODE optimizer에 넘길 `mu(t)` 후보가 논문 제출 기준에서 leakage, overfit, underfit, calibration 문제를 얼마나 방어할 수 있는지 점검한 결과다.

## 결론

현재 handoff의 `mu(t)` 입력은 **ODE 실험용 research input으로는 사용 가능**하다. 특히 `selected_mu_input.csv`의 `mu_signal`은 날짜별 ETF cross-section 안에서 만든 centered-rank signal이라 단위 폭주 위험이 작고, `returns_for_ode.csv`, `sigma_shrunk_wide.csv`와 날짜 grid가 맞는다.

다만 논문에서 “오류가 없다” 또는 “통계적으로 확정된 alpha다”라고 쓰면 안 된다. 가장 큰 이유는 두 가지다.

- 20일 forward return target을 쓰면서 walk-forward fold에 horizon embargo가 없다. train/validation 날짜는 test 날짜보다 앞서지만, 마지막 20일 내외의 학습 label은 test 기간과 forward-return window가 겹칠 수 있다.
- image-factor add-on의 rank-corr lift는 양의 점추정이지만, autocorrelation을 반영한 block bootstrap에서는 0을 배제하지 못한다.

따라서 논문용 표현은 다음 수준이 적절하다.

> “이미지 기반 `mu(t)` 후보는 OOS rank quality를 개선하는 경향을 보였고, ODE 입력으로 사용할 수 있는 leakage-controlled handoff 형식으로 정리했다. 다만 overlapping horizon과 regime sensitivity 때문에, 최종 alpha claim은 block-bootstrap 및 purged walk-forward 민감도 결과와 함께 제한적으로 해석한다.”

## 실제 파일 검증

검증 기준 경로는 `outputs/ode_handoff/`이다.
재현 가능한 파일 무결성 검사는 repo root에서 아래 명령으로 실행한다.

```bash
python3 verify_ode_mu_inputs.py
```

현재 실행 결과는 `PASS`이며, 이 검사는 CSV grid, 중복, 결측, centered-rank, calibration warm-up, returns/Sigma alignment를 확인한다. 통계적 유의성은 이 스크립트가 아니라 block-bootstrap 및 multiple-testing 보고서로 판단한다.

| 항목 | 결과 |
|---|---:|
| 추천 `mu` 파일 | `outputs/ode_handoff/02_mu_inputs/selected_mu_input.csv` |
| 행 수 | 20,160 |
| 날짜 수 | 2,880 |
| 자산 수 | 7 |
| 날짜 범위 | 2014-09-24 ~ 2026-03-09 |
| 날짜별 행 수 | 7 고정 |
| `date, asset` 중복 | 0 |
| 주요 `mu` 컬럼 결측 | 0 |
| `mu_signal` 날짜별 합 | 전 날짜 0, centered-rank 조건 만족 |
| `returns_for_ode.csv`와 grid | 완전 일치 |
| `sigma_shrunk_wide.csv`와 날짜 | 완전 일치 |

`selected_mu_input_calibrated.csv`에는 초반 1,897행의 `mu_calibrated_daily` 결측이 있다. 이는 오류가 아니라 252일 최소 학습기간과 20일 horizon embargo 때문에 생긴 정상 warm-up 구간이다. calibration이 필요한 ODE 실험에서는 2015-10-21 이후 non-null 구간부터 쓰면 된다.

## 추가 보완 작업공간

감사 이후 별도 작업공간을 만들어 논문 제출 리스크를 추가로 줄였다.

| 항목 | 내용 |
|---|---|
| 작업공간 | `outputs/ode_handoff/06_mu_submission_validation/` |
| 재현 명령 | `python3 run_mu_submission_validation.py` |
| 핵심 검증 | 후보 고정 평가, fold-boundary purge, stationary block bootstrap, calibration sanity, purged retraining smoke |
| purge 규칙 | 60일 OOS fold마다 첫 20일 제거 |
| bootstrap | B=5,000, mean block length=20 |

주요 결과:

| sample | 후보 | rank corr | Sharpe | 해석 |
|---|---|---:|---:|---|
| full OOS | `mu_image_factor_rank` | 0.0798 | 0.2982 | rank corr 최상위 |
| purged boundary | `mu_image_factor_rank` | 0.0742 | 0.2520 | fold 시작부 제거 후에도 rank 방향성 유지 |
| purged boundary | `mu_image_factor_strict_rank` | 0.0667 | 0.8122 | 보수적 MA 기준 Sharpe 최상위 |
| purged boundary | `mu_sharpe_baseline` | 0.0701 | 0.6803 | Sharpe 비교군 |

block-bootstrap lift:

| 비교 | sample | lift | 95% block CI | 판단 |
|---|---|---:|---|---|
| `mu_image_factor_rank - mu_rank_baseline` | full OOS | +0.0124 | [-0.0258, +0.0512] | not significant |
| `mu_image_factor_rank - mu_rank_baseline` | purged boundary | +0.0047 | [-0.0380, +0.0483] | not significant |
| `mu_image_factor_strict_rank - mu_rank_baseline` | purged boundary | -0.0028 | [-0.0438, +0.0391] | not significant |

이 보완의 의미는 명확하다. horizon-overlap에 가장 민감한 fold 시작부를 제거해도 `mu_image_factor_rank`의 rank corr는 양수로 유지된다. 하지만 baseline 대비 lift는 작아지고 CI가 0을 포함하므로, 논문에서는 “유의한 초과 alpha”가 아니라 “ODE 입력 후보로 방향성 있는 signal”이라고 표현해야 한다.

추가로 코드 레벨 보완도 완료했다.

- `PipelineConfig.wf_embargo_days`를 추가했다.
- `run_walkforward.py`, `build_image_factor_extension.py`, `build_image_factor_ablation.py`, `build_image_factor_robustness.py`에서 `--wf-embargo-days` 옵션을 받을 수 있다.
- `src.walkforward.generate_walkforward_folds()`는 embargo가 켜지면 test 직전 validation 날짜를 제거한다.
- `src.models.cnn.fit_torch_model()`은 `training_history_`, `best_epoch_`, `best_val_loss_`를 남긴다.
- image-factor 재학습 시 `cnn_training_history.csv`가 생성된다.
- 1 fold / 1 epoch purged retraining smoke test를 실행했고, `outputs/ode_handoff/06_mu_submission_validation/purged_retraining_smoke/`에 결과를 남겼다.

smoke 결과:

| 항목 | 값 |
|---|---:|
| status | PASS |
| rows | 420 |
| dates | 60 |
| folds | 1 |
| `wf_embargo_days` | 20 |
| `cnn_epochs` | 1 |
| `cnn_training_history.csv` 생성 | True |

## 추천 입력

| 목적 | 파일 | 컬럼 | 판단 |
|---|---|---|---|
| rank-scale ODE 실험 | `selected_mu_input.csv` | `mu_signal` | 1순위 추천 |
| return-scale ODE 실험 | `selected_mu_input_calibrated.csv` | `mu_calibrated_daily` | 단위가 수익률이어야 할 때 사용 |
| Sharpe 우선 비교군 | `final_mu_inputs_wide.csv` | `mu_signal_mu_sharpe_baseline` | ODE ablation 비교용 |
| 보수적 image robustness | `final_mu_inputs_wide.csv` | `mu_signal_mu_image_factor_strict_rank` | MA lead/lag 우려 대응용 |

`future_return`은 평가용 target이다. ODE optimizer input으로 넣으면 leakage가 된다.

## Overfit 점검

| 체크 | 현재 상태 | 판단 |
|---|---|---|
| OOS walk-forward | 시간 순서 기반 walk-forward 사용 | 통과 |
| feature PCA | train+validation feature에만 fit, test는 transform만 적용 | 통과 |
| rolling PCA control | signal date 이하 과거 252일만 사용 | 통과 |
| Sigma estimate | trailing window 기반, shrunk covariance 제공 | 통과 |
| calibration | expanding OLS + horizon embargo 적용 | 통과 |
| multiple testing | Bonferroni/FDR 보고서 존재 | 부분 통과 |
| autocorrelation-aware CI | stationary block bootstrap 보고서 존재 | 통과, 단 결과는 유의하지 않음 |
| subperiod stability | pre-covid, covid/inflation, recent 구간 보고 | 부분 통과 |
| 모델 선택 bias | 여러 모델/variant 탐색 후 최종 선택 | 추가 보완 필요 |
| walk-forward label embargo | fold 자체에는 horizon embargo 없음 | 추가 보완 필요 |

핵심은 “큰 feature leakage는 발견되지 않았지만, 모델 선택과 overlapping target 때문에 과적합 위험을 완전히 제거했다고 말할 수는 없다”이다.

## Underfit 점검

초기 2D CNN은 undertraining 문제가 있었다. 기존 8 epoch, patience 2 설정에서 validation loss가 충분히 내려가기 전에 멈췄고, 이후 `cnn_2d_residual_small`로 구조를 줄이고 `30 epoch`, `patience 5`, `weight_decay 5e-4`, `dropout 0.2`를 적용했다.

현재 메인 image-factor extractor는 이 조정된 `cnn_2d_residual_small`이다. 따라서 초기 2D CNN underfit 문제는 상당 부분 해결됐다고 볼 수 있다.

다만 논문 제출용으로는 다음 근거가 더 있으면 좋다.

- 최종 모델의 fold별 train/validation loss curve 요약
- early stopping epoch 분포
- seed별 성능 분산
- `cnn_2d_residual_small`과 더 작은 CNN/linear image model의 동일 grid 비교

## Calibration 점검

`mu_signal`은 기대수익률 단위가 아니라 cross-sectional centered rank다. ODE 코드가 `mu(t)`를 상대 매력도 점수로 받아도 된다면 `mu_signal`이 더 안정적이다.

수익률 단위의 `mu(t)`가 필요하면 `mu_calibrated_daily`를 사용한다. 이 값은 다음 방식으로 만들어졌다.

- 각 input별 expanding pooled OLS
- 식: `future_return ~ 1 + mu_signal`
- 최소 252 signal dates
- horizon embargo 적용
- `mu_calibrated_horizon / 20 = mu_calibrated_daily`

calibration 후 daily rank corr는 낮아진다. 예를 들어 `mu_image_factor_rank`는 calibrated 기준 0.0542이고, strict 후보는 0.0610이다. 이는 leakage를 줄인 결과로 해석하는 것이 맞다.

## 성능 해석

| 후보 | Rank Corr | Top-k Sharpe | 해석 |
|---|---:|---:|---|
| `mu_rank_baseline` | 0.0674 | 0.5430 | image factor 이전 4-family baseline |
| `mu_image_factor_rank` | 0.0798 | 0.3155 | rank quality 최상위, 추천 `mu` |
| `mu_image_factor_strict_rank` | 0.0771 | 0.6053 | 보수적 MA 기준, Sharpe 우수 |
| `mu_sharpe_baseline` | 0.0606 | 0.6425 | Sharpe 우선 비교군 |

`mu_image_factor_rank`가 rank corr는 가장 높지만 Sharpe는 낮다. 논문에서는 “예측 ranking 개선”과 “simple top-k portfolio 성과”를 분리해서 써야 한다. ODE에서는 optimizer가 `Sigma(t)`와 risk aversion을 함께 사용하므로, rank corr가 높은 후보와 Sharpe가 높은 후보를 모두 넣어 비교하는 것이 가장 방어 가능하다.

## 남은 제출 전 보완

위 보완으로 후보 고정 평가와 fold-boundary purge sensitivity는 완료했다. 아직 남은 엄격한 보완은 아래 순서다.

1. **Purged / embargoed walk-forward 재검증**  
   코드 옵션과 1-fold smoke는 완료했다. 더 엄격한 논문 제출을 위해서는 아래 명령을 전체 48 fold / 30 epoch로 끝까지 돌려 최종 성능표를 갱신해야 한다.

   ```bash
   python build_image_factor_extension.py \
     --output-dir outputs/ode_handoff/06_mu_submission_validation/purged_retraining_candidate \
     --lookback 60 --horizon 20 \
     --wf-embargo-days 20 \
     --cnn-epochs 30 --patience 5 --weight-decay 5e-4
   ```

2. **Final candidate lock 후 재평가**  
   완료. `run_mu_submission_validation.py`에서 `mu_image_factor_rank`, `mu_image_factor_strict_rank`, `mu_sharpe_baseline` 등 고정 후보만 평가했다.

3. **Seed stability**  
   CNN seed 3~5개로 최종 후보의 rank corr와 factor direction이 유지되는지 확인한다.

4. **Loss curve summary**  
   최종 `cnn_2d_residual_small`의 fold별 early stopping epoch와 validation loss 분포를 남긴다. underfit/overfit 지적에 대한 방어 자료다.

5. **ODE-level ablation**  
   ODE 팀에서 `mu_rank_baseline`, `mu_image_factor_rank`, `mu_image_factor_strict_rank`, `mu_sharpe_baseline`을 같은 `Sigma(t)`와 risk-aversion으로 비교한다. 최종 주장은 raw rank corr가 아니라 ODE portfolio 성과로 닫아야 한다.

## 최종 판단

현재 `mu(t)` 값은 다음 조건에서는 사용 가능하다.

- ODE 입력 후보로 사용한다.
- `future_return`을 input으로 쓰지 않는다.
- rank-scale은 `mu_signal`, return-scale은 `mu_calibrated_daily`로 구분한다.
- image-factor 성능은 “유의한 확정 alpha”가 아니라 “positive trend / mechanism evidence”로 표현한다.
- 논문 제출 전에는 purged/embargoed 재학습 또는 그에 준하는 sensitivity를 명확히 한계로 공개한다.

현재 상태를 한 줄로 정리하면:

> ODE팀에 넘겨서 실험할 수 있는 수준은 맞고, fold-boundary purge 후에도 방향성은 유지된다. 다만 논문에서 완전히 오류 없는 최종 `mu`라고 주장하려면 purged/embargoed 재학습과 seed/loss-curve 안정성 검증을 추가해야 한다.
