# ODE `mu(t)` 논문 제출 보완 검증 작업공간

## 목적

이 작업공간은 `mu(t)` 입력 감사에서 남은 논문 제출 리스크를 보완하기 위한 별도 검증 결과다.
CNN/LSTM 모델을 새로 재학습하지 않고, 이미 export된 OOS `mu(t)` handoff 파일을 대상으로
fold-boundary purge와 후보 고정 평가를 수행한다.

## 작업공간 설정

- 원본 파일: `outputs/ode_handoff/02_mu_inputs/final_mu_inputs_wide.csv`
- 출력 폴더: `outputs/ode_handoff/06_mu_submission_validation`
- handoff grid 기준 test fold 길이: `60` dates
- purge 규칙: 각 fold의 첫 `20` OOS dates 제거
- Top-k: `2`
- Stationary block bootstrap: B=`5000`, mean block length=`20`

## 핵심 결과

- Purged `mu_image_factor_rank` rank corr: `0.0742`.
- Purged `mu_image_factor_strict_rank` rank corr: `0.0667`.
- Purged Sharpe 최고 후보: `mu_image_factor_strict_rank`, Sharpe `0.8122`.

해석: horizon-overlap 우려가 가장 큰 fold 시작부 날짜를 제거해도 주요 `mu(t)` 후보의 방향성은 유지된다.
다만 rank-corr lift의 block-bootstrap CI는 0을 포함하므로, 통계적으로 확정된 alpha라고 표현하면 안 된다.

## Fold-Boundary Purge Audit

| fold | fold_start | purged_start | purged_end | kept_start | fold_end | raw_test_dates | purged_dates | kept_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2014-09-24 | 2014-09-24 | 2014-10-21 | 2014-10-22 | 2014-12-17 | 60 | 20 | 40 |
| 1 | 2014-12-18 | 2014-12-18 | 2015-01-16 | 2015-01-20 | 2015-03-17 | 60 | 20 | 40 |
| 2 | 2015-03-18 | 2015-03-18 | 2015-04-15 | 2015-04-16 | 2015-06-11 | 60 | 20 | 40 |
| 3 | 2015-06-12 | 2015-06-12 | 2015-07-10 | 2015-07-13 | 2015-09-04 | 60 | 20 | 40 |
| 4 | 2015-09-08 | 2015-09-08 | 2015-10-05 | 2015-10-06 | 2015-12-01 | 60 | 20 | 40 |
| 46 | 2025-09-16 | 2025-09-16 | 2025-10-13 | 2025-10-14 | 2025-12-09 | 60 | 20 | 40 |
| 47 | 2025-12-10 | 2025-12-10 | 2026-01-08 | 2026-01-09 | 2026-03-09 | 60 | 20 | 40 |

총 `48`개 OOS fold에 대해 동일한 purge 규칙을 적용했다. 표는 앞 5개와 마지막 2개 fold만 보여준다.

## 후보 고정 성능

| sample | candidate | n_dates | rank_corr | top_k_sharpe | top_k_cumulative_return | top_k_hit_rate | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_oos | mu_rank_baseline | 2880 | 0.0674 | 0.5489 | 1.2586 | 0.6042 | 0.6084 |
| full_oos | mu_image_factor_rank | 2880 | 0.0798 | 0.2982 | 0.4962 | 0.5764 | 0.5455 |
| full_oos | mu_image_factor_strict_rank | 2880 | 0.0771 | 0.6846 | 1.7671 | 0.6111 | 0.5524 |
| full_oos | mu_image_factor_balanced | 2880 | 0.0731 | 0.4431 | 0.9219 | 0.5486 | 0.5804 |
| full_oos | mu_sharpe_baseline | 2880 | 0.0606 | 0.6234 | 1.4415 | 0.5903 | 0.6678 |
| purged_boundary_oos | mu_rank_baseline | 1920 | 0.0695 | 0.6630 | 0.9462 | 0.6562 | 0.5842 |
| purged_boundary_oos | mu_image_factor_rank | 1920 | 0.0742 | 0.2520 | 0.2263 | 0.6250 | 0.5684 |
| purged_boundary_oos | mu_image_factor_strict_rank | 1920 | 0.0667 | 0.8122 | 1.2551 | 0.6562 | 0.5842 |
| purged_boundary_oos | mu_image_factor_balanced | 1920 | 0.0716 | 0.3907 | 0.4544 | 0.5833 | 0.6000 |
| purged_boundary_oos | mu_sharpe_baseline | 1920 | 0.0701 | 0.6803 | 0.9288 | 0.6458 | 0.6421 |

## Block Bootstrap Lift

| sample | comparison | rank_corr_lift | block_ci_low | block_ci_high | n_aligned_dates |
| --- | --- | --- | --- | --- | --- |
| full_oos | image_factor_rank_vs_rank_baseline | 0.0124 | -0.0258 | 0.0512 | 2880 |
| full_oos | strict_rank_vs_rank_baseline | 0.0097 | -0.0270 | 0.0448 | 2880 |
| full_oos | balanced_vs_sharpe_baseline | 0.0125 | -0.0245 | 0.0505 | 2880 |
| purged_boundary_oos | image_factor_rank_vs_rank_baseline | 0.0047 | -0.0380 | 0.0483 | 1920 |
| purged_boundary_oos | strict_rank_vs_rank_baseline | -0.0028 | -0.0438 | 0.0391 | 1920 |
| purged_boundary_oos | balanced_vs_sharpe_baseline | 0.0015 | -0.0423 | 0.0444 | 1920 |

## Calibration 점검

| sample | candidate | n_dates | rank_corr | top_k_sharpe | top_k_hit_rate |
| --- | --- | --- | --- | --- | --- |
| full_oos | selected_mu_signal | 2609 | 0.0542 | 0.2640 | 0.5725 |
| full_oos | selected_mu_calibrated_daily | 2609 | 0.0542 | 0.2640 | 0.5725 |
| purged_boundary_oos | selected_mu_signal | 1749 | 0.0555 | 0.1021 | 0.5455 |
| purged_boundary_oos | selected_mu_calibrated_daily | 1749 | 0.0555 | 0.1021 | 0.5455 |

## 이번 보완으로 해결한 것

- 별도 재현 가능 검증 작업공간을 추가했다.
- 새 조합 탐색이 아니라 이미 고정한 최종 후보만 평가했다.
- 20일 overlapping target 이슈에 대해 fold-boundary purged sensitivity를 추가했다.
- 최종 lift 해석에 IID CI가 아니라 stationary block bootstrap CI를 사용했다.
- embargo가 적용된 non-null calibration 구간에서 return-scale `mu`를 점검했다.

## 남은 한계

이 결과는 이미 학습된 OOS signal에 대한 sensitivity analysis다.
따라서 purged/embargoed CNN 재학습을 완전히 대체하지는 못한다.
더 엄격한 학회/저널 제출을 목표로 한다면, 학습 전에 label-overlap row를 제거하는 fold 구성으로 deep-learning stack을 다시 돌리는 것이 최종 보완이다.

현재 코드에는 이 재학습을 위한 옵션이 추가되어 있다.

```bash
python build_image_factor_extension.py \
  --output-dir outputs/ode_handoff/06_mu_submission_validation/purged_retraining_candidate \
  --lookback 60 --horizon 20 \
  --wf-embargo-days 20 \
  --cnn-epochs 30 --patience 5 --weight-decay 5e-4
```

재학습 시 `cnn_training_history.csv`가 생성되어 fold별 train/validation loss와 best epoch를 확인할 수 있다.

## Purged Retraining Smoke Test

purged/embargoed 재학습 코드 경로가 실제로 동작하는지 1 fold / 1 epoch smoke test를 수행했다.

- 상태: `PASS`
- 출력 폴더: `outputs/ode_handoff/06_mu_submission_validation/purged_retraining_smoke`
- OOS rows: `420`
- OOS dates: `60`
- folds: `1`
- wf_embargo_days: `20`
- cnn_epochs: `1`
- training history 생성: `True`

이 smoke test는 성능 결론용이 아니라, purged fold 재학습 코드와 loss-history 기록 경로가 정상 동작함을 확인하는 용도다.

## 출력 파일

- `candidate_locked_metrics.csv`
- `purged_lift_block_bootstrap.csv`
- `calibration_sanity_metrics.csv`
- `fold_boundary_audit.csv`
- `manifest.json`
