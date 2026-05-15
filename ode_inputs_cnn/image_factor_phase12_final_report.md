# Image Factor Phase 1-2 Final Report

## Scope

이번 문서는 Optimization으로 넘어가기 전, 네 담당 범위 중 아래 두 단계를 마감하기 위한 정리다.

- Phase 1: CNN image factor 형성 후 rolling PCA 공통요인 통제 유의성 검정
- Phase 2: 이미지 구성요소 ablation으로 어떤 chart element가 정보성을 만드는지 분해

Optimization, ODE solver 통합, 최종 portfolio weight 계산은 이 문서의 범위가 아니다.

## Experimental Setup

| 항목 | 설정 |
|---|---|
| 데이터 | `etfdata.csv`, 7 ETF assets |
| 모델 | `cnn_2d_residual_small` 고정 |
| Lookback / Horizon | `60 / 20` |
| 평가 방식 | walk-forward OOS |
| Fold 수 | 48 |
| OOS sample | 20,160 asset-date rows |
| OOS dates | 2,880 dates |
| Feature control | 252일 rolling return PCA loading 1-3 |
| 회귀 검정 | pooled panel OLS, date-clustered robust SE |
| 비교 대상 | 기존 `ensemble_best`, `ensemble_4family` |

모든 ablation variant는 같은 OOS `date-asset` grid를 사용했다. 각 variant의 panel은 중복 row와 factor NaN이 없다.

## Phase 1. Image Factor Significance

기존 Jiang-style full image factor extension에서는 `image_score`가 rolling PCA 공통요인을 통제한 뒤에도 유의했다.

| factor | t-stat | p-value | delta R2 | daily rank corr |
|---|---:|---:|---:|---:|
| `image_score` | 4.612 | 0.000004 | 0.002257 | 0.0121 |
| `image_factor_pc3` | -2.224 | 0.026196 | 0.000479 | -0.0136 |
| `image_factor_pc1` | 1.338 | 0.181100 | 0.000199 | 0.0406 |
| `image_factor_pc2` | -0.135 | 0.892576 | 0.000002 | 0.0130 |

해석은 명확하다. ETF 공통 움직임을 PCA loading으로 통제해도 CNN image score가 미래 수익률에 대해 추가 설명력을 가진다. 따라서 가격 chart path 정보가 단순 공통 market factor만 반영한 것은 아니라고 볼 수 있다.

다만 delta R2는 작다. 이 결과는 "강력한 단독 예측 모델"이라는 주장보다는, "이미지 기반 path factor가 통계적으로 남는다"는 근거로 사용하는 것이 맞다.

## Phase 2. Image Component Ablation

6개 이미지 variant를 같은 조건에서 비교했다.

| variant | 의미 |
|---|---|
| `close_only` | 종가 path 선만 사용 |
| `ohlc_full` | OHLC candle만 사용 |
| `ohlc_ma` | OHLC candle + moving average |
| `ohlc_volume` | OHLC candle + volume |
| `ohlc_ma_volume` | OHLC candle + moving average + volume |
| `high_low_range` | high-low range block만 강조 |

### 2.1 PCA Control 이후 가장 유의한 factor

| rank | variant::factor | t-stat | p-value | delta R2 | daily rank corr |
|---:|---|---:|---:|---:|---:|
| 1 | `ohlc_ma::image_score` | 7.232 | 6.07e-13 | 0.008839 | 0.0184 |
| 2 | `close_only::image_factor_pc2` | 6.859 | 8.44e-12 | 0.004431 | 0.0196 |
| 3 | `ohlc_volume::image_factor_pc1` | -6.537 | 7.41e-11 | 0.003877 | 0.0198 |
| 4 | `high_low_range::image_factor_pc3` | 6.273 | 4.08e-10 | 0.003972 | 0.0429 |
| 5 | `ohlc_ma::image_factor_pc2` | 5.146 | 2.84e-07 | 0.004413 | 0.0065 |

가장 강한 유의성은 `ohlc_ma::image_score`에서 나왔다. 이는 이동평균선이 추가된 가격 path가 공통요인 통제 이후에도 정보성을 갖는다는 근거다.

### 2.2 Daily Rank Correlation 기준 가장 좋은 factor

| rank | variant::factor | daily rank corr | t-stat | p-value | delta R2 |
|---:|---|---:|---:|---:|---:|
| 1 | `ohlc_ma_volume::image_factor_pc1` | 0.0592 | 3.551 | 0.000390 | 0.001586 |
| 2 | `high_low_range::image_factor_pc3` | 0.0429 | 6.273 | 4.08e-10 | 0.003972 |
| 3 | `ohlc_volume::image_factor_pc2` | 0.0304 | -1.648 | 0.099415 | 0.000351 |
| 4 | `ohlc_ma_volume::image_score` | 0.0285 | 3.338 | 0.000853 | 0.001174 |
| 5 | `ohlc_ma::image_factor_pc3` | 0.0265 | 1.994 | 0.046195 | 0.000379 |

ranking 관점에서는 `ohlc_ma_volume::image_factor_pc1`이 가장 좋다. 즉, OHLC candle에 MA와 volume을 모두 포함한 full chart representation이 cross-sectional ranking에는 가장 유용했다.

### 2.3 Ensemble 추가 결과

기존 ensemble과 image factor를 cross-sectional percentile rank 평균으로 결합했다.

| candidate | rank corr | Sharpe | rank corr lift | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| `ensemble_4family` | 0.0674 | 0.5430 | 0.0000 | baseline |
| `ensemble_4family + ohlc_ma_volume_pc1` | 0.0798 | 0.3155 | 0.0124 | [0.00001, 0.02505] |
| `ensemble_best` | 0.0606 | 0.6425 | 0.0000 | baseline |
| `ensemble_best + ohlc_ma_volume_pc1` | 0.0731 | 0.5109 | 0.0125 | [0.00057, 0.02448] |

핵심은 `ohlc_ma_volume::image_factor_pc1`이 기존 ensemble의 rank corr를 올렸다는 점이다. 특히 `ensemble_4family` 기준 lift의 bootstrap CI가 거의 0 위에 있으므로, ranking 개선 evidence는 있다.

반면 Sharpe는 감소했다. 따라서 이 factor는 단독 trading rule 또는 Sharpe 최적 신호라기보다 ODE에 넘길 expected-return ranking input 후보로 쓰는 것이 적절하다.

## What We Learned

1. 가격 이미지에는 정보가 있다.

PCA 공통요인 통제 이후에도 image factor가 유의했다. 이는 ETF들이 같이 움직이는 공통 market component만으로 설명되지 않는 path 정보가 남는다는 의미다.

2. 이동평균선은 통계적 유의성에 중요했다.

`ohlc_ma::image_score`가 가장 강한 p-value와 delta R2를 보였다. 추세선 형태의 path feature가 정보성을 만든다는 해석이 가능하다.

3. MA와 volume을 모두 포함한 full chart는 ranking에 가장 유용했다.

`ohlc_ma_volume::image_factor_pc1`은 daily rank correlation이 가장 높았고, 기존 ensemble에 추가했을 때 rank corr를 개선했다.

4. Sharpe와 rank corr는 같은 방향으로 움직이지 않았다.

image factor를 추가하면 rank corr는 개선됐지만 Sharpe는 낮아졌다. 따라서 최종 목적이 ODE input 제공이라면, 이 신호는 portfolio rule이 아니라 `mu(t)` ranking 후보로 넘기는 것이 맞다.

## Final Phase 1-2 Conclusion

Phase 1-2의 결론은 다음 한 문장으로 정리할 수 있다.

> Jiang-style ETF 가격 이미지에서 추출한 CNN image factor는 rolling PCA 공통요인을 통제한 뒤에도 유의하며, 특히 MA와 volume을 포함한 full chart factor는 기존 CNN/LSTM ensemble의 cross-sectional return ranking을 추가로 개선한다.

따라서 1번과 2번은 마감 가능하다. 다음 단계는 이 결과를 바탕으로 ODE 팀에 넘길 `mu(t)` candidate input을 정리하는 것이다.

## Files To Use

| 목적 | 파일 |
|---|---|
| Phase 1 image factor panel | `ode_inputs_cnn/image_factor_extension/image_factor_panel.csv` |
| Phase 1 significance | `ode_inputs_cnn/image_factor_extension/image_factor_significance.csv` |
| Phase 1 report | `ode_inputs_cnn/image_factor_extension/image_factor_report.md` |
| Phase 2 ablation summary | `ode_inputs_cnn/image_factor_ablation/ablation_summary.csv` |
| Phase 2 all significance tests | `ode_inputs_cnn/image_factor_ablation/ablation_significance.csv` |
| Phase 2 ensemble search | `ode_inputs_cnn/image_factor_ablation/ablation_ensemble_search.csv` |
| Phase 2 report | `ode_inputs_cnn/image_factor_ablation/image_factor_ablation_report.md` |
| Best ranking signal candidate | `ode_inputs_cnn/image_factor_ablation/ohlc_ma_volume/ode_mu_candidate_signals.csv` |

## Status Checklist

| item | status |
|---|---|
| Image factor extraction | complete |
| Rolling PCA controls | complete |
| PCA-control significance test | complete |
| 6-way image component ablation | complete |
| Same OOS grid across variants | complete |
| Ensemble add-on test | complete |
| Signal files exported | complete |
| Optimization / ODE solver | intentionally excluded |
