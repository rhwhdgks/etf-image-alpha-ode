# 발표용 Figure 색인

[presentation_outline.md](../presentation_outline.md) 의 각 슬라이드와 1:1 매핑.

## 사용 figure (필수, 7장)

| 파일명 | 슬라이드 | 용도 | 권장 위치 |
|---|---|---|---|
| `slide02_jiang_sample_image.png` | Slide 2 | Jiang-style 입력 이미지 샘플 | 왼쪽 절반 |
| `slide04_model_comparison.png` | Slide 4 | 17 entries OOS 비교 (rank corr + Sharpe) | 큰 가로형, 슬라이드 60% |
| `slide05_ablation_image_vs_cnn.png` | Slide 5 | 2×2 ablation heatmap | 오른쪽 절반 |
| `slide06_correlation_matrix.png` | Slide 6 ★ | ρ 매트릭스 heatmap (thesis 핵심) | 오른쪽, 14×14 |
| `slide07_2d_cnn_loss_curves.png` | Slide 7 | undertrain 진단 학습곡선 | 왼쪽 60% |
| `slide09_3stage_lift_progression.png` | Slide 9 | Phase 1→3→4 lift 막대 그래프 | 큰 가로형 |
| `slide10_bootstrap_ci.png` | Slide 10 ★ | Bootstrap CI 분포 (정직성) | 큰 가로형 |

## 보조 figure (extras/, 6장)

발표 본 슬라이드엔 안 들어가지만 Q&A 또는 백업 슬라이드용:

| 파일명 | 용도 |
|---|---|
| `02_mu_timeseries_ensemble.png` | μ 시계열 sanity (특정 자산 drift 없음 증명) |
| `03_mu_distribution_ensemble.png` | μ 분포 (±0.003 scale, daily log return 규약 일치) |
| `04_sigma_condition_number.png` | Σ 조건수 (median 2715, 안정성 검증) |
| `06_rolling_rank_corr.png` | 60일 rolling rank corr (시간 안정성, regime shift) |
| `07_coverage_heatmap.png` | 데이터 커버리지 (2014-09-24~2026-04, 모든 자산 동시 유효) |
| `08_risk_zscore_timeseries.png` | risk_score 시계열 (γ(t) 입력 후보) |

## 폴더 구조

```
presentation_figures/  (4.0 MB)
├── INDEX.md
├── slide02_jiang_sample_image.png
├── slide04_model_comparison.png
├── slide05_ablation_image_vs_cnn.png
├── slide06_correlation_matrix.png
├── slide07_2d_cnn_loss_curves.png
├── slide09_3stage_lift_progression.png
├── slide10_bootstrap_ci.png
└── extras/
    ├── 02_mu_timeseries_ensemble.png
    ├── 03_mu_distribution_ensemble.png
    ├── 04_sigma_condition_number.png
    ├── 06_rolling_rank_corr.png
    ├── 07_coverage_heatmap.png
    └── 08_risk_zscore_timeseries.png
```

## 사용 팁

- **파일명에 슬라이드 번호 포함** → PPT 작업 시 순서대로 드래그 가능
- **★ 표시된 6, 10 번 슬라이드의 figure** 가 발표 thesis 의 핵심 — 가장 크게, 가장 선명하게 배치
- **160 DPI 해상도** 로 저장됨 — PPT 에서 크게 키워도 깨지지 않음
- 만약 figure 갱신 필요하면:
  - 코드 수정 후 [build_handoff_figures.py](../build_handoff_figures.py), [build_lift_progression_fig.py](../build_lift_progression_fig.py), [bootstrap_significance.py](../bootstrap_significance.py) 재실행
  - 그 다음 이 폴더 재배포
