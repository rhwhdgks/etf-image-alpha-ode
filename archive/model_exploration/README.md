# Model Exploration Archive

이 폴더는 최종 발표/제출 메시지를 단순하게 유지하기 위해 분리한 탐색 산출물입니다.
삭제한 것은 아니며, 재현성이나 세부 ablation 확인이 필요할 때만 참고합니다.

## 최종 폴더에 남긴 모델

- `ode_inputs_cnn/cnn_1d_cumulative_scale`: 기존 ensemble에 기여한 1D CNN 대표
- `ode_inputs_cnn/cnn_2d_residual_small`: Jiang-style image factor extractor
- `ode_inputs_cnn/ensemble_best`: ODE 기본 `mu(t)` 후보
- `ode_inputs_cnn/ensemble_4family`: rank-corr 우선 비교 후보
- `ode_inputs_cnn/image_factor_extension`: image factor 검정 및 ODE 후보 신호

## 여기로 이동한 것

- `ode_inputs_cnn/`: multiscale, dilated, attention, rendered 2D, 초기 residual 2D 등 탐색용 CNN 번들
- `walkforward_outputs/`: 중간 walk-forward 실험 결과
- `ensemble_top3`: 이전 단계 ensemble 산출물

## 발표/보고서에서의 표현

초기에는 여러 CNN 변형을 실험했지만, 최종 연구에서는 두 역할로 압축한다.

- 1D CNN: ensemble member로서 cross-sectional score에 기여
- 2D CNN: Jiang-style 가격 이미지에서 image factor를 추출하는 extractor

