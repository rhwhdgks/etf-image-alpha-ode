#!/usr/bin/env python3
"""Generate sprint2 presentation deck (.pptx) using FIND_A_템플릿.pptx as the base.

Strategy:
- Slide 0 of template: cover (logos) — rewrite as title slide
- Slide 1 of template: content layout w/ header bar + section title rectangle +
  body text box + slide number. Duplicate this 12 times for content slides.

Each content slide gets:
- section title (the rectangle "섹션 설명")
- body text or figure (replacing/extending the "TextBox 28" body)
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE

ROOT = Path(__file__).parent
TEMPLATE = ROOT / "FIND_A_템플릿.pptx"
OUT = ROOT / "sprint2_presentation.pptx"
FIG = ROOT / "ode_inputs_cnn" / "figures"
SAMPLE_IMG = ROOT / "lstm" / "sample_images" / "alternative_20120702.png"

FONT_KOR = "Pretendard"
TITLE_SIZE = Pt(20)
BODY_SIZE = Pt(14)
SMALL_SIZE = Pt(11)


# === slide content (12 content slides) ============================
SLIDES = [
    # 0 — title
    {
        "kind": "title",
        "title": "Sprint 2 — ETF μ Signal Pipeline",
        "subtitle": "CNN + LSTM ensemble · ODE 동적 포트폴리오 입력",
    },
    # 1 — 문제
    {
        "section": "1. 무엇을 푸는가",
        "title": "ODE 기반 동적 포트폴리오 — μ 시그널 생성",
        "bullets": [
            "7개 ETF 자산에 매일 얼마씩 넣을지를 미분방정식으로 결정",
            "dw/dt = f(μ(t), Σ(t), γ(t)) — 우리 역할은 μ(t) 시그널 생성",
            "Σ·R 은 과거 데이터로 자동, μ 는 미래 예측 — 어려운 부분",
            "랭킹 (어떤 자산이 더 오를지) 만 잘 맞아도 포트폴리오엔 충분",
        ],
    },
    # 2 — Jiang 이미지
    {
        "section": "2. 입력 — Jiang-style 이미지 변환",
        "title": "60일 OHLCV → 2D 흑백 이미지 → CNN",
        "bullets": [
            "사람 트레이더가 차트 보고 판단하듯 모델도 이미지로 본다 (Jiang-Kelly-Xiu 2023)",
            "캔들 모양 + 이동평균선 + 거래량 막대 → 픽셀 인코딩",
            "왼쪽: 60일 alternative 자산 입력 이미지 샘플",
        ],
        "image": str(SAMPLE_IMG),
        "image_pos": (Inches(0.64), Inches(2.7), Inches(4.5), Inches(3.5)),
    },
    # 3 — 실험 설계
    {
        "section": "3. 실험 설계",
        "title": "13 모델 walk-forward OOS — 24~48 folds × 2,880일",
        "bullets": [
            "8 CNN (1D 4종 + 2D 3종 + Phase 2 rehab 1종) + 2 logistic + 2 LSTM + 2 CNN+LSTM hybrid",
            "Walk-forward expanding: train_max_date < test_min_date 항상 보장 → leakage 차단",
            "Lookback 60일, horizon 20일 (월 1회 rebalance 가정)",
            "평가 metric: Spearman rank corr (점예측) + top-2 Sharpe (포트폴리오)",
        ],
    },
    # 4 — 결과 (fig 01)
    {
        "section": "4. 결과 — 단독은 모두 같은 티어",
        "title": "rank corr 1위 ensemble_4family · Sharpe 1위 ensemble_best",
        "bullets": [
            "단독 model 격차 (CNN 0.043 ↔ logistic 0.039 ↔ LSTM 0.051) 는 noise 범위",
            "★★ ensemble_4family (rank corr 0.0674) ★ ensemble_best (Sharpe 0.643)",
        ],
        "image": str(FIG / "01_model_comparison.png"),
        "image_pos": (Inches(2.0), Inches(2.7), Inches(9.0), Inches(4.0)),
    },
    # 5 — Ablation
    {
        "section": "5. Ablation — 이미지 효과 vs 모델 효과",
        "title": "rank corr lift 의 대부분은 이미지 변환에서",
        "bullets": [
            "Logistic + 이미지: −0.007 → +0.039 (+0.046 lift) ← Jiang 의 공",
            "CNN + 이미지: +0.027 → +0.028 (≈0 lift) ← 이미지 위에 CNN 얹는 효과 미미",
            "→ rank corr 만 보면 CNN 이 logistic 을 이긴다고 단정 못함. Sharpe 는 별개.",
        ],
        "image": str(FIG / "09_ablation_image_vs_cnn.png"),
        "image_pos": (Inches(7.5), Inches(2.7), Inches(5.3), Inches(4.0)),
    },
    # 6 — 상관 구조
    {
        "section": "6. ★ 진짜 레버 — 상관 구조",
        "title": "ρ 가 ensemble 천장을 결정한다",
        "bullets": [
            "CNN-CNN ρ ≈ 0.5~0.6 (같은 family, 닮음) → CNN-only 앙상블 천장 명확",
            "LSTM-CNN ρ ≈ 0.03~0.18 ← 가장 독립적인 family",
            "★ cnnlstm vs logistic: ρ = −0.17 (음의 상관!) ← 분산 감소 효과 최대",
            "Var(평균) ≈ ρ·σ² (n→∞): 신호 강도보다 ρ 가 결과를 지배",
        ],
        "image": str(FIG / "05_model_raw_correlation.png"),
        "image_pos": (Inches(7.5), Inches(2.7), Inches(5.3), Inches(4.0)),
    },
    # 7 — Phase 2 rehab
    {
        "section": "7. Phase 2 — 2D CNN underperform 진단",
        "title": "원인은 데이터 부족 아니라 undertrain + overparam",
        "bullets": [
            "원본 cnn_2d_residual: 8 epoch + patience 2 → val loss 가 epoch 18에 minimum 인데 일찍 끊김",
            "Rehab: capacity 60K→23K + dropout 0.2 + wd 5e-4 + patience 5 → rank 0.007 → 0.043 (4.3배)",
            "교훈: \"underperform\" 의 원인을 단정짓지 말고 학습곡선부터 찍어라",
        ],
        "image": str(FIG / "10_2d_cnn_loss_curves.png"),
        "image_pos": (Inches(0.64), Inches(2.7), Inches(7.5), Inches(4.0)),
    },
    # 8 — Phase 4 cnnlstm
    {
        "section": "8. Phase 4 — cnnlstm hybrid 합류",
        "title": "★ 단독 1위 LSTM 이 ensemble winner 에서 빠지는 패턴",
        "bullets": [
            "단독 LSTM 1위: lstm_image (rank 0.0506), 단독 hybrid 1위: cnnlstm_image (rank 0.0448)",
            "ensemble_4family winner = logistic + 1D + 2D + cnnlstm_image (NOT lstm_image)",
            "이유: cnnlstm vs logistic ρ = −0.17 (음의 상관), lstm vs cnnlstm ρ = 0.63 (둘 다 못 들어감)",
            "→ \"단독 성능 ≠ ensemble 기여\" 의 정확한 사례",
        ],
    },
    # 9 — 3-stage lift
    {
        "section": "9. 3단계 lift progression",
        "title": "rank corr 단조 증가: 0.038 → 0.061 → 0.067",
        "bullets": [
            "Phase 1 (CNN-only top-3): 0.0375 / Sharpe 0.275",
            "Phase 3 (+ logistic, 3-family): 0.0606 / Sharpe 0.643 ← 두 metric 동시 1위",
            "Phase 4 (+ cnnlstm, 4-family): 0.0674 / Sharpe 0.543 ← rank 천장 갱신, Sharpe trade-off",
        ],
        "image": str(FIG / "11_3stage_lift_progression.png"),
        "image_pos": (Inches(2.0), Inches(2.7), Inches(9.0), Inches(4.0)),
    },
    # 10 — Bootstrap CI
    {
        "section": "10. 정직성 — Bootstrap CI",
        "title": "Phase 3 → 4 lift 의 95% CI 가 0 살짝 포함 (borderline)",
        "bullets": [
            "ensemble_4family − ensemble_best: +0.0068, 95% CI [−0.0024, +0.0157] → borderline NOT significant",
            "이전 (lstm 버전) lift +0.0027 / CI [−0.007, +0.012] 대비 점추정 2.5배, CI 0 에 거의 닿음",
            "→ statistical proof 부족, 단 mechanism evidence (점추정 일관 + ρ = −0.17) 강하게 일관",
        ],
        "image": str(FIG / "12_bootstrap_ci.png"),
        "image_pos": (Inches(2.0), Inches(2.7), Inches(9.0), Inches(4.0)),
    },
    # 11 — 4줄 교훈
    {
        "section": "11. 정리",
        "title": "4줄 교훈",
        "bullets": [
            "① \"새 모델이 베이스라인을 이긴다\" 프레임 경계 — 같은 티어에서 서로 다른 강점을 본다",
            "② 앙상블의 진짜 레버는 상관 구조 — 0.038 → 0.061 → 0.067 단조 증가의 mechanism",
            "③ Underperform 원인을 단정짓지 말고 학습곡선부터 찍어라 — 2D CNN rehab",
            "④ Lift 가 보여도 CI 까지 보자 — \"lift 있다\" ≠ \"mechanism evidence 일관\"",
        ],
    },
    # 12 — 다음
    {
        "section": "12. 다음 — ODE 통합 + 확장",
        "title": "두 production 번들 + 후속 실험",
        "bullets": [
            "ensemble_best (Sharpe-prio default, 0.643) + ensemble_4family (rank-prio, +cnnlstm)",
            "ODE 팀: 두 입력으로 weight trajectory 비교 → \"점예측 미세 lift 가 portfolio level 어떻게 펼쳐지나\"",
            "확장 후보: 5번째 family (tree / transformer), γ(t) 시변 위험회피, sigma shrinkage",
        ],
    },
]


def duplicate_slide(prs, src_slide_index: int):
    """Duplicate a slide by copying all shapes; returns the new slide."""
    src = prs.slides[src_slide_index]
    blank_layout = prs.slide_layouts[11]  # 빈 레이아웃
    new_slide = prs.slides.add_slide(blank_layout)
    for shp in src.shapes:
        el = shp.element
        new_slide.shapes._spTree.insert_element_before(deepcopy(el), "p:extLst")
    return new_slide


def find_shape(slide, *, name: str | None = None, contains_text: str | None = None):
    for sh in slide.shapes:
        if name and sh.name == name:
            return sh
        if contains_text and sh.has_text_frame and contains_text in sh.text_frame.text:
            return sh
    return None


def set_text(shape, text: str, *, size=BODY_SIZE, bold: bool | None = None, color=None):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.name = FONT_KOR
    run.font.size = size
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def set_bullets(shape, bullets: list[str], *, size=BODY_SIZE):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, line in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run()
        run.text = "• " + line
        run.font.name = FONT_KOR
        run.font.size = size


def fill_content_slide(slide, cfg):
    sec_shape = find_shape(slide, contains_text="섹션 설명") or find_shape(slide, name="직사각형 7191")
    if sec_shape:
        set_text(sec_shape, cfg["section"], size=Pt(16), bold=True)
    body_shape = find_shape(slide, contains_text="목차목차") or find_shape(slide, contains_text="본문본문")
    if body_shape:
        tf = body_shape.text_frame
        tf.clear()
        tf.word_wrap = True
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = cfg["title"]
        run.font.name = FONT_KOR
        run.font.size = Pt(22)
        run.font.bold = True
        body_shape.left = Inches(0.64)
        body_shape.top = Inches(0.95)
        body_shape.width = Inches(12.05)
        body_shape.height = Inches(1.2)
    if cfg.get("bullets"):
        img_pos = cfg.get("image_pos")
        if img_pos:
            ix = img_pos[0]
            if ix.emu < Inches(6).emu:
                bullets_box = slide.shapes.add_textbox(Inches(7.5), Inches(2.7), Inches(5.3), Inches(4.0))
            else:
                bullets_box = slide.shapes.add_textbox(Inches(0.64), Inches(2.7), Inches(6.7), Inches(4.0))
        else:
            bullets_box = slide.shapes.add_textbox(Inches(0.64), Inches(2.4), Inches(12.05), Inches(4.5))
        set_bullets(bullets_box, cfg["bullets"], size=Pt(15))
    if cfg.get("image"):
        x, y, w, h = cfg["image_pos"]
        slide.shapes.add_picture(cfg["image"], x, y, width=w, height=h)


def main():
    prs = Presentation(str(TEMPLATE))

    cover_slide = prs.slides[0]

    # First duplicate the clean template (slide 1) for each additional content slide,
    # BEFORE filling anything. This way each new slide is a clean copy.
    n_extra = len(SLIDES) - 2  # slides 2..N (slide 1 reuses template)
    new_slides = [duplicate_slide(prs, 1) for _ in range(n_extra)]

    # === Slide 0 — cover (title) ==========================
    cover = SLIDES[0]
    title_box = cover_slide.shapes.add_textbox(Inches(0.8), Inches(2.4), Inches(11.5), Inches(1.2))
    set_text(title_box, cover["title"], size=Pt(36), bold=True)
    sub_box = cover_slide.shapes.add_textbox(Inches(0.8), Inches(3.7), Inches(11.5), Inches(0.8))
    set_text(sub_box, cover["subtitle"], size=Pt(20), color=RGBColor(0x55, 0x55, 0x55))

    # === Slide 1 — first content (reuse template's slide 1) =====
    fill_content_slide(prs.slides[1], SLIDES[1])

    # === Slides 2..N — fill the duplicates =====================
    for cfg, new in zip(SLIDES[2:], new_slides):
        fill_content_slide(new, cfg)

    prs.save(str(OUT))
    print(f"saved: {OUT}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
