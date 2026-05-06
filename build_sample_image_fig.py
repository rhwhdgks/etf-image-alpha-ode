#!/usr/bin/env python3
"""Slide 2 용 — Jiang-style 입력 이미지 grid figure.

원본 입력 이미지가 64x32 정도로 작아서 PPT 에 넣으면 픽셀화. 6개 샘플을 큰
figure 에 grid 로 배치하고 자산명/날짜 라벨링.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
SAMPLES_DIR = ROOT / "lstm" / "sample_images"
OUT = ROOT / "presentation_figures" / "slide02_jiang_sample_image.png"


def main() -> None:
    files = sorted(SAMPLES_DIR.glob("*.png"))[:6]
    if not files:
        raise FileNotFoundError(SAMPLES_DIR)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, f in zip(axes.flat, files):
        img = mpimg.imread(f)
        ax.imshow(img, cmap="gray", aspect="auto", interpolation="nearest")
        # filename: alternative_20120702.png → "alternative · 2012-07-02"
        stem = f.stem
        asset, date = stem.rsplit("_", 1)
        date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        ax.set_title(f"{asset} · {date_fmt}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Jiang-style input image samples — 60-day OHLCV encoded as 2D grayscale pixels",
                 fontsize=15, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
