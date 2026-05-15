from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


CHART_VARIANTS = {
    "close_only",
    "ohlc_full",
    "ohlc_ma",
    "ohlc_volume",
    "ohlc_ma_volume",
    "high_low_range",
}


def chart_variant_uses_moving_average(chart_variant: str) -> bool:
    _validate_chart_variant(chart_variant)
    return chart_variant in {"ohlc_ma", "ohlc_ma_volume"}


def chart_variant_uses_volume(chart_variant: str) -> bool:
    _validate_chart_variant(chart_variant)
    return chart_variant in {"ohlc_volume", "ohlc_ma_volume"}


def _validate_chart_variant(chart_variant: str) -> None:
    if chart_variant not in CHART_VARIANTS:
        options = ", ".join(sorted(CHART_VARIANTS))
        raise ValueError(f"unsupported chart_variant={chart_variant!r}; choose one of: {options}")


def _scaled_price_values(window: pd.DataFrame, fields: list[str]) -> dict[str, np.ndarray]:
    arrays = [window[field].to_numpy(dtype=float) for field in fields]
    price_min = float(np.nanmin(np.concatenate(arrays)))
    price_max = float(np.nanmax(np.concatenate(arrays)))
    price_range = price_max - price_min
    if not np.isfinite(price_range) or price_range < 1e-8:
        price_range = 1.0

    return {
        field: np.clip((window[field].to_numpy(dtype=float) - price_min) / price_range, 0.0, 1.0)
        for field in fields
    }


def _scaled_volume_values(window: pd.DataFrame) -> np.ndarray:
    volume = window["volume"].to_numpy(dtype=float)
    volume_max = float(np.nanmax(volume))
    if not np.isfinite(volume_max) or volume_max < 1e-8:
        volume_max = 1.0
    return np.clip(volume / volume_max, 0.0, 1.0)


def _price_fields_for_variant(chart_variant: str, use_moving_average: bool) -> list[str]:
    if chart_variant == "close_only":
        return ["close"]
    if chart_variant == "high_low_range":
        return ["high", "low"]

    fields = ["open", "high", "low", "close"]
    if use_moving_average:
        fields.append("ma")
    return fields


def render_jiang_chart(
    window: pd.DataFrame,
    image_height: int,
    include_moving_average: bool,
    include_volume: bool,
    chart_variant: str = "ohlc_ma_volume",
) -> np.ndarray:
    """Render a Jiang-style chart image.

    The include flags preserve the old API. The chart variant decides which
    visual elements are eligible; the flags decide whether optional MA/volume
    inputs are actually available for this call.
    """
    _validate_chart_variant(chart_variant)
    use_moving_average = include_moving_average and chart_variant_uses_moving_average(chart_variant)
    use_volume = include_volume and chart_variant_uses_volume(chart_variant)

    width = len(window) * 3
    height = image_height
    price_height = int(round(height * 0.8)) if use_volume else height
    volume_height = max(height - price_height, 1)
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)

    price_fields = _price_fields_for_variant(chart_variant, use_moving_average)
    scaled_price = _scaled_price_values(window, price_fields)
    scaled_volume = _scaled_volume_values(window) if use_volume else None

    def to_price_row(value: float) -> int:
        return int(round((price_height - 1) * (1.0 - float(value))))

    def to_volume_row(value: float) -> int:
        return price_height + int(round((volume_height - 1) * (1.0 - float(value))))

    previous_close_point: tuple[int, int] | None = None
    previous_ma_point: tuple[int, int] | None = None

    for day_idx in range(len(window)):
        x0 = day_idx * 3
        x1 = x0 + 1
        x2 = x0 + 2

        if chart_variant == "close_only":
            close_point = (x1, to_price_row(scaled_price["close"][day_idx]))
            draw.point(close_point, fill=255)
            if previous_close_point is not None:
                draw.line([previous_close_point, close_point], fill=255)
            previous_close_point = close_point

        elif chart_variant == "high_low_range":
            high_row = to_price_row(scaled_price["high"][day_idx])
            low_row = to_price_row(scaled_price["low"][day_idx])
            y0, y1 = sorted((high_row, low_row))
            draw.rectangle([(x0, y0), (x2, y1)], fill=255)

        else:
            open_row = to_price_row(scaled_price["open"][day_idx])
            high_row = to_price_row(scaled_price["high"][day_idx])
            low_row = to_price_row(scaled_price["low"][day_idx])
            close_row = to_price_row(scaled_price["close"][day_idx])

            draw.line([(x1, high_row), (x1, low_row)], fill=255)
            draw.line([(x0, open_row), (x1, open_row)], fill=255)
            draw.line([(x1, close_row), (x2, close_row)], fill=255)

            if use_moving_average:
                ma_point = (x1, to_price_row(scaled_price["ma"][day_idx]))
                if previous_ma_point is not None:
                    draw.line([previous_ma_point, ma_point], fill=255)
                previous_ma_point = ma_point

        if use_volume and scaled_volume is not None:
            volume_top = to_volume_row(scaled_volume[day_idx])
            draw.rectangle([(x0, volume_top), (x2, height - 1)], fill=255)

    return np.asarray(image, dtype=np.uint8)


def save_chart_preview(image_array: np.ndarray, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array, mode="L").save(path)
