"""Проверки, которые надёжнее посчитать кодом, чем спрашивать у LLM.

Они только ужесточают итог: если проверка кодом не пройдена, пункт будет failed,
даже если LLM ответила successed. Если пройдена — решает LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from .config import Settings
from .prompt import FAILED, SUCCESS

TARGET_ASPECT = 35.0 / 45.0  # ширина / высота


@dataclass
class LocalResult:
    test_type: str
    result: str
    comments: List[str] = field(default_factory=list)

    @property
    def comment(self) -> str:
        return " ".join(self.comments)


def _center(arr: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    h, w = arr.shape[:2]
    return arr[int(h * y0): int(h * y1), int(w * x0): int(w * x1)]


def _laplacian_var(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def analyze(img: Image.Image, s: Settings) -> Tuple[Dict[str, LocalResult], Dict[str, float]]:
    """img — исходное изображение (RGB, уже повёрнутое по EXIF, до уменьшения)."""
    results: Dict[str, LocalResult] = {}
    metrics: Dict[str, float] = {}

    # 1. Пропорции кадра 35×45
    w, h = img.size
    aspect = w / float(h)
    deviation = abs(aspect / TARGET_ASPECT - 1.0)
    metrics.update(width=w, height=h, aspect=round(aspect, 4), aspect_deviation=round(deviation, 4))
    fmt = LocalResult("format", SUCCESS)
    if deviation > s.aspect_tolerance:
        fmt.result = FAILED
        fmt.comments.append(
            f"Пропорции кадра {w}×{h} (≈{aspect:.2f}) не соответствуют 35×45 мм (≈0.78) — обрежьте фото по формату 35×45."
        )
    results["format"] = fmt

    # Для остальных метрик работаем с уменьшенной копией: быстро и не зависит от разрешения
    small = img.copy()
    small.thumbnail((512, 512))
    rgb = np.asarray(small, dtype=np.float32)
    gray = np.asarray(small.convert("L"), dtype=np.float32)
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)

    # 2. Цветное ли фото (средняя насыщенность и разброс каналов)
    sat = float(hsv[..., 1].mean() / 255.0)
    spread = float((np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 1] - rgb[..., 2])).mean())
    metrics.update(mean_saturation=round(sat, 4), channel_spread=round(spread, 2))

    light = LocalResult("lighting_color", SUCCESS)
    if sat < 0.05 and spread < 6:
        light.result = FAILED
        light.comments.append("Фото чёрно-белое — требуется цветная фотография.")

    # 3. Пересвет / затемнение в зоне лица (центр кадра). Фон не учитываем:
    #    белый фон для фото на документы — норма. Порог по «выбитым» пикселям,
    #    а не по средней яркости, чтобы не зависеть от цвета кожи.
    face = _center(gray, 0.28, 0.72, 0.22, 0.70)
    over = float((face >= 248).mean())
    under = float((face <= 12).mean())
    metrics.update(face_overexposed=round(over, 4), face_underexposed=round(under, 4))
    if over > s.overexposed_frac:
        light.result = FAILED
        light.comments.append("Пересвет: значительная часть лица «выбелена», детали не видны.")
    if under > s.underexposed_frac:
        light.result = FAILED
        light.comments.append("Затемнение: значительная часть лица слишком тёмная, детали не видны.")
    results["lighting_color"] = light

    # 4. Резкость (дисперсия лапласиана в зоне лица)
    sharp = _laplacian_var(face) if face.size else 0.0
    metrics.update(sharpness=round(sharp, 2))
    quality = LocalResult("quality", SUCCESS)
    if sharp < s.blur_threshold:
        quality.result = FAILED
        quality.comments.append("Снимок размыт или не в фокусе — нужна чёткая фотография.")
    results["quality"] = quality

    return results, metrics
