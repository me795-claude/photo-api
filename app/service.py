"""Конвейер проверки: подготовка фото → проверки кодом → LLM → слияние результатов."""
from __future__ import annotations

import io
import logging
import time
from typing import Any, Dict, Optional

from PIL import Image, ImageOps, UnidentifiedImageError

from . import local_checks
from .config import Settings
from .llm import LLMClient
from .prompt import FAILED, SUCCESS, TEST_TYPES

log = logging.getLogger("photo_check")


class InvalidImageError(ValueError):
    pass


def load_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImageError("файл не является изображением или повреждён") from e
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def to_jpeg(img: Image.Image, max_side: int, quality: int) -> bytes:
    im = img.copy()
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


class PhotoChecker:
    def __init__(self, settings: Settings):
        self.s = settings
        self.llm = LLMClient(settings)

    async def check(self, data: bytes, filename: Optional[str] = None) -> Dict[str, Any]:
        started = time.monotonic()
        img = load_image(data)

        local: Dict[str, local_checks.LocalResult] = {}
        metrics: Dict[str, float] = {}
        if self.s.local_checks:
            local, metrics = local_checks.analyze(img, self.s)

        jpeg = to_jpeg(img, self.s.image_max_side, self.s.jpeg_quality)
        llm_checks, llm_meta = await self.llm.check_image(jpeg)

        # Слияние: проверка кодом может только «завалить» пункт
        checks = []
        for item in llm_checks:
            lr = local.get(item["test_type"])
            if lr is not None and lr.result == FAILED:
                comment = lr.comment
                if item["result"] == FAILED and item["comment"]:
                    comment = f"{comment} {item['comment']}"
                item = {"test_type": item["test_type"], "result": FAILED, "comment": comment.strip()}
            checks.append(item)

        overall = SUCCESS if all(c["result"] == SUCCESS for c in checks) else FAILED
        return {
            "checks": checks,
            "overall": overall,
            "meta": {
                "file": filename,
                "latency_ms": int((time.monotonic() - started) * 1000),
                **llm_meta,
                "llm_checks": llm_checks,
                "local_checks": {k: {"result": v.result, "comment": v.comment} for k, v in local.items()},
                "local_metrics": metrics,
            },
        }


__all__ = ["PhotoChecker", "InvalidImageError", "TEST_TYPES"]
