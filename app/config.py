"""Настройки сервиса из переменных окружения (.env)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _json(name: str) -> Dict[str, Any]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} должен быть JSON-объектом")
    return value


@dataclass(frozen=True)
class Settings:
    # Подключение к Open WebUI
    base_url: str
    api_key: str
    model: str
    verify_ssl: bool = True
    timeout: float = 180.0

    # Параметры запроса
    temperature: float = 0.0
    max_tokens: int = 2000
    seed: int = 42
    response_format: str = "json_schema"  # json_schema | json_object | none
    max_retries: int = 2
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # Изображение
    image_max_side: int = 1024
    jpeg_quality: int = 90
    max_upload_mb: int = 15

    # Проверки кодом
    local_checks: bool = True
    aspect_tolerance: float = 0.04
    blur_threshold: float = 2.0
    overexposed_frac: float = 0.20
    underexposed_frac: float = 0.35

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/api/chat/completions"

    @property
    def models_url(self) -> str:
        return self.base_url.rstrip("/") + "/api/models"


def load_settings() -> Settings:
    fmt = os.getenv("LLM_RESPONSE_FORMAT", "json_schema").strip().lower()
    if fmt not in {"json_schema", "json_object", "none"}:
        raise ValueError("LLM_RESPONSE_FORMAT: допустимо json_schema | json_object | none")
    return Settings(
        base_url=os.getenv("OPENWEBUI_URL", "http://localhost:3000"),
        api_key=os.getenv("OPENWEBUI_API_KEY", ""),
        model=os.getenv("LLM_MODEL", "qwen3.8-flash-next"),
        verify_ssl=_bool("OPENWEBUI_VERIFY_SSL", True),
        timeout=_float("LLM_TIMEOUT", 180.0),
        temperature=_float("LLM_TEMPERATURE", 0.0),
        max_tokens=_int("LLM_MAX_TOKENS", 2000),
        seed=_int("LLM_SEED", 42),
        response_format=fmt,
        max_retries=_int("LLM_MAX_RETRIES", 2),
        extra_params=_json("LLM_EXTRA_PARAMS"),
        image_max_side=_int("IMAGE_MAX_SIDE", 1024),
        jpeg_quality=_int("IMAGE_JPEG_QUALITY", 90),
        max_upload_mb=_int("MAX_UPLOAD_MB", 15),
        local_checks=_bool("LOCAL_CHECKS", True),
        aspect_tolerance=_float("LOCAL_ASPECT_TOLERANCE", 0.04),
        blur_threshold=_float("LOCAL_BLUR_THRESHOLD", 2.0),
        overexposed_frac=_float("LOCAL_OVEREXPOSED_FRAC", 0.20),
        underexposed_frac=_float("LOCAL_UNDEREXPOSED_FRAC", 0.35),
    )
