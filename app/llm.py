"""Клиент Open WebUI: POST /api/chat/completions (OpenAI-совместимый)."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .config import Settings
from .prompt import FAILED, RESPONSE_SCHEMA, SUCCESS, SYSTEM_PROMPT, TEST_TYPES, USER_PROMPT

log = logging.getLogger("photo_check.llm")

_FORMAT_FALLBACK = {"json_schema": "json_object", "json_object": "none", "none": None}

_RESULT_ALIASES = {
    "successed": SUCCESS, "succeeded": SUCCESS, "success": SUCCESS, "successful": SUCCESS,
    "passed": SUCCESS, "pass": SUCCESS, "ok": SUCCESS, "true": SUCCESS, "yes": SUCCESS,
    "failed": FAILED, "fail": FAILED, "failure": FAILED, "false": FAILED, "no": FAILED,
}


class LLMError(Exception):
    """LLM недоступна или так и не вернула корректный ответ."""


class ResponseValidationError(ValueError):
    pass


# ---------------------------------------------------------------- разбор ответа

def _strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    if "</think>" in text.lower():  # незакрытое начало рассуждений
        text = re.split(r"</think>", text, flags=re.I)[-1]
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    return text.strip()


def _extract_json(text: str) -> Any:
    text = _strip_reasoning(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ResponseValidationError("в ответе нет JSON")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    try:
        return json.loads(text[start: end + 1])
    except json.JSONDecodeError as e:
        raise ResponseValidationError(f"некорректный JSON: {e}") from e


def parse_checks(text: str) -> List[Dict[str, str]]:
    """Достаёт из ответа модели 9 проверок и приводит их к строгому формату."""
    data = _extract_json(text)

    if isinstance(data, dict) and isinstance(data.get("checks"), list):
        items = data["checks"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict) and set(data) & set(TEST_TYPES):  # {"format": {...}, ...}
        items = [dict(v, test_type=k) if isinstance(v, dict) else {"test_type": k, "result": v}
                 for k, v in data.items()]
    else:
        raise ResponseValidationError('ожидался объект {"checks": [...]}')

    found: Dict[str, Dict[str, str]] = {}
    problems: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            problems.append(f"элемент не объект: {item!r}")
            continue
        tt = str(item.get("test_type", "")).strip().lower()
        if tt not in TEST_TYPES:
            problems.append(f"неизвестный test_type {tt!r}")
            continue
        if tt in found:
            problems.append(f"test_type {tt!r} повторяется")
            continue
        raw_result = str(item.get("result", "")).strip().lower()
        result = _RESULT_ALIASES.get(raw_result)
        if result is None:
            problems.append(f"{tt}: недопустимое значение result {raw_result!r}")
            continue
        comment = str(item.get("comment") or "").strip()
        if result == FAILED and not comment:
            comment = f"Не пройдена проверка «{TEST_TYPES[tt]}»."
        found[tt] = {"test_type": tt, "result": result, "comment": comment if result == FAILED else ""}

    missing = [t for t in TEST_TYPES if t not in found]
    if missing:
        problems.append("нет проверок: " + ", ".join(missing))
    if problems:
        raise ResponseValidationError("; ".join(problems))
    return [found[t] for t in TEST_TYPES]


# ---------------------------------------------------------------- клиент

class OpenWebUIClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self._format: Optional[str] = settings.response_format  # может откатиться при 400

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.s.api_key:
            h["Authorization"] = f"Bearer {self.s.api_key}"
        return h

    def _payload(self, messages: List[Dict[str, Any]], fmt: Optional[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.s.model,
            "stream": False,
            "temperature": self.s.temperature,
            "max_tokens": self.s.max_tokens,
            "seed": self.s.seed,
            "messages": messages,
        }
        if fmt == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "photo_check", "strict": True, "schema": RESPONSE_SCHEMA},
            }
        elif fmt == "json_object":
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.s.extra_params)
        return payload

    @staticmethod
    def _read_content(resp: httpx.Response) -> str:
        """Текст ответа модели — и для обычного JSON, и для SSE (если админ включил стриминг)."""
        ctype = resp.headers.get("content-type", "")
        if "text/event-stream" in ctype:
            parts: List[str] = []
            for line in resp.text.splitlines():
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if not chunk or chunk == "[DONE]":
                    continue
                try:
                    event = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                for choice in event.get("choices") or []:  # первое событие может быть {"sources": ...}
                    delta = choice.get("delta") or choice.get("message") or {}
                    if delta.get("content"):
                        parts.append(delta["content"])
            return "".join(parts)

        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ResponseValidationError(f"неожиданная структура ответа: {str(data)[:300]}")
        if isinstance(content, list):  # контент частями
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content or ""

    async def list_models(self) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=30, verify=self.s.verify_ssl) as client:
            r = await client.get(self.s.models_url, headers=self._headers())
            r.raise_for_status()
            data = r.json()
        return data.get("data", data) if isinstance(data, dict) else data

    async def check_image(self, jpeg: bytes) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]

        attempts = 0
        last_error = "нет ответа"
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=self.s.timeout, verify=self.s.verify_ssl) as client:
            while attempts <= self.s.max_retries:
                attempts += 1
                fmt = self._format
                try:
                    resp = await client.post(self.s.chat_url, headers=self._headers(),
                                             json=self._payload(messages, fmt))
                except httpx.HTTPError as e:
                    last_error = f"ошибка соединения: {e!r}"
                    log.warning("попытка %d: %s", attempts, last_error)
                    await asyncio.sleep(min(2 ** attempts, 10))
                    continue

                # Модель/бэкенд не поддерживает response_format — откатываемся и пробуем снова
                # (если 400 по другой причине — дойдём до "none" и получим исходную ошибку)
                if resp.status_code in (400, 422) and fmt in ("json_schema", "json_object"):
                    nxt = _FORMAT_FALLBACK[fmt]
                    log.warning("response_format=%s отклонён (HTTP %s: %s), пробую %s",
                                fmt, resp.status_code, resp.text[:200], nxt)
                    self._format = nxt
                    attempts -= 1  # смена формата не считается попыткой
                    continue

                if resp.status_code in (401, 403):
                    raise LLMError(f"доступ запрещён (HTTP {resp.status_code}): проверьте OPENWEBUI_API_KEY")
                if resp.status_code == 404:
                    raise LLMError(f"HTTP 404: проверьте OPENWEBUI_URL и LLM_MODEL ({resp.text[:200]})")
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    retry_after = resp.headers.get("retry-after", "")
                    delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else min(2 ** attempts, 10)
                    log.warning("попытка %d: %s, повтор через %.1fс", attempts, last_error, delay)
                    await asyncio.sleep(delay)
                    continue
                if resp.status_code >= 400:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")

                content = ""
                try:
                    content = self._read_content(resp)
                    checks = parse_checks(content)
                except ResponseValidationError as e:
                    last_error = f"ответ не по схеме: {e}"
                    log.warning("попытка %d: %s", attempts, last_error)
                    # Даём модели её ответ и описание ошибки, просим исправить
                    if content:
                        messages = messages[:2] + [
                            {"role": "assistant", "content": content[:4000]},
                            {"role": "user", "content": (
                                f"Ответ не соответствует требуемой схеме: {e}. "
                                "Верни исправленный JSON: объект {\"checks\": [...]} ровно с 9 элементами "
                                "(test_type, result = successed|failed, comment). Только JSON.")},
                        ]
                    continue

                meta = {
                    "model": self.s.model,
                    "attempts": attempts,
                    "response_format": fmt,
                    "llm_latency_ms": int((time.monotonic() - started) * 1000),
                }
                return checks, meta

        raise LLMError(f"LLM не вернула корректный ответ за {attempts} попыт.: {last_error}")
