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

# Заголовки шлюза, полезные для разбора проблем (Bifrost и др.)
_TRACE_HEADERS = ("x-request-id", "x-bifrost-trace-id", "x-bifrost-resolved-model", "x-bifrost-routing-info-key")


class _Reply:
    """Результат одного HTTP-запроса: статус, заголовки, текст ответа модели или тело ошибки."""

    def __init__(self, status: int, headers: httpx.Headers, text: str):
        self.status, self.headers, self.text = status, headers, text

    @property
    def trace(self) -> Dict[str, str]:
        return {h: self.headers[h] for h in _TRACE_HEADERS if h in self.headers}


def _sse_content(lines) -> Tuple[str, Optional[str]]:
    """Собирает текст из строк SSE «data: {...}». Возвращает (текст, ошибка-из-потока)."""
    parts: List[str] = []
    for line in lines:
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            event = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if event.get("error"):
            return "".join(parts), str(event["error"])[:500]
        for choice in event.get("choices") or []:  # служебные события без choices пропускаем
            delta = choice.get("delta") or choice.get("message") or {}
            if delta.get("content"):
                parts.append(delta["content"])
    return "".join(parts), None


class LLMClient:
    """Клиент OpenAI-совместимого chat/completions: шлюз LLM (/v1/...) или Open WebUI (/api/...)."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._format: Optional[str] = settings.response_format  # может откатиться при 400

    def _headers(self, stream: bool = False) -> Dict[str, str]:
        h = {"Content-Type": "application/json",
             "Accept": "text/event-stream" if stream else "application/json"}
        if self.s.api_key:
            h["Authorization"] = f"Bearer {self.s.api_key}"
        return h

    def _payload(self, messages: List[Dict[str, Any]], fmt: Optional[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.s.model,
            "stream": self.s.stream,
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
    def _json_content(resp: httpx.Response) -> str:
        if "text/event-stream" in resp.headers.get("content-type", ""):
            text, err = _sse_content(resp.text.splitlines())
            if err:
                raise ResponseValidationError(f"ошибка в потоке ответа: {err}")
            return text
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ResponseValidationError(f"неожиданная структура ответа: {str(data)[:300]}")
        if isinstance(content, list):  # контент частями
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content or ""

    async def _send(self, client: httpx.AsyncClient, payload: Dict[str, Any]) -> _Reply:
        """Один запрос. В потоковом режиме текст собирается по мере прихода событий:
        шлюз без потока рвёт долгие соединения (~30 с)."""
        if not payload.get("stream"):
            resp = await client.post(self.s.chat_url, headers=self._headers(), json=payload)
            if resp.status_code >= 400:
                return _Reply(resp.status_code, resp.headers, resp.text)
            return _Reply(resp.status_code, resp.headers, self._json_content(resp))

        async with client.stream("POST", self.s.chat_url, headers=self._headers(stream=True), json=payload) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "replace")
                return _Reply(resp.status_code, resp.headers, body)
            if "text/event-stream" not in resp.headers.get("content-type", ""):
                await resp.aread()  # сервер ответил без потока — разбираем обычный JSON
                return _Reply(resp.status_code, resp.headers, self._json_content(resp))
            lines = [line async for line in resp.aiter_lines()]
        text, err = _sse_content(lines)
        if err:
            return _Reply(502, resp.headers, f"ошибка в потоке ответа: {err}")
        return _Reply(resp.status_code, resp.headers, text)

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
        connect_failed = False
        trace: Dict[str, str] = {}
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=self.s.timeout, verify=self.s.verify_ssl) as client:
            while attempts <= self.s.max_retries:
                attempts += 1
                fmt = self._format
                try:
                    reply = await self._send(client, self._payload(messages, fmt))
                except ResponseValidationError as e:  # сервер ответил без потока, но JSON не тот
                    reply = _Reply(200, httpx.Headers(), "")
                    last_error = str(e)
                except httpx.ConnectError as e:
                    connect_failed = True
                    last_error = f"не удалось подключиться к {self.s.chat_url}: {e!r}"
                    log.warning("попытка %d: %s", attempts, last_error)
                    await asyncio.sleep(min(2 ** attempts, 10))
                    continue
                except httpx.HTTPError as e:
                    last_error = f"сетевая ошибка при запросе к {self.s.chat_url}: {type(e).__name__}: {e}"
                    log.warning("попытка %d: %s", attempts, last_error)
                    await asyncio.sleep(min(2 ** attempts, 10))
                    continue
                trace = reply.trace or trace

                # Модель/шлюз не поддерживает response_format — откатываемся и пробуем снова
                # (если 400 по другой причине — дойдём до "none" и получим исходную ошибку)
                if reply.status in (400, 422) and fmt in ("json_schema", "json_object"):
                    nxt = _FORMAT_FALLBACK[fmt]
                    log.warning("response_format=%s отклонён (HTTP %s: %s), пробую %s",
                                fmt, reply.status, reply.text[:200], nxt)
                    self._format = nxt
                    attempts -= 1  # смена формата не считается попыткой
                    continue

                if reply.status in (401, 403):
                    raise LLMError(f"доступ запрещён (HTTP {reply.status}): проверьте LLM_API_KEY. {reply.text[:200]}")
                if reply.status == 404:
                    raise LLMError(f"HTTP 404 на {self.s.chat_url}: проверьте LLM_BASE_URL, LLM_CHAT_PATH "
                                   f"и LLM_MODEL ({reply.text[:200]})")
                if reply.status == 429 or reply.status >= 500:
                    last_error = f"HTTP {reply.status}: {reply.text[:300]}"
                    retry_after = reply.headers.get("retry-after", "")
                    delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else min(2 ** attempts, 10)
                    log.warning("попытка %d: %s, повтор через %.1fс %s", attempts, last_error, delay, reply.trace)
                    await asyncio.sleep(delay)
                    continue
                if reply.status >= 400:
                    raise LLMError(f"HTTP {reply.status}: {reply.text[:500]} {reply.trace or ''}".strip())

                content = reply.text
                try:
                    if not content:
                        raise ResponseValidationError(last_error if last_error != "нет ответа" else "пустой ответ")
                    checks = parse_checks(content)
                except ResponseValidationError as e:
                    last_error = f"ответ не по схеме: {e}"
                    log.warning("попытка %d: %s %s", attempts, last_error, reply.trace)
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
                    "stream": self.s.stream,
                    "llm_latency_ms": int((time.monotonic() - started) * 1000),
                    **({"gateway": reply.trace} if reply.trace else {}),
                }
                return checks, meta

        if connect_failed and last_error.startswith("не удалось подключиться"):
            last_error += (" — адрес недоступен из контейнера. Проверьте LLM_BASE_URL; для внутреннего шлюза "
                           "контейнеру нужен доступ во внутреннюю сеть (VPN, DNS, прокси HTTPS_PROXY).")
        raise LLMError(f"LLM не вернула корректный ответ за {attempts} попыт.: {last_error} {trace or ''}".strip())


# Старое имя — для совместимости
OpenWebUIClient = LLMClient
