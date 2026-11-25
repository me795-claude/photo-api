"""Заглушка Open WebUI для проверки сервиса без настоящей LLM.

Эмулирует GET /api/models и POST /api/chat/completions. Проверяет, что запрос
собран правильно (модель, картинка в base64), и возвращает ответ по схеме.
Ответ намеренно «грязный» — с блоком <think> и markdown, — чтобы проверить разбор.

  MOCK_REJECT_JSON_SCHEMA=true — отвечать 400 на response_format=json_schema (проверка отката)
  MOCK_FAIL=glasses,face_eyes  — какие пункты вернуть failed
"""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .prompt import FAILED, SUCCESS, TEST_TYPES

app = FastAPI(title="Mock Open WebUI")
MODEL = os.getenv("LLM_MODEL", "qwen3.8-flash-next")


@app.get("/api/models")
async def models():
    return {"data": [{"id": MODEL, "name": MODEL}, {"id": "pplx-embed-v1-4b"}, {"id": "nemotron-rerank-1b-v"}]}


@app.post("/api/chat/completions")
async def chat(request: Request):
    if not request.headers.get("authorization", "").startswith("Bearer "):
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    body = await request.json()
    if body.get("model") != MODEL:
        return JSONResponse({"detail": "Model not found"}, status_code=404)
    fmt = (body.get("response_format") or {}).get("type")
    if fmt == "json_schema" and os.getenv("MOCK_REJECT_JSON_SCHEMA", "").lower() == "true":
        return JSONResponse({"detail": "response_format json_schema is not supported"}, status_code=400)

    user = next(m for m in body["messages"] if m["role"] == "user")
    images = [p for p in user["content"] if isinstance(p, dict) and p.get("type") == "image_url"]
    if not images or not images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"):
        return JSONResponse({"detail": "image_url (data:image/jpeg;base64,...) not found"}, status_code=400)

    fail = {t.strip() for t in os.getenv("MOCK_FAIL", "").split(",") if t.strip()}
    checks = [
        {"test_type": t, "result": FAILED if t in fail else SUCCESS,
         "comment": f"[mock] не пройдено: {title}" if t in fail else ""}
        for t, title in TEST_TYPES.items()
    ]
    content = "<think>mock reasoning</think>\n```json\n" + json.dumps({"checks": checks}, ensure_ascii=False) + "\n```"
    return {
        "id": "mock-1", "object": "chat.completion", "model": MODEL,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
    }
