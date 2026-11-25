"""HTTP-сервис проверки фото: веб-страница загрузки + REST API."""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

from .config import load_settings
from .llm import LLMError
from .prompt import TEST_TYPES
from .service import InvalidImageError, PhotoChecker

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

settings = load_settings()
logging.getLogger("photo_check").info("LLM: %s (поток: %s), модель: %s, ключ: %s", settings.chat_url,
                                      "да" if settings.stream else "нет", settings.model,
                                      "задан" if settings.api_key else "НЕ задан")
checker = PhotoChecker(settings)
app = FastAPI(title="Photo-API: проверка фото на пропуск", version="1.0")


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.model, "llm_url": settings.chat_url, "stream": settings.stream}


@app.get("/health/llm")
async def health_llm():
    """Проверка связи с LLM (список моделей) и наличия нужной модели."""
    try:
        models = await checker.llm.list_models()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"LLM недоступна по адресу {settings.models_url}: {e!r}")
    ids = [m.get("id") for m in models if isinstance(m, dict)]
    return {"status": "ok" if settings.model in ids else "model_not_found", "model": settings.model, "available": ids}


@app.get("/api/test-types")
async def test_types():
    return [{"test_type": k, "title": v} for k, v in TEST_TYPES.items()]


@app.post("/api/check")
async def check(file: UploadFile = File(...), meta: bool = Query(True, description="добавить служебный блок meta")):
    data = await file.read()
    if not data:
        raise HTTPException(400, "пустой файл")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"файл больше {settings.max_upload_mb} МБ")
    try:
        result = await checker.check(data, file.filename)
    except InvalidImageError as e:
        raise HTTPException(400, str(e))
    except LLMError as e:
        raise HTTPException(502, f"ошибка LLM: {e}")
    if not meta:
        result.pop("meta", None)
    return result


INDEX_HTML = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Проверка фото на пропуск</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:32px auto;padding:0 16px;color:#1d1d1f}
 h1{font-size:22px} .row{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}
 #preview{max-width:280px;max-height:360px;border:1px solid #ddd;border-radius:6px;display:none}
 table{border-collapse:collapse;width:100%;min-width:420px;flex:1} td,th{border-bottom:1px solid #eee;padding:8px;text-align:left;vertical-align:top;font-size:14px}
 .ok{color:#1a7f37;font-weight:600}.bad{color:#cf222e;font-weight:600}
 .verdict{font-size:18px;margin:12px 0} button{padding:8px 16px;font-size:14px;cursor:pointer}
 pre{background:#f6f8fa;padding:12px;border-radius:6px;overflow:auto;font-size:12px;max-height:400px}
 .muted{color:#6e7781;font-size:13px}
</style></head><body>
<h1>Проверка фото на пропуск</h1>
<p class="muted">Модель: <b>__MODEL__</b>. Фото отправляется в LLM, ответ — JSON по 9 пунктам требований.</p>
<form id="f"><input type="file" id="file" accept="image/*" required> <button>Проверить</button> <span id="st" class="muted"></span></form>
<div class="verdict" id="verdict"></div>
<div class="row"><img id="preview"><table id="tbl"></table></div>
<details><summary class="muted">JSON-ответ</summary><pre id="json"></pre></details>
<script>
const $=id=>document.getElementById(id);
$('file').onchange=e=>{const f=e.target.files[0]; if(f){$('preview').src=URL.createObjectURL(f);$('preview').style.display='block'}};
$('f').onsubmit=async e=>{
  e.preventDefault(); const f=$('file').files[0]; if(!f) return;
  $('st').textContent='Проверяю…'; $('tbl').innerHTML=''; $('verdict').textContent=''; $('json').textContent='';
  const fd=new FormData(); fd.append('file',f); const t0=performance.now();
  try{
    const r=await fetch('api/check',{method:'POST',body:fd}); const j=await r.json();
    $('json').textContent=JSON.stringify(j,null,2);
    if(!r.ok){$('st').textContent='Ошибка: '+(j.detail||r.status);return}
    $('st').textContent=`Готово за ${((performance.now()-t0)/1000).toFixed(1)} с`;
    $('verdict').innerHTML=j.overall==='successed'?'<span class="ok">✔ Фото подходит</span>':'<span class="bad">✘ Фото не подходит</span>';
    const titles=await (await fetch('api/test-types')).json(); const tmap=Object.fromEntries(titles.map(x=>[x.test_type,x.title]));
    $('tbl').innerHTML='<tr><th>Проверка</th><th>Результат</th><th>Комментарий</th></tr>'+j.checks.map(c=>
      `<tr><td>${tmap[c.test_type]||c.test_type}<br><span class="muted">${c.test_type}</span></td>
       <td class="${c.result==='successed'?'ok':'bad'}">${c.result}</td><td>${(c.comment||'').replace(/</g,'&lt;')}</td></tr>`).join('');
  }catch(err){$('st').textContent='Ошибка: '+err}
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML.replace("__MODEL__", settings.model)
