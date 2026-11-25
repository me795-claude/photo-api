# Photo-API — проверка фото на пропуск через LLM

Сервис принимает фотографию, отправляет её в LLM через API Open WebUI и возвращает JSON с результатом по каждому из 9 требований документа «Требование к фото на пропуск». Архитектура описана в [Схема проверки фото через LLM.md](Схема%20проверки%20фото%20через%20LLM.md).

```json
{
  "checks": [
    { "test_type": "format",  "result": "successed", "comment": "" },
    { "test_type": "glasses", "result": "failed",    "comment": "Тонированные стёкла — снимите очки или замените на прозрачные" }
  ],
  "overall": "failed",
  "meta": { "...": "служебная информация: модель, время, ответ LLM и метрики проверок кодом" }
}
```

`test_type`: `format`, `quality`, `lighting_color`, `background`, `headwear`, `head_position`, `gaze_expression`, `face_eyes`, `glasses`.

## Быстрый старт

```bash
cp .env.example .env
```

В `.env` укажите как минимум `OPENWEBUI_URL`, `OPENWEBUI_API_KEY` и `LLM_MODEL`, затем:

```bash
docker compose up -d --build
```

- Веб-страница загрузки: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Связь с Open WebUI: http://localhost:8000/health/llm

### API

```bash
curl -F "file=@photo.jpg" http://localhost:8000/api/check
```

`?meta=false` — вернуть только `checks` и `overall`.

### Прогон тестового набора

```bash
docker compose run --rm testset
```

В консоли — какие фото совпали с эталоном, точность по каждому `test_type`; полный отчёт — `results/report.json`.

Другие команды CLI (внутри контейнера):

```bash
docker compose run --rm photo-check python -m app.cli models
```

```bash
docker compose run --rm -v "$PWD/photos:/photos:ro" photo-check python -m app.cli check /photos
```

### Проверка без настоящей LLM

В `.env` поставьте `OPENWEBUI_URL=http://mock-llm:3000`, затем:

```bash
docker compose --profile mock up -d --build
```

Заглушка (`app/mock_llm.py`) проверяет формат запроса и отвечает по схеме. `MOCK_FAIL=glasses,face_eyes` — какие пункты «провалить», `MOCK_REJECT_JSON_SCHEMA=true` — проверить откат на `json_object`.

## Настройки (.env)

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `OPENWEBUI_URL` | `http://host.docker.internal:3000` | адрес Open WebUI без `/api` |
| `OPENWEBUI_API_KEY` | — | ключ `sk-...` (Settings → Account → API Keys) |
| `OPENWEBUI_VERIFY_SSL` | `true` | проверять TLS-сертификат |
| `LLM_MODEL` | `qwen3.8-flash-next` | модель с поддержкой изображений |
| `LLM_RESPONSE_FORMAT` | `json_schema` | `json_schema` / `json_object` / `none`; при отказе модели — автоматический откат |
| `LLM_MAX_RETRIES` | `2` | повторы, если ответ не по схеме или сервер вернул 429/5xx |
| `LLM_TEMPERATURE`, `LLM_SEED`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT` | `0`, `42`, `2000`, `180` | параметры запроса |
| `LLM_EXTRA_PARAMS` | — | доп. поля запроса в JSON, напр. `{"chat_template_kwargs":{"enable_thinking":false}}` |
| `IMAGE_MAX_SIDE` | `1024` | до какого размера уменьшать фото перед отправкой |
| `LOCAL_CHECKS` | `true` | проверки кодом: пропорции, ч/б, пересвет/затемнение, резкость |
| `LOCAL_ASPECT_TOLERANCE` | `0.04` | допустимое отклонение от 35×45 |
| `LOCAL_BLUR_THRESHOLD` | `2` | порог резкости (только явный расфокус; для фото с камеры можно 10–20) |
| `LOCAL_OVEREXPOSED_FRAC`, `LOCAL_UNDEREXPOSED_FRAC` | `0.20`, `0.35` | доля «выбитых» пикселей в зоне лица |
| `APP_PORT` | `8000` | порт сервиса на хосте |

## Как устроена проверка

1. Фото поворачивается по EXIF, уменьшается до `IMAGE_MAX_SIDE` и кодируется в JPEG.
2. **Проверки кодом** (`app/local_checks.py`): пропорции кадра, цветное ли фото, пересвет/затемнение лица, резкость. Они могут только перевести пункт в `failed`.
3. **LLM** (`app/llm.py`): `POST /api/chat/completions`, фото — `image_url` с `data:image/jpeg;base64,...`, строгая JSON Schema ответа. Ответ очищается от `<think>` и markdown, проверяется (все 9 пунктов, без повторов). Если ответ не по схеме — модель переспрашивается с описанием ошибки.
4. Итог: результат LLM + проверки кодом; `overall = successed`, только если все 9 пунктов пройдены.

## Тестовый набор

`testset/images/` — 25 фото из иллюстраций документа (по каждому требованию примеры «не подходит» и «подходит»). Значки ✘/✔ удалены, чтобы не подсказывать модели ответ; разрешение повышено нейросетью EDSR x4 и кадр обрезан до 35×45.
`testset/expected.json` — эталон: какой результат ожидается по проверяемому пункту и итоговый вердикт.

Пересобрать набор (нужны исходный документ `Требование к фото на пропуск.docx` — в репозиторий он не входит, — OpenCV и веса EDSR):

```bash
pip install -r tools/requirements-tools.txt
```

```bash
curl -L -o EDSR_x4.pb https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x4.pb
```

```bash
python tools/build_testset.py --docx "Требование к фото на пропуск.docx" --model EDSR_x4.pb --out testset
```

## Перед запуском с настоящей LLM

- У модели в Open WebUI включена **Vision** (Admin → Settings → Models → модель → Capabilities). Иначе картинка не дойдёт до модели.
- API-ключи разрешены администратором (Admin → Settings → General → Enable API Keys).
- Если Open WebUI не на этой машине — укажите его адрес в `OPENWEBUI_URL`.
- `pplx-embed-v1-4b` и `nemotron-rerank-1b-v` — текстовые модели, в проверке фото не используются.
