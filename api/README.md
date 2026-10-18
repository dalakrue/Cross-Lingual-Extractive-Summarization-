# api/ — FastAPI backend for the frontend

## Files

| File | What it is |
|---|---|
| `main.py` | FastAPI app. This file did not exist in the original shared zip — the frontend called it, but there was nothing to answer. It is added here so the demo actually runs. |

## Endpoints (all under `/api`, as the frontend expects)

| Method + path | Request | Response |
|---|---|---|
| `GET /api/health` | — | `{ status, models_ready }` |
| `POST /api/clean` | `{ article }` | `{ cleaned_text, paragraph_count, sentence_count, original_character_count, cleaned_character_count }` |
| `POST /api/summarize` | `{ article, sentence_count }` | `{ summary, sentence_count, analyzed_sentence_count }` |
| `POST /api/translate` | `{ text }` | `{ translated_text }` (English → Myanmar) |

## How to run

```bash
# From the repo root. Train first so results/ has the improved models.
uvicorn api.main:app --reload
# API on http://127.0.0.1:8000, frontend dev server proxies /api to it.
```

## Translation

`POST /api/translate` calls MyMemory (`api.mymemory.translated.net`,
English → Myanmar). Only the generated summary is sent, never the full
article. MyMemory is a free third-party service — availability and
translation quality may vary, which is what the frontend notice says.

## Errors

- `503` from `/api/clean` / `/api/summarize` means the trained models are
  missing from `results/` — run the training pipeline (root README).
- `502` from `/api/translate` means the translation service is unreachable.
