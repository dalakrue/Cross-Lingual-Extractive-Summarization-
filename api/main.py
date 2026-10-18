"""FastAPI backend for the Briefly frontend.

Endpoints expected by frontend/src/App.jsx:
    POST /api/clean     { article }                     -> cleaned text + counts
    POST /api/summarize { article, sentence_count }    -> summary + counts
    POST /api/translate { text }                       -> Myanmar translation
    GET  /api/health                                    -> status

Run from the repo root:
    uvicorn api.main:app --reload

Models must be trained first (see README.md):
    python src/fasttext/improved_features.py
    python src/training/train_improved_classifier.py

Translation uses MyMemory (free third-party service), the same service
the frontend notice describes. Only the generated summary is sent.
"""

import sys
import urllib.parse
import urllib.request
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Repo root on sys.path so `src.*` imports work when run as `uvicorn api.main:app`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = FastAPI(title="Briefly — Cross-Lingual Extractive Summarization API")


class CleanRequest(BaseModel):
    article: str = Field(..., max_length=1_000_000)


class SummarizeRequest(BaseModel):
    article: str = Field(..., max_length=1_000_000)
    sentence_count: int = Field(5, ge=1, le=20)


class TranslateRequest(BaseModel):
    text: str = Field(..., max_length=1_000_000)


def _load_summarizer():
    """Import the improved summarizer. Raises 503 if models are not trained."""
    try:
        from src.summarization import summarize_improved as sim
        return sim
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Trained models not found in results/. "
                "Run: python src/fasttext/improved_features.py, then "
                "python src/training/train_improved_classifier.py"
            ),
        ) from exc
    except Exception as exc:  # e.g. missing dependency
        raise HTTPException(
            status_code=503, detail=f"Summarizer could not be loaded: {exc}"
        ) from exc


@app.get("/api/health")
def health():
    models_exist = (
        (ROOT / "results" / "improved_fasttext.model").exists()
        and (ROOT / "results" / "improved_svm_model.pkl").exists()
        and (ROOT / "results" / "improved_scaler.pkl").exists()
    )
    return {"status": "ok", "models_ready": models_exist}


@app.post("/api/clean")
def clean(req: CleanRequest):
    sim = _load_summarizer()
    if not req.article.strip():
        raise HTTPException(status_code=400, detail="Article is empty.")
    cleaned = sim.clean_article_text(req.article)
    sentences = sim.split_sentences(cleaned)
    paragraphs = [p for p in cleaned.split("\n\n") if p.strip()]
    return {
        "cleaned_text": cleaned,
        "paragraph_count": len(paragraphs),
        "sentence_count": len(sentences),
        "original_character_count": len(req.article),
        "cleaned_character_count": len(cleaned),
    }


@app.post("/api/summarize")
def summarize(req: SummarizeRequest):
    sim = _load_summarizer()
    if not req.article.strip():
        raise HTTPException(status_code=400, detail="Article is empty.")
    summary = sim.summarize_article(req.article, top_k=req.sentence_count)
    if not summary:
        raise HTTPException(status_code=400, detail="No sentences found in article.")
    analyzed = sim.split_sentences(sim.clean_article_text(req.article))
    selected = sim.split_sentences(summary)
    return {
        "summary": summary,
        "sentence_count": len(selected),
        "analyzed_sentence_count": len(analyzed),
    }


@app.post("/api/translate")
def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")
    params = urllib.parse.urlencode(
        {"q": req.text, "langpair": "en|my"}
    )
    url = f"https://api.mymemory.translated.net/get?{params}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Translation service unavailable: {exc}",
        ) from exc
    translated = (payload.get("responseData") or {}).get("translatedText")
    if not translated:
        raise HTTPException(status_code=502, detail="Translation service returned no text.")
    return {"translated_text": translated}
