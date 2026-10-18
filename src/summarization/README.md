# src/summarization/ — Summarize any new article

## Files

| File | What it does |
|---|---|
| `summarize.py` | Baseline summarizer: 64 FastText features + SVM. |
| `summarize_improved.py` | **Used by the app.** 69 features (64 FastText + 5 sentence features) + improved SVM. Section-based selection for coverage. |

## How to run

```bash
# Quick demo (needs results/improved_*.model/.pkl from training)
python src/summarization/summarize_improved.py

# Baseline demo (needs results/fasttext.model, results/svm_model.pkl, results/scaler.pkl)
python src/summarization/summarize.py
```

## How to use in your own code

```python
from src.summarization.summarize_improved import summarize_article

summary = summarize_article(article_text, top_k=5)
print(summary)
```

## How it works

1. Split the article into sentences.
2. Convert each sentence to its feature vector (FastText mean + 5 sentence features).
3. Scale with the same scaler used in training.
4. SVM `predict_proba` gives P(sentence is important) for each sentence.
5. **Improved:** divide the article into `top_k` consecutive sections and
   pick the highest-scoring sentence from each section — this keeps the
   summary covering the whole article instead of clustering at the start.
   **Baseline:** pick the global top-K, then restore original order.
6. Join the selected sentences in original article order.

## Notes

- `summarize.py` tokenization lowercases, matching its training code
  (`fasttext_features.py`). Do not remove the lowercasing.
- `summarize_improved.py` also exposes `clean_article_text()` and
  `split_sentences()` — the API reuses them so the frontend preview and
  the summarizer clean text the same way.
- Models are loaded at import time. If `results/` has no trained models
  yet, run the training pipeline first (see root README).
