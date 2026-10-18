# src/evaluation/ — Summary-selection evaluation

## Files

| File | What it does |
|---|---|
| `evaluate_summarizer.py` | Evaluates the baseline summarizer on the 10 test articles. |
| `evaluate_improved.py` | Evaluates the improved summarizer on the 10 test articles. |

## How to run

```bash
# Needs trained models in results/ (see root README)
python src/evaluation/evaluate_summarizer.py
# Writes: results/summarization_selection_results.csv (avg per top_k)

python src/evaluation/evaluate_improved.py
# Writes: results/improved_summarization_selection_results.csv (per-article + per top_k)
```

## What it measures

For each test article and each `top_k` in {5, 7, 10}:

- Rebuild the article from `data/sentence_labels.csv`.
- Generate a summary with `summarize_article(article, top_k)`.
- Mark a sentence as predicted-important if it appears in the summary.
- Compare against the true 1/0 labels: precision, recall, F1.

## Committed results

Baseline (avg F1): top-5 ≈ 0.33, top-7 ≈ 0.44, top-10 ≈ 0.38.
Improved (avg F1): top-5 ≈ 0.57, top-7 ≈ 0.54, top-10 ≈ 0.48
(see `results/improved_summarization_selection_results.csv` for per-article scores).

ROUGE-1/2/L is planned future work — it is **not** computed in this repo,
and no ROUGE scores are claimed.
