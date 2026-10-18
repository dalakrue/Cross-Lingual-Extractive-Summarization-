# src/ — Python pipeline

| Folder | What it does |
|---|---|
| `fasttext/` | Train FastText, build sentence vectors (+ 5 sentence features in the improved version). |
| `training/` | Train Logistic Regression / SVM / Random Forest, compare by F1. |
| `summarization/` | Summarize any new article with a trained model. |
| `evaluation/` | Measure summary selection (precision / recall / F1) on the 10 test articles. |

Run order (from repo root): `fasttext` → `training` → `evaluation`.
`summarization` and `api/` use the trained models in `results/`.
Each subfolder has its own README with commands and outputs.
