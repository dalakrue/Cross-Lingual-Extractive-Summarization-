# Cross-Lingual Domain-Specific Extractive Text Summarization

English article → extractive summary → Myanmar summary.
Sports + Technology domains. FastText + Machine Learning.

MIIT CSE group project — 2021-MIIT-CSE-077 Thet Oo Maung (dataset & preprocessing),
2021-MIIT-CSE-076 Thet Hume San (FastText & feature extraction),
2021-MIIT-CSE-068 Swam Linn Htun (ML model & summarization),
2021-MIIT-CSE-066 Soe Pyae Pyae (cross-language, evaluation & system).

## How it works

1. **Dataset** (`data/`) — 50 articles, 1500 sentences, sentence-level 1/0 labels
   (1 = belongs in the summary).
2. **FastText** (`src/fasttext/`) — each sentence becomes a 64-dim vector
   (skip-gram, char n-grams 3–6). The improved pipeline adds 5 sentence features.
3. **ML model** (`src/training/`) — Logistic Regression / SVM / Random Forest
   predict P(sentence is important). SVM wins on F1, so it is used for summaries.
4. **Summary** (`src/summarization/`) — pick the top-K scored sentences,
   keep them in original article order (improved version picks the best
   sentence from each section for coverage).
5. **Cross-language** (`api/`) — the English summary is translated to Myanmar
   via MyMemory. Only the generated summary is sent to the translation service.

## Real results (from the committed `results/` tables)

| Pipeline | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| FastText only (64 features) | SVM | 74.3% | 0.28 | 0.58 | 0.37 |
| FastText only | Logistic Regression | 72.7% | 0.26 | 0.58 | 0.36 |
| FastText only | Random Forest | 87.3% | 0.75 | 0.08 | 0.14 |
| **Improved (69 features)** | **SVM** | **83.7%** | **0.44** | **0.90** | **0.60** |
| Improved | Logistic Regression | 84.3% | 0.45 | 0.73 | 0.55 |
| Improved | Random Forest | 89.7% | 0.66 | 0.48 | 0.55 |

Why SVM, not the 89.7% Random Forest? Accuracy alone lies on imbalanced
labels (most sentences are 0). Random Forest barely finds important sentences
(recall 0.08–0.48). SVM with the improved features finds 90% of them
(recall 0.90) with the best F1 (0.60). That is why the app uses the improved SVM.

Summary selection (improved, 10 test articles, avg F1): top-5 ≈ 0.57,
top-7 ≈ 0.54, top-10 ≈ 0.48. ROUGE-1/2/L is planned, not yet computed —
no ROUGE scores are claimed here.

## Run it end-to-end

```bash
pip install -r requirements.txt

# 1. FastText features (creates results/article_split.csv, results/sentence_vectors.csv, results/fasttext.model)
python src/fasttext/fasttext_features.py

# 2. Train baseline classifiers
python src/training/train_classifier.py

# 3. Improved features (64 FastText + 5 sentence features)
python src/fasttext/improved_features.py

# 4. Train improved classifiers (saves improved SVM + scaler)
python src/training/train_improved_classifier.py

# 5. Evaluate summary selection
python src/evaluation/evaluate_summarizer.py
python src/evaluation/evaluate_improved.py

# 6. API + frontend
uvicorn api.main:app --reload
cd frontend && npm install && npm run dev
```

Open the frontend, paste an English article, choose 5 / 7 / 10 sentences,
summarize, then toggle English / မြန်မာ.

Quick test without the API:

```bash
python src/summarization/summarize_improved.py
```

## Project structure

| Folder | What it is | README |
|---|---|---|
| `data/` | Labeled dataset (1500 sentences) | [data/README.md](data/README.md) |
| `src/fasttext/` | FastText training + feature extraction | [src/fasttext/README.md](src/fasttext/README.md) |
| `src/training/` | Classifier training + model comparison | [src/training/README.md](src/training/README.md) |
| `src/summarization/` | Summarize any new article with a trained model | [src/summarization/README.md](src/summarization/README.md) |
| `src/evaluation/` | Summary-selection evaluation (precision/recall/F1) | [src/evaluation/README.md](src/evaluation/README.md) |
| `api/` | FastAPI backend (`/api/clean`, `/api/summarize`, `/api/translate`) | [api/README.md](api/README.md) |
| `frontend/` | React + Vite UI ("Briefly") | [frontend/README.md](frontend/README.md) |
| `notebooks/` | Original Colab notebook, cleaned (relative paths) | [notebooks/README.md](notebooks/README.md) |
| `results/` | Committed result tables (model comparison + selection scores) | [results/README.md](results/README.md) |

Each folder has its own README explaining what is inside, how to run it,
and what it produces.

## Notes on this repo

- Trained binaries (`*.pkl`, `*.model`) and generated vector CSVs are
  **not committed** (see `.gitignore`). Retrain with the commands above —
  it takes a few minutes on a laptop.
- The original shared zip had everything flat in one folder, no `api/`
  folder, and paths pointing at `data/` / `results/` / `summarization/`
  folders that did not exist, so it could not run as-is. This repo fixes
  the layout and paths, adds the missing `api/main.py` the frontend calls,
  and fixes `summarize.py` tokenization to lowercase like its training code.
  No model, result number, or dataset row was changed.
- The original Colab notebook had a `whafrom` typo in its first cell and
  hard-coded Google Drive paths. The copy in `notebooks/` fixes the typo
  and uses relative paths.
