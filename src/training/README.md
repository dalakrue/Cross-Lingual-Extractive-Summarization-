# src/training/ — Classifier training + model comparison

## Files

| File | What it does |
|---|---|
| `train_classifier.py` | Trains Logistic Regression, SVM, Random Forest on the 64 FastText features. |
| `train_improved_classifier.py` | Trains the same three models on the 69 improved features. |

## How to run

```bash
# Baseline — needs results/sentence_vectors.csv from src/fasttext/
python src/training/train_classifier.py
# Writes: results/model_comparison.csv, results/{logistic_regression,svm,random_forest}_model.pkl, results/scaler.pkl

# Improved — needs results/improved_sentence_vectors.csv
python src/training/train_improved_classifier.py
# Writes: results/improved_model_comparison.csv, results/improved_{svm,logistic_regression,random_forest}_model.pkl, results/improved_scaler.pkl
```

## What it measures

For each model: accuracy, precision, recall, F1, classification report,
confusion matrix. Models are ranked by **F1**, not accuracy, because most
sentences are label 0 — a model that predicts 0 for everything would get
high accuracy and be useless.

## Results (committed in `results/`)

| Pipeline | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baseline (64) | SVM | 74.3% | 0.28 | 0.58 | 0.37 |
| Baseline (64) | Logistic Regression | 72.7% | 0.26 | 0.58 | 0.36 |
| Baseline (64) | Random Forest | 87.3% | 0.75 | 0.08 | 0.14 |
| **Improved (69)** | **SVM** | **83.7%** | **0.44** | **0.90** | **0.60** |
| Improved (69) | Logistic Regression | 84.3% | 0.45 | 0.73 | 0.55 |
| Improved (69) | Random Forest | 89.7% | 0.66 | 0.48 | 0.55 |

The app uses the **improved SVM**: best F1 and best recall (finds 90% of
important sentences). All models use `class_weight="balanced"`.
Random Forest is trained on unscaled features; Logistic Regression and
SVM use `StandardScaler` fitted on train only.
