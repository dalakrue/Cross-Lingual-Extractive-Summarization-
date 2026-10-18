from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_FILE = DATA_DIR / "sentence_labels.csv"
SPLIT_FILE = RESULTS_DIR / "article_split.csv"


# ============================================================
# IMPORT SUMMARIZER
# ============================================================

SUMMARIZATION_DIR = BASE_DIR / "src" / "summarization"

sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(SUMMARIZATION_DIR))

from summarization.summarize import summarize_article


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
split_df = pd.read_csv(SPLIT_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# GET TEST ARTICLES
# ============================================================

test_articles = split_df[
    split_df["split"] == "test"
]["article_id"].tolist()

print(f"\nNumber of test articles: {len(test_articles)}")

print("\nTest articles:")
for article_id in test_articles:
    print(f" - {article_id}")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_top_k(top_k):

    all_results = []

    print("\n")
    print("=" * 60)
    print(f"EVALUATING TOP {top_k}")
    print("=" * 60)

    for article_id in test_articles:

        article_df = df[
            df["article_id"] == article_id
        ].sort_values("sentence_id")

        sentences = article_df["sentence"].tolist()

        true_labels = article_df["label"].tolist()

        article_text = " ".join(sentences)

        # ----------------------------------------------------
        # Generate summary
        # ----------------------------------------------------

        result = summarize_article(
            article_text,
            top_k=top_k
        )

        selected_sentences = result["selected_sentences"]

        # ----------------------------------------------------
        # Convert selected sentences into predictions
        # ----------------------------------------------------

        predicted_labels = []

        for sentence in sentences:

            if sentence in selected_sentences:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        precision = precision_score(
            true_labels,
            predicted_labels,
            zero_division=0
        )

        recall = recall_score(
            true_labels,
            predicted_labels,
            zero_division=0
        )

        f1 = f1_score(
            true_labels,
            predicted_labels,
            zero_division=0
        )

        all_results.append({
            "article_id": article_id,
            "top_k": top_k,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        print(
            f"{article_id}: "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}"
        )

    return pd.DataFrame(all_results)


# ============================================================
# RUN EVALUATION
# ============================================================

results = []

for top_k in [5, 7, 10]:

    result_df = evaluate_top_k(top_k)

    results.append(result_df)


# Combine results

results_df = pd.concat(
    results,
    ignore_index=True
)


# ============================================================
# OVERALL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("OVERALL SUMMARIZATION SELECTION RESULTS")
print("=" * 60)

summary_results = (
    results_df
    .groupby("top_k")
    [["precision", "recall", "f1"]]
    .mean()
    .reset_index()
)

print(summary_results.to_string(index=False))


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = (
    RESULTS_DIR
    / "summarization_selection_results.csv"
)

summary_results.to_csv(
    output_file,
    index=False
)

print("\nResults saved to:")
print(output_file)

print("\n")
print("=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)