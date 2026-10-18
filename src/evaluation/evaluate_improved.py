# %%

import os
import sys
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "sentence_labels.csv"
)

SPLIT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "article_split.csv"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "results",
    "improved_summarization_selection_results.csv"
)

# ------------------------------------------------------------
# Import improved summarizer
# ------------------------------------------------------------

SUMMARIZATION_DIR = os.path.join(
    BASE_DIR,
    "src",
    "summarization"
)

sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, SUMMARIZATION_DIR)

from summarization.summarize_improved import summarize_article


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
split_df = pd.read_csv(SPLIT_PATH)

print("=" * 70)
print("IMPROVED SUMMARIZATION EVALUATION")
print("=" * 70)

print("\nDataset shape:", df.shape)

# ------------------------------------------------------------
# GET TEST ARTICLES
# ------------------------------------------------------------

test_articles = split_df[
    split_df["split"] == "test"
]["article_id"].unique()

test_df = df[
    df["article_id"].isin(test_articles)
].copy()

print("Test articles:", len(test_articles))
print("Test sentences:", len(test_df))

print("\nTest articles:")
for article_id in test_articles:
    print(" -", article_id)


# ------------------------------------------------------------
# EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate_article(article_df, top_k):
    """
    Evaluate sentence selection for one article.

    label = 1 means the sentence is considered
    an important/reference sentence.

    The summarizer selects top_k sentences.
    """

    article_df = article_df.sort_values(
        "sentence_id"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Reconstruct original article
    # --------------------------------------------------------

    article_text = " ".join(
        article_df["sentence"].astype(str)
    )

    # --------------------------------------------------------
    # Generate summary
    # --------------------------------------------------------

    summary = summarize_article(
        article_text,
        top_k=top_k
    )

    # --------------------------------------------------------
    # Determine which original sentences were selected
    # --------------------------------------------------------

    selected_indices = []

    for i, sentence in enumerate(
        article_df["sentence"]
    ):

        if sentence in summary:
            selected_indices.append(i)

    # --------------------------------------------------------
    # Convert selected sentences to predictions
    # --------------------------------------------------------

    y_true = article_df["label"].astype(int).values

    y_pred = np.zeros(
        len(article_df),
        dtype=int
    )

    for index in selected_indices:
        y_pred[index] = 1

    # --------------------------------------------------------
    # Calculate TP / FP / FN
    # --------------------------------------------------------

    tp = np.sum(
        (y_true == 1) & (y_pred == 1)
    )

    fp = np.sum(
        (y_true == 0) & (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1) & (y_pred == 0)
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if precision + recall > 0:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "selected_count": len(selected_indices)
    }


# ------------------------------------------------------------
# RUN TOP 5 / TOP 7 / TOP 10
# ------------------------------------------------------------

all_results = []

for top_k in [5, 7, 10]:

    print("\n" + "=" * 70)
    print(f"TOP {top_k}")
    print("=" * 70)

    article_results = []

    for article_id in test_articles:

        article_df = test_df[
            test_df["article_id"] == article_id
        ].copy()

        result = evaluate_article(
            article_df,
            top_k
        )

        result["article_id"] = article_id
        result["top_k"] = top_k

        article_results.append(result)
        all_results.append(result)

        print(
            f"{article_id:20s} "
            f"Precision={result['precision']:.4f}  "
            f"Recall={result['recall']:.4f}  "
            f"F1={result['f1']:.4f}"
        )

    # --------------------------------------------------------
    # Average across test articles
    # --------------------------------------------------------

    article_results_df = pd.DataFrame(
        article_results
    )

    avg_precision = article_results_df[
        "precision"
    ].mean()

    avg_recall = article_results_df[
        "recall"
    ].mean()

    avg_f1 = article_results_df[
        "f1"
    ].mean()

    print("\nAVERAGE")
    print(
        f"Precision = {avg_precision:.4f}"
    )

    print(
        f"Recall    = {avg_recall:.4f}"
    )

    print(
        f"F1        = {avg_f1:.4f}"
    )


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

# ------------------------------------------------------------
# OVERALL RESULTS
# ------------------------------------------------------------

overall = (
    results_df
    .groupby("top_k")[
        ["precision", "recall", "f1"]
    ]
    .mean()
    .reset_index()
)

print("\n")
print("=" * 70)
print("OVERALL IMPROVED SUMMARIZATION RESULTS")
print("=" * 70)

print(
    overall.to_string(
        index=False
    )
)

print("\nResults saved to:")
print(RESULTS_PATH)