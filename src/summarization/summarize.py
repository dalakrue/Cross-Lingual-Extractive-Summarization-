from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from gensim.models import FastText


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = BASE_DIR / "results"

FASTTEXT_MODEL_PATH = RESULTS_DIR / "fasttext.model"
SVM_MODEL_PATH = RESULTS_DIR / "svm_model.pkl"
SCALER_PATH = RESULTS_DIR / "scaler.pkl"


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

print("=" * 60)
print("Loading trained models...")
print("=" * 60)

fasttext_model = FastText.load(str(FASTTEXT_MODEL_PATH))
svm_model = joblib.load(SVM_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("FastText model loaded.")
print("SVM model loaded.")
print("Scaler loaded.")


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):
    """
    Split an English article into individual sentences.
    """

    if not isinstance(text, str):
        raise TypeError("Article must be a string.")

    text = text.strip()

    if not text:
        return []

    # Normalize multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)

    # Basic English sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Clean sentences
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


# ============================================================
# FASTTEXT TOKENIZATION
# ============================================================

def tokenize_sentence(sentence):
    """
    Tokenization method used when creating sentence vectors.
    """

    return re.findall(
        r"[A-Za-z0-9]+(?:'[A-Za-z]+)?",
        str(sentence).lower()
    )


# ============================================================
# CONVERT SENTENCES TO FASTTEXT VECTORS
# ============================================================

def sentence_to_vector(sentence):
    """
    Convert one sentence into a 64-dimensional FastText vector.
    """

    tokens = tokenize_sentence(sentence)

    if not tokens:
        return np.zeros(fasttext_model.vector_size)

    vectors = []

    for token in tokens:
        vectors.append(
            fasttext_model.wv[token]
        )

    return np.mean(vectors, axis=0)


def sentences_to_vectors(sentences):
    """
    Convert multiple sentences into FastText vectors.
    """

    vectors = [
        sentence_to_vector(sentence)
        for sentence in sentences
    ]

    return np.array(vectors)


# ============================================================
# SCORE SENTENCES USING SVM
# ============================================================

def score_sentences(vectors):
    """
    Use the trained scaler and SVM to calculate
    the probability that each sentence is important.
    """

    feature_names = [
        f"ft_{i}"
        for i in range(fasttext_model.vector_size)
    ]

    scaled_vectors = scaler.transform(
        pd.DataFrame(
            vectors,
            columns=feature_names
        )
    )

    probabilities = svm_model.predict_proba(
        scaled_vectors
    )[:, 1]

    return probabilities


# ============================================================
# SELECT TOP K SENTENCES
# ============================================================

def select_top_sentences(sentences, scores, top_k=5):
    """
    Select the highest-scoring sentences while
    preserving the original article order.
    """

    if len(sentences) == 0:
        return []

    # Make sure we don't request more sentences
    # than the article actually contains.
    top_k = min(top_k, len(sentences))

    # Get indices sorted by importance score
    ranked_indices = np.argsort(scores)[::-1]

    # Select the top K
    selected_indices = ranked_indices[:top_k]

    # Restore original article order
    selected_indices = sorted(selected_indices)

    selected_sentences = [
        sentences[index]
        for index in selected_indices
    ]

    return selected_sentences


# ============================================================
# MAIN SUMMARIZATION FUNCTION
# ============================================================

def summarize_article(text, top_k=5):
    """
    Complete extractive summarization pipeline.

    Input:
        text   = article text
        top_k  = number of sentences in summary

    Output:
        dictionary containing:
            sentences
            scores
            selected_sentences
            summary
    """

    # --------------------------------------------------------
    # Step 1: Split article into sentences
    # --------------------------------------------------------

    sentences = split_sentences(text)

    if not sentences:
        return {
            "sentences": [],
            "scores": [],
            "selected_sentences": [],
            "summary": ""
        }

    # --------------------------------------------------------
    # Step 2: Convert sentences to FastText vectors
    # --------------------------------------------------------

    vectors = sentences_to_vectors(sentences)

    # --------------------------------------------------------
    # Step 3: Calculate importance scores using SVM
    # --------------------------------------------------------

    scores = score_sentences(vectors)

    # --------------------------------------------------------
    # Step 4: Select top K sentences
    # --------------------------------------------------------

    selected_sentences = select_top_sentences(
        sentences,
        scores,
        top_k
    )

    # --------------------------------------------------------
    # Step 5: Combine selected sentences
    # --------------------------------------------------------

    summary = " ".join(selected_sentences)

    return {
        "sentences": sentences,
        "scores": scores.tolist(),
        "selected_sentences": selected_sentences,
        "summary": summary
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("EXTRACTIVE SUMMARIZATION TEST")
    print("=" * 60)

    article = """
    Manchester United defeated Liverpool 3-1 in an exciting
    Premier League match. Liverpool had more possession during
    the first half. Marcus Rashford scored the opening goal.
    United defended well in the second half. The victory moved
    United to third place in the league table. The manager
    praised the players after the match.
    """

    TOP_K = 3

    result = summarize_article(
        article,
        top_k=TOP_K
    )

    print("\nOriginal sentences:")
    print("-" * 60)

    for i, sentence in enumerate(
        result["sentences"],
        start=1
    ):
        print(
            f"{i}. {sentence}"
        )

    print("\nImportance scores:")
    print("-" * 60)

    for i, score in enumerate(
        result["scores"],
        start=1
    ):
        print(
            f"S{i}: {score:.4f}"
        )

    print("\nSelected sentences:")
    print("-" * 60)

    for sentence in result["selected_sentences"]:
        print(sentence)

    print("\nFinal Summary:")
    print("-" * 60)
    print(result["summary"])

    print("\n")
    print("=" * 60)
    print("Summarization test completed.")
    print("=" * 60)