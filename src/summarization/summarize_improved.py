import os
import re
import sys
import joblib
import numpy as np
import pandas as pd
import unicodedata
from html import unescape
from gensim.models import FastText


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

FASTTEXT_MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "improved_fasttext.model"
)

SVM_MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "improved_svm_model.pkl"
)

SCALER_PATH = os.path.join(
    RESULTS_DIR,
    "improved_scaler.pkl"
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

fasttext_model = FastText.load(FASTTEXT_MODEL_PATH)
svm_model = joblib.load(SVM_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ============================================================
# FEATURE NAMES
# Must match train_improved_classifier.py exactly
# ============================================================

FASTTEXT_FEATURES = [
    f"ft_{i}"
    for i in range(64)
]

EXTRA_FEATURES = [
    "sentence_position",
    "relative_position",
    "word_count",
    "character_count",
    "normalized_word_count"
]

FEATURE_NAMES = FASTTEXT_FEATURES + EXTRA_FEATURES


# ============================================================
# TOKENIZATION
# Must match improved_features.py
# ============================================================

TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:'[A-Za-z]+)?"
)


def tokenize(sentence):
    """
    Convert a sentence into tokens using the
    same pattern used during FastText training.
    """
    return TOKEN_PATTERN.findall(sentence)


# ============================================================
# ARTICLE TEXT CLEANUP
# ============================================================

def clean_article_text(text):
    """
    Normalize common copy/paste artifacts without removing article content.
    """
    if not isinstance(text, str):
        raise TypeError("Article must be a string.")

    text = unicodedata.normalize("NFC", unescape(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u00ad", "")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    paragraphs = re.split(r"\n[ \t]*\n+", text)
    cleaned_paragraphs = []

    for paragraph in paragraphs:
        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in paragraph.split("\n")
        ]
        cleaned_paragraph = " ".join(line for line in lines if line)
        if cleaned_paragraph:
            cleaned_paragraphs.append(cleaned_paragraph)

    return "\n\n".join(cleaned_paragraphs)


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):
    """
    Split article text into sentences.

    Example:
        "Hello world. This is NLP!"
        ->
        ["Hello world.", "This is NLP!"]
    """

    normalized_text = re.sub(r"\s+", " ", clean_article_text(text)).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        normalized_text
    )

    # Remove empty sentences
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


# ============================================================
# FASTTEXT SENTENCE VECTOR
# ============================================================

def get_sentence_vector(sentence):
    """
    Create a 64-dimensional sentence vector
    by averaging the FastText vectors of its tokens.
    """

    tokens = tokenize(sentence)

    if not tokens:
        return np.zeros(64)

    vectors = []

    for token in tokens:
        try:
            vectors.append(
                fasttext_model.wv[token]
            )
        except KeyError:
            # FastText normally handles unseen words
            # using subword information.
            continue

    if not vectors:
        return np.zeros(64)

    return np.mean(vectors, axis=0)


# ============================================================
# CREATE IMPROVED FEATURES
# ============================================================

def create_features(sentences):
    """
    Create the exact 69 features used during training:

    64 FastText features
    +
    5 sentence-level features

    Features:
        sentence_position
        relative_position
        word_count
        character_count
        normalized_word_count
    """

    total_sentences = len(sentences)

    # --------------------------------------------------------
    # Calculate word counts first because
    # normalized_word_count needs the maximum.
    # --------------------------------------------------------

    word_counts = []

    for sentence in sentences:
        tokens = tokenize(sentence)
        word_counts.append(len(tokens))

    max_word_count = max(word_counts) if word_counts else 1

    rows = []

    for i, sentence in enumerate(sentences):

        # ----------------------------------------------------
        # 64-dimensional FastText vector
        # ----------------------------------------------------

        sentence_vector = get_sentence_vector(sentence)

        # ----------------------------------------------------
        # Additional sentence-level features
        # ----------------------------------------------------

        sentence_position = i

        if total_sentences > 1:
            relative_position = i / (total_sentences - 1)
        else:
            relative_position = 0.0

        word_count = word_counts[i]

        character_count = len(sentence)

        normalized_word_count = (
            word_count / max_word_count
            if max_word_count > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Combine all 69 features
        # ----------------------------------------------------

        feature_row = list(sentence_vector) + [
            sentence_position,
            relative_position,
            word_count,
            character_count,
            normalized_word_count
        ]

        rows.append(feature_row)

    # --------------------------------------------------------
    # Convert to DataFrame
    #
    # Important:
    # The scaler was trained using feature names.
    # Therefore we provide the same feature names here.
    # --------------------------------------------------------

    feature_df = pd.DataFrame(
        rows,
        columns=FEATURE_NAMES
    )

    return feature_df


# ============================================================
# SCORE SENTENCES
# ============================================================

def score_sentences(sentences):
    """
    Generate importance probability for every sentence.
    """

    feature_df = create_features(sentences)

    # Scale using the SAME scaler used during training
    scaled_features = scaler.transform(feature_df)

    # SVM probability of class 1
    scores = svm_model.predict_proba(
        scaled_features
    )[:, 1]

    return scores


# ============================================================
# SUMMARIZE ARTICLE
# ============================================================

def summarize_article(text, top_k=7):
    """
    Generate an extractive summary.

    Steps:
        1. Split article into sentences
        2. Generate improved features
        3. Scale features
        4. SVM predicts importance probability
        5. Select the highest-scoring sentence from each article section
        6. Keep selected sentences in original article order
        7. Return summary
    """

    sentences = split_sentences(clean_article_text(text))

    if not sentences:
        return ""

    # --------------------------------------------------------
    # If article has fewer sentences than requested top_k,
    # simply use all sentences.
    # --------------------------------------------------------

    top_k = min(top_k, len(sentences))

    # --------------------------------------------------------
    # Get importance scores
    # --------------------------------------------------------

    scores = score_sentences(sentences)

    # Divide the full article into consecutive sections and select
    # each section's highest-scoring sentence for broad coverage.
    selected_indices = []
    for section in range(top_k):
        start = section * len(sentences) // top_k
        end = (section + 1) * len(sentences) // top_k
        selected_indices.append(
            start + int(np.argmax(scores[start:end]))
        )

    summary_sentences = [
        sentences[i]
        for i in selected_indices
    ]

    summary = " ".join(summary_sentences)

    return summary


# ============================================================
# DEBUG / TEST
# ============================================================

if __name__ == "__main__":

    sample_text = """
    Artificial intelligence is becoming increasingly important
    in modern technology. Machine learning allows computers to
    learn patterns from large datasets. Natural language
    processing helps computers understand human language.
    Researchers are developing new models for text summarization.
    Extractive summarization selects important sentences from
    an original document. These systems can help users read
    long documents more efficiently.
    """

    print("=" * 70)
    print("IMPROVED EXTRACTIVE SUMMARIZATION TEST")
    print("=" * 70)

    sentences = split_sentences(sample_text)

    print("\nNumber of sentences:", len(sentences))

    # --------------------------------------------------------
    # Show sentence scores
    # --------------------------------------------------------

    scores = score_sentences(sentences)

    print("\nSentence Scores:")
    print("-" * 70)

    for i, (sentence, score) in enumerate(
        zip(sentences, scores)
    ):
        print(
            f"S{i + 1}: {score:.4f} | {sentence}"
        )

    # --------------------------------------------------------
    # Generate summary
    # --------------------------------------------------------

    summary = summarize_article(
        sample_text,
        top_k=3
    )

    print("\n" + "=" * 70)
    print("TOP 3 SUMMARY")
    print("=" * 70)

    print(summary)