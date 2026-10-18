from pathlib import Path
import re
import pandas as pd
import numpy as np
from gensim.models import FastText


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_FILE = DATA_DIR / "sentence_labels.csv"
SPLIT_FILE = RESULTS_DIR / "article_split.csv"

MODEL_FILE = RESULTS_DIR / "improved_fasttext.model"
OUTPUT_FILE = RESULTS_DIR / "improved_sentence_vectors.csv"


# ============================================================
# SETTINGS
# ============================================================

VECTOR_SIZE = 64


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_sentence(sentence):

    return re.findall(
        r"[A-Za-z0-9]+(?:'[A-Za-z]+)?",
        sentence
    )


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
# MERGE ARTICLE SPLIT
# ============================================================

df = df.merge(
    split_df[["article_id", "split"]],
    on="article_id",
    how="left"
)

print("\nSplit counts:")
print(df["split"].value_counts())


# ============================================================
# CHECK ARTICLE LEAKAGE
# ============================================================

train_articles = set(
    df.loc[df["split"] == "train", "article_id"]
)

test_articles = set(
    df.loc[df["split"] == "test", "article_id"]
)

print("\nTraining articles:", len(train_articles))
print("Testing articles:", len(test_articles))

print(
    "Article overlap:",
    train_articles.intersection(test_articles)
)


# ============================================================
# TRAIN FASTTEXT ONLY ON TRAINING ARTICLES
# ============================================================

train_df = df[df["split"] == "train"]

training_sentences = []

for sentence in train_df["sentence"]:

    tokens = tokenize_sentence(sentence)

    if tokens:
        training_sentences.append(tokens)


print("\nFastText training sentences:")
print(len(training_sentences))


# ============================================================
# TRAIN FASTTEXT
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING IMPROVED FASTTEXT")
print("=" * 60)

fasttext_model = FastText(
    vector_size=VECTOR_SIZE,
    window=4,
    min_count=1,
    workers=1,
    sg=1,
    seed=7,
    min_n=3,
    max_n=6
)

fasttext_model.build_vocab(
    corpus_iterable=training_sentences
)

fasttext_model.train(
    corpus_iterable=training_sentences,
    total_examples=len(training_sentences),
    epochs=50
)

print("FastText training completed.")


# ============================================================
# SAVE FASTTEXT MODEL
# ============================================================

fasttext_model.save(
    str(MODEL_FILE)
)

print(f"FastText model saved to:\n{MODEL_FILE}")


# ============================================================
# CREATE FEATURES
# ============================================================

print("\n")
print("=" * 60)
print("CREATING IMPROVED FEATURES")
print("=" * 60)


feature_rows = []


for article_id, article_df in df.groupby(
    "article_id",
    sort=False
):

    # Keep original sentence order
    article_df = article_df.sort_values(
        "sentence_id"
    )

    total_sentences = len(article_df)

    for _, row in article_df.iterrows():

        sentence = row["sentence"]

        tokens = tokenize_sentence(sentence)

        # ----------------------------------------------------
        # FastText sentence vector
        # ----------------------------------------------------

        if tokens:

            token_vectors = [
                fasttext_model.wv[token]
                for token in tokens
            ]

            sentence_vector = np.mean(
                token_vectors,
                axis=0
            )

        else:

            sentence_vector = np.zeros(
                VECTOR_SIZE
            )

        # ----------------------------------------------------
        # Sentence position
        # ----------------------------------------------------

        sentence_id = int(row["sentence_id"])

        if total_sentences > 1:

            relative_position = (
                sentence_id /
                (total_sentences - 1)
            )

        else:

            relative_position = 0.0

        # ----------------------------------------------------
        # Sentence length
        # ----------------------------------------------------

        word_count = len(tokens)

        character_count = len(sentence)

        # Normalized word count
        max_word_count = max(
            len(
                tokenize_sentence(s)
            )
            for s in article_df["sentence"]
        )

        if max_word_count > 0:

            normalized_word_count = (
                word_count /
                max_word_count
            )

        else:

            normalized_word_count = 0.0

        # ----------------------------------------------------
        # Build row
        # ----------------------------------------------------

        feature_row = {
            "article_id": article_id,
            "domain": row["domain"],
            "sentence_id": sentence_id,
            "sentence": sentence,
            "label": row["label"],
            "split": row["split"],

            "sentence_position": sentence_id,
            "relative_position": relative_position,

            "word_count": word_count,
            "character_count": character_count,
            "normalized_word_count": normalized_word_count
        }

        # Add FastText features
        for i in range(VECTOR_SIZE):

            feature_row[
                f"ft_{i}"
            ] = sentence_vector[i]

        feature_rows.append(feature_row)


# ============================================================
# CREATE DATAFRAME
# ============================================================

feature_df = pd.DataFrame(
    feature_rows
)


# ============================================================
# FEATURE ORDER
# ============================================================

feature_columns = [
    f"ft_{i}"
    for i in range(VECTOR_SIZE)
] + [
    "sentence_position",
    "relative_position",
    "word_count",
    "character_count",
    "normalized_word_count"
]


# ============================================================
# SAVE FEATURES
# ============================================================

feature_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nFeature matrix shape: "
    f"{feature_df.shape}"
)

print(
    f"Number of features: "
    f"{len(feature_columns)}"
)

print("\nFeatures:")
for feature in feature_columns:
    print(f" - {feature}")

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n")
print("=" * 60)
print("IMPROVED FEATURE EXTRACTION COMPLETED")
print("=" * 60)