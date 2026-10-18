import pandas as pd
import numpy as np
import re
from pathlib import Path

from gensim.models import FastText
from sklearn.model_selection import train_test_split


class FastTextSentenceEncoder:
    def __init__(
        self,
        vector_size=64,
        min_n=3,
        max_n=6,
        epochs=50
    ):
        self.vector_size = vector_size
        self.min_n = min_n
        self.max_n = max_n
        self.epochs = epochs
        self.model = None

    # --------------------------------------------------
    # Tokenization
    # --------------------------------------------------

    def tokenize(self, text):
        return re.findall(
            r"[A-Za-z0-9]+(?:'[A-Za-z]+)?",
            str(text).lower()
        )

    # --------------------------------------------------
    # Train FastText
    # --------------------------------------------------

    def fit(self, sentences):

        tokenized = [
            self.tokenize(sentence)
            for sentence in sentences
        ]

        self.model = FastText(
            vector_size=self.vector_size,
            window=4,
            min_count=1,
            min_n=self.min_n,
            max_n=self.max_n,
            sg=1,
            workers=1,
            seed=7
        )

        self.model.build_vocab(
            corpus_iterable=tokenized
        )

        self.model.train(
            corpus_iterable=tokenized,
            total_examples=len(tokenized),
            epochs=self.epochs
        )

        return self

    # --------------------------------------------------
    # Convert sentences into vectors
    # --------------------------------------------------

    def transform(self, sentences):

        if self.model is None:
            raise ValueError(
                "FastText model has not been trained yet."
            )

        vectors = []

        for sentence in sentences:

            words = self.tokenize(sentence)

            # Empty sentence
            if len(words) == 0:

                vectors.append(
                    np.zeros(
                        self.vector_size,
                        dtype=np.float32
                    )
                )

                continue

            # Get word vectors
            word_vectors = [
                self.model.wv[word]
                for word in words
            ]

            # Average word vectors
            sentence_vector = np.mean(
                word_vectors,
                axis=0
            )

            vectors.append(
                sentence_vector.astype(np.float32)
            )

        return np.array(
            vectors,
            dtype=np.float32
        )


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    # --------------------------------------------------
    # Project paths
    # --------------------------------------------------

    BASE_DIR = Path(__file__).resolve().parents[2]

    input_file = (
        BASE_DIR
        / "data"
        / "sentence_labels.csv"
    )

    output_dir = (
        BASE_DIR
        / "results"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    vector_output_file = (
        output_dir
        / "sentence_vectors.csv"
    )

    split_output_file = (
        output_dir
        / "article_split.csv"
    )

    fasttext_output_file = (
        output_dir
        / "fasttext.model"
    )

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    print("=" * 60)
    print("Loading sentence labels...")
    print("=" * 60)

    df = pd.read_csv(input_file)

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Total articles: {df['article_id'].nunique()}"
    )

    print(
        f"Total sentences: {len(df)}"
    )

    print("\nLabel distribution:")
    print(
        df["label"].value_counts()
    )

    # --------------------------------------------------
    # Validate dataset
    # --------------------------------------------------

    required_columns = [
        "article_id",
        "domain",
        "sentence_id",
        "sentence",
        "label"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------
    # Article-level train/test split
    #
    # IMPORTANT:
    # We split articles instead of sentences.
    #
    # This prevents sentences from the same article
    # appearing in both training and testing data.
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Creating article-level train/test split...")
    print("=" * 60)

    article_ids = df["article_id"].unique()

    train_articles, test_articles = train_test_split(
        article_ids,
        test_size=0.2,
        random_state=42
    )

    train_articles = sorted(train_articles)
    test_articles = sorted(test_articles)

    print(
        f"Training articles: {len(train_articles)}"
    )

    print(
        f"Testing articles: {len(test_articles)}"
    )

    # --------------------------------------------------
    # Create train/test dataframes
    # --------------------------------------------------

    train_df = df[
        df["article_id"].isin(train_articles)
    ].copy()

    test_df = df[
        df["article_id"].isin(test_articles)
    ].copy()

    print(
        f"Training sentences: {len(train_df)}"
    )

    print(
        f"Testing sentences: {len(test_df)}"
    )

    # --------------------------------------------------
    # Verify no article leakage
    # --------------------------------------------------

    overlap = (
        set(train_articles)
        &
        set(test_articles)
    )

    if overlap:

        raise ValueError(
            f"Article leakage detected: {overlap}"
        )

    print(
        "\nArticle overlap:",
        overlap
    )

    # --------------------------------------------------
    # Save article split
    # --------------------------------------------------

    split_rows = []

    for article_id in train_articles:

        split_rows.append({
            "article_id": article_id,
            "split": "train"
        })

    for article_id in test_articles:

        split_rows.append({
            "article_id": article_id,
            "split": "test"
        })

    split_df = pd.DataFrame(
        split_rows
    )

    split_df.to_csv(
        split_output_file,
        index=False
    )

    print(
        f"\nArticle split saved to:\n"
        f"{split_output_file}"
    )

    # --------------------------------------------------
    # Train FastText ONLY on training articles
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Training FastText...")
    print("=" * 60)

    train_sentences = (
        train_df["sentence"]
        .tolist()
    )

    print(
        f"FastText training sentences: "
        f"{len(train_sentences)}"
    )

    encoder = FastTextSentenceEncoder(
        vector_size=64,
        min_n=3,
        max_n=6,
        epochs=50
    )

    encoder.fit(
        train_sentences
    )

    print(
        "FastText training completed."
    )

    # --------------------------------------------------
    # Save FastText model
    # --------------------------------------------------

    encoder.model.save(
        str(fasttext_output_file)
    )

    print(
        f"FastText model saved to:\n"
        f"{fasttext_output_file}"
    )

    # --------------------------------------------------
    # Generate vectors for ALL sentences
    #
    # The model was trained only on training articles.
    # It can still generate vectors for unseen test
    # sentences because FastText uses subword information.
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Generating sentence vectors...")
    print("=" * 60)

    all_sentences = (
        df["sentence"]
        .tolist()
    )

    vectors = encoder.transform(
        all_sentences
    )

    print(
        f"Vector shape: {vectors.shape}"
    )

    # --------------------------------------------------
    # Create vector dataframe
    # --------------------------------------------------

    vector_columns = [
        f"ft_{i}"
        for i in range(64)
    ]

    vector_df = pd.DataFrame(
        vectors,
        columns=vector_columns
    )

    # --------------------------------------------------
    # Combine original data + vectors
    # --------------------------------------------------

    result = pd.concat(
        [
            df.reset_index(drop=True),
            vector_df
        ],
        axis=1
    )

    # Add split information
    split_mapping = (
        split_df
        .set_index("article_id")["split"]
        .to_dict()
    )

    result["split"] = (
        result["article_id"]
        .map(split_mapping)
    )

    # --------------------------------------------------
    # Save sentence vectors
    # --------------------------------------------------

    result.to_csv(
        vector_output_file,
        index=False
    )

    print(
        f"\nSentence vectors saved to:\n"
        f"{vector_output_file}"
    )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FastText feature extraction completed!")
    print("=" * 60)

    print(
        f"Articles: {df['article_id'].nunique()}"
    )

    print(
        f"Sentences: {len(df)}"
    )

    print(
        f"Vector dimensions: 64"
    )

    print(
        f"Training articles: {len(train_articles)}"
    )

    print(
        f"Testing articles: {len(test_articles)}"
    )