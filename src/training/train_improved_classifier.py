from pathlib import Path
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = (
    RESULTS_DIR
    / "improved_sentence_vectors.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

IMPROVED_SVM_FILE = (
    RESULTS_DIR
    / "improved_svm_model.pkl"
)

IMPROVED_SCALER_FILE = (
    RESULTS_DIR
    / "improved_scaler.pkl"
)

COMPARISON_FILE = (
    RESULTS_DIR
    / "improved_model_comparison.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING IMPROVED FEATURES")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# DEFINE FEATURES
# ============================================================

FASTTEXT_FEATURES = [
    f"ft_{i}"
    for i in range(64)
]

ADDITIONAL_FEATURES = [
    "sentence_position",
    "relative_position",
    "word_count",
    "character_count",
    "normalized_word_count"
]

FEATURES = (
    FASTTEXT_FEATURES
    + ADDITIONAL_FEATURES
)


print("\nNumber of features:", len(FEATURES))


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        f"Missing features: {missing_features}"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_df = df[
    df["split"] == "train"
].copy()

test_df = df[
    df["split"] == "test"
].copy()


X_train = train_df[FEATURES]
y_train = train_df["label"]

X_test = test_df[FEATURES]
y_test = test_df["label"]


print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print(
    "\nTraining articles:",
    train_df["article_id"].nunique()
)

print(
    "Testing articles:",
    test_df["article_id"].nunique()
)


# ============================================================
# VERIFY ARTICLE LEAKAGE
# ============================================================

train_articles = set(
    train_df["article_id"]
)

test_articles = set(
    test_df["article_id"]
)

overlap = (
    train_articles
    .intersection(test_articles)
)

print("\nArticle overlap:", overlap)

if overlap:

    raise ValueError(
        "Article leakage detected!"
    )


# ============================================================
# SCALE FEATURES
# ============================================================

print("\n")
print("=" * 60)
print("SCALING FEATURES")
print("=" * 60)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# TRAIN MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ),

    "SVM":
        SVC(
            probability=True,
            random_state=42,
            class_weight="balanced"
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
}


results = []


# ============================================================
# TRAIN + EVALUATE
# ============================================================

for name, model in models.items():

    print("\n")
    print("=" * 60)
    print(f"TRAINING {name.upper()}")
    print("=" * 60)

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    results.append({

        "model": name,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1
    })


# ============================================================
# MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    results
)

comparison_df = comparison_df.sort_values(
    by="f1",
    ascending=False
)


print("\n")
print("=" * 60)
print("IMPROVED MODEL COMPARISON")
print("=" * 60)

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODELS
# ============================================================

svm_model = models["SVM"]

joblib.dump(
    svm_model,
    IMPROVED_SVM_FILE
)

joblib.dump(
    scaler,
    IMPROVED_SCALER_FILE
)

comparison_df.to_csv(
    COMPARISON_FILE,
    index=False
)


# ============================================================
# SAVE OTHER MODELS
# ============================================================

joblib.dump(
    models["Logistic Regression"],
    RESULTS_DIR
    / "improved_logistic_regression_model.pkl"
)

joblib.dump(
    models["Random Forest"],
    RESULTS_DIR
    / "improved_random_forest_model.pkl"
)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 60)
print("IMPROVED CLASSIFIER TRAINING COMPLETED")
print("=" * 60)

print(
    "\nImproved SVM saved to:"
)

print(
    IMPROVED_SVM_FILE
)

print(
    "\nImproved scaler saved to:"
)

print(
    IMPROVED_SCALER_FILE
)

print(
    "\nModel comparison saved to:"
)

print(
    COMPARISON_FILE
)