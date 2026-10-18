import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import joblib


# ======================================================
# PROJECT PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[2]

input_file = (
    BASE_DIR
    / "results"
    / "sentence_vectors.csv"
)

output_dir = (
    BASE_DIR
    / "results"
)

output_dir.mkdir(
    exist_ok=True
)


# ======================================================
# LOAD DATA
# ======================================================

print("=" * 60)
print("Loading sentence vectors...")
print("=" * 60)

df = pd.read_csv(
    input_file
)

print(
    f"Dataset shape: {df.shape}"
)

print(
    f"Articles: {df['article_id'].nunique()}"
)

print(
    f"Sentences: {len(df)}"
)


# ======================================================
# FASTTEXT FEATURES
# ======================================================

feature_columns = [
    f"ft_{i}"
    for i in range(64)
]

missing_features = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing_features:

    raise ValueError(
        f"Missing FastText features: "
        f"{missing_features}"
    )


X = df[feature_columns]

y = df["label"]


print(
    "\nFeature shape:",
    X.shape
)

print(
    "\nLabel distribution:"
)

print(
    y.value_counts()
)


# ======================================================
# USE THE PREDEFINED ARTICLE SPLIT
# ======================================================

if "split" not in df.columns:

    raise ValueError(
        "The 'split' column is missing. "
        "Run fasttext_features.py first."
    )


train_df = df[
    df["split"] == "train"
].copy()

test_df = df[
    df["split"] == "test"
].copy()


print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(
    "Training articles:",
    train_df["article_id"].nunique()
)

print(
    "Testing articles:",
    test_df["article_id"].nunique()
)

print(
    "Training sentences:",
    len(train_df)
)

print(
    "Testing sentences:",
    len(test_df)
)


# ======================================================
# VERIFY NO ARTICLE LEAKAGE
# ======================================================

train_article_ids = set(
    train_df["article_id"]
)

test_article_ids = set(
    test_df["article_id"]
)

overlap = (
    train_article_ids
    &
    test_article_ids
)

print(
    "Article overlap:",
    overlap
)

if overlap:

    raise ValueError(
        f"Article leakage detected: {overlap}"
    )


# ======================================================
# CREATE TRAIN / TEST FEATURES
# ======================================================

X_train = train_df[
    feature_columns
]

y_train = train_df[
    "label"
]

X_test = test_df[
    feature_columns
]

y_test = test_df[
    "label"
]


# ======================================================
# STANDARD SCALER
#
# Used for Logistic Regression and SVM.
# ======================================================

print("\n" + "=" * 60)
print("Scaling features...")
print("=" * 60)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print(
    "Scaling completed."
)


# ======================================================
# RESULTS STORAGE
# ======================================================

model_results = []


# ======================================================
# LOGISTIC REGRESSION
# ======================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

lr = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

lr.fit(
    X_train_scaled,
    y_train
)

lr_pred = lr.predict(
    X_test_scaled
)


lr_accuracy = accuracy_score(
    y_test,
    lr_pred
)

lr_precision = precision_score(
    y_test,
    lr_pred,
    zero_division=0
)

lr_recall = recall_score(
    y_test,
    lr_pred,
    zero_division=0
)

lr_f1 = f1_score(
    y_test,
    lr_pred,
    zero_division=0
)


print(
    f"Accuracy:  {lr_accuracy:.4f}"
)

print(
    f"Precision: {lr_precision:.4f}"
)

print(
    f"Recall:    {lr_recall:.4f}"
)

print(
    f"F1:        {lr_f1:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        lr_pred,
        zero_division=0
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        lr_pred
    )
)


model_results.append({
    "model": "Logistic Regression",
    "accuracy": lr_accuracy,
    "precision": lr_precision,
    "recall": lr_recall,
    "f1": lr_f1
})


# ======================================================
# SUPPORT VECTOR MACHINE
# ======================================================

print("\n" + "=" * 60)
print("SUPPORT VECTOR MACHINE")
print("=" * 60)

svm = SVC(
    probability=True,
    random_state=42,
    class_weight="balanced"
)

svm.fit(
    X_train_scaled,
    y_train
)

svm_pred = svm.predict(
    X_test_scaled
)


svm_accuracy = accuracy_score(
    y_test,
    svm_pred
)

svm_precision = precision_score(
    y_test,
    svm_pred,
    zero_division=0
)

svm_recall = recall_score(
    y_test,
    svm_pred,
    zero_division=0
)

svm_f1 = f1_score(
    y_test,
    svm_pred,
    zero_division=0
)


print(
    f"Accuracy:  {svm_accuracy:.4f}"
)

print(
    f"Precision: {svm_precision:.4f}"
)

print(
    f"Recall:    {svm_recall:.4f}"
)

print(
    f"F1:        {svm_f1:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        svm_pred,
        zero_division=0
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        svm_pred
    )
)


model_results.append({
    "model": "SVM",
    "accuracy": svm_accuracy,
    "precision": svm_precision,
    "recall": svm_recall,
    "f1": svm_f1
})


# ======================================================
# RANDOM FOREST
# ======================================================

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf.fit(
    X_train,
    y_train
)

rf_pred = rf.predict(
    X_test
)


rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_pred,
    zero_division=0
)


print(
    f"Accuracy:  {rf_accuracy:.4f}"
)

print(
    f"Precision: {rf_precision:.4f}"
)

print(
    f"Recall:    {rf_recall:.4f}"
)

print(
    f"F1:        {rf_f1:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_pred,
        zero_division=0
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        rf_pred
    )
)


model_results.append({
    "model": "Random Forest",
    "accuracy": rf_accuracy,
    "precision": rf_precision,
    "recall": rf_recall,
    "f1": rf_f1
})


# ======================================================
# MODEL COMPARISON
# ======================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(
    model_results
)

results_df = results_df.sort_values(
    by="f1",
    ascending=False
)

print(
    results_df.to_string(
        index=False
    )
)


# ======================================================
# SAVE MODEL COMPARISON
# ======================================================

results_file = (
    output_dir
    / "model_comparison.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print(
    f"\nModel comparison saved to:\n"
    f"{results_file}"
)


# ======================================================
# SAVE MODELS
# ======================================================

print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)


lr_file = (
    output_dir
    / "logistic_regression_model.pkl"
)

svm_file = (
    output_dir
    / "svm_model.pkl"
)

rf_file = (
    output_dir
    / "random_forest_model.pkl"
)

scaler_file = (
    output_dir
    / "scaler.pkl"
)


joblib.dump(
    lr,
    lr_file
)

joblib.dump(
    svm,
    svm_file
)

joblib.dump(
    rf,
    rf_file
)

joblib.dump(
    scaler,
    scaler_file
)


print(
    f"Logistic Regression saved:\n"
    f"{lr_file}"
)

print(
    f"SVM saved:\n"
    f"{svm_file}"
)

print(
    f"Random Forest saved:\n"
    f"{rf_file}"
)

print(
    f"Scaler saved:\n"
    f"{scaler_file}"
)


# ======================================================
# BEST MODEL
# ======================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"Model: {best_model['model']}"
)

print(
    f"F1: {best_model['f1']:.4f}"
)

print(
    f"Accuracy: {best_model['accuracy']:.4f}"
)

print(
    f"Precision: {best_model['precision']:.4f}"
)

print(
    f"Recall: {best_model['recall']:.4f}"
)

print("\nTraining completed successfully.")
