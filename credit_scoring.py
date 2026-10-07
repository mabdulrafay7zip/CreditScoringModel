"""
Credit Scoring Model — CodeAlpha Machine Learning Internship
Author: Muhammad Abdul Rafay — CodeAlpha ML Intern

Predicts whether a loan applicant is creditworthy (good / bad credit risk)
using the German Credit dataset (UCI Machine Learning Repository).

Models compared:
    1. Logistic Regression
    2. Random Forest Classifier

Run:
    pip install -r requirements.txt
    python credit_scoring.py
Outputs:
    outputs/metrics.json               - accuracy / precision / recall / F1 for both models
    outputs/confusion_matrix_<model>.png
    outputs/model_comparison.png
    data/german_credit.csv             - cached copy of the dataset (downloaded on first run)
"""

import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_FILE = DATA_DIR / "german_credit.csv"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# Column names for the UCI German Credit (statlog) file — it ships without a header.
COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age_years", "other_installment_plans", "housing", "existing_credits",
    "job", "people_liable", "telephone", "foreign_worker", "credit_risk",
]


def load_dataset() -> pd.DataFrame:
    """Load the German Credit dataset, downloading and caching it on first run."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        print(f"Loading cached dataset: {DATA_FILE}")
        return pd.read_csv(DATA_FILE)

    print(f"Downloading German Credit dataset from UCI...\n  {UCI_URL}")
    try:
        raw = pd.read_csv(UCI_URL, sep=r"\s+", header=None, names=COLUMNS)
    except Exception as exc:  # pragma: no cover - network fallback
        raise RuntimeError(
            f"Could not download the dataset ({exc}). "
            f"Download german.data manually from {UCI_URL} and place a parsed "
            f"CSV at {DATA_FILE} with columns {COLUMNS}."
        ) from exc

    # UCI target: 1 = good credit, 2 = bad credit -> convert to 1 = good, 0 = bad
    raw["credit_risk"] = (raw["credit_risk"] == 1).astype(int)
    raw.to_csv(DATA_FILE, index=False)
    print(f"Saved cached copy to {DATA_FILE}")
    return raw


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric = df.select_dtypes(include=[np.number]).columns.drop("credit_risk").tolist()
    categorical = [c for c in df.columns if c not in numeric + ["credit_risk"]]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ]
    )


def evaluate(name: str, y_true, y_pred) -> dict:
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def save_confusion_matrix(name: str, y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bad risk", "Good risk"],
                yticklabels=["Bad risk", "Good risk"])
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


def save_comparison_chart(results: list[dict]) -> None:
    metrics = pd.DataFrame(results).set_index("model")
    metrics.plot(kind="bar", figsize=(9, 5), ylim=(0, 1))
    plt.title("Credit Scoring — Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['credit_risk'].value_counts().to_dict()}  (1 = good, 0 = bad)\n")

    X = df.drop(columns=["credit_risk"])
    y = df["credit_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(df)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    results = []
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("classifier", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append(evaluate(name, y_test, pred))
        save_confusion_matrix(name, y_test, pred)
        print(f"--- {name} ---")
        print(classification_report(y_test, pred, target_names=["Bad risk", "Good risk"]))

    save_comparison_chart(results)
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=== Summary ===")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nPlots and metrics saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
