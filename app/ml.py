"""Model training, evaluation, persistence, and inference."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from nlp import preprocess

DATA_DIR = Path(__file__).parent
MODEL_DIR = DATA_DIR / "models"
MODEL_PATH = MODEL_DIR / "recommendation_model.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_results.json"

RANDOM_STATE = 42
DEFAULT_PIPELINE = ("count", "logistic")


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build combined and preprocessed text columns from review data."""
    frame = df.copy()
    frame["combined"] = frame["Title"].fillna("") + " " + frame["Review Text"].fillna("")
    frame["processed"] = frame["combined"].apply(preprocess)
    return frame


def build_pipeline(vectorizer_type: str, classifier_type: str) -> Pipeline:
    """Create a sklearn pipeline for a vectorizer + classifier pair."""
    if vectorizer_type == "count":
        vectorizer = CountVectorizer(min_df=2)
    elif vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(min_df=2)
    else:
        raise ValueError(f"Unknown vectorizer: {vectorizer_type}")

    if classifier_type == "logistic":
        classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif classifier_type == "naive_bayes":
        classifier = MultinomialNB()
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")

    return Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier),
    ])


def train_pipeline(df: pd.DataFrame, vectorizer_type: str = "count", classifier_type: str = "logistic") -> Pipeline:
    """Fit a vectorizer + classifier pipeline on all rows in df."""
    frame = prepare_training_frame(df)
    pipeline = build_pipeline(vectorizer_type, classifier_type)
    pipeline.fit(frame["processed"], frame["Recommended IND"].values)
    return pipeline


def predict_recommendation(pipeline: Pipeline, title: str, review_text: str) -> tuple[int, float]:
    """Predict recommendation label and confidence percentage."""
    combined = preprocess(title + " " + review_text)
    label = int(pipeline.predict([combined])[0])
    proba = float(pipeline.predict_proba([combined])[0][label])
    return label, round(proba * 100, 1)


def _metrics_dict(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def evaluate_holdout(
    df: pd.DataFrame,
    vectorizer_type: str = "count",
    classifier_type: str = "logistic",
    test_size: float = 0.2,
) -> dict:
    """Evaluate one pipeline using a stratified train/test split."""
    frame = prepare_training_frame(df)
    x_train, x_test, y_train, y_test = train_test_split(
        frame["processed"],
        frame["Recommended IND"].values,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=frame["Recommended IND"].values,
    )

    start = time.perf_counter()
    pipeline = build_pipeline(vectorizer_type, classifier_type)
    pipeline.fit(x_train, y_train)
    train_seconds = round(time.perf_counter() - start, 3)

    y_pred = pipeline.predict(x_test)
    metrics = _metrics_dict(y_test, y_pred)
    metrics.update({
        "vectorizer": vectorizer_type,
        "classifier": classifier_type,
        "train_seconds": train_seconds,
        "test_size": test_size,
        "n_train": len(x_train),
        "n_test": len(x_test),
    })
    return metrics


def compare_models(df: pd.DataFrame, test_size: float = 0.2) -> list[dict]:
    """Compare three baseline pipelines on a held-out test set."""
    configs = [
        ("count", "logistic"),
        ("tfidf", "logistic"),
        ("tfidf", "naive_bayes"),
    ]
    results = []
    for vectorizer_type, classifier_type in configs:
        results.append(evaluate_holdout(
            df,
            vectorizer_type=vectorizer_type,
            classifier_type=classifier_type,
            test_size=test_size,
        ))
    return results


def save_model(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: Path = MODEL_PATH) -> Pipeline | None:
    if not path.exists():
        return None
    return joblib.load(path)


def save_evaluation_results(
    holdout_metrics: dict,
    comparison: list[dict],
    path: Path = METRICS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "holdout_evaluation": holdout_metrics,
        "model_comparison": comparison,
        "production_model": {
            "vectorizer": DEFAULT_PIPELINE[0],
            "classifier": DEFAULT_PIPELINE[1],
            "trained_on": "full dataset",
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_evaluation_results(path: Path = METRICS_PATH) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_or_train_model(df: pd.DataFrame, retrain: bool = False) -> Pipeline:
    """Load a saved model or train on the full dataset."""
    if not retrain:
        existing = load_model()
        if existing is not None:
            return existing

    pipeline = train_pipeline(df, *DEFAULT_PIPELINE)
    save_model(pipeline)
    return pipeline
