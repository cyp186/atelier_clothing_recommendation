"""Run hold-out evaluation and model comparison; train and save production model."""

from pathlib import Path

import pandas as pd

from ml import (
    DEFAULT_PIPELINE,
    compare_models,
    evaluate_holdout,
    get_or_train_model,
    save_evaluation_results,
)

DATA_DIR = Path(__file__).parent


def main() -> None:
    df = pd.read_csv(DATA_DIR / "assignment3_II.csv")
    print(f"Loaded {len(df):,} reviews\n")

    holdout = evaluate_holdout(df, *DEFAULT_PIPELINE)
    comparison = compare_models(df)

    save_evaluation_results(holdout, comparison)
    get_or_train_model(df, retrain=True)

    print("Hold-out evaluation (CountVectorizer + Logistic Regression)")
    print(f"  Accuracy:  {holdout['accuracy']:.1%}")
    print(f"  Precision: {holdout['precision']:.1%}")
    print(f"  Recall:    {holdout['recall']:.1%}")
    print(f"  F1:        {holdout['f1']:.1%}")
    print(f"  Confusion matrix: {holdout['confusion_matrix']}")
    print()

    print("Model comparison (held-out test set)")
    print(f"{'Vectorizer':<12} {'Classifier':<14} {'Accuracy':>10} {'F1':>8} {'Train (s)':>10}")
    print("-" * 58)
    for row in comparison:
        name = f"{row['vectorizer']} + {row['classifier']}"
        print(
            f"{row['vectorizer']:<12} {row['classifier']:<14} "
            f"{row['accuracy']:>9.1%} {row['f1']:>8.1%} {row['train_seconds']:>10.3f}"
        )

    print("\nSaved:")
    print("  models/recommendation_model.joblib")
    print("  models/evaluation_results.json")


if __name__ == "__main__":
    main()
