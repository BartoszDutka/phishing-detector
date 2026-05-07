import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer


def run_cross_validation(X, y):
    """5-fold stratified cross-validation."""
    print("\n=== Cross-Validation (5-fold Stratified) ===")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

    scoring = {
        "accuracy": "accuracy",
        "precision_phishing": make_scorer(precision_score, pos_label=0),
        "recall_phishing": make_scorer(recall_score, pos_label=0),
        "f1_phishing": make_scorer(f1_score, pos_label=0),
    }

    results = []

    for model_name, model in models.items():
        print(f"\n{model_name}:")
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

        for metric_name in scoring.keys():
            scores = cv_results[f"test_{metric_name}"]
            mean_score = scores.mean()
            std_score = scores.std()

            print(f"  {metric_name}:")
            for fold_idx, score in enumerate(scores):
                print(f"    Fold {fold_idx + 1}: {score:.4f}")
            print(f"    Mean: {mean_score:.4f} (+/- {std_score:.4f})")

            results.append({
                "model": model_name,
                "metric": metric_name,
                "mean": mean_score,
                "std": std_score,
            })

    results_df = pd.DataFrame(results)
    return results_df
