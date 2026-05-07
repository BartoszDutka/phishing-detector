import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_models(models, X_test, y_test):
    """
    Evaluate Logistic Regression and Random Forest models on test set.

    Returns:
        DataFrame: model, accuracy, precision_phishing, recall_phishing, f1_phishing
    """
    results = []

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=0)
        recall = recall_score(y_test, y_pred, pos_label=0)
        f1 = f1_score(y_test, y_pred, pos_label=0)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (phishing): {precision:.4f}")
        print(f"Recall (phishing): {recall:.4f}")
        print(f"F1-Score (phishing): {f1:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"                 Predicted Phishing  Predicted Legitimate")
        print(f"Actual Phishing             {cm[0,0]:4d}                {cm[0,1]:4d}")
        print(f"Actual Legitimate           {cm[1,0]:4d}                {cm[1,1]:4d}")

        print(f"\nClassification Report:")
        report = classification_report(
            y_test, y_pred, labels=[0, 1], target_names=["phishing", "legitimate/legal"], digits=4
        )
        print(report)

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision_phishing": precision,
            "recall_phishing": recall,
            "f1_phishing": f1,
        })

    results_df = pd.DataFrame(results)
    return results_df
