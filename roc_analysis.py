import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def run_roc_auc_analysis(models, X_test, y_test, experiment_name="baseline", output_filename=None):
    """
    Compute ROC/AUC for phishing detection (class 0).
    Plots ROC curve for all models and saves to outputs/roc_curve_{experiment_name}.png.

    Parameters:
        models: Dict of trained models
        X_test: Test features
        y_test: Test labels
        experiment_name: Name of experiment (default: "baseline")
        output_filename: Custom output filename (if None, uses outputs/roc_curve_{experiment_name}.png)

    Returns:
        DataFrame: experiment, model, auc_phishing
    """
    os.makedirs("outputs", exist_ok=True)

    if output_filename is None:
        output_filename = f"outputs/roc_curve_{experiment_name}.png"

    results = []
    plt.figure(figsize=(10, 8))

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name} - ROC/AUC Analysis")
        print(f"{'='*60}")

        proba = model.predict_proba(X_test)

        # Get phishing class index (0 = phishing)
        classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["lr"].classes_
        phishing_class_index = list(classes).index(0)

        phishing_scores = proba[:, phishing_class_index]

        # Create binary target: 1 if phishing, 0 otherwise
        y_test_phishing = (y_test == 0).astype(int)

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test_phishing, phishing_scores)
        auc_score = roc_auc_score(y_test_phishing, phishing_scores)

        print(f"AUC (Phishing): {auc_score:.4f}")

        # Plot ROC curve
        plt.plot(
            fpr, tpr, label=f"{model_name} (AUC = {auc_score:.6f})", linewidth=2
        )

        results.append({
            "experiment": experiment_name,
            "model": model_name,
            "auc_phishing": auc_score,
        })

    # Plot random classifier baseline
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5000)", linewidth=2)

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve - Phishing Detection ({experiment_name})", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_filename, dpi=100, bbox_inches="tight")
    print(f"\nROC curve saved to {output_filename}")
    plt.close()

    results_df = pd.DataFrame(results)
    return results_df
