import pandas as pd


def analyze_feature_importance(models: dict, X_train: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Extract feature importance from Random Forest model."""
    print("\n=== Feature Importance Analysis ===")

    rf_model = models.get("Random Forest")
    if rf_model is None:
        print("Random Forest model not found.")
        return pd.DataFrame()

    importances = rf_model.feature_importances_
    feature_names = X_train.columns

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print(f"\nTop {top_n} Features:")
    for idx, row in importance_df.head(top_n).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.6f}")

    return importance_df
