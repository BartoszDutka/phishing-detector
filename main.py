from config import DEFAULT_LABEL_COL, DEFAULT_URL_COL
from data_io import load_df
from eda import quick_eda
from features import prepare_features
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    df = load_df()
    print("head():")
    print(df.head())
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    quick_eda(df, label_col=DEFAULT_LABEL_COL)

    X, y = prepare_features(df, label_col=DEFAULT_LABEL_COL, url_col=DEFAULT_URL_COL)
    print("\n=== Features ready ===")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X columns (first 30):", list(X.columns[:30]))
    print("y distribution:", y.value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print("\n=== Train/test split (80/20, stratified) ===")
    print("X_train shape:", X_train.shape, "| y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape, "| y_test shape:", y_test.shape)
    print("y_train distribution:", y_train.value_counts().to_dict())
    print("y_test distribution:", y_test.value_counts().to_dict())
