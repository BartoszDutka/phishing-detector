import pandas as pd


def deduplicate_and_split_safe(
    df: pd.DataFrame,
    url_col: str = "URL",
) -> pd.DataFrame:
    """Remove duplicate URLs and keep only first occurrence to prevent data leakage."""
    initial_rows = len(df)

    # Remove exact duplicates (full record)
    df_dedup = df.drop_duplicates()
    exact_dup_count = initial_rows - len(df_dedup)

    # Remove URL duplicates (keep first, remove rest)
    df_dedup = df_dedup.drop_duplicates(subset=[url_col], keep='first')
    url_dup_count = len(df) - exact_dup_count - len(df_dedup)

    final_rows = len(df_dedup)
    removed = initial_rows - final_rows

    print(f"\n=== Data Deduplication ===")
    print(f"Initial rows: {initial_rows}")
    print(f"Exact duplicates removed: {exact_dup_count}")
    print(f"URL duplicates removed: {url_dup_count}")
    print(f"Final rows: {final_rows} (removed {removed} total)")

    return df_dedup


def preprocess_df(
    df: pd.DataFrame,
    label_col: str = "label",
    drop_text_columns: bool = True,
    impute_missing: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    if label_col not in df.columns:
        raise KeyError(f"Missing label column: `{label_col}`")

    out = df.copy()

    # Ensure label is {0,1} with 1=legitimate/legal, 0=phishing
    out = out.dropna(subset=[label_col])
    y_raw = out[label_col]

    # If it's already numeric 0/1, keep. Otherwise map common string labels.
    if not pd.api.types.is_numeric_dtype(y_raw):
        y_norm = y_raw.astype(str).str.strip().str.lower()
        mapping = {
            "phishing": 0,
            "phish": 0,
            "1": 1,
            "true": 1,
            "yes": 1,
            "legal": 1,
            "legit": 1,
            "legitimate": 1,
            "0": 0,
            "false": 0,
            "no": 0,
        }
        y = y_norm.map(mapping)
        if y.isna().any():
            bad = sorted(set(y_norm[y.isna()].unique().tolist()))
            raise ValueError(
                f"Unrecognized labels in `{label_col}` (cannot map to 0/1): {bad[:20]}"
            )
        y = y.astype("int64")
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        uniques = set(y.dropna().unique().tolist())
        if not uniques.issubset({0, 1}):
            raise ValueError(f"`{label_col}` must be binary 0/1. Found values: {sorted(uniques)}")
        y = y.astype("int64")

    # Build feature matrix
    X = out.drop(columns=[label_col])

    if drop_text_columns:
        # Keep only numeric columns (model-friendly without encoding)
        X = X.select_dtypes(include=["number"]).copy()

    if impute_missing:
        # Fill numeric NaNs with median (robust default)
        for col in X.columns:
            if X[col].isna().any():
                med = X[col].median()
                if pd.isna(med):
                    # all-NaN column -> fill with 0 then keep (or later drop if desired)
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(med)

    # Final safety check
    if X.isna().any().any():
        still_missing = X.isna().sum().sort_values(ascending=False)
        still_missing = still_missing[still_missing > 0].head(20)
        raise ValueError(f"Missing values remain after preprocessing (top columns):\n{still_missing}")

    return X, y

