import pandas as pd


def select_url_only_features(X: pd.DataFrame) -> pd.DataFrame:
    """Keep ONLY basic URL structure features per requirements."""

    # Only features explicitly required: URL length, IP presence, subdomains, TLD
    url_only_features = [
        'URLLength',
        'DomainLength',
        'IsDomainIP',
        'TLDLength',
        'NoOfSubDomain',
        'IsHTTPS',
        'CharContinuationRate',
        'HasObfuscation',
        'NoOfObfuscatedChar',
        'ObfuscationRatio',
    ]

    features_present = [f for f in url_only_features if f in X.columns]
    removed = set(X.columns) - set(features_present)

    print(f"\n=== REQUIREMENT: URL-Only Features ===")
    print(f"Keeping {len(features_present)} URL features: {features_present}")
    print(f"Removed {len(removed)} non-URL features (website content, analysis, etc)")

    return X[features_present].copy()

