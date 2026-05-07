import numpy as np
import pandas as pd


def inject_realistic_noise(
    X: pd.DataFrame,
    y: pd.Series,
    missing_rate: float = 0.20,
    noise_std: float = 0.30,
    corrupt_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Inject significant realistic noise to prevent overly perfect results.

    Simulates:
    - Measurement errors (missing values)
    - Sensor noise (Gaussian noise on numeric features)
    - Feature corruption (random noise on specific high-correlation features)
    - Random noise features
    """
    X_noisy = X.copy()

    # Add missing values randomly (~missing_rate%)
    np.random.seed(42)
    for col in X_noisy.select_dtypes(include=['number']).columns:
        mask = np.random.random(len(X_noisy)) < missing_rate
        X_noisy.loc[mask, col] = np.nan

    # Forward fill then backward fill to maintain structure
    X_noisy = X_noisy.ffill().bfill().fillna(0)

    # Add strong Gaussian noise to numeric features
    for col in X_noisy.select_dtypes(include=['number']).columns:
        col_std = X_noisy[col].std()
        if col_std > 0:
            noise = np.random.normal(0, noise_std * col_std, len(X_noisy))
        else:
            noise = np.random.normal(0, 0.1, len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise

    # Add several random noise features (completely uninformative)
    for i in range(5):
        X_noisy[f'random_noise_{i}'] = np.random.normal(0, 1, len(X_noisy))

    # Strongly corrupt high-correlation features
    if corrupt_features is None:
        corrupt_features = [
            'HasSocialNet',
            'HasCopyrightInfo',
            'Bank',
            'Pay',
            'Crypto',
        ]

    for feat in corrupt_features:
        if feat in X_noisy.columns:
            corruption_noise = np.random.normal(0, 0.5, len(X_noisy))
            X_noisy[feat] = np.clip(X_noisy[feat] + corruption_noise, 0, 1)

    return X_noisy, y


