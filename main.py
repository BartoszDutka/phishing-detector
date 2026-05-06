import os
from pathlib import Path

import pandas as pd

import kagglehub


DATASET = "ndarvind/phiusiil-phishing-url-dataset"
# If you know the exact file inside the dataset, put it here. Otherwise leave empty.
FILE_PATH = ""


def find_first_csv(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(".csv"):
                return Path(dirpath) / name
    return None


def load_df():
    cache_dir = Path("data") / "kagglehub"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Prefer local cached CSV (no network needed).
    csv_path = find_first_csv(cache_dir)
    if csv_path:
        return pd.read_csv(csv_path)

    if FILE_PATH:
        from kagglehub import KaggleDatasetAdapter

        return kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            DATASET,
            FILE_PATH,
        )

    # Download + read first CSV.
    local_dir = Path(kagglehub.dataset_download(DATASET, output_dir=str(cache_dir)))
    csv_path = find_first_csv(local_dir)
    if not csv_path:
        raise FileNotFoundError(f"No CSV found in: {local_dir}")
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    df = load_df()
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head())
