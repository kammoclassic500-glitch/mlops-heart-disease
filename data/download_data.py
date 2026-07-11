"""
Data Acquisition
-------------------------
Usage:
    pip install ucimlrepo pandas
    python data/download_data.py
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_heart_disease(output_path: str = "data/heart_disease_raw.csv") -> pd.DataFrame:

    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets

    df = pd.concat([X, y], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {df.shape[0]} rows, {df.shape[1]} columns -> {output_path}")
    print("\nMetadata:")
    print(heart_disease.metadata.name, "|", heart_disease.metadata.repository_url)
    return df


if __name__ == "__main__":
    download_heart_disease()
