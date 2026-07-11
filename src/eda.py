"""
Exploratory Data Analysis (EDA)
-----------------------------------------
Usage:
    python src/eda.py --data data/heart_disease_raw.csv
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import ALL_FEATURES, NUMERIC_FEATURES, clean_data, load_raw_csv

OUT_DIR = "reports/eda"


def run_eda(data_path: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = clean_data(load_raw_csv(data_path))

    # 1. Class balance
    plt.figure(figsize=(5, 4))
    sns.countplot(x="target", data=df)
    plt.title("Class Balance: Heart Disease Presence (0=No, 1=Yes)")
    plt.savefig(f"{OUT_DIR}/class_balance.png", bbox_inches="tight")
    plt.close()

    # 2. Histograms of numeric features
    df[NUMERIC_FEATURES].hist(figsize=(12, 8), bins=20)
    plt.suptitle("Distributions of Numeric Features")
    plt.savefig(f"{OUT_DIR}/histograms.png", bbox_inches="tight")
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df[ALL_FEATURES + ["target"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    # 4. Missing values summary
    missing = df[ALL_FEATURES].isna().sum()
    print("Missing values per column:\n", missing)

    # 5. Basic descriptive stats -> saved as CSV for the report
    df[NUMERIC_FEATURES].describe().to_csv(f"{OUT_DIR}/summary_stats.csv")

    print(f"EDA plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heart_disease_raw.csv")
    args = parser.parse_args()
    run_eda(args.data)
