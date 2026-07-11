import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import clean_data, build_preprocessing_pipeline, split_features_target


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "age": [63, 45, None],
        "sex": [1, 0, 1],
        "cp": [3, 2, 1],
        "trestbps": [145, 130, 120],
        "chol": [233, 250, 199],
        "fbs": [1, 0, 0],
        "restecg": [0, 1, 0],
        "thalach": [150, 187, 172],
        "exang": [0, 0, 1],
        "oldpeak": [2.3, 3.5, 0.0],
        "slope": [0, 0, 2],
        "ca": [0, 0, 1],
        "thal": [1, 2, 2],
        "num": [1, 0, 2],
    })


def test_clean_data_binarizes_target(raw_df):
    df = clean_data(raw_df)
    assert set(df["target"].unique()).issubset({0, 1})
    # original num=1 -> target 1, num=0 -> target 0, num=2 -> target 1
    assert df["target"].tolist() == [1, 0, 1]


def test_clean_data_drops_duplicates(raw_df):
    dup_df = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
    cleaned = clean_data(dup_df)
    assert cleaned.shape[0] == raw_df.shape[0] + 999


def test_clean_data_requires_target_column():
    df = pd.DataFrame({"age": [1, 2]})
    with pytest.raises(ValueError):
        clean_data(df)


def test_preprocessing_pipeline_handles_missing_values(raw_df):
    df = clean_data(raw_df)
    X, y = split_features_target(df)
    pipeline = build_preprocessing_pipeline()
    Xt = pipeline.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
    assert not pd.isna(Xt).any()
