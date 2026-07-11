"""
Data Cleaning + Feature Engineering
------------------------------------------------
Shared, reusable preprocessing logic used by:
  - the training script (src/train.py)
  - the FastAPI service (api/main.py)
  - the unit tests (tests/test_preprocessing.py)
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Raw UCI column names (dataset id=45). "num" is the original 0-4 severity target.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "num"


def load_raw_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
      - drop exact duplicate rows
      - coerce feature columns to numeric (UCI sometimes stores '?' for NaN)
      - binarize the target: 0 = no disease, 1 = disease (original values 1-4 -> 1)
    Missing values themselves are handled later inside the sklearn Pipeline
    (SimpleImputer) so that the same logic runs identically at inference time.
    """
    df = df.drop_duplicates().copy()

    for col in ALL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET_COL in df.columns:
        df["target"] = (df[TARGET_COL] > 0).astype(int)
    elif "target" not in df.columns:
        raise ValueError("Expected a 'num' or 'target' column in the dataset.")

    return df


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Returns an sklearn ColumnTransformer:
      - numeric features -> median impute + standard scale
      - categorical features -> most-frequent impute + one-hot encode
    This object gets saved alongside the model (see src/train.py) so that
    raw JSON sent to the API can be transformed exactly like training data.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def split_features_target(df: pd.DataFrame):
    X = df[ALL_FEATURES]
    y = df["target"]
    return X, y
