"""
Model Development + Experiment Tracking + Packaging
------------------------------------------------------------------
Trains Logistic Regression and Random Forest, evaluates both with
cross-validation, logs everything to MLflow, and saves the final
(best) model + preprocessing pipeline as reusable joblib artifacts.

Usage:
    python src/train.py --data data/heart_disease_raw.csv

MLflow UI:
    mlflow ui --backend-store-uri ./mlruns
"""

import argparse
import os
import sys

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import build_preprocessing_pipeline, clean_data, load_raw_csv, split_features_target

MODEL_DIR = "models"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = ["accuracy", "precision", "recall", "roc_auc"]


def get_candidates():
    """Two model families + a small hyperparameter grid each (tuning)."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {"model__n_estimators": [100, 300], "model__max_depth": [4, 8, None]},
        ),
    }


def evaluate_and_log(name, pipeline, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=name):
        cv_results = cross_validate(pipeline, X_train, y_train, cv=CV, scoring=SCORING)
        for metric in SCORING:
            mlflow.log_metric(f"cv_{metric}_mean", cv_results[f"test_{metric}"].mean())

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        test_accuracy = accuracy_score(y_test, preds)
        test_precision = precision_score(y_test, preds)
        test_recall = recall_score(y_test, preds)
        test_roc_auc = roc_auc_score(y_test, probs)

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_params(pipeline.named_steps["model"].get_params())

        os.makedirs("reports", exist_ok=True)
        fig_path = f"reports/roc_{name}.png"
        RocCurveDisplay.from_predictions(y_test, probs)
        plt.title(f"ROC Curve - {name}")
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(fig_path)

        mlflow.sklearn.log_model(pipeline, artifact_path="model", serialization_format="cloudpickle")

        print(f"[{name}] test_accuracy={test_accuracy:.3f} roc_auc={test_roc_auc:.3f}")
        return test_roc_auc, pipeline


def main(data_path: str):
    mlflow.set_experiment("heart-disease-risk")

    df = clean_data(load_raw_csv(data_path))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_score, best_pipeline, best_name = -1, None, None

    for name, (model, param_grid) in get_candidates().items():
        base_pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessing_pipeline()),
            ("model", model),
        ])
        search = GridSearchCV(base_pipeline, param_grid, cv=CV, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        tuned_pipeline = search.best_estimator_

        score, fitted_pipeline = evaluate_and_log(name, tuned_pipeline, X_train, y_train, X_test, y_test)
        if score > best_score:
            best_score, best_pipeline, best_name = score, fitted_pipeline, name

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "heart_disease_model.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"\nBest model: {best_name} (roc_auc={best_score:.3f}) -> saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heart_disease_raw.csv")
    args = parser.parse_args()
    main(args.data)
