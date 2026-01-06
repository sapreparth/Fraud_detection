"""
train_models.py
---------------

Train Isolation Forest and XGBoost models on labelled transaction data.  For
demonstration purposes this script synthesizes a dataset with class imbalance
using `sklearn.datasets.make_classification`.  In a production deployment you
would replace this with your own feature engineering and training pipeline.

The script saves the trained models to `models/` and logs them to MLflow.  It
also exports a simple JSON metadata file with training statistics.
"""

import os
import json
import joblib
from pathlib import Path

import mlflow
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier


def synthesize_data(n_samples: int = 10000, fraud_ratio: float = 0.01, n_features: int = 20):
    """Generate an imbalanced classification dataset to simulate fraud."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[1 - fraud_ratio, fraud_ratio],
        flip_y=0.01,
        random_state=42,
    )
    return X, y


def train_isolation_forest(X_train):
    """Train an Isolation Forest on the training data."""
    clf = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    clf.fit(X_train)
    return clf


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train an XGBoost classifier and tune a few hyper‑parameters."""
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        eval_metric="auc",
        n_jobs=4,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def evaluate_xgboost(clf, X_test, y_test):
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)
    auc = roc_auc_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred)
    return {"auc": auc, "f1": f1}


def main() -> None:
    # Prepare data
    X, y = synthesize_data()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Create output directory
    model_dir = Path(__file__).resolve().parents[2] / "models"
    model_dir.mkdir(exist_ok=True)

    mlflow.set_experiment("fraud_detection")
    with mlflow.start_run(run_name="model_training"):
        # Train Isolation Forest (unsupervised)
        iso_clf = train_isolation_forest(X_train)
        iso_path = model_dir / "isolation_forest.pkl"
        joblib.dump(iso_clf, iso_path)
        mlflow.log_artifact(str(iso_path), artifact_path="models")

        # Train XGBoost (supervised)
        xgb_clf = train_xgboost(X_train, y_train, X_val, y_val)
        xgb_path = model_dir / "xgboost_model.pkl"
        joblib.dump(xgb_clf, xgb_path)
        mlflow.log_artifact(str(xgb_path), artifact_path="models")
        metrics = evaluate_xgboost(xgb_clf, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Save metrics to JSON
        meta = {"xgboost_metrics": metrics, "samples": len(y)}
        meta_path = model_dir / "training_metadata.json"
        with open(meta_path, "w") as fp:
            json.dump(meta, fp, indent=2)
        mlflow.log_artifact(str(meta_path), artifact_path="metadata")

    print("Models trained and saved to", model_dir)


if __name__ == "__main__":
    main()
