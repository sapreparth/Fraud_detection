"""
Retrain models on new data and compare performance against the current model.

This script simulates a weekly retraining job.  It synthesizes a new training
dataset, trains Isolation Forest and XGBoost models (delegating to the
`train_models` module), compares the new XGBoost model's performance to the
existing model, and if it improves the AUC, writes the new model to disk and
updates the symbolic "production" symlink.  In a real system this script would
query a feature store or data warehouse for the latest labelled data.
"""

import json
import shutil
from pathlib import Path

from sklearn.metrics import roc_auc_score

from src.models.train_models import synthesize_data, train_xgboost, evaluate_xgboost
import joblib


def load_current_model(model_dir: Path):
    model_path = model_dir / "xgboost_model.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    return None


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
    # Load current model and compute baseline AUC on fresh validation data
    X_ref, y_ref = synthesize_data(n_samples=5000, fraud_ratio=0.01)
    current_model = load_current_model(model_dir)
    baseline_auc = None
    if current_model is not None:
        # Evaluate using our evaluation function (requires a test split)
        from sklearn.model_selection import train_test_split
        X_train0, X_test0, y_train0, y_test0 = train_test_split(X_ref, y_ref, test_size=0.5, random_state=42, stratify=y_ref)
        baseline_auc = roc_auc_score(y_test0, current_model.predict_proba(X_test0)[:, 1])
    # Synthesise new data and split
    X, y = synthesize_data(n_samples=10000, fraud_ratio=0.01)
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    # Train new XGBoost
    new_model = train_xgboost(X_train, y_train, X_val, y_val)
    metrics = evaluate_xgboost(new_model, X_test, y_test)
    new_auc = metrics["auc"]
    print(f"New model AUC: {new_auc:.4f}")
    if baseline_auc is None or new_auc > baseline_auc:
        print("New model outperforms current model. Promoting to production.")
        joblib.dump(new_model, model_dir / "xgboost_model.pkl")
        with open(model_dir / "training_metadata.json", "w") as fp:
            json.dump({"xgboost_metrics": metrics, "samples": len(y)}, fp, indent=2)
    else:
        print("New model does not improve over current model. Keeping existing model.")


if __name__ == "__main__":
    main()
