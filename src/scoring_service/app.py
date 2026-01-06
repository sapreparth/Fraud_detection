"""
FastAPI scoring service for fraud detection.

This service exposes endpoints to score incoming feature vectors using pre‑trained
models (Isolation Forest or XGBoost).  Suspicious transactions (above a
threshold or with high anomaly scores) are persisted to a case management table
in PostgreSQL via SQLAlchemy.  A simple REST API allows listing open cases and
updating their status.

To run the service::

    uvicorn src.scoring_service.app:app --reload

"""

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func


# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cases.db")
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Case(Base):
    __tablename__ = "cases"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    timestamp = Column(String)
    amount = Column(Float)
    score = Column(Float)
    model = Column(String)
    status = Column(String, default="open")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# Create tables on startup
Base.metadata.create_all(bind=engine)


# Load models
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"
ISOLATION_MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"
XGB_MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"

if ISOLATION_MODEL_PATH.exists():
    isolation_model = joblib.load(ISOLATION_MODEL_PATH)
else:
    isolation_model = None
if XGB_MODEL_PATH.exists():
    xgb_model = joblib.load(XGB_MODEL_PATH)
else:
    xgb_model = None


class FeatureVector(BaseModel):
    user_id: str
    timestamp: str
    amount: float
    sum_5m: float
    count_5m: int
    sum_30m: float
    count_30m: int
    category: Optional[str] = None


app = FastAPI(title="Fraud Detection Scoring Service")


def score_isolation(features: FeatureVector) -> float:
    if isolation_model is None:
        raise HTTPException(status_code=500, detail="Isolation model not loaded")
    # Build feature array – Isolation Forest expects only the aggregated numeric features
    X = np.array([[features.amount, features.sum_5m, features.count_5m, features.sum_30m, features.count_30m]])
    score = -isolation_model.decision_function(X)[0]  # higher score → more anomalous
    return float(score)


def score_xgboost(features: FeatureVector) -> float:
    if xgb_model is None:
        raise HTTPException(status_code=500, detail="XGBoost model not loaded")
    # XGBoost expects the same numeric features
    X = np.array([[features.amount, features.sum_5m, features.count_5m, features.sum_30m, features.count_30m]])
    prob = xgb_model.predict_proba(X)[0, 1]
    return float(prob)


@app.post("/score")
def score_transaction(features: FeatureVector, model: str = "isolation", threshold: float = 0.5):
    """Score a feature vector and persist suspicious transactions."""
    if model == "isolation":
        score = score_isolation(features)
        is_suspicious = score > threshold  # for Isolation Forest we treat anomaly score > threshold as suspicious
    elif model == "xgboost":
        score = score_xgboost(features)
        is_suspicious = score > threshold  # probability > threshold triggers alert
    else:
        raise HTTPException(status_code=400, detail="Unknown model")
    # Persist if suspicious
    if is_suspicious:
        db = SessionLocal()
        case = Case(
            user_id=features.user_id,
            timestamp=features.timestamp,
            amount=features.amount,
            score=score,
            model=model,
            status="open",
        )
        db.add(case)
        db.commit()
        db.refresh(case)
        db.close()
    return {"score": score, "suspicious": is_suspicious}


@app.get("/cases")
def list_cases(status: Optional[str] = None):
    db = SessionLocal()
    query = db.query(Case)
    if status:
        query = query.filter(Case.status == status)
    results = query.all()
    cases = [
        {
            "id": c.id,
            "user_id": c.user_id,
            "timestamp": c.timestamp,
            "amount": c.amount,
            "score": c.score,
            "model": c.model,
            "status": c.status,
        }
        for c in results
    ]
    db.close()
    return cases


@app.post("/cases/{case_id}/update")
def update_case(case_id: int, status: str):
    db = SessionLocal()
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        db.close()
        raise HTTPException(status_code=404, detail="Case not found")
    case.status = status
    db.commit()
    db.refresh(case)
    db.close()
    return {"id": case.id, "status": case.status}
