"""Run a reproducible IBM Telco *illustrative-sample* churn analysis.

This is not real customer churn from a bank or live telecom provider.
Sample dataset provenance is documented; outputs are held-out empirical
results on the actual bundled IBM dataset, never invented performance.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "bank-churn-analysis/data/Telco-Customer-Churn.csv"
PUBLISHER = "IBM publicly distributed Telco Customer Churn example dataset"
PUBLISHER_URL = "https://github.com/IBM/telco-customer-churn-on-icp4d"
NUMERIC = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")


def load_sample(path=SOURCE):
    raw = Path(path).read_bytes()
    if len(raw) < 100_000:
        raise ValueError("Missing or truncated IBM Telco example")
    df = pd.read_csv(path, dtype={"customerID": "string"})
    required = set(NUMERIC) | {"customerID", "Churn", "gender", "Contract", "PaymentMethod"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing IBM Telco fields: {required - set(df.columns)}")
    if df.shape[0] != 7043 or not df.customerID.is_unique or df.customerID.isna().any():
        raise ValueError("Original IBM published sample should have 7043 unique row IDs")
    if not df.Churn.isin(["Yes", "No"]).all():
        raise ValueError("Unexpected churn label; do not relabel missing values")
    df = df.copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[list(NUMERIC)].isna().all(axis=1).any():
        raise ValueError("Entirely missing numeric feature row")
    if df["tenure"].lt(0).any() or df["MonthlyCharges"].le(0).any():
        raise ValueError("Implausible sample tenure or billed amount")
    return df, sha256(raw).hexdigest()


def build_estimator(base_model, X):
    number = list(NUMERIC)
    category = [c for c in X.columns if c not in number]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), number),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), category),
    ])
    return Pipeline([("prepare", preprocessor), ("classify", base_model)])


def run(output):
    df, raw_sha = load_sample()
    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    X = df.drop(columns=["customerID", "Churn"])
    if X.shape[1] < 15:
        raise ValueError("Unexpectedly small IBM churn feature set")
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    models = {
        "Prior-probability baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(max_iter=1200, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10,
                                                min_samples_leaf=3, random_state=42,
                                                n_jobs=1),
    }
    metrics = {}
    curves = {}
    for name, model in models.items():
        classifier = build_estimator(model, train_x)
        classifier.fit(train_x, train_y)
        predictions = classifier.predict(test_x)
        probs = classifier.predict_proba(test_x)[:, 1]
        if not np.isfinite(probs).all():
            raise ValueError("Non-finite churn model probabilities")
        fpr, tpr, _ = roc_curve(test_y, probs)
        curves[name] = (fpr, tpr)
        metrics[name] = {
            "holdout_accuracy": float(accuracy_score(test_y, predictions)),
            "holdout_balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
            "holdout_roc_auc": float(roc_auc_score(test_y, probs)),
            "holdout_average_precision": float(average_precision_score(test_y, probs)),
            "confusion_matrix_0_1": confusion_matrix(test_y, predictions, labels=[0, 1]).tolist(),
        }
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} AUC={metrics[name]['holdout_roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="IBM illustrative Telco Churn sample: unseen held-out rows")
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    fig.savefig(folder / "holdout_roc.png", dpi=150)
    plt.close(fig)
    result = {
        "dataset_publisher": PUBLISHER,
        "publisher_url": PUBLISHER_URL,
        "data_origin": "Bundled public IBM demonstration/example CSV, not verified actual customers",
        "source_path": SOURCE.relative_to(ROOT).as_posix(),
        "sha256_bundled_source_csv": raw_sha,
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(df)),
        "features_excluding_id": int(X.shape[1]),
        "positive_churn_rows": int(y.sum()),
        "numeric_missing_cells": int(df[list(NUMERIC)].isna().sum().sum()),
        "split": {"train_rows": int(len(train_x)), "test_rows": int(len(test_x)),
                  "heldout_churn_rows": int(test_y.sum()), "seed": 42, "stratified": True},
        "leakage_controls": "Exclude customerID; fit imputation, encoders and scalers only in train fold",
        "models": metrics,
        "limitations": "Illustrative public telecom sample only; no real bank/customer database, no prospective outcomes, calibration assessment, external validity or deployment",
    }
    (folder / "verified_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="bank-churn-analysis/results")
    run(parser.parse_args().output)
