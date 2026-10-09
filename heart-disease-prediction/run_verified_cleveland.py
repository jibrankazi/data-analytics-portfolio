"""End-to-end, provenance-checked research benchmark on genuine UCI Cleveland data.

This is historical, small-sample model research, not a clinical decision tool.
It downloads the official published input every run and does not synthesize rows.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SOURCE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
CATALOG = "https://archive.ics.uci.edu/dataset/45/heart+disease"
COLUMNS = ("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num")


def parse_official_bytes(raw: bytes):
    """Reject empty, partial, altered-schema or implausible inputs."""
    if len(raw) < 8000:
        raise ValueError("Empty or incomplete UCI data download")
    df = pd.read_csv(BytesIO(raw), header=None, names=COLUMNS, na_values=["?"],
                     skipinitialspace=True)
    if df.shape != (303, 14):
        raise ValueError(f"Expected 303 original Cleveland rows x 14 columns, got {df.shape}")
    for column in COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")
    if df["num"].isna().any() or not df["num"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Original UCI Cleveland target codes must be 0 through 4")
    if not df["age"].between(18, 110).all() or not df["sex"].isin([0, 1]).all():
        raise ValueError("Unexpected original UCI clinical field values")
    if df.drop(columns=["num"]).isna().all(axis=0).any():
        raise ValueError("An original data feature is wholly missing")
    if df.drop_duplicates().shape[0] < 290:
        raise ValueError("Too many duplicated data rows; possible malformed source")
    return df


def get_observed_source():
    request = Request(SOURCE, headers={"User-Agent": "AcademicPortfolioReproduction/1.0"})
    with urlopen(request, timeout=35) as response:
        raw = response.read()
    return parse_official_bytes(raw), raw


def run(output: str):
    observations, raw = get_observed_source()
    features = observations.drop(columns="num")
    # Original num=0 means no disease; values 1-4 indicate presence.
    labels = (observations["num"] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.30, random_state=42, stratify=labels
    )
    candidates = {
        "Prior-prevalence baseline": DummyClassifier(strategy="prior"),
        "KNN k=5": Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)),
        ]),
    }
    scores = {}
    curves = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        if not np.isfinite(probability).all() or not np.logical_and(
            probability >= 0, probability <= 1
        ).all():
            raise ValueError(f"Invalid predicted probabilities for {name}")
        fpr, tpr, _ = roc_curve(y_test, probability)
        curves[name] = (fpr, tpr)
        scores[name] = {
            "test_accuracy": float(accuracy_score(y_test, prediction)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
            "test_roc_auc": float(roc_auc_score(y_test, probability)),
            "test_average_precision": float(average_precision_score(y_test, probability)),
            "test_confusion_matrix": confusion_matrix(
                y_test, prediction, labels=[0, 1]
            ).astype(int).tolist(),
        }
    if len(y_train) + len(y_test) != 303 or y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Holdout dataset size or stratification invalid")
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=f"{name}: AUC {scores[name]['test_roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="UCI Cleveland historical held-out ROC")
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    fig.savefig(target / "heldout_roc.png", dpi=150)
    plt.close(fig)
    results = {
        "source": SOURCE,
        "catalog": CATALOG,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": sha256(raw).hexdigest(),
        "dataset": "Original processed Cleveland research observations",
        "original_records": int(len(observations)),
        "input_features": int(features.shape[1]),
        "missing_feature_cells": int(features.isna().sum().sum()),
        "positive_original_records": int(labels.sum()),
        "split": {"method": "stratified single holdout; random seed 42",
                  "train_rows": int(len(X_train)),
                  "test_rows": int(len(X_test)),
                  "test_positive_cases": int(y_test.sum())},
        "preprocessing": "Training-only most-frequent imputation; KNN scales training data only",
        "model_metrics": scores,
        "limitations": (
            "Historical single-center research sample; small one-time split; "
            "no external validation, confidence intervals, prospective trial, "
            "diagnostic deployment or clinical decision support."
        ),
    }
    (target / "verified_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, allow_nan=False))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="heart-disease-prediction/results")
    args = parser.parse_args()
    run(args.output)
