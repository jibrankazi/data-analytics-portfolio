"""Real anonymized credit-card fraud benchmark, OpenML id=1597 (ULB/Worldline).

Full source -> validation-selected model/threshold -> untouched test set.
OpenML edition lacks original Time field, so splits are *stratified random*,
not chronological; this is historical research, not live fraud monitoring.
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
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OPENML_DATA_ID = 1597
ORIGINAL_SOURCE = "https://www.openml.org/d/1597"
ORIGINAL_RESEARCH = "Worldline and ULB Machine Learning Group, 2013 cardholders"


def actual_ulb_dataset():
    dataset = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
    source = dataset.frame.copy()
    if "Class" not in source:
        raise ValueError("Official OpenML 1597 response missing Class fraud label")
    if len(source) != 284807:
        raise ValueError(f"Unexpected official ULB observation count: {len(source)}")
    labels = pd.to_numeric(source.pop("Class"), errors="raise").astype("int8")
    if not labels.isin([0, 1]).all() or int(labels.sum()) != 492:
        raise ValueError("ULB source label contract failed (492 original fraud cases)")
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    if any(f not in source.columns for f in feature_names):
        raise ValueError("OpenML ULB published PCA/Amount schema missing")
    # The original Kaggle download contains Time, but OpenML's mirror often
    # omits it. Do not quietly describe random splitting as time-ordered.
    extra = set(source.columns) - set(feature_names)
    if extra - {"Time"}:
        raise ValueError(f"Unexpected unlabeled source features: {sorted(extra)}")
    X = source[feature_names].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy(dtype="float64")).all():
        raise ValueError("Original financial transaction input has missing/infinite features")
    if not X["Amount"].ge(0).all():
        raise ValueError("Impossible negative transaction amount in dataset")
    # Prevent exact same anonymized observation from being in both partitions.
    repeated = X.duplicated(keep="first")
    duplicate_rows = int(repeated.sum())
    clean_X = X.loc[~repeated].reset_index(drop=True)
    clean_y = labels.loc[~repeated].reset_index(drop=True)
    if len(clean_X) < 280000 or clean_y.sum() < 450:
        raise ValueError("Deduplication excluded more actual records than expected")
    # Fingerprint the exact ordered floating-point source for audit; do not
    # claim an original network-payload SHA when using fetch_openml's parser.
    values = X.to_numpy(dtype="<f8", copy=True)
    digest = sha256(values.tobytes() + labels.to_numpy(dtype="int8").tobytes()).hexdigest()
    return clean_X, clean_y, {"original_rows": int(len(X)),
                             "original_positive_labels": 492,
                             "duplicate_feature_rows_removed": duplicate_rows,
                             "source_numeric_values_sha256": digest,
                             "deduplicated_rows": int(len(clean_X)),
                             "deduplicated_fraud_labels": int(clean_y.sum()),
                             "features": feature_names,
                             "time_field_available_from_openml": "Time" in source}


def validation_threshold(y, probability):
    # Deterministically maximize F1 on validation only. No test-set tuning.
    prec, recall, thresholds = precision_recall_curve(y, probability)
    if len(thresholds) < 1:
        raise ValueError("Insufficient validation score variation")
    f1s = 2 * prec[:-1] * recall[:-1] / np.maximum(prec[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1s))
    return float(thresholds[best]), float(f1s[best])


def metric_summary(y, probs, threshold):
    predictions = (probs >= threshold).astype("int8")
    return {
        "roc_auc": float(roc_auc_score(y, probs)),
        "average_precision_pr_auc": float(average_precision_score(y, probs)),
        "precision_fraud": float(precision_score(y, predictions, zero_division=0)),
        "recall_fraud": float(recall_score(y, predictions, zero_division=0)),
        "f1_fraud": float(f1_score(y, predictions, zero_division=0)),
        "confusion_matrix_true_0_1": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
    }


def run(output="fraud-detection-finance/results/ulb_real_fraud"):
    X, y, provenance = actual_ulb_dataset()
    source_rows = np.arange(len(X))
    train_val_idx, test_idx = train_test_split(
        source_rows, test_size=.2, random_state=42, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=.25, random_state=42, stratify=y.iloc[train_val_idx]
    )
    if set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
        raise ValueError("Train/test partitions overlap")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    models = {
        "class_prior_baseline": DummyClassifier(strategy="prior"),
        "balanced_logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(
                class_weight="balanced", max_iter=500, random_state=42)
        ),
        "histogram_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=130, learning_rate=.08, max_leaf_nodes=15,
            l2_regularization=1.0, min_samples_leaf=25,
            random_state=42, early_stopping=True),
    }
    fitted = {}
    for name, estimator in models.items():
        print(f"Training actual ULB research classifier: {name}", flush=True)
        estimator.fit(X_train, y_train)
        val_proba = estimator.predict_proba(X_val)[:, 1]
        validation_ap = float(average_precision_score(y_val, val_proba))
        if name == "class_prior_baseline":
            threshold, val_f1 = .5, 0.
        else:
            threshold, val_f1 = validation_threshold(y_val, val_proba)
        fitted[name] = (estimator, threshold, validation_ap, val_f1)
    selected = max(("balanced_logistic_regression", "histogram_gradient_boosting"),
                   key=lambda key: fitted[key][2])
    results = {}
    test_predictions = {}
    for name, (estimator, threshold, validation_ap, validation_f1) in fitted.items():
        proba = estimator.predict_proba(X_test)[:, 1]
        if not np.isfinite(proba).all() or not np.logical_and(proba >= 0, proba <= 1).all():
            raise ValueError("Invalid probabilities in fraud model")
        results[name] = {
            "threshold_selected_on_validation_only": threshold,
            "validation_average_precision": validation_ap,
            "validation_f1_at_selected_threshold": validation_f1,
            "untouched_test": metric_summary(y_test, proba, threshold),
        }
        test_predictions[name] = proba
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_scoring = pd.DataFrame({
        "deduplicated_original_row_position": test_idx,
        "original_fraud_label": y_test.to_numpy(),
    })
    for name, proba in test_predictions.items():
        test_scoring[f"probability_{name}"] = proba
    test_scoring.sort_values("deduplicated_original_row_position").to_csv(
        output_dir / "heldout_real_ulb_probabilities.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, proba in test_predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ax.plot(recall, precision,
                label=f"{name}: AP {results[name]['untouched_test']['average_precision_pr_auc']:.3f}")
    ax.set(title="Real anonymized ULB 2013 fraud labels: untouched test PR curves",
           xlabel="Fraud recall", ylabel="Fraud precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / "heldout_precision_recall.png", dpi=150)
    plt.close(fig)
    report = {
        "publisher": ORIGINAL_RESEARCH,
        "openml_dataset_id": OPENML_DATA_ID,
        "source_url": ORIGINAL_SOURCE,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "data_contract": provenance,
        "split": {
            "method": "60/20/20 stratified random after original-source feature deduplication",
            "why_not_chronological": "OpenML 1597 version lacks the original Time feature",
            "train_rows": len(train_idx), "validation_rows": len(val_idx),
            "test_rows": len(test_idx), "train_fraud": int(y_train.sum()),
            "validation_fraud": int(y_val.sum()), "test_fraud": int(y_test.sum()),
            "seed": 42,
        },
        "threshold_selection": "Validation partition only; maximize F1 independently",
        "model_selection": "Maximum validation average precision among the two fitted alternatives, not test metrics",
        "selected_model": selected,
        "models": results,
        "limitations": (
            "Genuine but historical 2013 anonymized fraud cases (not PaySim). "
            "Stratified random split is not forward-time validation; no future "
            "drift, time, cardholder identity, financial cost weighting, "
            "institutional permissions or live production controls measured."
        ),
    }
    (output_dir / "verified_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--output", default="fraud-detection-finance/results/ulb_real_fraud")
    run(args.parse_args().output)
