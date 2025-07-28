"""
Financial Fraud Detection
========================

This script builds and evaluates fraud detection models using the
PaySim mobile money transaction dataset. The dataset includes
simulated transactions that mimic real mobile money operations with
different transaction types and a fraud flag. Because fraudulent
events are extremely rare, the script demonstrates techniques for
dealing with class imbalance before training classification models.

Steps covered:

1. **Load Data** –  Read the PaySim CSV and inspect the basic
distribution of the target class.
2. **Preprocessing** –  Encode categorical transaction types, scale
numerical features and split the data into training and test sets.
3. **Handle Class Imbalance** –  Apply either SMOTE (synthetic
minority oversampling) or random undersampling.  Both strategies
are available via the `imblearn` library.
4. **Model Training** –  Fit a Random Forest and an XGBoost classifier
on the balanced training data.
5. **Evaluation** –  Generate confusion matrices, ROC–AUC and
precision–recall curves.  Save the plots to the `images` folder.
6. **Feature Importance** –  Plot feature importances from the tree
models to understand which variables contribute most to fraud
detection.

Note that training on the full PaySim dataset (millions of rows) can
be computationally expensive.  Feel free to sample the data or use a
subset for prototyping.
"""

import os
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ✅ Direct import of XGBoost
from xgboost import XGBClassifier

# Load PaySim data
def load_paysim_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

# Preprocessing: encode + scale
def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series, ColumnTransformer]:
    y = df['isFraud']
    X = df.drop(columns=['isFraud', 'isFlaggedFraud'])

    cat_cols = ['type']
    num_cols = [col for col in X.columns if col not in cat_cols and np.issubdtype(X[col].dtype, np.number)]

    preprocessor = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('scale', StandardScaler(), num_cols)
    ])

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor

# Balance with SMOTE
def balance_data(X, y) -> Tuple[np.ndarray, np.ndarray]:
    sampler = SMOTE(random_state=42)
    return sampler.fit_resample(X, y)

# Train RandomForest
def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

# Train XGBoost
def train_xgboost(X_train, y_train):
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr'
    )
    xgb.fit(X_train, y_train)
    return xgb

# Evaluation with plots
def plot_evaluation(y_test, y_pred, y_pred_proba, model_name: str):
    roc_auc = auc(*roc_curve(y_test, y_pred_proba)[:2])
    pr_auc = average_precision_score(y_test, y_pred_proba)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axs[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axs[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axs[0].set_title(f"ROC Curve ({model_name})")
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].legend()

    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    axs[1].plot(recall, precision, label=f"AP = {pr_auc:.3f}")
    axs[1].set_title(f"Precision–Recall Curve ({model_name})")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].legend()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(ax=axs[2], cmap="Blues")
    axs[2].set_title(f"Confusion Matrix ({model_name})")

    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    path = os.path.join("images", f"{model_name.lower()}_evaluation.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"✅ {model_name} Evaluation Complete | AUC: {roc_auc:.3f}, AP: {pr_auc:.3f}")
    print(f"📁 Saved plots to: {path}")

# Main function
def main():
    file_path = r"C:\Users\jibra\Downloads\PS_20174392719_1491204439457_log.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print("📥 Loading data...")
    df = load_paysim_data(file_path)
    df = df.sample(n=100_000, random_state=42)
    print(f"✅ Sampled shape: {df.shape}")

    print("🔄 Preprocessing...")
    X, y, _ = preprocess_data(df)

    print("⚖️ Balancing with SMOTE...")
    X_bal, y_bal = balance_data(X, y)

    print("📚 Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal)

    print("🌲 Training RandomForest...")
    rf = train_random_forest(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    plot_evaluation(y_test, y_pred_rf, y_proba_rf, model_name="RandomForest")

    print("🚀 Training XGBoost...")
    xgb = train_xgboost(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    plot_evaluation(y_test, y_pred_xgb, y_proba_xgb, model_name="XGBoost")

    print("✅ All tasks completed.")

# Run
main()
