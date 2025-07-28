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
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Optional: import XGBoost if available
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def load_paysim_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series, ColumnTransformer]:
    y = df['isFraud']
    X = df.drop(columns=['isFraud', 'isFlaggedFraud'])
    cat_cols = ['type']
    num_cols = [col for col in X.columns if col not in cat_cols and X[col].dtype in [np.float64, np.int64]]

    preprocessor = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('scale', StandardScaler(), num_cols)
    ])

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor

def balance_data(X, y, method: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
    if method == 'smote':
        sampler = SMOTE(random_state=42)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError("method must be 'smote' or 'undersample'")

    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train):
    if not HAS_XGB:
        raise RuntimeError("XGBoost is not installed. Install xgboost to use this model.")

    xgb = XGBClassifier(
        n_estimators=300,
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

def plot_evaluation(y_test, y_pred_proba, model_name: str) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='grey')
    axes[0].set_title(f'ROC Curve ({model_name})')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()

    axes[1].plot(recall, precision, label=f'AP = {pr_auc:.3f}')
    axes[1].set_title(f'Precision–Recall Curve ({model_name})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()

    plt.tight_layout()
    images_dir = os.path.abspath(os.path.join("images"))
    os.makedirs(images_dir, exist_ok=True)
    path = os.path.join(images_dir, f'{model_name.lower()}_evaluation.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ Saved evaluation plots for {model_name} to {path}")

def main():
    file_path = r"C:\\Users\\jibra\\Downloads\\PS_20174392719_1491204439457_log.csv 2\\PS_20174392719_1491204439457_log.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = load_paysim_data(file_path)
    print(f"📥 Loaded PaySim data with shape {df.shape}")

    X, y, _ = preprocess_data(df)
    X_bal, y_bal = balance_data(X, y, method='smote')

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    rf = train_random_forest(X_train, y_train)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    plot_evaluation(y_test, y_proba_rf, model_name='RandomForest')

    if HAS_XGB:
        xgb = train_xgboost(X_train, y_train)
        y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
        plot_evaluation(y_test, y_proba_xgb, model_name='XGBoost')
    else:
        print("⚠️ XGBoost not available; skipping XGBoost model.")

if __name__ == '__main__':
    main()
