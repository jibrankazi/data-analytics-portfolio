"""
Heart Disease Risk Prediction
==================================

This script performs a complete exploratory data analysis (EDA) and
classification workflow on the UCI Cleveland heart‑disease data set.  The
goal is to predict the presence of heart disease using a variety of
physiological and diagnostic measurements.  Throughout the analysis I
focus on clarity and reproducibility, documenting each step so that
another analyst could follow along and understand the decision making
process.

Key steps in this notebook:

1. Download and parse the raw ``processed.cleveland.data`` file from the
   StatQuest GitHub repository.  The original UCI site can be fragile,
   but the raw file hosted on GitHub is reliable and easier to access
   programmatically.
2. Clean the data by assigning column names, converting strings to
   numerical types and handling missing values in ``ca``, ``thal`` and
   ``slope``.  Missing values are encoded as ``?`` in the raw data.
3. Perform a quick EDA to understand the distribution of features and
   relationships with the target variable.  Basic summary statistics and
   correlation matrices are produced to guide model building.
4. Split the data into training and test sets, build two classifiers
   (K‑Nearest Neighbours and Random Forest), evaluate them using
   accuracy, confusion matrices and ROC curves.  Feature importance is
   extracted from the Random Forest model.
5. Save key visualisations (confusion matrices, ROC curves and
   feature importances) to the ``images`` directory for inclusion in
   reports or presentations.

Running this script will download the data and produce output in the
``images`` folder.  Make sure the required Python packages are
installed (see the README for dependencies).
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, RocCurveDisplay)


def download_heart_data(url: str, dest: str) -> None:
    """Download the heart disease dataset from a raw GitHub URL.

    Parameters
    ----------
    url : str
        Direct link to the raw ``processed.cleveland.data`` file.  The
        StatQuest repository hosts a copy that is accessible without
        authentication.
    dest : str
        Destination filepath where the downloaded file should be saved.
    """
    if not os.path.exists(dest):
        print(f"Downloading heart disease data from {url}…")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved data to {dest}")
    else:
        print(f"Data already downloaded at {dest}")


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load the Cleveland heart disease dataset and perform basic cleaning.

    The raw file lacks column headers and represents missing values
    as ``?``.  After loading the data we assign descriptive column
    names, convert the data to numeric types and replace missing
    values with ``NaN``.  Finally, we coerce the target variable to a
    binary indicator: 0 for absence of disease and 1 for presence (any
    value >0 in the ``num`` column indicates heart disease).

    Parameters
    ----------
    filepath : str
        Path to the downloaded ``processed.cleveland.data`` file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for analysis.
    """
    # Column names from the UCI dataset documentation【684202365134492†L90-L104】
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    df = pd.read_csv(
        filepath,
        header=None,
        names=columns,
        na_values="?",
        comment="",
        skipinitialspace=True
    )
    # Convert all columns to numeric where possible
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop rows with missing values (or impute if desired)
    # For simplicity we drop; in practice one might impute or model them.
    df.dropna(inplace=True)
    # Convert target to binary (0 = no disease, 1 = disease)
    df["target"] = (df["num"] > 0).astype(int)
    df.drop(columns=["num"], inplace=True)
    return df


def perform_eda(df: pd.DataFrame) -> None:
    """Perform exploratory data analysis and save correlation heatmap.

    A correlation matrix gives a quick overview of relationships
    between variables and helps identify multicollinearity or strong
    associations.  The heatmap is saved to the ``images`` folder.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned heart disease DataFrame.
    """
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    # Ensure images directory exists
    img_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(img_dir, exist_ok=True)
    plt.tight_layout()
    heatmap_path = os.path.join(img_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {heatmap_path}")


def train_models(df: pd.DataFrame) -> None:
    """Train KNN and Random Forest models and evaluate their performance.

    Splits the data into training and test sets, scales features for
    KNN, fits both models and computes accuracy, confusion matrices
    and ROC curves.  Visualisations are saved to the ``images`` folder.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data with features and binary ``target`` column.
    """
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # Scale features for KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    # Random Forest classifier
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    # Metrics
    print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))
    print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
    # Confusion matrices
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    # Create directory for images
    img_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(img_dir, exist_ok=True)
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("KNN Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", ax=axes[1])
    axes[1].set_title("Random Forest Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(img_dir, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices to {cm_path}")
    # ROC curves
    y_proba_knn = knn.predict_proba(X_test_scaled)[:, 1]
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    roc_auc_knn = auc(fpr_knn, tpr_knn)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {roc_auc_knn:.2f})")
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Heart Disease Classifiers")
    plt.legend(loc="lower right")
    roc_path = os.path.join(img_dir, 'roc_curves.png')
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves to {roc_path}")
    # Feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[indices], y=X.columns[indices], palette="viridis")
    plt.title("Random Forest Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    fi_path = os.path.join(img_dir, 'feature_importances.png')
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"Saved feature importances to {fi_path}")


def main() -> None:
    """Main execution logic for the heart disease analysis."""
    data_url = (
        "https://raw.githubusercontent.com/StatQuest/logistic_regression_demo/"
        "master/processed.cleveland.data"
    )
    data_path = os.path.join(os.path.dirname(__file__), 'processed.cleveland.data')
    download_heart_data(data_url, data_path)
    df = load_and_clean_data(data_path)
    perform_eda(df)
    train_models(df)


if __name__ == "__main__":
    main()
