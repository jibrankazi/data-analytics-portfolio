"""
Bank Churn Analysis

This script performs customer churn analysis using the IBM Telco Customer Churn dataset.
It trains multiple models, evaluates them, and saves visualizations into the images folder.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)


def ensure_image_folder():
    """Ensure the local ./images directory exists without permission issues."""
    image_dir = os.path.join(".", "images")
    os.makedirs(image_dir, exist_ok=True)
    return image_dir


def load_data():
    """Load the Telco Customer Churn dataset from IBM's GitHub repository."""
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    return df


def preprocess_data(df):
    """Clean and preprocess the churn data."""
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df.drop(columns=['customerID'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    return X, y, preprocessor


def train_and_evaluate(X, y, preprocessor, image_dir):
    """Train models and evaluate them with metrics and saved plots."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(max_depth=6),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    }

    results = {}
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
            'report': classification_report(y_test, y_pred, output_dict=True),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'conf_matrix': confusion_matrix(y_test, y_pred)
        }

        # Save Confusion Matrix
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=['No Churn', 'Churn'], cmap='Blues'
        )
        disp.ax_.set_title(f'{name} Confusion Matrix')
        cm_path = os.path.join(image_dir, f'{name}_confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        print(f"✅ Saved {name} confusion matrix to {cm_path}")

        # Add ROC curve to combined plot
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["roc_auc"]:.2f})')

    # Final ROC curve plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Churn Models')
    plt.legend()
    roc_path = os.path.join(image_dir, 'roc_curve.png')
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    print(f"✅ Saved combined ROC curve to {roc_path}")

    return results


def main():
    image_dir = ensure_image_folder()
    df = load_data()
    X, y, preprocessor = preprocess_data(df)
    results = train_and_evaluate(X, y, preprocessor, image_dir)

    print("\n📊 === Summary ===")
    for name, metrics in results.items():
        print(f"\n{name}")
        print(f"ROC AUC: {metrics['roc_auc']:.2f}")
        print(f"Precision (Churn): {metrics['report']['1']['precision']:.2f}")
        print(f"Recall (Churn): {metrics['report']['1']['recall']:.2f}")


if __name__ == '__main__':
    main()
