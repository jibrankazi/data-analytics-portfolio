"""
Bank Churn Analysis

This script performs customer churn analysis on the IBM Telco Customer Churn dataset. It loads the dataset from a remote source, cleans and preprocesses the data, builds several classification models, evaluates them using ROC AUC and confusion matrices, and saves visualisations to the images directory.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    """Load the Telco Customer Churn dataset from IBM's GitHub repository."""
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    return df


def preprocess_data(df):
    """Clean and preprocess the churn data, returning features, labels and a preprocessing pipeline."""
    df = df.copy()
    # Convert TotalCharges to numeric and fill missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    # Drop customerID column
    df.drop(columns=['customerID'], inplace=True)
    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    return X, y, preprocessor


def train_and_evaluate(X, y, preprocessor):
    """Train multiple models and evaluate them, returning a dictionary of results."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(max_depth=6),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    }
    results = {}
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        results[name] = {
            'report': classification_report(y_test, y_pred, output_dict=True),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'conf_matrix': confusion_matrix(y_test, y_pred)
        }
        # Plot confusion matrix and save
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['No Churn', 'Churn'], cmap='Blues')
        disp.ax_.set_title(f'{name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'../images/{name}_confusion_matrix.png')
        plt.close()
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC={results[name]['roc_auc']:.2f})')
    # Finalise ROC curve plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Churn Models')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../images/roc_curve.png')
    plt.close()
    return results


def main():
    df = load_data()
    X, y, preprocessor = preprocess_data(df)
    results = train_and_evaluate(X, y, preprocessor)
    # Print a simple summary of the results
    for name, metrics in results.items():
        print(f"--- {name} ---")
        print(f"ROC AUC: {metrics['roc_auc']:.2f}")
        print("Precision (Churn):", metrics['report']['1']['precision'])
        print("Recall (Churn):", metrics['report']['1']['recall'])
        print()


if __name__ == '__main__':
    main()
