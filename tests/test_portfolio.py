from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    "bank-churn-analysis": "bank_churn_analysis.py",
    "fraud-detection-finance": "fraud_detection.py",
    "heart-disease-prediction": "heart_disease_analysis.py",
    "public-health-sentiment": "public_health_sentiment.py",
    "toronto-311-analysis": "toronto_311_analysis.py",
}


def test_five_scripts_exist():
    for folder, name in PROJECTS.items():
        assert (ROOT / folder / "notebooks" / name).is_file()


def test_toronto_311_sample_is_parseable():
    path = ROOT / "toronto-311-analysis" / "data" / "SR2025.csv"
    data = pd.read_csv(path, nrows=20, encoding="latin1", on_bad_lines="skip")
    assert len(data) > 0
    assert len(data.columns) > 1


def test_bank_churn_dataset():
    path = ROOT / "bank-churn-analysis" / "data" / "Telco-Customer-Churn.csv"
    data = pd.read_csv(path, nrows=20)
    assert len(data) > 0
