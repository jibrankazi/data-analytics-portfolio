# IBM Telco Customer Churn — verified example-data machine-learning benchmark

The code in this folder uses **IBM's published Telco Customer Churn demonstration CSV**, a historical teaching/sample dataset, **not verified live telecom customer records and not bank account data**. It cannot establish real-world retention savings, future churn lift or deployment readiness.

### Independently executed, reproducible model lifecycle

[Successful October 9, 2026 GitHub Actions train-and-holdout run](https://github.com/jibrankazi/data-analytics-portfolio/actions/runs/37962422491) checked the existing published sample and completed fitting, evaluation, plot-generation and artifact checks.

| Measured observation from the actual executed run | Value |
| --- | ---: |
| Original IBM example CSV rows | 7,043 |
| Churn labels marked Yes | 1,869 |
| Feature columns after removing sample customer ID and target | 19 |
| Numeric missing cells (original blank TotalCharges) | 11 |
| Stratified training rows | 5,282 |
| Untouched holdout rows | 1,761 |

| Model | Held-out accuracy | Held-out ROC AUC |
| --- | ---: | ---: |
| Prior-probability baseline | 73.48% | 0.500 |
| Logistic regression | 80.81% | 0.846 |
| Random Forest | 80.12% | 0.843 |

Note that naive accuracy can exaggerate performance on imbalanced labels: the always-majority baseline reaches 73.5% accuracy despite a useless 0.500 ROC AUC. The model evaluation also records **balanced accuracy, average precision and confusion matrices**.

### Run the verified code

```bash
python -m pip install numpy pandas matplotlib scikit-learn
python bank-churn-analysis/run_verified_telco.py
```

This program uses `bank-churn-analysis/data/Telco-Customer-Churn.csv`, checks that it is the **7,043-row IBM demonstration schema**, calculates the SHA-256 fingerprint of the actual input file, and trains three models with a fixed 75/25 stratified split. IDs are never features; median numeric imputation, categoricals and scaling are fitted on the **training partition only** to prevent data leakage.

It writes the following generated results to `bank-churn-analysis/results/`:

- `verified_results.json`: exact sample-data provenance, file fingerprint, split and actual measured metrics.
- `holdout_roc.png`: ROC curves generated from the held-out predictions.

GitHub Actions uploads both as the `verified-ibm-telco-illustrative-churn-analysis` run artifact.

**Not validated:** Actual customer records, financial institution churn, independent out-of-sample institutions, fairness, statistical confidence intervals, retention uplift, real-world decisions, production integration or economic ROI.

### Existing historical script

The pre-existing `notebooks/bank_churn_analysis.py` is an older educational example with a remotely hosted CSV dependency and saved exploratory charts. The newly tested `run_verified_telco.py` is the reproducible source-to-results workflow; prior static images alone are not evidence of a new execution.

Original publicly distributed example: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d).
