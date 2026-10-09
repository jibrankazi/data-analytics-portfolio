# Financial fraud detection — real ULB transaction benchmark

This folder now contains an **executed historical credit-card fraud classification system** using **original, genuine, anonymized European cardholder transactions**, not generated or substituted PaySim records. It is an educational research benchmark; it is **not deployed bank software**.

## The original problem is fixed

Previously, `notebooks/fraud_detection.py` could not run on another computer because it opened a fixed `C:\\Users\\...\\Downloads\\...csv` path, which is absent from the repository. It also applied SMOTE **before** a train/test split, contaminating the evaluation with synthetic observations derived from training-and-test data.

The actual executable `notebooks/fraud_detection.py` now delegates to `run_real_ulb_fraud.py`, which directly downloads the authentic [OpenML ULB fraud benchmark dataset, ID 1597](https://www.openml.org/d/1597) from the original public published repository, then runs data validation, training, separated model/threshold selection, untouched test evaluation, precision–recall plots and reproducibility artifacts.

## Actual successful full historical data-to-model run

[**GitHub Actions: independently verified successful execution**](https://github.com/jibrankazi/data-analytics-portfolio/actions/runs/37969687947) — October 9, 2026.

| Source and partition | Actual observations |
| --- | ---: |
| Original anonymized 2013 European credit-card transaction records | 284,807 |
| Original fraud-labeled transactions | 492 |
| Identical anonymized feature vectors assigned to the same split | 9,144 repeated instances |
| Training records | 170,773 |
| Validation records | 57,069 |
| **Untouched test records** | **56,965** |
| Fraud cases in untouched test | 82 |

**No original input rows were deleted** to manufacture model performance. OpenML's public version does not include the original transaction `Time` field, so the program groups identical public 29-feature vectors in the **same partition** to prevent an exact-match contamination across samples and splits. The three partitions are randomized **by feature group**, not chronological. Selecting a model and its decision threshold uses **validation data only**. The held-out test is scored after selection.

| Fully executed model | Held-out ROC AUC | Held-out average precision (PR AUC) |
| --- | ---: | ---: |
| Prior-prevalence baseline | 0.500 | 0.0014 |
| Class-weighted logistic regression | 0.993 | 0.7105 |
| Histogram gradient boosting | 0.985 | **0.7744** |

The model selected by validation average precision was histogram gradient boosting. With its validation-selected threshold, its genuine untouched test confusion matrix showed **61 correctly detected frauds**, **21 missed frauds**, **11 false alerts**, and **56,872 correctly ignored nonfraud transactions**. Precision was **84.72%**, recall **74.39%**, fraud F1 **0.7922**. These scores are limited to this partition of the **historical dataset**, not operational fraud detection accuracy.

## Actually run the application

With Python 3.11+ and internet access:

```bash
python -m pip install pandas numpy scikit-learn matplotlib scipy
python fraud-detection-finance/notebooks/fraud_detection.py
```

The script uses OpenML dataset **1597**, checks all original published row counts and labels, verifies original PCA/amount fields, guards against grouping duplicate feature vectors into multiple splits, trains the baseline and two fitted classifiers, and saves:

- `fraud-detection-finance/results/ulb_real_fraud/verified_results.json` — actual publisher, original data fingerprint, all real split sizes and all real validation/test metrics.
- `fraud-detection-finance/results/ulb_real_fraud/heldout_real_ulb_probabilities.csv` — genuine original source row positions, fraud labels and all three models' untouched-test predicted probabilities; **no cardholder names or IDs**.
- `fraud-detection-finance/results/ulb_real_fraud/heldout_precision_recall.png` — calculated curve for each model on the actual untouched test.

The GitHub Actions workflow retrieves and checks these as the `actual-ULB-2013-fraud-train-validation-test-results` downloadable workflow artifact.

**Research boundaries:** The ULB data are only two days in September 2013, with privacy-protected PCA features. The OpenML copy lacks original time, and **no forward-time prospective validation** was done. No live processor integration, real-time alert review, concept drift, operational loss, cost-sensitive fraud thresholds or customer-protection outcome is established. These research models must **not** be used as production fraud decisions without substantial additional work.

### Historical PaySim material

The earlier material discussed **PaySim**, a mobile-money transaction *simulator*. PaySim records are not verified real cardholder activity. The repository did not contain the original PaySim source CSV. To avoid masking this distinction and to eliminate invalid pre-split SMOTE evaluation, the executable entrypoint has been upgraded to the independently verified ULB source pipeline rather than pretending missing PaySim files exist. A dedicated and separately validated PaySim simulation study could be added only when its actual simulation CSV is supplied and documented.
