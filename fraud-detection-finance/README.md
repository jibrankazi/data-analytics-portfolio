# Financial fraud detection — real ULB transaction benchmark

This folder now contains an **executed historical credit-card fraud classification system** using **original, genuine, anonymized European cardholder transactions**, not generated or substituted PaySim records. It is an educational research benchmark; it is **not deployed bank software**.

## Interview engineering addition: fixed-threshold cost sensitivity

[**Passed end-to-end original OpenML source, training, fixed-threshold decision-analysis and verification run**](https://github.com/jibrankazi/data-analytics-portfolio/actions/runs/38342914993). The same workflow independently rechecked 56,965 historic heldout rows: **61 detected frauds, 21 missed frauds, 11 false alerts and 56,872 correctly unflagged nonfrauds**. Under explicitly invented cost assumptions (for example, $10 per false alert and $100 per missed fraud), the illustrative test-set accounting is 61 × $100 − 11 × $10 = **$5,990 lower hypothetical cost** versus a never-flag policy. These numbers are **not realized dollar savings, a bank's actual fraud losses or a model-selection result**.

A separate `analyze_fraud_costs.py` script evaluates the **already frozen
validation-selected model and threshold** on the existing ULB heldout probability
artifact. It independently reconstructs TN, FP, FN and TP from the published
test rows and checks they exactly match the original model report. Then it
calculates the possible consequences of false alerts versus undetected fraud
under three **explicit hypothetical** cost assumptions.

```bash
# Once the source-to-model benchmark has already produced its results:
python -m unittest discover -s fraud-detection-finance/tests -p 'test_fraud_cost_sensitivity.py' -v
python fraud-detection-finance/analyze_fraud_costs.py
```

The resulting `frozen_threshold_cost_sensitivity.json` records original-result
checks, file hashes, real historical classification counts, and separate
illustrative cost arithmetic. It is uploaded by the
[fraud workflow](https://github.com/jibrankazi/data-analytics-portfolio/actions/workflows/real-ulb-fraud.yml)
when the new workflow passes.

This demonstrates the difference between **classification quality** and a
**decision policy**. The assumed false-alert and missed-fraud costs **are not
real bank losses, consumer impacts or realized cost savings**. The original
threshold was selected on validation data for F1, *not* optimized on the
holdout labels or these hypothetical cost scenarios. All cost analyses are
post-hoc sensitivities, not evidence of the profitability or safety of a live
fraud detection program.

---

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
