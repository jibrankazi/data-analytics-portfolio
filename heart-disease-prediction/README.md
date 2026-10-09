# Heart Disease Research — Real UCI Cleveland Data

This project contains a reproducible **historical** supervised-learning evaluation on the genuine [UCI Heart Disease Cleveland dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). UCI documents **303 original Cleveland observations**, 13 input features and a diagnosis-severity code `num` from 0 to 4. We use **0 = no recorded heart disease** and **1–4 = recorded heart disease present**. Missing original measurements are retained as missing values and imputed **using training data only**.

**This is an educational research benchmark, not a validated medical screening or diagnostic product.** The dataset is small and historical, with no independent external clinical validation.

## Run the actual complete lifecycle

With Python 3.11 and an internet connection:

```bash
python -m pip install pandas numpy scikit-learn matplotlib
python heart-disease-prediction/run_verified_cleveland.py
```

The verified runner:
1. Downloads original `processed.cleveland.data` **directly from UCI**, not a generated example or an unofficial GitHub mirror.
2. Checks the full **303 × 14** shape, numeric fields, target codes, reasonable original-value ranges and expected missingness; a missing, empty or malformed download fails.
3. Uses a reproducible **70% training / 30% stratified holdout**, fixed random seed 42.
4. Fits median-free, **most-frequent imputation within each model's training pipeline**, scaling KNN from the training split only. Evaluates a prior-prevalence baseline, KNN (k=5) and a 200-tree Random Forest.
5. Saves accuracy, balanced accuracy, ROC AUC, average precision, confusion matrices and a holdout ROC plot. The results are computed from the executed models, not copied from pre-existing plots.

### Outputs

- `heart-disease-prediction/results/verified_results.json`: source URL, UTC retrieval timestamp, SHA-256 fingerprint of the **actual downloaded raw bytes**, observations, features, missing values, split sizes, model scores and limitations.
- `heart-disease-prediction/results/heldout_roc.png`: generated holdout ROC curves.

The data and plots are generated into the GitHub Actions **artifact**, not committed as new patient-level datasets. Check the [real Cleveland lifecycle workflow](https://github.com/jibrankazi/data-analytics-portfolio/actions/workflows/real-heart-cleveland.yml) and its latest completed run; an unexecuted job is not proof of results.

## Why the original older script was not sufficient

The older `notebooks/heart_disease_analysis.py` uses an unofficial GitHub URL, discards patients with missing features and contains plots from an earlier context with no currently reproducible metric record. Its original narrative also mentions imputation and cross-validation that the code does not actually perform. **Those original plots are not evidence of a newly re-executed clinical experiment.** Use `run_verified_cleveland.py` for the current independently checked source-to-results pipeline.

The pre-existing tracked `data/processed.cleveland.data` had **zero bytes** and was not a valid data source; the verified pipeline retrieves directly from the original publisher instead.

### Research boundaries

This is a single 70/30 held-out experiment without repeated confidence intervals, external-site validation, decision-threshold optimization, drift monitoring, patient consent pathway, or regulatory approval. **Do not apply these outputs to individual medical decisions.**

Original reference: Jánosi A., Steinbrunn W., Pfisterer M. and Detrano R., *Heart Disease*, UCI Machine Learning Repository, DOI: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X).
