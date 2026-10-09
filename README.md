# Data Analytics Portfolio — evidence and real-data status

The repository contains several **separate analytics projects**. Their data sources and execution maturity differ. Successful workflows now independently verify **Toronto 311 descriptive analysis and the original UCI Cleveland heart-disease research lifecycle**. Other modules must still be evaluated individually.

## Independently executed original observations

**Toronto 311 service requests:** `toronto-311-analysis/data/SR2025.csv` contains historical City of Toronto service requests. `validate_real_311.py` audits the original CSV fields, parses 2025 creation dates, summarizes request types and timing, and saves JSON results. [Successful October 9 workflow](https://github.com/jibrankazi/data-analytics-portfolio/actions/runs/37936583459).

```bash
pip install pandas
python validate_real_311.py
```

Original publisher: [City of Toronto Open Data — 311 Service Requests](https://open.toronto.ca/dataset/311-service-requests-customer-initiated/).

**Scope restriction:** These are service requests from multiple intake channels. They are **not exclusively telephone calls**, and cannot by themselves establish live queue volumes, handling times or staffing.

## Other projects, described without unverified model results

| Project folder | What the repository contains | Evidence boundary |
| --- | --- | --- |
| `bank-churn-analysis/` | The **IBM Telco Customer Churn** CSV and analytical Python code | A telecommunications churn dataset, **not real bank account churn** |
| `fraud-detection-finance/` | PaySim-related financial fraud analysis code | PaySim is a **simulator**, not verified actual customer transaction fraud |
| `heart-disease-prediction/` | [Actual UCI Cleveland 303-row end-to-end classification benchmark](https://github.com/jibrankazi/data-analytics-portfolio/actions/runs/37954748781): direct publisher download, 212/91 holdout split, pipeline imputation, three model evaluations and uploaded JSON/ROC figure | **Verified historical research only**; no external hospital validation or clinical use |
| `public-health-sentiment/` | Public-health text/sentiment analysis script | Live historical source scraping and complete sentiment evaluation not verified |
| `toronto-311-analysis/` | Actual 2025 service-request CSV and audited descriptive counts | Public-data descriptive analysis confirmed; individual service outcomes need separate assessment |

**Additional October 9, 2026 actual observed-data result:** The UCI Cleveland source yielded 303 original records, 13 features, 6 missing feature cells, and a 91-row stratified holdout. Measured ROC AUC was **0.500** for the prior-prevalence baseline, **0.865** for KNN and **0.925** for Random Forest. These scores come directly from the [successful model training/evaluation job](https://github.com/jibrankazi/data-analytics-portfolio/actions/runs/37954748781), not historical unverified images. The sample is small and the split is single, with no uncertainty intervals; results are **not** clinical performance estimates. The original zero-byte Cleveland file was removed. The other modules still need their own full source-to-results CI execution before being called end-to-end.

The scripts may use simulated records in mathematical tests. Such fixtures must never be reported as measured economic or medical outcomes. No confidential municipal or banking data are published here.
