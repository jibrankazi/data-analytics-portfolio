# Data Analytics Portfolio — evidence and real-data status

The repository contains several **separate analytics projects**. Their data sources and execution maturity differ. Successful workflows now independently verify **Toronto 311 descriptive analysis**, the **original UCI Cleveland heart-disease research lifecycle** **IBM illustrative Telco churn model training/evaluation** and **real ULB credit-card fraud model training**. Other modules must still be evaluated individually.

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
| `bank-churn-analysis/` | [Verified IBM example source-to-holdout training](https://github.com/jibrankazi/data-analytics-portfolio/actions/runs/37962422491): 7,043 rows, three models, plot and auditable metrics | **Published illustrative telecom sample, not genuine bank or live telco outcomes** |
| `fraud-detection-finance/` | [Completed original ULB 284,807-row genuine anonymized transaction model](https://github.com/jibrankazi/data-analytics-portfolio/actions/runs/37969687947), validation-selected classifier, 56,965-row untouched holdout, actual PR curves and scored rows | Genuine **historical** 2013 credit card fraud; OpenML omits transaction time so not forward-time/live bank validation; older PaySim material remains simulated |
| `heart-disease-prediction/` | [Actual UCI Cleveland 303-row end-to-end classification benchmark](https://github.com/jibrankazi/data-analytics-portfolio/actions/runs/37954748781): direct publisher download, 212/91 holdout split, pipeline imputation, three model evaluations and uploaded JSON/ROC figure | **Verified historical research only**; no external hospital validation or clinical use |
| `public-health-sentiment/` | [Authentic PHAC official feed-to-lexicon language audit](https://github.com/jibrankazi/data-analytics-portfolio/actions/runs/37963620463): 43 unique dated public releases, one preview-domain link excluded, true CSV/plot/JSON artifact | **Agency-authored communications only; not Twitter posts or population vaccination attitudes** |
| `toronto-311-analysis/` | Actual 2025 service-request CSV and audited descriptive counts | Public-data descriptive analysis confirmed; individual service outcomes need separate assessment |

**Additional October 9, 2026 actual observed-data result:** The UCI Cleveland source yielded 303 original records, 13 features, 6 missing feature cells, and a 91-row stratified holdout. Measured ROC AUC was **0.500** for the prior-prevalence baseline, **0.865** for KNN and **0.925** for Random Forest. These scores come directly from the [successful model training/evaluation job](https://github.com/jibrankazi/data-analytics-portfolio/actions/runs/37954748781), not historical unverified images. The sample is small and the split is single, with no uncertainty intervals; results are **not** clinical performance estimates. The original zero-byte Cleveland file was removed. The prior PaySim simulator file is not a genuine labeled transaction source. Public-health Twitter/X opinions remain unverified; the separate PHAC official-publisher messaging pipeline is independently executed. Further external, prospective and cost-weighted fraud validation is still missing.

**IBM illustrative churn results (October 9):** 7,043 sample rows and a 1,761-row stratified holdout. Measured ROC AUC was **0.500** baseline, **0.846** logistic regression, and **0.843** Random Forest; results are from the [successful executed model job](https://github.com/jibrankazi/data-analytics-portfolio/actions/runs/37962422491) and are not verified performance on real bank subscribers.\n\n**Public health publisher-language results:** The [successful PHAC run](https://github.com/jibrankazi/data-analytics-portfolio/actions/runs/37963620463) included **43 actual published agency updates**, excluding one preview-host entry; VADER lexical classifications were 17 negative, 15 neutral and 11 positive. These are mechanical scores of government-authored text, **not public opinions or independently labeled sentiment truths**.\n\n**Verified genuine cardholder fraud (October 9):** [successful ULB full execution](https://github.com/jibrankazi/data-analytics-portfolio/actions/runs/37969687947) loaded **284,807 original historical anonymized card transactions and 492 fraud labels**. The validation-selected histogram gradient boosting model found 61 of 82 fraud events in its 56,965-record untouched test (11 false alarms; 84.7% precision, 74.4% recall; average precision 0.774). Original identical anonymized feature vectors were grouped across partitions to prevent exact input duplicates leaking; OpenML omits the original time, so this is **not** chronological prospective validation or real production fraud prevention.

The scripts may use simulated records in mathematical tests. Such fixtures must never be reported as measured economic or medical outcomes. No confidential municipal or banking data are published here.
