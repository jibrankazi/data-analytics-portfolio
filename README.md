
The repository contains several **separate analytics projects**. Their data sources and execution maturity differ. A successful workflow on Toronto 311 data does not verify all the other project scripts.

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
| `heart-disease-prediction/` | Cleveland Heart Disease analysis script | The checked-in `processed.cleveland.data` file is **empty**; no complete source-data execution was established |
| `public-health-sentiment/` | Public-health text/sentiment analysis script | Live historical source scraping and complete sentiment evaluation not verified |
| `toronto-311-analysis/` | Actual 2025 service-request CSV and audited descriptive counts | Public-data descriptive analysis confirmed; individual service outcomes need separate assessment |

The existence of a script or chart does not prove it was executed successfully on real input. The Toronto 311 CI job is the concrete evidence linked above; the other projects need their own validated data contracts, provenance, full executions and artifacts before being called end-to-end.

The scripts may use simulated records in mathematical tests. Such fixtures must never be reported as measured economic or medical outcomes. No confidential municipal or banking data are published here.
