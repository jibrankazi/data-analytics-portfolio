
# Data Analytics Portfolio

This repository showcases a collection of five end‑to‑end analytics projects built from publicly available data.  Each project addresses a real‑world question and demonstrates the workflow I follow as a junior data analyst—from acquiring and cleaning the data, through exploratory analysis, modelling and interpretation, to communicating findings with clear narratives and visuals.  All work is organised into separate folders with well‑commented notebooks, raw data (or scripts to download it), images and project‑specific documentation.

## Projects

| Project | Domain | Brief Description |
| --- | --- | --- |
| **Heart Disease Risk Prediction** | Healthcare | Predict the presence of heart disease using the Cleveland Clinic subset of the UCI Heart Disease dataset.  Models include K‑Nearest Neighbours and Random Forests; key risk factors are identified. |
| **Public Health Sentiment Analysis** | Public Communication | Scrape tweets mentioning COVID‑11 vaccines, clean and score sentiment with VADER, and visualise temporal trends and word frequencies. |
| **Financial Fraud Detection** | Banking & FinTech | Detect fraudulent mobile money transactions using the PaySim simulation dataset.  Address class imbalance with SMOTE and train ensemble models such as Random Forests and XGBoost. |
| **Bank Customer Churn** | Retail Banking | Analyse the IBM Telco customer churn data to predict which customers are likely to leave and understand the drivers of churn. |
| **Toronto 311 Service Requests** | Civic Analytics | Pull 311 service request data from Toronto’s Open Data portal, analyse complaint types and patterns, and map hotspots and seasonal trends. |

## How to Use This Portfolio

Navigate into each project folder to find:

* A **README** explaining the business context, problem statement, data source, tools, approach and findings.
* A **notebooks** directory containing a Python script or Jupyter notebook with the full analysis.  These scripts are written in a clear, step‑by‑step manner with comments that explain my reasoning at each stage.
* A **data** directory with either the raw data or a script to download it (depending on licensing restrictions).  For large or restricted datasets, you may need to run the download script yourself.
* An **images** directory storing plots generated during the analysis.  Feel free to browse these for a quick visual overview.

The code has been written to be reproducible and easy to follow.  To replicate an analysis, create a Python environment with the required libraries (see the imports in each script) and run the notebook from top to bottom.  Where external downloads are required (e.g. scraping tweets or fetching 311 data), network connectivity is necessary.
