## Bank Customer Churn Analysis

### Overview

Retaining customers is vital for financial institutions, as the cost of acquiring new clients often exceeds the cost of keeping existing ones. This project analyses customer behaviour to predict churn—whether a customer will close their account—and identifies the factors driving that decision. Using the Telco customer churn dataset from IBM, the aim is to build interpretable models that can be used to guide retention strategies.

### Problem Statement

Banks and telco companies face churn when customers cancel their services. Understanding which customers are likely to leave and why allows proactive engagement. The challenge is to model churn as a function of demographic information (e.g. gender, senior citizen), account tenure, services subscribed and billing details. The goal is to predict churn and rank drivers to inform targeted interventions.

### Data Source

The dataset comes from IBM’s Telco Customer Churn sample, which contains about 7 000 customers and includes features such as customer ID, gender, whether the person is a senior citizen, partner and dependent status, tenure, types of phone and internet service, contract type, payment method and charges【990940360085099†L0-L9】. The target variable `Churn` indicates whether the customer left within the last month. Some records include missing or categorical values (e.g. “No internet service”) that need to be encoded.

### Tools Used

- **Python** with `pandas` and `NumPy` for cleaning and preparation.
- **Scikit‑learn** for modelling (logistic regression, decision trees, random forests and gradient boosting) and evaluation.
- **Matplotlib** and **Seaborn** for visualisations.
- **Imbalanced‑Learn** for optional resampling if classes are imbalanced.

### Business Value

Reducing churn improves profitability and customer lifetime value. By identifying high‑risk customers and the factors associated with their decisions, management can offer personalised retention incentives (e.g. loyalty programmes, discounts or service upgrades) and improve customer satisfaction. Insights from churn analysis also inform marketing campaigns and resource allocation.

### Approach and Key Findings

1. **Data Cleaning** – The dataset is loaded and inspected. Total charges are converted to numeric, and missing values are handled. Categorical features (e.g. gender, contract type, internet service) are encoded using one‑hot encoding.
2. **Exploratory Analysis** – Churn rates are examined across demographics, tenure ranges and subscription types. For example, customers on month‑to‑month contracts or with fibre‑optic internet tend to churn at higher rates.
3. **Modelling** – Several algorithms are compared:
   * *Logistic Regression* – A baseline linear model that provides interpretable coefficients.
   * *Decision Tree* and *Random Forest* – Non‑linear models that capture complex interactions.
   * *Gradient Boosting* – XGBoost or light gradient boosting to improve predictive accuracy.
4. **Evaluation** – Accuracy, precision, recall, F1‑score and ROC‑AUC are computed. Models are validated using cross‑validation. Feature importance from tree‑based models highlights influential variables such as tenure, contract type and monthly charges.
5. **Segmentation and Recommendations** – Customers are segmented by risk level, and specific retention actions are proposed (e.g. long‑term contract offers for high‑tenure customers on month‑to‑month plans). Visualisations of churn rates by segment are included in the `images` folder.

### Visualisations

- Bar charts showing churn rates by contract type, tenure buckets and service subscriptions
- Correlation heatmap of numerical variables
- Confusion matrix and ROC curves for each model
- Feature importance plots for tree‑based models

These analyses enable data‑driven recommendations to reduce churn and enhance customer satisfaction.
