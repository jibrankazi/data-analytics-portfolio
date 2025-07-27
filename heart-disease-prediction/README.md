# Heart Disease Risk Prediction

### Overview

This project focuses on building a predictive model that estimates the likelihood that a patient has heart disease.  The work is based on the Cleveland clinic subset of the UCI Heart Disease dataset, which contains just over 300 records and 14 clinically relevant attributes.  Each row represents a patient and includes demographic information (age and sex), medical measurements (resting blood pressure, serum cholesterol, maximum heart rate achieved) and results of diagnostic tests (chest pain type, fasting blood sugar, resting electrocardiographic results, exercise‑induced angina, ST depression and slope, number of major vessels coloured by fluoroscopy, and thalassemia).  The goal variable indicates the presence or absence of heart disease.

### Problem Statement

Cardiovascular diseases remain a leading cause of mortality globally.  Identifying individuals at high risk enables clinicians to intervene earlier, tailor treatments and reduce costs.  The challenge is to construct a model that can discriminate between patients who have heart disease (a positive diagnosis) and those who do not based on the available clinical measurements.

### Data Source

The raw data originate from the Cleveland Clinic and were made publicly available through the UCI Machine Learning Repository.  In the Cleveland subset only 14 attributes are used and the target variable `num` indicates the presence of heart disease.  The features include age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise induced angina, ST depression induced by exercise, the slope of the peak exercise ST segment, number of major vessels coloured by fluoroscopy and thalassemia【684202365134492†L90-L104】.  Missing values exist for some categorical features (e.g. `ca`, `thal` and `slope`) and need to be imputed before modelling.

### Tools Used

- **Python** for data loading, cleaning, and modelling.
- **Pandas** and **NumPy** for data manipulation.
- **Matplotlib** and **Seaborn** for exploratory visualisations.
- **Scikit‑learn** for machine learning models (K‑nearest neighbours and random forests) and evaluation metrics.

### Business Value

Predictive models for heart disease risk assist healthcare providers in triaging patients and allocating resources.  An accurate classifier can highlight high‑risk individuals who may benefit from further diagnostic tests or lifestyle interventions.  Conversely, it can reduce unnecessary testing for those at low risk, improving patient experience and lowering costs.

### Approach and Key Findings

1. **Data Preparation** – The raw `.data` file from UCI was parsed with appropriate column names.  Missing values in the `ca`, `thal` and `slope` features were imputed using the most common value for each feature.  All categorical variables were encoded numerically.
2. **Exploratory Analysis** – Pairwise correlations were examined to understand relationships between risk factors.  Age and maximum heart rate showed moderate correlations with the presence of disease, while features like chest pain type and ST depression exhibited stronger associations.
3. **Modelling** – Two supervised models were trained:
   * *K‑Nearest Neighbours* (KNN) –  A simple classification algorithm that makes predictions based on the most common class among the nearest neighbours in feature space.  Cross validation was used to select the optimal number of neighbours.
   * *Random Forest* –  An ensemble of decision trees that improves predictive performance through bagging.  Feature importance scores from the forest highlighted chest pain type, the slope of the ST segment and number of major vessels as influential predictors.
4. **Evaluation** – Models were evaluated using accuracy, confusion matrices and ROC‑AUC curves.  The random forest outperformed KNN with a higher area under the ROC curve and better balanced sensitivity and specificity.  Visualisations of the ROC curve and feature importance are provided in the `images` folder.

### Visualisations

Plots generated during the analysis include:

- Correlation heatmap of clinical variables
- Confusion matrix comparing actual vs. predicted diagnoses
- ROC curve illustrating model sensitivity vs. specificity
- Bar chart of feature importances from the random forest

These images are saved under `heart-disease-prediction/images`.  An interactive dashboard (Tableau or Power BI) can be added later to share findings with a non‑technical audience.
