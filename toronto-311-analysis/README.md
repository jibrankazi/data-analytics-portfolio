# Toronto 311 Service Request Analysis

**Overview**

This project analyzes service requests submitted by Toronto residents via the 311 customer service system. The goal is to understand what types of issues residents report most frequently, how requests vary over time and across neighbourhoods, and where resources could be allocated to improve city services.

**Problem Statement**

The City of Toronto receives thousands of service requests each month through its 311 system. Without analysis, it's difficult to prioritize issues, identify hotspots, and allocate resources effectively. This project uses open data to categorize requests by type, analyse temporal patterns, and visualize geographic trends to inform city management and community planning.

**Data Source**

The dataset comes from Toronto's open data portal. The "311 Service Requests – Customer Initiated" dataset contains records of all service requests submitted by residents. Each record includes the creation date/time, service type, status, and location. The dataset is updated monthly and is accessible via the City of Toronto's CKAN API【111782869124138†L117-L160】.

**Tools Used**

- Python, pandas, geopandas for spatial analysis, matplotlib and seaborn for static charts, and Plotly for interactive maps.
- requests for API access and data retrieval.
- A Jupyter notebook orchestrates data downloading, cleaning, analysis, and visualization.
- Optional: Tableau or Power BI dashboard for interactive exploration.

**Business Value**

By analysing 311 data, the city can identify the most common issues and geographic hotspots, enabling targeted interventions. Understanding seasonal patterns helps anticipate service demand and allocate crews accordingly. Insights from this analysis support data-driven policy decisions and improve resident satisfaction.

**Approach and Key Findings**

1. **Data Acquisition** – Use the CKAN API to download the latest 311 service request records. Parse timestamps into datetime objects and select relevant fields such as request type, status, and location.
2. **Cleaning and Categorization** – Handle missing values, standardise request categories, and derive additional features such as year, month, weekday, and season.
3. **Exploratory Analysis** – Identify the most frequent service request types and compute monthly counts to reveal seasonal trends. Analyse status distribution (Initiated, In Progress, Closed) and average resolution times.
4. **Geospatial Analysis** – Map requests by latitude and longitude using Plotly or geopandas to highlight neighbourhoods with high concentrations of complaints.
5. **Visualization** – Produce bar charts of top request types, line charts showing monthly volume, and interactive heat maps. A dashboard summarises key metrics for stakeholders.

**Visualisations**

- Bar plot of top 10 service request types.
- Line chart of monthly service request volume.
- Status distribution chart.
- Choropleth or heat map of requests by neighbourhood.
- Optional interactive dashboard link.
