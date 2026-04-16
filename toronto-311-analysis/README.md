# Toronto 311 Service Request Analysis (2025)

A full-year analysis of 464,000+ service requests from the City of Toronto's 311 open data portal. Built to demonstrate how municipal service data can support staffing decisions, demand forecasting, and resource allocation.

## What This Project Does

This project connects to the City of Toronto's CKAN open data API, downloads the latest 311 service request dataset, cleans and preprocesses the data, and generates 14 interactive visualizations covering demand patterns, geographic distribution, ward-level breakdowns, correlation analysis, outlier detection, and short-term forecasting.

## Key Findings (2025 Data)

- **464,080 service requests** across 510 request types and 26 wards
- **Peak month:** February (53,268 requests) driven by winter storms
- **Peak hour:** 10:00 AM (39,549 requests across the year)
- **Biggest single-day spike:** February 19 with 6,153 requests (winter storm)
- **Top request type:** Residential Bin Lid Damaged (24,000+)
- **Highest volume ward:** Toronto-Danforth (28,246 requests)
- **Top correlated pair:** Bin Lid Damaged and Bin Body Damaged (0.88 correlation)
- **Pothole forecast:** Average 7.9 requests per day predicted for the next 30 days

## Visualizations Generated

| Chart | What It Shows | Operational Use |
|-------|--------------|-----------------|
| Top 15 Request Types | Most common service categories | Prioritize staffing and training |
| Monthly Volume Trend | Seasonal demand patterns | Plan shift coverage by month |
| Hourly Volume | Peak hours during the day | Schedule staff for peak times |
| Day-Hour Heatmap | Demand by day and hour combined | Detailed shift planning |
| Ward Breakdown | Request volume by geographic area | Allocate resources by area |
| Division Workload | Which city departments handle what | Understand workload distribution |
| Interactive Heatmap | Geographic density of requests | Identify hotspot neighborhoods |
| Spike Detection | Days with unusual volume | Detect weather and event triggers |
| Outlier Detection | Statistical anomalies (Z-score) | Flag days needing surge response |
| Correlation Heatmap | Which request types move together | Predict cascade demand |
| Ward vs Issue Heatmap | Which issues are in which wards | Localized resource planning |
| Top 3 Ward Comparison | Issue profile by ward | Curate reports per area |
| Status Distribution | Open, closed, cancelled breakdown | Track resolution performance |
| Pothole Forecast | 30-day demand prediction | Proactive crew scheduling |

## Technical Stack

- **Data Source:** City of Toronto CKAN API (package ID: `2e54bc0e-4399-4076-b717-351df5918ae7`)
- **Data Ingestion:** Python, requests, zipfile (downloads and extracts ZIP/CSV programmatically)
- **Data Processing:** pandas, NumPy (date parsing, feature engineering, missing value handling)
- **Visualization:** Plotly (interactive HTML charts), matplotlib/seaborn (static heatmap), Folium (interactive map)
- **Machine Learning:** scikit-learn LinearRegression (demand forecasting)
- **Geocoding:** FSA (Forward Sortation Area) postal code lookup table for 96 Toronto neighborhoods
- **Error Handling:** Structured try/except for network, ZIP, data, and schema errors
- **Local Fallback:** Auto-saves downloaded data to `./data/` for offline use

## Project Structure

```
toronto-311-analysis/
├── analysis.py              # Main analysis script
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── data/                    # Downloaded CSV data (auto-generated)
│   └── SR2025.csv
└── images/                  # Generated visualizations
    ├── top_request_types.html
    ├── monthly_volume.html
    ├── hourly_volume.html
    ├── demand_heatmap.png
    ├── ward_breakdown.html
    ├── division_workload.html
    ├── 311_heatmap.html
    ├── outlier_detection.html
    ├── correlation_heatmap.html
    ├── ward_issue_heatmap.html
    ├── ward_comparison.html
    ├── status_distribution.html
    └── pothole_forecast.html
```

## How to Run

```bash
# Clone the repository
git clone https://github.com/jibrankazi/data-analytics-portfolio.git
cd data-analytics-portfolio/toronto-311-analysis

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python analysis.py
```

The script will automatically download the 2025 dataset from Toronto's open data portal on first run and save a local copy in the `data/` folder. Subsequent runs will use the local copy for faster execution.

## Data Notes

- The dataset covers ALL service request channels combined (phone, app, email, web). No channel breakdown is available in the open data.
- Geographic coordinates are not included in the dataset. This project uses a built-in FSA (Forward Sortation Area) postal code lookup table to geocode requests to approximate neighborhood coordinates.
- The 2025 dataset provides a full 12-month view, which is essential for identifying seasonal patterns and planning staffing levels.

## Author

**Jibran Kazi, PMP**
- GitHub: [github.com/jibrankazi](https://github.com/jibrankazi)
- LinkedIn: [linktr.ee/jibrankazi](https://linktr.ee/jibrankazi)
