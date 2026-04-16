"""
Toronto 311 Service Request Analysis (2025) - Complete Notebook
Full-year analysis: charts, heatmaps, ward breakdown, correlation, outlier detection, forecasting
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from folium.plugins import HeatMap
from sklearn.linear_model import LinearRegression
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = './images'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('./data', exist_ok=True)

CKAN_BASE = "https://ckan0.cf.opendata.inter.prod-toronto.ca"

TORONTO_FSA_COORDS = {
    'M1B': (43.8067, -79.1944), 'M1C': (43.7845, -79.1605), 'M1E': (43.7636, -79.1887),
    'M1G': (43.7710, -79.2169), 'M1H': (43.7731, -79.2395), 'M1J': (43.7448, -79.2395),
    'M1K': (43.7279, -79.2620), 'M1L': (43.7112, -79.2845), 'M1M': (43.7164, -79.2395),
    'M1N': (43.6927, -79.2648), 'M1P': (43.7574, -79.2730), 'M1R': (43.7500, -79.2956),
    'M1S': (43.7942, -79.2620), 'M1T': (43.7816, -79.3040), 'M1V': (43.8153, -79.2845),
    'M1W': (43.7995, -79.3183), 'M1X': (43.8361, -79.2056), 'M2H': (43.8037, -79.3549),
    'M2J': (43.7785, -79.3465), 'M2K': (43.7869, -79.3857), 'M2L': (43.7574, -79.3746),
    'M2M': (43.7890, -79.4082), 'M2N': (43.7700, -79.4082), 'M2P': (43.7527, -79.4082),
    'M2R': (43.7827, -79.4421), 'M3A': (43.7532, -79.3296), 'M3B': (43.7459, -79.3521),
    'M3C': (43.7258, -79.3408), 'M3H': (43.7543, -79.4421), 'M3J': (43.7679, -79.4872),
    'M3K': (43.7374, -79.4647), 'M3L': (43.7390, -79.5097), 'M3M': (43.7279, -79.4985),
    'M3N': (43.7616, -79.5210), 'M4A': (43.7258, -79.3183), 'M4B': (43.7069, -79.3070),
    'M4C': (43.6953, -79.3183), 'M4E': (43.6763, -79.2930), 'M4G': (43.7090, -79.3634),
    'M4H': (43.7048, -79.3465), 'M4J': (43.6848, -79.3408), 'M4K': (43.6795, -79.3521),
    'M4L': (43.6689, -79.3155), 'M4M': (43.6595, -79.3409), 'M4N': (43.7280, -79.3888),
    'M4P': (43.7127, -79.3888), 'M4R': (43.7153, -79.4057), 'M4S': (43.7043, -79.3888),
    'M4T': (43.6895, -79.3831), 'M4V': (43.6864, -79.3972), 'M4W': (43.6795, -79.3747),
    'M4X': (43.6679, -79.3676), 'M4Y': (43.6658, -79.3831), 'M5A': (43.6543, -79.3606),
    'M5B': (43.6572, -79.3789), 'M5C': (43.6515, -79.3733), 'M5E': (43.6447, -79.3676),
    'M5G': (43.6573, -79.3873), 'M5H': (43.6501, -79.3831), 'M5J': (43.6405, -79.3817),
    'M5K': (43.6472, -79.3817), 'M5L': (43.6487, -79.3789), 'M5M': (43.7332, -79.4197),
    'M5N': (43.7111, -79.4197), 'M5P': (43.6969, -79.4113), 'M5R': (43.6727, -79.4057),
    'M5S': (43.6627, -79.3972), 'M5T': (43.6532, -79.3958), 'M5V': (43.6289, -79.3944),
    'M5W': (43.6464, -79.3845), 'M5X': (43.6481, -79.3817), 'M6A': (43.7185, -79.4478),
    'M6B': (43.7090, -79.4478), 'M6C': (43.6937, -79.4281), 'M6E': (43.6890, -79.4506),
    'M6G': (43.6700, -79.4225), 'M6H': (43.6690, -79.4422), 'M6J': (43.6479, -79.4197),
    'M6K': (43.6368, -79.4281), 'M6L': (43.7137, -79.4900), 'M6M': (43.6911, -79.4760),
    'M6N': (43.6731, -79.4760), 'M6P': (43.6616, -79.4647), 'M6R': (43.6489, -79.4564),
    'M6S': (43.6515, -79.4760), 'M7A': (43.6621, -79.3920), 'M7Y': (43.6627, -79.3214),
    'M8V': (43.6056, -79.5013), 'M8W': (43.6024, -79.5434), 'M8X': (43.6536, -79.5069),
    'M8Y': (43.6363, -79.4985), 'M8Z': (43.6289, -79.5210), 'M9A': (43.6678, -79.5322),
    'M9B': (43.6509, -79.5547), 'M9C': (43.6435, -79.5772), 'M9L': (43.7564, -79.5547),
    'M9M': (43.7248, -79.5434), 'M9N': (43.7064, -79.5182), 'M9P': (43.6964, -79.5322),
    'M9R': (43.6889, -79.5547), 'M9V': (43.7395, -79.5884), 'M9W': (43.7064, -79.5941),
}

# ============================================================
# CELL 1: LOAD DATA
# ============================================================
def load_data_2025(package_id='2e54bc0e-4399-4076-b717-351df5918ae7'):
    local_csv = './data/SR2025.csv'
    if os.path.exists(local_csv):
        print(f"Loading from local: {local_csv}")
        return pd.read_csv(local_csv, on_bad_lines='skip', encoding='latin-1')
    print("Downloading 2025 data...")
    url = f"{CKAN_BASE}/api/3/action/package_show?id={package_id}"
    pkg = requests.get(url, timeout=30).json()
    res_url = None
    for r in pkg['result']['resources']:
        if '2025' in r.get('name', '') and r.get('format', '').lower() == 'zip':
            res_url = r['url']
            break
    resp = requests.get(res_url, timeout=60)
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    csv_name = [n for n in z.namelist() if n.endswith('.csv')][0]
    df = pd.read_csv(z.open(csv_name), on_bad_lines='skip', encoding='latin-1')
    df.to_csv(local_csv, index=False)
    return df

print("=" * 60)
print("TORONTO 311 SERVICE REQUEST ANALYSIS (2025)")
print("=" * 60)

df = load_data_2025()
print(f"Total Records: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head().to_string())

# ============================================================
# CELL 2: PREPROCESS
# ============================================================
df['created_date'] = pd.to_datetime(df['Creation Date'], errors='coerce')
df = df.dropna(subset=['created_date'])
df['year'] = df['created_date'].dt.year
df['month'] = df['created_date'].dt.month_name()
df['month_num'] = df['created_date'].dt.month
df['hour'] = df['created_date'].dt.hour
df['day_name'] = df['created_date'].dt.day_name()
df['date'] = df['created_date'].dt.date

print(f"\nDate Range: {df['created_date'].min().strftime('%Y-%m-%d')} to {df['created_date'].max().strftime('%Y-%m-%d')}")
print(f"Years in data: {sorted(df['year'].unique())}")
print(f"Months covered: {df['month_num'].nunique()}")
print(f"Unique Request Types: {df['Service Request Type'].nunique()}")
print(f"Unique Wards: {df['Ward'].nunique()}")

# Check for source/channel columns
possible_source_cols = [c for c in df.columns if 'source' in c.lower() or 'method' in c.lower() or 'channel' in c.lower()]
print(f"\nColumns containing 'Source', 'Method', or 'Channel': {possible_source_cols if possible_source_cols else 'NONE FOUND'}")
print("Note: This dataset covers ALL channels (phone, app, email, web) combined. No channel breakdown available.")

# ============================================================
# CELL 3: TOP 15 SERVICE REQUEST TYPES
# ============================================================
print("\n--- 1. Top 15 Service Request Types ---")
top_types = df['Service Request Type'].value_counts().nlargest(15).reset_index()
top_types.columns = ['Service Request Type', 'count']

fig_types = px.bar(top_types, x='count', y='Service Request Type', orientation='h',
                   title='Top 15 Service Request Types - Toronto 311 (2025)',
                   labels={'count': 'Number of Requests'},
                   color='count', color_continuous_scale='Viridis')
fig_types.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
fig_types.write_html(f'{OUTPUT_DIR}/top_request_types.html')
print("Saved: top_request_types.html")

# ============================================================
# CELL 4: STATUS DISTRIBUTION (PIE)
# ============================================================
print("\n--- 2. Status Distribution ---")
status_counts = df['Status'].value_counts().reset_index()
status_counts.columns = ['Status', 'count']

fig_status = px.pie(status_counts, values='count', names='Status',
                    title='Service Request Status Distribution - Toronto 311 (2025)',
                    color_discrete_sequence=px.colors.qualitative.Set2)
fig_status.write_html(f'{OUTPUT_DIR}/status_distribution.html')
print("Saved: status_distribution.html")
print(status_counts.to_string(index=False))

# ============================================================
# CELL 5: MONTHLY VOLUME TREND
# ============================================================
print("\n--- 3. Monthly Volume Trend ---")
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_vol = df['month'].value_counts().reindex(month_order).reset_index()
monthly_vol.columns = ['month', 'count']

fig_month = px.line(monthly_vol, x='month', y='count', markers=True,
                    title='Monthly Service Request Volume - Toronto 311 (2025)',
                    labels={'month': 'Month', 'count': 'Total Requests'})
fig_month.write_html(f'{OUTPUT_DIR}/monthly_volume.html')
print("Saved: monthly_volume.html")
peak_m = monthly_vol.loc[monthly_vol['count'].idxmax()]
low_m = monthly_vol.loc[monthly_vol['count'].idxmin()]
print(f"  Peak: {peak_m['month']} ({peak_m['count']:,.0f})")
print(f"  Lowest: {low_m['month']} ({low_m['count']:,.0f})")

# ============================================================
# CELL 6: HOURLY VOLUME (SHIFT PLANNING)
# ============================================================
print("\n--- 4. Hourly Volume (Shift Planning) ---")
hourly_vol = df['hour'].value_counts().sort_index().reset_index()
hourly_vol.columns = ['hour', 'count']

fig_hour = px.bar(hourly_vol, x='hour', y='count',
                  title='Request Volume by Hour of Day - Toronto 311 (2025)',
                  labels={'hour': 'Hour (24h)', 'count': 'Total Requests'},
                  color='count', color_continuous_scale='YlOrRd')
fig_hour.write_html(f'{OUTPUT_DIR}/hourly_volume.html')
print("Saved: hourly_volume.html")
peak_h = hourly_vol.loc[hourly_vol['count'].idxmax()]
print(f"  Peak Hour: {int(peak_h['hour']):02d}:00 ({peak_h['count']:,.0f} requests)")

# ============================================================
# CELL 7: DAY-HOUR DEMAND HEATMAP
# ============================================================
print("\n--- 5. Day-Hour Demand Heatmap ---")
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['day_name'] = pd.Categorical(df['day_name'], categories=day_order, ordered=True)
heatmap_data = df.groupby(['day_name', 'hour']).size().reset_index(name='count')
heatmap_pivot = heatmap_data.pivot(index='day_name', columns='hour', values='count').fillna(0)

plt.figure(figsize=(16, 5))
sns.heatmap(heatmap_pivot, cmap='YlOrRd', linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Number of Requests'})
plt.xlabel('Hour of Day')
plt.ylabel('')
plt.title('Service Request Volume by Day and Hour - Toronto 311 (2025)\nUse this for shift coverage and staffing')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/demand_heatmap.png', dpi=150)
plt.close()
print("Saved: demand_heatmap.png")

# ============================================================
# CELL 8: WARD BREAKDOWN
# ============================================================
print("\n--- 6. Request Volume by Ward ---")
ward_counts = df['Ward'].value_counts().reset_index()
ward_counts.columns = ['Ward', 'count']

fig_ward = px.bar(ward_counts, x='Ward', y='count',
                  title='Request Volume by Ward - Toronto 311 (2025)',
                  labels={'count': 'Number of Requests'},
                  color='count', color_continuous_scale='Blues')
fig_ward.update_xaxes(tickangle=45)
fig_ward.write_html(f'{OUTPUT_DIR}/ward_breakdown.html')
print("Saved: ward_breakdown.html")
print(f"  Top Ward: {ward_counts.iloc[0]['Ward']} ({ward_counts.iloc[0]['count']:,.0f})")

# ============================================================
# CELL 9: DIVISION WORKLOAD
# ============================================================
print("\n--- 7. Division Workload ---")
division_counts = df['Division'].value_counts().reset_index()
division_counts.columns = ['Division', 'Request Count']
total = division_counts['Request Count'].sum()
division_counts['Percentage'] = (division_counts['Request Count'] / total * 100).round(2)

fig_div = px.bar(division_counts, x='Request Count', y='Division', orientation='h',
                 text='Percentage',
                 title='Total Service Requests by City Division - Toronto 311 (2025)',
                 labels={'Request Count': 'Number of Requests'})
fig_div.update_traces(texttemplate='%{text}%', textposition='outside')
fig_div.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=200))
fig_div.write_html(f'{OUTPUT_DIR}/division_workload.html')
print("Saved: division_workload.html")
print(division_counts.to_string(index=False))

# ============================================================
# CELL 10: FOLIUM HEATMAP
# ============================================================
print("\n--- 8. Interactive Heatmap ---")
df_geo = df.dropna(subset=['First 3 Chars of Postal Code'])
geo_counts = df_geo['First 3 Chars of Postal Code'].value_counts().reset_index()
geo_counts.columns = ['FSA', 'count']

heat_data = []
for _, row in geo_counts.iterrows():
    fsa = str(row['FSA']).strip().upper()
    if fsa in TORONTO_FSA_COORDS:
        lat, lon = TORONTO_FSA_COORDS[fsa]
        heat_data.append([lat, lon, row['count']])

m = folium.Map(location=[43.7, -79.4], zoom_start=11, tiles='OpenStreetMap')
HeatMap(heat_data, radius=25, blur=15, max_zoom=13).add_to(m)
m.save(f'{OUTPUT_DIR}/311_heatmap.html')
print(f"Saved: 311_heatmap.html ({len(heat_data)} FSA areas)")

# ============================================================
# CELL 11: SPIKE / EVENT DAY DETECTION
# ============================================================
print("\n--- 9. Top 10 Spike Days (Weather/Event Triggers) ---")
daily_issue = df.groupby(['date', 'Service Request Type']).size().reset_index(name='daily_count')
top_spikes = daily_issue.sort_values('daily_count', ascending=False).head(10)
print(top_spikes.to_string(index=False))

# ============================================================
# CELL 12: OUTLIER DETECTION (Z-SCORE)
# ============================================================
print("\n--- 10. Daily Volume Outlier Detection ---")
daily_total = df.groupby('date').size().reset_index(name='total_requests')
daily_total['z_score'] = (daily_total['total_requests'] - daily_total['total_requests'].mean()) / daily_total['total_requests'].std()
outliers = daily_total[daily_total['z_score'].abs() > 2]

fig_outliers = px.scatter(daily_total, x='date', y='total_requests', color='z_score',
                          title='Daily Volume with Outlier Detection (Z-Score > 2) - Toronto 311 (2025)',
                          labels={'total_requests': 'Total Daily Requests', 'date': 'Date'},
                          color_continuous_scale='RdYlGn_r')
fig_outliers.write_html(f'{OUTPUT_DIR}/outlier_detection.html')
print(f"Saved: outlier_detection.html")
print(f"  Outlier days detected (z > 2): {len(outliers)}")
if len(outliers) > 0:
    print(f"  Highest spike: {outliers.loc[outliers['total_requests'].idxmax(), 'date']} ({outliers['total_requests'].max():,.0f} requests)")

# ============================================================
# CELL 13: CORRELATION HEATMAP
# ============================================================
print("\n--- 11. Request Type Correlation ---")
top_10_types = df['Service Request Type'].value_counts().nlargest(10).index
corr_df = df[df['Service Request Type'].isin(top_10_types)]
corr_pivot = corr_df.groupby(['date', 'Service Request Type']).size().unstack(fill_value=0)
correlation_matrix = corr_pivot.corr()

fig_corr = px.imshow(correlation_matrix, text_auto='.2f',
                     title='Correlation Heatmap: Do specific issues trigger together? (2025)',
                     color_continuous_scale='RdBu_r')
fig_corr.write_html(f'{OUTPUT_DIR}/correlation_heatmap.html')
print("Saved: correlation_heatmap.html")

# Top correlated pairs
corr_unstacked = correlation_matrix.unstack()
high_corr = corr_unstacked[corr_unstacked < 1.0].sort_values(ascending=False).drop_duplicates()
print("\nTop 5 Correlated Service Request Pairs:")
for (type1, type2), val in high_corr.head(5).items():
    print(f"  {val:.3f}  {type1} <-> {type2}")

# ============================================================
# CELL 14: WARD x ISSUE HEATMAP
# ============================================================
print("\n--- 12. Ward vs Issue Heatmap ---")
top_wards = df['Ward'].value_counts().nlargest(10).index
top_issues = df['Service Request Type'].value_counts().nlargest(10).index
df_filtered = df[df['Ward'].isin(top_wards) & df['Service Request Type'].isin(top_issues)]
ward_issue_map = pd.crosstab(df_filtered['Ward'], df_filtered['Service Request Type'])

fig_wi = px.imshow(ward_issue_map, text_auto=True, aspect="auto",
                   labels=dict(x="Request Type", y="Ward", color="Count"),
                   title="Which issues are reported in which Wards? - Toronto 311 (2025)")
fig_wi.update_xaxes(side="top", tickangle=45)
fig_wi.write_html(f'{OUTPUT_DIR}/ward_issue_heatmap.html')
print("Saved: ward_issue_heatmap.html")

max_issue = ward_issue_map.stack().idxmax()
print(f"  Most frequent localized issue: '{max_issue[1]}' in '{max_issue[0]}'")

# ============================================================
# CELL 15: TOP 3 WARD COMPARISON
# ============================================================
print("\n--- 13. Top 3 Ward Comparison ---")
top_3_wards = df['Ward'].value_counts().nlargest(3).index
df_top_3 = df[df['Ward'].isin(top_3_wards) & df['Service Request Type'].isin(top_issues)]
comparison = df_top_3.groupby(['Ward', 'Service Request Type']).size().reset_index(name='count')

fig_compare = px.bar(comparison, x='Service Request Type', y='count', color='Ward', barmode='group',
                     title='Request Type Comparison: Top 3 Most Active Wards (2025)',
                     labels={'count': 'Number of Requests'})
fig_compare.update_xaxes(tickangle=45)
fig_compare.write_html(f'{OUTPUT_DIR}/ward_comparison.html')
print("Saved: ward_comparison.html")

# ============================================================
# CELL 16: POTHOLE DEMAND FORECAST
# ============================================================
print("\n--- 14. Pothole Request Forecast ---")
pothole_df = df[df['Service Request Type'].str.contains('Pothole', case=False, na=False)].copy()
print(f"  Total pothole requests in 2025: {len(pothole_df):,}")

daily_potholes = pothole_df.groupby('date').size().reset_index(name='count')
daily_potholes['date'] = pd.to_datetime(daily_potholes['date'])
daily_potholes['date_ordinal'] = daily_potholes['date'].apply(lambda x: x.toordinal())

X = daily_potholes[['date_ordinal']]
y = daily_potholes['count']
model = LinearRegression()
model.fit(X, y)

last_date = daily_potholes['date'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
future_ordinals = np.array([[d.toordinal()] for d in future_dates])
predictions = np.maximum(model.predict(future_ordinals), 0)

forecast_df = pd.DataFrame({'date': future_dates, 'count': predictions, 'type': 'Forecast'})
history_df = daily_potholes[['date', 'count']].copy()
history_df['type'] = 'Actual'
combined = pd.concat([history_df, forecast_df], ignore_index=True)

fig_pred = px.scatter(combined, x='date', y='count', color='type',
                      title='Daily Pothole Requests: 2025 Actuals + 30-Day Forecast',
                      labels={'date': 'Date', 'count': 'Daily Requests', 'type': ''},
                      color_discrete_map={'Actual': '#1f77b4', 'Forecast': '#ff7f0e'})
fig_pred.write_html(f'{OUTPUT_DIR}/pothole_forecast.html')
print("Saved: pothole_forecast.html")
print(f"  Avg predicted per day (next 30 days): {np.mean(predictions):.1f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE - 14 OUTPUTS GENERATED")
print("=" * 60)
print(f"Records: {len(df):,}")
print(f"Date Range: {df['created_date'].min().strftime('%b %d')} to {df['created_date'].max().strftime('%b %d, %Y')}")
print(f"\nOutput files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(f'{OUTPUT_DIR}/{f}')
    print(f"  {f} ({size/1024:.0f} KB)")
