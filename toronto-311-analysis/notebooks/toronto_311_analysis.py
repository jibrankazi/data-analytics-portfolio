notebooks/toronto_311_analysis.py"""
Toronto 311 Service Request Analysis

This script downloads and analyzes service request data from the City of Toronto's open data portal. It uses the CKAN API to locate the most recent resource, loads the data into pandas, processes timestamps, categorizes request types, and generates visualizations including bar charts and maps.
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# CKAN base URL for Toronto open data
CKAN_BASE = "https://ckan0.cf.opendata.inter.prod-toronto.ca"


def get_package_show(package_id: str):
    """Call the CKAN package_show endpoint and return the JSON response."""
    url = f"{CKAN_BASE}/api/3/action/package_show?id={package_id}"
    return requests.get(url).json()


def get_latest_resource_url(package_json: dict) -> str:
    """Identify the latest yearly ZIP resource for the service requests dataset and return its URL."""
    resources = package_json['result']['resources']
    year_resources = [r for r in resources if r['name'].lower().endswith('.zip')]
    latest = sorted(year_resources, key=lambda x: x['name'], reverse=True)[0]
    return latest['url']


def load_latest_data(package_id: str = '2e54bc0e-4399-4076-b717-351df5918ae7') -> pd.DataFrame:
    """Download the most recent 311 service request data and return it as a pandas DataFrame."""
    package_json = get_package_show(package_id)
    resource_url = get_latest_resource_url(package_json)
    response = requests.get(resource_url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    csv_name = [name for name in z.namelist() if name.lower().endswith('.csv')][0]
    df = pd.read_csv(z.open(csv_name))
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the dataframe for analysis by parsing dates and deriving additional features."""
    df = df.copy()
    df['created_date'] = pd.to_datetime(df['Service Request Creation Date and Time'])
    df['year'] = df['created_date'].dt.year
    df['month'] = df['created_date'].dt.to_period('M')
    df['week_day'] = df['created_date'].dt.day_name()
    df['request_type'] = df['Original Service Request Type']
    return df


def plot_top_types(df: pd.DataFrame, n: int = 10) -> None:
    """Generate and save a bar chart of the most common service request types."""
    top_counts = df['request_type'].value_counts().head(n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_counts.values, y=top_counts.index, palette='viridis')
    plt.xlabel('Number of Requests')
    plt.ylabel('Service Request Type')
    plt.title(f'Top {n} Service Request Types')
    plt.tight_layout()
    plt.savefig('../images/top_request_types.png')
    plt.close()


def plot_monthly_trend(df: pd.DataFrame) -> None:
    """Plot and save a line chart showing the volume of requests per month."""
    monthly_counts = df.groupby('month').size().reset_index(name='count')
    monthly_counts['month'] = monthly_counts['month'].astype(str)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='month', y='count', data=monthly_counts, marker='o')
    plt.xticks(rotation=45)
    plt.xlabel('Month')
    plt.ylabel('Number of Requests')
    plt.title('Monthly Service Request Volume')
    plt.tight_layout()
    plt.savefig('../images/monthly_volume.png')
    plt.close()


def create_map(df: pd.DataFrame) -> None:
    """Create an interactive density map of service requests using Plotly and save it as an HTML file."""
    df_geo = df.dropna(subset=['Latitude', 'Longitude'])
    if df_geo.empty:
        return
    fig = px.density_mapbox(df_geo, lat='Latitude', lon='Longitude',
                            radius=5,
                            center=dict(lat=df_geo['Latitude'].mean(), lon=df_geo['Longitude'].mean()),
                            zoom=10,
                            mapbox_style='stamen-terrain',
                            title='Service Request Density Map')
    fig.write_html('../images/311_density_map.html')


def main() -> None:
    df = load_latest_data()
    df = preprocess(df)
    plot_top_types(df)
    plot_monthly_trend(df)
    create_map(df)
    print('311 analysis complete. Visualisations saved to images folder.')


if __name__ == '__main__':
    main()
