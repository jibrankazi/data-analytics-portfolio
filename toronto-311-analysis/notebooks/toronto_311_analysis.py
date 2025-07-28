"""
Toronto 311 Service Request Analysis

This script downloads and analyzes service request data from the City of Toronto's open data portal.
It uses the CKAN API to locate the most recent resource, loads the data into pandas,
processes timestamps, categorizes request types, and generates visualizations
including bar charts and maps.
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os # Import the os module for path operations

# CKAN base URL for Toronto open data
CKAN_BASE = "https://ckan0.cf.opendata.inter.prod-toronto.ca"


def get_package_show(package_id: str):
    """Call the CKAN package_show endpoint and return the JSON response."""
    url = f"{CKAN_BASE}/api/3/action/package_show?id={package_id}"
    response = requests.get(url)
    response.raise_for_status() # Raise an exception for HTTP errors
    return response.json()


def get_latest_resource_url(package_json: dict) -> str:
    """Identify the latest yearly ZIP resource for the service requests dataset and return its URL."""
    resources = package_json['result']['resources']
    
    # Debugging: Print all resource names to inspect
    print("Available resources:")
    for r in resources:
        print(f"- Name: {r.get('name')}, Format: {r.get('format')}, URL: {r.get('url')}")

    # Filter for resources that have format 'ZIP' and contain '311' or 'service request' in their name
    year_resources = [
        r for r in resources 
        if r.get('format', '').lower() == 'zip' and 
           ('311' in r.get('name', '').lower() or 'service request' in r.get('name', '').lower())
    ]
    
    if not year_resources:
        # Fallback: If no specific '311' or 'service request' zip is found, try to find any zip file.
        print("No specific '311' or 'service request' ZIP found. Attempting to find any ZIP resource.")
        year_resources = [r for r in resources if r.get('format', '').lower() == 'zip']
        
        if not year_resources:
            raise ValueError("No relevant ZIP resources found in the package.")
    
    # Sort by name, assuming the name contains the year in a sortable format (e.g., '2023_311_data.zip')
    latest = sorted(year_resources, key=lambda x: x.get('name', ''), reverse=True)[0]
    return latest['url']


def load_latest_data(package_id: str = '2e54bc0e-4399-4076-b717-351df5918ae7') -> pd.DataFrame:
    """Download the most recent 311 service request data and return it as a pandas DataFrame."""
    print(f"Fetching package info for ID: {package_id}")
    package_json = get_package_show(package_id)
    print("Identifying latest resource URL...")
    resource_url = get_latest_resource_url(package_json)
    print(f"Downloading data from: {resource_url}")
    response = requests.get(resource_url)
    response.raise_for_status() # Raise an exception for HTTP errors
    
    print("Extracting CSV from ZIP file...")
    z = zipfile.ZipFile(io.BytesIO(response.content))
    csv_names = [name for name in z.namelist() if name.lower().endswith('.csv')]
    if not csv_names:
        raise ValueError("No CSV file found in the downloaded ZIP archive.")
    csv_name = csv_names[0] # Assuming there's only one CSV or the first one is the correct one
    
    print(f"Loading {csv_name} into DataFrame...")
    df = pd.read_csv(z.open(csv_name))
    print("Data loaded successfully.")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the dataframe for analysis by parsing dates and deriving additional features."""
    print("Preprocessing data...")
    print(f"DataFrame columns: {df.columns.tolist()}")

    df = df.copy()
    
    # --- UPDATED: Use actual column names from the provided output ---
    DATE_COL = 'Creation Date'
    TYPE_COL = 'Service Request Type'
    STATUS_COL = 'Status' # New column for status distribution
    LAT_COL = 'Latitude' # These are not present in the current dataset
    LON_COL = 'Longitude' # These are not present in the current dataset

    # Check if the required columns exist and assign them
    if DATE_COL in df.columns:
        df['created_date'] = pd.to_datetime(df[DATE_COL], errors='coerce')
        df = df.dropna(subset=['created_date']) # Drop rows where date parsing failed
        df['year'] = df['created_date'].dt.year
        df['month'] = df['created_date'].dt.to_period('M')
        df['week_day'] = df['created_date'].dt.day_name()
    else:
        print(f"Warning: Column '{DATE_COL}' not found in DataFrame. Date-related features will be missing.")
        df['created_date'] = pd.NaT
        df['year'] = np.nan
        df['month'] = pd.NaT
        df['week_day'] = np.nan
        
    if TYPE_COL in df.columns:
        df['request_type'] = df[TYPE_COL]
    else:
        print(f"Warning: Column '{TYPE_COL}' not found in DataFrame. Request type will be 'Unknown'.")
        df['request_type'] = 'Unknown'

    if STATUS_COL in df.columns:
        df['status'] = df[STATUS_COL]
    else:
        print(f"Warning: Column '{STATUS_COL}' not found in DataFrame. Status distribution cannot be plotted.")
        df['status'] = 'Unknown'
        
    # Handle missing Latitude/Longitude columns for mapping
    if LAT_COL not in df.columns or LON_COL not in df.columns:
        print(f"Note: Columns '{LAT_COL}' or '{LON_COL}' are not found in this dataset. The interactive map cannot be created.")
        # Ensure these columns exist, even if filled with NaN, to prevent KeyError in create_map
        df[LAT_COL] = np.nan
        df[LON_COL] = np.nan
    else:
        # Ensure Latitude and Longitude are numeric if they exist
        df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors='coerce')
        df[LON_COL] = pd.to_numeric(df[LON_COL], errors='coerce')
        
    print("Preprocessing complete.")
    return df


def plot_top_types(df: pd.DataFrame, n: int = 10) -> None:
    """Generate and save a bar chart of the most common service request types."""
    print(f"Generating top {n} request types plot...")
    if 'request_type' not in df.columns or df['request_type'].empty or df['request_type'].eq('Unknown').all():
        print("Cannot plot top types: 'request_type' column is missing, empty, or all values are 'Unknown'.")
        return

    top_counts = df['request_type'].value_counts().head(n)
    if top_counts.empty:
        print("No data to plot for top request types.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_counts.values, y=top_counts.index, palette='viridis', hue=top_counts.index, legend=False)
    plt.xlabel('Number of Requests')
    plt.ylabel('Service Request Type')
    plt.title(f'Top {n} Service Request Types')
    plt.tight_layout()
    plt.savefig('./images/top_request_types.png')
    plt.close()
    print("Top request types plot saved.")


def plot_monthly_trend(df: pd.DataFrame) -> None:
    """Plot and save a line chart showing the volume of requests per month."""
    print("Generating monthly trend plot...")
    if 'month' not in df.columns or df['month'].empty or df['month'].isnull().all():
        print("Cannot plot monthly trend: 'month' column is missing, empty, or all values are NaN.")
        return

    monthly_counts = df.groupby('month').size().reset_index(name='count')
    if monthly_counts.empty:
        print("No data to plot for monthly trend.")
        return

    monthly_counts['month'] = monthly_counts['month'].astype(str)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='month', y='count', data=monthly_counts, marker='o')
    plt.xticks(rotation=45)
    plt.xlabel('Month')
    plt.ylabel('Number of Requests')
    plt.title('Monthly Service Request Volume')
    plt.tight_layout()
    plt.savefig('./images/monthly_volume.png')
    plt.close()
    print("Monthly trend plot saved.")


def plot_status_distribution(df: pd.DataFrame) -> None:
    """Generate and save a bar chart of the service request status distribution."""
    print("Generating status distribution plot...")
    if 'status' not in df.columns or df['status'].empty or df['status'].eq('Unknown').all():
        print("Cannot plot status distribution: 'status' column is missing, empty, or all values are 'Unknown'.")
        return

    status_counts = df['status'].value_counts()
    if status_counts.empty:
        print("No data to plot for status distribution.")
        return

    plt.figure(figsize=(8, 5))
    sns.barplot(x=status_counts.index, y=status_counts.values, palette='plasma', hue=status_counts.index, legend=False)
    plt.xlabel('Status')
    plt.ylabel('Number of Requests')
    plt.title('Service Request Status Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('./images/status_distribution.png')
    plt.close()
    print("Status distribution plot saved.")


def create_map(df: pd.DataFrame) -> None:
    """Create an interactive density map of service requests using Plotly and save it as an HTML file."""
    print("Generating interactive density map...")
    # The preprocess function now ensures these columns exist, even if filled with NaN.
    # We only proceed if there's actual non-NaN geographical data.
    df_geo = df.dropna(subset=['Latitude', 'Longitude'])
    
    if df_geo.empty:
        print("No valid geographical data (Latitude/Longitude) to create map after dropping NaNs. Map will not be generated.")
        return
        
    # Ensure Latitude and Longitude are numeric (already done in preprocess, but good to be safe)
    df_geo['Latitude'] = pd.to_numeric(df_geo['Latitude'], errors='coerce')
    df_geo['Longitude'] = pd.to_numeric(df_geo['Longitude'], errors='coerce')
    df_geo = df_geo.dropna(subset=['Latitude', 'Longitude'])

    if df_geo.empty:
        print("No numeric geographical data after conversion and dropping NaNs to create map. Map will not be generated.")
        return

    # Calculate mean only if df_geo is not empty
    center_lat = df_geo['Latitude'].mean() if not df_geo['Latitude'].empty else 43.6532
    center_lon = df_geo['Longitude'].mean() if not df_geo['Longitude'].empty else -79.3832

    fig = px.density_mapbox(df_geo, lat='Latitude', lon='Longitude',
                            radius=5,
                            center=dict(lat=center_lat, lon=center_lon),
                            zoom=10,
                            mapbox_style='stamen-terrain',
                            title='Service Request Density Map')
    fig.write_html('./images/311_density_map.html')
    print("Interactive density map saved.")


def main() -> None:
    """Main function to run the analysis workflow."""
    # Ensure the images directory exists
    output_dir = './images'
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        try:
            os.makedirs(output_dir)
        except PermissionError:
            print(f"Permission denied: Could not create directory '{output_dir}'.")
            print("Please ensure the script has write permissions to the current directory, or create the 'images' folder manually.")
            return # Exit if directory cannot be created due to permissions
    
    try:
        df = load_latest_data()
        df = preprocess(df)
        plot_top_types(df)
        plot_monthly_trend(df)
        plot_status_distribution(df) # Call the new status distribution plot function
        create_map(df)
        print('311 analysis complete. Visualizations saved to images folder.')
    except requests.exceptions.RequestException as e:
        print(f"Network or API error: {e}. Please check your internet connection or the CKAN API status.")
    except zipfile.BadZipFile as e:
        print(f"Error unzipping file: {e}. The downloaded file might be corrupted.")
    except ValueError as e:
        print(f"Data processing error: {e}. Check data structure or resource availability.")
    except KeyError as e:
        print(f"Missing expected column: {e}. The dataset structure might have changed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    main()
