import os, re, shutil, warnings
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_squared_error, r2_score, accuracy_score
from xgboost import XGBClassifier
from scipy.stats import mode, stats
import matplotlib.pyplot as plt
import geopandas as gpd

from config import Config

warnings.filterwarnings('ignore')

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def save_dataset(df, folder, file_name):
    create_folder_if_not_exists(folder)
    file_path = os.path.join(folder, file_name)
    df.to_csv(file_path, index=False)

def load_csv_with_cols(csv_path, name_list):
    return pd.read_csv(csv_path, names=name_list if name_list else None)

def download_file(url, folder):
    file_name = url.split('/')[-1]
    file_path = os.path.join(folder, file_name)
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")        

def find_data_start_row(file_path):
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if line.startswith("   "):
                return i
    return 0

def extract_links(url, href_contains):
    """
    Extracts and compiles a list of URLs from a given webpage where the 'href' attribute contains a specific pattern.
    
    Parameters:
    - url (str): The URL of the webpage to scrape.
    - href_contains (str): A substring that must be present in the 'href' attributes of the links to be included.
    
    Returns:
    - urls (list): A list of fully qualified URLs that match the criteria.
    """
    urls = []
    try:
        response = requests.get(url)
        if response.ok:
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)
            for link in links:
                if href_contains in link['href']:
                    full_url = urljoin(url, link['href'])
                    urls.append(full_url)
        else:
            print(f"Error accessing page: Status code {response.status_code}")
    except requests.RequestException as e:
        print(f"Error during requests to {url} : {str(e)}") 
    
    return urls

def download_main_dataset(url, output_folder, pattern):
    """
    Downloads files from URLs that contain a specific pattern and saves them in a designated folder with a timestamp.
    
    Parameters:
    - url (str): The URL to scrape for file links.
    - output_folder (str): The base path where the downloaded files should be saved.
    - pattern (str): The substring to look for in URLs to identify downloadable files.
    """
    timestamp_suffix = datetime.now().strftime("_%Y%m%d_%H%M%S")
    output_folder_with_timestamp = f"{output_folder}{timestamp_suffix}"
    csv_links = extract_links(url, pattern)
    create_folder_if_not_exists(output_folder_with_timestamp)

    for link in csv_links:
        # Pass the modified folder path with the timestamp to the download function
        download_file(link, output_folder_with_timestamp)
        
def txt_to_csv(file_path, output_folder):    #Dataset 1
    """
    Converts a text file to CSV format, selecting specific columns and saving the result in the specified output folder.
    
    Parameters:
    - file_path (str): The path to the input .txt file.
    - output_folder (str): The folder where the output .csv file should be saved.
    
    Returns:
    - output_folder (str): The path to the folder containing the newly created .csv file.
    """
    start_row = find_data_start_row(file_path)
    df = pd.read_csv(file_path, skiprows=start_row, delim_whitespace=True, usecols=[0, 1, 2, 3, 4, 5, 6])
    df.reset_index(drop=True, inplace=True)
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_folder, file_name.replace('.txt', '.csv'))
    df.to_csv(output_file, index=False)
    return output_folder

def convert_txts_to_csvs(input_folder, output_folder):    #Dataset 1
    """
    Converts all .txt files in a given folder to .csv format and saves them in an output folder.
    
    Parameters:
    - input_folder (str): The folder containing the .txt files to convert.
    - output_folder (str): The folder where the converted .csv files will be saved.
    """
    create_folder_if_not_exists(output_folder)
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        if file_path.endswith('.txt'):
            txt_to_csv(file_path, output_folder)

def clean_and_rename_csv_in_folder(input_folder, output_folder, remove_rows_criteria=[], special_chars_to_remove=[]):   #Dataset 1
    """
    Cleans CSV files by removing specified rows and special characters, then saves the cleaned files in an output folder.
    
    Parameters:
    - input_folder (str): The folder containing the .csv files to clean.
    - output_folder (str): The folder where the cleaned .csv files will be saved.
    - remove_rows_criteria (list): Criteria for rows to remove, specified as a list of dictionaries.
    - special_chars_to_remove (list): A list of special characters to remove from the data.
    """
    create_folder_if_not_exists(output_folder)
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            df = df.drop(0).reset_index(drop=True)
            df.columns = ['yyyy', 'mm', 'tmax (degC)', 'tmin (degC)', 'af (days)', 'rain (mm)', 'sun (hours)']
            df.replace('---', pd.NA, inplace=True)
            for criterion in remove_rows_criteria:
                for col_name, value in criterion.items():
                    df = df[df[col_name] != value]
            for col in df.columns:
                if df[col].dtype == 'object':
                    for char in special_chars_to_remove:
                        df[col] = df[col].str.replace(char, "", regex=True)
            output_file_path = os.path.join(output_folder, file_name)
            df.to_csv(output_file_path, index=False)

def extract_and_save_coordinates(input_folder, output_folder):  #Dataset 1
    """
    Extracts the latest latitude and longitude values from text files and saves them in CSV format.

    Parameters:
    - input_folder (str): The folder containing the .txt files to extract coordinates from.
    - output_folder (str): The folder where the extracted coordinates will be saved.
    """
    create_folder_if_not_exists(output_folder)
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as file:
                text = file.read()
            lat_lon_pattern = re.compile(r'Lat\s*([+-]?\d+\.\d+)\s*Lon\s*([+-]?\d+\.\d+)')
            matches = lat_lon_pattern.findall(text)
            if matches:
                latest_lat, latest_lon = matches[-1]
            else:
                latest_lat, latest_lon = None, None
            df = pd.DataFrame({
                'Station': [file_name.replace("data.txt", "")],
                'Latitude': [latest_lat],
                'Longitude': [latest_lon]
            })
            output_file_name = os.path.splitext(file_name)[0].replace("data", "") + '_coordinates.csv'
            output_file_path = os.path.join(output_folder, output_file_name)
            df.to_csv(output_file_path, index=False)

def process_happiness_excel(file_path, sheet_name, word):   #Dataset 2
    """
    Processes the data from an Excel file containing happiness statistics and returns a cleaned DataFrame.

    Parameters:
    - file_path (str): The path to the Excel file to process.
    - sheet_name (str): The name of the sheet to read from the Excel file.
    - word (str): The word to search for in the Excel file to identify the start of the data. 
    """
    # Load the CSV data, ignoring the first column which is just an index
    data = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=2)

    col_names = [
        "Codes", "Area names", "Region", "Location", "Total % 0-4", "Total % 5-6", "Total % 7-8", "Total % 9-10", 
        "Total Average rating", "Total Standard deviation", "0-4 CV", "0-4 Lower limit", "0-4 Upper limit", 
        "5-6 CV", " 5-6 Lower limit", "5-6 Upper limit", "7-8 CV", "7-8 Lower limit", "7-8 Upper limit", 
        "9-10 CV", "9-10 Lower limit", "9-10 Upper limit", "Total CV", "Total Lower limit", "Total Upper limit", 
        "Sample size"
    ]
    for i in range(len(data)):
        if word in str(data.iloc[i, 0]):
            data.columns = data.iloc[i]
            data = data.iloc[i+1:]
            break
    data = data.dropna(how="all")
    data.reset_index(drop=True, inplace=True)
    data.columns = data.columns.str.strip()
    if "2012" in file_path:
        new_columns = col_names
    else:
        del col_names[9]
        new_columns = col_names
    data.columns = new_columns[:len(data.columns)]

    return data

def process_excel_folder(input_folder, output_folder, sheet_name, word):   #Dataset 2
    """
    Processes all Excel files in a folder containing happiness statistics and saves the cleaned data as CSV files.

    Parameters:
    - input_folder (str): The folder containing the Excel files to process.
    - output_folder (str): The folder where the processed CSV files will be saved.
    - sheet_name (str): The name of the sheet to read from the Excel files.
    - word (str): The word to search for in the Excel files to identify the start of the data. 
    """
    create_folder_if_not_exists(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(".xls"):
            file_path = os.path.join(input_folder, filename)
            updated_filename = filename[:-3] + "xlsx"
            updated_file_path = os.path.join(input_folder, updated_filename)
            shutil.copy(file_path, updated_file_path)
            updated_file_path = os.path.join(input_folder, updated_filename)
            processed_data = process_happiness_excel(updated_file_path, sheet_name, word)
            output_file_path = os.path.join(output_folder, filename.replace(".xls", ".csv"))
            processed_data.to_csv(output_file_path, index=False)

def clean_census_rows(input_folder, output_folder):   #Dataset 2
    """
    Cleans the rows of all CSV files in a folder containing census data and saves the cleaned files in an output folder.

    Parameters:
    - input_folder (str): The folder containing the .csv files to clean.
    - output_folder (str): The folder where the cleaned .csv files will be saved.
    """
    create_folder_if_not_exists(output_folder)
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            df = df[~df['Codes'].str.contains(' ', na=True)]
            df.replace('x', np.nan, inplace=True)
            df.replace('#', np.nan, inplace=True)
            df.replace('x ', np.nan, inplace=True)
            output_file_path = os.path.join(output_folder, file_name)
            df.to_csv(output_file_path, index=False)

def combine_dataset_2_average(input_folder, output_folder, drop_columns=["Total Standard deviation"], no_group=False):  #Dataset 2
    """
    Combines all CSV files in a folder containing happiness statistics, calculates averages, and saves the combined data.

    Parameters:
    - input_folder (str): The folder containing the .csv files to combine.
    - output_folder (str): The folder where the combined .csv file will be saved.
    - drop_columns (list): Columns to drop from the combined dataset.
    - no_group (bool): If True, the data will not be grouped and averaged.
    """
    all_codes_dfs = []
    
    # Step 1: Collect 'Codes' from each file
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            df = pd.read_csv(file_path)
            df.replace('x', np.nan, inplace=True)
            all_codes_dfs.append(df[['Codes']].drop_duplicates())
    
    # Step 2: Find distinct 'Codes'
    distinct_codes_df = pd.concat(all_codes_dfs).drop_duplicates().reset_index(drop=True)
    
    all_data_dfs = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            df = pd.read_csv(file_path)
            df.replace('x', np.nan, inplace=True)
            df = df.drop(columns=drop_columns, errors='ignore')
            all_data_dfs.append(df)
    
    # Concatenate all DataFrames vertically and sort by 'Codes'
    concatenated_df = pd.concat(all_data_dfs, axis=0).sort_values(by='Codes')
    concatenated_df["Sample size"] = concatenated_df["Sample size"].astype(float)

    # If no_group is True, skip the grouping and averaging process
    if no_group:
        final_df = concatenated_df
    else:
        # Step 5: Compute final values based on your criteria
        def compute_final_values(group):
            numeric_cols = group.select_dtypes(include=np.number).columns
            non_numeric_cols = group.select_dtypes(exclude=np.number).columns.drop('Codes')

            # For numeric columns, calculate mean
            group[numeric_cols] = group[numeric_cols].mean()

            # For non-numeric columns, keep the first value
            group[non_numeric_cols] = group[non_numeric_cols].apply(lambda x: x.dropna().head(1).item() if not x.dropna().empty else np.nan)

            return group.head(1)

        final_df = concatenated_df.groupby('Codes', as_index=False).apply(compute_final_values).reset_index(drop=True)

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define the output file path within the output folder
    output_file_path = os.path.join(output_folder, "combined_dataset.csv" if not no_group else "merged_dataset.csv")

    # Save the final DataFrame
    final_df.to_csv(output_file_path, index=False)

def impute_missing_values(file_path, columns_to_impute, group_by=None):   #Both Datasets
    """ 
    Imputes missing values in a DataFrame by filling them with the mean of the respective column.
    
    Parameters:
    - file_path (str): The path to the input CSV file.
    - columns_to_impute (list): A list of column names to impute missing values for.
    - group_by (str): A column name to group by before imputing missing values.
    """
    df = pd.read_csv(file_path)
    
    if group_by:
        for col in columns_to_impute:
            df[col] = df.groupby(group_by)[col].transform(lambda x: x.fillna(x.mean()))
    else:
        for col in columns_to_impute:
            df[col] = df[col].fillna(df[col].mean())
    
    return df

def impute_missing_values_in_folder(input_folder, output_folder, columns_to_impute, group_by=None):   #Both Datasets
    """
    Imputes missing values in all CSV files in a folder and saves the cleaned files in an output folder.

    Parameters:
    - input_folder (str): The folder containing the .csv files to clean.
    - output_folder (str): The folder where the cleaned .csv files will be saved.
    - columns_to_impute (list): A list of column names to impute missing values for.
    - group_by (str): A column name to group by before imputing missing values.
    """
    create_folder_if_not_exists(output_folder)
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if file_path.endswith('.csv'):
            try:
                df = impute_missing_values(file_path, columns_to_impute, group_by)
                output_file_path = os.path.join(output_folder, file_name)
                df.to_csv(output_file_path, index=False)
            except Exception as e:
                print(f"Error occurred in file: {file_name}")
                print(str(e))

def generate_weather_dataset(input_folder, use_station_average=True, drop_rows_criteria=["stations_total_average"]):   #Dataset 2
    """ 
    Generates a weather dataset by combining and averaging the data from multiple CSV files.
    
    Parameters:
    - input_folder (str): The folder containing the CSV files to combine.
    - use_station_average (bool): If True, the data will be averaged by station. If False, the data will be averaged by station, year, and month.
    - drop_rows_criteria (list): A list of criteria to drop rows based on the 'station' column.
    """
    all_data = []

    # Step 1: Concatenate all CSVs with a 'station' column
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            station_name = os.path.splitext(file_name)[0].replace("data", "")
            temp_df = pd.read_csv(os.path.join(input_folder, file_name))
            temp_df['station'] = station_name
            all_data.append(temp_df)

    # Combine into one DataFrame
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Ensure numeric columns are treated as such
    numeric_cols = ['tmax (degC)', 'tmin (degC)', 'af (days)', 'rain (mm)', 'sun (hours)']
    df_combined[numeric_cols] = df_combined[numeric_cols].apply(pd.to_numeric, errors='coerce')

    if use_station_average:
        # Step 2: Group by 'station' and calculate averages
        df_result = df_combined.groupby('station')[numeric_cols].mean().reset_index()
    else:
        # Step 3: Group by 'station', 'yyyy', and 'mm' then calculate averages
        df_result = df_combined.groupby(['station', 'mm'])[numeric_cols].mean().reset_index()
    df_result = df_result.drop(df_result[df_result['station'].isin(drop_rows_criteria)].index)
    return df_result

def find_elbow_point(wcss):   #Dataset 1
    """
    Finds the elbow point in a list of within-cluster sum of squares (WCSS) values.

    Parameters:
    - wcss (list): A list of WCSS values to analyze.
    """
    # Calculate the second derivative of the wcss list
    second_derivative = np.diff(wcss, n=2)
    # The elbow point is where the second derivative is maximum (in absolute value)
    elbow_point = np.argmax(np.abs(second_derivative)) + 2  # Adding 2 because np.diff reduces the length by 1 for each differentiation
    return elbow_point

def apply_clustering(df, num_clusters_range, optimal_num_clusters=None):   #Dataset 1
    """
    Applies K-means clustering to the data and visualizes the results to determine the optimal number of clusters.

    Parameters:
    - df (DataFrame): The input DataFrame to cluster.
    - num_clusters_range (range): A range of values to test for the number of clusters.
    - optimal_num_clusters (int): The optimal number of clusters to use, if known.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numerical_cols]
    
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    wcss = []
    silhouette_scores = []
    davies_bouldin_scores = []

    for n_clusters in num_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_scaled)
        labels = kmeans.labels_
        wcss.append(kmeans.inertia_)
        
        if n_clusters > 1:  # These metrics require at least 2 clusters to be meaningful
            silhouette_scores.append(silhouette_score(X_scaled, labels))
            davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))

    # Visualizing the metrics to help determine the optimal number of clusters
    plt.figure(figsize=(18, 5))
    
    # Plot WCSS
    plt.subplot(1, 3, 1)
    plt.plot(num_clusters_range, wcss, marker='o', linestyle='-', color='blue')
    plt.title('Elbow Method (WCSS)')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    
    # Plot Silhouette Score
    plt.subplot(1, 3, 2)
    plt.plot(num_clusters_range[1:], silhouette_scores, marker='o', linestyle='-', color='green')
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    
    # Plot Davies-Bouldin Score
    plt.subplot(1, 3, 3)
    plt.plot(num_clusters_range[1:], davies_bouldin_scores, marker='o', linestyle='-', color='red')
    plt.title('Davies-Bouldin Score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Davies-Bouldin Score')
    
    plt.tight_layout()
    plt.show()

    optimal_num_clusters = optimal_num_clusters if optimal_num_clusters else find_elbow_point(wcss)

    # Apply clustering with the chosen optimal number of clusters
    kmeans_optimal = KMeans(n_clusters=optimal_num_clusters, random_state=42, n_init=10).fit(X_scaled)
    df['cluster'] = kmeans_optimal.labels_
    
    return df

def add_coordinates_to_df(df, locations_folder, join_on):   #Dataset 1
    locations_df = pd.DataFrame()

    # Concatenate all location files into one DataFrame
    for file_name in os.listdir(locations_folder):
        if file_name.endswith('.csv'):
            # Read the coordinates file
            temp_df = pd.read_csv(os.path.join(locations_folder, file_name))
            # Concatenate to the main locations DataFrame
            locations_df = pd.concat([locations_df, temp_df], ignore_index=True)

    # Ensure there are no duplicate columns in the locations DataFrame before merging
    locations_df = locations_df.drop_duplicates(subset=join_on[1])

    # Merge the main df with the locations DataFrame on the specified columns
    df_with_coords = df.merge(locations_df, how='left', left_on=join_on[0], right_on=join_on[1])

    return df_with_coords

def stats_by_group(df, group_col):   #Dataset 1
    # Ensure the group column exists in the DataFrame
    if group_col not in df.columns:
        raise ValueError(f"The column '{group_col}' does not exist in the DataFrame.")

    # Group the DataFrame by the specified column and compute descriptive statistics
    grouped_stats = df.groupby(group_col).describe()
    
    return grouped_stats

def plot_clusters_on_map(df, map_col='cluster', lat_col='Latitude', lon_col='Longitude'):   #Both Datasets
    # Create a GeoDataFrame from the DataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]))

    # Define the bounds for the UK (these are approximate and may need adjusting)
    uk_bounds = {
        "min_lon": -10.5, "max_lon": 1.8,
        "min_lat": 49.8, "max_lat": 60.9
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    uk = world.cx[uk_bounds["min_lon"]:uk_bounds["max_lon"], uk_bounds["min_lat"]:uk_bounds["max_lat"]]
    uk.plot(ax=ax, color='white', edgecolor='black')

    # Plot the clusters
    gdf.plot(ax=ax, markersize=50, column=map_col, legend=True, cmap='viridis')

    # Annotate each point with its cluster label
    for idx, row in gdf.iterrows():
        ax.annotate(text=row[map_col], xy=(row[lon_col], row[lat_col]),
                    xytext=(3,3), # Position label slightly off of the point
                    textcoords="offset points", # Offset (3,3) points
                    fontsize=8, color='darkred')

    # Set axis limits to zoom in to the UK
    ax.set_xlim(uk_bounds["min_lon"], uk_bounds["max_lon"])
    ax.set_ylim(uk_bounds["min_lat"], uk_bounds["max_lat"])

    plt.show()

def plot_cluster_distribution_on_map(df, lat_col='Latitude', lon_col='Longitude', cluster_col='cluster'):   #Both Datasets
    """ 
    Plots the distribution of clusters on a map, showing the percentage of each cluster at each location.
    
    Parameters:
    - df (DataFrame): The input DataFrame containing the cluster data.
    - lat_col (str): The name of the column containing latitude values.
    - lon_col (str): The name of the column containing longitude values.
    - cluster_col (str): The name of the column containing cluster labels.
    """
    # Aggregate data by location and cluster and count the occurrences
    agg_df = df.groupby([lat_col, lon_col, cluster_col]).size().reset_index(name='counts')

    # Calculate the total counts by location
    total_counts = agg_df.groupby([lat_col, lon_col])['counts'].sum().reset_index(name='total_counts')

    # Merge back to get the total counts per location
    agg_df = agg_df.merge(total_counts, on=[lat_col, lon_col])

    # Calculate the percentage of each cluster at each location
    agg_df['percentage'] = (agg_df['counts'] / agg_df['total_counts'] * 100).round(2)
    
    # Create labels for plotting
    agg_df['label'] = 'C' + agg_df[cluster_col].astype(str) + ': ' + agg_df['percentage'].astype(str) + '%'
    labels_df = agg_df.groupby([lat_col, lon_col])['label'].apply(list).reset_index(name='labels')

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        labels_df, 
        geometry=gpd.points_from_xy(labels_df[lon_col], labels_df[lat_col])
    )

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.plot(ax=ax, color='lightgrey', edgecolor='black')
    
    # Define the bounds for the UK
    uk_bounds = {"min_lon": -10.5, "max_lon": 1.8, "min_lat": 49.8, "max_lat": 60.9}
    ax.set_xlim(uk_bounds["min_lon"], uk_bounds["max_lon"])
    ax.set_ylim(uk_bounds["min_lat"], uk_bounds["max_lat"])

    # Plot each location with its cluster distribution label
    for idx, row in gdf.iterrows():
        ax.scatter(row.geometry.x, row.geometry.y, color='blue', alpha=0.6, edgecolor='black')
        ax.text(row.geometry.x, row.geometry.y, ' | '.join(row['labels']), fontsize=4.5,  # Adjusted fontsize here
                verticalalignment='bottom', horizontalalignment='right')
    
    plt.show()

def determine_dominant_cluster_manual(df, lat_col='Latitude', lon_col='Longitude', cluster_col='cluster'):   #Dataset 1
    """
    Determines the dominant cluster for each unique location in the DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame containing the cluster data.
    - lat_col (str): The name of the column containing latitude values.
    - lon_col (str): The name of the column containing longitude values.
    - cluster_col (str): The name of the column containing cluster labels.
    """
    # Group the DataFrame by latitude and longitude
    grouped = df.groupby([lat_col, lon_col])

    # Initialize a list to hold the data for the dominant cluster DataFrame
    dominant_clusters_data = []

    # Iterate through each group
    for (lat, lon), group in grouped:
        # Calculate the most frequent cluster for this group
        most_frequent_cluster = group[cluster_col].mode()[0]  # mode()[0] ensures we're getting the first mode in case of ties

        # Calculate the count of the most frequent cluster
        count = group[group[cluster_col] == most_frequent_cluster].shape[0]

        # Append the information to our list
        dominant_clusters_data.append({
            lat_col: lat,
            lon_col: lon,
            'dominant_cluster': most_frequent_cluster,
            'count': count
        })

    # Create a DataFrame from the collected data
    dominant_clusters_df = pd.DataFrame(dominant_clusters_data)

    return dominant_clusters_df

def apply_dbscan_clustering(df, eps=0.8, min_samples=5):   #Dataset 1
    """
    Applies DBSCAN clustering to the data and calculates silhouette and Davies-Bouldin scores.

    Parameters:
    - df (DataFrame): The input DataFrame to cluster.
    - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numerical_cols]
    
    # Scale the data
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    labels = dbscan.labels_
    
    # Add the cluster labels to the dataframe
    df['cluster'] = labels
    
    # Calculate silhouette and Davies-Bouldin scores for the clustering
    # Note: These scores require more than one cluster to be meaningful, 
    # excluding noise points labeled as -1 by DBSCAN.
    if len(set(labels)) > 2:  # More than 1 cluster excluding noise
        silhouette_avg = silhouette_score(X_scaled, labels)
        davies_bouldin_avg = davies_bouldin_score(X_scaled, labels)
        print(f"Silhouette Score: {silhouette_avg}")
        print(f"Davies-Bouldin Score: {davies_bouldin_avg}")
    else:
        print("Not enough clusters to calculate silhouette and Davies-Bouldin scores.")
    
    return df

def generate_latitude_ds(df, number_groups):   #Dataset 1
    """
    Generates latitude groups based on quantiles and adds them to the DataFrame.
    
    Parameters:
    - df (DataFrame): The input DataFrame to add latitude groups to.
    - number_groups (int): The number of groups to create based on latitude quantiles.
    """
    df['lat_group'] = pd.qcut(df['Latitude'], q=number_groups, labels=False)
    return df

def classify_latitude_groups(df, testing_fraction, target_column):   #Dataset 1
    """
    Trains an XGBoost classifier using RandomizedSearchCV to predict the latitude group based on other features in the DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame containing the features and target variable.
    - testing_fraction (float): The fraction of data to use for testing the classifier.
    - target_column (str): The name of the target column to predict.
    """
    # Save the latitude values before dropping them
    latitudes = df['Latitude'].copy()
    
    # Prepare the features and target variable
    X = df.drop(columns=[target_column, 'Latitude', 'station'])  # Exclude non-feature columns + latitude
    y = df[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_fraction, random_state=10, stratify=y)
    
    # Define the hyperparameter space to search
    param_grid = {
        'n_estimators': np.arange(50, 400, 50),
        'max_depth': np.arange(3, 10),
        'learning_rate': np.linspace(0.01, 0.2, 10),
        'subsample': np.linspace(0.7, 1, 4),
        'colsample_bytree': np.linspace(0.7, 1, 4),
    }
    
    # Initialize the XGBoost classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    
    # Set up RandomizedSearchCV
    randomized_cv = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, scoring='accuracy', cv=5, verbose=1, random_state=42, n_jobs=-1)
    
    # Fit the RandomizedSearchCV to find the best model
    randomized_cv.fit(X_train, y_train)
    
    # Best model after RandomizedSearchCV
    best_model = randomized_cv.best_estimator_
    
    # Predict the classes for the test set using the best model
    y_pred = best_model.predict(X_test)
    
    # Create a DataFrame with actual and predicted classes for the test set
    test_results_df = X_test.copy()
    test_results_df['Actual_Class'] = y_test
    test_results_df['Predicted_Class'] = y_pred
    
    # Re-add the latitude values to the test_results_df
    test_results_df['Latitude'] = latitudes[X_test.index]
    
    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Model Accuracy: {accuracy}\n----------------------------------------")
    print(f"Best Model Parameters: {randomized_cv.best_params_}\n----------------------------------------")

    # Print the relative feature importance
    feature_importances = pd.DataFrame(best_model.feature_importances_,
                                       index = X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    print("\nFeature Importances:\n", feature_importances)

    # Optionally, plot the feature importances
    feature_importances.plot(kind='bar', title='Feature Importance')
    plt.ylabel('Relative Importance')
    plt.show()

    return test_results_df

def haversine(lat1, lon1, lat2, lon2):   #Dataset 2
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)

    Parameters:
    - lat1, lon1, lat2, lon2 (float): Latitude and longitude of the two points
    """
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula 
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def assign_closest_station(census_df, stations_df):   #Dataset 2
    """
    Assigns the closest weather station to each census location based on latitude and longitude.

    Parameters:
    - census_df (DataFrame): The DataFrame containing census data with latitude and longitude columns.
    - stations_df (DataFrame): The DataFrame containing weather station data with latitude and longitude columns.
    """
    closest_station_codes = []
    
    for census_index, census_row in census_df.iterrows():
        min_distance = np.inf
        closest_station_code = None
        
        # Use haversine function to calculate distance between census location and each weather station
        for station_index, station_row in stations_df.iterrows():
            distance = haversine(census_row['Latitude'], census_row['Longitude'],
                                 station_row['Latitude'], station_row['Longitude'])
            
            if distance < min_distance:
                min_distance = distance
                closest_station_code = station_row['station']  # Using 'station' as the code identifier
                
        closest_station_codes.append(closest_station_code)
    
    census_df['station'] = closest_station_codes
    return census_df

def calculate_descriptive_statistics(data, climate_cols, happiness_cols, group_by=None):
    """ 
    Calculates descriptive statistics for the specified columns in the dataset, optionally grouped by a column.
    
    Parameters:
    - data (DataFrame): The input DataFrame containing the data.
    - climate_cols (list): A list of column names for climate data.
    - happiness_cols (list): A list of column names for happiness data.
    - group_by (str): The column name to group the data by (optional).
    """
    if group_by:
        climate_descriptive_stats = data.groupby(group_by)[climate_cols].describe()
        happiness_descriptive_stats = data.groupby(group_by)[happiness_cols].describe()
    else:
        climate_descriptive_stats = data[climate_cols].describe()
        happiness_descriptive_stats = data[happiness_cols].describe()
    
    return (climate_descriptive_stats, happiness_descriptive_stats)

def perform_correlation_analysis(data, cols):   #Dataset 2
    """
    Performs correlation analysis between the specified columns and the 'Total Average rating' column.

    Parameters:
    - data (DataFrame): The input DataFrame containing the data.
    - cols (list): A list of column names to analyze for correlation.
    """
    correlation_matrix = data[cols].corr()
    return correlation_matrix[['Total Average rating']]

def perform_regression_analysis(data, independent_vars, dependent_var):
    """ 
    Performs linear regression analysis on the specified independent and dependent variables.
    
    Parameters:
    - data (DataFrame): The input DataFrame containing the data.
    - independent_vars (list): A list of column names for the independent variables.
    - dependent_var (str): The column name for the dependent variable.
    """

    X = data[independent_vars]
    y = data[dependent_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'intercept': model.intercept_,
        'coefficients': model.coef_
    }

def random_forest_feature_importance(data, independent_vars, dependent_var):
    """
    Calculates the feature importances using a Random Forest model.

    Parameters:
    - data (DataFrame): The input DataFrame containing the data.
    - independent_vars (list): A list of column names for the independent variables.
    - dependent_var (str): The column name for the dependent variable.
    """
    X = data[independent_vars]
    y = data[dependent_var]
    
    # Training the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Getting the feature importances
    importances = model.feature_importances_
    
    # Creating a DataFrame for easier visualization
    features_df = pd.DataFrame({
        'Feature': independent_vars,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    # Printing the feature importances
    print("Feature Importances:\n")
    print(features_df.to_string(index=False))

def process_dataset_1(station_url, pattern_station, clean_symbols, output_folder_ds1_txt, output_folder_ds1_csv, output_folder_ds1_csv_clean):
    """
    Downloads, converts, and cleans dataset 1, then extracts and saves coordinates.

    - station_url: URL to download the dataset from.
    - pattern_station: Pattern to match files for download.
    - clean_symbols: List of symbols to clean from the dataset.
    - output_folder_ds1_txt: Folder to save downloaded txt files.
    - output_folder_ds1_csv: Folder to save converted csv files.
    - output_folder_ds1_csv_clean: Folder to save cleaned csv files.
    """
    # Download and convert the files for dataset 1
    download_main_dataset(station_url, output_folder_ds1_txt, pattern_station)
    convert_txts_to_csvs(output_folder_ds1_txt, output_folder_ds1_csv)
    clean_and_rename_csv_in_folder(output_folder_ds1_csv, output_folder_ds1_csv_clean, [{'yyyy': "Site"}], clean_symbols)

    # Extract and save coordinates
    output_folder_ds1_coordinates = '../data/ds1/reference/locations'
    extract_and_save_coordinates(output_folder_ds1_txt, output_folder_ds1_coordinates)

def process_dataset_2(census_url, pattern_census, output_folder_ds2_xlsx, output_folder_ds2_csv, output_folder_ds2_csv_clean, output_folder_ds2_csv_combined):
    """
    Downloads, converts, cleans, and combines dataset 2.

    - census_url: URL to download the dataset from.
    - pattern_census: Pattern to match files for download.
    - output_folder_ds2_xlsx: Folder to save downloaded xlsx files.
    - output_folder_ds2_csv: Folder to save converted csv files.
    - output_folder_ds2_csv_clean: Folder to save cleaned csv files.
    - output_folder_ds2_csv_combined: Folder to save combined csv files.
    """
    # Ensure the combined data folder exists
    create_folder_if_not_exists(output_folder_ds2_csv_combined)

    # Download and convert the files for dataset 2
    download_main_dataset(census_url, output_folder_ds2_xlsx, pattern_census)
    process_excel_folder(output_folder_ds2_xlsx, output_folder_ds2_csv, 'Happiness', 'Codes')

    # Clean and combine the census data
    clean_census_rows(output_folder_ds2_csv, output_folder_ds2_csv_clean)
    combine_dataset_2_average(output_folder_ds2_csv_clean, output_folder_ds2_csv_combined, no_group=False)

def impute_missing_values_for_dataset_1(output_folder_ds1_csv_clean, output_folder_ds1_csv_clean_imputed, columns_to_impute, group_by):
    """
    Imputes missing values in the cleaned dataset 1 files and saves the imputed files.

    - output_folder_ds1_csv_clean: Folder containing the cleaned csv files.
    - output_folder_ds1_csv_clean_imputed: Folder to save csv files after imputation.
    - columns_to_impute: List of column names where missing values should be imputed.
    - group_by: Column name to group by before imputation.
    """
    impute_missing_values_in_folder(output_folder_ds1_csv_clean, output_folder_ds1_csv_clean_imputed, columns_to_impute, group_by=group_by)

def process_and_cluster_weather_data(output_folder_ds1_csv_clean_imputed, output_folder_ds1_coordinates, output_folder_clustering_df, output_folder_clustering_df_labeled, optimal_num_clusters):
    """
    Generates weather datasets, applies clustering, adds coordinates, saves, and plots the results. Returns the main DataFrames used or created.

    - output_folder_ds1_csv_clean_imputed: Path to the folder with imputed weather data.
    - output_folder_ds1_coordinates: Path to the folder with station coordinates.
    - output_folder_clustering_df: Path to the folder for saving clustering datasets.
    - output_folder_clustering_df_labeled: Path to the folder for saving labeled clustering datasets.
    :return: A tuple of DataFrames and possibly plot objects or paths depending on the implementation of plotting functions.
    """
    # Generate datasets
    stations_total_average_df = generate_weather_dataset(output_folder_ds1_csv_clean_imputed, use_station_average=True)
    stations_monthly_average_df = generate_weather_dataset(output_folder_ds1_csv_clean_imputed, use_station_average=False)

    # Save initial datasets
    save_dataset(stations_total_average_df, output_folder_clustering_df, 'stations_total_average_kmeans.csv')
    save_dataset(stations_monthly_average_df, output_folder_clustering_df, 'stations_monthly_average_kmeans.csv')

    # Apply clustering and label dataframes
    stations_total_average_df_labeled = apply_clustering(stations_total_average_df, range(1, 10), optimal_num_clusters=optimal_num_clusters)
    stations_monthly_average_df_labeled = apply_clustering(stations_monthly_average_df, range(1, 10), optimal_num_clusters=optimal_num_clusters)

    # Add coordinates
    stations_total_average_df_labeled = add_coordinates_to_df(stations_total_average_df_labeled, output_folder_ds1_coordinates, ['station', 'Station']).drop(columns=['Station'])
    stations_monthly_average_df_labeled = add_coordinates_to_df(stations_monthly_average_df_labeled, output_folder_ds1_coordinates, ['station', 'Station']).drop(columns=['Station'])

    # Save labeled datasets
    save_dataset(stations_total_average_df_labeled, output_folder_clustering_df_labeled, 'stations_total_average_kmeans.csv')
    save_dataset(stations_monthly_average_df_labeled, output_folder_clustering_df_labeled, 'stations_monthly_average_kmeans.csv')

    # Manual cluster determination and saving
    stations_monthly_average_df_labeled_manual = determine_dominant_cluster_manual(stations_monthly_average_df_labeled[['Latitude', 'Longitude', 'cluster']])
    save_dataset(stations_monthly_average_df_labeled_manual, output_folder_clustering_df_labeled, 'stations_monthly_average_manual_kmeans.csv')

    # DBSCAN clustering and saving
    stations_total_average_df_dbscan = apply_dbscan_clustering(stations_total_average_df)
    save_dataset(stations_total_average_df_dbscan, output_folder_clustering_df_labeled, 'stations_total_average_dbscan.csv')

    # Plotting
    plot_clusters_on_map(stations_total_average_df_labeled)
    plot_cluster_distribution_on_map(stations_monthly_average_df_labeled)

    cluster_stats = stats_by_group(stations_total_average_df_labeled[["cluster", "tmax (degC)", "sun (hours)"]], 'cluster')

    return stations_total_average_df, stations_monthly_average_df, stations_total_average_df_labeled, stations_monthly_average_df_labeled,stations_monthly_average_df_labeled_manual, stations_total_average_df_dbscan, cluster_stats

def process_and_classify_by_latitude(stations_total_labeled_df, stations_monthly_labeled_df, output_folder, latitude_groups, testing_fraction, target_column):
    """
    Process station data to classify and plot by latitude groups.
    
    - stations_total_labeled_df: DataFrame with labeled data for total averages.
    - stations_monthly_labeled_df: DataFrame with labeled data for monthly averages.
    - output_folder: Output directory for saving processed files.
    - latitude_groups: Number of latitude groups for classification.
    - testing_fraction: Fraction of data to be used for testing in classification.
    - target_column: The name of the target column for classification.
    """
    create_folder_if_not_exists(output_folder)
    
    # Process and save total average latitude dataset
    latitude_total_dataset = "latitude_total_dataset.csv"
    stations_latitude_total_average_df = generate_latitude_ds(stations_total_labeled_df, latitude_groups).drop(columns=['cluster'])
    save_dataset(stations_latitude_total_average_df, output_folder, latitude_total_dataset)
    plot_clusters_on_map(stations_latitude_total_average_df, map_col='lat_group')
    
    # Process and save monthly average latitude dataset
    latitude_monthly_dataset = "latitude_monthly_dataset.csv"
    stations_latitude_monthly_average_df = generate_latitude_ds(stations_monthly_labeled_df, latitude_groups).drop(columns=['cluster'])
    save_dataset(stations_latitude_monthly_average_df, output_folder, latitude_monthly_dataset)
    
    # Classify and plot total average latitude groups
    classified_df_total = classify_latitude_groups(stations_latitude_total_average_df, testing_fraction, target_column)
    plot_clusters_on_map(classified_df_total, "Predicted_Class")
    plot_clusters_on_map(classified_df_total, "Actual_Class")

    classified_df_monthly = classify_latitude_groups(stations_latitude_monthly_average_df, testing_fraction, target_column)

def process_census_happiness_data(reference_csv_census_location, file_census_happiness_csv, stations_latitude_label_csv, output_folder):
    """
    Processes the census happiness data to include location, merge with weather station data,
    and plot the results.

    - reference_csv_census_location: Path to the reference CSV with census locations.
    - file_census_happiness_csv: Path to the CSV file with census happiness data.
    - stations_latitude_label_csv: Path to the CSV file with latitude labeled station data.
    - output_folder: Output directory for saving processed files.
    """
    create_folder_if_not_exists(output_folder)
    
    # Load the latitude labeled station data
    stations_latitude_total_average_df = load_csv_with_cols(stations_latitude_label_csv, None)
    
    # Prepare the census happiness dataset with location information
    df_reference_census_location = load_csv_with_cols(reference_csv_census_location, ["Codes", "Area", "Latitude", "Longitude"])
    df_census_happiness = load_csv_with_cols(file_census_happiness_csv, None)
    df_census_happiness_locations = df_census_happiness.merge(df_reference_census_location, on="Codes")
    
    file_census_happiness_dataset = "happines_regions_location.csv"
    save_dataset(df_census_happiness_locations, output_folder, file_census_happiness_dataset)
    
    # Assign the closest weather station to each census region and merge
    file_census_closest_station = "happines_regions_location_station.csv"
    df_census_closest_station = assign_closest_station(df_census_happiness_locations, stations_latitude_total_average_df)
    df_census_closest_station = pd.merge(df_census_closest_station, stations_latitude_total_average_df, on='station')
    save_dataset(df_census_closest_station, output_folder, file_census_closest_station)
    
    # Round off the 'Total Average rating' and plot
    df_census_happiness_locations['Total Average rating'] = df_census_happiness_locations['Total Average rating'].round(2)
    plot_clusters_on_map(df_census_happiness_locations, map_col='Total Average rating')
    plot_clusters_on_map(df_census_closest_station, map_col='lat_group', lon_col='Longitude_x', lat_col='Latitude_x')

    return df_census_closest_station

def analyze_weather_and_happiness_impact(df, climate_columns, happiness_columns, dependent_var):
    """
    Performs a series of analyses to understand the impact of weather (climate) on happiness.

    - df: The DataFrame containing both climate and happiness data.
    - climate_columns: List of climate-related columns in the DataFrame for analysis.
    - happiness_columns: List of happiness-related columns in the DataFrame for analysis.
    - dependent_var: The dependent variable (from happiness data) for regression analysis.
    """
    # Descriptive Statistics
    climate_stats, happiness_stats = calculate_descriptive_statistics(df, climate_columns, happiness_columns, group_by='lat_group')
    print("Climate Descriptive Statistics:\n", climate_stats)
    print("\nHappiness Descriptive Statistics:\n", happiness_stats)

    # Correlation Analysis
    correlation_results = perform_correlation_analysis(df, climate_columns + [dependent_var])
    print("\nCorrelation Analysis Results:\n", correlation_results)

    # Regression Analysis
    regression_results = perform_regression_analysis(df, climate_columns, dependent_var)
    print("\nRegression Analysis Results:\n", regression_results)

    # Random Forest Feature Importance
    random_forest_feature_importance(df, climate_columns, dependent_var)