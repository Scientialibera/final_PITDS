from dotenv import load_dotenv
import os

class Config:
    def __init__(self, path='./environment.env'):
        # Load environment variables from specified file path
        load_dotenv(dotenv_path=path)

        # Station and climate configurations
        self.station_url = os.getenv("STATION_URL") # URL for dataset 1
        self.pattern_station = os.getenv("PATTERN_STATION") # Pattern for dataset 1 download
        self.clean_symbols = os.getenv("CLEAN_SYMBOLS").split(',') # Symbols to clean from dataset 1
        self.census_url = os.getenv("CENSUS_URL") # URL for dataset 2
        self.pattern_census = os.getenv("PATTERN_CENSUS") # Pattern for dataset 2 download
        self.columns_to_impute = os.getenv("COLUMNS_TO_IMPUTE").split(',') # Columns to impute in dataset 1 (Missing data in weather dataset)
        self.testing_fraction = float(os.getenv("TESTING_FRACTION")) # Fraction of data to use for testing for Part 2
        self.latitude_groups = int(os.getenv("LATITUDE_GROUPS")) # Number of latitude groups for Part 2 - Will create this many clusters
        self.target_column = os.getenv("TARGET_COLUMN") # Column to predict in Part 2
        self.climate_columns = os.getenv("CLIMATE_COLUMNS").split(',') # Columns to use for for EDA in Part 3 (and to set as independent variables)
        self.happiness_columns = os.getenv("HAPPINESS_COLUMNS").split(',') # Happiness Cols to use for EDA in Part 3
        self.dependent_var = os.getenv("DEPENDENT_VAR") # Dependent variable for Part 3 (We decided to use Average Happoiness Score, but can be easily changed)
        self.reference_csv_census_location = os.getenv("REFERENCE_CSV_CENSUS_LOCATION") # Location of the reference csv for the census
        self.optimal_num_clusters = int(os.getenv("OPTIMAL_NUM_CLUSTERS")) # Optimal number of clusters for KMeans