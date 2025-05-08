#!/usr/bin/env python
# coding: utf-8

# Final Notebook for CS 3500 

import sys
import time
import threading
import itertools
import warnings
import os
import pickle
import joblib

# Initialize all global variables to prevent reference before assignment
train_df, test_df, df = None, None, None
train_bool, test_bool, clean_bool, trained_bool = None, None, None, None
torch_available = False
libraries_status = {"core": True, "data": False, "ml": False, "nn": False}
predictions_data = None
trans_model, mlp_model = None, None

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

def spinner(message, show_done=False):
    """
    Display a spinner animation while a task is running.
    
    Args:
        message: Message to display alongside the spinner
        show_done: Whether to show "Done!" when complete
        
    Returns:
        Function to call to stop the spinner
    """
    done_event = threading.Event()
    spinner_active = True  # Flag to track if spinner is active

    def spin():
        for c in itertools.cycle('|/-\\'):
            if done_event.is_set():
                break
            sys.stdout.write(f'\r{message}... {c}') 
            sys.stdout.flush()
            time.sleep(0.1)
            
        # Clear the spinner line completely
        sys.stdout.write('\r' + ' ' * (len(message) + 10) + '\r')
        sys.stdout.flush()
        if show_done:
            print('Done! ✅')

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()

    # Wrap the stop function to also join the thread
    def stop_and_join():
        nonlocal spinner_active
        if spinner_active:  # Only stop if spinner is active
            done_event.set()
            try:
                thread.join(timeout=1.0)  # Add timeout to prevent hanging
            except Exception:
                pass
            spinner_active = False
    
    return stop_and_join

# Clear screen
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# ----------------------
# IMPORT LIBRARIES
# ----------------------
def import_libraries():
    """
    This function imports all necessary libraries and sets up the global variables.
    """
    stop_spinner = spinner("Loading dependencies...", show_done=True)
    
    # Dictionary to track available libraries
    libraries_available = {"core": True, "data": True, "ml": True, "nn": torch_available}
    
    # Core modules
    try:
        import os
        import sys
        import time
        import random
        from datetime import datetime, timedelta
        from collections import Counter
        import itertools
    except ImportError as e:
        print(f"Warning: Some core libraries not available: {e}")
        libraries_available["core"] = False

    # Data handling & visualization
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from tqdm import tqdm
    except ImportError as e:
        print(f"Warning: Some data libraries not available: {e}")
        libraries_available["data"] = False

    # Sklearn tools
    try:
        from sklearn.preprocessing import LabelEncoder, PowerTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import (
            classification_report,
            precision_recall_curve,
            roc_curve,
            auc,
            confusion_matrix,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score
        )
    except ImportError as e:
        print(f"Warning: Machine learning libraries not available: {e}")
        libraries_available["ml"] = False


    # PyTorch - already imported at module level 
    if not torch_available:
        print("Note: PyTorch not available. Training and prediction functionality will be limited.")

    # Resampling
    try:
        from imblearn.over_sampling import SMOTE
        smote_available = True
    except ImportError:
        print("Warning: SMOTE not available. Will not be able to handle class imbalance.")
        smote_available = False


    # Promote available libraries to global scope
    globals_dict = {}
    
    # Add core libraries if available
    if libraries_available["core"]:
        globals_dict.update({
            'os': os,
            'sys': sys,
            'time': time,
            'random': random,
            'datetime': datetime,
            'timedelta': timedelta,
            'Counter': Counter,
            'itertools': itertools,
        })
    
    # Add data libraries if available
    globals_dict.update({
        'pd': pd,
        'np': np,
    })
    
    if libraries_available["data"]:
        globals_dict.update({
            'plt': plt,
            'sns': sns,
            'tqdm': tqdm,
        })
    
    # Add ML libraries if available
    globals_dict.update({
        'LabelEncoder': LabelEncoder,
        'PowerTransformer': PowerTransformer,
        'CountVectorizer': CountVectorizer,
        'PCA': PCA,
        'KMeans': KMeans,
    })
    
    if libraries_available["ml"]:
        globals_dict.update({
            'classification_report': classification_report,
            'precision_recall_curve': precision_recall_curve,
            'roc_curve': roc_curve,
            'auc': auc,
            'confusion_matrix': confusion_matrix,
            'accuracy_score': accuracy_score,
            'precision_score': precision_score,
            'recall_score': recall_score,
            'f1_score': f1_score,
            'roc_auc_score': roc_auc_score,
        })
    
    # Add SMOTE if available
    if smote_available:
        globals_dict['SMOTE'] = SMOTE
    
    # Update global namespace
    globals().update(globals_dict)
    
    # Store library status for function availability checks
    global libraries_status
    libraries_status = libraries_available

    stop_spinner()

# Neural Network Global Class Definitions
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    torch_available = True
except ImportError:
    print("Warning: PyTorch libraries not found. Neural network functionality will be limited.")


# ---------------- Device ----------------
if torch_available:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

# ---------------- Dataset ----------------
class CrimeDataset(Dataset):
    def __init__(self, features, targets):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

# ---------------- Focal Loss ----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# ---------------- Models ----------------
class TabTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, hidden_dim=128):
        super(TabTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*2, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.head(x)

class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.selu = nn.SELU()
        self.dropout = nn.AlphaDropout(0.1)

    def forward(self, x):
        x = self.dropout(self.selu(self.fc1(x)))
        x = self.dropout(self.selu(self.fc2(x)))
        return self.fc3(x)

# Globals (To Prevent Errors)
test_bool, train_bool, clean_bool, trained_bool = None, None, None, None
trans_model, mlp_model = None, None

# Helper function for timestamp
def timestamp(): 
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

# ----------------------
# LOAD DATA (TRAIN OR TEST)
# ----------------------
def load_data(file_path, label):
    """
    This function loads the training or test data.
    """
    stop_spinner = spinner(f"\nLoading {label}ing set")
    print("\n*********************")

    global train_df, test_df, train_bool, test_bool
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            stop_spinner()  # Ensure spinner is stopped
            return None
            
        if label == "train":
            train_bool = True
            start = time.time()
            train_df = pd.read_csv(file_path)
            print(f"Total Columns Read: {train_df.shape[1]}")
            print(f"Total Rows Read: {train_df.shape[0]}")
            print("\nTime to load is:", round(time.time() - start, 2), "seconds")
            stop_spinner()  # Stop spinner before return
            return train_df
        elif label == "test":
            test_bool = True
            start = time.time()
            test_df = pd.read_csv(file_path)
            print(f"Total Columns Read: {test_df.shape[1]}")
            print(f"Total Rows Read: {test_df.shape[0]}")
            print("\nTime to load is:", round(time.time() - start, 2), "seconds")
            stop_spinner()  # Stop spinner before return
            return test_df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        stop_spinner()  # Ensure spinner is stopped on exception
        return None

# ----------------------
# PROCESS (CLEAN) DATA
# ----------------------
def process_data():
    """
    This function processes the data.
    """
    # Mark Bool As Clean For NN
    global train_df, test_df, clean_bool, df
    clean_bool = True

    # Combine Data For Cleaning
    print("\n*********************")
    stop_spinner = spinner("Cleaning Data....", show_done=True)
    start = time.time()

    try:
        # Check if we have training data, test data, or both
        if train_bool and test_bool:
            print("Processing both training and test data...")
            # Fix the origin flags 
            train_df["__origin"] = 0
            test_df["__origin"] = 1
            
            # Create backup of test data
            test_backup = test_df.copy()
            
            # Combine datasets
            df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        elif train_bool:
            print("Processing training data only...")
            df = train_df.copy()
            df["__origin"] = 0
        elif test_bool:
            print("Processing test data only...")
            df = test_df.copy()
            df["__origin"] = 1
        else:
            print("🛑 No data loaded. Please load either training or test data first.")
            stop_spinner()
            return None
            
        # Remove Unnamed Columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Convert the columns to a suitable data type
        df['Date Rptd'] = pd.to_datetime(df['Date Rptd'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%Y-%m-%d', errors='coerce')
        df['AREA NAME'] = df['AREA NAME'].astype('string')
        df['Crm Cd Desc'] = df['Crm Cd Desc'].astype('string')
        df['Mocodes'] = df['Mocodes'].astype('string')
        df['Vict Sex'] = df['Vict Sex'].astype('string')
        df['Vict Descent'] = df['Vict Descent'].astype('string')
        df['Premis Desc'] = df['Premis Desc'].astype('string')
        df['Weapon Desc'] = df['Weapon Desc'].astype('string')
        df['Status'] = df['Status'].astype('string')
        df['Status Desc'] = df['Status Desc'].astype('string')

        # Mapping dictionary
        mapping = {
                    'IC': 'No Arrest'
                    ,'AA': 'Arrest'
                    ,'AO': 'No Arrest'
                    ,'JO': 'No Arrest'
                    ,'JA': 'Arrest'
                    ,'CC': 'No Arrest'
        }
        # Create target variable based in the status variable 
        df['Target'] = df['Status'].map(mapping)

        # Change data type
        df['TIME OCC'] = df['TIME OCC'].astype('string')

        # Pad the 'TIME OCC' column values with leading zeros to ensure a 4-digit format
        df['TIME OCC'] = df['TIME OCC'].str.zfill(4)

        # Format the 'TIME OCC' column as 'HH:MM' (hour:minute)
        df['TIME OCC'] = df['TIME OCC'].str[:-2] + ':' + df['TIME OCC'].str[-2:]

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Fill missing values (NaN) in 'Weapon Used Cd' column with 0
        df.loc[df['Weapon Used Cd'].isna(), 'Weapon Used Cd'] = 0

        # Fill missing values (NaN) in 'Weapon Desc' column with 'No weapons identified'
        df.loc[df['Weapon Desc'].isna(), 'Weapon Desc'] = 'No weapons identified'

        # Filter the DataFrame 'df' to exclude rows where 'Vict Age' is either 0 or NaN
        df = df[(df['Vict Age'] != 0) & (df['Vict Age'].notna())]

        # Filter the DataFrame 'df' to exclude rows where 'Vict Sex' is 'X' (Unknown), 'H' (invalid), or NaN
        df = df[(df['Vict Sex'] != 'X') & (df['Vict Sex'] != 'H')&(df['Vict Sex'].notna())]

        # Filter the DataFrame 'df' to exclude rows where 'Vict Descent' is '-' or missing (NaN)
        df = df[(df['Vict Descent'] != '-') & (df['Vict Descent'].notna())]

        # Conver to DateTime
        df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])

        # Extract the year from the "Date Occurred" column and create a new column "Year"
        df['Year'] = df['DATE OCC'].dt.year

        # Convert Hour To DateTime
        df['TIME OCC'] = pd.to_datetime(df['TIME OCC'], format='%H:%M')

        # Extract the time component (hours and minutes)
        df['Time'] = df['TIME OCC'].dt.time

        # Define time intervals
        intervals = [(pd.Timestamp('00:01:00').time(), pd.Timestamp('06:00:00').time()),
                    (pd.Timestamp('06:01:00').time(), pd.Timestamp('12:00:00').time()),
                    (pd.Timestamp('12:01:00').time(), pd.Timestamp('18:00:00').time()),
                    (pd.Timestamp('18:01:00').time(), pd.Timestamp('23:59:59').time())]

        # Create labels for the intervals
        labels = ['00:01-06:00', '06:01-12:00', '12:01-18:00', '18:01-24:00']

        # Define a custom categorization function
        def categorize_time(time):
            for i, interval in enumerate(intervals):
                if interval[0] <= time <= interval[1]:
                    return labels[i]
            return None

        # Apply the custom categorization function to create the 'Time Interval' column
        df['Time Interval'] = df['Time'].apply(categorize_time)

        # Create Function To Clean Outliers Using IQR Method
        def handle_outliers(df):
            for col in df.select_dtypes(include='number').columns:
                    # Identify Quartiles
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
            
                    # Identify Upper And Lower Limit
                    lower_lim = Q1 - 1.5 * IQR
                    upper_lim = Q3 + 1.5 * IQR
            
                    # Drop All Values Outside Lower/Upper Limit
                    df = df[(df[col] >= lower_lim) & (df[col] <= upper_lim)]

            return df

        df = handle_outliers(df)

        # Remove 'CC' Column Since It Accounts FOR Such A Small Percentage Of Data
        df = df.drop(df[df['Status'] == 'CC'].index)
        df['Status'].value_counts()

        # Days To Holiday Column
        def get_us_holidays(year):
            # Fixed-date holidays
            holidays = [
                datetime(year, 1, 1),    # New Year's Day
                datetime(year, 7, 4),    # Independence Day
                datetime(year, 11, 11),  # Veterans Day
                datetime(year, 12, 25),  # Christmas Day
            ]
            
            # Floating holidays
            # Martin Luther King Jr. Day (3rd Monday of January)
            mlk = datetime(year, 1, 1) + timedelta(days=(14 - datetime(year, 1, 1).weekday()) % 7 + 14)
            holidays.append(mlk)
            
            # Presidents' Day (3rd Monday of February)
            presidents_day = datetime(year, 2, 1) + timedelta(days=(14 - datetime(year, 2, 1).weekday()) % 7 + 14)
            holidays.append(presidents_day)
            
            # Memorial Day (last Monday of May)
            memorial_day = datetime(year, 5, 31)
            while memorial_day.weekday() != 0:
                memorial_day -= timedelta(days=1)
            holidays.append(memorial_day)
            
            # Labor Day (1st Monday of September)
            labor_day = datetime(year, 9, 1)
            while labor_day.weekday() != 0:
                labor_day += timedelta(days=1)
            holidays.append(labor_day)
            
            # Columbus Day (2nd Monday of October)
            columbus_day = datetime(year, 10, 1) + timedelta(days=(7 - datetime(year, 10, 1).weekday()) % 7 + 7)
            holidays.append(columbus_day)
            
            # Thanksgiving Day (4th Thursday of November)
            thanksgiving = datetime(year, 11, 1)
            thursdays = 0
            while thursdays < 4:
                if thanksgiving.weekday() == 3:
                    thursdays += 1
                thanksgiving += timedelta(days=1)
            holidays.append(thanksgiving - timedelta(days=1))  # because it overshoots
            
            return holidays

        # Ensure DATE OCC is datetime
        df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])

        # Calculate Days_To_Holiday
        def days_to_nearest_holiday(date):
            year = date.year
            holidays = get_us_holidays(year)
            return min(abs((date - h).days) for h in holidays)

        df['Days_To_Holiday'] = df['DATE OCC'].apply(days_to_nearest_holiday)

        # Seperate Date, Time, Month into Individual Columns
        # Reported Date
        df['RPTD_Year'] = df['Date Rptd'].dt.year
        df['RPTD_Month'] = df['Date Rptd'].dt.month
        df['RPTD_Day'] = df['Date Rptd'].dt.day

        # Date Occured
        df['OCC_Year'] = df['DATE OCC'].dt.year
        df['OCC_Month'] = df['DATE OCC'].dt.month
        df['OCC_Date'] = df['DATE OCC'].dt.day

        # Time Occured
        df['OCC_Hour'] = df['TIME OCC'].dt.hour
        df['OCC_Minute'] = df['TIME OCC'].dt.minute
        df['OCC_Second'] = df['TIME OCC'].dt.second

        # Time Of Day Occured
        def map_time_of_day(hour):
            if 0 <= hour < 5:
                return 'Late Night'
            elif 5 <= hour < 8:
                return 'Early Morning'
            elif 8 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:
                return 'Night'

        df['OCC_TimeOfDay'] = df['OCC_Hour'].apply(map_time_of_day)

        # Drop Parsed Columns
        df.drop(['Date Rptd', 'DATE OCC', 'TIME OCC', 'Time', 'Year'], axis = 1, inplace = True)

        df.rename(columns={
            'AREA': 'area_code',
            'AREA NAME': 'area_name',
            'Rpt Dist No': 'reporting_district',
            'Part 1-2': 'crime_part',
            'Crm Cd': 'crime_code',
            'Crm Cd Desc': 'crime_description',
            'Mocodes': 'mo_codes',
            'Vict Age': 'victim_age',
            'Vict Sex': 'victim_sex',
            'Vict Descent': 'victim_descent',
            'Premis Cd': 'premise_code',
            'Premis Desc': 'premise_description',
            'Weapon Used Cd': 'weapon_code',
            'Weapon Desc': 'weapon_description',
            'Status Desc': 'status_description',
            'Target': 'arrest_type',
            'Time Interval': 'occ_time_interval',
            'Days_To_Holiday': 'days_to_holiday',
            'RPTD_Year': 'report_year',
            'RPTD_Month': 'report_month',
            'RPTD_Day': 'report_day',
            'OCC_Year': 'occurrence_year',
            'OCC_Month': 'occurrence_month',
            'OCC_Date': 'occurrence_day',
            'OCC_Hour': 'occurrence_hour',
            'OCC_Minute': 'occurrence_minute',
            'OCC_Second': 'occurrence_second',
            'OCC_TimeOfDay': 'occurrence_time_of_day',
        }, inplace=True)

        # Count The Number Mo_Codes
        df['mo_code_count'] = df['mo_codes'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

        # Top K Binary Flags
        def binary_flags(df, k=20):
            # Step 1: Flatten all codes into one list
            all_mo_lists = df['mo_codes'].fillna("").apply(lambda x: str(x).split())
            flat_list = [code for sublist in all_mo_lists for code in sublist]

            # Step 2: Count frequency of each code
            mo_counts = Counter(flat_list)

            # Step 3: Get top K codes
            top_k_codes = [code for code, _ in mo_counts.most_common(k)]

            # Step 4: Create binary flags
            for code in top_k_codes:
                df[f'mo_{code}'] = df['mo_codes'].apply(lambda x: int(code in str(x).split()))

            return df

        df = binary_flags(df, k=20)

        # Cluster Codes
        def cluster(k = 5):
            vectorizer = CountVectorizer(analyzer = str.split) # Tokenizes on Each Mo_Code Sep by ' '
            x_mo = vectorizer.fit_transform(df['mo_codes'].fillna("").astype(str)) # Learn the Vocabulary 
            
            kmeans = KMeans(n_clusters = k, random_state = 42) # Assign Codes To Cluster
            df['mo_cluster'] = kmeans.fit_predict(x_mo) # Assign Clusters To Row on df

            pca = PCA(n_components = 2) # Dimensionality Reduction for visualization
            x_pca = pca.fit_transform(x_mo.toarray())

        cluster(10)

        df.drop(['status_description', 'Status'], inplace = True, axis = 1)

        def encode_columns(df):
            categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

            for col in ['mo_codes', 'arrest_type']: # Ignore These Two Cols For Further Processing
                if col in categorical_cols:
                    categorical_cols.remove(col)
            
            df = pd.get_dummies(df, columns=categorical_cols) # One-Hot Encode Features

            # Label Encode Target
            le = LabelEncoder()
            df['arrest_type'] = le.fit_transform(df['arrest_type'])
            
            return df
        
        df = encode_columns(df)

        # Move Target Column To Last Position
        df = df.assign(arrest_type=df.pop('arrest_type'))

        # Print data shapes before splitting
        print(f"Total data shape: {df.shape}")
        print(f"Train indicators: {df['__origin'].value_counts().get(0, 0)}")
        print(f"Test indicators: {df['__origin'].value_counts().get(1, 0)}")

        print(f"Time Elapsed: {round(time.time() - start, 2)} seconds")
        stop_spinner()
        return df
    except Exception as e:
        print(f"Error during data processing: {str(e)}")
        stop_spinner()
        return None

# ----------------------
# TRAIN NEURAL NETWORK
# ----------------------
# Helper Function
def init_models(input_dim):
    """
    This function initializes the models.
    """
    global trans_model, mlp_model
    trans_model = TabTransformer(input_dim).to(device)
    mlp_model = MLPModel(input_dim).to(device)

def train_nn(df):
    """
    This function trains the neural network.
    """
    # Mark Trained Bool as True (Globally)
    global trans_model, mlp_model, train_loader, test_loader, trained_bool
    
    print("\n🚨 Model training is a computationally intensive process 🚨")
    print("It may take a significant amount of time and system resources")
    confirm = input("Do you want to proceed with training? (y/n): ")
    
    if confirm.lower() != 'y':
        print("Training cancelled. Returning to main menu.")
        return
    
    # Mark Trained Bool as True (Globally)
    trained_bool = True

    print("\n🚨 Model training in progress. This will take a while... 🚨")

    learning_rate = 0.001
    epochs = 500
    batch_size = 1024
    patience = 15

    stop_spinner = spinner("Performing Final Preprocessing...")
    df.drop(columns=["mo_codes"], inplace=True)
    df = df[df["arrest_type"].isin([0, 1])].reset_index(drop=True)
    stop_spinner()

    stop_spinner = spinner("Splitting Data...")
    test_size = int(0.2 * len(df))
    train_df = df[:-test_size].reset_index(drop=True)
    test_df = df[-test_size:].reset_index(drop=True)

    # Split features and labels before scaling
    X_train = train_df.drop(columns="arrest_type")
    y_train = train_df["arrest_type"]
    X_test = test_df.drop(columns="arrest_type")
    y_test = test_df["arrest_type"]
    stop_spinner()

    # Scale training and test data
    stop_spinner = spinner("Scaling Data...")
    
    # Completely suppress all warnings from scaling operation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        scaler = PowerTransformer(method='yeo-johnson')
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # Create directory if it doesn't exist
    os.makedirs("models/script_models", exist_ok=True)
    
    # Save the scaler with joblib
    scaler_path = "models/script_models/scaler.pkl"
    try:
        if joblib is not None:
            joblib.dump(scaler, scaler_path)
        else:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
    except Exception as e:
        print(f"Error saving scaler: {str(e)}. Using pickle instead.")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Save column names for feature alignment during prediction
    shared_columns = list(X_train.columns)
    with open("models/script_models/shared_columns.pkl", "wb") as f:
        pickle.dump(shared_columns, f)
    print(f"Column information saved for prediction alignment")
    
    stop_spinner()

    # SMOTE Oversampling (Only On Training Data)
    stop_spinner = spinner("Upsampling the minority class...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    stop_spinner()

    # Wrap datasets
    stop_spinner = spinner("Loading DataLoaders...")
    train_dataset = CrimeDataset(X_train_resampled, y_train_resampled)
    test_dataset = CrimeDataset(X_test_scaled, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    stop_spinner()

    stop_spinner = spinner("Initializing Models...")
    input_dim = X_train.shape[1]
    init_models(input_dim)

    models = [trans_model, mlp_model]
    optimizers = [torch.optim.Adam(m.parameters(), lr=learning_rate, weight_decay=1e-4) for m in models]
    criterion = FocalLoss(alpha=0.8, gamma=1.0)
    schedulers = [ReduceLROnPlateau(opt, mode='min', patience=10) for opt in optimizers]
    stop_spinner()

    for i, model in enumerate(models):
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        best_loss, epochs_no_improve = float('inf'), 0
        for epoch in tqdm(range(epochs), desc=f"Training model {i+1}"):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizers[i].zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1).float())
                loss.backward()
                optimizers[i].step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            schedulers[i].step(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
                torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), f"models/script_models/model_{i+1}.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("🛑 Early Stopping Triggered at Epoch: ", epoch)
                    break

# ----------------------
# MAKE PREDICTIONS
# ----------------------
def predict(df):
    """
    This function evaluates the model given a test set.
    """
    global trans_model, mlp_model, device, predictions_data
    
    print("\n*********************")
    print("Evaluating models with the test data")
    print(f"Data shape: {df.shape}")
    
    # Initial spinner for the overall process
    stop_spinner = spinner("Preparing data for prediction")
    
    try:
        # Check if test data has the target column to evaluate accuracy
        if 'arrest_type' in df.columns:
            df = df[df['arrest_type'].isin([0, 1])].reset_index(drop=True)
            has_target = True
            print("Target column 'arrest_type' found. Will evaluate model accuracy.")
        else:
            has_target = False
            print("No target column found. Will generate predictions only.")
        
        # Check which model sets are available
        script_models_available = (os.path.exists("models/script_models/model_1.pt") and 
                                  os.path.exists("models/script_models/model_2.pt") and 
                                  os.path.exists("models/script_models/scaler.pkl") and 
                                  os.path.exists("models/script_models/shared_columns.pkl"))
                                 
        bolt_models_available = (os.path.exists("models/bolt_models/model_1.pt") and 
                               os.path.exists("models/bolt_models/model_2.pt") and 
                               os.path.exists("models/bolt_models/scaler.pkl") and 
                               os.path.exists("models/bolt_models/shared_columns.pkl"))
        
        # Debug information
        print(f"Checking for models:")
        print(f"  - script_models: {'Available' if script_models_available else 'Not found'}")
        print(f"  - bolt_models: {'Available' if bolt_models_available else 'Not found'}")
        
        # Set default model path prefix based on working directory
        model_path_prefix = ""
        
        # Stop the spinner before asking for user input
        stop_spinner()
        
        # Ask user which model set to use if both are available
        model_set = None
        if script_models_available and bolt_models_available:
            print("\nBoth user-trained models and pre-trained models are available.")
            choice = input("Use (1) (HIGHLY RECOMMENDED) user-trained models or (2) (OLD) pre-trained models? (1/2): ")
            model_set = "script_models" if choice == "1" else "bolt_models"
        elif script_models_available:
            print("\nUsing user-trained models (script_models).")
            model_set = "script_models"
        elif bolt_models_available:
            print("\nUsing pre-trained models (bolt_models).")
            model_set = "bolt_models"
        else:
            print("\n❌ No models found. Please train models first (option 3).")
            return None
        
        model_path = f"{model_path_prefix}models/{model_set}"
        print(f"Loading models from {model_path}")
        
        # Start a new spinner for preprocessing
        stop_spinner = spinner("Preprocessing data")
        
        try:
            if has_target:
                X_test_scaled, y_test = preprocess_for_prediction(df, model_set, model_path_prefix)
            else:
                X_test_scaled = preprocess_for_prediction(df, model_set, model_path_prefix)
                y_test = pd.Series(np.zeros(X_test_scaled.shape[0]))
        except Exception as e:
            print(f"\nError in preprocessing: {str(e)}")
            stop_spinner()
            return None
            
        stop_spinner()
        
        # Load models
        stop_spinner = spinner("Loading models")
        
        input_dim = X_test_scaled.shape[1]
        
        # Initialize models with correct input dimension
        if trans_model is None or mlp_model is None:
            init_models(input_dim)
        
        # Load the pre-trained weights
        model1_path = f"{model_path}/model_1.pt"
        model2_path = f"{model_path}/model_2.pt"
        
        try:
            trans_model.load_state_dict(torch.load(model1_path, map_location=device))
            mlp_model.load_state_dict(torch.load(model2_path, map_location=device))
            print(f"Successfully loaded model weights from {model_path}")
        except Exception as e:
            print(f"\nFailed to load models: {e}")
            stop_spinner()
            return None
        
        # Set models to evaluation mode
        trans_model.eval()
        mlp_model.eval()
        
        stop_spinner()
        
        # Make predictions
        stop_spinner = spinner("Running with ensemble models")
        
        # Create a dataset and dataloader
        test_dataset = CrimeDataset(X_test_scaled, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                
                # Get predictions from both models
                trans_outputs = torch.sigmoid(trans_model(X_batch))
                mlp_outputs = torch.sigmoid(mlp_model(X_batch))
                
                # Ensemble predictions (average of both models)
                ensemble_probs = (trans_outputs + mlp_outputs) / 2
                
                # Store predictions and targets
                all_targets.extend(y_batch.cpu().numpy())
                all_probabilities.extend(ensemble_probs.cpu().numpy())
        
        # Convert probabilities to binary predictions (threshold = 0.5)
        all_predictions = [1 if p[0] > 0.5 else 0 for p in all_probabilities]
        
        stop_spinner()
        
        # Calculate metrics if we have ground truth labels
        if has_target:
            accuracy = accuracy_score(all_targets, all_predictions)
            precision = precision_score(all_targets, all_predictions)
            recall = recall_score(all_targets, all_predictions)
            f1 = f1_score(all_targets, all_predictions)
            auc = roc_auc_score(all_targets, [p[0] for p in all_probabilities])
            
            print("\n=== Model Performance ===")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC AUC: {auc:.4f}")
            
            # Save predictions for accuracy function
            predictions_data = {
                'predictions': all_predictions,
                'probabilities': all_probabilities,
                'targets': all_targets,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc
                }
            }
        else:
            # Just show predictions without metrics
            predictions_data = {
                'predictions': all_predictions,
                'probabilities': all_probabilities
            }
        
        # Display 5 random predictions
        import random
        random_indices = random.sample(range(len(all_predictions)), min(5, len(all_predictions)))
        
        print("\n=== Sample Predictions ===")
        for idx in random_indices:
            sample_idx = df.index[idx]
            pred_label = "Arrest" if all_predictions[idx] == 1 else "No Arrest"
            prob = all_probabilities[idx][0]
            
            if has_target:
                true_label = "Arrest" if all_targets[idx] == 1 else "No Arrest"
                print(f"Sample #{sample_idx}:")
                print(f"  True: {true_label}")
                print(f"  Predicted: {pred_label} (Confidence: {prob:.4f})")
            else:
                print(f"Sample #{sample_idx}:")
                print(f"  Predicted: {pred_label} (Confidence: {prob:.4f})")
        
        print("\nPrediction complete! ✅")
        
        # If data doesn't have a target column, save predictions to a CSV file
        if not has_target:
            # Save predictions to a CSV file
            results_df = df.copy()
            results_df['prediction'] = all_predictions
            results_df['confidence'] = [p[0] for p in all_probabilities]
            results_df['predicted_label'] = ["Arrest" if p == 1 else "No Arrest" for p in all_predictions]
            
            os.makedirs("Final/output", exist_ok=True)
            output_path = "Final/output/predictions.csv"
            results_df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up spinner in case of error
        stop_spinner()
        return None
    
    return predictions_data

# ----------------------
# ACCURACY FUNCTION
# ----------------------
def accuracy():
    """
    Returns accuracy and generates performance plots.
    """
    global predictions_data
    
    if 'predictions_data' not in globals() or predictions_data is None:
        print("🛑 No predictions available. Please run predictions first.")
        return
    
    print("\n*********************")
    print("Detailed Model Performance Metrics")
    
    metrics = predictions_data['metrics']
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['auc']:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(predictions_data['targets'], predictions_data['predictions'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Arrest', 'Arrest'],
                yticklabels=['No Arrest', 'Arrest'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save the confusion matrix
    os.makedirs("Final/output", exist_ok=True)
    plt.savefig("Final/output/confusion_matrix.png")
    plt.close()
    
    print("\nConfusion matrix saved to Final/output/confusion_matrix.png")
    
    # Generate ROC curve
    fpr, tpr, _ = roc_curve(predictions_data['targets'], [p[0] for p in predictions_data['probabilities']])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {metrics["auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Save the ROC curve
    plt.savefig("Final/output/roc_curve.png")
    plt.close()
    
    print("ROC curve saved to Final/output/roc_curve.png")

# ----------------------
# PREPROCESS FOR PREDICTION
# ----------------------
def preprocess_for_prediction(df, model_set, model_path_prefix="Final/"):
    """
    Prepare test data for prediction by aligning with training data format and scaling.
    """
    # Set model path based on specified model set
    model_path = f"{model_path_prefix}models/{model_set}"
    
    # Ensure the model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory {model_path} not found")
    
    # Load shared column information
    shared_columns_path = f"{model_path}/shared_columns.pkl"
    if not os.path.exists(shared_columns_path):
        raise FileNotFoundError(f"shared_columns.pkl not found in {model_path}")
        
    with open(shared_columns_path, 'rb') as f:
        shared_columns = pickle.load(f)
    print(f"Loaded column information from {shared_columns_path}")
    
    # Load saved scaler
    scaler_path = f"{model_path}/scaler.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"scaler.pkl not found in {model_path}")
    
    # Try using joblib, fall back to pickle if that fails
    try:    
        scaler = joblib.load(scaler_path)
    except (ImportError, ModuleNotFoundError):
        print("Joblib not available. Trying pickle instead...")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    print(f"Loaded scaler from {scaler_path}")
    
    # Prepare data for prediction
    print(f"Original test data shape: {df.shape}")
    
    # Remove mo_codes if present (not used in prediction)
    if 'mo_codes' in df.columns:
        df = df.drop(columns=['mo_codes'])
        print("Removed 'mo_codes' column (not used in prediction)")
    
    # Check for string columns that need conversion
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    if len(string_cols) > 0:
        print(f"Found string columns that need to be encoded: {list(string_cols)}")
        df = df.drop(columns=string_cols)
    
    # Remove potential leakage columns
    potential_leakage = ['__origin']
    for col in potential_leakage:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Removed potential leakage column: {col}")
    
    # Extract features (exclude target if present)
    if 'arrest_type' in df.columns:
        X_test = df.drop(columns=['arrest_type'])
        y_test = df['arrest_type']
        has_target = True
    else:
        X_test = df
        has_target = False
    
    # Standard preprocessing: align columns with training data
    missing_cols = set(shared_columns) - set(X_test.columns)
    extra_cols = set(X_test.columns) - set(shared_columns)
    
    # Add missing columns
    for col in missing_cols:
        X_test[col] = 0
    
    # Ensure column order matches training data
    X_test = X_test[shared_columns]
        
    # Apply scaling
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        X_test_scaled = scaler.transform(X_test)
    
    print("Feature scaling applied successfully")
    
    if has_target:
        return X_test_scaled, y_test
    else:
        return X_test_scaled

# ----------------------
# MAIN MENU FUNCTION
# ----------------------
def menu():
    # Initialize globals
    global train_df, test_df, df, train_bool, test_bool, clean_bool, trained_bool, libraries_status
    
    if 'libraries_status' not in globals():
        libraries_status = {"core": True, "data": False, "ml": False, "nn": False}

    # Initialize flags if they don't exist
    if 'train_bool' not in globals():
        train_bool = None
    if 'test_bool' not in globals():
        test_bool = None
    if 'clean_bool' not in globals():
        clean_bool = None
    if 'trained_bool' not in globals():
        trained_bool = None

    # Read In Libraries
    import_libraries()

    while True:
        print("""
        Menu Options:
        (1) Load training data
        (2) Process (Clean) data
        (3) Train Neural Network
        \t↳ Note: This step may take a while and requires significant resources.
        \t   Pre-trained models are also available.
        (4) Load testing data
        (5) Generate Predictions
        \t↳ Works with both user-trained and pre-trained models.
        (6) Display Model Accuracy & Visualizations
        (7) Clear Screen
        (8) Quit
        """)

        # Display current state
        status = []
        if train_bool:
            status.append("Training data loaded ✅")
        if test_bool:
            status.append("Test data loaded ✅")
        if clean_bool:
            status.append("Data processed ✅")
        if trained_bool:
            status.append("Models trained ✅")
            
        if status:
            print("Current state:", " | ".join(status))
            
        # Display library status warnings
        if not libraries_status["data"]:
            print("⚠️  Warning: Data processing libraries not available. Some functions may not work.")
        if not libraries_status["ml"]:
            print("⚠️  Warning: Machine learning libraries not available. Analysis functions may not work.")
        if not libraries_status["nn"]:
            print("⚠️  Warning: Neural network libraries not available. Training and prediction will not work.")

        # Buffer Delay
        sys.stdout.flush()
        time.sleep(0.1)
        
        # Get user choice with clear input
        choice = input("Enter your choice: ")
        
        # Process the user choice
        if choice == '1':
            # Load training data
            if not libraries_status["data"]:
                print("🛑 Data libraries (pandas) not available. Cannot load training data.")
                continue
                
            file_path = input("Please enter the file path for the training data (Enter for default):")
            if file_path == "":
                file_path = "../Data/Dirty/LA_Crime_Data_2023_to_Present_data.csv"
            train_df = load_data(file_path, "train")
            
        elif choice == '2':
            # Process data
            if not libraries_status["data"] or not libraries_status["ml"]:
                print("🛑 Required libraries not available. Cannot process data.")
                continue
                
            if train_bool is None and test_bool is None:
                print("🛑 You need to load either training or test data first.")
            else:
                df = process_data()
                
        elif choice == '3':
            # Train neural network - requires cleaned data
            if not libraries_status["nn"]:
                print("🛑 PyTorch libraries not available. Cannot train neural network.")
                continue
                
            if clean_bool is None:
                print("🛑 Dataset must be cleaned & loaded prior to training.")
            elif train_bool is None:
                print("🛑 Training data must be loaded to train models.")
            else:
                train_nn(df)
                
        elif choice == '4':
            # Load test data
            if not libraries_status["data"]:
                print("🛑 Data libraries (pandas) not available. Cannot load test data.")
                continue
                
            file_path = input("Please enter the file path for the test data (Enter for default):")
            if file_path == "":
                file_path = "../Data/Dirty/LA_Crime_Data_2023_to_Present_test1.csv"
            test_df = load_data(file_path, "test")
            
        elif choice == '5':
            # Generate predictions
            if not libraries_status["nn"]:
                print("🛑 PyTorch libraries not available. Cannot generate predictions.")
                continue
                
            if test_bool is None:
                print("🛑 Test data must be loaded before generating predictions.")
            elif clean_bool is None:
                print("🛑 Data must be processed before generating predictions.")
            else:
                predict(df)
                
        elif choice == '6':
            # Display accuracy
            if not libraries_status["data"] or not libraries_status["ml"]:
                print("🛑 Required libraries not available. Cannot display accuracy.")
                continue
                
            accuracy()
            
        elif choice == '7':
            # Clear screen
            clear_screen()
            
        elif choice == '8':
            # Exit
            print("👋 Goodbye! Thank you for using the Crime Prediction System.")
            break
            
        else:
            print("🛑 Invalid Choice. Please Try Again.")

if __name__ == "__main__":
    menu()