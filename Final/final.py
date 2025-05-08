#!/usr/bin/env python
# coding: utf-8

# Final Notebook for CS 3500 

import sys
import time
import threading
import itertools
import warnings
import os

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

def spinner(message, show_done=False):
    """
    
    """
    done_event = threading.Event()

    def spin():
        for c in itertools.cycle('|/-\\'):
            if done_event.is_set():
                break
            sys.stdout.write(f'\r{message}... {c}')
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * 50 + '\r')
        if show_done:
            sys.stdout.write('Done! ✅\n')
            sys.stdout.flush()

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()

    # Wrap the stop function to also join the thread
    def stop_and_join():
        done_event.set()
        thread.join()

    return stop_and_join

# Clear screen function for better UI organization
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# ----------------------
# IMPORT LIBRARIES
# ----------------------
def import_libraries():
    stop_spinner = spinner("Loading dependencies...", show_done=True)

    # Core modules
    import os
    import sys
    import time
    import random
    from datetime import datetime, timedelta
    from collections import Counter
    import itertools

    # Data handling & visualization
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm

    # Sklearn tools
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

    # PyTorch
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    # Resampling
    from imblearn.over_sampling import SMOTE

    # Promote to global scope
    globals().update({
        # Core
        'os': os,
        'sys': sys,
        'time': time,
        'random': random,
        'datetime': datetime,
        'timedelta': timedelta,
        'Counter': Counter,
        'itertools': itertools,

        # Data & viz
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns,
        'tqdm': tqdm,

        # Sklearn
        'LabelEncoder': LabelEncoder,
        'PowerTransformer': PowerTransformer,
        'CountVectorizer': CountVectorizer,
        'PCA': PCA,
        'KMeans': KMeans,
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

        # PyTorch
        'torch': torch,
        'nn': nn,
        'Dataset': Dataset,
        'DataLoader': DataLoader,
        'ReduceLROnPlateau': ReduceLROnPlateau,

        # Others
        'SMOTE': SMOTE,
    })

    stop_spinner()

# Neural Network Global Class Definitions
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    stop_spinner = spinner(f"\nLoading {label}ing set:")
    print("\n*********************")

    global train_bool, test_bool
    if label == "train":
        train_bool = True
    elif label == "test":
        test_bool = True
    
    stop_spinner()
    start = time.time()
    df = pd.read_csv(file_path)
    print(f"Total Columns Read: {df.shape[1]}")
    print(f"Total Rows Read: {df.shape[0]}")
    print("\nTime to load is:", round(time.time() - start, 2), "seconds")

    return df

# ----------------------
# PROCESS (CLEAN) DATA
# ----------------------
def process_data():
    # Mark Bool As Clean For NN
    global train_df, test_df, clean_bool
    clean_bool = True

    # Combine Data For Cleaning
    stop_spinner = spinner("Cleaning Data....", show_done=True)
    start = time.time()

    # Fix the origin flags 
    train_df["__origin"] = 0
    test_df["__origin"] = 1
    
    # Create backup of test data
    test_backup = test_df.copy()
    
    # Combine datasets
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

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

    stop_spinner()
    print(f"Time Elapsed: {round(time.time() - start, 2)} seconds")
    return df

# ----------------------
# TRAIN NEURAL NETWORK
# ----------------------
# Helper Function
def init_models(input_dim):
    global trans_model, mlp_model
    trans_model = TabTransformer(input_dim).to(device)
    mlp_model = MLPModel(input_dim).to(device)

def train_nn(df):
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
    
    # Save the scaler - remove joblib
    os.makedirs("models/script_models", exist_ok=True)
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
    global trans_model, mlp_model, device, predictions_data
    
    # Initial spinner for the overall process
    stop_spinner = spinner("Preparing data for prediction", show_done=False)
    
    try:
        print("\n*********************")
        print("Evaluating models with the test data")
        print(f"Data shape: {df.shape}")
        
        # Make sure we're only dealing with binary classification
        df = df[df['arrest_type'].isin([0, 1])].reset_index(drop=True)
        
        # Drop mo_codes column if it exists
        if 'mo_codes' in df.columns:
            print("Dropping mo_codes column...")
            df = df.drop(columns=['mo_codes'])
        
        # Check for any remaining string columns that might cause issues
        string_cols = df.select_dtypes(include=['object', 'string']).columns
        if len(string_cols) > 0:
            print(f"Found and dropped string columns: {list(string_cols)}")
            df = df.drop(columns=string_cols)
        
        # Check for potential leakage columns
        potential_leakage = [col for col in df.columns if any(term in col.lower() for term in 
                            ['id', 'dr_no', 'code', 'status', 'report', '__origin'])]
        
        if potential_leakage:
            if '__origin' in df.columns:
                print("Removing __origin column used for data splitting")
                df = df.drop(columns=['__origin'])
        
        # Split data 80/20
        train_size = int(0.8 * len(df))
        train_data = df[:train_size].reset_index(drop=True)
        test_data = df[train_size:].reset_index(drop=True)
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Testing data shape: {test_data.shape}")
        
        # Separate features and target
        X_test = test_data.drop(columns=["arrest_type"])
        y_test = test_data["arrest_type"]
        X_train = train_data.drop(columns=["arrest_type"])
        y_train = train_data["arrest_type"]
        
        stop_spinner()
        
        # Preprocess the data with warnings suppressed
        stop_spinner = spinner("Scaling features", show_done=False)
        
        # Suppress ALL warnings from scaling operation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            scaler = PowerTransformer(method='yeo-johnson')
            scaler.fit(X_train)  # Fit on training data
            X_test_scaled = scaler.transform(X_test)  # Transform test data
        
        stop_spinner()
        
        # Load models
        stop_spinner = spinner("Loading pre-trained models", show_done=False)
        
        input_dim = X_test.shape[1]
        
        # Check model directory structure
        model_dirs = [
            "models/bolt_models",
            "Final/models/bolt_models", 
            "../models/bolt_models",
            "./models/bolt_models",
            "/Users/noahgallego/Desktop/CS3500_Project/Final/models/bolt_models"
        ]
        
        model_path = None
        for dir_path in model_dirs:
            if os.path.exists(f"{dir_path}/model_1.pt"):
                model_path = dir_path
                break
        
        if model_path is None:
            raise FileNotFoundError("Could not find model directory")
        
        # Initialize models with correct input dimension
        if trans_model is None or mlp_model is None:
            init_models(input_dim)
        
        # Load the pre-trained weights
        model1_path = f"{model_path}/model_1.pt"
        model2_path = f"{model_path}/model_2.pt"
        
        try:
            trans_model.load_state_dict(torch.load(model1_path, map_location=device))
            mlp_model.load_state_dict(torch.load(model2_path, map_location=device))
        except Exception as e:
            raise Exception(f"Failed to load models: {e}")
        
        # Set models to evaluation mode
        trans_model.eval()
        mlp_model.eval()
        
        stop_spinner()
        
        # Make predictions
        stop_spinner = spinner("Running inference with ensemble models", show_done=False)
        
        # Create a PyTorch dataset and dataloader for efficient batching
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
        
        # Calculate metrics
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
        
        # Display 5 random predictions
        import random
        random_indices = random.sample(range(len(all_predictions)), min(5, len(all_predictions)))
        
        print("\n=== Sample Predictions ===")
        for idx in random_indices:
            sample_idx = test_data.index[idx]
            true_label = "Arrest" if all_targets[idx] == 1 else "No Arrest"
            pred_label = "Arrest" if all_predictions[idx] == 1 else "No Arrest"
            prob = all_probabilities[idx][0]
            
            print(f"Sample #{sample_idx}:")
            print(f"  True: {true_label}")
            print(f"  Predicted: {pred_label} (Confidence: {prob:.4f})")
        
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
        
        print("\nPrediction complete! ✅")
        
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
# MAIN MENU FUNCTION
# ----------------------
def menu():
    # Mark DF as Global for NN
    global train_df, test_df, df

    # Read In Libraries
    import_libraries()

    while True:
        print("""
        Menu Options:
        (1) Load training data
        (2) Process (Clean) data
        (3) Train Neural Network
        \t↳ Note: This step may take a while and requires significant resources.
        \t   Choose option (5) to use pre-trained models instead.
        (4) Load testing data
        (5) Generate Predictions using pre-trained models
        (6) Display Model Accuracy & Visualizations
        (7) Clear Screen
        (8) Quit
        """)

        # Buffer Delay
        sys.stdout.flush()
        time.sleep(0.1)
        choice = input("Enter your choice: ")

        if choice == '1':
            train_df = load_data("../Data/Dirty/LA_Crime_Data_2023_to_Present_data.csv", "train")
        elif choice == '2':
            if train_bool is None or test_bool is None:
                print("🛑 Both training and testing data must be loaded before cleaning.")
            else:
                df = process_data()
        elif choice == '3':
            if clean_bool is None:
                print("🛑 Dataset must be cleaned & loaded prior to training.")
            else:
                train_nn(df)
        elif choice == '4':
            test_df = load_data("../Data/Dirty/LA_Crime_Data_2023_to_Present_test1.csv", "test")
        elif choice == '5':
            if clean_bool is None:
                print("🛑 Dataset must be cleaned before generating predictions.")
            else:
                predict(df)
        elif choice == '6':
            accuracy()
        elif choice == '7':
            clear_screen()
        elif choice == '8':
            print("👋 Goodbye! Thank you for using the Crime Prediction System.")
            break
        else:
            print("🛑 Invalid Choice. Please Try Again.")

if __name__ == "__main__":
    menu()