import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import log  # Import logging utility
from src.data_process.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4,
    PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
)
import datetime

# Dictionary mapping dataset names to corresponding classes and paths
DATASETS_INFO = {
    "ETTh1": {"class": Dataset_ETT_hour, "path": "data/ETTh1.csv"},
    "ETTh2": {"class": Dataset_ETT_hour, "path": "../data/ETTh2.csv"},
    "ETTm1": {"class": Dataset_ETT_minute, "path": "../data/ETTm1.csv"},
    "ETTm2": {"class": Dataset_ETT_minute, "path": "../data/ETTm2.csv"},
    "electricity": {"class": Dataset_Custom, "path": "../data/electricity.csv"},
    "traffic": {"class": Dataset_Custom, "path": "../data/traffic.csv"},
    "weather": {"class": Dataset_Custom, "path": "../data/weather.csv"},
    "m4": {"class": Dataset_M4, "path": "../data/m4.csv"},
    "PSM": {"class": PSMSegLoader, "path": "../data/PSM.csv"},
    "MSL": {"class": MSLSegLoader, "path": "../data/MSL.csv"},
    "SMAP": {"class": SMAPSegLoader, "path": "../data/SMAP.csv"},
    "SMD": {"class": SMDSegLoader, "path": "../data/SMD.csv"},
    "SWAT": {"class": SWATSegLoader, "path": "../data/SWAT.csv"},
    "UEA": {"class": UEAloader, "path": "../data/UEA.csv"}
}

class TimeSeriesDataset(Dataset):
    """Custom PyTorch Dataset for time series data."""
    def __init__(self, X, y, seq_x_mark, seq_y_mark):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        self.seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.seq_x_mark[idx], self.seq_y_mark[idx]

def preprocess_df(df):
    """Preprocess the dataset by filling missing values."""
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    return df

def preprocess_data(df, context_length, prediction_horizon, freq='h', n_samples=None):
    """
    Preprocess the data, apply sliding windows, and generate time-related features.

    Parameters:
    - df (DataFrame): The time series data.
    - context_length (int): The length of the context (input sequence).
    - prediction_horizon (int): The length of the prediction horizon (output sequence).
    - freq (str): The frequency of the time series data ('h' for hourly, 'd' for daily, etc.).

    Returns:
    - X (np.array): Input data with shape (n_samples, context_length, n_features).
    - y (np.array): Target data with shape (n_samples, prediction_horizon, n_features).
    - seq_x_mark (np.array): Time-related features for input with shape (n_samples, context_length, 4).
    - seq_y_mark (np.array): Time-related features for target with shape (n_samples, prediction_horizon, 4).
    """
    if n_samples is None:
        n_samples = len(df) - context_length - prediction_horizon  # Number of samples for the sliding window

    # Ensure n_samples does not exceed the available data length
    n_samples = min(n_samples, len(df) - context_length - prediction_horizon + 1)

    # Extract numerical columns, excluding the 'date' column
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude date-related columns by checking column names
    numerical_cols = [col for col in numerical_cols if not col.startswith('date')]
    n_features = len(numerical_cols)  # Number of numerical features (columns)

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the numerical columns of the dataframe and transform the data
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Create arrays for X, y, seq_x_mark, seq_y_mark
    X = np.zeros((n_samples, context_length, n_features))  # Features (input sequence)
    y = np.zeros((n_samples, prediction_horizon, n_features))  # Targets (output sequence)
    seq_x_mark = np.zeros((n_samples, context_length, 4))  # Time-related features for seq_x
    seq_y_mark = np.zeros((n_samples, prediction_horizon, 4))  # Time-related features for seq_y

    # Check if 'date' column exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        # If 'date' column does not exist, create a default datetime index
        start_date = datetime.datetime(2020, 1, 1)
        if freq == 'h':  # hourly
            time_index = [start_date + datetime.timedelta(hours=i) for i in range(len(df))]
        elif freq == 'd':  # daily
            time_index = [start_date + datetime.timedelta(days=i) for i in range(len(df))]
        else:
            raise ValueError(f"Unsupported frequency: {freq}. Please use 'h' for hourly or 'd' for daily.")
        df['date'] = pd.to_datetime(time_index)  # Assign the generated date column

    # Now that we have a date column, we can generate time-related features
    for i in range(n_samples):
        # Define X and y as the sliding window
        X[i] = df.iloc[i:i + context_length][numerical_cols].values
        y[i] = df.iloc[i + context_length:i + context_length + prediction_horizon][numerical_cols].values

        # Extract time-related features for each window (seq_x_mark and seq_y_mark)
        for j in range(context_length):
            time_stamp_x = df.iloc[i + j]['date']
            seq_x_mark[i, j, 0] = time_stamp_x.month      # month
            seq_x_mark[i, j, 1] = time_stamp_x.day        # day
            seq_x_mark[i, j, 2] = time_stamp_x.weekday()  # weekday (0=Monday, 6=Sunday)
            seq_x_mark[i, j, 3] = time_stamp_x.hour       # hour of the day

        for j in range(prediction_horizon):
            time_stamp_y = df.iloc[i + context_length + j]['date'] if i + context_length + j < len(df) else time_stamp_x
            seq_y_mark[i, j, 0] = time_stamp_y.month      # month
            seq_y_mark[i, j, 1] = time_stamp_y.day        # day
            seq_y_mark[i, j, 2] = time_stamp_y.weekday()  # weekday
            seq_y_mark[i, j, 3] = time_stamp_y.hour       # hour

    return X, y, seq_x_mark, seq_y_mark

def load_predefined_dataset(config):
    """Loads predefined datasets based on the configuration."""
    dataset_name = config.get("data_name", "")
    if dataset_name not in DATASETS_INFO:
        log(f"❌ Dataset '{dataset_name}' not found in predefined datasets.")
        return None, None, None

    dataset_info = DATASETS_INFO[dataset_name]
    dataset_class = dataset_info["class"]
    dataset_path = dataset_info["path"]
    # root_path = "/".join(dataset_path.split("/")[:-1])
    root_path = ""
    size = (config["window_size"], config["label_len"], config["prediction_horizon"])

    print(dataset_path)
    print(root_path)

    log(f"📂 Loading dataset: {dataset_name} from {dataset_path}")

    train_loader = DataLoader(dataset_class(root_path=root_path, data_path=dataset_path, flag='train', size=size,
                                            features='M', target='OT'),
                              batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset_class(root_path=root_path, data_path=dataset_path, flag='val', size=size,
                                          features='M', target='OT'),
                            batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(dataset_class(root_path=root_path, data_path=dataset_path, flag='test', size=size,
                                           features='M', target='OT'),
                             batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader

def load_custom_dataset(config):
    """Loads a custom dataset from CSV and processes it for training."""
    train_file = config.get("data_name", "")
    if not train_file:
        log("❌ No train file specified in config.")
        return None, None, None

    log(f"📂 Loading custom dataset from {train_file}")
    train_df = preprocess_df(pd.read_csv(train_file))
    # print(train_df.shape)

    X, y, seq_x_mark, seq_y_mark = preprocess_data(train_df, config["window_size"], config["prediction_horizon"], freq='h',
                                                n_samples=config.get("n_samples"))

    X_train, X_temp, y_train, y_temp, seq_x_mark_train, seq_x_mark_temp, seq_y_mark_train, seq_y_mark_temp = train_test_split(
        X, y, seq_x_mark, seq_y_mark, test_size=0.4, random_state=42)

    X_val, X_test, y_val, y_test, seq_x_mark_val, seq_x_mark_test, seq_y_mark_val, seq_y_mark_test = train_test_split(
        X_temp, y_temp, seq_x_mark_temp, seq_y_mark_temp, test_size=0.5, random_state=42)

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train, seq_x_mark_train, seq_y_mark_train),
                              batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val, seq_x_mark_val, seq_y_mark_val),
                            batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test, seq_x_mark_test, seq_y_mark_test),
                             batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader

def load_data_sota(config):
    """
    Loads dataset based on configuration.
    - Uses predefined dataset if available.
    - Otherwise, loads and processes a custom dataset.
    """
    log("🔄 Loading dataset...")
    print(config.get("data_name"))
    if config.get("data_name", "") in DATASETS_INFO:
        return load_predefined_dataset(config)
    return load_custom_dataset(config)
