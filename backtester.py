import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from ta.others import DailyReturnIndicator
import warnings

# Suppress specific warnings from pandas or ta library that might clutter output
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configuration ---
DATA_DIR = "/workspaces/Catboost/data/1_min" # Your data directory

# --- Target/Stop-Loss Multipliers for Target Creation (how the model was trained) ---
# These are used in the create_target function to define the 'target_class' labels
# that the CatBoost model was originally trained to predict.
TARGET_MULTIPLIER_FOR_TARGET_CREATION = 1.0  
STOPLOSS_MULTIPLIER_FOR_TARGET_CREATION = 0.5 

# --- NEW: Percentage-based Target and Stop-Loss for Signal Output ---
# These are used in the signal generation loop to calculate the target/stop-loss
# levels that will be printed in your signal list.
TARGET_PERCENTAGE = 0.02  # 2% target profit
STOPLOSS_PERCENTAGE = 0.01 # 1% stop-loss

RANDOM_STATE = 42
MAX_FILES_TO_PROCESS = 5 # Limit to 5 files for faster execution/debugging

# Best Hyperparameters from your tuning process
# This is kept for reference, but the model will be loaded, not initialized with these.
BEST_PARAMS = {
    'subsample': 0.9,
    'min_data_in_leaf': 1,
    'learning_rate': 0.05,
    'l2_leaf_reg': 9,
    'iterations': 1000,
    'depth': 6,
    'colsample_bylevel': 0.6
}

# Path to your saved CatBoost model
# IMPORTANT: Ensure this path is correct and the model file exists!
MODEL_LOAD_PATH = "/workspaces/Catboost/catboost_stock_classifier.cbm"

# Custom Prediction Threshold for generating trade signals
# Adjust this value to balance precision and recall for Class 1
# A higher threshold increases precision (fewer false positives) but may decrease recall.
# A lower threshold increases recall (more true positives) but may decrease precision.
CUSTOM_PREDICTION_THRESHOLD = 0.65 # Changed to 0.65 for a stronger filter

# Backtesting Window Configuration (for single backtest)
TEST_WINDOW_MONTHS = 1  # 1 month for the final test set

# --- Data Preprocessing Functions (Copied for self-containment) ---

def calculate_features(df):
    """
    Calculate all required technical indicators and custom features for the given DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.
        
    Returns:
        pd.DataFrame: DataFrame with calculated technical and custom features, with initial NaN rows dropped.
    """
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd_hist'] = macd.macd_diff()
    df['close_pct_change'] = DailyReturnIndicator(df['close']).daily_return()
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr.average_true_range()
    df['hourly_range'] = (df['high'].rolling(60).max() - df['low'].rolling(60).min()) / df['close'].rolling(60).mean()
    vwap = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=20)
    df['volume_osc'] = (df['close'] - vwap.volume_weighted_average_price()) / vwap.volume_weighted_average_price()
    df['sma'] = df['close'].rolling(window=20).mean()
    df['volatility_adj_rsi'] = df['rsi'] / (df['atr'].rolling(window=14).std() + 1e-9)
    df['momentum_ratio'] = df['close'].pct_change(5) / (df['close'].pct_change(10) + 1e-9)
    df['normalized_range'] = (df['high'] - df['low']) / df['close']

    df['close_pct_change_3min'] = df['close'].pct_change(3)
    df['close_pct_change_5min'] = df['close'].pct_change(5)
    df['volume_change_1min'] = df['volume'].pct_change(1)
    df['volume_avg_5min'] = df['volume'].rolling(window=5).mean()
    df['volume_avg_ratio_5min'] = df['volume'] / (df['volume_avg_5min'] + 1e-9)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df.dropna()

def create_target(df):
    """
    Create classification target based on ATR-driven target and stoploss levels.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'close', 'high', 'low', and 'atr' columns.
        
    Returns:
        pd.DataFrame: DataFrame with 'target' and 'target_class' columns added.
    """
    df_copy = df.copy()
    entry_price = df_copy['close'].values
    atr = df_copy['atr'].values
    high_prices = df_copy['high'].values
    low_prices = df_copy['low'].values

    # Using the multipliers specifically for target creation (how the model was trained)
    target_level = entry_price + (TARGET_MULTIPLIER_FOR_TARGET_CREATION * atr) 
    stop_level = entry_price - (STOPLOSS_MULTIPLIER_FOR_TARGET_CREATION * atr) 
    target_values = np.zeros(len(df_copy), dtype=int)
    
    lookahead = 360 # 6 hours

    for i in range(len(df_copy) - lookahead):
        future_high_window = high_prices[i+1 : i+lookahead+1]
        future_low_window = low_prices[i+1 : i+lookahead+1]
        
        if future_high_window.size == 0:
            continue

        current_target = target_level[i]
        current_stop = stop_level[i]
        
        target_hit_indices = np.where(future_high_window >= current_target)[0]
        stop_hit_indices = np.where(future_low_window <= current_stop)[0]

        target_hit_first = False
        stop_hit_first = False

        # Initialize these variables to None or a default value before conditional assignment
        first_target_hit_idx = None
        first_stop_hit_idx = None

        if target_hit_indices.size > 0:
            first_target_hit_idx = target_hit_indices[0]
            target_hit_first = True
        
        if stop_hit_indices.size > 0:
            first_stop_hit_idx = stop_hit_indices[0]
            stop_hit_first = True

        if target_hit_first and stop_hit_first:
            if first_target_hit_idx < first_stop_hit_idx:
                target_values[i] = 1
            else:
                target_values[i] = -1
        elif target_hit_first:
            target_values[i] = 1
        elif stop_hit_first:
            target_values[i] = -1
            
    df_copy['target'] = target_values
    df_copy['target_class'] = (df_copy['target'] == 1).astype(int)
    
    return df_copy

def load_and_preprocess_single_file(file_path):
    """
    Load and preprocess a single stock data file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with features and target, or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        df = calculate_features(df)
        df = create_target(df)
        
        features = [
            'rsi', 'macd_hist', 'close_pct_change', 'atr', 'hourly_range',
            'volume_osc', 'sma', 'volatility_adj_rsi', 'momentum_ratio', 'normalized_range',
            'close_pct_change_3min', 'close_pct_change_5min', 'volume_change_1min', 'volume_avg_ratio_5min'
        ]
        
        return df[['datetime'] + features + ['target_class', 'close', 'high', 'low', 'atr']] # Include price data for trade simulation
    except Exception as e:
        print(f"Error loading or preprocessing {file_path}: {e}")
        return None

def load_all_data():
    """
    Loads and preprocesses all stock files up to MAX_FILES_TO_PROCESS.
    """
    all_data = []
    processed_files_count = 0
    print(f"Loading data from: {DATA_DIR}")
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith('.csv'):
            if processed_files_count >= MAX_FILES_TO_PROCESS:
                print(f"Reached limit of {MAX_FILES_TO_PROCESS} files. Skipping remaining files.")
                break
            file_path = os.path.join(DATA_DIR, file_name)
            df = load_and_preprocess_single_file(file_path)
            if df is not None:
                all_data.append(df)
                processed_files_count += 1
                print(f"Processed {file_name} with {len(df)} samples")
    
    if not all_data:
        raise ValueError("No valid data files found in the directory.")
    
    combined_df = pd.concat(all_data)
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    
    print(f"\nTotal samples across all files: {len(combined_df)}")
    print(f"Combined data covers period from {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
    
    return combined_df

def main():
    """
    Main function to load a pre-trained model and generate a signal list on the last month of data.
    """
    combined_data = load_all_data()

    min_date = combined_data['datetime'].min()
    max_date = combined_data['datetime'].max()

    # --- Single Backtest Logic ---
    # Define the test period as the last 1 month of data
    test_end_date = max_date
    test_start_date = max_date - pd.DateOffset(months=TEST_WINDOW_MONTHS)
    
    # Ensure test_start_date doesn't go before the beginning of the data
    if test_start_date < min_date:
        test_start_date = min_date
        print(f"\nWarning: Not enough data for a full {TEST_WINDOW_MONTHS}-month test period. Test start date adjusted to {test_start_date}.")

    # Training data is all data before the test period (used for context, but model is loaded)
    train_end_date = test_start_date
    train_start_date = min_date 

    print(f"\n--- Signal Generation: Using data from {test_start_date} to {test_end_date} ---")

    # Slice data for testing (signal generation)
    test_df = combined_data[(combined_data['datetime'] >= test_start_date) & (combined_data['datetime'] <= test_end_date)].copy()

    if test_df.empty:
        raise ValueError("Not enough data to form a test set for signal generation. Ensure combined data has at least 1 month of data.")

    # Define features based on what was used during training (this list should match the model's expectation)
    # This list is used to extract features from test_df, and then reordered to match the loaded model.
    all_possible_features = [col for col in combined_data.columns if col not in ['datetime', 'target_class', 'target', 'close', 'high', 'low', 'atr']]
    
    X_test = test_df[all_possible_features]
    y_test = test_df['target_class'] # Keep y_test for evaluation metrics

    # --- Load the pre-trained model ---
    print(f"\nLoading pre-trained model from {MODEL_LOAD_PATH}...")
    try:
        model = CatBoostClassifier()
        model.load_model(MODEL_LOAD_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading model from {MODEL_LOAD_PATH}: {e}. Please ensure the model file exists and is valid.")

    # --- FIX: Ensure X_test columns match model's expected feature names and order ---
    # Get the feature names the loaded model expects
    expected_features_from_model = model.feature_names_
    
    # Reindex X_test to match the exact order and names expected by the loaded model
    # This will align columns and fill any missing ones (though ideally all should be present)
    X_test = X_test.reindex(columns=expected_features_from_model, fill_value=0.0)

    # Get probabilities on the test set for signal generation
    test_proba = model.predict_proba(X_test)
    test_predictions_adjusted = (test_proba[:, 1] >= CUSTOM_PREDICTION_THRESHOLD).astype(int)

    # Log model performance for the test set
    report = classification_report(y_test, test_predictions_adjusted, output_dict=True, zero_division=0)
    print(f"  Test Metrics (Threshold {CUSTOM_PREDICTION_THRESHOLD}):")
    print(classification_report(y_test, test_predictions_adjusted, zero_division=0))

    # --- Generate Signal List ---
    signal_data = []
    print("\nGenerating signal list on the test set...")
    for i in range(len(test_df)):
        if test_predictions_adjusted[i] == 1:
            signal_row = test_df.iloc[i]
            entry_datetime = signal_row['datetime']
            
            # --- FIX: Robust scalar extraction using try-except with .item() ---
            # This handles cases where pandas might return a Series of length 1 or a direct scalar.
            try:
                entry_price = float(signal_row['close'].item())
            except (ValueError, AttributeError): # Catch both errors if .item() fails or is not available
                entry_price = float(signal_row['close']) # Fallback to direct conversion if it's already a scalar

            try:
                atr_at_entry = float(signal_row['atr'].item())
            except (ValueError, AttributeError):
                atr_at_entry = float(signal_row['atr'])
            
            # --- Percentage-based Target and Stop-Loss Calculation ---
            target_level = entry_price * (1 + TARGET_PERCENTAGE)
            stop_level = entry_price * (1 - STOPLOSS_PERCENTAGE)

            signal_data.append({
                'signal_datetime': entry_datetime,
                'entry_price': entry_price, 
                'target_level': target_level,
                'stop_level': stop_level,
                'predicted_class': 1 # Always 1 for signals
            })
    
    signal_log_df = pd.DataFrame(signal_data)

    print("\n--- Signal List Summary ---")
    if not signal_log_df.empty:
        total_signals = len(signal_log_df)
        print(f"Total Signals Generated: {total_signals}")
        
        # Filter for signals on the last day of the test period
        # test_end_date is already defined and is the last day
        last_day_signals_df = signal_log_df[signal_log_df['signal_datetime'].dt.date == test_end_date.date()]
        
        total_last_day_signals = len(last_day_signals_df) # Count filtered signals
        print(f"\nTotal Signals Generated on the Last Day ({test_end_date.date()}): {total_last_day_signals}")
        
        # Print the DataFrame to console
        print("\n--- Detailed Signal List (Last Day Only) ---")
        if not last_day_signals_df.empty:
            print(last_day_signals_df.to_string()) # Use to_string() to avoid truncation
        else:
            print("No signals generated on the last day with the current threshold.")
    else:
        print("No signals were generated during the backtest period with the current threshold.")

if __name__ == "__main__":
    main()
