import os
import pandas as pd
import numpy as np
import sqlite3
from catboost import CatBoostClassifier
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
DB_PATH = "/workspaces/Catboost/data/market_data.db" # Path to your SQLite database
TABLE_NAME = "ohlcv_data" # Name of your table containing OHLCV data

# --- Target/Stop-Loss Multipliers for Target Creation (how the model was trained) ---
# These are used in the create_target function to define the 'target_class' labels
# that the CatBoost model was originally trained to predict.
TARGET_MULTIPLIER_FOR_TARGET_CREATION = 1.0  
STOPLOSS_MULTIPLIER_FOR_TARGET_CREATION = 0.5 

# --- Percentage-based Target and Stop-Loss for Signal Output ---
# These are used in the signal generation loop to calculate the target/stop-loss
# levels that will be printed in your signal list.
TARGET_PERCENTAGE = 0.02  # 2% target profit
STOPLOSS_PERCENTAGE = 0.01 # 1% stop-loss

RANDOM_STATE = 42

# Path to your saved CatBoost model
# IMPORTANT: Ensure this path is correct and the model file exists!
MODEL_LOAD_PATH = "/workspaces/Catboost/catboost_stock_classifier.cbm"

# Custom Prediction Threshold for generating trade signals
# Adjust this value to balance precision and recall for Class 1
# A higher threshold increases precision (fewer false positives) but may decrease recall.
# A lower threshold increases recall (more true positives) but may decrease precision.
CUSTOM_PREDICTION_THRESHOLD = 0.65 

# Minimum number of historical bars required to calculate all features
# This should be at least the largest window size used in calculate_features (e.g., 60 for hourly_range, 26 for MACD)
MIN_HISTORY_FOR_FEATURES = 60 

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

    return df # Do not dropna here, handle NaN for latest row in live loop

def create_target(df):
    """
    Create classification target based on ATR-driven target and stoploss levels.
    This function is primarily for how the model was trained, not for live signal output.
    
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

        first_target_hit_idx = None # Initialize
        first_stop_hit_idx = None # Initialize

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

def load_data_from_db(db_path, table_name, days_history=30):
    """
    Loads OHLCV data from SQLite database for the last 'days_history' days.
    
    Args:
        db_path (str): Path to the SQLite database.
        table_name (str): Name of the table containing OHLCV data.
        days_history (int): Number of days of historical data to load.
        
    Returns:
        pd.DataFrame: DataFrame with loaded OHLCV data, sorted by datetime.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        # Query for data from the last 'days_history' days
        query = f"""
        SELECT datetime, open, high, low, close, volume 
        FROM {table_name} 
        WHERE datetime >= strftime('%Y-%m-%d %H:%M:%S', date('now', '-{days_history} days'))
        ORDER BY datetime ASC;
        """
        df = pd.read_sql_query(query, conn)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"Loaded {len(df)} rows from {db_path} for the last {days_history} days.")
        if not df.empty:
            print(f"Data range: {df['datetime'].min()} to {df['datetime'].max()}")
        return df
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def main():
    """
    Main function to run the live signal filter.
    """
    if not os.path.exists(MODEL_LOAD_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_LOAD_PATH}. Please ensure your trained model is saved there.")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found at: {DB_PATH}. Please ensure market_data.db is in the correct location.")

    # --- Load the pre-trained model ---
    print(f"\nLoading pre-trained model from {MODEL_LOAD_PATH}...")
    try:
        model = CatBoostClassifier()
        model.load_model(MODEL_LOAD_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading model from {MODEL_LOAD_PATH}: {e}. Please ensure the model file exists and is valid.")

    # Get the feature names the loaded model expects
    expected_features_from_model = model.feature_names_
    print(f"Model expects features: {expected_features_from_model}")

    # --- Load initial historical data from DB ---
    # We load more than 30 days if available, to ensure enough history for feature calculation
    # and to simulate live data from the end of this history.
    full_history_df = load_data_from_db(DB_PATH, TABLE_NAME, days_history=60) # Load more history to ensure rolling window works
    if full_history_df.empty:
        raise ValueError("No data loaded from the database. Cannot proceed with live signal generation.")

    # Separate initial history for feature warm-up and data for live simulation
    # Use the first part for initial rolling_df, and the rest to simulate live data.
    # For simplicity, let's say the 'live' simulation starts after the first MIN_HISTORY_FOR_FEATURES bars
    # or after 30 days, whichever is later.
    
    # Calculate initial features for the entire history
    processed_full_history_df = calculate_features(full_history_df.copy())
    
    if processed_full_history_df.empty:
        raise ValueError("No valid data after feature calculation. Check your raw data and feature logic.")

    # Determine the split point for initial history vs. simulated live data
    # We need at least MIN_HISTORY_FOR_FEATURES bars to calculate features reliably for the first live bar.
    # Let's use the last 30 days as the "test" period, and data before that as "training history" for features.
    
    # Find the datetime 30 days ago from the latest data point
    thirty_days_ago = processed_full_history_df['datetime'].max() - pd.DateOffset(days=30)
    
    # Initial rolling history for feature calculation (data before the last 30 days)
    # Ensure this part is large enough for MIN_HISTORY_FOR_FEATURES
    initial_rolling_history = processed_full_history_df[processed_full_history_df['datetime'] < thirty_days_ago].copy()

    # Data to simulate as "live" incoming bars (the last 30 days)
    live_simulation_data = processed_full_history_df[processed_full_history_df['datetime'] >= thirty_days_ago].copy()

    if initial_rolling_history.empty or len(initial_rolling_history) < MIN_HISTORY_FOR_FEATURES:
        print(f"Warning: Initial history for feature calculation is less than {MIN_HISTORY_FOR_FEATURES} bars. "
              "Features for early live bars might be NaN. Consider loading more historical data.")
        # If initial history is too short, just start rolling_df with whatever we have
        rolling_df = processed_full_history_df.head(MIN_HISTORY_FOR_FEATURES).copy()
        live_simulation_data_start_idx = MIN_HISTORY_FOR_FEATURES
    else:
        rolling_df = initial_rolling_history.tail(MIN_HISTORY_FOR_FEATURES).copy() # Start with the tail for rolling window
        live_simulation_data_start_idx = processed_full_history_df[processed_full_history_df['datetime'] >= thirty_days_ago].index.min()
        if pd.isna(live_simulation_data_start_idx): # Handle case where thirty_days_ago is beyond max_date
            live_simulation_data_start_idx = len(processed_full_history_df) # No live data, empty loop
        else:
            live_simulation_data_start_idx = processed_full_history_df.index.get_loc(live_simulation_data_start_idx)


    print(f"\nSimulating live data arrival from {live_simulation_data['datetime'].min()} to {live_simulation_data['datetime'].max()}...")
    
    generated_signals = []

    # Simulate live data bar by bar
    for i in range(len(live_simulation_data)):
        current_live_bar = live_simulation_data.iloc[i:i+1].copy() # Get one row as a DataFrame
        
        # Append new bar and trim rolling_df to maintain size for feature calculation
        rolling_df = pd.concat([rolling_df, current_live_bar], ignore_index=True)
        if len(rolling_df) > MIN_HISTORY_FOR_FEATURES:
            rolling_df = rolling_df.iloc[-MIN_HISTORY_FOR_FEATURES:].copy()

        # Only calculate features and predict if we have enough history
        if len(rolling_df) >= MIN_HISTORY_FOR_FEATURES:
            # Recalculate features on the rolling window
            features_for_prediction_df = calculate_features(rolling_df.copy())
            
            # Get the features for the latest bar (last row)
            # Ensure the last row is not NaN for features
            live_features_row = features_for_prediction_df.iloc[-1]
            
            # Check for NaN in critical features before prediction
            if live_features_row[expected_features_from_model].isnull().any():
                # print(f"Skipping prediction for {live_features_row['datetime']} due to NaN features.")
                continue

            # Prepare X_live for prediction, ensuring correct column order
            X_live = pd.DataFrame([live_features_row[expected_features_from_model].values], columns=expected_features_from_model)
            
            # Get probabilities
            prediction_proba = model.predict_proba(X_live)[:, 1][0] # Get the probability for Class 1

            # Check for signal
            if prediction_proba >= CUSTOM_PREDICTION_THRESHOLD:
                entry_datetime = live_features_row['datetime']
                entry_price = float(live_features_row['close']) # Directly access scalar
                
                # Percentage-based Target and Stop-Loss Calculation
                target_level = entry_price * (1 + TARGET_PERCENTAGE)
                stop_level = entry_price * (1 - STOPLOSS_PERCENTAGE)

                generated_signals.append({
                    'signal_datetime': entry_datetime,
                    'entry_price': entry_price,
                    'target_level': target_level,
                    'stop_level': stop_level,
                    'predicted_proba': prediction_proba
                })
                # print(f"SIGNAL: {entry_datetime} | Price: {entry_price:.2f} | Target: {target_level:.2f} | Stop: {stop_level:.2f} | Proba: {prediction_proba:.4f}")

    signal_log_df = pd.DataFrame(generated_signals)

    print("\n--- Live Signal Generation Summary ---")
    if not signal_log_df.empty:
        total_signals = len(signal_log_df)
        print(f"Total Signals Generated in the simulation period: {total_signals}")
        
        # Filter for signals on the last day of the overall data
        last_day_signals_df = signal_log_df[signal_log_df['signal_datetime'].dt.date == max_date.date()]
        
        total_last_day_signals = len(last_day_signals_df) 
        print(f"\nTotal Signals Generated on the Last Day ({max_date.date()}): {total_last_day_signals}")
        
        print("\n--- Detailed Signal List (Last Day Only) ---")
        if not last_day_signals_df.empty:
            print(last_day_signals_df.to_string()) 
        else:
            print("No signals generated on the last day with the current threshold.")
    else:
        print("No signals were generated during the simulated live period with the current threshold.")

if __name__ == "__main__":
    main()
