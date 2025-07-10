import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
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


# Configuration
DATA_DIR = "/workspaces/Catboost/data/1_min"
TARGET_MULTIPLIER = 1.0  # Changed to 1 ATR for target
STOPLOSS_MULTIPLIER = 0.5 # Changed to 0.5 ATR for stoploss
RANDOM_STATE = 42
MODEL_SAVE_PATH = "catboost_stock_classifier.cbm"
MAX_FILES_TO_PROCESS = 5 # Limit to 5 files for faster execution/debugging

# --- NEW: Best Hyperparameters from Tuning ---
# Replace these with the actual best parameters found by your catboost_tuner.py script
BEST_PARAMS = {
    'subsample': 0.9,
    'min_data_in_leaf': 1,
    'learning_rate': 0.05,
    'l2_leaf_reg': 9,
    'iterations': 1000,
    'depth': 6,
    'colsample_bylevel': 0.6
}

# --- NEW: Custom Prediction Threshold ---
# Adjust this value to balance precision and recall for Class 1
# A higher threshold increases precision (fewer false positives) but may decrease recall.
# A lower threshold increases recall (more true positives) but may decrease precision.
CUSTOM_PREDICTION_THRESHOLD = 0.7 # Start with 0.5, then try 0.6, 0.7, etc.


def calculate_features(df):
    """
    Calculate all required technical indicators and custom features for the given DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.
        
    Returns:
        pd.DataFrame: DataFrame with calculated technical and custom features, with initial NaN rows dropped.
    """
    # Ensure columns are numeric for calculations, coercing errors will turn non-numeric into NaN
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

    # Custom Features for Early Momentum Finding
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
    Optimized to use NumPy arrays for faster processing of future price checks.
    
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

    target_level = entry_price + (TARGET_MULTIPLIER * atr)
    stop_level = entry_price - (STOPLOSS_MULTIPLIER * atr)
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

def load_and_preprocess_data(file_path):
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
        
        return df[['datetime'] + features + ['target_class']]
    except Exception as e:
        print(f"Error loading or preprocessing {file_path}: {e}")
        return None

def train_model(X_train, y_train, X_test, y_test, class_weights=None, best_params=None):
    """
    Train and evaluate a CatBoost classifier.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        class_weights (dict, optional): Dictionary of class weights. Defaults to None.
        best_params (dict, optional): Dictionary of best hyperparameters from tuning. Defaults to None.
        
    Returns:
        CatBoostClassifier: Trained CatBoost model.
    """
    # Use best_params if provided, otherwise use default parameters
    model_params = best_params if best_params else {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
    }
    
    model = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='Accuracy',
        random_seed=RANDOM_STATE,
        early_stopping_rounds=50,
        verbose=100,
        class_weights=class_weights,
        **model_params # Unpack the best_params dictionary here
    )
    
    print("Starting CatBoost model training...")
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )
    print("Model training complete.")
    
    return model

def main():
    """
    Main function to orchestrate data loading, preprocessing, model training,
    evaluation, and saving.
    """
    all_data = []
    processed_files_count = 0
    
    print(f"Loading data from: {DATA_DIR}")
    for file in os.listdir(DATA_DIR):
        if file.endswith('.csv'):
            if processed_files_count >= MAX_FILES_TO_PROCESS:
                print(f"Reached limit of {MAX_FILES_TO_PROCESS} files. Skipping remaining files.")
                break
            
            file_path = os.path.join(DATA_DIR, file)
            df = load_and_preprocess_data(file_path)
            if df is not None:
                all_data.append(df)
                processed_files_count += 1
                print(f"Processed {file} with {len(df)} samples")
    
    if not all_data:
        raise ValueError("No valid data files found in the directory. Please check DATA_DIR and file formats.")
    
    combined_df = pd.concat(all_data)
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    
    print(f"\nTotal samples across all files: {len(combined_df)}")
    
    class_counts = combined_df['target_class'].value_counts()
    total_samples = len(combined_df)
    print("Class distribution of 'target_class':")
    print(class_counts / total_samples)

    weight_for_0 = total_samples / (2.0 * class_counts[0])
    weight_for_1 = total_samples / (2.0 * class_counts[1])
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"Calculated class weights: {class_weights}")

    min_date = combined_df['datetime'].min()
    max_date = combined_df['datetime'].max()

    print(f"\nData covers period from {min_date} to {max_date}")

    train_duration_months = 6
    test_duration_months = 1

    test_end_date = max_date
    test_start_date = max_date - pd.DateOffset(months=test_duration_months)
    
    if test_start_date < min_date:
        test_start_date = min_date
        print(f"Warning: Not enough data for a full {test_duration_months}-month test period. Test start date adjusted to {test_start_date}.")

    train_end_date = test_start_date
    train_start_date = train_end_date - pd.DateOffset(months=train_duration_months)
    
    if train_start_date < min_date:
        train_start_date = min_date
        print(f"Warning: Not enough data for a full {train_duration_months}-month training period. Train start date adjusted to {train_start_date}.")

    print(f"\nCalculated split dates:")
    print(f"  Train Start: {train_start_date}")
    print(f"  Train End:   {train_end_date}")
    print(f"  Test Start:  {test_start_date}")
    print(f"  Test End:    {test_end_date}")

    train_df = combined_df[(combined_df['datetime'] >= train_start_date) & (combined_df['datetime'] < train_end_date)]
    test_df = combined_df[(combined_df['datetime'] >= test_start_date) & (combined_df['datetime'] <= test_end_date)]

    print(f"\nDataFrame lengths after splitting:")
    print(f"  train_df length: {len(train_df)}")
    print(f"  test_df length: {len(test_df)}")

    if train_df.empty:
        raise ValueError("Not enough data to form a training set. Please ensure your combined data has sufficient history.")
    
    features = [col for col in combined_df.columns if col not in ['datetime', 'target_class', 'target']]
    
    X_train = train_df[features]
    y_train = train_df['target_class']
    
    X_test = test_df[features]
    y_test = test_df['target_class']

    if y_test.empty:
        raise ValueError(
            "Test set labels (y_test) are empty. Cannot proceed with model training and evaluation. "
            "This typically means there's no data in the calculated test date range. "
            "Please ensure your combined data spans at least 1 month for the test period. "
            "Current data range: {} to {}. Calculated test range: {} to {}."
            .format(min_date, max_date, test_start_date, test_end_date)
        )

    # Train the CatBoost model using the best parameters found by tuning
    model = train_model(X_train, y_train, X_test, y_test, class_weights=class_weights, best_params=BEST_PARAMS)
    
    # --- START OF THRESHOLD ADJUSTMENT ---
    # Get prediction probabilities for the test set
    test_proba = model.predict_proba(X_test)
    # The probabilities are returned as an array where test_proba[:, 1] is the probability of Class 1
    
    # Apply the custom threshold to get binary predictions
    test_pred_adjusted = (test_proba[:, 1] >= CUSTOM_PREDICTION_THRESHOLD).astype(int)
    
    # Evaluate with adjusted predictions
    print(f"\n--- Test Metrics (Adjusted Threshold: {CUSTOM_PREDICTION_THRESHOLD}) ---")
    print(classification_report(y_test, test_pred_adjusted))
    print(f"Test Accuracy (Adjusted Threshold): {accuracy_score(y_test, test_pred_adjusted):.4f}")
    # --- END OF THRESHOLD ADJUSTMENT ---

    # Evaluate training predictions (no threshold adjustment needed here, as it's for training performance)
    train_pred = model.predict(X_train)
    print("\n--- Training Metrics (Default Threshold) ---")
    print(classification_report(y_train, train_pred))
    print(f"Training Accuracy (Default Threshold): {accuracy_score(y_train, train_pred):.4f}")
        
    # Save the trained model
    model.save_model(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    
    # Display feature importances
    print("\n--- Feature Importances ---")
    feature_importances = model.get_feature_importance()
    for score, name in sorted(zip(feature_importances, X_train.columns), reverse=True):
        print(f"{name}: {score:.4f}")

if __name__ == "__main__":
    main()