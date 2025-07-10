import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, classification_report, accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from ta.others import DailyReturnIndicator
import warnings

# Suppress specific warnings from pandas or ta library that might clutter output
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuration - Ensure this matches your data setup
DATA_DIR = "/workspaces/Catboost/data/1_min"
TARGET_MULTIPLIER = 1.0  # From your current model
STOPLOSS_MULTIPLIER = 0.5 # From your current model
RANDOM_STATE = 42
MAX_FILES_TO_PROCESS = 5 # Limit to 5 files for tuning speed

# --- Data Preprocessing Functions (Copied from your main script for self-containment) ---

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

# --- Hyperparameter Tuning Logic ---

def main_tuning():
    """
    Main function for orchestrating data loading, splitting, and hyperparameter tuning.
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
        raise ValueError("No valid data files found in the directory for tuning.")
    
    combined_df = pd.concat(all_data)
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    
    print(f"\nTotal samples across all files: {len(combined_df)}")
    
    class_counts = combined_df['target_class'].value_counts()
    total_samples = len(combined_df)
    print("Class distribution of 'target_class':")
    print(class_counts / total_samples)

    # Calculate class weights for imbalance handling
    weight_for_0 = total_samples / (2.0 * class_counts[0])
    weight_for_1 = total_samples / (2.0 * class_counts[1])
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"Calculated class weights: {class_weights}")

    # --- Time-based Split for Training Data for Tuning ---
    # We will use the training portion of your data for tuning,
    # leaving the final test set completely untouched.
    min_date = combined_df['datetime'].min()
    max_date = combined_df['datetime'].max()

    print(f"\nData covers period from {min_date} to {max_date}")

    train_duration_months = 6
    test_duration_months = 1 # This is the duration for the *final* test set, not for tuning's internal CV

    # Calculate the start and end dates for the training data (for tuning)
    # This will be the same training set as in your main script
    temp_test_end_date = max_date
    temp_test_start_date = max_date - pd.DateOffset(months=test_duration_months)
    
    if temp_test_start_date < min_date:
        temp_test_start_date = min_date
        print(f"Warning: Not enough data for a full {test_duration_months}-month test period. Test start date adjusted to {temp_test_start_date}.")

    train_end_date_for_tuning = temp_test_start_date
    train_start_date_for_tuning = train_end_date_for_tuning - pd.DateOffset(months=train_duration_months)
    
    if train_start_date_for_tuning < min_date:
        train_start_date_for_tuning = min_date
        print(f"Warning: Not enough data for a full {train_duration_months}-month training period. Train start date adjusted to {train_start_date_for_tuning}.")

    # Filter data for tuning (this is your X_train, y_train from the main script)
    tuning_df = combined_df[(combined_df['datetime'] >= train_start_date_for_tuning) & (combined_df['datetime'] < train_end_date_for_tuning)]

    if tuning_df.empty:
        raise ValueError("Not enough data to form a tuning set. Please ensure your combined data has sufficient history for the training period.")
    
    features = [col for col in combined_df.columns if col not in ['datetime', 'target_class', 'target']]
    X_tuning = tuning_df[features]
    y_tuning = tuning_df['target_class']

    print(f"\nData for tuning covers period from {tuning_df['datetime'].min()} to {tuning_df['datetime'].max()} ({len(tuning_df)} samples)")
    print(f"Tuning data class distribution:\n{y_tuning.value_counts(normalize=True)}")

    # --- Hyperparameter Search Space ---
    param_distributions = {
        'iterations': [100, 250, 500, 750, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9], # L2 regularization coefficient
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0], # Percentage of samples to use for training each tree
        'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0], # Percentage of features to use at each split
        'min_data_in_leaf': [1, 5, 10, 20], # Minimum number of samples in a leaf
    }

    # Initialize CatBoostClassifier with fixed parameters and class weights
    # These will be the base parameters for the search
    base_model = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='F1', # Using F1 for evaluation during tuning
        random_seed=RANDOM_STATE,
        early_stopping_rounds=50,
        verbose=0, # Suppress verbose output during tuning
        class_weights=class_weights # Apply class weights during tuning
    )

    # Define the scoring metric for RandomizedSearchCV
    # We want to optimize for F1-score of the positive class (Class 1)
    f1_scorer = make_scorer(f1_score, pos_label=1)

    # Setup RandomizedSearchCV
    # n_iter: Number of parameter settings that are sampled.
    # cv: Cross-validation strategy. StratifiedKFold is good for imbalanced data.
    # scoring: The metric to optimize.
    # n_jobs: Number of jobs to run in parallel (-1 means use all available processors).
    print("\nStarting Randomized Search for Hyperparameter Tuning...")
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=50, # Number of random combinations to try. Adjust based on computational resources.
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE), # 3-fold stratified CV
        scoring=f1_scorer,
        verbose=2, # More verbose output during search
        random_state=RANDOM_STATE,
        n_jobs=-1 # Use all available cores
    )

    random_search.fit(X_tuning, y_tuning)

    print("\nHyperparameter Tuning Complete.")
    print(f"Best F1-score for Class 1: {random_search.best_score_:.4f}")
    print("Best parameters found:")
    print(random_search.best_params_)

    # You can now use random_search.best_estimator_ to get the best model
    # or retrain a new model with random_search.best_params_ on your full training data.
    print("\nRetraining final model with best parameters on the full training data...")
    final_model_params = random_search.best_params_
    final_model_params['loss_function'] = 'Logloss'
    final_model_params['eval_metric'] = 'Accuracy' # Use Accuracy for final eval if preferred, or F1
    final_model_params['random_seed'] = RANDOM_STATE
    final_model_params['early_stopping_rounds'] = 50
    final_model_params['verbose'] = 100
    final_model_params['class_weights'] = class_weights

    final_model = CatBoostClassifier(**final_model_params)
    
    # Load the full training and test data again for final model evaluation
    # This part assumes you have enough data for a proper 6-month train, 1-month test split
    # If not, the warnings from load_and_preprocess_data will apply.
    
    # Re-load all data to ensure the final train/test split is consistent with the main script
    reloaded_all_data = []
    reloaded_processed_files_count = 0
    for file in os.listdir(DATA_DIR):
        if file.endswith('.csv'):
            if reloaded_processed_files_count >= MAX_FILES_TO_PROCESS:
                break
            file_path = os.path.join(DATA_DIR, file)
            df = load_and_preprocess_data(file_path)
            if df is not None:
                reloaded_all_data.append(df)
                reloaded_processed_files_count += 1
    
    if not reloaded_all_data:
        raise ValueError("No valid data files found for final model training.")
    
    reloaded_combined_df = pd.concat(reloaded_all_data)
    reloaded_combined_df = reloaded_combined_df.sort_values('datetime').reset_index(drop=True)

    min_date = reloaded_combined_df['datetime'].min()
    max_date = reloaded_combined_df['datetime'].max()

    test_end_date = max_date
    test_start_date = max_date - pd.DateOffset(months=test_duration_months)
    if test_start_date < min_date:
        test_start_date = min_date

    train_end_date = test_start_date
    train_start_date = train_end_date - pd.DateOffset(months=train_duration_months)
    if train_start_date < min_date:
        train_start_date = min_date

    final_train_df = reloaded_combined_df[(reloaded_combined_df['datetime'] >= train_start_date) & (reloaded_combined_df['datetime'] < train_end_date)]
    final_test_df = reloaded_combined_df[(reloaded_combined_df['datetime'] >= test_start_date) & (reloaded_combined_df['datetime'] <= test_end_date)]

    final_features = [col for col in reloaded_combined_df.columns if col not in ['datetime', 'target_class', 'target']]
    X_final_train = final_train_df[final_features]
    y_final_train = final_train_df['target_class']
    X_final_test = final_test_df[final_features]
    y_final_test = final_test_df['target_class']

    if y_final_test.empty:
        raise ValueError("Final test set labels (y_final_test) are empty. Cannot evaluate final model.")

    final_model.fit(
        X_final_train, y_final_train,
        eval_set=(X_final_test, y_final_test),
        use_best_model=True
    )

    print("\n--- Final Model Evaluation (on Test Set) ---")
    final_test_pred = final_model.predict(X_final_test)
    print(classification_report(y_final_test, final_test_pred))
    print(f"Final Test Accuracy: {accuracy_score(y_final_test, final_test_pred):.4f}")

    # Save the final tuned model
    final_model.save_model("catboost_tuned_stock_classifier.cbm")
    print(f"\nFinal Tuned Model saved to catboost_tuned_stock_classifier.cbm")

if __name__ == "__main__":
    main_tuning()
