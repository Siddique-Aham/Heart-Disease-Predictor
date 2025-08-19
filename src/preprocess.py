import os
import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_data(file_path):
    """Load raw data from CSV"""
    return pd.read_csv(file_path)

def clean_data(df):
    """Handle missing values and outliers"""
    df_clean = df.copy()
    
    # Fill missing values with median for numerical columns
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    return df_clean

def normalize_data(df, feature_cols):
    """Normalize features using StandardScaler"""
    scaler = StandardScaler()
    df_normalized = df.copy()
    df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_normalized, scaler

def preprocess_pipeline(input_file, output_file, target_col):
    """Complete preprocessing pipeline"""
    config = load_config()
    
    # Load and clean data
    df = load_data(input_file)
    df = clean_data(df)
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    # Normalize features
    X_normalized, scaler = normalize_data(X, feature_cols)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42
    )
    
    # Save processed data
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_cols,
        'scaler': scaler
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    joblib.dump(processed_data, output_file)
    
    return processed_data

if __name__ == "__main__":
    config = load_config()
    
    # Process heart data
    heart_raw = os.path.join(config['paths']['data_raw'], "heart.csv")
    heart_processed = os.path.join(config['paths']['data_processed'], "heart_processed.joblib")
    preprocess_pipeline(heart_raw, heart_processed, config['model_params']['heart']['target'])
    print("Heart data preprocessing completed successfully!")