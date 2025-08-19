import os
import yaml
import joblib
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Suppress warnings
warnings.filterwarnings("ignore")

def load_config():
    """Load configuration from YAML file"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def train_heart_model(data_path):
    """Train random forest model for heart disease prediction"""
    data = joblib.load(data_path)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data['X_train'], data['y_train'])
    
    # Evaluate
    y_pred = model.predict(data['X_test'])
    acc = accuracy_score(data['y_test'], y_pred)
    roc_auc = roc_auc_score(data['y_test'], y_pred)
    
    print("\nHeart Disease Model Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(data['y_test'], y_pred))
    
    return model

def save_model(model, model_path):
    """Save trained model to disk"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

if __name__ == "__main__":
    config = load_config()
    
    # Train and save heart model
    heart_data = os.path.join(config['paths']['data_processed'], "heart_processed.joblib")
    heart_model = train_heart_model(heart_data)
    save_model(heart_model, os.path.join(config['paths']['models'], "heart_rf.pkl"))
    print("\nHeart disease model trained and saved successfully!")