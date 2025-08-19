import os
import joblib
import numpy as np
import pandas as pd
import yaml
from flask import jsonify

class HeartDiseasePredictor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load_model()
    
    def _load_model(self):
        """Load model and scaler"""
        try:
            # Load heart model and data
            model_path = os.path.join(self.config['paths']['models'], "heart_rf.pkl")
            data_path = os.path.join(self.config['paths']['data_processed'], "heart_processed.joblib")
            
            self.model = joblib.load(model_path)
            heart_data = joblib.load(data_path)
            self.scaler = heart_data['scaler']
            self.feature_names = heart_data['feature_names']
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _validate_input(self, input_data):
        """Validate input data"""
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
            
        # Check all required features are present
        missing_features = set(self.feature_names) - set(input_data.keys())
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
            
        return True

    def _preprocess_input(self, input_data):
        """Preprocess input data"""
        try:
            # Create DataFrame with correct feature order
            df = pd.DataFrame([input_data], columns=self.feature_names)
            
            # Scale features
            scaled_data = self.scaler.transform(df)
            return scaled_data
            
        except Exception as e:
            raise ValueError(f"Input preprocessing failed: {str(e)}")

    def predict(self, input_data):
        """Make prediction"""
        try:
            self._validate_input(input_data)
            processed_data = self._preprocess_input(input_data)
            
            # Get prediction probabilities
            probability = self.model.predict_proba(processed_data)[0][1]
            
            return {
                'probability': float(probability),
                'prediction': int(probability > 0.5)
            }
            
        except Exception as e:
            return {'error': str(e)}

def load_config(config_path='config.yaml'):
    """Load configuration with error handling"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Config loading failed: {str(e)}")

if __name__ == "__main__":
    # Example usage
    try:
        config = load_config()
        predictor = HeartDiseasePredictor(config)
        
        # Example input (using values from heart.csv)
        example_input = {
            'age': 52,
            'sex': 1,
            'cp': 0,
            'trestbps': 125,
            'chol': 212,
            'fbs': 0,
            'restecg': 1,
            'thalach': 168,
            'exang': 0,
            'oldpeak': 1.0,
            'slope': 2,
            'ca': 2,
            'thal': 3
        }
        
        result = predictor.predict(example_input)
        print("Prediction Result:")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Prediction: {result['prediction']} (1 means heart disease likely)")
        
    except Exception as e:
        print(f"Error: {str(e)}")