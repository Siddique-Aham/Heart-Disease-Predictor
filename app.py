from flask import Flask, render_template, request, jsonify
import yaml
from predict import HeartDiseasePredictor
import os

app = Flask(__name__)

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

# Initialize predictor
config = load_config()
predictor = HeartDiseasePredictor(config)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()
        
        # Convert numeric fields
        numeric_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                         'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        for field in numeric_fields:
            data[field] = float(data[field])
        
        # Make prediction
        result = predictor.predict(data)
        
        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']})
        
        # Prepare response
        response = {
            'success': True,
            'probability': round(result['probability'] * 100, 2),
            'prediction': result['prediction']
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True)