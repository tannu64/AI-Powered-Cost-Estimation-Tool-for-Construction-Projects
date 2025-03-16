from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import sys
import json

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.model import CostEstimationModel

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                         'models', 'cost_estimation_model.joblib')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                'models', 'preprocessor.joblib')

# Check if model exists, otherwise use a placeholder
if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
    model = CostEstimationModel.load_model(MODEL_PATH, PREPROCESSOR_PATH)
    model_loaded = True
else:
    model = None
    model_loaded = False


@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', model_loaded=model_loaded)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 400
    
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert numeric fields
        numeric_fields = ['area_sqm', 'floors', 'has_basement', 'has_elevator', 
                         'has_parking', 'labor_rate', 'material_cost_index']
        
        for field in numeric_fields:
            if field in data:
                try:
                    data[field] = float(data[field])
                except ValueError:
                    return jsonify({
                        'error': f'Invalid value for {field}. Must be a number.'
                    }), 400
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Format prediction
        formatted_prediction = f"${prediction:,.2f}"
        
        return jsonify({
            'prediction': prediction,
            'formatted_prediction': formatted_prediction
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 400
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided'
            }), 400
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'formatted_prediction': f"${prediction:,.2f}"
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500


if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            'models'), exist_ok=True)
    
    # Run the app
    app.run(debug=True) 