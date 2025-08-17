"""
Flask API for churn prediction service.
Production-ready with error handling and logging.
"""
from flask import Flask, request, jsonify
import pandas as pd
from pathlib import Path
import logging
from functools import wraps
import time

from prediction_service import ChurnPredictionService
from logger_config import setup_logger

# Initialize Flask app
app = Flask(__name__)
logger = setup_logger(__name__)

# Initialize prediction service
MODEL_PATH = Path("models/model_production")
prediction_service = ChurnPredictionService(model_path=MODEL_PATH)


def validate_request(f):
    """Decorator to validate API requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        return f(*args, **kwargs)
    return decorated_function


def log_prediction(f):
    """Decorator to log predictions"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        duration = time.time() - start_time
        
        logger.info("Prediction made", extra={
            'endpoint': request.endpoint,
            'duration_ms': duration * 1000,
            'status_code': result[1] if isinstance(result, tuple) else 200
        })
        
        return result
    return decorated_function


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_info = prediction_service.pipeline.get_model_info()
    return jsonify({
        'status': 'healthy',
        'model_version': model_info['version'],
        'model_loaded': model_info['is_trained']
    })


@app.route('/predict', methods=['POST'])
@validate_request
@log_prediction
def predict():
    """Single prediction endpoint"""
    try:
        customer_data = request.json
        
        # Validate required fields
        required_fields = ['customerID', 'tenure', 'MonthlyCharges', 'Contract']
        missing_fields = [f for f in required_fields if f not in customer_data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Make prediction
        result = prediction_service.predict_single(customer_data)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/predict_batch', methods=['POST'])
@validate_request
@log_prediction
def predict_batch():
    """Batch prediction endpoint"""
    try:
        data = request.json
        
        if 'customers' not in data:
            return jsonify({'error': 'Missing customers field'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['customers'])
        
        # Make predictions
        results = prediction_service.predict_batch(df)
        
        # Convert to JSON-friendly format
        response = {
            'predictions': results.to_dict('records'),
            'summary': {
                'total_customers': len(results),
                'high_risk_count': len(results[results['risk_segment'] == 'Very High']),
                'total_revenue_at_risk': float(results['annual_revenue_at_risk'].sum())
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/update_threshold', methods=['POST'])
@validate_request
def update_threshold():
    """Update business threshold endpoint"""
    try:
        data = request.json
        
        if 'threshold' not in data:
            return jsonify({'error': 'Missing threshold field'}), 400
        
        new_threshold = float(data['threshold'])
        
        if not 0 <= new_threshold <= 1:
            return jsonify({'error': 'Threshold must be between 0 and 1'}), 400
        
        prediction_service.update_threshold(new_threshold)
        
        return jsonify({
            'status': 'success',
            'new_threshold': new_threshold
        })
        
    except Exception as e:
        logger.error(f"Threshold update error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Don't use this in production - use gunicorn instead
    app.run(debug=False, host='0.0.0.0', port=5000)