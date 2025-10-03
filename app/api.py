"""
Flask REST API for Churn Prediction

Endpoints:
- POST /predict - Single customer prediction
- POST /predict/batch - Batch predictions
- GET /health - API health check
- GET /model/info - Model metadata

Usage:
    python app/api.py
    
    API will be available at http://localhost:5000
"""

from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.feature_engineering import FeatureEngineer
from src.retention_strategy import classify_risk_tier, recommend_action

app = Flask(__name__)

# Load models at startup
MODEL_PATH = Path(__file__).parent.parent / "models"
model = joblib.load(MODEL_PATH / "logistic_regression.pkl")
feature_engineer = joblib.load(MODEL_PATH / "feature_engineer.pkl")

print("✅ Models loaded successfully")
print(f"   - Model: Logistic Regression")
print(f"   - Feature Engineer: Ready")


@app.route('/', methods=['GET'])
def home():
    """API home page"""
    return jsonify({
        'service': 'Customer Churn Prediction API',
        'version': '1.0',
        'model': 'Logistic Regression',
        'endpoints': {
            'health': '/health',
            'model_info': '/model/info',
            'predict_single': '/predict (POST)',
            'predict_batch': '/predict/batch (POST)'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'Logistic Regression',
        'version': '1.0'
    })


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'Logistic Regression',
        'roc_auc': 0.846,
        'precision': 0.514,
        'recall': 0.805,
        'top_decile_precision': 0.752,
        'features': len(model.coef_[0]) if hasattr(model, 'coef_') else 'N/A'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Single customer prediction
    
    Request JSON:
    {
        "customerID": "1234-ABCD",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        customer_id = data.get('customerID', 'Unknown')
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Remove customerID and Churn if present
        X = df.drop(['customerID', 'Churn'], axis=1, errors='ignore')
        
        # Transform features
        X_transformed = feature_engineer.transform(X)
        
        # Get prediction
        churn_prob = model.predict_proba(X_transformed)[0, 1]
        churn_pred = model.predict(X_transformed)[0]
        
        # Get risk tier and action
        risk_tier = classify_risk_tier(churn_prob)
        action = recommend_action(churn_prob, data)
        
        return jsonify({
            'customerID': customer_id,
            'churn_probability': float(churn_prob),
            'churn_probability_pct': f"{float(churn_prob):.2%}",
            'churn_prediction': 'Yes' if churn_pred == 1 else 'No',
            'risk_tier': risk_tier,
            'recommended_action': action,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction
    
    Request JSON:
    {
        "customers": [
            {...customer1 data...},
            {...customer2 data...}
        ]
    }
    """
    try:
        data = request.json
        
        if not data or 'customers' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No customers data provided. Expected {"customers": [...]}'
            }), 400
        
        customers = data.get('customers', [])
        
        if not customers:
            return jsonify({'status': 'error', 'message': 'No customers provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(customers)
        customer_ids = df['customerID'].tolist() if 'customerID' in df.columns else [f'Customer_{i}' for i in range(len(df))]
        
        # Prepare features
        X = df.drop(['customerID', 'Churn'], axis=1, errors='ignore')
        X_transformed = feature_engineer.transform(X)
        
        # Get predictions
        churn_probs = model.predict_proba(X_transformed)[:, 1]
        churn_preds = model.predict(X_transformed)
        
        # Build response
        results = []
        for i, (cid, prob, pred) in enumerate(zip(customer_ids, churn_probs, churn_preds)):
            results.append({
                'customerID': cid,
                'churn_probability': float(prob),
                'churn_probability_pct': f"{float(prob):.2%}",
                'churn_prediction': 'Yes' if pred == 1 else 'No',
                'risk_tier': classify_risk_tier(prob)
            })
        
        # Sort by risk (highest first)
        results.sort(key=lambda x: x['churn_probability'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'count': len(results),
            'high_risk_count': sum(1 for r in results if r['risk_tier'] == 'High Risk'),
            'medium_risk_count': sum(1 for r in results if r['risk_tier'] == 'Medium Risk'),
            'low_risk_count': sum(1 for r in results if r['risk_tier'] == 'Low Risk'),
            'predictions': results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Customer Churn Prediction API")
    print("="*60)
    print(f"Model: Logistic Regression (ROC-AUC: 0.846)")
    print(f"API: http://localhost:5000")
    print(f"\nEndpoints:")
    print(f"  GET  /              - API info")
    print(f"  GET  /health        - Health check")
    print(f"  GET  /model/info    - Model metadata")
    print(f"  POST /predict       - Single customer prediction")
    print(f"  POST /predict/batch - Batch predictions")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
