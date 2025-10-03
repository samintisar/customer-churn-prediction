# 🚀 Deployment Guide - Logistic Regression Model

This guide covers multiple deployment options for the Customer Churn Prediction model (Logistic Regression).

## 📋 Table of Contents

1. [Quick Start - Use the Existing Model](#1-quick-start---use-the-existing-model)
2. [Deploy with Streamlit Dashboard (Local)](#2-deploy-with-streamlit-dashboard-local)
3. [Deploy as REST API with Flask](#3-deploy-as-rest-api-with-flask)
4. [Deploy to Production (Cloud)](#4-deploy-to-production-cloud)
5. [Batch Prediction Script](#5-batch-prediction-script)
6. [Model Artifacts & Files](#6-model-artifacts--files)

---

## 1. Quick Start - Use the Existing Model

The Logistic Regression model is already trained and saved! Here's how to use it:

### Load and Use the Model

```python
import pandas as pd
import joblib
from src.feature_engineering import FeatureEngineer

# Load the saved models
logistic_model = joblib.load('models/logistic_regression.pkl')
feature_engineer = joblib.load('models/feature_engineer.pkl')

# Load your customer data
customers = pd.read_csv('data/raw/Telco-Customer-Churn.csv')

# Prepare features (remove customerID and target if present)
X = customers.drop(['customerID', 'Churn'], axis=1, errors='ignore')

# Transform features using the saved feature engineer
X_transformed = feature_engineer.transform(X)

# Get predictions
churn_probabilities = logistic_model.predict_proba(X_transformed)[:, 1]
churn_predictions = logistic_model.predict(X_transformed)

# Add predictions to dataframe
customers['ChurnProbability'] = churn_probabilities
customers['ChurnPrediction'] = ['Yes' if pred == 1 else 'No' for pred in churn_predictions]

# Sort by risk (highest first)
at_risk = customers.sort_values('ChurnProbability', ascending=False)
print(at_risk[['customerID', 'ChurnProbability', 'ChurnPrediction']].head(20))
```

---

## 2. Deploy with Streamlit Dashboard (Local)

**Best for:** Interactive exploration, demos, internal stakeholder use

### Step 1: Ensure Model is Active

The dashboard automatically loads `models/churn_predictor.pkl`. To use Logistic Regression:

```bash
# Make sure logistic_regression.pkl is copied as churn_predictor.pkl
cd models
cp logistic_regression.pkl churn_predictor.pkl  # Mac/Linux
# OR
copy logistic_regression.pkl churn_predictor.pkl  # Windows
```

### Step 2: Launch Dashboard

```bash
# Activate your environment
conda activate customer-churn

# Run the dashboard
streamlit run app/dashboard.py
```

### Step 3: Access Dashboard

Open browser to: `http://localhost:8501`

**Features:**
- 📊 Overview metrics and risk distribution
- 🚨 At-risk customer list with filters
- 📈 Model performance metrics
- 🔍 Individual customer predictions
- 📥 Download reports as CSV

---

## 3. Deploy as REST API with Flask

**Best for:** Integration with other applications, automated workflows

### Step 1: Create Flask API

Create `app/api.py`:

```python
"""
Flask REST API for Churn Prediction

Endpoints:
- POST /predict - Single customer prediction
- POST /predict/batch - Batch predictions
- GET /health - API health check
- GET /model/info - Model metadata
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
        'features': feature_engineer.feature_names_ if hasattr(feature_engineer, 'feature_names_') else 'N/A'
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
        "InternetService": "Fiber optic",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20,
        ...
    }
    """
    try:
        data = request.json
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
                'churn_prediction': 'Yes' if pred == 1 else 'No',
                'risk_tier': classify_risk_tier(prob)
            })
        
        return jsonify({
            'status': 'success',
            'count': len(results),
            'predictions': results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### Step 2: Install Flask

```bash
pip install flask
```

### Step 3: Run the API

```bash
python app/api.py
```

API will be available at: `http://localhost:5000`

### Step 4: Test the API

**Single Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Python Client Example:**
```python
import requests

url = "http://localhost:5000/predict"
customer_data = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "tenure": 1,
    "Contract": "Month-to-month",
    "MonthlyCharges": 29.85,
    # ... other features
}

response = requests.post(url, json=customer_data)
result = response.json()

print(f"Churn Probability: {result['churn_probability']:.2%}")
print(f"Risk Tier: {result['risk_tier']}")
print(f"Action: {result['recommended_action']}")
```

---

## 4. Deploy to Production (Cloud)

### Option A: Deploy to Streamlit Cloud (Free)

**Best for:** Quick deployment, demos, MVPs

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Deploy churn prediction app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repo: `customer-churn-prediction`
   - Main file: `app/dashboard.py`
   - Click "Deploy"

3. **Access your app:**
   - URL: `https://yourapp.streamlit.app`

### Option B: Deploy API to Heroku

**Best for:** Production REST APIs

1. **Create `Procfile`:**
   ```
   web: gunicorn app.api:app
   ```

2. **Install Gunicorn:**
   ```bash
   pip install gunicorn
   pip freeze > requirements.txt
   ```

3. **Deploy to Heroku:**
   ```bash
   heroku login
   heroku create your-churn-api
   git push heroku main
   ```

### Option C: Deploy to AWS/Azure/GCP

**Best for:** Enterprise production

1. **Containerize with Docker:**
   
   Create `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 5000
   
   CMD ["python", "app/api.py"]
   ```

2. **Build and Test:**
   ```bash
   docker build -t churn-predictor .
   docker run -p 5000:5000 churn-predictor
   ```

3. **Deploy to Cloud:**
   - **AWS ECS/Fargate**: Push to ECR and deploy
   - **Azure Container Apps**: Push to ACR and deploy
   - **GCP Cloud Run**: Push to GCR and deploy

---

## 5. Batch Prediction Script

**Best for:** Regular batch scoring of customer base

Create `scripts/batch_predict.py`:

```python
"""
Batch Churn Prediction Script

Usage:
    python scripts/batch_predict.py --input data/customers.csv --output results/predictions.csv
"""

import argparse
import pandas as pd
import joblib
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from src.feature_engineering import FeatureEngineer
from src.retention_strategy import classify_risk_tier, recommend_action

def batch_predict(input_file: str, output_file: str):
    """Run batch predictions on customer data"""
    
    print(f"🚀 Starting batch prediction...")
    print(f"Input: {input_file}")
    
    # Load models
    print("Loading models...")
    model = joblib.load('models/logistic_regression.pkl')
    feature_engineer = joblib.load('models/feature_engineer.pkl')
    
    # Load data
    print("Loading customer data...")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} customers")
    
    # Store customer IDs
    if 'customerID' in df.columns:
        customer_ids = df['customerID']
    else:
        customer_ids = [f'Customer_{i}' for i in range(len(df))]
    
    # Prepare features
    X = df.drop(['customerID', 'Churn'], axis=1, errors='ignore')
    X_transformed = feature_engineer.transform(X)
    
    # Get predictions
    print("Generating predictions...")
    churn_probs = model.predict_proba(X_transformed)[:, 1]
    churn_preds = model.predict(X_transformed)
    
    # Create results dataframe
    results = pd.DataFrame({
        'customerID': customer_ids,
        'churn_probability': churn_probs,
        'churn_prediction': ['Yes' if pred == 1 else 'No' for pred in churn_preds],
        'risk_tier': [classify_risk_tier(prob) for prob in churn_probs]
    })
    
    # Add recommended actions
    results['recommended_action'] = results.apply(
        lambda row: recommend_action(row['churn_probability'], {}), 
        axis=1
    )
    
    # Sort by risk
    results = results.sort_values('churn_probability', ascending=False)
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n✅ Predictions complete!")
    print(f"  Output: {output_file}")
    print(f"\n📊 Summary:")
    print(f"  Total customers: {len(results)}")
    print(f"  Predicted churners: {(results['churn_prediction'] == 'Yes').sum()}")
    print(f"  Churn rate: {(results['churn_prediction'] == 'Yes').mean():.2%}")
    print(f"\n🎯 Risk Distribution:")
    print(results['risk_tier'].value_counts())
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch churn prediction')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    batch_predict(args.input, args.output)
```

**Usage:**
```bash
python scripts/batch_predict.py \
  --input data/raw/Telco-Customer-Churn.csv \
  --output results/churn_predictions_$(date +%Y%m%d).csv
```

---

## 6. Model Artifacts & Files

Your trained models are stored in the `models/` directory:

```
models/
├── logistic_regression.pkl      # ⭐ Recommended production model
├── random_forest.pkl             # Alternative model
├── churn_predictor.pkl           # Currently active model (used by dashboard)
└── feature_engineer.pkl          # Feature transformation pipeline
```

### Switch to Logistic Regression

If `churn_predictor.pkl` is not the Logistic Regression model:

```bash
cd models
cp logistic_regression.pkl churn_predictor.pkl
```

Or in Python:
```python
import shutil
shutil.copy('models/logistic_regression.pkl', 'models/churn_predictor.pkl')
```

---

## 📋 Deployment Checklist

Before deploying to production:

- [ ] Test model on validation data
- [ ] Verify feature engineering pipeline works correctly
- [ ] Set up monitoring for prediction latency
- [ ] Implement logging for all predictions
- [ ] Add error handling and fallback logic
- [ ] Set up model retraining pipeline
- [ ] Document API endpoints (if applicable)
- [ ] Add authentication/authorization (for production APIs)
- [ ] Configure rate limiting
- [ ] Set up alerting for model drift
- [ ] Plan for model versioning
- [ ] Create rollback procedure

---

## 🔧 Troubleshooting

### Model File Not Found
```python
# Verify model exists
from pathlib import Path
model_path = Path('models/logistic_regression.pkl')
print(f"Model exists: {model_path.exists()}")
```

### Feature Mismatch
```python
# Check expected features
feature_engineer = joblib.load('models/feature_engineer.pkl')
print(f"Expected features: {feature_engineer.feature_names_}")
```

### Version Incompatibility
```bash
# Check scikit-learn version
python -c "import sklearn; print(sklearn.__version__)"

# If needed, match the version used during training
pip install scikit-learn==1.3.0  # Adjust version as needed
```

---

## 📞 Support

For deployment issues:
1. Check logs for error messages
2. Verify all dependencies are installed
3. Ensure model files are not corrupted
4. Review the [README.md](README.md) for setup instructions

---

## 🎯 Next Steps

After deployment:
1. Monitor model performance in production
2. Collect feedback from users
3. Track prediction accuracy over time
4. Plan for model retraining (quarterly/annually)
5. Implement A/B testing for model improvements

Good luck with your deployment! 🚀
