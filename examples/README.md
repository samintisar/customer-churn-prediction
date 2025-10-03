# Examples - Using the Churn Prediction Model

This folder contains practical examples demonstrating how to use the trained Logistic Regression model for churn prediction.

## 📁 Files

### `simple_prediction.py`
Basic examples showing how to:
- Load the trained model
- Make single customer predictions
- Process batch predictions
- Interpret results and risk tiers

## 🚀 Quick Start

### Run the Simple Example

```bash
python examples/simple_prediction.py
```

This will show you:
1. **High-Risk Customer** - New customer with month-to-month contract
2. **Low-Risk Customer** - Loyal customer with 2-year contract
3. **Batch Predictions** - Process multiple customers at once

## 📊 Example Output

```
Customer ID:          DEMO-0001
Churn Probability:    78.45%
Prediction:           Yes
Risk Tier:            High Risk
Recommended Action:   Immediate retention outreach recommended
```

## 💡 Usage in Your Code

### Basic Prediction

```python
import joblib
import pandas as pd
from src.feature_engineering import FeatureEngineer

# Load models
model = joblib.load('models/logistic_regression.pkl')
feature_engineer = joblib.load('models/feature_engineer.pkl')

# Prepare customer data
customer = {
    'customerID': '1234-ABCD',
    'gender': 'Female',
    'tenure': 1,
    'Contract': 'Month-to-month',
    'MonthlyCharges': 70.35,
    # ... other features
}

# Convert to DataFrame and transform
df = pd.DataFrame([customer])
X = df.drop(['customerID', 'Churn'], axis=1, errors='ignore')
X_transformed = feature_engineer.transform(X)

# Predict
churn_prob = model.predict_proba(X_transformed)[0, 1]
print(f"Churn Probability: {churn_prob:.2%}")
```

### Batch Prediction

```python
# Load customer data
customers = pd.read_csv('data/customers.csv')

# Prepare and transform
X = customers.drop(['customerID', 'Churn'], axis=1, errors='ignore')
X_transformed = feature_engineer.transform(X)

# Predict for all
churn_probs = model.predict_proba(X_transformed)[:, 1]

# Add to dataframe
customers['ChurnProbability'] = churn_probs
customers['RiskTier'] = ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' 
                         for p in churn_probs]

# Get high-risk customers
high_risk = customers[customers['RiskTier'] == 'High'].sort_values(
    'ChurnProbability', ascending=False
)
```

## 🎯 Risk Tiers

The model classifies customers into three risk tiers:

| Risk Tier | Probability Range | Action |
|-----------|------------------|---------|
| **High Risk** | > 70% | Immediate retention outreach |
| **Medium Risk** | 40-70% | Proactive engagement |
| **Low Risk** | < 40% | Standard service |

## 📚 Next Steps

1. **For Development**: Modify `simple_prediction.py` with your own customer data
2. **For REST API**: Use `app/api.py` for production deployments
3. **For Dashboard**: Use `app/dashboard.py` for interactive exploration
4. **For Batch Jobs**: Use `scripts/batch_predict.py` for regular scoring

## 📖 Documentation

- See `DEPLOYMENT_GUIDE.md` for comprehensive deployment options
- See `README.md` for project overview and setup
- See `reports/business_report.md` for business insights

## 🔧 Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root directory
cd /path/to/customer-churn-prediction

# Run from root
python examples/simple_prediction.py
```

### Model Not Found
```bash
# Train models first
jupyter notebook notebooks/02_model_experiments.ipynb

# Or check if models exist
ls models/
```

### Feature Mismatch
Ensure your customer data has all required features:
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Services: PhoneService, InternetService, OnlineSecurity, etc.
- Account: tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges

## 💬 Need Help?

Check the deployment guide for more detailed instructions:
```bash
cat DEPLOYMENT_GUIDE.md
```
