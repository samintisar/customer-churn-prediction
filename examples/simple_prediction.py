"""
Simple Example: Using the Logistic Regression Model

This script demonstrates how to load and use the trained model
for making predictions on new customers.

Usage:
    python examples/simple_prediction.py
"""

import pandas as pd
import joblib
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.feature_engineering import FeatureEngineer
from src.retention_strategy import classify_risk_tier, recommend_action


def load_models():
    """Load the trained model and feature engineer"""
    print("Loading models...")
    model = joblib.load('models/logistic_regression.pkl')
    feature_engineer = joblib.load('models/feature_engineer.pkl')
    print("✅ Models loaded successfully\n")
    return model, feature_engineer


def predict_single_customer(model, feature_engineer, customer_data):
    """
    Predict churn for a single customer
    
    Args:
        model: Trained logistic regression model
        feature_engineer: Fitted feature engineering pipeline
        customer_data: Dictionary with customer information
    
    Returns:
        Dictionary with prediction results
    """
    # Convert to DataFrame
    df = pd.DataFrame([customer_data])
    
    # Remove customerID if present
    customer_id = customer_data.get('customerID', 'Unknown')
    X = df.drop(['customerID', 'Churn'], axis=1, errors='ignore')
    
    # Transform features
    X_transformed = feature_engineer.transform(X)
    
    # Get prediction
    churn_prob = model.predict_proba(X_transformed)[0, 1]
    churn_pred = model.predict(X_transformed)[0]
    
    # Get risk tier and recommendation
    risk_tier = classify_risk_tier(churn_prob)
    action = recommend_action(churn_prob, customer_data)
    
    return {
        'customerID': customer_id,
        'churn_probability': churn_prob,
        'churn_prediction': 'Yes' if churn_pred == 1 else 'No',
        'risk_tier': risk_tier,
        'recommended_action': action
    }


def main():
    print("="*70)
    print("🎯 CUSTOMER CHURN PREDICTION - SIMPLE EXAMPLE")
    print("="*70 + "\n")
    
    # Load models
    model, feature_engineer = load_models()
    
    # Example 1: High-risk customer (month-to-month, new customer, high charges)
    print("📊 Example 1: High-Risk Customer Profile")
    print("-"*70)
    
    customer_high_risk = {
        'customerID': 'DEMO-0001',
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 1,  # New customer
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',  # High churn indicator
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',  # High churn indicator
        'MonthlyCharges': 89.95,  # High charges
        'TotalCharges': 89.95
    }
    
    result1 = predict_single_customer(model, feature_engineer, customer_high_risk)
    
    print(f"Customer ID:          {result1['customerID']}")
    print(f"Churn Probability:    {result1['churn_probability']:.2%}")
    print(f"Prediction:           {result1['churn_prediction']}")
    print(f"Risk Tier:            {result1['risk_tier']}")
    print(f"Recommended Action:   {result1['recommended_action']}")
    
    # Example 2: Low-risk customer (long-term contract, loyal)
    print("\n" + "="*70)
    print("📊 Example 2: Low-Risk Customer Profile")
    print("-"*70)
    
    customer_low_risk = {
        'customerID': 'DEMO-0002',
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 48,  # Loyal customer
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Two year',  # Low churn indicator
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Bank transfer (automatic)',
        'MonthlyCharges': 99.95,
        'TotalCharges': 4797.60
    }
    
    result2 = predict_single_customer(model, feature_engineer, customer_low_risk)
    
    print(f"Customer ID:          {result2['customerID']}")
    print(f"Churn Probability:    {result2['churn_probability']:.2%}")
    print(f"Prediction:           {result2['churn_prediction']}")
    print(f"Risk Tier:            {result2['risk_tier']}")
    print(f"Recommended Action:   {result2['recommended_action']}")
    
    # Example 3: Batch prediction from CSV
    print("\n" + "="*70)
    print("📊 Example 3: Batch Prediction from CSV")
    print("-"*70)
    
    # Load sample customers
    try:
        customers = pd.read_csv('data/raw/Telco-Customer-Churn.csv')
        
        # Take first 5 customers
        sample = customers.head(5).copy()
        
        # Prepare features
        X = sample.drop(['customerID', 'Churn'], axis=1, errors='ignore')
        X_transformed = feature_engineer.transform(X)
        
        # Get predictions
        churn_probs = model.predict_proba(X_transformed)[:, 1]
        
        # Add to dataframe
        sample['ChurnProbability'] = churn_probs
        sample['PredictedChurn'] = ['Yes' if p > 0.5 else 'No' for p in churn_probs]
        sample['RiskTier'] = [classify_risk_tier(p) for p in churn_probs]
        
        # Display results
        display_cols = ['customerID', 'tenure', 'Contract', 'MonthlyCharges', 
                       'ChurnProbability', 'PredictedChurn', 'RiskTier']
        
        print(sample[display_cols].to_string(index=False))
        
    except FileNotFoundError:
        print("⚠️  Sample data file not found")
        print("   Please ensure data/raw/Telco-Customer-Churn.csv exists")
    
    # Summary
    print("\n" + "="*70)
    print("✅ EXAMPLES COMPLETE")
    print("="*70)
    print("\n💡 Next Steps:")
    print("   1. Modify customer data above to test different scenarios")
    print("   2. Use app/api.py for REST API deployment")
    print("   3. Use app/dashboard.py for interactive web interface")
    print("   4. Use scripts/batch_predict.py for bulk predictions")
    print("\n📖 See DEPLOYMENT_GUIDE.md for more information")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
