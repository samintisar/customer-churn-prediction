"""
Batch Churn Prediction Script

Generates churn predictions for a batch of customers and saves results to CSV.

Usage:
    python scripts/batch_predict.py --input data/raw/Telco-Customer-Churn.csv --output results/predictions.csv
    
    python scripts/batch_predict.py -i data/customers.csv -o results/risk_report.csv --top-n 100
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


def batch_predict(input_file: str, output_file: str, top_n: int = None):
    """
    Run batch predictions on customer data
    
    Args:
        input_file: Path to input CSV file with customer data
        output_file: Path to save predictions CSV
        top_n: If specified, only save top N highest-risk customers
    """
    
    print("\n" + "="*70)
    print("🚀 BATCH CHURN PREDICTION")
    print("="*70)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    if top_n:
        print(f"Filter: Top {top_n} highest-risk customers")
    print()
    
    # Load models
    print("📦 Loading models...")
    try:
        model = joblib.load('models/logistic_regression.pkl')
        feature_engineer = joblib.load('models/feature_engineer.pkl')
        print("   ✅ Models loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading models: {e}")
        return None
    
    # Load data
    print("\n📊 Loading customer data...")
    try:
        df = pd.read_csv(input_file)
        print(f"   ✅ Loaded {len(df):,} customers")
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return None
    
    # Store customer IDs and original data
    if 'customerID' in df.columns:
        customer_ids = df['customerID']
    else:
        customer_ids = [f'Customer_{i}' for i in range(len(df))]
    
    # Keep some original columns for context
    original_cols = ['customerID', 'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges']
    original_data = df[[col for col in original_cols if col in df.columns]].copy()
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X = df.drop(['customerID', 'Churn'], axis=1, errors='ignore')
    
    try:
        X_transformed = feature_engineer.transform(X)
        print(f"   ✅ Features transformed ({X_transformed.shape[1]} features)")
    except Exception as e:
        print(f"   ❌ Error transforming features: {e}")
        return None
    
    # Get predictions
    print("\n🎯 Generating predictions...")
    try:
        churn_probs = model.predict_proba(X_transformed)[:, 1]
        churn_preds = model.predict(X_transformed)
        print(f"   ✅ Predictions generated")
    except Exception as e:
        print(f"   ❌ Error generating predictions: {e}")
        return None
    
    # Create results dataframe
    results = pd.DataFrame({
        'customerID': customer_ids,
        'churn_probability': churn_probs,
        'churn_probability_pct': [f"{prob:.2%}" for prob in churn_probs],
        'churn_prediction': ['Yes' if pred == 1 else 'No' for pred in churn_preds],
        'risk_tier': [classify_risk_tier(prob) for prob in churn_probs]
    })
    
    # Add recommended actions
    print("\n💡 Generating recommendations...")
    results['recommended_action'] = results.apply(
        lambda row: recommend_action(row['churn_probability'], {}), 
        axis=1
    )
    
    # Merge with original data for context
    results = pd.concat([results, original_data.drop('customerID', axis=1, errors='ignore')], axis=1)
    
    # Sort by risk (highest first)
    results = results.sort_values('churn_probability', ascending=False)
    
    # Filter top N if specified
    if top_n:
        results = results.head(top_n)
        print(f"   ✅ Filtered to top {top_n} highest-risk customers")
    
    # Save results
    print(f"\n💾 Saving results...")
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"   ✅ Results saved to: {output_file}")
    except Exception as e:
        print(f"   ❌ Error saving results: {e}")
        return None
    
    # Print summary
    print("\n" + "="*70)
    print("📈 PREDICTION SUMMARY")
    print("="*70)
    print(f"Total customers processed:  {len(results):,}")
    print(f"Predicted to churn:         {(results['churn_prediction'] == 'Yes').sum():,} ({(results['churn_prediction'] == 'Yes').mean():.1%})")
    print(f"Predicted to stay:          {(results['churn_prediction'] == 'No').sum():,} ({(results['churn_prediction'] == 'No').mean():.1%})")
    
    print(f"\n🎯 Risk Tier Distribution:")
    risk_counts = results['risk_tier'].value_counts()
    for tier in ['High Risk', 'Medium Risk', 'Low Risk']:
        count = risk_counts.get(tier, 0)
        pct = count / len(results) * 100
        print(f"   {tier:<15} {count:>6,} ({pct:>5.1f}%)")
    
    print(f"\n📊 Statistics:")
    print(f"   Average churn probability:  {results['churn_probability'].mean():.2%}")
    print(f"   Median churn probability:   {results['churn_probability'].median():.2%}")
    print(f"   Max churn probability:      {results['churn_probability'].max():.2%}")
    print(f"   Min churn probability:      {results['churn_probability'].min():.2%}")
    
    print(f"\n🔝 Top 5 Highest Risk Customers:")
    print(results[['customerID', 'churn_probability_pct', 'risk_tier', 'Contract']].head().to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ BATCH PREDICTION COMPLETE")
    print("="*70 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch Churn Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict all customers
  python scripts/batch_predict.py -i data/raw/Telco-Customer-Churn.csv -o results/predictions.csv
  
  # Get top 100 highest-risk customers
  python scripts/batch_predict.py -i data/customers.csv -o results/top_100.csv --top-n 100
  
  # Save with timestamp
  python scripts/batch_predict.py -i data/customers.csv -o results/predictions_$(date +%%Y%%m%%d).csv
        """
    )
    
    parser.add_argument(
        '-i', '--input', 
        required=True, 
        help='Input CSV file with customer data'
    )
    parser.add_argument(
        '-o', '--output', 
        required=True, 
        help='Output CSV file for predictions'
    )
    parser.add_argument(
        '--top-n', 
        type=int, 
        default=None, 
        help='Only save top N highest-risk customers (optional)'
    )
    
    args = parser.parse_args()
    
    # Run batch prediction
    batch_predict(args.input, args.output, args.top_n)


if __name__ == '__main__':
    main()
