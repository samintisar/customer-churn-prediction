"""
Generate Retention Action Plan Report

This script generates a comprehensive retention action plan by:
1. Loading the trained churn prediction model
2. Generating predictions for all customers in the test set
3. Classifying risk tiers and recommending actions
4. Calculating retention values
5. Identifying top risk factors for each customer
6. Creating an actionable CSV report

Usage:
    python scripts/generate_retention_report.py
    python scripts/generate_retention_report.py --top_n 200
    python scripts/generate_retention_report.py --top_n 100 --output custom_report.csv
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retention_strategy import (
    classify_risk_tier,
    recommend_action,
    calculate_retention_value,
    generate_retention_report
)
from src.data_loader import load_raw_data, clean_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_customer_ids(raw_data_path: str, sample_size: int = None) -> pd.Series:
    """
    Load customer IDs from raw data.
    
    Args:
        raw_data_path: Path to raw data CSV
        sample_size: Optional number of samples to load
        
    Returns:
        Series of customer IDs
    """
    logger.info("Loading customer IDs from raw data")
    df_raw = pd.read_csv(raw_data_path)
    
    if sample_size:
        df_raw = df_raw.head(sample_size)
    
    if 'customerID' in df_raw.columns:
        return df_raw['customerID']
    else:
        logger.warning("customerID not found, generating synthetic IDs")
        return pd.Series([f"CUST_{i:05d}" for i in range(len(df_raw))])


def identify_top_risk_factors(
    model,
    customer_features: pd.Series,
    feature_names: list,
    top_n: int = 3
) -> str:
    """
    Identify top risk factors for a customer based on feature importance.
    
    Args:
        model: Trained model
        customer_features: Single customer's feature values
        feature_names: List of feature names
        top_n: Number of top factors to return
        
    Returns:
        Comma-separated string of top risk factors
    """
    # Get feature importance from model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return "Unable to determine"
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'value': customer_features.values
    })
    
    # Sort by importance and get top N
    top_factors = importance_df.nlargest(top_n, 'importance')
    
    # Create human-readable factor names
    factor_list = []
    for _, row in top_factors.iterrows():
        feature = row['feature']
        
        # Clean up feature names for readability
        if feature.startswith('Contract_'):
            factor_list.append(f"Contract: {feature.split('_')[1]}")
        elif feature.startswith('PaymentMethod_'):
            factor_list.append(f"Payment: {feature.split('_', 1)[1]}")
        elif feature.startswith('InternetService_'):
            factor_list.append(f"Internet: {feature.split('_', 1)[1]}")
        elif feature == 'tenure':
            factor_list.append("Short tenure")
        elif feature == 'MonthlyCharges':
            factor_list.append("Monthly charges")
        elif feature == 'is_new_customer':
            factor_list.append("New customer")
        elif feature.startswith('has_'):
            factor_list.append(feature.replace('_', ' ').title())
        else:
            # Generic cleanup
            factor_list.append(feature.replace('_', ' ').title())
    
    return ", ".join(factor_list)


def generate_retention_action_plan(
    model_path: str = 'models/churn_predictor.pkl',
    test_data_path: str = 'data/processed/test.csv',
    raw_data_path: str = 'data/raw/Telco-Customer-Churn.csv',
    output_path: str = 'reports/retention_action_plan.csv',
    top_n: int = None
) -> pd.DataFrame:
    """
    Generate comprehensive retention action plan.
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to processed test data
        raw_data_path: Path to raw data (for customer IDs)
        output_path: Path to save output CSV
        top_n: Number of top customers to include (None = all)
        
    Returns:
        DataFrame with retention action plan
    """
    logger.info("=" * 80)
    logger.info("Generating Retention Action Plan")
    logger.info("=" * 80)
    
    # Step 1: Load trained model
    logger.info(f"\n[Step 1/6] Loading trained model from {model_path}")
    model = joblib.load(model_path)
    logger.info(f"[OK] Model loaded successfully: {type(model).__name__}")
    
    # Step 2: Load test data
    logger.info(f"\n[Step 2/6] Loading test data from {test_data_path}")
    df_test = pd.read_csv(test_data_path)
    logger.info(f"[OK] Loaded {len(df_test)} customer records")
    
    # Step 3: Generate predictions
    logger.info("\n[Step 3/6] Generating churn probability predictions")
    X_test = df_test.drop(columns=['Churn'], errors='ignore')
    
    churn_probabilities = model.predict_proba(X_test)[:, 1]
    df_test['churn_probability'] = churn_probabilities
    
    logger.info(f"[OK] Generated predictions")
    logger.info(f"  Mean churn probability: {churn_probabilities.mean():.2%}")
    logger.info(f"  High risk (>=70%): {(churn_probabilities >= 0.70).sum()} customers")
    logger.info(f"  Medium risk (40-70%): {((churn_probabilities >= 0.40) & (churn_probabilities < 0.70)).sum()} customers")
    logger.info(f"  Low risk (<40%): {(churn_probabilities < 0.40).sum()} customers")
    
    # Step 4: Load raw data for customer context
    logger.info(f"\n[Step 4/6] Loading customer context from {raw_data_path}")
    df_raw = load_raw_data(raw_data_path)
    df_clean = clean_data(df_raw)
    
    # Match test data size (assuming same order after cleaning and splitting)
    # We'll use the last portion for test data
    total_samples = len(df_clean)
    test_samples = len(df_test)
    df_context = df_clean.iloc[-test_samples:].reset_index(drop=True)
    
    # Add context columns to test data
    df_test['customerID'] = df_context['customerID'].values if 'customerID' in df_context.columns else [f"CUST_{i:05d}" for i in range(len(df_test))]
    df_test['Contract'] = df_context['Contract'].values
    df_test['tenure'] = df_context['tenure'].values
    df_test['MonthlyCharges'] = df_context['MonthlyCharges'].values
    df_test['TotalCharges'] = df_context['TotalCharges'].values
    
    logger.info(f"[OK] Added customer context (ID, Contract, tenure, charges)")
    
    # Step 5: Generate retention recommendations
    logger.info("\n[Step 5/6] Generating retention recommendations")
    
    # Classify risk tiers
    df_test['risk_tier'] = df_test['churn_probability'].apply(classify_risk_tier)
    
    # Generate recommendations
    recommendations = []
    retention_values = []
    
    for idx, row in df_test.iterrows():
        customer_profile = {
            'MonthlyCharges': row['MonthlyCharges'],
            'tenure': row['tenure'],
            'Contract': row['Contract'],
            'TotalCharges': row['TotalCharges']
        }
        
        # Get recommendation
        rec = recommend_action(row['risk_tier'], customer_profile)
        recommendations.append(rec['action'])
        
        # Calculate retention value
        value = calculate_retention_value(customer_profile)
        retention_values.append(value)
    
    df_test['recommended_action'] = recommendations
    df_test['retention_value'] = retention_values
    
    logger.info("[OK] Generated personalized recommendations")
    
    # Step 6: Identify top risk factors
    logger.info("\n[Step 6/6] Identifying top risk factors for each customer")
    
    feature_names = X_test.columns.tolist()
    top_risk_factors = []
    
    for idx, row in X_test.iterrows():
        factors = identify_top_risk_factors(model, row, feature_names, top_n=3)
        top_risk_factors.append(factors)
    
    df_test['key_risk_factors'] = top_risk_factors
    
    logger.info("[OK] Identified key risk factors")
    
    # Create final report
    logger.info("\n[Creating Final Report]")
    
    # Select and order columns for the report
    report_columns = [
        'customerID',
        'churn_probability',
        'risk_tier',
        'recommended_action',
        'retention_value',
        'key_risk_factors',
        'Contract',
        'tenure',
        'MonthlyCharges'
    ]
    
    report = df_test[report_columns].copy()
    
    # Sort by churn probability (highest risk first)
    report = report.sort_values('churn_probability', ascending=False)
    
    # Filter to top N if specified
    if top_n:
        report = report.head(top_n)
        logger.info(f"[OK] Filtered to top {top_n} at-risk customers")
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    
    logger.info(f"[OK] Saved retention action plan to {output_path}")
    
    return report


def print_summary_statistics(report: pd.DataFrame):
    """
    Print summary statistics from the retention report.
    
    Args:
        report: Retention action plan DataFrame
    """
    print("\n" + "=" * 80)
    print("RETENTION ACTION PLAN SUMMARY")
    print("=" * 80)
    
    # Count by risk tier
    print("\n[CUSTOMER COUNT BY RISK TIER]")
    print("-" * 40)
    risk_counts = report['risk_tier'].value_counts()
    for tier in ['HIGH', 'MEDIUM', 'LOW']:
        count = risk_counts.get(tier, 0)
        percentage = (count / len(report) * 100) if len(report) > 0 else 0
        print(f"  {tier:8s}: {count:5d} customers ({percentage:5.1f}%)")
    
    # Total retention value at stake
    print("\n[FINANCIAL IMPACT]")
    print("-" * 40)
    total_value = report['retention_value'].sum()
    print(f"  Total Retention Value at Stake: ${total_value:,.2f}")
    print(f"  Average Retention Value:        ${report['retention_value'].mean():,.2f}")
    
    # Average retention value by tier
    print("\n[AVERAGE RETENTION VALUE BY RISK TIER]")
    print("-" * 40)
    for tier in ['HIGH', 'MEDIUM', 'LOW']:
        tier_data = report[report['risk_tier'] == tier]
        if len(tier_data) > 0:
            avg_value = tier_data['retention_value'].mean()
            total_tier_value = tier_data['retention_value'].sum()
            print(f"  {tier:8s}: ${avg_value:8,.2f} avg  |  ${total_tier_value:10,.2f} total")
        else:
            print(f"  {tier:8s}: No customers in this tier")
    
    # Churn probability statistics
    print("\n[CHURN PROBABILITY STATISTICS]")
    print("-" * 40)
    print(f"  Mean:   {report['churn_probability'].mean():.2%}")
    print(f"  Median: {report['churn_probability'].median():.2%}")
    print(f"  Max:    {report['churn_probability'].max():.2%}")
    print(f"  Min:    {report['churn_probability'].min():.2%}")
    
    # Contract type breakdown
    print("\n[CONTRACT TYPE BREAKDOWN]")
    print("-" * 40)
    contract_counts = report['Contract'].value_counts()
    for contract, count in contract_counts.items():
        percentage = (count / len(report) * 100)
        avg_churn = report[report['Contract'] == contract]['churn_probability'].mean()
        print(f"  {contract:16s}: {count:4d} ({percentage:5.1f}%) | Avg churn: {avg_churn:.2%}")
    
    # Top 5 most at-risk customers
    print("\n[TOP 5 MOST AT-RISK CUSTOMERS]")
    print("-" * 40)
    top_5 = report.head(5)
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"\n  {idx}. Customer {row['customerID']}")
        print(f"     Churn Probability: {row['churn_probability']:.1%}")
        print(f"     Risk Tier: {row['risk_tier']}")
        print(f"     Retention Value: ${row['retention_value']:,.2f}")
        print(f"     Contract: {row['Contract']} | Tenure: {row['tenure']} months | Charges: ${row['MonthlyCharges']:.2f}/mo")
        print(f"     Key Risk Factors: {row['key_risk_factors']}")
        print(f"     Recommended Action: {row['recommended_action'][:80]}...")
    
    print("\n" + "=" * 80)
    print(f"[OK] Report contains {len(report)} customers sorted by churn risk")
    print("=" * 80 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate Retention Action Plan Report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report for all customers
  python scripts/generate_retention_report.py
  
  # Generate report for top 100 at-risk customers
  python scripts/generate_retention_report.py --top_n 100
  
  # Specify custom output path
  python scripts/generate_retention_report.py --top_n 200 --output reports/my_report.csv
        """
    )
    
    parser.add_argument(
        '--top_n',
        type=int,
        default=None,
        help='Number of top at-risk customers to include (default: all)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='reports/retention_action_plan.csv',
        help='Output CSV file path (default: reports/retention_action_plan.csv)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/churn_predictor.pkl',
        help='Path to trained model (default: models/churn_predictor.pkl)'
    )
    
    parser.add_argument(
        '--test_data',
        type=str,
        default='data/processed/test.csv',
        help='Path to test data (default: data/processed/test.csv)'
    )
    
    parser.add_argument(
        '--raw_data',
        type=str,
        default='data/raw/Telco-Customer-Churn.csv',
        help='Path to raw data (default: data/raw/Telco-Customer-Churn.csv)'
    )
    
    args = parser.parse_args()
    
    try:
        # Generate report
        report = generate_retention_action_plan(
            model_path=args.model,
            test_data_path=args.test_data,
            raw_data_path=args.raw_data,
            output_path=args.output,
            top_n=args.top_n
        )
        
        # Print summary statistics
        print_summary_statistics(report)
        
        logger.info("[SUCCESS] Retention action plan generated successfully!")
        logger.info(f"[OUTPUT] Report saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] Error generating retention report: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

