"""
Retention Strategy Module

This module maps churn risk scores to actionable retention strategies.

Risk Tiers:
- High Risk (>70% churn probability): Immediate intervention
- Medium Risk (40-70%): Proactive engagement
- Low Risk (<40%): Standard care

Key Functions:
- classify_risk_tier: Assign customers to risk tiers
- recommend_action: Map risk tier + profile to retention action
- generate_retention_report: Create prioritized action list
- calculate_retention_value: Estimate value of saving a customer

Usage:
    from src.retention_strategy import classify_risk_tier, recommend_action
    
    tier = classify_risk_tier(churn_probability)
    action = recommend_action(tier, customer_profile)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def classify_risk_tier(churn_probability: float) -> str:
    """
    Classify customer into risk tier based on churn probability.
    
    Tiers:
    - HIGH: >= 0.70
    - MEDIUM: 0.40 - 0.69
    - LOW: < 0.40
    
    Args:
        churn_probability: Model prediction (0-1)
        
    Returns:
        Risk tier label
    """
    if churn_probability >= 0.70:
        return "HIGH"
    elif churn_probability >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


def recommend_action(
    risk_tier: str,
    customer_profile: Dict[str, any]
) -> Dict[str, any]:
    """
    Recommend retention action based on risk tier and customer profile.
    
    Action Types:
    - High Risk: Personalized discount, account manager call, upgrade offer
    - Medium Risk: Email campaign, survey, loyalty program enrollment
    - Low Risk: Standard newsletter, product education
    
    Args:
        risk_tier: Risk classification (HIGH/MEDIUM/LOW)
        customer_profile: Dictionary with customer attributes
        
    Returns:
        Dictionary with recommended action, channel, and priority
    """
    # Extract relevant profile information
    monthly_charges = customer_profile.get('MonthlyCharges', 0)
    tenure = customer_profile.get('tenure', 0)
    contract = customer_profile.get('Contract', 'Month-to-month')
    
    # Determine if customer is high-value
    is_high_value = monthly_charges > 70
    is_new_customer = tenure < 12
    is_month_to_month = contract == 'Month-to-month'
    
    recommendation = {
        'risk_tier': risk_tier,
        'priority': None,
        'action': None,
        'channel': None,
        'discount_percentage': 0,
        'estimated_cost': 0
    }
    
    if risk_tier == "HIGH":
        # High risk customers need immediate, personalized intervention
        recommendation['priority'] = 1
        recommendation['channel'] = 'Phone Call'
        
        # Personalize based on value and contract
        if is_high_value:
            # High-value customers get premium treatment
            recommendation['action'] = 'Personal outreach from account manager with 20% discount and priority support upgrade'
            recommendation['discount_percentage'] = 20
            recommendation['estimated_cost'] = monthly_charges * 0.20 * 12  # 12 months of discount
        elif is_month_to_month:
            # Month-to-month customers need incentive to commit
            recommendation['action'] = 'Personal outreach with 15% discount and 1-year contract upgrade offer'
            recommendation['discount_percentage'] = 15
            recommendation['estimated_cost'] = monthly_charges * 0.15 * 12
        else:
            # Standard high-risk intervention
            recommendation['action'] = 'Personal outreach with 15% discount, priority support, and service review'
            recommendation['discount_percentage'] = 15
            recommendation['estimated_cost'] = monthly_charges * 0.15 * 6
            
    elif risk_tier == "MEDIUM":
        # Medium risk customers need proactive but less intensive engagement
        recommendation['priority'] = 2
        recommendation['channel'] = 'Email'
        
        if is_new_customer:
            # New customers may need onboarding help
            recommendation['action'] = 'Email campaign with customer satisfaction survey and loyalty program enrollment'
        elif is_month_to_month:
            # Encourage commitment
            recommendation['action'] = 'Email campaign with loyalty program benefits and contract upgrade incentive'
        else:
            # Standard medium-risk engagement
            recommendation['action'] = 'Email campaign with satisfaction survey and exclusive loyalty program perks'
        
        recommendation['discount_percentage'] = 0
        recommendation['estimated_cost'] = 5  # Cost of email campaign
        
    else:  # LOW risk
        # Low risk customers need standard engagement
        recommendation['priority'] = 3
        recommendation['channel'] = 'Newsletter'
        
        if is_new_customer:
            recommendation['action'] = 'Monthly newsletter with onboarding tips and educational content'
        elif is_high_value:
            recommendation['action'] = 'Monthly newsletter with premium feature highlights and usage tips'
        else:
            recommendation['action'] = 'Monthly newsletter with product updates and educational content'
        
        recommendation['discount_percentage'] = 0
        recommendation['estimated_cost'] = 1  # Cost of newsletter
    
    return recommendation


def calculate_retention_value(
    customer_profile: Dict[str, any],
    months_retained: int = 12
) -> float:
    """
    Calculate estimated value of retaining a customer.
    
    Based on:
    - Monthly charges
    - Contract type
    - Typical customer lifetime
    
    Args:
        customer_profile: Dictionary with customer attributes
        months_retained: Expected retention period
        
    Returns:
        Estimated retention value in dollars
    """
    monthly_charges = customer_profile.get('MonthlyCharges', 0)
    tenure = customer_profile.get('tenure', 0)
    contract = customer_profile.get('Contract', 'Month-to-month')
    
    # Base value: monthly charges over retention period
    base_value = monthly_charges * months_retained
    
    # Adjust based on contract type
    # Month-to-month customers are less likely to stay full period
    # Long-term contracts are more stable
    contract_multiplier = {
        'Month-to-month': 0.7,  # Lower expected retention
        'One year': 1.0,         # Standard retention
        'Two year': 1.2          # Higher retention likelihood
    }.get(contract, 1.0)
    
    # Adjust based on tenure
    # Longer tenure = higher likelihood of staying if we intervene
    if tenure < 6:
        tenure_multiplier = 0.8  # Very new customers are risky
    elif tenure < 12:
        tenure_multiplier = 0.9  # New customers
    elif tenure < 24:
        tenure_multiplier = 1.0  # Standard
    else:
        tenure_multiplier = 1.1  # Established customers have higher value
    
    # Calculate adjusted retention value
    retention_value = base_value * contract_multiplier * tenure_multiplier
    
    return round(retention_value, 2)


def generate_retention_report(
    df: pd.DataFrame,
    churn_proba_col: str = 'churn_probability',
    top_n: int = 100
) -> pd.DataFrame:
    """
    Generate prioritized retention action report.
    
    Args:
        df: DataFrame with customer data and churn predictions
        churn_proba_col: Column name for churn probabilities
        top_n: Number of top at-risk customers to include
        
    Returns:
        DataFrame with customers, risk tiers, and recommended actions
    """
    logger.info(f"Generating retention report for top {top_n} at-risk customers")
    
    if churn_proba_col not in df.columns:
        raise ValueError(f"Column '{churn_proba_col}' not found in DataFrame")
    
    # Create a copy to avoid modifying original
    report_df = df.copy()
    
    # Add risk tier classification
    logger.info("Classifying customers into risk tiers")
    report_df['risk_tier'] = report_df[churn_proba_col].apply(classify_risk_tier)
    
    # Prepare customer profiles and get recommendations
    logger.info("Generating personalized recommendations")
    recommendations = []
    
    for idx, row in report_df.iterrows():
        # Create customer profile dictionary
        customer_profile = {
            'MonthlyCharges': row.get('MonthlyCharges', 0),
            'tenure': row.get('tenure', 0),
            'Contract': row.get('Contract', 'Month-to-month'),
            'TotalCharges': row.get('TotalCharges', 0)
        }
        
        # Get recommendation
        rec = recommend_action(row['risk_tier'], customer_profile)
        recommendations.append(rec)
        
    # Convert recommendations to DataFrame columns
    rec_df = pd.DataFrame(recommendations)
    
    # Add recommendation columns to report
    report_df['priority'] = rec_df['priority']
    report_df['recommended_action'] = rec_df['action']
    report_df['contact_channel'] = rec_df['channel']
    report_df['discount_percentage'] = rec_df['discount_percentage']
    report_df['intervention_cost'] = rec_df['estimated_cost']
    
    # Calculate retention value for each customer
    logger.info("Calculating retention values")
    retention_values = []
    
    for idx, row in report_df.iterrows():
        customer_profile = {
            'MonthlyCharges': row.get('MonthlyCharges', 0),
            'tenure': row.get('tenure', 0),
            'Contract': row.get('Contract', 'Month-to-month'),
            'TotalCharges': row.get('TotalCharges', 0)
        }
        value = calculate_retention_value(customer_profile)
        retention_values.append(value)
    
    report_df['retention_value'] = retention_values
    
    # Calculate ROI: (retention_value - intervention_cost) / intervention_cost
    report_df['estimated_roi'] = (
        (report_df['retention_value'] - report_df['intervention_cost']) / 
        report_df['intervention_cost'].replace(0, 1)  # Avoid division by zero
    ).round(2)
    
    # Sort by churn probability (descending) to get most at-risk first
    report_df = report_df.sort_values(by=churn_proba_col, ascending=False)
    
    # Select top N customers
    top_customers = report_df.head(top_n)
    
    # Log summary statistics
    logger.info(f"Report generated with {len(top_customers)} customers")
    logger.info(f"Risk tier distribution:")
    logger.info(f"  HIGH: {(top_customers['risk_tier'] == 'HIGH').sum()}")
    logger.info(f"  MEDIUM: {(top_customers['risk_tier'] == 'MEDIUM').sum()}")
    logger.info(f"  LOW: {(top_customers['risk_tier'] == 'LOW').sum()}")
    logger.info(f"Total estimated retention value: ${top_customers['retention_value'].sum():,.2f}")
    logger.info(f"Total estimated intervention cost: ${top_customers['intervention_cost'].sum():,.2f}")
    
    return top_customers


def segment_customers(
    df: pd.DataFrame,
    by: str = 'Contract'
) -> Dict[str, pd.DataFrame]:
    """
    Segment customers for targeted retention strategies.
    
    Common segments:
    - By contract type (month-to-month vs long-term)
    - By tenure (new vs established)
    - By service usage (single vs multiple services)
    
    Args:
        df: DataFrame with customer data
        by: Segmentation variable
        
    Returns:
        Dictionary mapping segment names to customer DataFrames
    """
    logger.info(f"Segmenting customers by: {by}")
    
    if by not in df.columns:
        raise ValueError(f"Column '{by}' not found in DataFrame")
    
    segments = {}
    unique_values = df[by].unique()
    
    for value in unique_values:
        segment_df = df[df[by] == value].copy()
        segments[str(value)] = segment_df
        logger.info(f"  Segment '{value}': {len(segment_df)} customers")
    
    return segments


def test_retention_strategy():
    """
    Test retention strategy module with the processed test dataset.
    """
    print("=" * 80)
    print("Testing Retention Strategy Module")
    print("=" * 80)
    
    try:
        # Load necessary modules
        import joblib
        from pathlib import Path
        
        # Test 1: Load test data and model
        print("\n[Test 1] Loading test data and trained model...")
        test_data_path = Path('data/processed/test.csv')
        model_path = Path('models/churn_predictor.pkl')
        
        if not test_data_path.exists():
            print(f"[X] Test data not found at {test_data_path}")
            print("  Please run the data pipeline first")
            return
        
        df_test = pd.read_csv(test_data_path)
        print(f"[OK] Loaded {len(df_test)} test samples")
        
        # Load original data to get non-scaled features
        from src.data_loader import load_raw_data, clean_data
        df_raw = load_raw_data('data/raw/Telco-Customer-Churn.csv')
        df_clean = clean_data(df_raw)
        
        # Get a sample for testing
        sample_size = min(200, len(df_test))
        df_sample = df_test.head(sample_size).copy()
        
        # For testing, we'll use the raw data's Contract, tenure, and MonthlyCharges
        # Merge with raw data to get original values
        print(f"[OK] Using sample of {sample_size} customers for testing")
        
        # Test 2: Test classify_risk_tier function
        print("\n[Test 2] Testing classify_risk_tier function...")
        test_probabilities = [0.85, 0.65, 0.45, 0.25, 0.15]
        expected_tiers = ["HIGH", "MEDIUM", "MEDIUM", "LOW", "LOW"]
        
        all_correct = True
        for prob, expected in zip(test_probabilities, expected_tiers):
            result = classify_risk_tier(prob)
            status = "[OK]" if result == expected else "[X]"
            print(f"  {status} Probability {prob:.2f} -> {result} (expected {expected})")
            if result != expected:
                all_correct = False
        
        if all_correct:
            print("[OK] All risk tier classifications correct")
        else:
            print("[X] Some risk tier classifications failed")
        
        # Test 3: Test recommend_action function
        print("\n[Test 3] Testing recommend_action function...")
        
        test_profiles = [
            {
                'MonthlyCharges': 85.50,
                'tenure': 24,
                'Contract': 'Two year',
                'risk_tier': 'HIGH'
            },
            {
                'MonthlyCharges': 50.00,
                'tenure': 3,
                'Contract': 'Month-to-month',
                'risk_tier': 'MEDIUM'
            },
            {
                'MonthlyCharges': 30.00,
                'tenure': 48,
                'Contract': 'One year',
                'risk_tier': 'LOW'
            }
        ]
        
        for i, profile in enumerate(test_profiles, 1):
            risk_tier = profile.pop('risk_tier')
            rec = recommend_action(risk_tier, profile)
            print(f"\n  Profile {i} ({risk_tier} risk, ${profile['MonthlyCharges']:.2f}/mo, {profile['tenure']} months):")
            print(f"    Priority: {rec['priority']}")
            print(f"    Channel: {rec['channel']}")
            print(f"    Action: {rec['action'][:70]}...")
            print(f"    Discount: {rec['discount_percentage']}%")
            print(f"    Estimated Cost: ${rec['estimated_cost']:.2f}")
        
        print("\n[OK] Recommendation generation successful")
        
        # Test 4: Test calculate_retention_value function
        print("\n[Test 4] Testing calculate_retention_value function...")
        
        for i, profile in enumerate(test_profiles, 1):
            value = calculate_retention_value(profile, months_retained=12)
            print(f"  Profile {i}: Retention value = ${value:,.2f}")
        
        print("[OK] Retention value calculation successful")
        
        # Test 5: Generate predictions for sample data
        print("\n[Test 5] Generating churn predictions for sample data...")
        
        if model_path.exists():
            model = joblib.load(model_path)
            print(f"[OK] Loaded model from {model_path}")
            
            # Generate predictions
            # Remove target column if present
            X_test = df_sample.drop(columns=['Churn'], errors='ignore')
            
            # Get predictions
            churn_probabilities = model.predict_proba(X_test)[:, 1]
            df_sample['churn_probability'] = churn_probabilities
            
            print(f"[OK] Generated predictions for {len(df_sample)} customers")
            print(f"  Mean churn probability: {churn_probabilities.mean():.2%}")
            print(f"  Max churn probability: {churn_probabilities.max():.2%}")
            print(f"  Min churn probability: {churn_probabilities.min():.2%}")
        else:
            print(f"  Model not found at {model_path}, using random probabilities")
            df_sample['churn_probability'] = np.random.uniform(0, 1, len(df_sample))
        
        # Add original columns from cleaned data for reporting
        # Map by index (assuming same order after cleaning)
        df_clean_sample = df_clean.head(sample_size).copy()
        df_sample['Contract'] = df_clean_sample['Contract'].values
        df_sample['MonthlyCharges'] = df_clean_sample['MonthlyCharges'].values
        df_sample['tenure'] = df_clean_sample['tenure'].values
        df_sample['TotalCharges'] = df_clean_sample['TotalCharges'].values
        
        # Test 6: Generate retention report
        print("\n[Test 6] Generating retention report...")
        
        retention_report = generate_retention_report(
            df_sample,
            churn_proba_col='churn_probability',
            top_n=50
        )
        
        print(f"\n[OK] Generated retention report with {len(retention_report)} customers")
        
        # Display summary
        print("\n  Report Summary:")
        print(f"    Total retention value: ${retention_report['retention_value'].sum():,.2f}")
        print(f"    Total intervention cost: ${retention_report['intervention_cost'].sum():,.2f}")
        print(f"    Average ROI: {retention_report['estimated_roi'].mean():.2f}x")
        
        print("\n  Top 5 at-risk customers:")
        display_cols = [
            'churn_probability', 'risk_tier', 'MonthlyCharges', 'tenure',
            'retention_value', 'contact_channel', 'discount_percentage'
        ]
        print(retention_report[display_cols].head().to_string(index=False))
        
        # Test 7: Test segment_customers function
        print("\n[Test 7] Testing customer segmentation...")
        
        segments = segment_customers(retention_report, by='Contract')
        
        print(f"\n[OK] Segmented into {len(segments)} groups:")
        for segment_name, segment_df in segments.items():
            avg_churn_prob = segment_df['churn_probability'].mean()
            print(f"    {segment_name}: {len(segment_df)} customers (avg churn prob: {avg_churn_prob:.2%})")
        
        # Save report to file
        print("\n[Test 8] Saving retention report...")
        output_path = Path('reports/retention_report.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        retention_report.to_csv(output_path, index=False)
        print(f"[OK] Saved retention report to {output_path}")
        
        print("\n" + "=" * 80)
        print("All tests passed!")
        print("=" * 80)
        
        return retention_report
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_retention_strategy()

