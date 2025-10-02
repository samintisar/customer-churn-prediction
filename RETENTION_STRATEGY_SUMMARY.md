# Retention Strategy Module - Implementation Summary

## Overview
Successfully implemented a comprehensive retention strategy module (`src/retention_strategy.py`) that maps churn risk scores to actionable retention strategies with personalized recommendations.

## Implemented Functions

### 1. `classify_risk_tier(churn_probability)`
Classifies customers into risk tiers based on their churn probability:
- **HIGH**: >= 0.70 (Immediate intervention needed)
- **MEDIUM**: 0.40-0.69 (Proactive engagement)
- **LOW**: < 0.40 (Standard care)

**Example:**
```python
risk_tier = classify_risk_tier(0.85)  # Returns "HIGH"
```

### 2. `recommend_action(risk_tier, customer_profile)`
Generates personalized retention recommendations based on risk tier and customer profile.

**Recommendation Strategy:**
- **HIGH Risk Customers:**
  - Channel: Phone Call
  - Priority: 1 (Highest)
  - Actions:
    - High-value customers (>$70/mo): 20% discount + account manager + priority support
    - Month-to-month: 15% discount + contract upgrade offer
    - Others: 15% discount + priority support + service review

- **MEDIUM Risk Customers:**
  - Channel: Email Campaign
  - Priority: 2
  - Actions:
    - New customers: Satisfaction survey + loyalty program
    - Month-to-month: Loyalty program + contract upgrade incentive
    - Others: Satisfaction survey + exclusive perks

- **LOW Risk Customers:**
  - Channel: Newsletter
  - Priority: 3
  - Actions:
    - New customers: Onboarding tips + educational content
    - High-value: Premium feature highlights + usage tips
    - Others: Product updates + educational content

**Returns:**
```python
{
    'risk_tier': 'HIGH',
    'priority': 1,
    'action': 'Personal outreach from account manager with 20% discount...',
    'channel': 'Phone Call',
    'discount_percentage': 20,
    'estimated_cost': 205.20
}
```

### 3. `calculate_retention_value(customer_profile, months_retained=12)`
Calculates the estimated dollar value of retaining a customer.

**Value Calculation Formula:**
```
retention_value = monthly_charges × months_retained × contract_multiplier × tenure_multiplier
```

**Multipliers:**
- **Contract Type:**
  - Month-to-month: 0.7 (lower retention likelihood)
  - One year: 1.0 (standard)
  - Two year: 1.2 (higher retention likelihood)

- **Tenure:**
  - < 6 months: 0.8 (very new, risky)
  - 6-12 months: 0.9 (new)
  - 12-24 months: 1.0 (standard)
  - > 24 months: 1.1 (established, higher value)

**Example:**
```python
profile = {
    'MonthlyCharges': 85.50,
    'tenure': 24,
    'Contract': 'Two year'
}
value = calculate_retention_value(profile)  # Returns $1,354.32
```

### 4. `generate_retention_report(df, churn_proba_col='churn_probability', top_n=100)`
Generates a comprehensive retention action report with the top N at-risk customers.

**Report Columns:**
- `churn_probability`: Model prediction score
- `risk_tier`: HIGH/MEDIUM/LOW classification
- `priority`: 1-3 (1 = highest priority)
- `recommended_action`: Personalized action description
- `contact_channel`: Phone Call/Email/Newsletter
- `discount_percentage`: Recommended discount (0-20%)
- `intervention_cost`: Estimated cost of intervention
- `retention_value`: Estimated value of retaining customer
- `estimated_roi`: ROI calculation (retention_value - cost) / cost
- All original customer features

**Usage:**
```python
from src.retention_strategy import generate_retention_report

# Assuming df has customer data and churn predictions
report = generate_retention_report(
    df,
    churn_proba_col='churn_probability',
    top_n=100
)

# Report is sorted by churn probability (descending)
# Focus on highest risk customers first
```

### 5. `segment_customers(df, by='Contract')`
Segments customers for targeted retention strategies.

**Common Segmentation Variables:**
- `Contract`: Month-to-month vs. long-term contracts
- `tenure_group`: New vs. established customers
- Custom fields: Service usage, demographics, etc.

**Returns:** Dictionary mapping segment names to DataFrames

## Test Results

Successfully tested with 200 customer samples:

### Test Summary:
- ✓ Risk tier classification: 100% accurate
- ✓ Recommendation generation: Working correctly for all tiers
- ✓ Retention value calculation: Properly adjusting for contract and tenure
- ✓ Report generation: Successfully created for 50 top at-risk customers
- ✓ Customer segmentation: Segmented by contract type

### Sample Report Metrics:
```
Total customers analyzed: 50 (top at-risk)
Risk tier distribution:
  - HIGH: 50 customers
  - MEDIUM: 0 customers
  - LOW: 0 customers

Financial Metrics:
  - Total retention value: $34,075.57
  - Total intervention cost: $6,395.22
  - Average ROI: 6.80x

Segmentation by Contract:
  - Month-to-month: 29 customers (avg churn prob: 83.87%)
  - Two year: 15 customers (avg churn prob: 77.78%)
  - One year: 6 customers (avg churn prob: 78.78%)
```

### Top At-Risk Customer Example:
```
Churn Probability: 94.21%
Risk Tier: HIGH
Monthly Charges: $103.80
Tenure: 42 months
Retention Value: $959.11
Recommended Action: Personal outreach from account manager with 20% discount and priority support upgrade
Contact Channel: Phone Call
Discount: 20%
Intervention Cost: $249.12
ROI: 2.85x
```

## Output Files

The module generates:
1. **Retention Report CSV**: `reports/retention_report.csv`
   - Contains all customers with their risk tiers and recommendations
   - Sorted by churn probability (descending)
   - Ready for business action

## Business Value

### Key Benefits:
1. **Prioritized Action List**: Focus on highest-risk customers first
2. **Personalized Recommendations**: Tailored to customer value and profile
3. **Cost-Benefit Analysis**: Clear ROI for each intervention
4. **Segmentation**: Targeted campaigns for different customer groups

### Sample Business Insights:
- **Month-to-month contracts** have the highest average churn probability (83.87%)
- **Average ROI of 6.80x** indicates high value in retention efforts
- **Top 50 at-risk customers** represent **$34K in potential annual revenue**
- **Investment of $6.4K** in retention could save significant revenue

## Usage Example

```python
from src.retention_strategy import (
    classify_risk_tier,
    recommend_action,
    calculate_retention_value,
    generate_retention_report
)
import pandas as pd
import joblib

# Load model and test data
model = joblib.load('models/churn_predictor.pkl')
df = pd.read_csv('data/processed/test.csv')

# Generate predictions
X = df.drop(columns=['Churn'], errors='ignore')
df['churn_probability'] = model.predict_proba(X)[:, 1]

# Add original features for recommendation
df_raw = pd.read_csv('data/raw/Telco-Customer-Churn.csv')
df['Contract'] = df_raw['Contract']
df['MonthlyCharges'] = df_raw['MonthlyCharges']
df['tenure'] = df_raw['tenure']

# Generate retention report
retention_report = generate_retention_report(
    df,
    churn_proba_col='churn_probability',
    top_n=100
)

# Save for business action
retention_report.to_csv('reports/retention_action_plan.csv', index=False)

# Example: Process individual customer
customer_profile = {
    'MonthlyCharges': 75.00,
    'tenure': 18,
    'Contract': 'Month-to-month'
}
churn_prob = 0.82

tier = classify_risk_tier(churn_prob)
recommendation = recommend_action(tier, customer_profile)
value = calculate_retention_value(customer_profile)

print(f"Customer Risk: {tier}")
print(f"Action: {recommendation['action']}")
print(f"Retention Value: ${value:,.2f}")
```

## Testing

Run the test suite:
```bash
python -m src.retention_strategy
```

This will:
1. Load test data and trained model
2. Test all four main functions
3. Generate sample retention report
4. Display results and metrics
5. Save report to `reports/retention_report.csv`

## Dependencies

- pandas
- numpy
- logging
- joblib (for model loading in tests)
- scikit-learn (for model predictions in tests)

All dependencies are listed in `requirements.txt`.

## Notes

- The module uses logging for tracking report generation progress
- All monetary values are rounded to 2 decimal places
- ROI calculation avoids division by zero for low-cost interventions
- Unicode characters replaced with ASCII for Windows compatibility
- Report is automatically sorted by churn probability (highest risk first)

