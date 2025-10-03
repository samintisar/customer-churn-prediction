# Retention Report Generator

## Overview

The `generate_retention_report.py` script creates comprehensive, actionable retention reports by combining churn predictions with strategic retention recommendations.

## Features

✓ **Automated Predictions**: Loads trained model and generates churn probabilities for all customers  
✓ **Risk Classification**: Categorizes customers into HIGH/MEDIUM/LOW risk tiers  
✓ **Personalized Actions**: Recommends specific retention strategies based on customer profile  
✓ **Value Calculation**: Estimates the dollar value of retaining each customer  
✓ **Risk Factor Analysis**: Identifies top 3 risk factors contributing to churn probability  
✓ **Summary Statistics**: Displays comprehensive analytics and top at-risk customers  
✓ **CSV Export**: Saves actionable report ready for business teams  

## Usage

### Basic Usage

Generate report for all customers:
```bash
python scripts/generate_retention_report.py
```

### Generate Report for Top N Customers

Focus on the most at-risk customers:
```bash
# Top 50 at-risk customers
python scripts/generate_retention_report.py --top_n 50

# Top 100 at-risk customers
python scripts/generate_retention_report.py --top_n 100

# Top 200 at-risk customers
python scripts/generate_retention_report.py --top_n 200
```

### Custom Output Path

Specify where to save the report:
```bash
python scripts/generate_retention_report.py --top_n 100 --output reports/monthly_retention_plan.csv
```

### Advanced Options

Use custom model or data files:
```bash
python scripts/generate_retention_report.py \
    --model models/custom_model.pkl \
    --test_data data/processed/custom_test.csv \
    --top_n 150 \
    --output reports/custom_report.csv
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--top_n` | int | None (all) | Number of top at-risk customers to include |
| `--output` | str | `reports/retention_action_plan.csv` | Output CSV file path |
| `--model` | str | `models/churn_predictor.pkl` | Path to trained model |
| `--test_data` | str | `data/processed/test.csv` | Path to test dataset |
| `--raw_data` | str | `data/raw/Telco-Customer-Churn.csv` | Path to raw data (for customer IDs) |

## Output Format

The generated CSV report contains the following columns:

### Core Columns
- **customerID**: Unique customer identifier
- **churn_probability**: Model prediction (0-1, higher = more likely to churn)
- **risk_tier**: HIGH (≥70%), MEDIUM (40-70%), or LOW (<40%)

### Recommendation Columns
- **recommended_action**: Specific retention strategy to implement
- **retention_value**: Estimated dollar value of retaining this customer (12-month projection)
- **key_risk_factors**: Top 3 factors contributing to churn risk

### Context Columns
- **Contract**: Contract type (Month-to-month, One year, Two year)
- **tenure**: Customer tenure in months
- **MonthlyCharges**: Current monthly charge amount

## Report Structure

The report is **sorted by churn probability** (descending), so the most at-risk customers appear first. This allows teams to prioritize their retention efforts.

## Summary Statistics

The script automatically displays comprehensive statistics:

### 1. Customer Count by Risk Tier
Shows distribution across HIGH, MEDIUM, and LOW risk categories with percentages.

### 2. Financial Impact
- Total retention value at stake
- Average retention value per customer

### 3. Average Retention Value by Risk Tier
Breakdown of average and total retention value for each risk tier.

### 4. Churn Probability Statistics
- Mean, median, max, and min churn probabilities

### 5. Contract Type Breakdown
Customer counts and average churn probability by contract type.

### 6. Top 5 Most At-Risk Customers
Detailed view of the 5 customers with highest churn probability, including:
- Customer ID
- Churn probability
- Risk tier
- Retention value
- Contract details
- Key risk factors
- Recommended action

## Example Output

```
================================================================================
RETENTION ACTION PLAN SUMMARY
================================================================================

[CUSTOMER COUNT BY RISK TIER]
----------------------------------------
  HIGH    :   100 customers (100.0%)
  MEDIUM  :     0 customers (  0.0%)
  LOW     :     0 customers (  0.0%)

[FINANCIAL IMPACT]
----------------------------------------
  Total Retention Value at Stake: $69,921.67
  Average Retention Value:        $699.22

[TOP 5 MOST AT-RISK CUSTOMERS]
----------------------------------------

  1. Customer 7359-SSBJK
     Churn Probability: 96.2%
     Risk Tier: HIGH
     Retention Value: $1,111.97
     Contract: Two year | Tenure: 64 months | Charges: $70.20/mo
     Key Risk Factors: Has Fiber, Monthly charges, Avg Monthly Spend
     Recommended Action: Personal outreach from account manager with 20%...
```

## Retention Strategies

The script recommends different strategies based on risk tier and customer profile:

### HIGH Risk (≥70% churn probability)
**Channel**: Phone Call  
**Priority**: 1 (Highest)

**Actions**:
- **High-value customers** (>$70/month): 20% discount + account manager + priority support
- **Month-to-month contracts**: 15% discount + contract upgrade offer
- **Standard**: 15% discount + priority support + service review

### MEDIUM Risk (40-70%)
**Channel**: Email Campaign  
**Priority**: 2

**Actions**:
- **New customers**: Satisfaction survey + loyalty program enrollment
- **Month-to-month contracts**: Loyalty benefits + contract upgrade incentive
- **Standard**: Satisfaction survey + exclusive perks

### LOW Risk (<40%)
**Channel**: Newsletter  
**Priority**: 3

**Actions**:
- **New customers**: Onboarding tips + educational content
- **High-value customers**: Premium feature highlights + usage tips
- **Standard**: Product updates + educational content

## Retention Value Calculation

The retention value estimates the financial benefit of keeping a customer for 12 months:

```
retention_value = monthly_charges × 12 × contract_multiplier × tenure_multiplier
```

**Contract Multipliers**:
- Month-to-month: 0.7 (lower retention likelihood)
- One year: 1.0 (standard)
- Two year: 1.2 (higher retention likelihood)

**Tenure Multipliers**:
- < 6 months: 0.8 (very new, risky)
- 6-12 months: 0.9 (new)
- 12-24 months: 1.0 (standard)
- > 24 months: 1.1 (established, higher value)

## Key Risk Factors

The script identifies the top 3 features contributing to each customer's churn risk based on model feature importance. Common factors include:

- **Contract Type**: Month-to-month contracts are higher risk
- **Internet Service**: Fiber optic service patterns
- **Payment Method**: Electronic check vs. automatic payments
- **Tenure**: Short tenure indicates higher risk
- **Monthly Charges**: Higher charges may lead to churn
- **Service Usage**: Lack of multiple services or premium features

## Integration with Business Workflow

### Recommended Process:

1. **Run Script Weekly/Monthly**
   ```bash
   python scripts/generate_retention_report.py --top_n 200
   ```

2. **Review Summary Statistics**
   - Identify trends in churn risk
   - Assess total value at stake
   - Check contract type patterns

3. **Prioritize Actions**
   - Start with HIGH risk customers (Priority 1)
   - Focus on highest retention value customers first
   - Consider intervention cost vs. value

4. **Execute Retention Campaigns**
   - Phone calls for HIGH risk
   - Email campaigns for MEDIUM risk
   - Newsletters for LOW risk

5. **Track Results**
   - Monitor which customers respond
   - Measure actual retention vs. predictions
   - Refine strategies based on outcomes

## Performance

- **Speed**: Processes ~1,400 customers in ~2 seconds
- **Memory**: Lightweight, runs on standard machines
- **Scalability**: Can handle datasets with 10,000+ customers

## Dependencies

Required packages (from `requirements.txt`):
- pandas
- numpy
- scikit-learn
- joblib

## Troubleshooting

### Model Not Found
```
Error: [Errno 2] No such file or directory: 'models/churn_predictor.pkl'
```
**Solution**: Train the model first by running:
```bash
python scripts/run_model_experiments.py
```

### Data File Not Found
```
Error: [Errno 2] No such file or directory: 'data/processed/test.csv'
```
**Solution**: Run the data pipeline first:
```bash
python scripts/test_pipeline.py
```

### Unicode Encoding Issues
If you see encoding errors on Windows, the script has been updated to use ASCII characters only.

## Related Scripts

- `test_pipeline.py`: Prepares data for modeling
- `run_model_experiments.py`: Trains churn prediction models
- `test_models.py`: Evaluates model performance

## Related Modules

- `src/retention_strategy.py`: Core retention logic
- `src/models.py`: Model training utilities
- `src/data_loader.py`: Data loading and cleaning
- `src/feature_engineering.py`: Feature transformation

## Support

For issues or questions:
1. Check that all dependencies are installed: `pip install -r requirements.txt`
2. Verify model and data files exist
3. Review the script output for error messages
4. Check the RETENTION_STRATEGY_SUMMARY.md for module details

## License

Part of the Customer Churn Prediction project.

