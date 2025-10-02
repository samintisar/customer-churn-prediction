# At-Risk Customers Page - User Guide

## Overview

The **At-Risk Customers** page is the operational hub for retention teams, providing advanced filtering, sorting, and detailed customer analysis with SHAP explanations.

## Features

### 🎛️ Advanced Filters (Sidebar)

#### 1. Risk Tier Filter (Multiselect)
- **Options**: HIGH, MEDIUM, LOW
- **Default**: HIGH and MEDIUM selected
- **Use Case**: Focus on customers that need immediate attention

#### 2. Minimum Churn Probability Slider
- **Range**: 0% - 100%
- **Default**: 40%
- **Step**: 5%
- **Use Case**: Set a threshold for customer inclusion

#### 3. Contract Type Filter (Multiselect)
- **Options**: Month-to-month, One year, Two year
- **Default**: All selected
- **Use Case**: Target specific contract segments

#### 4. Tenure Range Filter (Slider)
- **Range**: Dynamic (based on dataset)
- **Default**: Full range
- **Use Case**: Focus on new customers or long-term relationships

### 🔄 Sort Options

#### Sort By Options:
1. **Churn Probability** (Default) - Highest risk first
2. **Retention Value** - Highest value customers first
3. **Tenure** - Newest or oldest customers
4. **Monthly Charges** - Highest paying customers first

#### Sort Order:
- **Descending** (Default) - High to low
- **Ascending** - Low to high

### 👁️ Display Options

**Show Top N Customers:**
- 10 customers
- 25 customers
- 50 customers (Default)
- 100 customers
- All customers

## Page Components

### 📊 Summary Metrics

Four key metrics displayed at the top:

1. **Customers Found** - Total matching your filters
2. **Avg Churn Probability** - Mean churn risk of filtered customers
3. **Total Retention Value** - Combined 12-month retention value
4. **High Risk Count** - Number of HIGH risk customers

### 📋 Customer List Table

**Columns Displayed:**
- **Customer ID** - Unique identifier
- **Churn Probability** - Color-coded gradient:
  - 🔴 Red (70%+): Highest risk
  - 🟠 Orange (50-69%): High risk
  - 🟡 Yellow (30-49%): Medium risk
  - 🟢 Green (<30%): Lower risk
  
- **Risk Tier** - Badge-style display (HIGH/MEDIUM/LOW)
- **Contract Type** - Month-to-month, One year, or Two year
- **Tenure** - Months as customer
- **Monthly Charges** - Current monthly bill
- **Recommended Action** - Personalized retention strategy
- **Retention Value** - Estimated 12-month value

**Features:**
- Color-coded cells for quick visual scanning
- Formatted currency and percentages
- Scrollable with 400px height
- Full-width responsive design

### 🔍 Customer Detail Expanders

**Top 10 customers** from filtered list have expandable detail views:

#### Expander Header:
```
🔴 Customer-ID - 85.2% Churn Risk - HIGH Priority
```

#### Inside Each Expander:

**1. Risk Profile**
- Churn Probability
- Risk Tier
- Action Priority (1-3)

**2. Account Information**
- Tenure (months)
- Contract Type
- Monthly Charges

**3. Retention Value**
- Estimated Value (12 months)
- Recommended Discount %
- Contact Channel

**4. Recommended Action**
- Detailed retention strategy
- Displayed in green success box

**5. All Features Table**
- Top 20 features for this customer
- Feature name and value
- Scrollable table

**6. SHAP Explanation** 🔬
- **Top 10 Contributing Factors**
- Each factor shows:
  - 🔴 Red emoji: Increases churn risk
  - 🟢 Green emoji: Decreases churn risk
  - Feature name
  - SHAP value (numeric impact)
  - Impact description

**Example SHAP Output:**
```
🔴 tenure: -0.245 (Increases Risk)
🔴 Contract_Month-to-month: +0.189 (Increases Risk)
🟢 Contract_Two year: -0.156 (Decreases Risk)
🟢 OnlineSecurity_Yes: -0.098 (Decreases Risk)
```

### 📥 Export Options

Two download buttons for different use cases:

#### 1. Download Filtered Customer List
- **File**: `at_risk_customers_YYYYMMDD_HHMMSS.csv`
- **Contains**: ALL columns including engineered features
- **Use Case**: Complete data for analysis

#### 2. Download Action Plan
- **File**: `retention_action_plan_YYYYMMDD_HHMMSS.csv`
- **Contains**: Essential campaign fields only
  - Customer ID
  - Churn Probability
  - Risk Tier
  - Contact Channel
  - Recommended Action
  - Discount %
  - Retention Value
- **Use Case**: Direct input for CRM campaigns

### 📊 Analysis Charts

#### 1. Customers by Contract Type (Left)
- **Type**: Grouped histogram
- **Groups**: Risk tiers (color-coded)
- **X-axis**: Contract types
- **Y-axis**: Count
- **Use Case**: Identify which contract types are most at-risk

#### 2. Churn Risk vs Retention Value (Right)
- **Type**: Scatter plot
- **X-axis**: Churn Probability
- **Y-axis**: Retention Value ($)
- **Color**: Risk tier
- **Size**: Monthly Charges (if available)
- **Hover**: Customer ID
- **Use Case**: Prioritize high-value, high-risk customers

## Usage Scenarios

### Scenario 1: Daily High-Risk Review

**Goal**: Review and action on highest risk customers

**Steps:**
1. Set filters:
   - Risk Tier: **HIGH only**
   - Min Probability: **70%**
   - Sort by: **Churn Probability**
   - Show: **25 customers**

2. Review customer list
3. Expand top 5 customers for details
4. Download Action Plan
5. Upload to CRM for outreach

### Scenario 2: Month-to-Month Contract Campaign

**Goal**: Target month-to-month customers for contract upgrades

**Steps:**
1. Set filters:
   - Contract Type: **Month-to-month only**
   - Risk Tier: **HIGH, MEDIUM**
   - Sort by: **Retention Value**
   - Show: **50 customers**

2. Analyze contract chart
3. Review recommendations (should suggest contract upgrades)
4. Download Action Plan
5. Create targeted email campaign

### Scenario 3: New Customer Onboarding

**Goal**: Prevent early churn (first 6 months)

**Steps:**
1. Set filters:
   - Tenure Range: **0-6 months**
   - Min Probability: **50%**
   - Sort by: **Churn Probability**

2. Review patterns in new customers
3. Expand customers to see SHAP factors
4. Identify common issues (e.g., lack of services)
5. Create onboarding improvement plan

### Scenario 4: High-Value Retention

**Goal**: Focus on most valuable at-risk customers

**Steps:**
1. Set filters:
   - Sort by: **Retention Value**
   - Sort Order: **Descending**
   - Show: **All**

2. Use scatter plot to identify high-value + high-risk quadrant
3. Expand top customers
4. Review SHAP explanations for patterns
5. Create VIP retention program

## Color Coding Guide

### Churn Probability Gradient
- **#ff4444** (Bright Red): 70%+ probability
- **#ff6666** (Red): 60-69% probability
- **#ff9944** (Orange): 50-59% probability
- **#ffbb44** (Yellow-Orange): 40-49% probability
- **#ffdd77** (Yellow): 30-39% probability
- **#88ff88** (Green): <30% probability

### Risk Tier Badges
- **HIGH**: Red badge, white text, bold
- **MEDIUM**: Orange badge, white text, bold
- **LOW**: Green badge, black text, bold

## Tips & Best Practices

### 💡 Filtering Tips

1. **Start Broad, Then Narrow**
   - Begin with default filters
   - Review summary metrics
   - Adjust filters based on insights

2. **Use Probability Slider Strategically**
   - 70%+: Immediate intervention needed
   - 50-70%: Proactive campaigns
   - 40-50%: Monitoring and engagement

3. **Combine Multiple Filters**
   - Example: High risk + New customers + Month-to-month
   - Creates highly targeted segments

### 🎯 Action Planning Tips

1. **Prioritize by Risk × Value**
   - Use scatter plot to find "sweet spot"
   - Focus on high-value, high-risk customers first

2. **Review SHAP Explanations**
   - Understand WHY customers are at risk
   - Tailor actions to specific issues
   - Example: If "No OnlineSecurity" is a factor, offer security package

3. **Export for Different Purposes**
   - **Full List**: Data analysis, reporting
   - **Action Plan**: CRM upload, campaign execution

### 📊 Chart Interpretation

**Contract Type Chart:**
- If month-to-month shows mostly HIGH risk → Focus on contract conversions
- If two-year shows HIGH risk → Service quality issues

**Scatter Plot:**
- **Top Right Quadrant** (High risk, High value): Priority 1
- **Bottom Right** (Low risk, High value): Maintain & upsell
- **Top Left** (High risk, Low value): Consider divestment vs. simple retention

## Performance Notes

- **SHAP calculations** may take 2-3 seconds per customer
- **Top 10 customers** only show expanders (for performance)
- **Caching** speeds up repeated SHAP requests
- **Large datasets** (1000+ customers): Use filters to reduce display count

## Troubleshooting

### SHAP Explanations Not Showing

**Possible Causes:**
1. Model is not Random Forest (SHAP only for tree models)
2. `shap` library not installed
3. Model/data not in session state

**Solutions:**
- Ensure Random Forest model is loaded
- Run: `pip install shap`
- Reload the dashboard

### No Customers Found

**Cause:** Filters are too restrictive

**Solution:**
- Reset filters to defaults
- Gradually add filters one at a time
- Check tenure range hasn't excluded all customers

### Export Files Empty

**Cause:** Browser blocking downloads

**Solution:**
- Check browser download permissions
- Try different browser
- Ensure CSV generated successfully (check for errors)

## Advanced Features

### Custom Segmentation

Create custom customer segments by combining filters:

**Example Segments:**

1. **Flight Risk VIPs**
   - Risk: HIGH
   - Monthly Charges: Sort descending
   - Limit: Top 25

2. **New Customer Struggling**
   - Tenure: 0-3 months
   - Probability: 60%+
   - Contract: Month-to-month

3. **Long-Term At-Risk**
   - Tenure: 36+ months
   - Risk: HIGH, MEDIUM
   - Sort by: Retention Value

### SHAP-Driven Insights

Use SHAP values to:
1. **Identify Common Issues** - Look for recurring factors across customers
2. **Personalize Outreach** - Address specific concerns
3. **Product Development** - See which features prevent churn
4. **Pricing Strategy** - Understand impact of charges

## Summary

The At-Risk Customers page provides:

✅ **Advanced filtering** for precise targeting  
✅ **Multiple sorting options** for prioritization  
✅ **Color-coded visualization** for quick assessment  
✅ **Detailed customer profiles** with SHAP explanations  
✅ **Multiple export formats** for different workflows  
✅ **Interactive charts** for pattern recognition  

Use this page to drive data-driven retention campaigns and reduce customer churn effectively.

