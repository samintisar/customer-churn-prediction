# Quick Start Guide - Streamlit Dashboard

## Prerequisites

Ensure you have completed the data pipeline and model training:

```bash
# 1. Run data pipeline (if not already done)
python scripts/test_pipeline.py

# 2. Train models (if not already done)
python scripts/run_model_experiments.py
```

This will create the required files:
- `models/churn_predictor.pkl` - Trained model
- `models/feature_engineer.pkl` - Feature transformation pipeline
- `data/processed/test.csv` - Test dataset
- `data/processed/cleaned_data.csv` - Original customer data

## Installation

Install Streamlit and required dependencies:

```bash
pip install streamlit plotly
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Running the Dashboard

From the project root directory:

```bash
streamlit run app/dashboard.py
```

The dashboard will automatically open in your default browser at:
```
http://localhost:8501
```

## First Time Usage

1. **Wait for Initial Load**: The first time may take a few seconds to load the model and data
2. **Explore the Tabs**:
   - **Overview**: See summary metrics and risk distribution
   - **At-Risk Customers**: Filter and export high-risk customer lists
   - **Model Performance**: Review model accuracy and metrics
   - **Individual Prediction**: Test predictions on new customers

3. **Adjust Settings**: Use the sidebar to:
   - Change risk tier thresholds
   - Upload custom data (requires feature engineer)
   - Apply filters

## Dashboard Pages

### 📈 Overview
- Total customer count and risk tier breakdown
- Interactive pie chart of risk distribution
- Churn probability histogram with threshold markers
- Top 10 most important risk factors

### 🚨 At-Risk Customers
- Filter customers by risk tier and contract type
- View recommended retention actions
- Download customer lists for campaigns
- Analyze patterns by tenure and contract

### 📊 Model Performance
- ROC-AUC, Precision, Recall metrics
- Confusion matrix visualization
- ROC curve
- Top-decile precision (targeting effectiveness)

### 🔍 Individual Prediction
- Enter customer attributes in a form
- Get real-time churn prediction
- See risk tier and retention value
- Receive personalized retention recommendation

## Common Issues

### Port Already in Use

If port 8501 is already in use:
```bash
streamlit run app/dashboard.py --server.port 8502
```

### Model Not Found

If you see "Model not found", run:
```bash
python scripts/run_model_experiments.py
```

### Data Not Found

If you see "Test data not found", run:
```bash
python scripts/test_pipeline.py
```

## Tips

- **Performance**: The dashboard caches model and data for fast performance
- **Custom Data**: To use custom CSV uploads, ensure the feature engineer is trained
- **Thresholds**: Adjust risk thresholds in the sidebar to match your business needs
- **Export**: Download at-risk customer lists directly from the dashboard

## Next Steps

1. **Analyze High-Risk Customers**: Go to "At-Risk Customers" tab and filter for HIGH risk
2. **Review Recommendations**: Check the recommended actions for each customer
3. **Download Campaign List**: Export the filtered list for your retention campaign
4. **Test Individual Predictions**: Use the prediction page to score new customers

## Support

For more details, see:
- `app/README.md` - Full dashboard documentation
- `README.md` - Project overview
- `IMPLEMENTATION_SUMMARY.md` - Technical details

