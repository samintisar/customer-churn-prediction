# Customer Churn Prediction Dashboard

Interactive Streamlit dashboard for analyzing customer churn risk and planning retention strategies.

## Features

### 📈 Overview Page
- **Summary Metrics**: Total customers, high/medium/low risk counts, average churn probability
- **Risk Distribution**: Interactive pie chart showing percentage in each risk tier
- **Churn Probability Distribution**: Histogram with risk tier threshold indicators
- **Top Risk Factors**: Bar chart of top 10 most important features from the model

### 🚨 At-Risk Customers Page
- **Filters**: Filter by risk tier, contract type, and number of customers to display
- **Customer Table**: Sortable table with churn probability, risk tier, and recommended actions
- **Export**: Download filtered customer list as CSV
- **Analysis Charts**: Distribution by contract type and tenure

### 📊 Model Performance Page
- **Performance Metrics**: ROC-AUC, Precision, Recall, Top-Decile Precision
- **Confusion Matrix**: Visual representation of model predictions vs. actual
- **ROC Curve**: Receiver Operating Characteristic curve
- **Model Information**: Model type, hyperparameters, and dataset statistics

### 🔍 Individual Prediction Page
- **Customer Input Form**: Enter customer attributes for real-time prediction
- **Risk Assessment**: Get churn probability and risk tier
- **Retention Recommendations**: Personalized action plan with channel and discount suggestions
- **Retention Value**: Estimated value of retaining the customer

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Required packages:
- `streamlit>=1.25.0`
- `pandas>=1.5.0`
- `numpy>=1.23.0`
- `plotly>=5.11.0`
- `scikit-learn>=1.2.0`

## Usage

### Run the Dashboard

From the project root directory:

```bash
streamlit run app/dashboard.py
```

The dashboard will open in your default browser at `http://localhost:8501`

### Required Files

The dashboard requires the following files to be present:

1. **Trained Model**: `models/churn_predictor.pkl`
   - Main churn prediction model (Random Forest or Logistic Regression)

2. **Test Data**: `data/processed/test.csv`
   - Processed test dataset with features

3. **Original Data** (optional): `data/processed/cleaned_data.csv` or `data/raw/Telco-Customer-Churn.csv`
   - Original customer data for displaying readable attributes

4. **Feature Engineer** (optional): `models/feature_engineer.pkl`
   - Required for individual predictions on custom input

If any files are missing, run the data pipeline and model training first:

```bash
# Run data pipeline
python scripts/test_pipeline.py

# Train models
python scripts/run_model_experiments.py
```

## Dashboard Configuration

### Sidebar Settings

- **Data Source**: Choose between test data or upload custom CSV
- **Risk Tier Thresholds**: Customize high and medium risk probability thresholds
  - Default High Risk: ≥ 70%
  - Default Medium Risk: 40-69%
  - Default Low Risk: < 40%

### Navigation

Use the tabs at the top to switch between different pages:
1. Overview
2. At-Risk Customers
3. Model Performance
4. Individual Prediction

## Technical Details

### Architecture

- **Framework**: Streamlit
- **Visualization**: Plotly for interactive charts
- **Model**: scikit-learn Random Forest or Logistic Regression
- **Feature Engineering**: Custom pipeline with tenure, spending, and service features

### Caching

The dashboard uses Streamlit's caching mechanisms:
- `@st.cache_resource`: For loading model and feature engineer (persists across sessions)
- `@st.cache_data`: For loading datasets (refreshes when data changes)

### Performance

- Initial load may take a few seconds to load model and data
- Interactive elements (filters, sliders) update in real-time
- Predictions on test data (~1400 customers) are computed once and cached

## Customization

### Modify Risk Thresholds

Risk tier thresholds can be adjusted in the sidebar or by modifying `src/retention_strategy.py`:

```python
def classify_risk_tier(churn_probability: float) -> str:
    if churn_probability >= 0.70:  # Adjust threshold
        return "HIGH"
    elif churn_probability >= 0.40:  # Adjust threshold
        return "MEDIUM"
    else:
        return "LOW"
```

### Add Custom Visualizations

Add new charts in the respective render functions:
- `render_overview_page()`: Overview visualizations
- `render_atrisk_customers_page()`: At-risk customer analysis
- `render_model_performance_page()`: Model metrics and evaluation

### Upload Custom Data

To use custom customer data:
1. Select "Upload New Data" in the sidebar
2. Upload a CSV file with customer attributes
3. **Note**: Custom data requires feature engineering - ensure `models/feature_engineer.pkl` exists

## Troubleshooting

### Model Not Found Error

```
❌ Model not found at models/churn_predictor.pkl
```

**Solution**: Run model training:
```bash
python scripts/run_model_experiments.py
```

### Test Data Not Found Error

```
❌ Test data not found at data/processed/test.csv
```

**Solution**: Run data pipeline:
```bash
python scripts/test_pipeline.py
```

### Feature Engineer Not Available

```
⚠️ Feature engineer not found. Cannot process custom input.
```

**Solution**: This only affects individual predictions. The feature engineer is created during model training. Run:
```bash
python scripts/run_model_experiments.py
```

### Port Already in Use

If port 8501 is already in use:
```bash
streamlit run app/dashboard.py --server.port 8502
```

## Examples

### Viewing High-Risk Customers

1. Go to "At-Risk Customers" tab
2. Select "HIGH" in risk tier filter
3. Sort by churn probability
4. Review recommended actions
5. Download CSV for retention campaign

### Making Individual Predictions

1. Go to "Individual Prediction" tab
2. Fill in customer details
3. Click "Predict Churn Risk"
4. Review churn probability and recommended action

### Analyzing Model Performance

1. Go to "Model Performance" tab
2. Review ROC-AUC and precision metrics
3. Examine confusion matrix
4. Check top-decile precision for targeting effectiveness

## Contact & Support

For questions or issues:
- Review the main project README.md
- Check the implementation summaries in the docs folder
- Review model training logs in the console output

## License

This dashboard is part of the Customer Churn Prediction project.

