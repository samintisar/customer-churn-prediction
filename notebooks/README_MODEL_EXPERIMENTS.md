# Model Experiments Notebook Guide

## Overview

The `02_model_experiments.ipynb` notebook implements a complete machine learning pipeline for customer churn prediction, comparing a baseline Logistic Regression model against an advanced Random Forest classifier.

## Implementation Status

✅ **FULLY IMPLEMENTED** - All TODO sections have been completed.

The notebook has been implemented in two formats:
1. **Jupyter Notebook**: `02_model_experiments.ipynb` (interactive)
2. **Python Script**: `../scripts/run_model_experiments.py` (automated)

## Running the Notebook

### Option 1: Using Jupyter Notebook (Interactive)

```bash
# Navigate to notebooks directory
cd notebooks

# Start Jupyter
jupyter notebook

# Open 02_model_experiments.ipynb and run all cells
```

### Option 2: Using the Python Script (Automated)

```bash
# From project root
python scripts/run_model_experiments.py
```

This script executes all the steps from the notebook in sequence.

## Notebook Sections

### 1. Data Preparation
- Loads pre-processed train/val/test splits
- Displays data shapes and target distribution
- Prepares feature names for model training

### 2. Baseline Model: Logistic Regression
- Trains baseline logistic regression with balanced class weights
- Evaluates on test set with comprehensive metrics
- ROC-AUC, Precision, Recall, F1-Score, Top-Decile Precision

### 3. Advanced Model: Random Forest
- Trains Random Forest with hyperparameter tuning (GridSearchCV)
- Optimizes for ROC-AUC score
- Evaluates on test set with same metrics as baseline

### 4. Model Comparison
- Creates comparison table showing all metrics side-by-side
- Calculates improvement percentage
- Generates visualization charts:
  - Bar plot comparing model performance
  - Improvement chart showing RF gains over baseline

### 5. Feature Importance and Explainability
- **Feature Importance**: Extracts and plots top 20 most important features
- **SHAP Analysis**: Generates SHAP summary plot showing:
  - Feature impact on predictions
  - Direction of influence (positive/negative)
  - Feature value distribution

### 6. Model Selection and Saving
- Selects best model based on ROC-AUC score
- Saves all trained models to `models/` directory:
  - `churn_predictor.pkl` (best model)
  - `logistic_regression.pkl`
  - `random_forest.pkl`

## Expected Outputs

After running the notebook, you will have:

### Models (in `models/`)
- `churn_predictor.pkl` - Best performing model
- `logistic_regression.pkl` - Baseline model
- `random_forest.pkl` - Advanced model
- `feature_engineer.pkl` - Feature transformation pipeline (from data pipeline)

### Visualizations (in `reports/figures/`)
- `model_comparison.png` - Side-by-side model performance comparison
- `feature_importance.png` - Top 20 most important features
- `shap_summary.png` - SHAP values showing feature impact

### Console Output
- Detailed metrics for both models
- Confusion matrices
- Model selection reasoning
- Key findings summary

## Implementation Details

### Source Modules Used

All core functionality is implemented in the `src/` modules:

```python
from src.data_loader import load_raw_data, clean_data, split_data
from src.feature_engineering import FeatureEngineer
from src.models import (
    train_baseline_model, 
    train_random_forest, 
    evaluate_model, 
    save_model
)
from src.explainability import (
    plot_feature_importance, 
    generate_shap_summary
)
```

### Key Functions Implemented

**models.py**:
- `train_baseline_model()` - Trains LogisticRegression with balanced weights
- `train_random_forest()` - Trains RandomForest with GridSearchCV
- `evaluate_model()` - Calculates ROC-AUC, Precision, Recall, F1, Top-Decile Precision
- `calculate_top_decile_precision()` - Business-critical metric for retention targeting
- `save_model()` / `load_model()` - Model persistence

**explainability.py**:
- `get_feature_importance()` - Extracts importance from tree/linear models
- `plot_feature_importance()` - Creates horizontal bar charts
- `generate_shap_summary()` - SHAP summary plots with TreeExplainer
- `create_explainability_report()` - Comprehensive report generation

## Expected Performance

Based on the Telco Customer Churn dataset, you should see:

**Logistic Regression (Baseline)**:
- ROC-AUC: ~0.82-0.84
- Precision: ~0.60-0.65
- Recall: ~0.55-0.60
- Top-Decile Precision: ~0.60-0.65

**Random Forest (Advanced)**:
- ROC-AUC: ~0.84-0.86
- Precision: ~0.65-0.70
- Recall: ~0.58-0.63
- Top-Decile Precision: ~0.65-0.75

**Improvement**: Random Forest typically shows 2-10% improvement across all metrics.

## Key Findings

### Top Churn Predictors
1. **Contract Type** - Month-to-month contracts = highest churn
2. **Tenure** - New customers (< 6 months) at greatest risk
3. **Internet Service Type** - Fiber optic customers churn more
4. **Monthly Charges** - Higher prices correlate with churn
5. **Payment Method** - Electronic check users more likely to churn

### Business Value
- **Top-Decile Precision of 65-75%** means:
  - For every 100 high-risk customers identified, 65-75 will actually churn
  - Highly efficient targeting for retention campaigns
  - Minimize wasted retention spending on loyal customers

### Actionable Insights
1. **Contract Incentives**: Offer discounts for 1-year or 2-year contracts
2. **Onboarding**: Extra support for customers in first 6 months
3. **Fiber Optic**: Investigate service quality issues causing churn
4. **Payment Options**: Encourage automatic credit card payments
5. **Pricing**: Review pricing strategy for high monthly charge segments

## Troubleshooting

### SHAP Taking Too Long
The SHAP calculation samples 500 rows maximum. If still slow:
- Reduce `max_display` parameter in `generate_shap_summary()`
- Use smaller test set sample

### Memory Issues
If running out of memory during GridSearchCV:
- Set `tune_hyperparams=False` in `train_random_forest()`
- Reduce `param_grid` options in `src/models.py`

### Module Import Errors
Ensure you're in the `notebooks/` directory and `sys.path.append()` is working:
```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
```

## Next Steps

After completing this notebook:

1. **Review Results**: Analyze model performance and feature importance
2. **Business Report**: Document findings for stakeholders
3. **Dashboard Deployment**: Use saved model in Streamlit dashboard
4. **A/B Testing**: Test retention strategies on high-risk customers
5. **Monitoring**: Set up model performance tracking
6. **Retraining**: Schedule quarterly model updates

## Additional Resources

- **Data Pipeline**: See `DATA_PIPELINE_SUMMARY.md` for preprocessing details
- **EDA Notebook**: See `01_eda.ipynb` for exploratory analysis
- **Dashboard**: See `app/dashboard.py` for deployment interface
- **Business Report**: See `reports/business_report.md` for stakeholder summary

