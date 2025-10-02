# Quick Start: Model Experiments

## Prerequisites

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run data pipeline** (if not done already):
   ```bash
   python scripts/test_pipeline.py
   ```

## Running the Experiments

### Option 1: Quick Test (5 seconds)
Verify the implementation works:

```bash
python scripts/test_models.py
```

**Output**: Trains both models without tuning, shows basic metrics.

### Option 2: Full Pipeline (5-10 minutes)
Complete model training with hyperparameter tuning:

```bash
python scripts/run_model_experiments.py
```

**Output**:
- 3 saved models in `models/`
- 3 visualization plots in `reports/figures/`
- Comprehensive console output with all metrics

### Option 3: Interactive Notebook
For step-by-step exploration:

```bash
cd notebooks
jupyter notebook
# Open 02_model_experiments.ipynb
# Run cells sequentially
```

## What Gets Generated

After running the experiments, you'll have:

### Models (`models/`)
- ✅ `churn_predictor.pkl` - Best model (ready for deployment)
- ✅ `logistic_regression.pkl` - Baseline model
- ✅ `random_forest.pkl` - Advanced model

### Visualizations (`reports/figures/`)
- ✅ `model_comparison.png` - Performance comparison chart
- ✅ `feature_importance.png` - Top 20 important features
- ✅ `shap_summary.png` - SHAP explainability plot

### Console Output
- Model training progress
- Performance metrics (ROC-AUC, Precision, Recall, F1, Top-Decile)
- Confusion matrices
- Model comparison table
- Key findings summary

## Expected Results

### Performance Metrics

| Metric | Logistic Regression | Random Forest | Improvement |
|--------|---------------------|---------------|-------------|
| ROC-AUC | ~0.82-0.84 | ~0.84-0.86 | +2-5% |
| Precision | ~0.60-0.65 | ~0.65-0.70 | +5-10% |
| Recall | ~0.55-0.60 | ~0.58-0.63 | +3-5% |
| Top-Decile Precision | ~0.60-0.65 | ~0.65-0.75 | +5-15% |

### Top Features

1. **Contract_Month-to-month** - Strongest predictor of churn
2. **tenure** - Customer lifetime
3. **InternetService_Fiber optic** - Service type
4. **MonthlyCharges** - Pricing
5. **PaymentMethod_Electronic check** - Payment behavior

## Next Steps

1. **Review Results**:
   ```bash
   # Check generated visualizations
   ls reports/figures/
   
   # View feature importance
   open reports/figures/feature_importance.png  # macOS
   # or
   start reports/figures/feature_importance.png  # Windows
   ```

2. **Use the Model**:
   ```python
   from src.models import load_model
   
   # Load best model
   model = load_model('models/churn_predictor.pkl')
   
   # Make predictions
   predictions = model.predict_proba(X_new)[:, 1]
   ```

3. **Deploy Dashboard**:
   ```bash
   streamlit run app/dashboard.py
   ```

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

**Solution**: Make sure you're running from project root:
```bash
cd /path/to/customer-churn-prediction
python scripts/run_model_experiments.py
```

### "FileNotFoundError: data/processed/train.csv"

**Solution**: Run the data pipeline first:
```bash
python scripts/test_pipeline.py
```

### SHAP taking too long

**Solution**: It's normal for SHAP to take 1-2 minutes. It automatically samples to max 500 rows for speed.

### GridSearchCV taking too long

**Solution**: For faster results, edit `src/models.py` and reduce the parameter grid, or run with:
```python
rf_model = train_random_forest(X_train, y_train, tune_hyperparams=False)
```

## Command Cheat Sheet

```bash
# Quick test (5 sec)
python scripts/test_models.py

# Full pipeline (5-10 min)
python scripts/run_model_experiments.py

# Interactive notebook
cd notebooks && jupyter notebook

# Check outputs
ls models/
ls reports/figures/

# View specific results
cat IMPLEMENTATION_SUMMARY.md
```

## Files Implemented

```
✅ src/models.py                     - All training & evaluation functions
✅ src/explainability.py             - All explainability functions  
✅ scripts/run_model_experiments.py  - Complete automation script
✅ scripts/test_models.py            - Quick verification test
✅ notebooks/README_MODEL_EXPERIMENTS.md - Detailed documentation
✅ IMPLEMENTATION_SUMMARY.md         - Full implementation overview
✅ QUICKSTART_MODEL_EXPERIMENTS.md   - This quick start guide
```

## Support

For detailed documentation, see:
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Notebook Guide**: `notebooks/README_MODEL_EXPERIMENTS.md`
- **Source Code**: `src/models.py`, `src/explainability.py`

---

**Ready to go!** Run `python scripts/run_model_experiments.py` to get started.

