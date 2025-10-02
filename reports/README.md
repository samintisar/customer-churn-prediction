# Reports Directory

## Structure

```
reports/
├── figures/                    # Generated plots and visualizations
│   ├── feature_importance.png
│   ├── shap_summary.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── [other generated plots]
├── business_report.md          # Executive business summary
└── README.md                   # This file
```

## Generated Artifacts

### Visualizations (figures/)

TODO: Once implemented, this directory will contain:

1. **Model Performance Plots**
   - ROC curves comparing models
   - Precision-recall curves
   - Confusion matrices
   - Calibration plots

2. **Feature Analysis**
   - Feature importance bar charts
   - SHAP summary plots
   - SHAP waterfall plots for sample predictions
   - Feature correlation heatmaps

3. **Business Insights**
   - Churn rate by customer segment
   - Risk tier distribution
   - Retention value analysis

### Business Report

The `business_report.md` file contains:
- Executive summary of findings
- Key churn drivers and insights
- Model performance in business terms
- Retention strategy recommendations
- ROI analysis
- Implementation roadmap

## Usage

Reports are generated through:
1. Jupyter notebooks (`notebooks/01_eda.ipynb`, `notebooks/02_model_experiments.ipynb`)
2. Python scripts in `src/` modules
3. Dashboard exports from Streamlit app

## Automation

TODO: Set up automated report generation:
```bash
# Run report generation script
python scripts/generate_reports.py
```

