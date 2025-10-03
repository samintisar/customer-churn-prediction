# Customer Churn Prediction & Retention Strategy

A comprehensive end-to-end machine learning solution for predicting customer churn and implementing data-driven retention strategies. Built with Python, scikit-learn, SHAP, and Streamlit.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.846-success.svg)
![Top-Decile](https://img.shields.io/badge/Top--Decile%20Precision-75.2%25-success.svg)

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Business Impact](#-business-impact)
- [Project Structure](#️-project-structure)
- [Quick Start](#-quick-start)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Key Findings & Insights](#-key-findings--insights)
- [Explainability & Interpretability](#-explainability--interpretability)
- [Retention Strategy](#-retention-strategy)
- [Interactive Dashboard](#-interactive-dashboard)
- [Data Pipeline & Feature Engineering](#-data-pipeline--feature-engineering)
- [Configuration](#-configuration)

---

## 📋 Project Overview

This production-ready project delivers a complete customer churn prediction and retention management system:

1. **Predict** customer churn with 84.6% ROC-AUC accuracy
2. **Explain** predictions using SHAP values and feature importance
3. **Recommend** personalized retention actions based on risk tiers
4. **Deliver** insights through an interactive Streamlit dashboard
5. **Optimize** retention campaigns with ROI calculations

### 🎯 Business Impact

**Key Results:**
- **84.6% ROC-AUC** - Excellent model discrimination
- **75.18% Top-Decile Precision** - 75% accuracy on highest-risk 10% of customers
- **80.48% Recall** - Identifies 80% of actual churners
- **$1.67M Annual Revenue at Risk** - Total exposure from 26.54% churn rate
- **$145K-$290K Potential Savings** - With 10-20% churn reduction

**Business Value Delivered:**
- 📊 Identify top 10% at-risk customers with 75% precision
- 💰 Enable proactive retention campaigns with 4-6x ROI
- 🎯 Personalized retention strategies by risk tier
- 📈 Data-driven customer segmentation and targeting
- ⚡ Real-time churn risk scoring for 7,000+ customers

---

## 🏗️ Project Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/                    # Original Telco Customer Churn dataset
│   ├── processed/              # Cleaned and feature-engineered data
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   └── 02_model_experiments.ipynb  # Model training and comparison
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading and splitting
│   ├── feature_engineering.py # Feature creation and preprocessing
│   ├── models.py              # Model training and evaluation
│   ├── explainability.py      # SHAP and feature importance
│   └── retention_strategy.py  # Risk tiers and action recommendations
├── app/
│   └── dashboard.py           # Streamlit interactive dashboard
├── reports/
│   ├── figures/               # Generated plots and visualizations
│   └── business_report.md     # Executive summary and insights
├── models/                     # Saved model artifacts
├── config.py                  # Central configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/samintisar/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Explore the Data
```bash
# Launch Jupyter notebooks
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Train Models
```bash
# Run model training notebook
jupyter notebook notebooks/02_model_experiments.ipynb

# Or use Python scripts (TODO: implement CLI)
# python src/train.py
```

### 5. Launch Dashboard

#### Option A: One-Click Launch (Recommended)
```bash
# For Windows Command Prompt:
run_dashboard.bat

# For PowerShell:
.\run_dashboard.ps1
```

#### Option B: Manual Launch
```bash
# First activate conda environment:
.\activate_env.bat        # Command Prompt
# OR
.\activate_env.ps1        # PowerShell

# Then start dashboard:
streamlit run app/dashboard.py
```

#### Option C: Traditional Method (if conda init is configured)
```bash
# Activate environment
conda activate customer-churn

# Start Streamlit app
streamlit run app/dashboard.py
```

**Note**: If you encounter `CondaError: Run 'conda init' before 'conda activate'`, see [CONDA_SETUP_GUIDE.md](CONDA_SETUP_GUIDE.md) for solutions.

The dashboard will open at `http://localhost:8501` with:
- 📈 **Overview**: Summary metrics and risk distribution
- 🚨 **At-Risk Customers**: Filterable list with retention actions
- 📊 **Model Performance**: Evaluation metrics and ROC curves
- 🔍 **Individual Prediction**: Score new customers in real-time

---

## 📊 Dataset

**Source**: [Kaggle Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

### Features
- **Demographics**: gender, senior citizen status, partner, dependents
- **Services**: phone, internet, online security, tech support, streaming
- **Account**: tenure, contract type, payment method, charges
- **Target**: Churn (Yes/No)

### Statistics
- **Size**: ~7,000 customers
- **Churn Rate**: ~27% (class imbalance)
- **Features**: 20+ original features, expanded through engineering

---

## 🧠 Model Performance

### Selected Model: Logistic Regression ✅
After comprehensive evaluation, **Logistic Regression** was selected as the production model due to:
- **Superior ROC-AUC** (0.846 vs 0.840 for Random Forest)
- **High Interpretability** - Clear coefficient weights for stakeholders
- **Fast Scoring** - Real-time prediction capability
- **Reliability** - Stable performance across validation sets

### Performance Metrics (Test Set - 1,409 Customers)

| Metric | Score | Business Interpretation |
|--------|-------|------------------------|
| **ROC-AUC** | **0.846** | Excellent ability to distinguish churners from non-churners |
| **Precision** | **0.514** | 51% of predicted churners actually churn |
| **Recall** | **0.805** | Identifies 80% of actual churners |
| **F1 Score** | **0.627** | Balanced performance measure |
| **Top-Decile Precision** | **0.752** | **75% accuracy on highest-risk 10%** ⭐ |

### Confusion Matrix Analysis
- **True Negatives**: 750 (correctly identified non-churners)
- **False Positives**: 285 (predicted to churn but didn't - acceptable for proactive campaigns)
- **False Negatives**: 73 (missed churners - only 20% of actual churners)
- **True Positives**: 301 (correctly identified churners)

### Model Comparison

| Metric | Logistic Regression | Random Forest | Winner |
|--------|-------------------|---------------|---------|
| ROC-AUC | **0.846** | 0.840 | ✓ Logistic |
| Precision | 0.514 | **0.548** | Random Forest |
| Recall | **0.805** | 0.727 | ✓ Logistic |
| F1 Score | **0.627** | 0.625 | ✓ Logistic |
| Top-Decile Precision | **0.752** | 0.752 | Tie |
| Training Time | **Fast** | Slower | ✓ Logistic |
| Interpretability | **High** | Medium | ✓ Logistic |

---

## 🔍 Key Findings & Insights

### Top 7 Churn Drivers (Ranked by Impact)

1. **📜 Contract Type** - *Strongest Predictor*
   - Month-to-month: **42.7% churn rate** (15x higher risk)
   - Two-year contracts: **2.8% churn rate**
   - **Action**: Incentivize contract upgrades with discounts

2. **⏰ Customer Tenure** - *Time with Company*
   - <1 year customers: **47.7% churn rate** (5x higher risk)
   - 4+ year customers: **9.5% churn rate**
   - **Action**: Enhanced onboarding and first-year engagement

3. **💰 Monthly Charges** - *Price Sensitivity*
   - Customers paying >$70/month show elevated risk
   - Strong positive correlation with churn
   - **Action**: Value-based pricing and bundled offerings

4. **💳 Payment Method** - *Electronic Check Users*
   - Electronic check users churn significantly more
   - Automatic payments (credit card, bank transfer) = lower churn
   - **Action**: Promote automatic payment enrollment

5. **🌐 Internet Service Type** - *Fiber Optic*
   - Fiber customers churn more than DSL users
   - Indicates competitive market pressures
   - **Action**: Enhanced support and loyalty programs

6. **🛡️ Service Adoption** - *Add-on Services*
   - Fewer services = higher churn risk
   - Each additional service reduces churn probability
   - **Action**: Cross-selling and service adoption campaigns

7. **📄 Paperless Billing**
   - Moderate predictor requiring combination with other factors
   - May indicate digital engagement preferences

### High-Risk Customer Segments 🚨

**Segment 1: New Month-to-Month Customers** (HIGHEST PRIORITY)
- Size: ~880 customers (12.5% of base)
- Churn Rate: ~55-60%
- Annual Revenue at Risk: ~$660,000
- Strategy: Aggressive retention with contract upgrade incentives

**Segment 2: High-Bill Fiber Optic Users**
- Size: ~450 customers (6.4% of base)
- Churn Rate: ~50%
- Annual Revenue at Risk: ~$480,000
- Strategy: Premium support and competitive matching

**Segment 3: Single-Service Customers**
- Size: ~1,200 customers (17% of base)
- Churn Rate: ~40%
- Annual Revenue at Risk: ~$370,000
- Strategy: Service expansion and engagement programs

### Revenue Impact Analysis 💵

**Current State:**
- Total Customers: 7,043
- Annual Churn: 1,869 customers (26.54%)
- Average Monthly Revenue: $64.76
- **Annual Revenue at Risk: $1.67 Million**

**Projected Savings (with Targeted Retention):**

| Churn Reduction | Customers Saved | Annual Revenue | ROI |
|-----------------|-----------------|----------------|-----|
| 5% | 93 | $72,000 | 3-5x |
| 10% | 187 | $145,000 | 4-6x |
| 15% | 280 | $217,000 | 5-7x |
| 20% | 374 | $290,000 | 6-8x |

---

## 🔬 Explainability & Interpretability

### Feature Importance Analysis
- **Top 20 Features** ranked by coefficient magnitude
- Visual bar charts showing relative importance
- Clear understanding of what drives churn predictions

### SHAP (SHapley Additive exPlanations)
- **Individual Predictions**: Waterfall plots showing how each feature contributes
- **Global Patterns**: Beeswarm plots revealing feature impact across all customers
- **Feature Interactions**: Identify combined effects of multiple factors
- **Transparency**: Full explainability for model decisions

### Model Interpretability Benefits
- ✅ Stakeholder trust and buy-in
- ✅ Regulatory compliance (explainable AI)
- ✅ Actionable insights for retention teams
- ✅ Debugging and model improvement
- ✅ Feature engineering validation

---

## 💼 Retention Strategy

### Risk Tiers
| Tier | Probability | Action Priority | Recommended Actions |
|------|------------|----------------|---------------------|
| **HIGH** | ≥70% | Immediate | Personal outreach, 15-20% discount, upgrade offers |
| **MEDIUM** | 40-69% | Proactive | Email campaigns, surveys, loyalty programs |
| **LOW** | <40% | Standard | Newsletters, education content, referral programs |

### Business Impact Estimation
- Calculate customer lifetime value (CLV)
- Estimate retention value per customer
- Prioritize interventions by ROI

---

## 📱 Interactive Dashboard

The **Streamlit dashboard** (`app/dashboard.py`) provides a complete business intelligence interface:

### 📈 Overview Page
- **Summary Metrics**: Total customers, risk tier counts, average churn probability
- **Risk Distribution**: Interactive pie chart with color-coded risk tiers
- **Probability Distribution**: Histogram with threshold markers
- **Top Risk Factors**: Bar chart of the 10 most important features

### 🚨 At-Risk Customers Page  
- **Smart Filtering**: By risk tier, contract type, and customer count
- **Actionable Table**: Churn probability, recommended actions, contact channels
- **Export Functionality**: Download filtered lists as CSV with timestamps
- **Analysis Charts**: Distribution by contract type and tenure

### 📊 Model Performance Page
- **Key Metrics**: ROC-AUC, Precision, Recall, Top-Decile Precision
- **Confusion Matrix**: Interactive heatmap with true/false positives/negatives
- **ROC Curve**: Visual performance comparison with baseline
- **Model Info**: Hyperparameters, feature count, dataset statistics

### 🔍 Individual Prediction Page
- **Input Form**: Enter all customer attributes (demographics, services, billing)
- **Real-Time Scoring**: Instant churn probability calculation
- **Risk Assessment**: Color-coded risk tier (🔴🟠🟢)
- **Retention Plan**: Personalized action, channel, discount, and estimated value

### ⚙️ Sidebar Settings
- **Risk Thresholds**: Customize high/medium/low cutoffs
- **Data Upload**: Score custom CSV files (requires feature engineer)
- **Filters**: Dynamic filtering across all pages

**Documentation**: See `app/README.md` and `app/QUICKSTART.md` for detailed usage guides

---

## � Data Pipeline & Feature Engineering

### Data Processing Summary

**Dataset Statistics:**
- Total Customers: 7,043
- Churn Rate: 26.54%
- Features: 19 original → 46 engineered
- Missing Values: 11 (0.16%) - cleaned and imputed
- Duplicate Rows: 0

**Data Splits (Stratified):**
- Training: 4,929 samples (70%)
- Validation: 705 samples (10%)
- Test: 1,409 samples (20%)

### Feature Engineering Pipeline

The `FeatureEngineer` class creates 27 new features across 4 categories:

**1. Tenure Features** (3 features)
- `tenure_group`: Categorical bins (0-1yr, 1-2yr, 2-4yr, 4yr+)
- `is_new_customer`: Binary flag for customers <6 months
- `tenure_years`: Continuous normalized tenure

**2. Spending Features** (3 features)
- `avg_monthly_spend`: TotalCharges / tenure (handles division by zero)
- `charge_increase`: Binary flag if MonthlyCharges > historical average
- `total_charges_per_month`: Alternative spending metric

**3. Service Features** (6 features)
- `total_services`: Count of active services (0-8)
- `has_multiple_services`: Binary flag for >1 service
- `has_premium_services`: Binary flag for security + backup
- `has_internet`: Binary flag for any internet service
- `has_fiber`: Binary flag for fiber optic

**4. Contract Features** (3 features)
- `contract_stability`: Ordinal encoding (0=month-to-month, 1=1yr, 2=2yr)
- `is_electronic_payment`: Binary flag for electronic payment
- `paperless_binary`: Binary flag for paperless billing

**5. One-Hot Encoding**
- 31 categorical features (drop_first=True to avoid multicollinearity)
- StandardScaler normalization on all numerical features

### Data Quality & Validation ✅
- Zero missing values after processing
- No data leakage (scaler fit only on training data)
- Consistent feature names across all splits
- Comprehensive logging for reproducibility
- Saved artifacts: `models/feature_engineer.pkl`

---

## 📝 Configuration

Edit `config.py` to customize project settings:

```python
# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Risk Tier Thresholds
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.40

# Retention Strategy
RETENTION_MONTHS = 12
DISCOUNT_PERCENTAGES = {
    'HIGH': 20,
    'MEDIUM': 10,
    'LOW': 0
}

# File Paths
DATA_DIR = 'data/'
MODEL_DIR = 'models/'
REPORTS_DIR = 'reports/'
```

---
