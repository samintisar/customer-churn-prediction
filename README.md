# Customer Churn Prediction & Retention Strategy

A comprehensive machine learning project for predicting customer churn and recommending targeted retention strategies. Built with Python, scikit-learn, SHAP, and Streamlit.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)

---

## 📋 Project Overview

This project aims to:
1. **Predict** which customers are at high risk of churning
2. **Explain** the key factors driving churn using interpretable ML
3. **Recommend** targeted retention actions based on risk tiers
4. **Deliver** actionable insights through an interactive dashboard

### Business Value
- Identify top 10% at-risk customers with high precision
- Enable proactive retention campaigns
- Estimate ROI of retention interventions
- Provide data-driven customer segmentation

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
git clone https://github.com/yourusername/customer-churn-prediction.git
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
```bash
# Validate files exist
python validate_dashboard_files.py

# Start Streamlit app
streamlit run app/dashboard.py
```

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

## 🧠 Models

### Baseline: Logistic Regression
- Fast, interpretable linear model
- Handles class imbalance with balanced weights
- Strong baseline for binary classification

### Advanced: Random Forest
- Ensemble method with hyperparameter tuning
- Captures non-linear relationships
- Provides feature importance rankings

### Evaluation Metrics
- **ROC-AUC**: Overall model discrimination
- **Precision/Recall**: Class-specific performance
- **Top-Decile Precision**: Accuracy for highest-risk 10% (key business metric)
- **Confusion Matrix**: Classification breakdown

---

## 🔍 Explainability

### Feature Importance
- Identify which features most influence churn predictions
- Bar charts showing top predictors

### SHAP Values
- Explain individual predictions
- Show how each feature contributes to a customer's risk score
- Generate summary plots across the dataset

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

## 🛠️ Development Roadmap

### Phase 1: Foundation ✅
- [x] Project scaffolding
- [ ] Data loading and EDA
- [ ] Feature engineering pipeline
- [ ] Baseline model training

### Phase 2: Core ML Pipeline
- [ ] Random Forest training and tuning
- [ ] Model evaluation and comparison
- [ ] SHAP explainability implementation
- [ ] Model persistence

### Phase 3: Business Logic
- [ ] Risk tier classification
- [ ] Retention action mapping
- [ ] ROI calculation
- [ ] Business report generation

### Phase 4: Dashboard
- [ ] Streamlit app implementation
- [ ] Interactive visualizations
- [ ] Filter and search functionality
- [ ] Export capabilities

### Phase 5: Polish
- [ ] Code documentation and tests
- [ ] Performance optimization
- [ ] Deployment guide
- [ ] User documentation

---

## 📈 Results

**TODO**: Once models are trained, populate with actual results:
- Best model performance
- Top churn predictors
- Risk distribution
- Expected retention value

---

## 🧪 Testing

```bash
# Run unit tests (TODO: implement)
pytest tests/

# Run code quality checks
flake8 src/
black src/ --check
```

---

## 📝 Configuration

Edit `config.py` to customize:
- Model hyperparameters
- Risk tier thresholds
- Retention action mappings
- File paths
- Visualization settings

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- **Your Name** - Initial work

---

## 🙏 Acknowledgments

- Dataset provided by [IBM Sample Data Sets](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)
- Inspired by real-world customer retention challenges
- Built with open-source tools from the Python data science community

---

## 📞 Contact

For questions or feedback:
- GitHub Issues: [Create an issue](https://github.com/yourusername/customer-churn-prediction/issues)
- Email: your.email@example.com

---

**Status**: 🚧 In Development - Scaffold Complete, Implementation in Progress
