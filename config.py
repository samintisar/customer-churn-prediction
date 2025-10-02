"""
Configuration Module

Central configuration for the customer churn prediction project.

Contains:
- File paths
- Model hyperparameters
- Feature lists
- Risk tier thresholds
- Evaluation metrics settings
"""

from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data files
RAW_DATA_FILE = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_data.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

# Model files
BASELINE_MODEL_FILE = MODELS_DIR / "logistic_regression.pkl"
RF_MODEL_FILE = MODELS_DIR / "random_forest.pkl"
BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"


# ============================================================================
# Model Configuration
# ============================================================================

# Random state for reproducibility
RANDOM_STATE = 42

# Train/validation/test split ratios
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # 10% of training data

# Baseline model (Logistic Regression) parameters
BASELINE_PARAMS = {
    'random_state': RANDOM_STATE,
    'max_iter': 1000,
    'solver': 'lbfgs',
    'class_weight': 'balanced'  # Handle class imbalance
}

# Random Forest hyperparameter search space
RF_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

# Default Random Forest parameters (if not tuning)
RF_DEFAULT_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'n_jobs': -1
}


# ============================================================================
# Feature Configuration
# ============================================================================

# Target variable
TARGET_COL = 'Churn'

# Columns to drop (non-predictive)
DROP_COLS = ['customerID']

# Numerical features
NUMERICAL_FEATURES = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

# Categorical features
CATEGORICAL_FEATURES = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod'
]


# ============================================================================
# Risk Tier Thresholds
# ============================================================================

RISK_TIERS = {
    'HIGH': 0.70,      # >= 70% churn probability
    'MEDIUM': 0.40,    # 40-69% churn probability
    'LOW': 0.0         # < 40% churn probability
}


# ============================================================================
# Evaluation Configuration
# ============================================================================

# Classification threshold
DEFAULT_THRESHOLD = 0.5

# Top-decile for precision calculation (for retention prioritization)
TOP_DECILE = 10  # Top 10% highest risk

# Cross-validation folds
CV_FOLDS = 5


# ============================================================================
# Retention Strategy Configuration
# ============================================================================

RETENTION_ACTIONS = {
    'HIGH': {
        'priority': 1,
        'actions': [
            'Immediate account manager outreach',
            'Personalized discount offer (15-20%)',
            'Premium service upgrade trial',
            'Loyalty program fast-track enrollment'
        ],
        'channel': 'Phone call + Email'
    },
    'MEDIUM': {
        'priority': 2,
        'actions': [
            'Targeted email campaign',
            'Customer satisfaction survey',
            'Service optimization recommendations',
            'Loyalty rewards reminder'
        ],
        'channel': 'Email + SMS'
    },
    'LOW': {
        'priority': 3,
        'actions': [
            'Standard newsletter',
            'Product education content',
            'Community engagement invitation',
            'Referral program promotion'
        ],
        'channel': 'Email'
    }
}

# Customer lifetime value estimation (months)
RETENTION_PERIOD = 12


# ============================================================================
# Visualization Configuration
# ============================================================================

# Figure size for plots
FIGURE_SIZE = (10, 6)

# Color palette
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#ff9800',
    'danger': '#d62728',
    'churn': '#e74c3c',
    'no_churn': '#3498db'
}

# SHAP plot settings
SHAP_SAMPLE_SIZE = 100  # Number of samples for SHAP calculation (for speed)


# ============================================================================
# Dashboard Configuration
# ============================================================================

DASHBOARD_TITLE = "Customer Churn Prediction & Retention Dashboard"
TOP_N_ATRISK = 100  # Default number of top at-risk customers to display

