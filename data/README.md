# Data Directory

## Structure

```
data/
├── raw/                    # Original, immutable data
│   └── Telco-Customer-Churn.csv
└── processed/              # Cleaned and transformed data
    ├── processed_data.csv  # Feature-engineered dataset
    ├── train.csv          # Training set
    ├── test.csv           # Test set
    └── val.csv            # Validation set
```

## Dataset Description

### Source
**Telco Customer Churn Dataset** from Kaggle

### Features

#### Customer Demographics
- `customerID`: Unique customer identifier
- `gender`: Customer gender (Male/Female)
- `SeniorCitizen`: Whether customer is a senior citizen (1, 0)
- `Partner`: Whether customer has a partner (Yes, No)
- `Dependents`: Whether customer has dependents (Yes, No)

#### Services
- `tenure`: Number of months customer has stayed with company
- `PhoneService`: Whether customer has phone service (Yes, No)
- `MultipleLines`: Whether customer has multiple lines (Yes, No, No phone service)
- `InternetService`: Customer's internet service provider (DSL, Fiber optic, No)
- `OnlineSecurity`: Whether customer has online security (Yes, No, No internet service)
- `OnlineBackup`: Whether customer has online backup (Yes, No, No internet service)
- `DeviceProtection`: Whether customer has device protection (Yes, No, No internet service)
- `TechSupport`: Whether customer has tech support (Yes, No, No internet service)
- `StreamingTV`: Whether customer has streaming TV (Yes, No, No internet service)
- `StreamingMovies`: Whether customer has streaming movies (Yes, No, No internet service)

#### Account Information
- `Contract`: Contract term (Month-to-month, One year, Two year)
- `PaperlessBilling`: Whether customer has paperless billing (Yes, No)
- `PaymentMethod`: Customer's payment method (Electronic check, Mailed check, Bank transfer, Credit card)
- `MonthlyCharges`: Amount charged to customer monthly
- `TotalCharges`: Total amount charged to customer

#### Target Variable
- `Churn`: Whether customer churned (Yes, No)

### Data Quality Notes

TODO: Document any data quality issues discovered during EDA:
- Missing values
- Data type corrections needed
- Outliers
- Inconsistencies

## Usage

```python
from src.data_loader import load_raw_data

# Load raw data
df = load_raw_data('data/raw/Telco-Customer-Churn.csv')
```

## Data Governance

- **Privacy**: No PII (Personally Identifiable Information) beyond customerID
- **Storage**: Data is stored locally and not committed to version control (except raw/)
- **Updates**: Raw data should never be modified; all transformations go to processed/

