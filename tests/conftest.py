"""
Pytest Configuration and Fixtures

This module contains shared fixtures and configuration for all tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return the data directory path."""
    return project_root / "data"


@pytest.fixture(scope="session")
def raw_data_path(data_dir):
    """Return the path to the raw data file."""
    return data_dir / "raw" / "Telco-Customer-Churn.csv"


@pytest.fixture(scope="session")
def models_dir(project_root):
    """Return the models directory path."""
    return project_root / "models"


@pytest.fixture
def sample_customer_data():
    """
    Create a sample customer record for testing.
    """
    return {
        'customerID': '1234-ABCD',
        'gender': 'Female',
        'SeniorCitizen': 'No',
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.35,
        'TotalCharges': 844.20,
        'Churn': 'No'
    }


@pytest.fixture
def sample_dataframe(sample_customer_data):
    """
    Create a sample DataFrame with multiple customers.
    """
    # Create 100 sample customers with variations
    np.random.seed(42)
    data = []
    
    for i in range(100):
        customer = sample_customer_data.copy()
        customer['customerID'] = f'CUST-{i:04d}'
        customer['tenure'] = np.random.randint(1, 72)
        customer['MonthlyCharges'] = np.random.uniform(20, 120)
        customer['TotalCharges'] = customer['MonthlyCharges'] * customer['tenure']
        customer['Churn'] = np.random.choice(['Yes', 'No'], p=[0.27, 0.73])
        data.append(customer)
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_features_dataframe():
    """
    Create a sample DataFrame with features only (no target).
    """
    np.random.seed(42)
    data = []
    
    for i in range(50):
        customer = {
            'gender': np.random.choice(['Female', 'Male']),
            'SeniorCitizen': np.random.choice(['Yes', 'No']),
            'Partner': np.random.choice(['Yes', 'No']),
            'Dependents': np.random.choice(['Yes', 'No']),
            'tenure': np.random.randint(1, 72),
            'PhoneService': np.random.choice(['Yes', 'No']),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service']),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No']),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service']),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service']),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service']),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service']),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service']),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service']),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year']),
            'PaperlessBilling': np.random.choice(['Yes', 'No']),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ]),
            'MonthlyCharges': np.random.uniform(20, 120),
        }
        customer['TotalCharges'] = customer['MonthlyCharges'] * customer['tenure']
        data.append(customer)
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test files.
    Automatically cleaned up after test.
    """
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_model():
    """
    Create a mock model for testing predictions.
    """
    from sklearn.linear_model import LogisticRegression
    
    # Create a simple mock model
    model = LogisticRegression()
    
    # Create some dummy training data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    model.fit(X, y)
    
    return model
