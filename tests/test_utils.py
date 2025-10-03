"""
Utility Functions for Tests

Helper functions and utilities used across multiple test modules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


def create_sample_customer(
    customer_id: str = "TEST-001",
    churn_prob: float = 0.5,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a sample customer dictionary with default values.
    
    Args:
        customer_id: Customer ID
        churn_prob: Churn probability (for testing predictions)
        **kwargs: Additional customer attributes to override defaults
    
    Returns:
        Dictionary with customer data
    """
    customer = {
        'customerID': customer_id,
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
    
    # Override with any provided kwargs
    customer.update(kwargs)
    
    return customer


def create_sample_customers(n: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Create a list of sample customers with random variations.
    
    Args:
        n: Number of customers to create
        seed: Random seed for reproducibility
    
    Returns:
        List of customer dictionaries
    """
    np.random.seed(seed)
    customers = []
    
    for i in range(n):
        customer = create_sample_customer(
            customer_id=f"TEST-{i:04d}",
            tenure=np.random.randint(1, 72),
            MonthlyCharges=np.random.uniform(20, 120),
            Contract=np.random.choice(['Month-to-month', 'One year', 'Two year']),
            Churn=np.random.choice(['Yes', 'No'], p=[0.27, 0.73])
        )
        customer['TotalCharges'] = customer['MonthlyCharges'] * customer['tenure']
        customers.append(customer)
    
    return customers


def assert_valid_probability(value: float, name: str = "probability") -> None:
    """
    Assert that a value is a valid probability (between 0 and 1).
    
    Args:
        value: Value to check
        name: Name of the value for error message
    
    Raises:
        AssertionError: If value is not a valid probability
    """
    assert isinstance(value, (int, float)), f"{name} must be numeric"
    assert 0 <= value <= 1, f"{name} must be between 0 and 1, got {value}"


def assert_dataframe_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    allow_extra: bool = True
) -> None:
    """
    Assert that a DataFrame has the required columns.
    
    Args:
        df: DataFrame to check
        required_columns: List of required column names
        allow_extra: Whether to allow extra columns
    
    Raises:
        AssertionError: If required columns are missing
    """
    missing_cols = set(required_columns) - set(df.columns)
    assert not missing_cols, f"Missing required columns: {missing_cols}"
    
    if not allow_extra:
        extra_cols = set(df.columns) - set(required_columns)
        assert not extra_cols, f"Found unexpected columns: {extra_cols}"


def assert_no_missing_values(df: pd.DataFrame, columns: List[str] = None) -> None:
    """
    Assert that DataFrame has no missing values.
    
    Args:
        df: DataFrame to check
        columns: Specific columns to check (None = check all)
    
    Raises:
        AssertionError: If missing values are found
    """
    if columns:
        df_check = df[columns]
    else:
        df_check = df
    
    missing = df_check.isnull().sum()
    cols_with_missing = missing[missing > 0]
    
    assert len(cols_with_missing) == 0, \
        f"Found missing values in columns: {cols_with_missing.to_dict()}"


def compare_model_predictions(
    model1,
    model2,
    X: pd.DataFrame,
    tolerance: float = 1e-5
) -> bool:
    """
    Compare predictions from two models.
    
    Args:
        model1: First model
        model2: Second model
        X: Features to predict on
        tolerance: Tolerance for numerical comparison
    
    Returns:
        True if predictions are similar within tolerance
    """
    pred1 = model1.predict_proba(X)
    pred2 = model2.predict_proba(X)
    
    return np.allclose(pred1, pred2, atol=tolerance)


def mock_churn_probability(
    customer: Dict[str, Any],
    high_risk_factors: Dict[str, Any] = None
) -> float:
    """
    Generate a mock churn probability based on customer attributes.
    
    Args:
        customer: Customer dictionary
        high_risk_factors: Attributes that increase churn risk
    
    Returns:
        Mock churn probability between 0 and 1
    """
    if high_risk_factors is None:
        high_risk_factors = {
            'Contract': 'Month-to-month',
            'tenure': lambda x: x < 12,
            'InternetService': 'Fiber optic'
        }
    
    base_prob = 0.3
    risk_score = 0
    
    for attr, value in high_risk_factors.items():
        if callable(value):
            if value(customer.get(attr, 0)):
                risk_score += 0.15
        elif customer.get(attr) == value:
            risk_score += 0.15
    
    return min(base_prob + risk_score, 0.95)
