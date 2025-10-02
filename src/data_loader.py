"""
Data Loading Module

This module handles loading and initial preprocessing of customer churn data.

Key Functions:
- load_raw_data: Load the Telco Customer Churn dataset
- validate_data: Check data quality and missing values
- split_data: Create train/validation/test splits with stratification
- save_processed_data: Save cleaned data for reproducibility

Usage:
    from src.data_loader import load_raw_data, split_data
    
    df = load_raw_data('data/raw/Telco-Customer-Churn.csv')
    X_train, X_test, y_train, y_test = split_data(df)
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with raw customer data
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If the file is empty or malformed
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.error(f"Data file not found: {filepath}")
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    logger.info(f"Loading data from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        
        if df.empty:
            raise ValueError("Loaded dataframe is empty")
        
        return df
    
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty")
        raise ValueError("CSV file is empty")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def validate_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data quality and return summary statistics.
    
    Args:
        df: Raw customer dataframe
        
    Returns:
        Dictionary containing validation results (missing values, dtypes, etc.)
    """
    logger.info("Validating data quality")
    
    validation_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'duplicates': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    # Log summary
    total_missing = df.isnull().sum().sum()
    logger.info(f"Shape: {validation_report['shape']}")
    logger.info(f"Total missing values: {total_missing}")
    logger.info(f"Duplicate rows: {validation_report['duplicates']}")
    
    # Log columns with missing values
    missing_cols = {k: v for k, v in validation_report['missing_values'].items() if v > 0}
    if missing_cols:
        logger.warning(f"Columns with missing values: {missing_cols}")
    
    return validation_report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data by handling missing values and data type issues.
    
    Args:
        df: Raw dataframe
        
    Returns:
        Cleaned dataframe
    """
    logger.info("Cleaning data")
    df_clean = df.copy()
    
    # Convert TotalCharges to numeric (it's sometimes stored as string with spaces)
    if 'TotalCharges' in df_clean.columns:
        # Replace empty strings with NaN
        df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # For customers with 0 tenure, set TotalCharges to 0
        mask = (df_clean['tenure'] == 0) | (df_clean['TotalCharges'].isnull())
        df_clean.loc[mask, 'TotalCharges'] = 0
        
        logger.info(f"Cleaned TotalCharges column, filled {mask.sum()} missing values")
    
    # Convert SeniorCitizen to string for consistency with other categorical variables
    if 'SeniorCitizen' in df_clean.columns:
        df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # Standardize categorical values (strip whitespace, ensure consistency)
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].str.strip()
    
    logger.info("Data cleaning completed")
    
    return df_clean


def split_data(
    df: pd.DataFrame, 
    target_col: str = 'Churn',
    test_size: float = 0.2, 
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train/validation/test sets with stratification.
    
    Args:
        df: Preprocessed dataframe
        target_col: Name of target column
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info(f"Splitting data: test_size={test_size}, val_size={val_size}")
    
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Drop customerID if present (not a feature)
    if 'customerID' in X.columns:
        X = X.drop(columns=['customerID'])
        logger.info("Dropped customerID column")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    # Second split: train vs val
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for already split test set
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        logger.info(f"Split sizes - Train: {len(X_temp)}, Test: {len(X_test)}")
        return X_temp, None, X_test, y_temp, None, y_test


def save_processed_data(
    df: pd.DataFrame, 
    output_path: str,
    create_dir: bool = True
) -> None:
    """
    Save processed data to disk for reproducibility.
    
    Args:
        df: Processed dataframe
        output_path: Output file path
        create_dir: Whether to create parent directories if they don't exist
    """
    output_path = Path(output_path)
    
    if create_dir:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed data to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved {len(df)} rows to {output_path}")


def test_data_loader():
    """
    Test function for data loading module.
    """
    print("=" * 60)
    print("Testing Data Loader Module")
    print("=" * 60)
    
    try:
        # Test 1: Load raw data
        print("\n[Test 1] Loading raw data...")
        df = load_raw_data('data/raw/Telco-Customer-Churn.csv')
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"  Columns: {df.columns.tolist()[:5]}...")
        
        # Test 2: Validate data
        print("\n[Test 2] Validating data...")
        validation = validate_data(df)
        print(f"✓ Shape: {validation['shape']}")
        print(f"  Missing values: {sum(validation['missing_values'].values())}")
        print(f"  Duplicates: {validation['duplicates']}")
        
        # Test 3: Clean data
        print("\n[Test 3] Cleaning data...")
        df_clean = clean_data(df)
        print(f"✓ Cleaned data shape: {df_clean.shape}")
        print(f"  Missing values after cleaning: {df_clean.isnull().sum().sum()}")
        
        # Test 4: Split data
        print("\n[Test 4] Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)
        print(f"✓ Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        print(f"  Features: {len(X_train.columns)}")
        
        # Check stratification
        print(f"\n  Class distribution:")
        print(f"  Train Churn Rate: {(y_train == 'Yes').mean():.2%}")
        print(f"  Val Churn Rate: {(y_val == 'Yes').mean():.2%}")
        print(f"  Test Churn Rate: {(y_test == 'Yes').mean():.2%}")
        
        # Test 5: Save data
        print("\n[Test 5] Saving processed data...")
        save_processed_data(df_clean, 'data/processed/processed_data.csv')
        print("✓ Data saved successfully")
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
        return df_clean
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_data_loader()

