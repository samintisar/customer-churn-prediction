"""
Tests for Data Loading Module

Tests the data loading, validation, cleaning, and splitting functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    load_raw_data,
    validate_data,
    clean_data,
    split_data,
    save_processed_data
)


class TestLoadRawData:
    """Tests for load_raw_data function."""
    
    def test_load_raw_data_success(self, raw_data_path):
        """Test successful data loading."""
        if not raw_data_path.exists():
            pytest.skip("Raw data file not found")
        
        df = load_raw_data(str(raw_data_path))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 0
        assert 'customerID' in df.columns
        assert 'Churn' in df.columns
    
    def test_load_raw_data_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_raw_data('nonexistent_file.csv')
    
    def test_load_raw_data_returns_dataframe(self, raw_data_path):
        """Test that function returns a DataFrame."""
        if not raw_data_path.exists():
            pytest.skip("Raw data file not found")
        
        df = load_raw_data(str(raw_data_path))
        assert isinstance(df, pd.DataFrame)


class TestValidateData:
    """Tests for validate_data function."""
    
    def test_validate_data_structure(self, sample_dataframe):
        """Test validation returns correct structure."""
        result = validate_data(sample_dataframe)
        
        assert isinstance(result, dict)
        assert 'shape' in result
        assert 'missing_values' in result
        assert 'missing_percentage' in result
        assert 'dtypes' in result
        assert 'duplicates' in result
        assert 'numeric_columns' in result
        assert 'categorical_columns' in result
    
    def test_validate_data_shape(self, sample_dataframe):
        """Test shape validation."""
        result = validate_data(sample_dataframe)
        assert result['shape'] == sample_dataframe.shape
    
    def test_validate_data_missing_values(self, sample_dataframe):
        """Test missing value detection."""
        # Add some missing values
        df_with_missing = sample_dataframe.copy()
        df_with_missing.loc[0, 'MonthlyCharges'] = np.nan
        
        result = validate_data(df_with_missing)
        assert result['missing_values']['MonthlyCharges'] == 1
    
    def test_validate_data_duplicates(self, sample_dataframe):
        """Test duplicate detection."""
        # Add a duplicate row
        df_with_dup = pd.concat([sample_dataframe, sample_dataframe.iloc[[0]]], ignore_index=True)
        
        result = validate_data(df_with_dup)
        assert result['duplicates'] >= 1
    
    def test_validate_data_column_types(self, sample_dataframe):
        """Test column type identification."""
        result = validate_data(sample_dataframe)
        
        assert 'tenure' in result['numeric_columns']
        assert 'MonthlyCharges' in result['numeric_columns']
        assert 'gender' in result['categorical_columns']


class TestCleanData:
    """Tests for clean_data function."""
    
    def test_clean_data_handles_total_charges(self, sample_dataframe):
        """Test TotalCharges cleaning."""
        # Add some problematic TotalCharges values
        df = sample_dataframe.copy()
        df.loc[0, 'TotalCharges'] = ' '
        df.loc[1, 'tenure'] = 0
        
        df_clean = clean_data(df)
        
        # Check that spaces are handled
        assert not pd.isna(df_clean.loc[0, 'TotalCharges'])
        # Check that 0 tenure customers have 0 TotalCharges
        assert df_clean.loc[1, 'TotalCharges'] == 0
    
    def test_clean_data_senior_citizen_conversion(self):
        """Test SeniorCitizen conversion to string."""
        df = pd.DataFrame({
            'SeniorCitizen': [0, 1, 0, 1],
            'tenure': [12, 24, 36, 48]
        })
        
        df_clean = clean_data(df)
        
        assert df_clean['SeniorCitizen'].dtype == 'object'
        assert set(df_clean['SeniorCitizen'].unique()) == {'Yes', 'No'}
    
    def test_clean_data_strips_whitespace(self):
        """Test that categorical values are stripped of whitespace."""
        df = pd.DataFrame({
            'gender': [' Male ', 'Female ', ' Female'],
            'tenure': [12, 24, 36]
        })
        
        df_clean = clean_data(df)
        
        assert all(df_clean['gender'].str.strip() == df_clean['gender'])
    
    def test_clean_data_preserves_shape(self, sample_dataframe):
        """Test that cleaning preserves DataFrame shape."""
        df_clean = clean_data(sample_dataframe)
        
        assert df_clean.shape == sample_dataframe.shape


class TestSplitData:
    """Tests for split_data function."""
    
    def test_split_data_default(self, sample_dataframe):
        """Test default data splitting."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(sample_dataframe)
        
        # Check that all splits are returned
        assert X_train is not None
        assert X_val is not None
        assert X_test is not None
        assert y_train is not None
        assert y_val is not None
        assert y_test is not None
        
        # Check sizes
        total_size = len(X_train) + len(X_val) + len(X_test)
        assert total_size == len(sample_dataframe)
    
    def test_split_data_no_validation(self, sample_dataframe):
        """Test splitting without validation set."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            sample_dataframe, 
            val_size=0
        )
        
        assert X_val is None
        assert y_val is None
        assert len(X_train) + len(X_test) == len(sample_dataframe)
    
    def test_split_data_removes_customer_id(self, sample_dataframe):
        """Test that customerID is removed from features."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(sample_dataframe)
        
        assert 'customerID' not in X_train.columns
        assert 'customerID' not in X_test.columns
    
    def test_split_data_stratification(self, sample_dataframe):
        """Test that class distribution is preserved."""
        original_churn_rate = (sample_dataframe['Churn'] == 'Yes').mean()
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(sample_dataframe)
        
        train_churn_rate = (y_train == 'Yes').mean()
        test_churn_rate = (y_test == 'Yes').mean()
        
        # Allow 10% deviation due to small sample size
        assert abs(train_churn_rate - original_churn_rate) < 0.1
        assert abs(test_churn_rate - original_churn_rate) < 0.1
    
    def test_split_data_reproducibility(self, sample_dataframe):
        """Test that splitting is reproducible with same random_state."""
        X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = split_data(
            sample_dataframe, random_state=42
        )
        X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = split_data(
            sample_dataframe, random_state=42
        )
        
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_series_equal(y_train1, y_train2)


class TestSaveProcessedData:
    """Tests for save_processed_data function."""
    
    def test_save_processed_data_success(self, sample_dataframe, temp_dir):
        """Test successful data saving."""
        output_path = temp_dir / "test_data.csv"
        
        save_processed_data(sample_dataframe, str(output_path))
        
        assert output_path.exists()
        
        # Load and verify
        df_loaded = pd.read_csv(output_path)
        assert len(df_loaded) == len(sample_dataframe)
        assert list(df_loaded.columns) == list(sample_dataframe.columns)
    
    def test_save_processed_data_creates_directory(self, sample_dataframe, temp_dir):
        """Test that parent directories are created."""
        output_path = temp_dir / "subdir" / "test_data.csv"
        
        save_processed_data(sample_dataframe, str(output_path), create_dir=True)
        
        assert output_path.exists()
        assert output_path.parent.exists()


class TestIntegration:
    """Integration tests for the data loading pipeline."""
    
    def test_full_pipeline(self, raw_data_path, temp_dir):
        """Test complete data loading pipeline."""
        if not raw_data_path.exists():
            pytest.skip("Raw data file not found")
        
        # Load
        df = load_raw_data(str(raw_data_path))
        
        # Validate
        validation = validate_data(df)
        assert validation['shape'][0] > 0
        
        # Clean
        df_clean = clean_data(df)
        
        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)
        
        # Verify splits
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Save
        output_path = temp_dir / "processed_data.csv"
        save_processed_data(df_clean, str(output_path))
        assert output_path.exists()
