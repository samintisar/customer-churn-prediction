"""
Tests for Feature Engineering Module

Tests feature creation, encoding, scaling, and the complete feature engineering pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer


class TestFeatureEngineerInit:
    """Tests for FeatureEngineer initialization."""
    
    def test_initialization(self):
        """Test that FeatureEngineer initializes correctly."""
        fe = FeatureEngineer()
        
        assert fe.scaler is not None
        assert fe.label_encoders == {}
        assert fe.feature_names is None
        assert fe.numeric_feature_names is None
        assert fe.is_fitted == False


class TestTenureFeatures:
    """Tests for tenure feature creation."""
    
    def test_create_tenure_features(self, sample_features_dataframe):
        """Test tenure feature creation."""
        fe = FeatureEngineer()
        df = sample_features_dataframe.copy()
        
        df_with_features = fe.create_tenure_features(df)
        
        # Check new features are created
        assert 'tenure_group' in df_with_features.columns
        assert 'is_new_customer' in df_with_features.columns
        assert 'tenure_years' in df_with_features.columns
        
        # Check tenure_years calculation
        assert np.allclose(
            df_with_features['tenure_years'], 
            df_with_features['tenure'] / 12.0
        )
    
    def test_tenure_groups_correct(self):
        """Test that tenure groups are correctly binned."""
        fe = FeatureEngineer()
        df = pd.DataFrame({
            'tenure': [3, 15, 30, 60]
        })
        
        df_with_features = fe.create_tenure_features(df)
        
        assert df_with_features.loc[0, 'tenure_group'] == '0-1yr'
        assert df_with_features.loc[1, 'tenure_group'] == '1-2yr'
        assert df_with_features.loc[2, 'tenure_group'] == '2-4yr'
        assert df_with_features.loc[3, 'tenure_group'] == '4yr+'
    
    def test_is_new_customer_flag(self):
        """Test new customer flag."""
        fe = FeatureEngineer()
        df = pd.DataFrame({
            'tenure': [3, 6, 12, 24]
        })
        
        df_with_features = fe.create_tenure_features(df)
        
        assert df_with_features.loc[0, 'is_new_customer'] == 1
        assert df_with_features.loc[1, 'is_new_customer'] == 0
        assert df_with_features.loc[2, 'is_new_customer'] == 0


class TestSpendingFeatures:
    """Tests for spending feature creation."""
    
    def test_create_spending_features(self):
        """Test spending feature creation."""
        fe = FeatureEngineer()
        df = pd.DataFrame({
            'tenure': [12, 24, 0],
            'MonthlyCharges': [50.0, 80.0, 30.0],
            'TotalCharges': [600.0, 1920.0, 0.0]
        })
        
        df_with_features = fe.create_spending_features(df)
        
        assert 'avg_monthly_spend' in df_with_features.columns
        assert 'charge_increase' in df_with_features.columns
        assert 'total_charges_per_month' in df_with_features.columns
    
    def test_avg_monthly_spend_calculation(self):
        """Test average monthly spend calculation."""
        fe = FeatureEngineer()
        df = pd.DataFrame({
            'tenure': [12, 24, 0],
            'MonthlyCharges': [50.0, 80.0, 30.0],
            'TotalCharges': [600.0, 1920.0, 0.0]
        })
        
        df_with_features = fe.create_spending_features(df)
        
        # tenure=12, TotalCharges=600 -> avg=50
        assert df_with_features.loc[0, 'avg_monthly_spend'] == 50.0
        # tenure=24, TotalCharges=1920 -> avg=80
        assert df_with_features.loc[1, 'avg_monthly_spend'] == 80.0
        # tenure=0 should use MonthlyCharges
        assert df_with_features.loc[2, 'avg_monthly_spend'] == 30.0
    
    def test_charge_increase_indicator(self):
        """Test charge increase indicator."""
        fe = FeatureEngineer()
        df = pd.DataFrame({
            'tenure': [12, 12],
            'MonthlyCharges': [60.0, 40.0],
            'TotalCharges': [600.0, 600.0]
        })
        
        df_with_features = fe.create_spending_features(df)
        
        # MonthlyCharges=60 > avg=50 -> increase=1
        assert df_with_features.loc[0, 'charge_increase'] == 1
        # MonthlyCharges=40 < avg=50 -> increase=0
        assert df_with_features.loc[1, 'charge_increase'] == 0


class TestServiceFeatures:
    """Tests for service feature creation."""
    
    def test_create_service_features(self):
        """Test service feature creation."""
        fe = FeatureEngineer()
        df = pd.DataFrame({
            'PhoneService': ['Yes', 'No'],
            'MultipleLines': ['Yes', 'No'],
            'OnlineSecurity': ['Yes', 'No'],
            'OnlineBackup': ['Yes', 'No'],
            'DeviceProtection': ['No', 'No'],
            'TechSupport': ['Yes', 'No'],
            'StreamingTV': ['No', 'Yes'],
            'StreamingMovies': ['Yes', 'No'],
            'InternetService': ['Fiber optic', 'No']
        })
        
        df_with_features = fe.create_service_features(df)
        
        assert 'total_services' in df_with_features.columns
        assert 'has_multiple_services' in df_with_features.columns
        assert 'has_premium_services' in df_with_features.columns
        assert 'has_internet' in df_with_features.columns
        assert 'has_fiber' in df_with_features.columns
    
    def test_total_services_count(self):
        """Test total services counting."""
        fe = FeatureEngineer()
        df = pd.DataFrame({
            'PhoneService': ['Yes', 'No'],
            'MultipleLines': ['Yes', 'No'],
            'OnlineSecurity': ['Yes', 'No'],
            'OnlineBackup': ['No', 'No'],
            'DeviceProtection': ['No', 'No'],
            'TechSupport': ['No', 'No'],
            'StreamingTV': ['No', 'No'],
            'StreamingMovies': ['No', 'No']
        })
        
        df_with_features = fe.create_service_features(df)
        
        # First customer has 3 services
        assert df_with_features.loc[0, 'total_services'] == 3
        # Second customer has 0 services
        assert df_with_features.loc[1, 'total_services'] == 0
    
    def test_premium_services_flag(self):
        """Test premium services indicator."""
        fe = FeatureEngineer()
        df = pd.DataFrame({
            'OnlineSecurity': ['Yes', 'Yes', 'No'],
            'OnlineBackup': ['Yes', 'No', 'Yes'],
            'PhoneService': ['Yes', 'Yes', 'Yes']
        })
        
        df_with_features = fe.create_service_features(df)
        
        # Only first customer has both premium services
        assert df_with_features.loc[0, 'has_premium_services'] == 1
        assert df_with_features.loc[1, 'has_premium_services'] == 0
        assert df_with_features.loc[2, 'has_premium_services'] == 0


class TestContractFeatures:
    """Tests for contract feature creation."""
    
    def test_create_contract_features(self):
        """Test contract feature creation."""
        fe = FeatureEngineer()
        df = pd.DataFrame({
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
            'PaperlessBilling': ['Yes', 'No', 'Yes']
        })
        
        df_with_features = fe.create_contract_features(df)
        
        assert 'contract_stability' in df_with_features.columns
        assert 'is_electronic_payment' in df_with_features.columns
        assert 'paperless_binary' in df_with_features.columns
    
    def test_contract_stability_mapping(self):
        """Test contract stability values."""
        fe = FeatureEngineer()
        df = pd.DataFrame({
            'Contract': ['Month-to-month', 'One year', 'Two year']
        })
        
        df_with_features = fe.create_contract_features(df)
        
        assert df_with_features.loc[0, 'contract_stability'] == 0
        assert df_with_features.loc[1, 'contract_stability'] == 1
        assert df_with_features.loc[2, 'contract_stability'] == 2


class TestCategoricalEncoding:
    """Tests for categorical encoding."""
    
    def test_encode_categorical(self):
        """Test categorical encoding creates dummy variables."""
        fe = FeatureEngineer()
        df = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'tenure': [12, 24, 36]
        })
        
        df_encoded = fe.encode_categorical(df, fit=True)
        
        # Check that categorical columns are encoded
        assert 'gender' not in df_encoded.columns
        assert 'Contract' not in df_encoded.columns
        
        # Check that numeric column is preserved
        assert 'tenure' in df_encoded.columns
        
        # Check encoded columns exist (drop_first=True, so n-1 columns)
        encoded_cols = [col for col in df_encoded.columns if 'gender' in col or 'Contract' in col]
        assert len(encoded_cols) > 0


class TestFitTransform:
    """Tests for fit_transform method."""
    
    def test_fit_transform_basic(self, sample_features_dataframe):
        """Test basic fit_transform functionality."""
        fe = FeatureEngineer()
        X = sample_features_dataframe.copy()
        
        X_transformed = fe.fit_transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed) == len(X)
        assert X_transformed.shape[1] > X.shape[1]  # More features after engineering
        assert fe.is_fitted == True
        assert fe.feature_names is not None
    
    def test_fit_transform_no_missing_values(self, sample_features_dataframe):
        """Test that fit_transform produces no missing values."""
        fe = FeatureEngineer()
        X = sample_features_dataframe.copy()
        
        X_transformed = fe.fit_transform(X)
        
        assert X_transformed.isnull().sum().sum() == 0
    
    def test_fit_transform_sets_feature_names(self, sample_features_dataframe):
        """Test that feature names are stored."""
        fe = FeatureEngineer()
        X = sample_features_dataframe.copy()
        
        X_transformed = fe.fit_transform(X)
        
        assert fe.feature_names is not None
        assert len(fe.feature_names) == X_transformed.shape[1]
        assert fe.feature_names == X_transformed.columns.tolist()


class TestTransform:
    """Tests for transform method."""
    
    def test_transform_requires_fit(self, sample_features_dataframe):
        """Test that transform requires fit first."""
        fe = FeatureEngineer()
        X = sample_features_dataframe.copy()
        
        with pytest.raises(ValueError, match="must be fitted"):
            fe.transform(X)
    
    def test_transform_consistency(self, sample_features_dataframe):
        """Test that transform produces same features as fit_transform."""
        fe = FeatureEngineer()
        X = sample_features_dataframe.copy()
        
        # Fit on first half
        X_train = X.iloc[:25]
        X_test = X.iloc[25:]
        
        X_train_transformed = fe.fit_transform(X_train)
        X_test_transformed = fe.transform(X_test)
        
        # Should have same number of features
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
        # Should have same column names
        assert list(X_train_transformed.columns) == list(X_test_transformed.columns)
    
    def test_transform_handles_missing_columns(self):
        """Test that transform handles missing categorical values."""
        fe = FeatureEngineer()
        
        # Train data with 3 contract types
        X_train = pd.DataFrame({
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'gender': ['Male', 'Female', 'Male'],
            'tenure': [12, 24, 36]
        })
        
        # Test data with only 2 contract types
        X_test = pd.DataFrame({
            'Contract': ['Month-to-month', 'One year'],
            'gender': ['Female', 'Male'],
            'tenure': [15, 20]
        })
        
        fe.fit_transform(X_train)
        X_test_transformed = fe.transform(X_test)
        
        # Should still have all features from training
        assert X_test_transformed.shape[1] == fe.fit_transform(X_train).shape[1]


class TestSaveLoad:
    """Tests for save and load functionality."""
    
    def test_save_and_load(self, sample_features_dataframe, temp_dir):
        """Test saving and loading feature engineer."""
        fe = FeatureEngineer()
        X = sample_features_dataframe.copy()
        
        # Fit and save
        X_transformed = fe.fit_transform(X)
        save_path = temp_dir / "feature_engineer.pkl"
        fe.save(str(save_path))
        
        assert save_path.exists()
        
        # Load and transform
        fe_loaded = FeatureEngineer.load(str(save_path))
        X_transformed_loaded = fe_loaded.transform(X)
        
        # Results should be identical
        pd.testing.assert_frame_equal(X_transformed, X_transformed_loaded)
    
    def test_loaded_engineer_has_same_attributes(self, sample_features_dataframe, temp_dir):
        """Test that loaded engineer has same attributes."""
        fe = FeatureEngineer()
        X = sample_features_dataframe.copy()
        
        fe.fit_transform(X)
        save_path = temp_dir / "feature_engineer.pkl"
        fe.save(str(save_path))
        
        fe_loaded = FeatureEngineer.load(str(save_path))
        
        assert fe_loaded.is_fitted == True
        assert fe_loaded.feature_names == fe.feature_names
        assert fe_loaded.numeric_feature_names == fe.numeric_feature_names


class TestIntegration:
    """Integration tests for feature engineering."""
    
    def test_complete_pipeline(self, sample_features_dataframe):
        """Test complete feature engineering pipeline."""
        fe = FeatureEngineer()
        X = sample_features_dataframe.copy()
        
        # Split data
        X_train = X.iloc[:30]
        X_test = X.iloc[30:]
        
        # Fit on training data
        X_train_transformed = fe.fit_transform(X_train)
        
        # Transform test data
        X_test_transformed = fe.transform(X_test)
        
        # Verify
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
        assert X_train_transformed.isnull().sum().sum() == 0
        assert X_test_transformed.isnull().sum().sum() == 0
        assert fe.is_fitted == True
    
    def test_preserves_sample_count(self, sample_features_dataframe):
        """Test that feature engineering preserves sample count."""
        fe = FeatureEngineer()
        X = sample_features_dataframe.copy()
        
        X_transformed = fe.fit_transform(X)
        
        assert len(X_transformed) == len(X)
