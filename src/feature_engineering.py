"""
Feature Engineering Module

This module handles feature creation, transformation, and encoding for churn prediction.

Key Features to Engineer:
- Tenure groups (new, medium, long-term customers)
- Spending patterns (charges per month of tenure)
- Service usage combinations
- Payment reliability indicators
- Contract value features

Key Functions:
- encode_categorical: Handle one-hot/label encoding
- create_tenure_features: Bin and derive tenure-based features
- create_spending_features: Calculate spending metrics
- create_interaction_features: Product/service combinations
- preprocess_pipeline: End-to-end feature transformation pipeline

Usage:
    from src.feature_engineering import FeatureEngineer
    
    fe = FeatureEngineer()
    X_train_processed = fe.fit_transform(X_train)
    X_test_processed = fe.transform(X_test)
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Optional, Dict, Tuple
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for customer churn prediction.
    
    Handles encoding, feature creation, and scaling in a reproducible way.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.is_fitted = False
        
    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure-based features.
        
        Features:
        - Tenure groups (bins)
        - Is new customer flag
        - Customer lifetime proxy
        
        Args:
            df: Input dataframe with 'tenure' column
            
        Returns:
            DataFrame with additional tenure features
        """
        df_new = df.copy()
        
        if 'tenure' in df_new.columns:
            # Tenure groups
            df_new['tenure_group'] = pd.cut(
                df_new['tenure'],
                bins=[0, 12, 24, 48, np.inf],
                labels=['0-1yr', '1-2yr', '2-4yr', '4yr+']
            )
            
            # Is new customer (< 6 months)
            df_new['is_new_customer'] = (df_new['tenure'] < 6).astype(int)
            
            # Tenure in years (continuous)
            df_new['tenure_years'] = df_new['tenure'] / 12.0
            
            logger.debug(f"Created tenure features: tenure_group, is_new_customer, tenure_years")
        
        return df_new
    
    def create_spending_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create spending pattern features.
        
        Features:
        - Average monthly spend (TotalCharges / tenure)
        - Charge ratio (MonthlyCharges / TotalCharges)
        - Spending velocity
        
        Args:
            df: Input dataframe with charges columns
            
        Returns:
            DataFrame with additional spending features
        """
        df_new = df.copy()
        
        # Average monthly spend (handle division by zero)
        if 'TotalCharges' in df_new.columns and 'tenure' in df_new.columns:
            df_new['avg_monthly_spend'] = np.where(
                df_new['tenure'] > 0,
                df_new['TotalCharges'] / df_new['tenure'],
                df_new.get('MonthlyCharges', 0)
            )
        
        # Charge increase indicator (if current monthly > historical average)
        if 'MonthlyCharges' in df_new.columns and 'avg_monthly_spend' in df_new.columns:
            df_new['charge_increase'] = (
                df_new['MonthlyCharges'] > df_new['avg_monthly_spend']
            ).astype(int)
        
        # Total charges per tenure month (another view)
        if 'TotalCharges' in df_new.columns and 'tenure' in df_new.columns:
            df_new['total_charges_per_month'] = np.where(
                df_new['tenure'] > 0,
                df_new['TotalCharges'] / df_new['tenure'],
                0
            )
        
        logger.debug("Created spending features")
        
        return df_new
    
    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create service usage features.
        
        Features:
        - Total services count
        - Has premium services
        - Service combination flags
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with service features
        """
        df_new = df.copy()
        
        # Services to count (common in Telco dataset)
        service_cols = [
            'PhoneService', 'MultipleLines', 'OnlineSecurity', 
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 
            'StreamingTV', 'StreamingMovies'
        ]
        
        # Count total services (Yes values)
        available_service_cols = [col for col in service_cols if col in df_new.columns]
        if available_service_cols:
            df_new['total_services'] = sum(
                (df_new[col] == 'Yes').astype(int) 
                for col in available_service_cols
            )
            
            # Has multiple services
            df_new['has_multiple_services'] = (df_new['total_services'] > 1).astype(int)
            
            # Has premium services (security + backup)
            if 'OnlineSecurity' in df_new.columns and 'OnlineBackup' in df_new.columns:
                df_new['has_premium_services'] = (
                    (df_new['OnlineSecurity'] == 'Yes') & 
                    (df_new['OnlineBackup'] == 'Yes')
                ).astype(int)
        
        # Has internet service
        if 'InternetService' in df_new.columns:
            df_new['has_internet'] = (df_new['InternetService'] != 'No').astype(int)
            df_new['has_fiber'] = (df_new['InternetService'] == 'Fiber optic').astype(int)
        
        logger.debug("Created service features")
        
        return df_new
    
    def create_contract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create contract and payment features.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with contract features
        """
        df_new = df.copy()
        
        # Contract stability (longer contract = more stable)
        if 'Contract' in df_new.columns:
            contract_map = {
                'Month-to-month': 0,
                'One year': 1,
                'Two year': 2
            }
            df_new['contract_stability'] = df_new['Contract'].map(contract_map).fillna(0)
        
        # Electronic payment flag
        if 'PaymentMethod' in df_new.columns:
            df_new['is_electronic_payment'] = (
                df_new['PaymentMethod'].str.contains('electronic', case=False, na=False)
            ).astype(int)
        
        # Paperless billing flag (already exists, but ensure it's numeric)
        if 'PaperlessBilling' in df_new.columns:
            df_new['paperless_binary'] = (df_new['PaperlessBilling'] == 'Yes').astype(int)
        
        logger.debug("Created contract features")
        
        return df_new
    
    def encode_categorical(
        self, 
        df: pd.DataFrame, 
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding.
        
        Args:
            df: Input dataframe
            fit: Whether to fit the encoder
            
        Returns:
            DataFrame with encoded features
        """
        df_new = df.copy()
        
        # Identify categorical columns (object dtype or specific columns)
        categorical_cols = df_new.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            logger.debug("No categorical columns to encode")
            return df_new
        
        logger.info(f"Encoding {len(categorical_cols)} categorical columns: {categorical_cols[:5]}...")
        
        # Use one-hot encoding
        df_encoded = pd.get_dummies(
            df_new, 
            columns=categorical_cols,
            prefix=categorical_cols,
            drop_first=True  # Avoid multicollinearity
        )
        
        logger.info(f"Encoded features shape: {df_encoded.shape}")
        
        return df_encoded
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit the feature engineer and transform the data.
        
        Args:
            X: Input features
            y: Target variable (optional, for future use)
            
        Returns:
            Transformed feature matrix
        """
        logger.info("Fitting and transforming features")
        
        # Step 1: Create engineered features
        X_eng = self.create_tenure_features(X)
        X_eng = self.create_spending_features(X_eng)
        X_eng = self.create_service_features(X_eng)
        X_eng = self.create_contract_features(X_eng)
        
        # Step 2: Encode categorical variables
        X_encoded = self.encode_categorical(X_eng, fit=True)
        
        # Step 3: Scale numerical features
        X_scaled = X_encoded.copy()
        numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            logger.info(f"Scaling {len(numeric_cols)} numerical features")
            X_scaled[numeric_cols] = self.scaler.fit_transform(X_scaled[numeric_cols])
        
        # Store feature names for consistency
        self.feature_names = X_scaled.columns.tolist()
        self.is_fitted = True
        
        logger.info(f"Final feature count: {len(self.feature_names)}")
        
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted transformers.
        
        Args:
            X: Input features
            
        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform. Use fit_transform first.")
        
        logger.info("Transforming features")
        
        # Step 1: Create engineered features
        X_eng = self.create_tenure_features(X)
        X_eng = self.create_spending_features(X_eng)
        X_eng = self.create_service_features(X_eng)
        X_eng = self.create_contract_features(X_eng)
        
        # Step 2: Encode categorical variables
        X_encoded = self.encode_categorical(X_eng, fit=False)
        
        # Step 3: Ensure same columns as training
        # Add missing columns with 0s
        for col in self.feature_names:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        
        # Remove extra columns
        X_encoded = X_encoded[self.feature_names]
        
        # Step 4: Scale numerical features
        X_scaled = X_encoded.copy()
        numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            X_scaled[numeric_cols] = self.scaler.transform(X_scaled[numeric_cols])
        
        logger.info(f"Transformed to {X_scaled.shape[1]} features")
        
        return X_scaled
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted feature engineer.
        
        Args:
            filepath: Path to save the object
        """
        joblib.dump(self, filepath)
        logger.info(f"Saved FeatureEngineer to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'FeatureEngineer':
        """
        Load a fitted feature engineer.
        
        Args:
            filepath: Path to the saved object
            
        Returns:
            Loaded FeatureEngineer
        """
        fe = joblib.load(filepath)
        logger.info(f"Loaded FeatureEngineer from {filepath}")
        return fe


def test_feature_engineering():
    """
    Test function for feature engineering module.
    """
    print("=" * 60)
    print("Testing Feature Engineering Module")
    print("=" * 60)
    
    try:
        # Import data loader
        from data_loader import load_raw_data, clean_data, split_data
        
        # Load and prepare data
        print("\n[Test 1] Loading data...")
        df = load_raw_data('data/raw/Telco-Customer-Churn.csv')
        df_clean = clean_data(df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)
        print(f"✓ Loaded data - Train: {X_train.shape}")
        
        # Test feature engineering
        print("\n[Test 2] Creating feature engineer...")
        fe = FeatureEngineer()
        print("✓ FeatureEngineer initialized")
        
        print("\n[Test 3] Fitting and transforming training data...")
        X_train_processed = fe.fit_transform(X_train, y_train)
        print(f"✓ Processed training data: {X_train_processed.shape}")
        print(f"  Original features: {X_train.shape[1]}")
        print(f"  Engineered features: {X_train_processed.shape[1]}")
        print(f"  Sample features: {X_train_processed.columns.tolist()[:10]}")
        
        print("\n[Test 4] Transforming validation data...")
        X_val_processed = fe.transform(X_val)
        print(f"✓ Processed validation data: {X_val_processed.shape}")
        assert X_val_processed.shape[1] == X_train_processed.shape[1], "Feature count mismatch!"
        
        print("\n[Test 5] Transforming test data...")
        X_test_processed = fe.transform(X_test)
        print(f"✓ Processed test data: {X_test_processed.shape}")
        assert X_test_processed.shape[1] == X_train_processed.shape[1], "Feature count mismatch!"
        
        print("\n[Test 6] Checking for NaN values...")
        train_nans = X_train_processed.isnull().sum().sum()
        val_nans = X_val_processed.isnull().sum().sum()
        test_nans = X_test_processed.isnull().sum().sum()
        print(f"  Train NaNs: {train_nans}")
        print(f"  Val NaNs: {val_nans}")
        print(f"  Test NaNs: {test_nans}")
        if train_nans + val_nans + test_nans == 0:
            print("✓ No missing values")
        
        print("\n[Test 7] Saving feature engineer...")
        fe.save('models/feature_engineer.pkl')
        print("✓ Saved to models/feature_engineer.pkl")
        
        print("\n[Test 8] Loading feature engineer...")
        fe_loaded = FeatureEngineer.load('models/feature_engineer.pkl')
        X_test_reprocessed = fe_loaded.transform(X_test)
        assert X_test_reprocessed.shape == X_test_processed.shape, "Loaded FE produces different output!"
        print("✓ Loaded and verified feature engineer")
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
        # Print summary
        print("\n📊 Feature Engineering Summary:")
        print(f"  Input features: {X_train.shape[1]}")
        print(f"  Output features: {X_train_processed.shape[1]}")
        print(f"  Feature increase: +{X_train_processed.shape[1] - X_train.shape[1]}")
        
        return fe, X_train_processed, X_val_processed, X_test_processed
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_feature_engineering()

