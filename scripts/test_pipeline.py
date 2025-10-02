"""
Test script for data loading and feature engineering pipeline.

This script tests the complete data pipeline from raw data to processed features.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import (
    load_raw_data, 
    validate_data, 
    clean_data, 
    split_data, 
    save_processed_data
)
from src.feature_engineering import FeatureEngineer


def main():
    """
    Run complete pipeline test.
    """
    print("\n" + "="*70)
    print("CUSTOMER CHURN PREDICTION - PIPELINE TEST")
    print("="*70)
    
    # ========================================================================
    # Step 1: Load and validate raw data
    # ========================================================================
    print("\n[STEP 1] Loading raw data...")
    df_raw = load_raw_data('data/raw/Telco-Customer-Churn.csv')
    print(f"[OK] Loaded {len(df_raw):,} rows, {len(df_raw.columns)} columns")
    
    # ========================================================================
    # Step 2: Validate data quality
    # ========================================================================
    print("\n[STEP 2] Validating data quality...")
    validation = validate_data(df_raw)
    print(f"[OK] Validation complete")
    print(f"  - Shape: {validation['shape']}")
    print(f"  - Missing values: {sum(validation['missing_values'].values())}")
    print(f"  - Duplicates: {validation['duplicates']}")
    
    # ========================================================================
    # Step 3: Clean data
    # ========================================================================
    print("\n[STEP 3] Cleaning data...")
    df_clean = clean_data(df_raw)
    print(f"[OK] Data cleaned")
    print(f"  - Remaining missing values: {df_clean.isnull().sum().sum()}")
    
    # ========================================================================
    # Step 4: Save processed data
    # ========================================================================
    print("\n[STEP 4] Saving processed data...")
    save_processed_data(df_clean, 'data/processed/cleaned_data.csv')
    print("[OK] Saved to data/processed/cleaned_data.csv")
    
    # ========================================================================
    # Step 5: Split data
    # ========================================================================
    print("\n[STEP 5] Splitting data into train/val/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)
    
    print(f"[OK] Data split complete:")
    print(f"  - Training set:   {len(X_train):,} samples ({len(X_train)/len(df_clean):.1%})")
    print(f"  - Validation set: {len(X_val):,} samples ({len(X_val)/len(df_clean):.1%})")
    print(f"  - Test set:       {len(X_test):,} samples ({len(X_test)/len(df_clean):.1%})")
    print(f"  - Features:       {len(X_train.columns)}")
    
    # Check class balance
    print(f"\n  Class distribution (Churn='Yes'):")
    print(f"  - Overall:     {(df_clean['Churn'] == 'Yes').mean():.2%}")
    print(f"  - Training:    {(y_train == 'Yes').mean():.2%}")
    print(f"  - Validation:  {(y_val == 'Yes').mean():.2%}")
    print(f"  - Test:        {(y_test == 'Yes').mean():.2%}")
    
    # ========================================================================
    # Step 6: Feature engineering
    # ========================================================================
    print("\n[STEP 6] Engineering features...")
    fe = FeatureEngineer()
    
    # Fit on training data
    X_train_processed = fe.fit_transform(X_train, y_train)
    print(f"[OK] Training features engineered: {X_train_processed.shape}")
    
    # Transform validation and test data
    X_val_processed = fe.transform(X_val)
    X_test_processed = fe.transform(X_test)
    print(f"[OK] Validation features engineered: {X_val_processed.shape}")
    print(f"[OK] Test features engineered: {X_test_processed.shape}")
    
    print(f"\n  Feature engineering summary:")
    print(f"  - Original features: {X_train.shape[1]}")
    print(f"  - Engineered features: {X_train_processed.shape[1]}")
    print(f"  - New features created: +{X_train_processed.shape[1] - X_train.shape[1]}")
    
    # Sample feature names
    print(f"\n  Sample engineered features:")
    for i, col in enumerate(X_train_processed.columns[:10], 1):
        print(f"    {i}. {col}")
    print(f"    ... ({len(X_train_processed.columns) - 10} more)")
    
    # ========================================================================
    # Step 7: Save artifacts
    # ========================================================================
    print("\n[STEP 7] Saving artifacts...")
    
    # Save feature engineer
    fe.save('models/feature_engineer.pkl')
    print("[OK] Saved feature engineer to models/feature_engineer.pkl")
    
    # Save processed datasets
    save_processed_data(
        X_train_processed.assign(Churn=y_train.values), 
        'data/processed/train.csv'
    )
    save_processed_data(
        X_val_processed.assign(Churn=y_val.values), 
        'data/processed/val.csv'
    )
    save_processed_data(
        X_test_processed.assign(Churn=y_test.values), 
        'data/processed/test.csv'
    )
    print("[OK] Saved train/val/test datasets to data/processed/")
    
    # ========================================================================
    # Step 8: Final validation
    # ========================================================================
    print("\n[STEP 8] Final validation...")
    
    # Check for NaN values
    total_nans = (
        X_train_processed.isnull().sum().sum() +
        X_val_processed.isnull().sum().sum() +
        X_test_processed.isnull().sum().sum()
    )
    
    if total_nans == 0:
        print("[OK] No missing values in processed data")
    else:
        print(f"[WARNING] {total_nans} missing values found")
    
    # Check feature consistency
    if (X_train_processed.shape[1] == X_val_processed.shape[1] == X_test_processed.shape[1]):
        print("[OK] Feature counts consistent across all splits")
    else:
        print("[ERROR] Feature count mismatch!")
    
    # Check target consistency
    if len(y_train) + len(y_val) + len(y_test) == len(df_clean):
        print("[OK] No samples lost during splitting")
    else:
        print("[ERROR] Sample count mismatch!")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("PIPELINE TEST COMPLETE [SUCCESS]")
    print("="*70)
    
    print("\n>> Summary Statistics:")
    print(f"  Total customers: {len(df_clean):,}")
    print(f"  Churn rate: {(df_clean['Churn'] == 'Yes').mean():.2%}")
    print(f"  Features: {X_train_processed.shape[1]:,}")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Ready for model training: YES")
    
    print("\n>> Saved Files:")
    print("  [OK] data/processed/cleaned_data.csv")
    print("  [OK] data/processed/train.csv")
    print("  [OK] data/processed/val.csv")
    print("  [OK] data/processed/test.csv")
    print("  [OK] models/feature_engineer.pkl")
    
    print("\n>> Next Steps:")
    print("  1. Run EDA notebook: notebooks/01_eda.ipynb")
    print("  2. Train models: notebooks/02_model_experiments.ipynb")
    print("  3. Build dashboard: streamlit run app/dashboard.py")
    
    print("\n" + "="*70 + "\n")
    
    return {
        'X_train': X_train_processed,
        'X_val': X_val_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_engineer': fe
    }


if __name__ == "__main__":
    results = main()

