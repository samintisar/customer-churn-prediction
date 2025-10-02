"""
Quick test script to verify model training implementation.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.models import (
    train_baseline_model, 
    train_random_forest, 
    evaluate_model,
    save_model,
    load_model
)

def test_models():
    """Test model training and evaluation functions."""
    
    print("=" * 60)
    print("Testing Model Training Implementation")
    print("=" * 60)
    
    try:
        # Load test data
        print("\n[1/6] Loading test data...")
        test_data = pd.read_csv('data/processed/test.csv')
        train_data = pd.read_csv('data/processed/train.csv')
        
        X_train = train_data.drop('Churn', axis=1)
        y_train = train_data['Churn']
        X_test = test_data.drop('Churn', axis=1)
        y_test = test_data['Churn']
        
        print(f"✓ Loaded {len(X_train)} training samples")
        print(f"✓ Loaded {len(X_test)} test samples")
        
        # Test baseline model
        print("\n[2/6] Testing baseline model training...")
        baseline_model = train_baseline_model(X_train, y_train)
        print(f"✓ Baseline model trained: {type(baseline_model).__name__}")
        
        # Test baseline evaluation
        print("\n[3/6] Testing model evaluation...")
        metrics = evaluate_model(baseline_model, X_test, y_test)
        print(f"✓ Evaluation complete")
        print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall: {metrics['recall']:.4f}")
        print(f"  - Top-Decile Precision: {metrics['top_decile_precision']:.4f}")
        
        # Test random forest (without tuning for speed)
        print("\n[4/6] Testing Random Forest training (no tuning)...")
        rf_model = train_random_forest(
            X_train, 
            y_train, 
            tune_hyperparams=False
        )
        print(f"✓ Random Forest trained: {type(rf_model).__name__}")
        
        # Test RF evaluation
        print("\n[5/6] Testing RF evaluation...")
        rf_metrics = evaluate_model(rf_model, X_test, y_test)
        print(f"✓ Evaluation complete")
        print(f"  - ROC-AUC: {rf_metrics['roc_auc']:.4f}")
        print(f"  - Precision: {rf_metrics['precision']:.4f}")
        print(f"  - Recall: {rf_metrics['recall']:.4f}")
        print(f"  - Top-Decile Precision: {rf_metrics['top_decile_precision']:.4f}")
        
        # Test model saving and loading
        print("\n[6/6] Testing model save/load...")
        test_path = 'models/test_model.pkl'
        save_model(rf_model, test_path)
        loaded_model = load_model(test_path)
        print(f"✓ Model saved and loaded successfully")
        
        # Verify loaded model works
        test_pred = loaded_model.predict_proba(X_test.iloc[:5])
        print(f"✓ Loaded model can make predictions")
        
        # Clean up test file
        Path(test_path).unlink()
        print(f"✓ Test file cleaned up")
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\n📊 Model Comparison:")
        print(f"  Baseline ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  RF ROC-AUC:       {rf_metrics['roc_auc']:.4f}")
        print(f"  Improvement:      {(rf_metrics['roc_auc'] - metrics['roc_auc']) / metrics['roc_auc'] * 100:.2f}%")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ TEST FAILED: Data files not found")
        print(f"   Error: {e}")
        print(f"\n   Please run the data pipeline first:")
        print(f"   python scripts/test_pipeline.py")
        return False
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_models()
    sys.exit(0 if success else 1)

