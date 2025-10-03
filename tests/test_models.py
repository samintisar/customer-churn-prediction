"""
Tests for Model Training and Evaluation Module

Tests model training, evaluation metrics, and model persistence.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    train_baseline_model,
    train_random_forest,
    evaluate_model,
    calculate_top_decile_precision,
    save_model,
    load_model
)


@pytest.fixture
def sample_training_data():
    """Create sample training data for model testing."""
    np.random.seed(42)
    
    # Create 200 samples with 10 features
    X = pd.DataFrame(
        np.random.randn(200, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )
    
    # Create target with ~30% churn
    y = pd.Series(np.random.choice(['Yes', 'No'], size=200, p=[0.3, 0.7]))
    
    return X, y


@pytest.fixture
def trained_model(sample_training_data):
    """Create a trained model for testing."""
    X, y = sample_training_data
    model = train_baseline_model(X, y)
    return model


class TestTrainBaselineModel:
    """Tests for baseline model training."""
    
    def test_train_baseline_model_returns_model(self, sample_training_data):
        """Test that baseline training returns a model."""
        X, y = sample_training_data
        
        model = train_baseline_model(X, y)
        
        assert model is not None
        assert isinstance(model, LogisticRegression)
    
    def test_train_baseline_model_is_fitted(self, sample_training_data):
        """Test that returned model is fitted."""
        X, y = sample_training_data
        
        model = train_baseline_model(X, y)
        
        # Fitted models have these attributes
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
    
    def test_train_baseline_model_can_predict(self, sample_training_data):
        """Test that model can make predictions."""
        X, y = sample_training_data
        
        model = train_baseline_model(X, y)
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert len(probabilities) == len(X)
        assert probabilities.shape[1] == 2  # Binary classification
    
    def test_train_baseline_model_with_custom_params(self, sample_training_data):
        """Test model training with custom parameters."""
        X, y = sample_training_data
        
        model = train_baseline_model(X, y, max_iter=500, C=0.5)
        
        assert model.max_iter == 500
        assert model.C == 0.5
    
    def test_train_baseline_model_handles_class_imbalance(self, sample_training_data):
        """Test that model uses balanced class weights."""
        X, y = sample_training_data
        
        model = train_baseline_model(X, y)
        
        assert model.class_weight == 'balanced'


class TestTrainRandomForest:
    """Tests for random forest training."""
    
    def test_train_random_forest_returns_model(self, sample_training_data):
        """Test that RF training returns a model."""
        X, y = sample_training_data
        
        model = train_random_forest(X, y, tune_hyperparams=False)
        
        assert model is not None
        assert isinstance(model, RandomForestClassifier)
    
    def test_train_random_forest_is_fitted(self, sample_training_data):
        """Test that returned model is fitted."""
        X, y = sample_training_data
        
        model = train_random_forest(X, y, tune_hyperparams=False)
        
        # Fitted models have these attributes
        assert hasattr(model, 'estimators_')
        assert len(model.estimators_) > 0
    
    def test_train_random_forest_can_predict(self, sample_training_data):
        """Test that model can make predictions."""
        X, y = sample_training_data
        
        model = train_random_forest(X, y, tune_hyperparams=False)
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert len(probabilities) == len(X)
        assert probabilities.shape[1] == 2
    
    def test_train_random_forest_without_tuning(self, sample_training_data):
        """Test RF training without hyperparameter tuning."""
        X, y = sample_training_data
        
        model = train_random_forest(X, y, tune_hyperparams=False)
        
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 200  # Default value
    
    def test_train_random_forest_has_feature_importance(self, sample_training_data):
        """Test that RF model has feature importance."""
        X, y = sample_training_data
        
        model = train_random_forest(X, y, tune_hyperparams=False)
        
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X.shape[1]
        assert np.allclose(model.feature_importances_.sum(), 1.0)


class TestEvaluateModel:
    """Tests for model evaluation."""
    
    def test_evaluate_model_returns_dict(self, trained_model, sample_training_data):
        """Test that evaluation returns a dictionary."""
        X, y = sample_training_data
        
        metrics = evaluate_model(trained_model, X, y)
        
        assert isinstance(metrics, dict)
    
    def test_evaluate_model_contains_required_metrics(self, trained_model, sample_training_data):
        """Test that all required metrics are present."""
        X, y = sample_training_data
        
        metrics = evaluate_model(trained_model, X, y)
        
        required_metrics = [
            'roc_auc', 'precision', 'recall', 'f1_score',
            'top_decile_precision', 'tn', 'fp', 'fn', 'tp'
        ]
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
    
    def test_evaluate_model_metric_ranges(self, trained_model, sample_training_data):
        """Test that metrics are within valid ranges."""
        X, y = sample_training_data
        
        metrics = evaluate_model(trained_model, X, y)
        
        # Metrics should be between 0 and 1
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['top_decile_precision'] <= 1
    
    def test_evaluate_model_confusion_matrix_sum(self, trained_model, sample_training_data):
        """Test that confusion matrix sums to total samples."""
        X, y = sample_training_data
        
        metrics = evaluate_model(trained_model, X, y)
        
        cm_sum = metrics['tn'] + metrics['fp'] + metrics['fn'] + metrics['tp']
        assert cm_sum == len(y)
    
    def test_evaluate_model_with_different_threshold(self, trained_model, sample_training_data):
        """Test evaluation with custom threshold."""
        X, y = sample_training_data
        
        metrics_default = evaluate_model(trained_model, X, y, threshold=0.5)
        metrics_high = evaluate_model(trained_model, X, y, threshold=0.7)
        
        # Higher threshold should generally reduce recall
        assert isinstance(metrics_default, dict)
        assert isinstance(metrics_high, dict)


class TestCalculateTopDecilePrecision:
    """Tests for top decile precision calculation."""
    
    def test_calculate_top_decile_precision_basic(self):
        """Test basic top decile precision calculation."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_pred_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        
        precision = calculate_top_decile_precision(y_true, y_pred_proba, decile=10)
        
        # Top 10% is 1 sample (index 0), which is positive
        assert precision == 1.0
    
    def test_calculate_top_decile_precision_range(self):
        """Test that precision is between 0 and 1."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.rand(100)
        
        precision = calculate_top_decile_precision(y_true, y_pred_proba)
        
        assert 0 <= precision <= 1
    
    def test_calculate_top_decile_precision_perfect_model(self):
        """Test precision for a perfect model."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_pred_proba = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        
        precision = calculate_top_decile_precision(y_true, y_pred_proba, decile=10)
        
        # Top 10% (1 sample) is positive
        assert precision == 1.0
    
    def test_calculate_top_decile_precision_random_model(self):
        """Test precision for a random model."""
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.randint(0, 2, n_samples)
        y_pred_proba = np.random.rand(n_samples)
        
        precision = calculate_top_decile_precision(y_true, y_pred_proba)
        
        # Random model should have precision close to base rate
        base_rate = y_true.mean()
        assert abs(precision - base_rate) < 0.2  # Allow some variance


class TestSaveLoadModel:
    """Tests for model persistence."""
    
    def test_save_model_creates_file(self, trained_model, temp_dir):
        """Test that save_model creates a file."""
        save_path = temp_dir / "test_model.pkl"
        
        save_model(trained_model, str(save_path))
        
        assert save_path.exists()
    
    def test_save_model_creates_parent_dirs(self, trained_model, temp_dir):
        """Test that save_model creates parent directories."""
        save_path = temp_dir / "models" / "subdir" / "test_model.pkl"
        
        save_model(trained_model, str(save_path))
        
        assert save_path.exists()
        assert save_path.parent.exists()
    
    def test_load_model_returns_model(self, trained_model, temp_dir):
        """Test that load_model returns a model."""
        save_path = temp_dir / "test_model.pkl"
        save_model(trained_model, str(save_path))
        
        loaded_model = load_model(str(save_path))
        
        assert loaded_model is not None
        assert isinstance(loaded_model, type(trained_model))
    
    def test_save_and_load_preserves_predictions(self, trained_model, sample_training_data, temp_dir):
        """Test that saved/loaded model makes same predictions."""
        X, y = sample_training_data
        save_path = temp_dir / "test_model.pkl"
        
        # Get predictions from original model
        original_predictions = trained_model.predict_proba(X)
        
        # Save and load
        save_model(trained_model, str(save_path))
        loaded_model = load_model(str(save_path))
        
        # Get predictions from loaded model
        loaded_predictions = loaded_model.predict_proba(X)
        
        # Should be identical
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)


class TestModelComparison:
    """Tests comparing different models."""
    
    def test_logistic_vs_random_forest(self, sample_training_data):
        """Test that both model types can be trained and evaluated."""
        X, y = sample_training_data
        
        # Train both models
        lr_model = train_baseline_model(X, y)
        rf_model = train_random_forest(X, y, tune_hyperparams=False)
        
        # Evaluate both
        lr_metrics = evaluate_model(lr_model, X, y)
        rf_metrics = evaluate_model(rf_model, X, y)
        
        # Both should produce valid metrics
        assert 0 <= lr_metrics['roc_auc'] <= 1
        assert 0 <= rf_metrics['roc_auc'] <= 1


class TestIntegration:
    """Integration tests for model training pipeline."""
    
    def test_complete_training_pipeline(self, sample_training_data, temp_dir):
        """Test complete model training and evaluation pipeline."""
        X, y = sample_training_data
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        model = train_baseline_model(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save
        save_path = temp_dir / "model.pkl"
        save_model(model, str(save_path))
        
        # Load
        loaded_model = load_model(str(save_path))
        
        # Verify loaded model works
        loaded_metrics = evaluate_model(loaded_model, X_test, y_test)
        
        assert metrics['roc_auc'] == loaded_metrics['roc_auc']
    
    def test_train_multiple_models_and_select_best(self, sample_training_data):
        """Test training multiple models and comparing them."""
        X, y = sample_training_data
        
        # Train multiple models
        lr_model = train_baseline_model(X, y)
        rf_model = train_random_forest(X, y, tune_hyperparams=False)
        
        # Evaluate all
        lr_metrics = evaluate_model(lr_model, X, y)
        rf_metrics = evaluate_model(rf_model, X, y)
        
        # Select best based on ROC-AUC
        best_model = lr_model if lr_metrics['roc_auc'] > rf_metrics['roc_auc'] else rf_model
        
        assert best_model is not None
