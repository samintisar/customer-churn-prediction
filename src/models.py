"""
Model Training and Evaluation Module

This module handles model training, evaluation, and comparison.

Models:
- Baseline: Logistic Regression
- Advanced: Random Forest Classifier

Key Functions:
- train_baseline_model: Train logistic regression
- train_random_forest: Train random forest with hyperparameter tuning
- evaluate_model: Calculate metrics (ROC-AUC, precision, recall, top-decile)
- generate_predictions: Produce churn probabilities for new data
- save_model: Persist trained models
- load_model: Load saved models

Usage:
    from src.models import train_baseline_model, evaluate_model
    
    model = train_baseline_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
"""

import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve
)
from typing import Dict, Any, Tuple
import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_baseline_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    **kwargs
) -> LogisticRegression:
    """
    Train baseline logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional parameters for LogisticRegression
        
    Returns:
        Trained logistic regression model
    """
    logger.info("Training baseline logistic regression model")
    
    # Convert target to binary
    y_binary = (y_train == 'Yes').astype(int)
    
    # Default parameters with class balancing
    default_params = {
        'random_state': 42,
        'max_iter': 1000,
        'class_weight': 'balanced',
        'solver': 'lbfgs'
    }
    default_params.update(kwargs)
    
    # Train model
    model = LogisticRegression(**default_params)
    model.fit(X_train, y_binary)
    
    logger.info(f"Baseline model trained successfully")
    
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    tune_hyperparams: bool = True
) -> RandomForestClassifier:
    """
    Train random forest classifier with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (for tuning)
        y_val: Validation labels (for tuning)
        tune_hyperparams: Whether to perform grid search
        
    Returns:
        Trained random forest model
    """
    logger.info("Training Random Forest model")
    
    # Convert target to binary
    y_binary = (y_train == 'Yes').astype(int)
    
    if tune_hyperparams and X_val is not None:
        logger.info("Performing hyperparameter tuning with GridSearchCV")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']
        }
        
        # Create base model
        rf_base = RandomForestClassifier(random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            rf_base,
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_binary)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
    else:
        # Train with default good parameters
        logger.info("Training with default parameters")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_binary)
    
    logger.info("Random Forest model trained successfully")
    
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model performance with comprehensive metrics.
    
    Metrics:
    - ROC-AUC score
    - Precision, Recall, F1-score
    - Top-decile precision (high-risk segment accuracy)
    - Confusion matrix
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model performance")
    
    # Convert target to binary
    y_binary = (y_test == 'Yes').astype(int)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'roc_auc': roc_auc_score(y_binary, y_pred_proba),
        'precision': precision_score(y_binary, y_pred),
        'recall': recall_score(y_binary, y_pred),
        'f1_score': f1_score(y_binary, y_pred),
        'top_decile_precision': calculate_top_decile_precision(y_binary, y_pred_proba),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_binary, y_pred)
    metrics['tn'] = int(cm[0, 0])
    metrics['fp'] = int(cm[0, 1])
    metrics['fn'] = int(cm[1, 0])
    metrics['tp'] = int(cm[1, 1])
    
    # Log metrics
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"Top-Decile Precision: {metrics['top_decile_precision']:.4f}")
    
    return metrics


def calculate_top_decile_precision(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    decile: int = 10
) -> float:
    """
    Calculate precision for top risk decile (most valuable metric for retention).
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        decile: Which decile to evaluate (10 = top 10%)
        
    Returns:
        Precision score for the top decile
    """
    # Calculate threshold for top decile
    threshold = np.percentile(y_pred_proba, 100 - decile)
    
    # Get top decile predictions
    top_decile_mask = y_pred_proba >= threshold
    
    # Calculate precision for top decile
    if top_decile_mask.sum() == 0:
        return 0.0
    
    precision = y_true[top_decile_mask].mean()
    
    return precision


def save_model(model: Any, filepath: str) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model object
        filepath: Output path for model file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to saved model file
        
    Returns:
        Loaded model object
    """
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model

