"""
Model Explainability Module

This module provides interpretability tools for churn prediction models.

Key Functions:
- get_feature_importance: Extract feature importance from tree-based models
- plot_feature_importance: Visualize top N most important features
- generate_shap_summary: Create SHAP summary plot
- generate_shap_waterfall: SHAP waterfall for individual prediction
- explain_prediction: Human-readable explanation for a customer's churn risk

Usage:
    from src.explainability import plot_feature_importance, generate_shap_summary
    
    plot_feature_importance(model, feature_names, top_n=20)
    generate_shap_summary(model, X_test, feature_names)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import logging
from typing import List, Optional, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_feature_importance(
    model: Any,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract feature importance from trained model.
    
    Args:
        model: Trained model (must have feature_importances_ or coef_)
        feature_names: List of feature names
        
    Returns:
        DataFrame with features and importance scores, sorted
    """
    logger.info("Extracting feature importance")
    
    # Check if model has feature_importances_ (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    # Check if model has coef_ (linear models)
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model doesn't have feature_importances_ or coef_ attribute")
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    logger.info(f"Extracted importance for {len(feature_names)} features")
    
    return importance_df


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Create and save feature importance bar plot.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to display
        save_path: Optional path to save figure
    """
    logger.info(f"Plotting top {top_n} feature importances")
    
    # Get feature importance
    importance_df = get_feature_importance(model, feature_names)
    
    # Select top N features
    top_features = importance_df.head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Horizontal bar plot
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()  # Highest importance at top
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Most Important Features')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def generate_shap_summary(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str] = None,
    save_path: Optional[str] = None,
    max_display: int = 20
) -> None:
    """
    Generate SHAP summary plot showing feature impact on predictions.
    
    Args:
        model: Trained model
        X: Feature matrix (sample for SHAP calculation)
        feature_names: List of feature names
        save_path: Optional path to save figure
        max_display: Maximum number of features to display
    """
    logger.info("Generating SHAP summary plot")
    
    # Use sample if dataset is too large (SHAP is computationally expensive)
    if len(X) > 500:
        logger.info(f"Sampling 500 rows from {len(X)} for SHAP calculation")
        X_sample = X.sample(500, random_state=42)
    else:
        X_sample = X
    
    # Create SHAP explainer
    try:
        # Try TreeExplainer for tree-based models (faster)
        explainer = shap.TreeExplainer(model)
        logger.info("Using TreeExplainer")
    except:
        # Fallback to general explainer
        explainer = shap.Explainer(model, X_sample)
        logger.info("Using general Explainer")
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, select positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Create summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        X_sample,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"SHAP summary plot saved to {save_path}")
    
    plt.show()


def generate_shap_waterfall(
    model: Any,
    X_sample: pd.DataFrame,
    customer_idx: int,
    feature_names: List[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Generate SHAP waterfall plot for a single customer prediction.
    
    Args:
        model: Trained model
        X_sample: Feature matrix sample
        customer_idx: Index of customer to explain
        feature_names: List of feature names
        save_path: Optional path to save figure
    """
    logger.info(f"Generating SHAP waterfall plot for customer {customer_idx}")
    
    # Create SHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
    except:
        explainer = shap.Explainer(model, X_sample)
    
    # Calculate SHAP values for single sample
    shap_values = explainer.shap_values(X_sample.iloc[[customer_idx]])
    
    # For binary classification, select positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Create waterfall plot
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
            data=X_sample.iloc[customer_idx].values,
            feature_names=feature_names
        ),
        show=False
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"SHAP waterfall plot saved to {save_path}")
    
    plt.show()


def explain_prediction(
    model: Any,
    X_sample: pd.DataFrame,
    feature_names: List[str],
    customer_id: str,
    top_factors: int = 5
) -> str:
    """
    Generate human-readable explanation for a customer's churn risk.
    
    Args:
        model: Trained model
        X_sample: Single customer feature vector
        feature_names: List of feature names
        customer_id: Customer identifier
        top_factors: Number of top contributing factors to include
        
    Returns:
        Text explanation of churn risk drivers
    """
    # Get prediction
    churn_prob = model.predict_proba(X_sample)[0, 1]
    
    # Get feature importance for this prediction
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]
        else:
            shap_values = shap_values[0]
    except:
        # Fallback to model feature importance
        if hasattr(model, 'feature_importances_'):
            shap_values = model.feature_importances_
        else:
            shap_values = np.abs(model.coef_[0])
    
    # Get top contributing features
    feature_contributions = pd.DataFrame({
        'feature': feature_names,
        'contribution': np.abs(shap_values)
    }).sort_values('contribution', ascending=False).head(top_factors)
    
    # Build explanation
    explanation = f"Customer {customer_id} has a {churn_prob:.1%} churn probability.\n\n"
    explanation += f"Top {top_factors} factors contributing to this prediction:\n"
    
    for idx, row in feature_contributions.iterrows():
        explanation += f"  - {row['feature']}\n"
    
    return explanation


def create_explainability_report(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: List[str] = None,
    output_dir: str = 'reports/figures'
) -> None:
    """
    Generate complete explainability report with all visualizations.
    
    Creates:
    - Feature importance plot
    - SHAP summary plot
    - Sample waterfall plots
    
    Args:
        model: Trained model
        X_test: Test set features
        feature_names: List of feature names
        output_dir: Directory to save figures
    """
    logger.info("Creating comprehensive explainability report")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if feature_names is None:
        feature_names = X_test.columns.tolist()
    
    # 1. Feature importance plot
    logger.info("Generating feature importance plot")
    plot_feature_importance(
        model,
        feature_names,
        top_n=20,
        save_path=output_path / 'feature_importance.png'
    )
    
    # 2. SHAP summary plot
    logger.info("Generating SHAP summary plot")
    generate_shap_summary(
        model,
        X_test,
        feature_names,
        save_path=output_path / 'shap_summary.png'
    )
    
    # 3. Sample waterfall plots (for a few customers)
    logger.info("Generating sample waterfall plots")
    for i in range(min(3, len(X_test))):
        generate_shap_waterfall(
            model,
            X_test,
            customer_idx=i,
            feature_names=feature_names,
            save_path=output_path / f'shap_waterfall_customer_{i}.png'
        )
    
    logger.info(f"Explainability report saved to {output_dir}")

