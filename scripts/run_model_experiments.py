"""
Model Training and Experimentation Script

This script implements the model training pipeline including:
- Baseline: Logistic Regression
- Advanced: Random Forest Classifier
- Model comparison and evaluation
- Feature importance analysis
- Model saving
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import custom modules
from src.data_loader import load_raw_data, clean_data, split_data
from src.feature_engineering import FeatureEngineer
from src.models import train_baseline_model, train_random_forest, evaluate_model, save_model
from src.explainability import plot_feature_importance, generate_shap_summary

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def main():
    """Main execution function."""
    
    print("=" * 80)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("=" * 80)
    
    # ============================================================================
    # 1. DATA PREPARATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("1. DATA PREPARATION")
    print("=" * 80)
    
    print("\nLoading processed data splits...")
    
    # Load the data
    train_data = pd.read_csv('data/processed/train.csv')
    val_data = pd.read_csv('data/processed/val.csv')
    test_data = pd.read_csv('data/processed/test.csv')
    
    # Separate features and target
    X_train = train_data.drop('Churn', axis=1)
    y_train = train_data['Churn']
    
    X_val = val_data.drop('Churn', axis=1)
    y_val = val_data['Churn']
    
    X_test = test_data.drop('Churn', axis=1)
    y_test = test_data['Churn']
    
    print(f"\n📊 Data Shapes:")
    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]:,} samples, {X_val.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
    
    print(f"\n🎯 Target Distribution (Train):")
    print(f"  Churn = No:  {(y_train == 'No').sum():,} ({(y_train == 'No').mean():.1%})")
    print(f"  Churn = Yes: {(y_train == 'Yes').sum():,} ({(y_train == 'Yes').mean():.1%})")
    
    # Store feature names
    feature_names = X_train.columns.tolist()
    
    # ============================================================================
    # 2. BASELINE MODEL: LOGISTIC REGRESSION
    # ============================================================================
    print("\n" + "=" * 80)
    print("2. BASELINE MODEL: LOGISTIC REGRESSION")
    print("=" * 80)
    
    baseline_model = train_baseline_model(X_train, y_train)
    
    print("\n✓ Baseline model trained successfully")
    print(f"  Model type: {type(baseline_model).__name__}")
    print(f"  Number of features: {len(baseline_model.coef_[0])}")
    
    # Evaluate baseline
    print("\n📊 Evaluating Baseline Model on Test Set...")
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test)
    
    print("\n📈 Baseline Model Performance:")
    print(f"  ROC-AUC Score:        {baseline_metrics['roc_auc']:.4f}")
    print(f"  Precision:            {baseline_metrics['precision']:.4f}")
    print(f"  Recall:               {baseline_metrics['recall']:.4f}")
    print(f"  F1 Score:             {baseline_metrics['f1_score']:.4f}")
    print(f"  Top-Decile Precision: {baseline_metrics['top_decile_precision']:.4f}")
    
    print(f"\n🎯 Confusion Matrix:")
    print(f"  True Negatives:  {baseline_metrics['tn']:,}")
    print(f"  False Positives: {baseline_metrics['fp']:,}")
    print(f"  False Negatives: {baseline_metrics['fn']:,}")
    print(f"  True Positives:  {baseline_metrics['tp']:,}")
    
    # ============================================================================
    # 3. ADVANCED MODEL: RANDOM FOREST
    # ============================================================================
    print("\n" + "=" * 80)
    print("3. ADVANCED MODEL: RANDOM FOREST")
    print("=" * 80)
    print("\nNote: This may take a few minutes due to hyperparameter tuning...")
    
    rf_model = train_random_forest(
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        tune_hyperparams=True
    )
    
    print("\n✓ Random Forest model trained successfully")
    print(f"  Model type: {type(rf_model).__name__}")
    print(f"  Number of estimators: {rf_model.n_estimators}")
    print(f"  Max depth: {rf_model.max_depth}")
    print(f"  Min samples split: {rf_model.min_samples_split}")
    
    # Evaluate Random Forest
    print("\n📊 Evaluating Random Forest Model on Test Set...")
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    
    print("\n📈 Random Forest Model Performance:")
    print(f"  ROC-AUC Score:        {rf_metrics['roc_auc']:.4f}")
    print(f"  Precision:            {rf_metrics['precision']:.4f}")
    print(f"  Recall:               {rf_metrics['recall']:.4f}")
    print(f"  F1 Score:             {rf_metrics['f1_score']:.4f}")
    print(f"  Top-Decile Precision: {rf_metrics['top_decile_precision']:.4f}")
    
    print(f"\n🎯 Confusion Matrix:")
    print(f"  True Negatives:  {rf_metrics['tn']:,}")
    print(f"  False Positives: {rf_metrics['fp']:,}")
    print(f"  False Negatives: {rf_metrics['fn']:,}")
    print(f"  True Positives:  {rf_metrics['tp']:,}")
    
    # ============================================================================
    # 4. MODEL COMPARISON
    # ============================================================================
    print("\n" + "=" * 80)
    print("4. MODEL COMPARISON")
    print("=" * 80)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Metric': ['ROC-AUC', 'Precision', 'Recall', 'F1 Score', 'Top-Decile Precision'],
        'Logistic Regression': [
            baseline_metrics['roc_auc'],
            baseline_metrics['precision'],
            baseline_metrics['recall'],
            baseline_metrics['f1_score'],
            baseline_metrics['top_decile_precision']
        ],
        'Random Forest': [
            rf_metrics['roc_auc'],
            rf_metrics['precision'],
            rf_metrics['recall'],
            rf_metrics['f1_score'],
            rf_metrics['top_decile_precision']
        ]
    })
    
    # Calculate improvement
    comparison_df['Improvement (%)'] = (
        (comparison_df['Random Forest'] - comparison_df['Logistic Regression']) / 
        comparison_df['Logistic Regression'] * 100
    )
    
    print("\n📊 Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Bar plot comparison
    metrics_to_plot = ['ROC-AUC', 'Precision', 'Recall', 'F1 Score', 'Top-Decile\nPrecision']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    axes[0].bar(x - width/2, comparison_df['Logistic Regression'], width, 
                label='Logistic Regression', alpha=0.8, color='steelblue')
    axes[0].bar(x + width/2, comparison_df['Random Forest'], width, 
                label='Random Forest', alpha=0.8, color='darkorange')
    axes[0].set_xlabel('Metrics', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_to_plot)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Improvement plot
    colors = ['green' if val > 0 else 'red' for val in comparison_df['Improvement (%)']]
    axes[1].bar(metrics_to_plot, comparison_df['Improvement (%)'], alpha=0.8, color=colors)
    axes[1].set_xlabel('Metrics', fontsize=12)
    axes[1].set_ylabel('Improvement (%)', fontsize=12)
    axes[1].set_title('Random Forest Improvement over Baseline', fontsize=14, fontweight='bold')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparison chart saved to reports/figures/model_comparison.png")
    plt.close()
    
    # ============================================================================
    # 5. FEATURE IMPORTANCE AND EXPLAINABILITY
    # ============================================================================
    print("\n" + "=" * 80)
    print("5. FEATURE IMPORTANCE AND EXPLAINABILITY")
    print("=" * 80)
    
    print("\nGenerating feature importance plot...")
    plot_feature_importance(
        rf_model, 
        feature_names, 
        top_n=20,
        save_path='reports/figures/feature_importance.png'
    )
    plt.close()
    print("✓ Feature importance plot saved to reports/figures/feature_importance.png")
    
    print("\nGenerating SHAP summary plot (this may take a minute)...")
    generate_shap_summary(
        rf_model,
        X_test,
        feature_names=feature_names,
        save_path='reports/figures/shap_summary.png',
        max_display=20
    )
    plt.close()
    print("✓ SHAP summary plot saved to reports/figures/shap_summary.png")
    
    # ============================================================================
    # 6. MODEL SELECTION AND SAVING
    # ============================================================================
    print("\n" + "=" * 80)
    print("6. MODEL SELECTION AND SAVING")
    print("=" * 80)
    
    # Select best model based on ROC-AUC
    if rf_metrics['roc_auc'] > baseline_metrics['roc_auc']:
        best_model = rf_model
        best_model_name = "Random Forest"
        best_metrics = rf_metrics
    else:
        best_model = baseline_model
        best_model_name = "Logistic Regression"
        best_metrics = baseline_metrics
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"  ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"  Top-Decile Precision: {best_metrics['top_decile_precision']:.4f}")
    
    # Save models
    save_model(best_model, 'models/churn_predictor.pkl')
    print("\n✓ Best model saved to models/churn_predictor.pkl")
    
    save_model(baseline_model, 'models/logistic_regression.pkl')
    save_model(rf_model, 'models/random_forest.pkl')
    print("✓ All models saved successfully")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    print("\n🎯 KEY FINDINGS:")
    print(f"  • Best Model: {best_model_name}")
    print(f"  • ROC-AUC Score: {best_metrics['roc_auc']:.4f}")
    print(f"  • Top-Decile Precision: {best_metrics['top_decile_precision']:.4f}")
    print(f"  • Improvement over baseline: {comparison_df.loc[0, 'Improvement (%)']:.2f}%")
    
    print("\n📈 BUSINESS VALUE:")
    print(f"  • Of top 10% highest-risk customers, {best_metrics['top_decile_precision']:.1%} will actually churn")
    print("  • Enables efficient targeting for retention campaigns")
    print("  • Reduces wasted retention spending on loyal customers")
    
    print("\n📁 OUTPUTS GENERATED:")
    print("  ✓ models/churn_predictor.pkl")
    print("  ✓ models/logistic_regression.pkl")
    print("  ✓ models/random_forest.pkl")
    print("  ✓ reports/figures/model_comparison.png")
    print("  ✓ reports/figures/feature_importance.png")
    print("  ✓ reports/figures/shap_summary.png")
    
    print("\n" + "=" * 80)
    print("✓ MODEL TRAINING COMPLETE!")
    print("=" * 80)
    
    return best_model, baseline_model, rf_model, comparison_df


if __name__ == "__main__":
    best_model, baseline_model, rf_model, comparison_df = main()

