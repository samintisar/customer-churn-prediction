"""
Quick Deployment Setup

This script helps you quickly set up and deploy the Logistic Regression model.

Usage:
    python scripts/deploy_model.py [option]
    
Options:
    --switch-to-lr     Switch active model to Logistic Regression
    --test-api         Test the Flask API
    --test-dashboard   Check if dashboard is ready
    --all              Run all checks
"""

import shutil
import joblib
from pathlib import Path
import sys
import argparse

sys.path.append(str(Path(__file__).parent.parent))


def switch_to_logistic_regression():
    """Switch the active model to Logistic Regression"""
    print("\n" + "="*60)
    print("🔄 Switching to Logistic Regression Model")
    print("="*60)
    
    source = Path('models/logistic_regression.pkl')
    target = Path('models/churn_predictor.pkl')
    
    if not source.exists():
        print(f"❌ Error: Logistic Regression model not found at {source}")
        print("   Please train the model first using 02_model_experiments.ipynb")
        return False
    
    try:
        shutil.copy(source, target)
        print(f"✅ Successfully copied:")
        print(f"   {source} → {target}")
        
        # Verify
        model = joblib.load(target)
        print(f"\n✅ Model verified:")
        print(f"   Type: {type(model).__name__}")
        print(f"   Features: {len(model.coef_[0]) if hasattr(model, 'coef_') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_api():
    """Test if Flask API dependencies are ready"""
    print("\n" + "="*60)
    print("🧪 Testing API Setup")
    print("="*60)
    
    try:
        import flask
        print(f"✅ Flask installed (version {flask.__version__})")
    except ImportError:
        print("❌ Flask not installed")
        print("   Install with: pip install flask")
        return False
    
    # Check if models exist
    lr_model = Path('models/logistic_regression.pkl')
    fe_model = Path('models/feature_engineer.pkl')
    
    if not lr_model.exists():
        print(f"❌ Logistic Regression model not found: {lr_model}")
        return False
    else:
        print(f"✅ Logistic Regression model found: {lr_model}")
    
    if not fe_model.exists():
        print(f"❌ Feature Engineer not found: {fe_model}")
        return False
    else:
        print(f"✅ Feature Engineer found: {fe_model}")
    
    # Check if API file exists
    api_file = Path('app/api.py')
    if not api_file.exists():
        print(f"❌ API file not found: {api_file}")
        return False
    else:
        print(f"✅ API file found: {api_file}")
    
    print("\n✅ API is ready to deploy!")
    print("   Start with: python app/api.py")
    
    return True


def test_dashboard():
    """Test if Streamlit dashboard is ready"""
    print("\n" + "="*60)
    print("🧪 Testing Dashboard Setup")
    print("="*60)
    
    try:
        import streamlit
        print(f"✅ Streamlit installed (version {streamlit.__version__})")
    except ImportError:
        print("❌ Streamlit not installed")
        print("   Install with: pip install streamlit")
        return False
    
    # Check if active model exists
    active_model = Path('models/churn_predictor.pkl')
    
    if not active_model.exists():
        print(f"❌ Active model not found: {active_model}")
        print("   Run: python scripts/deploy_model.py --switch-to-lr")
        return False
    else:
        print(f"✅ Active model found: {active_model}")
    
    # Check if dashboard file exists
    dashboard_file = Path('app/dashboard.py')
    if not dashboard_file.exists():
        print(f"❌ Dashboard file not found: {dashboard_file}")
        return False
    else:
        print(f"✅ Dashboard file found: {dashboard_file}")
    
    print("\n✅ Dashboard is ready to launch!")
    print("   Start with: streamlit run app/dashboard.py")
    
    return True


def print_deployment_summary():
    """Print deployment options summary"""
    print("\n" + "="*60)
    print("📋 Deployment Options Summary")
    print("="*60)
    
    print("\n1️⃣  Interactive Dashboard (Streamlit)")
    print("   Best for: Demos, internal tools, exploration")
    print("   Command: streamlit run app/dashboard.py")
    print("   URL:     http://localhost:8501")
    
    print("\n2️⃣  REST API (Flask)")
    print("   Best for: Production, integrations, automation")
    print("   Command: python app/api.py")
    print("   URL:     http://localhost:5000")
    
    print("\n3️⃣  Batch Predictions (Script)")
    print("   Best for: Regular scoring, reports, bulk processing")
    print("   Command: python scripts/batch_predict.py -i data.csv -o results.csv")
    
    print("\n4️⃣  Cloud Deployment")
    print("   - Streamlit Cloud (free)")
    print("   - Heroku (API)")
    print("   - AWS/Azure/GCP (Docker)")
    
    print("\n📖 See DEPLOYMENT_GUIDE.md for detailed instructions")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Quick Deployment Setup for Churn Prediction Model'
    )
    
    parser.add_argument(
        '--switch-to-lr',
        action='store_true',
        help='Switch active model to Logistic Regression'
    )
    parser.add_argument(
        '--test-api',
        action='store_true',
        help='Test if Flask API is ready'
    )
    parser.add_argument(
        '--test-dashboard',
        action='store_true',
        help='Test if Streamlit dashboard is ready'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all setup checks'
    )
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any([args.switch_to_lr, args.test_api, args.test_dashboard, args.all]):
        print_deployment_summary()
        parser.print_help()
        return
    
    success = True
    
    if args.switch_to_lr or args.all:
        success = switch_to_logistic_regression() and success
    
    if args.test_api or args.all:
        success = test_api() and success
    
    if args.test_dashboard or args.all:
        success = test_dashboard() and success
    
    if args.all:
        print_deployment_summary()
    
    if success:
        print("\n🎉 All checks passed! You're ready to deploy.")
    else:
        print("\n⚠️  Some checks failed. Please resolve the issues above.")


if __name__ == '__main__':
    main()
