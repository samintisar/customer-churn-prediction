"""
Validate that all required files exist for the dashboard to run.
"""

from pathlib import Path

def check_file_exists(filepath, description, required=True):
    """Check if a file exists and print status."""
    path = Path(filepath)
    exists = path.exists()
    
    if exists:
        if path.is_file():
            size = path.stat().st_size / 1024  # KB
            print(f"[OK] {description}")
            print(f"     Path: {filepath}")
            print(f"     Size: {size:.1f} KB")
        else:
            print(f"[OK] {description} (directory)")
            print(f"     Path: {filepath}")
    else:
        status = "[REQUIRED]" if required else "[OPTIONAL]"
        print(f"{status} {description} - NOT FOUND")
        print(f"          Path: {filepath}")
    
    print()
    return exists

def main():
    print("=" * 80)
    print("Dashboard File Validation")
    print("=" * 80)
    print()
    
    # Check required files
    print("REQUIRED FILES:")
    print("-" * 80)
    
    model_exists = check_file_exists(
        "models/churn_predictor.pkl",
        "Trained churn prediction model",
        required=True
    )
    
    test_data_exists = check_file_exists(
        "data/processed/test.csv",
        "Processed test dataset",
        required=True
    )
    
    dashboard_exists = check_file_exists(
        "app/dashboard.py",
        "Dashboard application file",
        required=True
    )
    
    # Check optional files
    print("OPTIONAL FILES (enhance functionality):")
    print("-" * 80)
    
    fe_exists = check_file_exists(
        "models/feature_engineer.pkl",
        "Feature engineering pipeline (for individual predictions)",
        required=False
    )
    
    cleaned_data_exists = check_file_exists(
        "data/processed/cleaned_data.csv",
        "Cleaned customer data (for readable attributes)",
        required=False
    )
    
    raw_data_exists = check_file_exists(
        "data/raw/Telco-Customer-Churn.csv",
        "Raw customer data (fallback)",
        required=False
    )
    
    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    required_count = sum([model_exists, test_data_exists, dashboard_exists])
    optional_count = sum([fe_exists, cleaned_data_exists, raw_data_exists])
    
    print(f"Required files: {required_count}/3")
    print(f"Optional files: {optional_count}/3")
    print()
    
    if required_count == 3:
        print("Status: READY TO RUN")
        print()
        print("The dashboard has all required files and can be launched.")
        print()
        print("To start the dashboard, run:")
        print("  streamlit run app/dashboard.py")
        print()
        
        if not fe_exists:
            print("Note: Individual predictions will not work without feature_engineer.pkl")
            print("      Run: python scripts/run_model_experiments.py")
            print()
        
        if not cleaned_data_exists and not raw_data_exists:
            print("Note: Some customer attributes may not display without original data")
            print()
            
    else:
        print("Status: MISSING REQUIRED FILES")
        print()
        print("The dashboard cannot run. Please ensure all required files exist.")
        print()
        
        if not model_exists:
            print("To create the model:")
            print("  python scripts/run_model_experiments.py")
            print()
        
        if not test_data_exists:
            print("To create the test data:")
            print("  python scripts/test_pipeline.py")
            print()
    
    print("=" * 80)

if __name__ == "__main__":
    main()

