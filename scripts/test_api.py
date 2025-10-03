#!/usr/bin/env python
"""
Simple API Test Script

Tests all endpoints of the Flask API to ensure it's working correctly.

Usage:
    # Test running API (from project root)
    python scripts/test_api.py
    
    # Test specific host/port
    python scripts/test_api.py --host localhost --port 5000
"""

import argparse
import json
import sys
import time

try:
    import requests
except ImportError:
    print("❌ Error: 'requests' library not installed")
    print("   Install with: conda run -n ml-conda pip install requests")
    sys.exit(1)


def test_health(base_url):
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Model: {data.get('model')}")
            return True
        else:
            print(f"❌ Health check failed: Status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False


def test_model_info(base_url):
    """Test model info endpoint"""
    print("\nTesting /model/info endpoint...")
    try:
        response = requests.get(f"{base_url}/model/info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Model Type: {data.get('model_type')}")
            print(f"   ROC-AUC: {data.get('roc_auc')}")
            print(f"   Precision: {data.get('precision')}")
            print(f"   Recall: {data.get('recall')}")
            return True
        else:
            print(f"❌ Model info failed: Status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False


def test_predict_single(base_url):
    """Test single prediction endpoint"""
    print("\nTesting /predict endpoint...")
    
    # Sample customer data
    customer = {
        "customerID": "TEST-0001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 70.35
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=customer,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful")
            print(f"   Customer ID: {data.get('customerID')}")
            print(f"   Churn Probability: {data.get('churn_probability_pct')}")
            print(f"   Prediction: {data.get('churn_prediction')}")
            print(f"   Risk Tier: {data.get('risk_tier')}")
            return True
        else:
            print(f"❌ Prediction failed: Status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False


def test_predict_batch(base_url):
    """Test batch prediction endpoint"""
    print("\nTesting /predict/batch endpoint...")
    
    # Sample batch of customers
    batch = {
        "customers": [
            {
                "customerID": "BATCH-001",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 99.95,
                "TotalCharges": 2398.80
            },
            {
                "customerID": "BATCH-002",
                "gender": "Female",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 45.50,
                "TotalCharges": 45.50
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict/batch",
            json=batch,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch prediction successful")
            print(f"   Total customers: {data.get('count')}")
            print(f"   High risk: {data.get('high_risk_count')}")
            print(f"   Medium risk: {data.get('medium_risk_count')}")
            print(f"   Low risk: {data.get('low_risk_count')}")
            return True
        else:
            print(f"❌ Batch prediction failed: Status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test Flask API endpoints')
    parser.add_argument('--host', default='localhost', help='API host (default: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='API port (default: 5000)')
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print("="*70)
    print("🧪 TESTING CUSTOMER CHURN PREDICTION API")
    print("="*70)
    print(f"API URL: {base_url}")
    print("="*70)
    print()
    
    # Run all tests
    tests = [
        ("Health Check", lambda: test_health(base_url)),
        ("Model Info", lambda: test_model_info(base_url)),
        ("Single Prediction", lambda: test_predict_single(base_url)),
        ("Batch Prediction", lambda: test_predict_batch(base_url)),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
        time.sleep(0.5)  # Small delay between tests
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print("-"*70)
    print(f"Total: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n🎉 All tests passed! API is working correctly.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the API.")
        sys.exit(1)


if __name__ == '__main__':
    main()
