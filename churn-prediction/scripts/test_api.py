"""
Test script for the churn prediction API
"""
import requests
import json
import pandas as pd
from pathlib import Path

# API endpoint
BASE_URL = "http://localhost:5001"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_single_prediction():
    """Test single prediction with complete customer data"""
    # Complete customer data matching the training dataset
    customer_data = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=customer_data,
        headers={"Content-Type": "application/json"}
    )
    
    print("Single Prediction:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_batch_prediction():
    """Test batch prediction"""
    # Load sample data
    data_path = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(data_path)
    
    # Take first 5 customers
    sample_customers = df.head(5).to_dict('records')
    
    batch_data = {
        "customers": sample_customers
    }
    
    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json=batch_data,
        headers={"Content-Type": "application/json"}
    )
    
    print("Batch Prediction:")
    result = response.json()
    if 'error' not in result:
        print(f"Total customers: {result['summary']['total_customers']}")
        print(f"High risk count: {result['summary']['high_risk_count']}")
        print(f"Total revenue at risk: ${result['summary']['total_revenue_at_risk']:,.2f}")
        print("\nFirst 3 predictions:")
        for pred in result['predictions'][:3]:
            print(f"  Customer {pred['customerID']}: {pred['churn_probability']:.2%} risk")
    else:
        print(result)
    print()

def test_update_threshold():
    """Test threshold update"""
    new_threshold = 0.35
    
    response = requests.post(
        f"{BASE_URL}/update_threshold",
        json={"threshold": new_threshold},
        headers={"Content-Type": "application/json"}
    )
    
    print("Update Threshold:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Testing Churn Prediction API\n")
    
    # Run tests
    test_health()
    test_single_prediction()
    test_batch_prediction()
    test_update_threshold()