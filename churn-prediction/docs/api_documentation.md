# Churn Prediction API Documentation

## Base URL
http://localhost:5001

## Authentication
No authentication required (for demo purposes)

## Endpoints

### 1. Health Check

Check if the API is running and model is loaded.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_version": "production",
  "model_loaded": true
}

Example:
curl http://localhost:5001/health

2. Single Customer Prediction
Predict churn probability for a single customer.
Endpoint: POST /predict
Headers:

Content-Type: application/json

Request Body:
json{
  "customerID": "string",
  "gender": "Male|Female",
  "SeniorCitizen": 0|1,
  "Partner": "Yes|No",
  "Dependents": "Yes|No",
  "tenure": integer,
  "PhoneService": "Yes|No",
  "MultipleLines": "No|Yes|No phone service",
  "InternetService": "DSL|Fiber optic|No",
  "OnlineSecurity": "Yes|No|No internet service",
  "OnlineBackup": "Yes|No|No internet service",
  "DeviceProtection": "Yes|No|No internet service",
  "TechSupport": "Yes|No|No internet service",
  "StreamingTV": "Yes|No|No internet service",
  "StreamingMovies": "Yes|No|No internet service",
  "Contract": "Month-to-month|One year|Two year",
  "PaperlessBilling": "Yes|No",
  "PaymentMethod": "Electronic check|Mailed check|Bank transfer (automatic)|Credit card (automatic)",
  "MonthlyCharges": float,
  "TotalCharges": "string"
}
Response:
json{
  "customer_id": "string",
  "churn_probability": 0.742,
  "will_churn": true,
  "recommendation": "High priority for retention",
  "predicted_at": "2024-08-19T15:30:45.123456"
}
Example:
bashcurl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "1234-ABCDE",
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
  }'

3. Batch Prediction
Predict churn probability for multiple customers.
Endpoint: POST /predict_batch
Headers:

Content-Type: application/json

Request Body:
json{
  "customers": [
    {
      "customerID": "string",
      "gender": "Male|Female",
      ... (same fields as single prediction)
    },
    ...
  ]
}
Response:
json{
  "predictions": [
    {
      "customerID": "string",
      "churn_probability": 0.742,
      "will_churn": true,
      "risk_segment": "Very High",
      "monthly_revenue": 70.0,
      "annual_revenue_at_risk": 623.28,
      "retention_priority": 1
    },
    ...
  ],
  "summary": {
    "total_customers": 100,
    "high_risk_count": 23,
    "total_revenue_at_risk": 45678.90
  }
}

4. Update Threshold
Update the business threshold for churn classification.
Endpoint: POST /update_threshold
Headers:

Content-Type: application/json

Request Body:
json{
  "threshold": 0.35
}
Response:
json{
  "status": "success",
  "new_threshold": 0.35
}
Example:
bashcurl -X POST http://localhost:5000/update_threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.35}'

Error Responses
400 Bad Request
json{
  "error": "Missing required fields: ['customerID', 'tenure']"
}
500 Internal Server Error
json{
  "error": "Internal server error"
}

Rate Limiting
No rate limiting implemented (demo purposes)

Notes

All numeric fields should be provided as numbers, not strings (except TotalCharges)
TotalCharges is expected as a string for compatibility with the original dataset
The API uses a pre-trained model stored in /models/model_production/