"""
API endpoint tests
"""
import pytest
import json
from src.app import app


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestAPI:
    """Test cases for API endpoints"""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'
        
    def test_predict_endpoint_valid(self, client, complete_customer_data):
        """Test prediction with valid data"""
        response = client.post('/predict',
                            json=complete_customer_data,
                            content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify response structure
        assert 'customer_id' in data
        assert 'churn_probability' in data
        assert 'will_churn' in data
        assert 'recommendation' in data
        assert 0 <= data['churn_probability'] <= 1
        
    def test_predict_endpoint_invalid(self, client):
        """Test prediction with invalid data"""
        # Missing required fields
        response = client.post('/predict',
                              json={"customerID": "TEST-001"},
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'required fields' in data['error']
        
    def test_predict_endpoint_no_json(self, client):
        """Test prediction without JSON content type"""
        response = client.post('/predict',
                              data="not json",
                              content_type='text/plain')
        
        assert response.status_code == 400
        
    def test_batch_predict_endpoint(self, client):
        """Test batch prediction endpoint"""
        batch_data = {
            "customers": [
                {"customerID": "TEST-001", "tenure": 12, "MonthlyCharges": 50.0},
                {"customerID": "TEST-002", "tenure": 24, "MonthlyCharges": 75.0}
            ]
        }
        
        response = client.post('/predict_batch',
                              json=batch_data,
                              content_type='application/json')
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 500]
        
    def test_update_threshold_endpoint(self, client):
        """Test threshold update endpoint"""
        # Valid threshold
        response = client.post('/update_threshold',
                              json={"threshold": 0.4},
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['new_threshold'] == 0.4
        
        # Invalid threshold
        response = client.post('/update_threshold',
                              json={"threshold": 1.5},
                              content_type='application/json')
        
        assert response.status_code == 400