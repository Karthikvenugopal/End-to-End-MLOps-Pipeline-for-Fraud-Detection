"""
Unit tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from serving.api import app

client = TestClient(app)

class TestAPI:
    """Test cases for API endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
        
    def test_predict_endpoint(self):
        """Test prediction endpoint."""
        transaction_data = {
            "TransactionAmt": 100.0,
            "ProductCD": "W",
            "card1": 12345,
            "card2": 123,
            "card3": 12,
            "card4": "visa",
            "card5": 12,
            "card6": "debit",
            "addr1": 123,
            "addr2": 12,
            "dist1": 5.0,
            "dist2": 10.0,
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "gmail.com"
        }
        
        response = client.post("/predict", json=transaction_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "confidence" in data
        
    def test_batch_predict(self):
        """Test batch prediction endpoint."""
        transactions = [
            {
                "TransactionAmt": 100.0,
                "ProductCD": "W",
                "card1": 12345,
                "card2": 123,
                "card3": 12,
                "card4": "visa",
                "card5": 12,
                "card6": "debit",
                "addr1": 123,
                "addr2": 12,
                "dist1": 5.0,
                "dist2": 10.0,
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "gmail.com"
            }
        ]
        
        response = client.post("/predict_batch", json=transactions)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 1

if __name__ == "__main__":
    pytest.main([__file__])
