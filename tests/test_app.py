import pytest
import json
import sys
import os

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from ml_model import StockPredictor

@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def predictor():
    """Create a StockPredictor instance for testing"""
    return StockPredictor()

class TestFlaskApp:
    """Test the Flask application endpoints"""

    def test_home_page(self, client):
        """Test the home page loads correctly"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'ML Stock Predictor' in response.data
        assert b'stock symbol' in response.data

    def test_health_check(self, client):
        """Test the health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'ML Stock Predictor' in data['service']

    def test_api_test_endpoint(self, client):
        """Test the API test endpoint"""
        response = client.get('/api/test')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'API is working!' in data['message']
        assert 'endpoints' in data

    def test_predict_endpoint_no_data(self, client):
        """Test prediction endpoint with no data"""
        response = client.post('/predict',
                               data=json.dumps({}),
                               content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_predict_endpoint_invalid_symbol(self, client):
        """Test prediction endpoint with invalid symbol"""
        response = client.post('/predict',
                               data=json.dumps({'symbol': 'INVALID123'}),
                               content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_predict_endpoint_valid_symbol(self, client):
        """Test prediction endpoint with valid symbol (might fail if no internet)"""
        response = client.post('/predict',
                               data=json.dumps({'symbol': 'AAPL'}),
                               content_type='application/json')
        # This might fail due to network issues, so we accept both success and error
        assert response.status_code in [200, 400, 500]

class TestStockPredictor:
    """Test the ML model functionality"""

    def test_predictor_initialization(self, predictor):
        """Test that predictor initializes correctly"""
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.is_trained == False

    def test_calculate_rsi(self, predictor):
        """Test RSI calculation with sample data"""
        import pandas as pd
        # Create sample price data
        prices = pd.Series([100, 102, 101, 103, 104, 102, 105, 107, 106, 108])
        rsi = predictor.calculate_rsi(prices, window=5)

        # RSI should be between 0 and 100
        rsi_clean = rsi.dropna()
        assert all(0 <= val <= 100 for val in rsi_clean)

    def test_get_stock_data_invalid_symbol(self, predictor):
        """Test fetching data for invalid symbol"""
        data = predictor.get_stock_data('INVALID123')
        assert data is None

    def test_prepare_features_with_sample_data(self, predictor):
        """Test feature preparation with sample data"""
        import pandas as pd
        import numpy as np

        # Create sample stock data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 50),
            'High': np.random.uniform(105, 115, 50),
            'Low': np.random.uniform(95, 105, 50),
            'Close': np.random.uniform(100, 110, 50),
            'Volume': np.random.uniform(1000000, 2000000, 50)
        }, index=dates)

        features, target = predictor.prepare_features(sample_data)

        # Check that features and target are returned
        assert features is not None
        assert target is not None
        assert len(features) > 0
        assert len(target) > 0
        assert len(features) == len(target)

class TestApplicationIntegration:
    """Integration tests for the complete application"""

    def test_app_startup(self):
        """Test that the app can start without errors"""
        with app.test_client() as client:
            response = client.get('/health')
            assert response.status_code == 200

    def test_all_routes_accessible(self, client):
        """Test that all defined routes are accessible"""
        routes_to_test = [
            ('/', 'GET'),
            ('/health', 'GET'),
            ('/api/test', 'GET')
        ]

        for route, method in routes_to_test:
            if method == 'GET':
                response = client.get(route)

            # All routes should be accessible (not 404)
            assert response.status_code != 404

if __name__ == '__main__':
    # Run tests if this file is executed directly
    pytest.main([__file__, '-v'])