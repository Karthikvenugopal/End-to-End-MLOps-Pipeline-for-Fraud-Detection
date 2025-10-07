"""
Unit tests for model training and evaluation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.train_xgboost import XGBoostFraudDetector
from models.train_neural_network import NeuralNetworkFraudDetector

class TestXGBoostModel:
    """Test cases for XGBoost model."""
    
    def test_initialization(self):
        """Test model initialization."""
        detector = XGBoostFraudDetector()
        assert detector.model is None
        assert detector.best_params is None
        
    def test_config_loading(self):
        """Test configuration loading."""
        detector = XGBoostFraudDetector()
        assert detector.config is not None
        assert 'models' in detector.config
        assert 'xgboost' in detector.config['models']

class TestNeuralNetworkModel:
    """Test cases for Neural Network model."""
    
    def test_initialization(self):
        """Test model initialization."""
        detector = NeuralNetworkFraudDetector()
        assert detector.model is None
        assert detector.categorical_dims is None
        assert detector.numerical_dim is None
        
    def test_config_loading(self):
        """Test configuration loading."""
        detector = NeuralNetworkFraudDetector()
        assert detector.config is not None
        assert 'models' in detector.config
        assert 'neural_network' in detector.config['models']

class TestDataProcessing:
    """Test cases for data processing."""
    
    def test_synthetic_data_creation(self):
        """Test synthetic data generation."""
        from data.prepare_data import FraudDataProcessor
        
        processor = FraudDataProcessor()
        data = processor._create_synthetic_data(n_samples=100, fraud_rate=0.1)
        
        assert len(data) == 100
        assert 'isFraud' in data.columns
        assert data['isFraud'].sum() == 10  # 10% fraud rate
        
    def test_data_preprocessing(self):
        """Test data preprocessing pipeline."""
        from data.prepare_data import FraudDataProcessor
        
        processor = FraudDataProcessor()
        processor.load_data()
        processor.preprocess_data()
        
        assert processor.train_df is not None
        assert processor.test_df is not None

if __name__ == "__main__":
    pytest.main([__file__])
