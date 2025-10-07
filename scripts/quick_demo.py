"""
Quick demo script for testing individual components.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_data_processing():
    """Test data processing components."""
    print("🧪 Testing Data Processing...")
    
    from data.prepare_data import FraudDataProcessor
    
    processor = FraudDataProcessor()
    data = processor._create_synthetic_data(n_samples=100, fraud_rate=0.1)
    
    print(f"✅ Created {len(data)} samples with {data['isFraud'].sum()} fraud cases")
    return True

def test_model_components():
    """Test model components."""
    print("🧪 Testing Model Components...")
    
    try:
        from models.train_xgboost import XGBoostFraudDetector
        from models.train_neural_network import NeuralNetworkFraudDetector
        
        xgb_detector = XGBoostFraudDetector()
        nn_detector = NeuralNetworkFraudDetector()
        
        print("✅ XGBoost detector initialized")
        print("✅ Neural network detector initialized")
        return True
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

def test_api_components():
    """Test API components."""
    print("🧪 Testing API Components...")
    
    try:
        from serving.api import app
        print("✅ FastAPI app created")
        return True
    except Exception as e:
        print(f"❌ API initialization failed: {e}")
        return False

def test_monitoring():
    """Test monitoring components."""
    print("🧪 Testing Monitoring Components...")
    
    try:
        from monitoring.drift_detection import DriftDetector
        
        detector = DriftDetector()
        print("✅ Drift detector initialized")
        return True
    except Exception as e:
        print(f"❌ Monitoring initialization failed: {e}")
        return False

def main():
    """Run quick component tests."""
    print("🚀 Quick Component Tests")
    print("=" * 30)
    
    tests = [
        test_data_processing,
        test_model_components,
        test_api_components,
        test_monitoring
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All components working correctly!")
    else:
        print("⚠️ Some components need attention")

if __name__ == "__main__":
    main()
