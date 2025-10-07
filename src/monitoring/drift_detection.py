"""
Data drift detection for fraud detection model.
Monitors model performance and data distribution changes.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging
from typing import Dict, List, Tuple
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftDetector:
    """Detects data drift in fraud detection pipeline."""
    
    def __init__(self, reference_data: pd.DataFrame = None):
        """Initialize drift detector with reference data."""
        self.reference_data = reference_data
        self.drift_threshold = 0.05  # 5% significance level
        
    def set_reference_data(self, data: pd.DataFrame):
        """Set reference data for drift detection."""
        self.reference_data = data
        logger.info(f"Reference data set with {len(data)} samples")
        
    def detect_drift(self, new_data: pd.DataFrame) -> Dict[str, bool]:
        """
        Detect drift between reference and new data.
        
        Args:
            new_data: New data to compare against reference
            
        Returns:
            Dictionary with drift detection results for each feature
        """
        if self.reference_data is None:
            logger.warning("No reference data set")
            return {}
            
        drift_results = {}
        
        # Get numerical columns
        numerical_cols = new_data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in self.reference_data.columns:
                try:
                    # Perform Kolmogorov-Smirnov test
                    ks_stat, p_value = stats.ks_2samp(
                        self.reference_data[col].dropna(),
                        new_data[col].dropna()
                    )
                    
                    # Drift detected if p-value is below threshold
                    drift_detected = p_value < self.drift_threshold
                    drift_results[col] = {
                        'drift_detected': drift_detected,
                        'p_value': p_value,
                        'ks_statistic': ks_stat
                    }
                    
                except Exception as e:
                    logger.error(f"Error detecting drift for {col}: {e}")
                    drift_results[col] = {
                        'drift_detected': False,
                        'error': str(e)
                    }
        
        return drift_results
    
    def detect_categorical_drift(self, new_data: pd.DataFrame) -> Dict[str, bool]:
        """Detect drift in categorical features using chi-square test."""
        if self.reference_data is None:
            return {}
            
        drift_results = {}
        categorical_cols = new_data.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col in self.reference_data.columns:
                try:
                    # Get value counts
                    ref_counts = self.reference_data[col].value_counts()
                    new_counts = new_data[col].value_counts()
                    
                    # Align indices and fill missing values with 0
                    all_categories = set(ref_counts.index) | set(new_counts.index)
                    ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
                    new_aligned = new_counts.reindex(all_categories, fill_value=0)
                    
                    # Perform chi-square test
                    chi2_stat, p_value = stats.chisquare(
                        new_aligned.values,
                        ref_aligned.values
                    )
                    
                    drift_detected = p_value < self.drift_threshold
                    drift_results[col] = {
                        'drift_detected': drift_detected,
                        'p_value': p_value,
                        'chi2_statistic': chi2_stat
                    }
                    
                except Exception as e:
                    logger.error(f"Error detecting categorical drift for {col}: {e}")
                    drift_results[col] = {
                        'drift_detected': False,
                        'error': str(e)
                    }
        
        return drift_results
    
    def generate_drift_report(self, new_data: pd.DataFrame) -> Dict:
        """Generate comprehensive drift detection report."""
        logger.info("Generating drift detection report...")
        
        numerical_drift = self.detect_drift(new_data)
        categorical_drift = self.detect_categorical_drift(new_data)
        
        # Count total drift
        total_features = len(numerical_drift) + len(categorical_drift)
        drifted_features = sum([
            result.get('drift_detected', False) 
            for result in list(numerical_drift.values()) + list(categorical_drift.values())
        ])
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_features': total_features,
            'drifted_features': drifted_features,
            'drift_percentage': (drifted_features / total_features * 100) if total_features > 0 else 0,
            'numerical_drift': numerical_drift,
            'categorical_drift': categorical_drift,
            'drift_detected': drifted_features > 0
        }
        
        logger.info(f"Drift report generated: {drifted_features}/{total_features} features drifted")
        
        return report

def main():
    """Demo function for drift detection."""
    logger.info("Drift detection module created")

if __name__ == "__main__":
    main()
