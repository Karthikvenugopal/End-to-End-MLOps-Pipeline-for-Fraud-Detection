"""
Data preparation pipeline for fraud detection dataset.
Handles data loading, preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import yaml
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDataProcessor:
    """Data processor for fraud detection dataset."""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, train_path=None, test_path=None):
        """
        Load training and test data.
        For demo purposes, we'll create synthetic data if files don't exist.
        """
        logger.info("Loading data...")
        
        if train_path and os.path.exists(train_path):
            self.train_df = pd.read_csv(train_path)
        else:
            logger.warning("Training data not found, creating synthetic data...")
            self.train_df = self._create_synthetic_data(n_samples=10000, fraud_rate=0.035)
        
        if test_path and os.path.exists(test_path):
            self.test_df = pd.read_csv(test_path)
        else:
            logger.warning("Test data not found, creating synthetic data...")
            self.test_df = self._create_synthetic_data(n_samples=2000, fraud_rate=0.035)
        
        logger.info(f"Loaded {len(self.train_df)} training samples")
        logger.info(f"Loaded {len(self.test_df)} test samples")
        
    def _create_synthetic_data(self, n_samples=10000, fraud_rate=0.035):
        """Create synthetic fraud detection data for demonstration."""
        np.random.seed(42)
        
        # Create base features
        data = {
            'TransactionAmt': np.random.lognormal(3, 1.5, n_samples),
            'ProductCD': np.random.choice(['W', 'R', 'C', 'S', 'H'], n_samples),
            'card1': np.random.randint(1, 10000, n_samples),
            'card2': np.random.randint(1, 1000, n_samples),
            'card3': np.random.randint(1, 200, n_samples),
            'card4': np.random.choice(['visa', 'mastercard', 'discover', 'amex'], n_samples),
            'card5': np.random.randint(1, 200, n_samples),
            'card6': np.random.choice(['debit', 'credit'], n_samples),
            'addr1': np.random.randint(1, 1000, n_samples),
            'addr2': np.random.randint(1, 100, n_samples),
            'dist1': np.random.exponential(5, n_samples),
            'dist2': np.random.exponential(10, n_samples),
            'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'], n_samples),
            'R_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'], n_samples),
        }
        
        # Add C features (count features)
        for i in range(1, 16):
            data[f'C{i}'] = np.random.poisson(2, n_samples)
        
        # Add D features (time-based features)
        for i in range(1, 16):
            data[f'D{i}'] = np.random.exponential(1, n_samples)
        
        # Add M features (match features)
        for i in range(1, 10):
            data[f'M{i}'] = np.random.choice(['T', 'F'], n_samples)
        
        df = pd.DataFrame(data)
        
        # Create fraud labels
        fraud_indices = np.random.choice(
            n_samples, 
            size=int(n_samples * fraud_rate), 
            replace=False
        )
        
        df['isFraud'] = 0
        df.loc[fraud_indices, 'isFraud'] = 1
        
        # Make fraud transactions more suspicious
        fraud_mask = df['isFraud'] == 1
        df.loc[fraud_mask, 'TransactionAmt'] *= np.random.uniform(2, 5, fraud_mask.sum())
        df.loc[fraud_mask, 'dist1'] *= np.random.uniform(2, 4, fraud_mask.sum())
        
        return df
    
    def preprocess_data(self):
        """Preprocess the data for model training."""
        logger.info("Preprocessing data...")
        
        # Handle missing values
        self.train_df = self._handle_missing_values(self.train_df)
        self.test_df = self._handle_missing_values(self.test_df)
        
        # Encode categorical variables
        self.train_df = self._encode_categorical(self.train_df, fit=True)
        self.test_df = self._encode_categorical(self.test_df, fit=False)
        
        # Scale numerical features
        self.train_df = self._scale_numerical(self.train_df, fit=True)
        self.test_df = self._scale_numerical(self.test_df, fit=False)
        
        logger.info("Data preprocessing completed")
        
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        # Fill missing values in categorical columns
        categorical_cols = self.config['features']['categorical_columns']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
        
        # Fill missing values in numerical columns
        numerical_cols = self.config['features']['numerical_columns']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _encode_categorical(self, df, fit=False):
        """Encode categorical variables."""
        categorical_cols = self.config['features']['categorical_columns']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        df[col] = df[col].astype(str)
                        unseen_mask = ~df[col].isin(self.label_encoders[col].classes_)
                        if unseen_mask.any():
                            # Add 'unknown' category to the encoder if it doesn't exist
                            if 'unknown' not in self.label_encoders[col].classes_:
                                # Create a new encoder with 'unknown' category
                                new_encoder = LabelEncoder()
                                all_classes = list(self.label_encoders[col].classes_) + ['unknown']
                                new_encoder.fit(all_classes)
                                self.label_encoders[col] = new_encoder
                            df.loc[unseen_mask, col] = 'unknown'
                        df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _scale_numerical(self, df, fit=False):
        """Scale numerical features."""
        numerical_cols = self.config['features']['numerical_columns']
        available_cols = [col for col in numerical_cols if col in df.columns]
        
        if fit:
            df[available_cols] = self.scaler.fit_transform(df[available_cols])
        else:
            df[available_cols] = self.scaler.transform(df[available_cols])
        
        return df
    
    def split_data(self):
        """Split data into train, validation, and test sets."""
        logger.info("Splitting data...")
        
        # Separate features and target
        feature_cols = (self.config['features']['categorical_columns'] + 
                       self.config['features']['numerical_columns'])
        available_cols = [col for col in feature_cols if col in self.train_df.columns]
        
        X = self.train_df[available_cols]
        y = self.train_df['isFraud']
        
        # Split into train/val/test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=1-self.config['data']['train_split'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        val_size = self.config['data']['val_split'] / (self.config['data']['val_split'] + self.config['data']['test_split'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1-val_size,
            random_state=self.config['data']['random_state'],
            stratify=y_temp
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, output_dir="data/processed/"):
        """Save processed data to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training data
        self.train_df.to_csv(f"{output_dir}/train_processed.csv", index=False)
        
        # Save test data
        self.test_df.to_csv(f"{output_dir}/test_processed.csv", index=False)
        
        logger.info(f"Processed data saved to {output_dir}")

def main():
    """Main function to run data preparation."""
    processor = FraudDataProcessor()
    
    # Load data
    processor.load_data()
    
    # Preprocess data
    processor.preprocess_data()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data()
    
    # Save processed data
    processor.save_processed_data()
    
    logger.info("Data preparation completed successfully!")

if __name__ == "__main__":
    main()
