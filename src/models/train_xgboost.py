"""
XGBoost model training with hyperparameter optimization.
Implements the baseline model for fraud detection.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import optuna
import mlflow
import mlflow.xgboost
import yaml
import joblib
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostFraudDetector:
    """XGBoost model for fraud detection with hyperparameter optimization."""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
    def objective(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective function for hyperparameter optimization."""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': 'gbtree',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict and calculate metrics
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate precision@100 (top 100 most suspicious transactions)
        top_100_indices = np.argsort(y_pred_proba)[-100:]
        precision_at_100 = precision_score(
            y_val.iloc[top_100_indices], 
            y_pred[top_100_indices]
        )
        
        return precision_at_100
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Optimize hyperparameters using Optuna."""
        logger.info("Starting hyperparameter optimization...")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.config['models']['xgboost']['n_trials']
        )
        
        self.best_params = study.best_params
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best precision@100: {study.best_value:.4f}")
        
        return study.best_value

    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the final model with best parameters."""
        logger.info("Training final XGBoost model...")
        
        # Add default parameters if not found in best_params
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': 'gbtree',
            'random_state': 42
        }
        
        final_params = {**default_params, **self.best_params}
        
        self.model = xgb.XGBClassifier(**final_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Model training completed")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance on test set."""
        logger.info("Evaluating model performance...")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate precision@100
        top_100_indices = np.argsort(y_pred_proba)[-100:]
        precision_at_100 = precision_score(
            y_test.iloc[top_100_indices], 
            y_pred[top_100_indices]
        )
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'precision_at_100': precision_at_100
        }
        
        logger.info(f"Test Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, model_path="models/xgboost_model.pkl"):
        """Save the trained model."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def log_to_mlflow(self, metrics, X_train, y_train):
        """Log experiment to MLflow."""
        with mlflow.start_run(run_name="xgboost_fraud_detection"):
            # Log parameters
            mlflow.log_params(self.best_params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.xgboost.log_model(
                self.model,
                "model",
                registered_model_name="xgboost_fraud_detector"
            )
            
            # Log feature importance
            mlflow.log_text(
                self.feature_importance.to_string(),
                "feature_importance.txt"
            )
            
            logger.info("Experiment logged to MLflow")

def main():
    """Main function to train XGBoost model."""
    # Load processed data
    train_df = pd.read_csv("data/processed/train_processed.csv")
    test_df = pd.read_csv("data/processed/test_processed.csv")
    
    # Prepare features and target
    feature_cols = [col for col in train_df.columns if col != 'isFraud']
    X_train = train_df[feature_cols]
    y_train = train_df['isFraud']
    
    X_test = test_df[feature_cols]
    y_test = test_df['isFraud']
    
    # Split training data into train/val
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Initialize and train model
    detector = XGBoostFraudDetector()
    
    # Optimize hyperparameters
    best_score = detector.optimize_hyperparameters(X_train, y_train, X_val, y_val)
    
    # Train final model
    detector.train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    metrics = detector.evaluate_model(X_test, y_test)
    
    # Save model
    detector.save_model()
    
    # Log to MLflow
    detector.log_to_mlflow(metrics, X_train, y_train)
    
    logger.info("XGBoost training completed successfully!")

if __name__ == "__main__":
    main()
