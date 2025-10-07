"""
MLflow setup and configuration for fraud detection experiments.
"""

import mlflow
import mlflow.xgboost
import mlflow.pytorch
import yaml
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow(config_path="config/mlflow_config.yaml"):
    """Setup MLflow tracking and experiments."""
    try:
        # Load MLflow configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Set tracking URI
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        
        # Create or get experiment
        experiment_name = config['experiments']['default']['name']
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    description=config['experiments']['default']['description']
                )
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"Could not create experiment: {e}")
            experiment_id = "0"  # Use default experiment
        
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow setup completed. Experiment: {experiment_name}")
        
        return config
        
    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        return None

def log_model_metrics(metrics, model_name, run_name=None):
    """Log model metrics to MLflow."""
    try:
        with mlflow.start_run(run_name=run_name):
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model info
            mlflow.log_param("model_name", model_name)
            
            logger.info(f"Metrics logged for {model_name}")
            
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")

def main():
    """Setup MLflow for fraud detection experiments."""
    config = setup_mlflow()
    if config:
        logger.info("MLflow setup completed successfully")
    else:
        logger.error("MLflow setup failed")

if __name__ == "__main__":
    main()
