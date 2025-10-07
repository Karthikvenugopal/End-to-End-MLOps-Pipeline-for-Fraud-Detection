# MLOps Pipeline Architecture

## System Overview

This document describes the architecture of the End-to-End MLOps Pipeline for Fraud Detection.

## Components

### 1. Data Pipeline
- **Data Source**: Synthetic fraud detection dataset
- **Preprocessing**: Feature engineering, encoding, scaling
- **Storage**: Processed data in CSV format

### 2. Model Training
- **XGBoost**: Gradient boosting with hyperparameter optimization
- **Neural Network**: Deep learning with embedding layers
- **Evaluation**: Precision@K, AUC, F1-score metrics

### 3. Experiment Tracking
- **MLflow**: Model versioning and experiment management
- **Metrics**: Performance tracking and comparison
- **Artifacts**: Model artifacts and metadata

### 4. Model Serving
- **FastAPI**: REST API for model inference
- **Docker**: Containerized deployment
- **Endpoints**: Single and batch prediction

### 5. Monitoring
- **Drift Detection**: Statistical tests for data drift
- **Performance**: Model performance monitoring
- **Alerts**: Automated drift alerts

## Data Flow

```
Raw Data → Preprocessing → Model Training → Model Registry → Model Serving → Monitoring
```

## Technology Stack

- **ML Framework**: XGBoost, PyTorch
- **MLOps**: MLflow, Prefect, FastAPI
- **Infrastructure**: Docker, SQLite
- **Monitoring**: Custom drift detection
- **Testing**: Pytest, FastAPI TestClient

## Deployment

The system can be deployed using:
1. Docker containers
2. Docker Compose for multi-service deployment
3. MLflow UI for experiment tracking
4. FastAPI for model serving
