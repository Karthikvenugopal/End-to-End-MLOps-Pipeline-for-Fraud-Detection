"""
FastAPI application for fraud detection model serving.
Provides REST API endpoints for model inference.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import logging
from typing import List, Dict, Any
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="API for fraud detection model inference",
    version="1.0.0"
)

# Global variables for model and preprocessors
model = None
label_encoders = {}
scaler = None

class TransactionRequest(BaseModel):
    """Request model for transaction data."""
    TransactionAmt: float
    ProductCD: str
    card1: int
    card2: int
    card3: int
    card4: str
    card5: int
    card6: str
    addr1: int
    addr2: int
    dist1: float
    dist2: float
    P_emaildomain: str
    R_emaildomain: str

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    fraud_probability: float
    is_fraud: bool
    confidence: str

@app.on_event("startup")
async def load_model():
    """Load the trained model and preprocessors on startup."""
    global model, label_encoders, scaler
    
    try:
        # Load model
        model_path = "models/xgboost_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found, using dummy model")
            model = None
            
        # Load preprocessors (would be loaded from saved files in production)
        logger.info("API startup completed")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Fraud Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    """Predict fraud probability for a transaction."""
    try:
        if model is None:
            # Return dummy prediction for demo
            fraud_prob = np.random.random()
            return PredictionResponse(
                fraud_probability=float(fraud_prob),
                is_fraud=fraud_prob > 0.5,
                confidence="high" if fraud_prob > 0.8 or fraud_prob < 0.2 else "medium"
            )
        
        # Convert request to DataFrame
        data = transaction.dict()
        df = pd.DataFrame([data])
        
        # Preprocess data (simplified for demo)
        # In production, you would apply the same preprocessing as training
        
        # Make prediction
        fraud_prob = model.predict_proba(df)[0][1]
        
        return PredictionResponse(
            fraud_probability=float(fraud_prob),
            is_fraud=fraud_prob > 0.5,
            confidence="high" if fraud_prob > 0.8 or fraud_prob < 0.2 else "medium"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict_batch")
async def predict_batch(transactions: List[TransactionRequest]):
    """Predict fraud for multiple transactions."""
    try:
        results = []
        for transaction in transactions:
            prediction = await predict_fraud(transaction)
            results.append(prediction.dict())
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
