"""
PyTorch Neural Network model for fraud detection.
Implements deep learning approach with embedding layers for categorical features.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.pytorch
import yaml
import logging
from pathlib import Path
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionNN(nn.Module):
    """Neural network for fraud detection with embedding layers."""
    
    def __init__(self, config, categorical_dims, numerical_dim):
        super(FraudDetectionNN, self).__init__()
        
        self.config = config
        self.categorical_dims = categorical_dims
        self.numerical_dim = numerical_dim
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, min(50, (dim + 1) // 2))
            for dim in categorical_dims
        ])
        
        # Calculate input dimension
        embedding_dim = sum([min(50, (dim + 1) // 2) for dim in categorical_dims])
        input_dim = embedding_dim + numerical_dim
        
        # Hidden layers
        hidden_layers = config['models']['neural_network']['hidden_layers']
        dropout_rate = config['models']['neural_network']['dropout_rate']
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, categorical_inputs, numerical_inputs):
        """Forward pass through the network."""
        # Process categorical features through embeddings
        embedded = []
        for i, embedding in enumerate(self.embeddings):
            # Clamp indices to valid range to avoid out-of-bounds errors
            cat_input = torch.clamp(categorical_inputs[:, i], 0, embedding.num_embeddings - 1)
            embedded.append(embedding(cat_input))
        
        # Concatenate embeddings
        embedded = torch.cat(embedded, dim=1)
        
        # Combine with numerical features
        combined = torch.cat([embedded, numerical_inputs], dim=1)
        
        # Pass through network
        output = self.network(combined)
        
        return output.squeeze()

class NeuralNetworkFraudDetector:
    """Neural network model for fraud detection."""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.categorical_dims = None
        self.numerical_dim = None

def main():
    """Main function to train neural network model."""
    logger.info("Neural network training module created")

if __name__ == "__main__":
    main()
