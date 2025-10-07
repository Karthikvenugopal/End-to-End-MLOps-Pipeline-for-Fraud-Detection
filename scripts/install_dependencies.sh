#!/bin/bash

# Installation script for fraud detection MLOps pipeline
# Handles Python version compatibility and dependency installation

echo "🚀 Installing MLOps Pipeline Dependencies"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "📋 Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies based on Python version
if [[ "$PYTHON_VERSION" == "3.13" ]]; then
    echo "🐍 Installing Python 3.13 compatible dependencies..."
    pip install -r requirements-py313-final.txt
elif [[ "$PYTHON_VERSION" == "3.12" ]]; then
    echo "🐍 Installing Python 3.12 compatible dependencies..."
    pip install -r requirements.txt
else
    echo "🐍 Installing standard dependencies..."
    pip install -r requirements.txt
fi

# Verify installation
echo "✅ Verifying installation..."
python -c "import pandas, numpy, xgboost, torch, mlflow, fastapi; print('All dependencies installed successfully!')"

echo ""
echo "🎉 Installation completed!"
echo "========================="
echo "To activate the environment: source venv/bin/activate"
echo "To run the demo: python scripts/simple_demo.py"
echo "To start the API: python src/serving/api.py"
