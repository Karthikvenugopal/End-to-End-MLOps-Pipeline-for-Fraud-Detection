FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/

# Create models directory if it doesn't exist
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models

# Run the API
CMD ["python", "src/serving/api.py"]
