#!/bin/bash
echo "Setting up AQI Predictor..."
echo "Creating models directory..."
mkdir -p models

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Training model... (This may take several minutes)"
python model_training.py

echo "Setup complete! Run 'python app.py' to start the server."
