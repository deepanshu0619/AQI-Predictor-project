#!/bin/bash
echo "Setting up AQI Predictor..."
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Training model..."
python model_training.py

echo "Setup complete! Run 'python app.py' to start the server."
