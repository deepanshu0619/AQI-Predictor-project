import os
import subprocess
import sys

def main():
    print("Setting up AQI Predictor...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Install requirements
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    # Train model
    print("Training model... (This may take several minutes)")
    subprocess.check_call([sys.executable, 'model_training.py'])
    
    print("Setup complete! Run 'python app.py' to start the server.")

if __name__ == '__main__':
    main()
