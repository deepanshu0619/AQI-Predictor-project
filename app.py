from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- Load Model and Assets ---
MODEL_DIR = 'models'
MODEL_FILE = os.path.join(MODEL_DIR, 'aqi_predictor_model.joblib')
CITY_AVG_FEATURES_FILE = os.path.join(MODEL_DIR, 'city_avg_features.joblib')

pipeline = None
city_avg_features = None
trained_cities = []
# To hold the feature names the model was actually trained on (extracted from pipeline)
model_numeric_features = []
model_categorical_features = []


def load_model_assets():
    global pipeline, city_avg_features, trained_cities, model_numeric_features, model_categorical_features
    try:
        pipeline = joblib.load(MODEL_FILE)
        city_avg_features = joblib.load(CITY_AVG_FEATURES_FILE)
        
        # Extract city names and feature names from the pipeline's preprocessor
        if pipeline and 'preprocessor' in pipeline.named_steps:
            preprocessor = pipeline.named_steps['preprocessor']
            for name, transformer_obj, columns in preprocessor.transformers_:
                if name == 'cat': # Categorical transformer
                    model_categorical_features.extend(columns)
                    if 'onehot' in transformer_obj.named_steps:
                        encoder = transformer_obj.named_steps['onehot']
                        if hasattr(encoder, 'categories_'):
                            # Find index of 'City' if multiple categorical features exist (though we only have one)
                            city_col_idx = columns.index('City') if 'City' in columns else -1
                            if city_col_idx != -1:
                                trained_cities = list(encoder.categories_[city_col_idx])
                elif name == 'num': # Numeric transformer
                    model_numeric_features.extend(columns)
        
        if not trained_cities and city_avg_features is not None: # Fallback for cities
            trained_cities = city_avg_features.index.tolist()

        print("Model and average features loaded successfully.")
        print(f"Model trained on cities (sample): {trained_cities[:5]}... (total {len(trained_cities)})")
        print(f"Model numeric features: {model_numeric_features}")
        print(f"Model categorical features: {model_categorical_features}")


    except FileNotFoundError:
        print(f"Error: Model file ({MODEL_FILE}) or city features ({CITY_AVG_FEATURES_FILE}) not found.")
        print("Please run model_training.py first.")
    except Exception as e:
        print(f"Error loading model or features: {e}")
        app.logger.error(f"Error loading model assets: {e}", exc_info=True)


load_model_assets()

@app.route('/')
def home():
    return render_template('index.html', cities=trained_cities)

@app.route('/predict', methods=['POST'])
def predict():
    if not pipeline or city_avg_features is None:
        return jsonify({'error': 'Model not loaded or city features missing. Train the model first.'}), 500

    try:
        data = request.get_json()
        city_name = data.get('city')

        if not city_name:
            return jsonify({'error': 'City name not provided'}), 400

        # --- Prepare input data for prediction ---
        input_data_dict = {}
        
        # 1. Add City (categorical feature)
        if 'City' in model_categorical_features:
             input_data_dict['City'] = [city_name]
        else:
            # This should ideally not happen if model training was correct
            app.logger.error("Critical: 'City' feature not found in model's categorical features.")
            return jsonify({'error': "Model configuration error: 'City' feature definition missing."}), 500


        # 2. Add numeric features using averages from city_avg_features
        if city_name in city_avg_features.index:
            avg_features_for_city = city_avg_features.loc[city_name]
            for feature_name in model_numeric_features:
                if feature_name in avg_features_for_city:
                    input_data_dict[feature_name] = [avg_features_for_city[feature_name]]
                else:
                    # Feature expected by model but not in city_avg_features (should not happen if training is consistent)
                    app.logger.warning(f"Numeric feature '{feature_name}' not found in city_avg_features for '{city_name}'. Using NaN.")
                    input_data_dict[feature_name] = [np.nan]
        else:
            # City not found in city_avg_features (unknown city)
            # Fill all numeric features with NaN; the pipeline's imputer should handle this.
            # The OneHotEncoder for 'City' also has handle_unknown='ignore'.
            app.logger.warning(f"City '{city_name}' not found in city_avg_features. Using NaNs for all numeric features.")
            for feature_name in model_numeric_features:
                input_data_dict[feature_name] = [np.nan]
        
        # Create DataFrame with columns in the order expected by the pipeline (implicitly handled by ColumnTransformer by name)
        # However, ensure all expected features are present.
        all_expected_model_features = model_categorical_features + model_numeric_features
        
        # Ensure all keys in input_data_dict are lists (for DataFrame creation)
        # And all expected features are present, even if with NaN
        final_input_for_df = {}
        for col in all_expected_model_features:
            if col in input_data_dict:
                final_input_for_df[col] = input_data_dict[col] if isinstance(input_data_dict[col], list) else [input_data_dict[col]]
            elif col == 'City': # Should have been handled
                 final_input_for_df[col] = [city_name]
            else: # A numeric feature might be missing if logic above failed.
                 final_input_for_df[col] = [np.nan]


        input_df = pd.DataFrame(final_input_for_df)
        
        # Reorder columns to match the order seen during training if necessary
        # (though ColumnTransformer uses names, maintaining order is good practice for clarity)
        # This step is more crucial if not using ColumnTransformer by name.
        # For now, let's assume ColumnTransformer handles it by name correctly.
        # If errors occur, uncomment and adapt:
        # try:
        #     input_df = input_df[all_expected_model_features] # Ensure correct column order
        # except KeyError as e:
        #     app.logger.error(f"Missing columns for prediction DataFrame: {e}")
        #     return jsonify({'error': f'Internal error: Missing expected feature columns for prediction.'}), 500


        print(f"Input DataFrame for prediction:\n{input_df}")

        # --- Make prediction ---
        prediction = pipeline.predict(input_df)
        predicted_aqi = round(float(prediction[0]), 2)

        return jsonify({'city': city_name, 'predicted_aqi': predicted_aqi})

    except Exception as e:
        app.logger.error(f"Prediction error: {e}", exc_info=True) # Log full traceback
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    # Ensure models directory exists for logging if needed, though training script creates it
    os.makedirs(MODEL_DIR, exist_ok=True) 
    app.run(debug=True)