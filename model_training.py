import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os
import numpy as np # For np.nan

# Define file paths
DATA_FILE = 'data.csv'
MODEL_SAVE_PATH = 'models'
MODEL_FILE = os.path.join(MODEL_SAVE_PATH, 'aqi_predictor_model.joblib')
CITY_AVG_FEATURES_FILE = os.path.join(MODEL_SAVE_PATH, 'city_avg_features.joblib')

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def train_model():
    print("Loading data...")
    try:
        data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please create it.")
        return

    if 'AQI' not in data.columns or 'City' not in data.columns:
        print("Error: 'AQI' and 'City' columns are required in the CSV.")
        return

    print("Data loaded successfully. Columns:", data.columns.tolist())
    print("Original data shape:", data.shape)
    print("Data types:\n", data.dtypes)

    # --- Feature Engineering ---
    # 1. Handle 'Date'
    if 'Date' in data.columns:
        try:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce') # errors='coerce' will turn unparseable dates to NaT
            data = data.dropna(subset=['Date']) # Drop rows where date couldn't be parsed
            data['Year'] = data['Date'].dt.year
            data['Month'] = data['Date'].dt.month
            data['Day'] = data['Date'].dt.day
            data['DayOfWeek'] = data['Date'].dt.dayofweek
            # Drop original 'Date' column as we've extracted features
            data = data.drop('Date', axis=1)
            print("Date features extracted.")
        except Exception as e:
            print(f"Error processing 'Date' column: {e}. Skipping date feature extraction.")
            # Ensure potential date feature columns don't cause issues later if they weren't created
            for col in ['Year', 'Month', 'Day', 'DayOfWeek']:
                if col not in data.columns:
                    data[col] = np.nan # Or some other default if preferred


    # 2. Drop 'AQI_Bucket' as it's derived from AQI (target leakage)
    if 'AQI_Bucket' in data.columns:
        data = data.drop('AQI_Bucket', axis=1)
        print("'AQI_Bucket' column dropped.")

    # 3. Handle potential non-numeric data in pollutant columns (e.g., strings like 'None', '<0.5')
    # Define expected pollutant and other numeric columns (excluding AQI, City, and date parts for now)
    potential_numeric_cols = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
    date_feature_cols = ['Year', 'Month', 'Day', 'DayOfWeek']
    
    all_numeric_feature_cols = [col for col in potential_numeric_cols if col in data.columns]
    all_numeric_feature_cols += [col for col in date_feature_cols if col in data.columns]


    for col in all_numeric_feature_cols:
        if data[col].dtype == 'object': # If column is object type, try to convert
            data[col] = pd.to_numeric(data[col], errors='coerce') # 'coerce' turns unconvertible to NaN

    # Impute NaNs created by 'coerce' or already present, BEFORE calculating averages
    # Using median for imputation for robustness to outliers in pollutant data
    for col in all_numeric_feature_cols:
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].median())
            print(f"Filled NaNs in '{col}' with median.")


    # --- Store average features for each city ---
    # This is used for prediction when only city name is given
    # We need averages for all features the model will be trained on, except 'City' itself
    if not all_numeric_feature_cols:
        print("Warning: No numeric features identified for averaging. Predictions might be unreliable.")
        city_avg_features = pd.DataFrame(columns=['City'] + all_numeric_feature_cols).set_index('City')
    else:
        city_avg_features = data.groupby('City')[all_numeric_feature_cols].mean()

    joblib.dump(city_avg_features, CITY_AVG_FEATURES_FILE)
    print(f"Average features per city saved to {CITY_AVG_FEATURES_FILE}")
    print("Sample of average features:\n", city_avg_features.head())

    # --- Define features (X) and target (y) ---
    # Ensure 'AQI' is numeric and drop rows with NaN AQI if any
    data['AQI'] = pd.to_numeric(data['AQI'], errors='coerce')
    data = data.dropna(subset=['AQI'])

    X = data.drop('AQI', axis=1)
    y = data['AQI']

    if X.empty:
        print("Error: No data left after processing. Check your CSV file and data cleaning steps.")
        return

    categorical_features = ['City']
    # numeric_features now includes date parts and pollutants
    numeric_features = [col for col in all_numeric_feature_cols if col in X.columns]

    print(f"Categorical features for model: {categorical_features}")
    print(f"Numeric features for model: {numeric_features}")


    # --- Preprocessing ---
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Impute any remaining NaNs (e.g., if a whole city had NaNs for a feature)
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for easier debugging if needed
    ])

    # Create a column transformer
    # Ensure that numeric_features and categorical_features only contain columns present in X
    valid_numeric_features = [f for f in numeric_features if f in X.columns]
    valid_categorical_features = [f for f in categorical_features if f in X.columns]

    # Check if features lists are empty, which can cause ColumnTransformer to fail
    transformers_list = []
    if valid_numeric_features:
        transformers_list.append(('num', numeric_transformer, valid_numeric_features))
    else:
        print("Warning: No valid numeric features for the preprocessor.")
    
    if valid_categorical_features:
        transformers_list.append(('cat', categorical_transformer, valid_categorical_features))
    else:
        print("Warning: No valid categorical features for the preprocessor. 'City' is essential.")
        # If 'City' is missing, this model design has a fundamental issue for prediction logic
        return


    if not transformers_list:
        print("Error: No features to transform. Aborting training.")
        return

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='drop' # Drop any columns not specified (should be none if X is defined correctly)
    )

    # --- Model ---
    model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)

    # --- Create a full pipeline ---
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    
    # --- Train-test split (optional for this example, training on full data) ---
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    print(f"Training on {len(X)} samples.")

    # --- Train the model ---
    try:
        pipeline.fit(X, y)
    except ValueError as e:
        print(f"Error during model training: {e}")
        print("This might be due to empty feature lists or data type issues not caught earlier.")
        print("Check feature definitions and data cleaning steps.")
        print("X columns:", X.columns)
        print("Numeric features for preprocessor:", valid_numeric_features)
        print("Categorical features for preprocessor:", valid_categorical_features)
        return

    print("Model training completed.")
    if hasattr(model, 'oob_score_') and model.oob_score_: # oob_score only available if n_estimators > 1 and bootstrap=True (default)
         print(f"Model OOB Score: {model.oob_score_:.4f}")


    # --- Save the trained pipeline ---
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Trained pipeline saved to {MODEL_FILE}")

if __name__ == '__main__':
    train_model()