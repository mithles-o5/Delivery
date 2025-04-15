import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load Railway Dataset
def load_railway_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXCEL_FILES_DIR = os.path.join(BASE_DIR, "excel_files")
    file_path = os.path.join(EXCEL_FILES_DIR, "railway.json")
    
    railway_data = pd.read_json(file_path)
    
    # Convert 'departure' to datetime with a default date
    railway_data['departure'] = pd.to_datetime(railway_data['departure'], format="%H:%M:%S", errors='coerce')
    
    # Add a default date (e.g., '2023-01-01') to ensure datetime compatibility
    railway_data['departure'] = railway_data['departure'].apply(
        lambda x: x.replace(year=2023, month=1, day=1) if not pd.isnull(x) else x
    )
    
    # Drop rows with missing datetime values
    railway_data = railway_data.dropna(subset=['departure'])
    
    # Calculate train duration in hours
    railway_data['train_duration'] = railway_data.groupby('train_number')['departure'].transform(
        lambda x: x.max() - x.min()
    ).dt.total_seconds() / 3600
    
    # Add a 'date' column based on departure time
    railway_data['date'] = railway_data['departure'].dt.date
    
    return railway_data[['train_number', 'station_code', 'train_duration', 'date']].rename(columns={
        'station_code': 'source',
        'train_number': 'train_id'
    })

# Step 2: Feature Engineering
def engineer_features(railway_data):
    # Add placeholder features for weather and other modes of transport
    railway_data['is_bad_weather'] = 0  # Placeholder for bad weather
    railway_data['flight_duration'] = 0  # Placeholder for flight data
    railway_data['road_duration'] = 0  # Placeholder for road data
    
    # Define Target Variable (Assume train is always the best mode for simplicity)
    railway_data['best_mode_encoded'] = 1  # 1 represents 'train'
    
    return railway_data

# Step 3: Train the Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model, scaler

# Main Execution
if __name__ == "__main__":
    # Load railway data
    railway_data = load_railway_data()
    
    # Feature engineering
    railway_data = engineer_features(railway_data)
    
    # Select features and target
    features = ['train_duration', 'is_bad_weather', 'flight_duration', 'road_duration']
    X = railway_data[features]
    y = railway_data['best_mode_encoded']
    
    # Train the model
    model, scaler = train_model(X, y)
    
    # Save the model and scaler
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, 'transport_mode_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("Model and scaler saved successfully.")