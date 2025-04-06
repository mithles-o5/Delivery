import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load and Preprocess Datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
EXCEL_FILES_DIR = os.path.join(BASE_DIR, "excel_files")
TEMPERATURE_DIR = os.path.join(EXCEL_FILES_DIR, "temperature")

# Load Airline Dataset
def load_airline_data():
    airline_data = pd.read_csv(os.path.join(EXCEL_FILES_DIR, "airline.csv"))
    
    # Convert 'Departure Time' and 'Arrival Time' to datetime with a default date
    airline_data['Departure Time'] = pd.to_datetime(airline_data['Departure Time'], format="%H:%M:%S", errors='coerce')
    airline_data['Arrival Time'] = pd.to_datetime(airline_data['Arrival Time'], format="%H:%M:%S", errors='coerce')
    
    # Add a default date (e.g., '2023-01-01') to ensure datetime compatibility
    airline_data['Departure Time'] = airline_data['Departure Time'].apply(
        lambda x: x.replace(year=2023, month=1, day=1) if not pd.isnull(x) else x
    )
    airline_data['Arrival Time'] = airline_data['Arrival Time'].apply(
        lambda x: x.replace(year=2023, month=1, day=1) if not pd.isnull(x) else x
    )
    
    # Drop rows with missing datetime values
    airline_data = airline_data.dropna(subset=['Departure Time', 'Arrival Time'])
    
    # Calculate flight duration in hours
    airline_data['flight_duration'] = (airline_data['Arrival Time'] - airline_data['Departure Time']).dt.total_seconds() / 3600
    
    # Add a 'date' column based on Departure Time
    airline_data['date'] = airline_data['Departure Time'].dt.date
    
    return airline_data[['Origin', 'Destination', 'flight_duration', 'Flight Price', 'date']].rename(columns={
        'Origin': 'source',
        'Destination': 'destination',
        'Flight Price': 'flight_cost'
    })

# Load Delhivery Dataset
def load_delhivery_data():
    delhivery_data = pd.read_csv(os.path.join(EXCEL_FILES_DIR, "delhivery.csv"))
    
    # Convert 'od_start_time' and 'od_end_time' to datetime with a default date
    delhivery_data['od_start_time'] = pd.to_datetime(delhivery_data['od_start_time'], format="%H:%M", errors='coerce')
    delhivery_data['od_end_time'] = pd.to_datetime(delhivery_data['od_end_time'], format="%H:%M", errors='coerce')
    
    # Add a default date (e.g., '2023-01-01') to ensure datetime compatibility
    delhivery_data['od_start_time'] = delhivery_data['od_start_time'].apply(
        lambda x: x.replace(year=2023, month=1, day=1) if not pd.isnull(x) else x
    )
    delhivery_data['od_end_time'] = delhivery_data['od_end_time'].apply(
        lambda x: x.replace(year=2023, month=1, day=1) if not pd.isnull(x) else x
    )
    
    # Drop rows with missing datetime values
    delhivery_data = delhivery_data.dropna(subset=['od_start_time', 'od_end_time'])
    
    # Calculate road duration in hours
    delhivery_data['road_duration'] = (delhivery_data['od_end_time'] - delhivery_data['od_start_time']).dt.total_seconds() / 3600
    
    # Add a 'date' column based on od_start_time
    delhivery_data['date'] = delhivery_data['od_start_time'].dt.date
    
    return delhivery_data[['source_name', 'destination_name', 'road_duration', 'actual_distance_to_destination', 'date']].rename(columns={
        'source_name': 'source',
        'destination_name': 'destination',
        'actual_distance_to_destination': 'road_distance'
    })

# Load Railway Dataset
def load_railway_data():
    railway_data = pd.read_json(os.path.join(EXCEL_FILES_DIR, "railway.json"))
    
    # Convert 'departure' to datetime with a default date
    railway_data['departure'] = pd.to_datetime(railway_data['departure'], format="%H:%M:%S", errors='coerce')
    
    # Add a default date (e.g., '2023-01-01') to ensure datetime compatibility
    railway_data['departure'] = railway_data['departure'].apply(
        lambda x: x.replace(year=2023, month=1, day=1) if not pd.isnull(x) else x
    )
    
    # Drop rows with missing datetime values
    railway_data = railway_data.dropna(subset=['departure'])
    
    # Calculate train duration in hours
    railway_data['train_duration'] = railway_data.groupby('train_number')['departure'].transform(lambda x: x.max() - x.min()).dt.total_seconds() / 3600
    
    # Add a 'date' column based on departure time
    railway_data['date'] = railway_data['departure'].dt.date
    
    return railway_data[['train_number', 'station_code', 'train_duration', 'date']].rename(columns={
        'station_code': 'source',
        'train_number': 'train_id'
    })

# Load Weather Data
def load_weather_data():
    weather_dfs = []
    for city in ['banglore', 'mumbai', 'chennai', 'delhi', 'lucknow', 'rajasthan']:
        file_path = os.path.join(TEMPERATURE_DIR, f"{city}.csv")
        df = pd.read_csv(file_path)
        
        # Ensure the 'time' column is in datetime format
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        
        # Drop rows with missing datetime values
        df = df.dropna(subset=['time'])
        
        df['city'] = city
        weather_dfs.append(df)
    
    weather_data = pd.concat(weather_dfs, ignore_index=True)
    
    # Rename 'time' to 'date' for consistency
    weather_data.rename(columns={'time': 'date'}, inplace=True)
    
    return weather_data[['city', 'date', 'tavg', 'prcp']].rename(columns={
        'tavg': 'temperature',
        'prcp': 'precipitation'
    })

# Step 2: Merge Datasets
def merge_datasets(airline_data, delhivery_data, railway_data, weather_data):
    # Merge Airline and Delhivery Data
    merged_data = pd.merge(airline_data, delhivery_data, on=['source', 'destination', 'date'], how='outer')

    # Merge Railway Data
    merged_data = pd.merge(merged_data, railway_data, on=['source', 'date'], how='left')

    # Merge Weather Data
    merged_data = pd.merge(merged_data, weather_data, left_on=['source', 'date'], right_on=['city', 'date'], how='left')

    # Handle Missing Values
    merged_data.fillna(0, inplace=True)

    return merged_data

# Step 3: Feature Engineering
def engineer_features(merged_data):
    # Add a feature to indicate bad weather
    merged_data['is_bad_weather'] = ((merged_data['precipitation'] > 5) | (merged_data['temperature'] < 10)).astype(int)

    # Define Target Variable
    merged_data['best_mode'] = merged_data.apply(
        lambda row: 'flight' if row['flight_duration'] < row['road_duration'] and row['flight_duration'] < row['train_duration']
        else 'train' if row['train_duration'] < row['road_duration'] else 'road', axis=1
    )

    # Encode Target Variable
    merged_data['best_mode_encoded'] = merged_data['best_mode'].map({'flight': 0, 'train': 1, 'road': 2})

    return merged_data

# Step 4: Train the Model
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
    # Load datasets
    airline_data = load_airline_data()
    delhivery_data = load_delhivery_data()
    railway_data = load_railway_data()
    weather_data = load_weather_data()

    # Debugging: Print dataset columns
    print("Columns in airline_data:", airline_data.columns.tolist())
    print("Columns in delhivery_data:", delhivery_data.columns.tolist())
    print("Columns in railway_data:", railway_data.columns.tolist())
    print("Columns in weather_data:", weather_data.columns.tolist())

    # Merge datasets
    merged_data = merge_datasets(airline_data, delhivery_data, railway_data, weather_data)

    # Feature engineering
    merged_data = engineer_features(merged_data)

    # Select features and target
    features = ['flight_duration', 'flight_cost', 'road_duration', 'road_distance', 'train_duration', 'temperature', 'precipitation', 'is_bad_weather']
    X = merged_data[features]
    y = merged_data['best_mode_encoded']

    # Train the model
    print(X)
    print(y)
    model, scaler = train_model(X, y)

    # Save the model and scaler
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, 'transport_mode_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    print("Model and scaler saved successfully.")
    