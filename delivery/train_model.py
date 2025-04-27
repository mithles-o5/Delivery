from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import joblib
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@csrf_exempt
def predict_transport_mode(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            origin = data.get('origin')
            destination = data.get('destination')
            flight_data = data.get('flight_data', {})
            road_data = data.get('road_data', [])
            weather_data = data.get('weather_data', {})

            # Extract features
            flight_duration = (pd.to_datetime(flight_data['arrival']) - pd.to_datetime(flight_data['departure'])).total_seconds() / 3600 if flight_data else 0
            flight_cost = flight_data['cost'] if flight_data else 0
            road_duration = sum([wp.duration for wp in road_data]) / 3600 if road_data else 0
            road_distance = sum([wp.distance for wp in road_data]) / 1000 if road_data else 0
            temperature = (weather_data['source']['temp_c'] + weather_data['destination']['temp_c']) / 2
            precipitation = (weather_data['source']['precip_mm'] + weather_data['destination']['precip_mm']) / 2
            is_bad_weather = int(precipitation > 5 or temperature < 10)

            # Prepare input for AI model
            input_data = np.array([[flight_duration, flight_cost, road_duration, road_distance, temperature, precipitation, is_bad_weather]])

            # Load AI model and scaler
            model_path = os.path.join(BASE_DIR, 'models', 'transport_mode_model.pkl')
            scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # Scale input data
            input_data_scaled = scaler.transform(input_data)

            # Predict
            prediction = model.predict(input_data_scaled)[0]
            transport_modes = {0: 'flight', 1: 'road'}
            best_mode = transport_modes.get(prediction, 'unknown')

            # Return result
            return JsonResponse({
                'best_mode': best_mode,
                'cost': flight_cost if best_mode == 'flight' else 'Variable'
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Sample training data structure (replace with your own dataset)
data = pd.read_csv('transport_dataset.csv')  # Replace with your actual path

# Features and target
X = data[['flight_duration', 'train_duration', 'road_duration', 'is_bad_weather']]
y = data['preferred_mode']  # 0 = flight, 1 = train, 2 = road

# Scale and train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save model and scaler
model_dir = os.path.join('your_django_project', 'transport', 'models')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, 'transport_mode_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

print("Model and scaler saved.")
