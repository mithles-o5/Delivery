import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Simulate Data
np.random.seed(42)
data = {
    'distance': np.random.uniform(1, 50, 1000),  # Distance in km
    'weather_condition': np.random.choice(['clear', 'rain', 'storm', 'snow'], 1000),
    'temperature': np.random.uniform(-10, 40, 1000),  # Temperature in Celsius
    'humidity': np.random.uniform(30, 100, 1000),  # Humidity in %
    'delayed': np.random.choice([0, 1], 1000)  # Binary: 0 = On-Time, 1 = Delayed
}

df = pd.DataFrame(data)

# Add a feature to indicate bad weather
df['is_bad_weather'] = df['weather_condition'].apply(lambda x: 1 if x in ['rain', 'storm', 'snow'] else 0)

# Step 2: Preprocess Data
X = df[['distance', 'is_bad_weather', 'temperature', 'humidity']]
y = df['delayed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train the Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the Model
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 4: Save the Model and Scaler
joblib.dump(model, 'delivery_delay_model.pkl')  # Save the model
joblib.dump(scaler, 'scaler.pkl')               # Save the scaler