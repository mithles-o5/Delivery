from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login as auth_login
from django.http import JsonResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
import joblib
import numpy as np
import os

# Transport Mode Prediction View
@api_view(['POST'])
@csrf_exempt
def predict_transport_mode(request):
    try:
        # Load the trained model and scaler
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(BASE_DIR, 'transport', 'models', 'transport_mode_model.pkl')
        scaler_path = os.path.join(BASE_DIR, 'transport', 'models', 'scaler.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Parse input data
        train_duration = request.data.get('train_duration')
        is_bad_weather = request.data.get('is_bad_weather', 0)
        flight_duration = request.data.get('flight_duration', 0)
        road_duration = request.data.get('road_duration', 0)
        
        # Validate input
        if None in [train_duration]:
            return Response({"error": "Missing data"}, status=400)
        
        # Prepare input for prediction
        input_data = np.array([[train_duration, is_bad_weather, flight_duration, road_duration]])
        
        # Scale input data
        input_data_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_data_scaled)[0]
        
        # Map prediction to transport mode
        transport_modes = {0: 'flight', 1: 'train', 2: 'road'}
        result = transport_modes.get(prediction, 'unknown')
        
        # Return result
        return Response({"prediction": result})
    except Exception as e:
        return Response({"error": str(e)}, status=500)
  # Rename login to avoid conflict

# @login_required  # Ensure only logged-in users can access the map
def map_view(request):
    return render(request, 'map.html')

# Updated Login View
def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)  # Authenticate the user
            return redirect('map')  # Redirect to map.html after login
        else:
            return render(request, 'login.html', {'error': 'Invalid credentials'})
    return render(request, 'login.html')



def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')  # Redirect to login page after registration
    else:
        form = UserCreationForm()
    return render(request, 'registeration.html', {'form': form})

def get_route(request):
    source = request.GET.get('source')
    destination = request.GET.get('destination')

    # Example: Use OSRM or another routing service to calculate the route
    # Replace with actual routing logic
    return JsonResponse({"route": {"coordinates": [[78.9629, 20.5937], [78.476, 17.385]]}})