from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import numpy as np
import os

class PredictDelay(APIView):
    def post(self, request):
        try:
            # Load the trained model and scaler
            model_path = os.path.join(os.path.dirname(__file__), '..', 'delivery_delay_model.pkl')
            scaler_path = os.path.join(os.path.dirname(__file__), '..', 'scaler.pkl')
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # Parse input data
            distance = request.data.get('distance')
            is_bad_weather = request.data.get('is_bad_weather')
            temperature = request.data.get('temperature')
            humidity = request.data.get('humidity')

            # Validate input
            if None in [distance, is_bad_weather, temperature, humidity]:
                return Response({"error": "Missing data"}, status=status.HTTP_400_BAD_REQUEST)

            # Prepare input for prediction
            input_data = np.array([[distance, is_bad_weather, temperature, humidity]])

            # Scale input data
            input_data_scaled = scaler.transform(input_data)

            # Predict
            prediction = model.predict(input_data_scaled)[0]

            # Return result
            return Response({"prediction": "Delayed" if prediction == 1 else "On-Time"})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

from django.shortcuts import render
from django.contrib.auth.decorators import login_required

# @login_required  # Ensure only logged-in users can access the map
def map_view(request):
    return render(request, 'map.html')

from django.http import JsonResponse

def get_route(request):
    source = request.GET.get('source')
    destination = request.GET.get('destination')

    # Example: Use OSRM or another routing service to calculate the route
    # Replace with actual routing logic
    return JsonResponse({"route": {"coordinates": [[78.9629, 20.5937], [78.476, 17.385]]}})