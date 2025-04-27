import requests
import pandas as pd
from datetime import datetime
import os
import math
from typing import Tuple, Dict, Any, Optional

# API keys
WEATHER_API_KEY = 'f65e7d3178e84ffdabf82714251104'
FLIGHT_API_KEY = '2137911892b31a1b3a1b76524fdca584'
RAIL_API_KEY = '16142bc4543c0e94ad892ec5d7d67958'

# Cost‐per‐kg dictionaries
ROAD_RATE = {
    'City': 35,
    'Within': 60,
    'Rest': 75
}
AIR_RATE = {
    'Within': 100,
    'Rest': 150
}

# Railway cost matrix (based on the image data)
RAIL_COSTS = {
    50: {  # Distance in km
        10: 7.53,   # Weight slabs (up to X kg): cost
        20: 15.06,
        30: 22.59,
        40: 30.12,
        50: 37.65,
        60: 45.18,
        70: 52.71,
        80: 60.24,
        90: 67.77,
        100: 75.3
    },
    60: {  # Distance in km
        10: 8.07,
        20: 16.14,
        30: 24.21,
        40: 32.28,
        50: 40.35,
        60: 48.42,
        70: 56.49,
        80: 64.56,
        90: 72.63,
        100: 80.7
    }
}

def get_coordinates(city: str) -> Tuple[float, float]:
    """Get coordinates using OpenStreetMap."""
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={city}&countrycodes=IN"
    headers = {"User-Agent": "DynamicMailingApp/1.0"}
    res = requests.get(url, headers=headers)
    data = res.json()
    return float(data[0]['lat']), float(data[0]['lon'])

def calculate_air_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate air distance using Haversine formula."""
    R = 6371  # Earth's radius in km
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def get_weather(city: str) -> Tuple[str, float]:
    """Get weather information."""
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        r = response.json()
        return r['current']['condition']['text'], r['current']['temp_c']
    except Exception as e:
        print(f"Weather API error for {city}: {e}")
        return "Unknown", 0

def get_rail_cost(distance_km: float, weight_kg: float) -> float:
    """Calculate railway cost based on distance and weight."""
    # Find the appropriate distance tier
    if distance_km <= 50:
        distance_tier = 50
    elif distance_km <= 60:
        distance_tier = 60
    else:
        # For distances > 60km, use the 60km rate with a multiplier
        distance_tier = 60
        
    # Find the appropriate weight slab
    weight_slabs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    weight_tier = next((slab for slab in weight_slabs if weight_kg <= slab), 100)
    
    # Get the base cost
    base_cost = RAIL_COSTS[distance_tier][weight_tier]
    
    # Apply distance multiplier for distances > 60km
    if distance_km > 60:
        multiplier = math.ceil(distance_km / 60)
        return base_cost * multiplier
    
    return base_cost

def predict_best_transport(data: Dict[str, Any]) -> str:
    """Predict the best mode of transport using multiple factors."""
    distance = data['road_distance_km']
    weight = data['weight_kg']
    
    # Ensure all costs exist and are numeric
    road_cost = float(data['road_cost_inr'] or 0)
    rail_cost = float(data['rail_cost_inr'] or 0)
    air_cost = float(data['air_cost_inr'] or 0)
    
    # Basic distance-based rules
    if distance <= 50:
        if road_cost <= rail_cost:
            return 'road'
        return 'rail'
    elif distance <= 300:
        if weight <= 30:
            return 'road' if road_cost <= rail_cost else 'rail'
        return 'rail'
    else:
        if weight >= 70:
            return 'rail'
        return 'air' if air_cost <= 1.5 * min(road_cost, rail_cost) else 'rail'

def get_road_data(src_lat: float, src_lon: float, dst_lat: float, dst_lon: float) -> Tuple[float, float]:
    """Get road duration and distance."""
    url = f"http://router.project-osrm.org/route/v1/driving/{src_lon},{src_lat};{dst_lon},{dst_lat}?overview=false"
    r = requests.get(url).json().get('routes', [{}])[0]
    return r.get('duration', 0) / 60, r.get('distance', 0) / 1000

def get_zone(src: str, dst: str) -> str:
    """Determine transport zone."""
    if src.lower() == dst.lower():
        return 'City'
    if src.split()[-1].lower() == dst.split()[-1].lower():
        return 'Within'
    return 'Rest'

def collect_transport_data(source: str, destination: str, weight_kg: float) -> Dict[str, Any]:
    """Main function to collect and process transport data."""
    # Get coordinates and calculate distances
    src_lat, src_lon = get_coordinates(source)
    dst_lat, dst_lon = get_coordinates(destination)
    
    air_distance = calculate_air_distance(src_lat, src_lon, dst_lat, dst_lon)
    road_dur, road_dist = get_road_data(src_lat, src_lon, dst_lat, dst_lon)
    
    # Get weather data
    src_weather, src_temp = get_weather(source)
    dst_weather, dst_temp = get_weather(destination)
    
    # Calculate costs
    zone = get_zone(source, destination)
    road_cost = ROAD_RATE[zone] * weight_kg
    rail_cost = get_rail_cost(road_dist, weight_kg)  # Using road distance for rail cost
    air_cost = AIR_RATE['Within'] * weight_kg if zone == 'Within' else AIR_RATE['Rest'] * weight_kg
    
    result = {
        "source": source,
        "destination": destination,
        "weight_kg": weight_kg,
        "src_weather": src_weather,
        "src_temp": src_temp,
        "dst_weather": dst_weather,
        "dst_temp": dst_temp,
        "road_duration_min": road_dur,
        "road_distance_km": road_dist,
        "air_distance_km": air_distance,
        "road_cost_inr": road_cost,
        "rail_cost_inr": rail_cost,
        "air_cost_inr": air_cost
    }
    
    # Predict best transport mode
    result["recommended_transport"] = predict_best_transport(result)
    
    # Save to CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'transport_data.csv')
    df = pd.DataFrame([result])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    
    return result
