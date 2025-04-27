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

# Railway cost dataset path
RAILWAY_COST_PATH = os.path.join(os.path.dirname(__file__), '..', 'excel_files', 'railway_cost.csv')

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

def get_rail_distance(src: str, dst: str) -> Optional[float]:
    """Get railway distance using Rail API."""
    try:
        # Get station codes
        station_url = f"https://api.railwayapi.com/v2/suggest-station/name/{src}/apikey/{RAIL_API_KEY}/"
        src_data = requests.get(station_url).json()
        src_code = src_data['stations'][0]['code']
        
        station_url = f"https://api.railwayapi.com/v2/suggest-station/name/{dst}/apikey/{RAIL_API_KEY}/"
        dst_data = requests.get(station_url).json()
        dst_code = dst_data['stations'][0]['code']
        
        # Get route information
        route_url = f"https://api.railwayapi.com/v2/route/source/{src_code}/dest/{dst_code}/apikey/{RAIL_API_KEY}/"
        route_data = requests.get(route_url).json()
        
        if route_data['response_code'] == 200:
            return float(route_data['route'][0]['distance'])
        return None
    except Exception as e:
        print(f"Error fetching rail distance: {e}")
        return None

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

def calculate_transport_costs(distance_km: float, weight_kg: float, zone: str) -> Dict[str, float]:
    """Calculate costs for different transport modes."""
    # Road cost
    road_cost = ROAD_RATE[zone] * weight_kg
    
    # Rail cost from CSV
    rail_cost = get_rail_cost(distance_km, weight_kg)
    
    # Air cost
    air_cost = AIR_RATE['Within'] * weight_kg if zone == 'Within' else AIR_RATE['Rest'] * weight_kg
    
    return {
        'road_cost': road_cost,
        'rail_cost': rail_cost,
        'air_cost': air_cost
    }

def predict_best_transport(data: Dict[str, Any]) -> str:
    """Predict the best mode of transport using simple rules."""
    # This is a simple rule-based system - replace with your AI model
    distance = data['road_distance_km']
    weight = data['weight_kg']
    
    if distance < 100:
        return 'road'
    elif distance < 500 and weight < 1000:
        return 'rail'
    else:
        return 'air'

def collect_transport_data(source: str, destination: str, weight_kg: float) -> Dict[str, Any]:
    """Main function to collect and process transport data."""
    # Get coordinates
    src_lat, src_lon = get_coordinates(source)
    dst_lat, dst_lon = get_coordinates(destination)
    
    # Calculate distances
    air_distance = calculate_air_distance(src_lat, src_lon, dst_lat, dst_lon)
    rail_distance = get_rail_distance(source, destination) or air_distance  # Fallback to air distance
    road_dur, road_dist = get_road_data(source, destination)
    
    # Get weather
    src_weather, src_temp = get_weather(source)
    dst_weather, dst_temp = get_weather(destination)
    
    # Calculate zone and costs
    zone = get_zone(source, destination)
    costs = calculate_transport_costs(road_dist, weight_kg, zone)
    
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
        "rail_distance_km": rail_distance,
        "air_distance_km": air_distance,
        "road_cost_inr": costs['road_cost'],
        "rail_cost_inr": costs['rail_cost'],
        "air_cost_inr": costs['air_cost']
    }
    
    # Predict best transport mode
    result["recommended_transport"] = predict_best_transport(result)
    
    # Save to CSV
    df = pd.DataFrame([result])
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'transport_data.csv')
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    
    return result

# Helper functions from original code remain unchanged
def get_road_data(src, dst):
    lat1, lon1 = get_coordinates(src)
    lat2, lon2 = get_coordinates(dst)
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
    r = requests.get(url).json().get('routes', [{}])[0]
    return r.get('duration', 0) / 60, r.get('distance', 0) / 1000

def get_rail_cost(distance_km, weight_kg):
    try:
        cost_df = pd.read_csv(RAILWAY_COST_PATH)
        cost_df.columns = cost_df.columns.str.strip()
        
        for _, row in cost_df.iterrows():
            dist_range = row.iloc[0]
            if isinstance(dist_range, str) and '-' in dist_range:
                low, high = map(int, dist_range.split('-'))
                if low <= distance_km <= high:
                    weight_ranges = [int(w.split('-')[1]) for w in cost_df.columns[1:]]
                    for i, max_weight in enumerate(weight_ranges):
                        if weight_kg <= max_weight:
                            return float(row.iloc[i + 1])
        return None
    except Exception as e:
        print(f"Error calculating rail cost: {e}")
        return None

def get_zone(src, dst):
    if src.lower() == dst.lower():
        return 'City'
    if src.split()[-1].lower() == dst.split()[-1].lower():
        return 'Within'
    return 'Rest'
