from langchain.tools import StructuredTool
from models.weather_models import GetWeatherArgs
from typing import Optional, Dict, List, Any
import requests
import os
from dotenv import load_dotenv
load_dotenv()
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')

def get_weather(city: str) -> Dict[str, Any]:
    """Mendapatkan informasi cuaca dari OpenWeatherMap"""
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if response.status_code == 200:
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"]
            }
        else:
            return {"error": "Failed to get weather data"}
    except Exception as e:
        return {"error": str(e)}

weather_tools = [
    StructuredTool.from_function(
        func=get_weather,
        name="get_weather",
        description="Mendapatkan informasi cuaca dari OpenWeatherMap",
        return_direct=False,
    )
]
