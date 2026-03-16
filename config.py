
import os

OWM_API_KEY = os.getenv("OWM_API_KEY", "YOUR_API_KEY_HERE")
OWM_BASE_URL = "https://api.openweathermap.org/data/2.5"
OWM_CURRENT_URL = f"{OWM_BASE_URL}/weather"
OWM_TIMEOUT_SEC = 10           
OWM_MAX_RETRIES = 3            
OWM_RETRY_BACKOFF_SEC = 5      


REGIONS = [
    {"name": "Frankfurt",  "country_code": "DE", "latitude": 50.1109, "longitude": 8.6821,  "timezone": "Europe/Berlin"},
    {"name": "Berlin",     "country_code": "DE", "latitude": 52.5200, "longitude": 13.4050, "timezone": "Europe/Berlin"},
    {"name": "Munich",     "country_code": "DE", "latitude": 48.1351, "longitude": 11.5820, "timezone": "Europe/Berlin"},
    {"name": "Hamburg",    "country_code": "DE", "latitude": 53.5753, "longitude": 10.0153, "timezone": "Europe/Berlin"},
]

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "3306")),
    "user":     os.getenv("DB_USER",     "weather_user"),
    "password": os.getenv("DB_PASSWORD", "changeme"),
    "database": os.getenv("DB_NAME",     "weather_db"),
    "charset":  "utf8mb4",
    "autocommit": False,
    "connect_timeout": 10,
    "pool_name": "weather_pool",
    "pool_size": 5,
}

FETCH_INTERVAL_MINUTES = 30    
SCHEDULER_TIMEZONE = "UTC"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE  = os.getenv("LOG_FILE",  "/var/log/weather_ingestion.log")
