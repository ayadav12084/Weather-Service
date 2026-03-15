

import logging
import time
from datetime import datetime, timezone

import requests

import db
from config import (
    OWM_API_KEY,
    OWM_CURRENT_URL,
    OWM_MAX_RETRIES,
    OWM_RETRY_BACKOFF_SEC,
    OWM_TIMEOUT_SEC,
)

logger = logging.getLogger(__name__)




def _get_with_retry(url: str, params: dict) -> dict:
    
    last_exc: Exception | None = None
    for attempt in range(1, OWM_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=OWM_TIMEOUT_SEC)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            logger.warning("Attempt %d/%d timed out for %s",
                           attempt, OWM_MAX_RETRIES, url)
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            logger.warning("Attempt %d/%d connection error: %s",
                           attempt, OWM_MAX_RETRIES, exc)
        except requests.HTTPError as exc:
            # 4xx errors (bad API key, city not found) are not retryable
            if exc.response is not None and exc.response.status_code < 500:
                raise
            last_exc = exc
            logger.warning("Attempt %d/%d HTTP %s for %s",
                           attempt, OWM_MAX_RETRIES,
                           exc.response.status_code if exc.response else "?",
                           url)

        if attempt < OWM_MAX_RETRIES:
            wait = OWM_RETRY_BACKOFF_SEC * attempt   # 5s, 10s, 15s …
            logger.info("Waiting %ds before retry …", wait)
            time.sleep(wait)

    raise RuntimeError(
        f"All {OWM_MAX_RETRIES} attempts failed for {url}"
    ) from last_exc



def _kelvin_to_celsius(k: float | None) -> float | None:
    if k is None:
        return None
    return round(k - 273.15, 2)


def _parse_owm_response(data: dict) -> dict:

    main    = data.get("main", {})
    wind    = data.get("wind", {})
    rain    = data.get("rain", {})
    clouds  = data.get("clouds", {})
    weather = data.get("weather", [{}])[0]

    owm_id  = weather.get("id", 0)
    cond_code = f"OWM_{owm_id}"
    cond_desc = weather.get("description", "unknown").capitalize()
    cond_cat  = weather.get("main", "Unknown")          # e.g. "Clear", "Rain"

    recorded_at = datetime.fromtimestamp(
        data.get("dt", 0), tz=timezone.utc
    ).replace(tzinfo=None)  

    return {
        "_cond_code": cond_code,
        "_cond_desc": cond_desc,
        "_cond_cat":  cond_cat,

        "recorded_at":      recorded_at,
        "temp_celsius":     _kelvin_to_celsius(main.get("temp")),
        "feels_like":       _kelvin_to_celsius(main.get("feels_like")),
        "humidity_pct":     main.get("humidity"),
        "pressure_hpa":     main.get("pressure"),
        "wind_speed_ms":    round(wind.get("speed", 0.0), 2),
        "wind_dir_deg":     wind.get("deg"),
        "precipitation_mm": round(rain.get("1h", 0.0), 2),
        "cloud_cover_pct":  clouds.get("all"),
        "source_api":       "openweathermap",
    }



def fetch_and_store(region: dict) -> bool:
    """
    Fetch current weather for one region dict (from config.REGIONS) and
    write it to the database.

    Returns True on success, False if the fetch or DB write failed.
    The exception is logged but not re-raised so the scheduler can
    continue with the next region.
    """
    name    = region["name"]
    lat     = region["latitude"]
    lon     = region["longitude"]

    logger.info("Fetching weather for %s (%.4f, %.4f)", name, lat, lon)

    try:
        raw = _get_with_retry(
            OWM_CURRENT_URL,
            {"lat": lat, "lon": lon, "appid": OWM_API_KEY},
        )
    except Exception as exc:
        logger.error("Failed to fetch %s: %s", name, exc)
        return False

    try:
        parsed = _parse_owm_response(raw)
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("Failed to parse OWM response for %s: %s", name, exc)
        logger.debug("Raw response: %s", raw)
        return False

    try:
        with db.get_conn() as conn:
            region_id = db.get_or_create_region(
                conn,
                name         = region["name"],
                country_code = region["country_code"],
                latitude     = region["latitude"],
                longitude    = region["longitude"],
                timezone     = region["timezone"],
            )
            condition_id = db.get_or_create_condition(
                conn,
                code        = parsed.pop("_cond_code"),
                description = parsed.pop("_cond_desc"),
                category    = parsed.pop("_cond_cat"),
            )
            reading = {
                "region_id":    region_id,
                "condition_id": condition_id,
                **parsed,
            }
            db.upsert_reading(conn, reading)

    except Exception as exc:
        logger.error("DB write failed for %s: %s", name, exc, exc_info=True)
        return False

    logger.info(
        "Stored reading for %s — %.1f°C, humidity %.0f%%",
        name,
        parsed.get("temp_celsius") or 0,
        parsed.get("humidity_pct") or 0,
    )
    return True
