"""Weather for the affected area (Open-Meteo: free, global, no API key).

Rain matters after an earthquake: shaken slopes are prone to landslides, and
anyone sleeping outside because their building is unsafe needs to know whether
they will be rained on tonight.
"""
import requests

_URL = "https://api.open-meteo.com/v1/forecast"
_UA = {"User-Agent": "QuakeSense/1.0 (earthquake information service)"}

# WMO weather codes -> short label + icon
_CODES = {
    0: ("Clear", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"), 45: ("Fog", "🌫️"), 48: ("Freezing fog", "🌫️"),
    51: ("Light drizzle", "🌦️"), 53: ("Drizzle", "🌦️"),
    55: ("Heavy drizzle", "🌦️"), 61: ("Light rain", "🌧️"),
    63: ("Rain", "🌧️"), 65: ("Heavy rain", "🌧️"),
    66: ("Freezing rain", "🌧️"), 67: ("Freezing rain", "🌧️"),
    71: ("Light snow", "🌨️"), 73: ("Snow", "🌨️"), 75: ("Heavy snow", "🌨️"),
    80: ("Rain showers", "🌦️"), 81: ("Rain showers", "🌦️"),
    82: ("Violent rain showers", "⛈️"), 95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with hail", "⛈️"), 99: ("Thunderstorm with hail", "⛈️"),
}


def conditions(lat: float, lon: float, timeout: int = 8):
    """Current conditions plus the next 24 hours of rain for a point.

    Returns None on any failure so callers can simply skip the section."""
    try:
        r = requests.get(_URL, params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,precipitation,weather_code",
            "hourly": "precipitation_probability,precipitation",
            "forecast_days": 2, "timezone": "auto"}, timeout=timeout,
            headers=_UA)
        r.raise_for_status()
        j = r.json()
        cur = j.get("current", {}) or {}
        hourly = j.get("hourly", {}) or {}
        probs = [p for p in (hourly.get("precipitation_probability") or [])[:24]
                 if p is not None]
        rain = [v for v in (hourly.get("precipitation") or [])[:24]
                if v is not None]
        code = int(cur.get("weather_code") or 0)
        label, icon = _CODES.get(code, ("—", "🌡️"))
        return {
            "temp": cur.get("temperature_2m"),
            "now_precip": cur.get("precipitation"),
            "label": label, "icon": icon,
            "rain_chance": max(probs) if probs else None,
            "rain_mm": round(sum(rain), 1) if rain else 0.0,
        }
    except Exception:
        return None


def advisory(w: dict) -> str:
    """One practical line about what the weather means for people affected."""
    if not w:
        return ""
    chance = w.get("rain_chance") or 0
    mm = w.get("rain_mm") or 0
    if mm >= 10 or chance >= 70:
        return ("⚠️ **Rain likely in the next 24 hours.** Shaken slopes and "
                "loose debris become landslide-prone when wet — avoid steep "
                "ground and damaged buildings. Anyone sheltering outdoors "
                "needs waterproof cover and a dry, level spot away from "
                "slopes and walls.")
    if mm >= 1 or chance >= 30:
        return ("Some rain is possible in the next 24 hours. Keep shelter "
                "materials handy and avoid steep or debris-covered slopes, "
                "which weaken when wet.")
    return ("Little or no rain expected in the next 24 hours — conditions "
            "should stay workable for people sheltering outdoors.")
