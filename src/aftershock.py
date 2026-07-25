"""Aftershock outlook.

Important distinction: QuakeSense never predicts earthquakes. Aftershock
*forecasting*, however, is established statistics (Omori/Reasenberg-Jones) and
USGS publishes official Operational Aftershock Forecasts (OAF) for some events.

This module therefore only ever reports:
  1. the OFFICIAL USGS forecast for an event, when USGS publishes one; and
  2. aftershocks that have ALREADY been recorded near the epicentre.
It never invents probabilities of its own.
"""
from datetime import timedelta

import pandas as pd
import requests

_UA = {"User-Agent": "QuakeSense/1.0 (earthquake information service)"}
_DETAIL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/detail/{}.geojson"


def official_forecast(event_id: str, timeout: int = 10):
    """USGS Operational Aftershock Forecast for an event, or None.

    Returns {"issued": str, "windows": [{"label", "mag", "probability"}...]}
    where probability is a percentage for at least one quake of that magnitude
    in the window. USGS only issues these for some events (mainly US)."""
    if not event_id:
        return None
    try:
        det = requests.get(_DETAIL.format(event_id), timeout=timeout,
                           headers=_UA)
        det.raise_for_status()
        prods = det.json().get("properties", {}).get("products", {})
        oaf = (prods.get("oaf") or [None])[0]
        if not oaf:
            return None
        url = oaf.get("contents", {}).get("forecast.json", {}).get("url")
        if not url:
            return None
        fc = requests.get(url, timeout=timeout, headers=_UA).json()
        out = []
        for window in fc.get("forecast", []):
            label = window.get("label")
            for b in window.get("bins", []):
                mag, prob = b.get("magnitude"), b.get("probability")
                if mag is None or prob is None:
                    continue
                if float(mag) in (3.0, 5.0):          # keep it readable
                    out.append({"label": label, "mag": float(mag),
                                "probability": round(float(prob) * 100)})
        if not out:
            return None
        return {"issued": fc.get("creationTime"), "windows": out}
    except Exception:
        return None


def observed_aftershocks(live_df: pd.DataFrame, ev: dict, radius_km: float = 150.0):
    """Earthquakes already recorded near this event AFTER it happened - real
    observations from the live feed, not a forecast."""
    try:
        from math import radians, sin, cos, asin, sqrt

        def _km(lat1, lon1, lat2, lon2):
            p1, p2 = radians(lat1), radians(lat2)
            dp, dl = radians(lat2 - lat1), radians(lon2 - lon1)
            a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
            return 2 * 6371 * asin(sqrt(a))

        if live_df is None or live_df.empty:
            return None
        t0 = pd.Timestamp(ev["time"])
        after = live_df[(live_df["time"] > t0)
                        & (live_df["time"] <= t0 + timedelta(days=14))].copy()
        if after.empty:
            return {"count": 0, "max_mag": None, "latest": None}
        d = after.apply(lambda r: _km(ev["lat"], ev["lon"], r["lat"], r["lon"]),
                        axis=1)
        near = after[d <= radius_km]
        if near.empty:
            return {"count": 0, "max_mag": None, "latest": None}
        return {"count": int(len(near)),
                "max_mag": float(near["mag"].max()),
                "latest": near.iloc[0]["time"]}
    except Exception:
        return None


# Established, non-predictive guidance (USGS/Red Cross public guidance).
GUIDANCE = (
    "Aftershocks are normal after any significant earthquake. They are most "
    "frequent in the first hours and days and become rarer over time, though "
    "they can continue for weeks. A small proportion of earthquakes — around "
    "1 in 20 — are followed by something larger. Nobody can predict when an "
    "aftershock will occur, so the practical response is preparation: stay "
    "clear of damaged structures, keep exits clear, and be ready to "
    "drop, cover and hold on again."
)
