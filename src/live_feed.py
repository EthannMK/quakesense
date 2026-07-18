"""USGS real-time earthquake feed (public, no API key).

Feed docs: https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php
Falls back to a bundled sample snapshot if the network is unavailable,
so the demo always renders.
"""
import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests

from src.config import USGS_LIVE_FEED, DATA_DIR

PAGER_LABEL = {
    "green": "GREEN - little/no impact expected",
    "yellow": "YELLOW - local impact possible",
    "orange": "ORANGE - regional impact likely",
    "red": "RED - major disaster likely",
}


def fetch_live(feed_url: str = USGS_LIVE_FEED) -> pd.DataFrame:
    """Fetch past-7-days M2.5+ quakes as a tidy DataFrame (with offline fallback)."""
    try:
        r = requests.get(feed_url, timeout=12)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        with open(os.path.join(DATA_DIR, "sample_feed.geojson"), encoding="utf-8") as f:
            payload = json.load(f)
    return parse_feed(payload)


def parse_feed(payload: dict) -> pd.DataFrame:
    rows = []
    for f in payload["features"]:
        p, (lon, lat, depth) = f["properties"], f["geometry"]["coordinates"]
        if p.get("type") != "earthquake" or p.get("mag") is None:
            continue
        rows.append({
            "id": f["id"],
            "time": datetime.fromtimestamp(p["time"] / 1000, tz=timezone.utc),
            "mag": float(p["mag"]),
            "place": p.get("place") or "unknown location",
            "lat": lat, "lon": lon, "depth_km": depth,
            "tsunami_flag": int(p.get("tsunami") or 0),
            "pager_alert": p.get("alert"),
            "felt_reports": int(p.get("felt") or 0),
            "significance": int(p.get("sig") or 0),
            "url": p.get("url"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("time", ascending=False).reset_index(drop=True)
    return df


def significant_events(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Most briefing-worthy events of the week."""
    if df.empty:
        return df
    return df.sort_values(["significance", "mag"], ascending=False).head(top_n)
