"""QuakeSense configuration."""
import os

GCP_PROJECT = os.environ.get("GCP_PROJECT", "usar-decision-intel")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
BQ_DATASET = os.environ.get("BQ_DATASET", "quakesense")
BQ_TABLE = "earthquakes_history"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Google Maps Platform key (enables live Places search + embedded maps in the
# Response Toolkit; the app falls back to OpenStreetMap when unset).
# Needs: Maps Embed API (free) + Places API (New) enabled on the GCP project.
MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# USGS public endpoints (no API key required)
USGS_LIVE_FEED = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.geojson"
USGS_FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Historical catalog scope: global M5+ since 1975
HISTORY_START_YEAR = 1975
HISTORY_MIN_MAG = 5.0

# Anomaly detection grid size in degrees
GRID_DEG = 5.0

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
