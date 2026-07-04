"""Anomaly / swarm detector.

Compares this week's live M4.5+ activity per 5-degree grid cell against the
historical weekly average (from the USGS catalog baseline). Flags cells with
unusually elevated activity.
"""
import os

import pandas as pd

from src.config import GRID_DEG, DATA_DIR

MIN_MAG = 4.5          # live events counted toward anomaly score
MIN_EVENTS = 3         # ignore cells with fewer current events
MIN_RATIO = 3.0        # current must exceed baseline by this factor


def load_baseline() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "baseline.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["cell_lat", "cell_lon", "total", "weekly_avg"])


def detect(live_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (flagged_cells, live_events_with_cells)."""
    base = load_baseline()
    df = live_df[live_df["mag"] >= MIN_MAG].copy()
    if df.empty:
        return pd.DataFrame(), df
    df["cell_lat"] = (df["lat"] // GRID_DEG) * GRID_DEG
    df["cell_lon"] = (df["lon"] // GRID_DEG) * GRID_DEG
    cur = df.groupby(["cell_lat", "cell_lon"]).agg(
        current=("id", "count"), max_mag=("mag", "max"),
        sample_place=("place", "first")).reset_index()
    merged = cur.merge(base[["cell_lat", "cell_lon", "weekly_avg"]],
                       on=["cell_lat", "cell_lon"], how="left")
    merged["weekly_avg"] = merged["weekly_avg"].fillna(0.05)  # rarely-active cell
    merged["ratio"] = merged["current"] / merged["weekly_avg"]
    flagged = merged[(merged["current"] >= MIN_EVENTS) &
                     (merged["ratio"] >= MIN_RATIO)].copy()
    flagged = flagged.sort_values("ratio", ascending=False).reset_index(drop=True)
    return flagged, df
