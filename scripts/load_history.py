"""Download USGS historical catalog (global M5+ since 1975) and load to BigQuery.

Chunked by year to respect the 20,000-event API limit. Also writes:
  data/history.csv    - full catalog (local fallback / offline demo)
  data/baseline.csv   - per-grid-cell weekly average counts (anomaly baseline)

Run:  python scripts/load_history.py            # download + BigQuery load
      python scripts/load_history.py --local    # skip BigQuery (no creds needed)
"""
import io
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (USGS_FDSN, HISTORY_START_YEAR, HISTORY_MIN_MAG, GRID_DEG,
                        GCP_PROJECT, BQ_DATASET, BQ_TABLE, DATA_DIR)  # noqa: E402

LOCAL_ONLY = "--local" in sys.argv
THIS_YEAR = datetime.now(timezone.utc).year


def download_year(year: int) -> pd.DataFrame:
    params = {
        "format": "csv",
        "starttime": f"{year}-01-01",
        "endtime": f"{year + 1}-01-01",
        "minmagnitude": HISTORY_MIN_MAG,
        "orderby": "time-asc",
    }
    r = requests.get(USGS_FDSN, params=params, timeout=120)
    r.raise_for_status()
    if not r.text.strip() or "\n" not in r.text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(r.text))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    out_csv = os.path.join(DATA_DIR, "history.csv")

    if os.path.exists(out_csv) and "--reload" not in sys.argv:
        print(f"Found existing {out_csv} - skipping download (use --reload to force).")
        df = pd.read_csv(out_csv)
        df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed")
    else:
        frames = []
        for year in range(HISTORY_START_YEAR, THIS_YEAR + 1):
            try:
                d = download_year(year)
                print(f"  {year}: {len(d)} events")
                if not d.empty:
                    frames.append(d)
            except Exception as e:
                print(f"  {year}: FAILED ({e}) - continuing")
        raw = pd.concat(frames, ignore_index=True)
        df = pd.DataFrame({
            "id": raw["id"],
            "time": pd.to_datetime(raw["time"], utc=True, format="mixed"),
            "latitude": raw["latitude"],
            "longitude": raw["longitude"],
            "depth_km": raw["depth"],
            "mag": raw["mag"],
            "mag_type": raw["magType"],
            "place": raw["place"].fillna("unknown"),
            "type": raw["type"],
        }).dropna(subset=["mag", "latitude", "longitude"])
        df = df[df["type"] == "earthquake"].drop_duplicates("id").reset_index(drop=True)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved {len(df):,} events -> {out_csv}")

    # ---- anomaly baseline: avg weekly M5+ count per grid cell -------------
    weeks = max(1.0, (df["time"].max() - df["time"].min()).days / 7)
    cells = df.assign(
        cell_lat=(df["latitude"] // GRID_DEG) * GRID_DEG,
        cell_lon=(df["longitude"] // GRID_DEG) * GRID_DEG,
    ).groupby(["cell_lat", "cell_lon"]).size().reset_index(name="total")
    cells["weekly_avg"] = cells["total"] / weeks
    cells.to_csv(os.path.join(DATA_DIR, "baseline.csv"), index=False)
    print(f"Saved anomaly baseline ({len(cells)} grid cells) -> data/baseline.csv")

    if LOCAL_ONLY:
        print("Local-only mode: skipping BigQuery load.")
        return

    # ---- BigQuery load ------------------------------------------------------
    from google.cloud import bigquery
    client = bigquery.Client(project=GCP_PROJECT)
    ds = bigquery.Dataset(f"{GCP_PROJECT}.{BQ_DATASET}")
    ds.location = "US"
    client.create_dataset(ds, exists_ok=True)
    table_id = f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    job = client.load_table_from_dataframe(
        df, table_id,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"))
    job.result()
    print(f"Loaded {client.get_table(table_id).num_rows:,} rows -> {table_id}")


if __name__ == "__main__":
    main()
