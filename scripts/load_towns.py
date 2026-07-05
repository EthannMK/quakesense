"""Download the GeoNames world towns database (cities500: every settlement
with population >= 500, ~200k entries) and write a compact data/towns.csv
used by the My Area country/town dropdowns.

Run once:  python scripts/load_towns.py
"""
import io
import os
import sys
import zipfile

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR  # noqa: E402

URL = "https://download.geonames.org/export/dump/cities500.zip"
COLS = ["geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
        "feature_class", "feature_code", "country_code", "cc2", "admin1", "admin2",
        "admin3", "admin4", "population", "elevation", "dem", "timezone", "modified"]

print("Downloading GeoNames cities500 (~10 MB)...")
r = requests.get(URL, timeout=180)
r.raise_for_status()
with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    with z.open("cities500.txt") as f:
        df = pd.read_csv(f, sep="\t", names=COLS, low_memory=False,
                         usecols=["name", "latitude", "longitude",
                                  "country_code", "admin1", "population"])

# country code -> name
try:
    import pycountry
    cc = {c.alpha_2: c.name for c in pycountry.countries}
except Exception:
    cc = {}
df["country"] = df["country_code"].map(cc).fillna(df["country_code"])

# admin1 code -> human-readable region name (e.g. TH.50 -> Chiang Mai)
print("Downloading region names...")
a1 = requests.get("https://download.geonames.org/export/dump/admin1CodesASCII.txt",
                  timeout=120)
amap = {}
for line in a1.text.splitlines():
    parts = line.split("\t")
    if len(parts) >= 2:
        amap[parts[0]] = parts[1]
df["admin1"] = (df["country_code"].astype(str) + "." + df["admin1"].astype(str)).map(amap)

out = df[["name", "admin1", "country", "latitude", "longitude", "population"]]
out = out.sort_values(["country", "population"], ascending=[True, False])
os.makedirs(DATA_DIR, exist_ok=True)
path = os.path.join(DATA_DIR, "towns.csv")
out.to_csv(path, index=False)
print(f"Saved {len(out):,} towns across {out['country'].nunique()} countries -> {path}")
