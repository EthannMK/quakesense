# QuakeSense

Live earthquake monitoring and AI analysis for communities. Built by Team KODA
for the Gen AI Academy APAC hackathon (theme: AI for Better Living and Smarter
Communities - disaster response).

**Live app:** https://quakesense-537926118329.asia-southeast1.run.app

## What it does

When an earthquake happens, USGS publishes the numbers within minutes. But
magnitude, depth and coordinates don't answer the questions people actually
have: was that dangerous, who is affected, what should we do, and is this
normal for our area?

QuakeSense sits on top of two real USGS data sources - the live global feed
and a 50-year catalog of about 86,000 events that we load into BigQuery - and
uses Gemini to turn them into answers:

- **Live Monitor** - world map of the past 7 days. Filter by magnitude or
  place, see tectonic plate boundaries, get tsunami-flagged events highlighted.
  Below the map you can generate a plain-language briefing for any significant
  event, written for residents rather than seismologists.
- **My Area** - pick your country and town, pick a language (English, Burmese,
  Thai, Hindi, Bengali, Telugu, Marathi or Tamil) and get a risk profile based
  on what has actually happened within 300 km of your town since 1975.
- **Ask about Earthquakes** - a chat agent. Questions about past events get
  answered by generating SQL against the catalog (the SQL is shown, and you
  can check every row). Science and safety questions are answered from
  Gemini's general knowledge. It keeps the conversation context, so
  follow-ups like "and for Japan?" work.
- **Anomaly Watch** - compares this week's activity in each region against
  that region's 50-year average and flags anything unusual, with an AI
  explanation of the pattern.

One thing we were careful about: the app never predicts earthquakes, and says
so on every page. Answers from the catalog end with a note that events are
verified against the USGS record, because we found that plain chatbots
happily invent earthquake lists (one gave us an M9.2 in Myanmar that never
happened there).

## How it's built

USGS live feed + FDSN catalog -> BigQuery (86k events) -> Gemini on Vertex AI
(briefings, NL-to-SQL, risk profiles, anomaly analysis) -> Streamlit app on
Cloud Run. The NL-to-SQL path only accepts SELECT statements, and every AI
call has a plain fallback so the app keeps working if a service is down.

## Running it yourself

You need a GCP project with BigQuery and Vertex AI enabled, and a service
account with BigQuery Data Editor, BigQuery Job User and Vertex AI User roles.

```bash
pip install -r requirements.txt

# point at your service account key
set GOOGLE_APPLICATION_CREDENTIALS=path\to\key.json   # Windows
export GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json # Linux/Mac

python scripts/load_history.py   # downloads the USGS catalog, loads BigQuery (~10 min)
python scripts/load_towns.py     # downloads the GeoNames towns list (~1 min)

streamlit run app.py
```

Project id and region are set in `src/config.py` (or via GCP_PROJECT /
GCP_LOCATION environment variables).

## Deploying

```bash
gcloud run deploy quakesense --source . --region asia-southeast1 \
  --allow-unauthenticated --memory 1Gi
```

Give the Cloud Run service account the same three roles listed above.

## Data sources

- USGS Earthquake Hazards Program (real-time GeoJSON feeds and the FDSN event
  service) - public domain
- GeoNames cities500 database (town names and coordinates) - CC-BY
- Tectonic plate boundaries from Bird (2003), via the fraxen/tectonicplates
  repository

## Team

Team KODA - Paing Thit Htoo, Rushitha Borra, Ardra T J, Mansi Ramesh Pardeshi.
