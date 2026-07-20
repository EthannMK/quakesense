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

On first open the app asks two things: how it should look - **Dark** (black
and deep navy), **Light**, or **Warm sand** (beige, brown and neutral earth
tones) - and which language the interface should be in. The whole UI (menus,
headers, captions, buttons) follows that choice. Eight languages ship
hand-translated (English, Burmese, Thai, Hindi, Bengali, Telugu, Marathi,
Tamil); pick "Other" and type **any** language - Gemini translates the entire
interface in one batched call (cached per language, falls back to English if
Vertex AI is unreachable). Both settings can be changed anytime from sidebar
dropdowns, and are kept in the URL so refreshes and shared links preserve
them.

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

Optional but recommended: set `GOOGLE_MAPS_API_KEY` to upgrade the Response
Toolkit's facility finder to live Google Maps data — free-text place search,
addresses, phone numbers, open-now status, an embedded results map, and
per-result Directions links that open Google Maps with live ETA. Enable
**Maps Embed API** (free) and **Places API (New)** on the GCP project and
create an API key. Without the key the app falls back to OpenStreetMap.

## Deploying

```bash
gcloud run deploy quakesense --source . --region asia-southeast1 \
  --allow-unauthenticated --memory 1Gi --min-instances 1
```

Give the Cloud Run service account the same three roles listed above.

`--min-instances 1` keeps one warm instance so visitors never hit a cold start
(30+ seconds of container boot). It costs a little while idle; drop the flag to
scale to zero if cost matters more than first-load speed.

## Data sources

- USGS Earthquake Hazards Program (real-time GeoJSON feeds and the FDSN event
  service) - public domain
- GeoNames cities500 database (town names and coordinates) - CC-BY
- Tectonic plate boundaries from Bird (2003), via the fraxen/tectonicplates
  repository

## Team

Team KODA - Paing Thit Htoo, Rushitha Borra, Ardra T J, Mansi Ramesh Pardeshi.
