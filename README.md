# QuakeSense

Live earthquake intelligence for communities. Built by Team KODA for the Google
Cloud Gen AI Academy APAC hackathon (theme: AI for Better Living and Smarter
Communities — disaster response).

**Live app:** https://quakesense-537926118329.asia-southeast1.run.app

## The problem

When an earthquake happens, USGS publishes the numbers within minutes. But
magnitude, depth and coordinates don't answer the questions people actually
have: *was that dangerous, does it affect me, can I trust what I'm reading,
and where do I go now?*

QuakeSense answers those four questions — and its four sections map to them
one-for-one.

## What it does

| Section | Answers | Data behind it |
|---|---|---|
| **🛰️ Live** | What's happening right now? | USGS live feed (7 days, M2.5+) |
| **📍 My Area** | What does it mean for my community? | 50-year USGS catalog in BigQuery |
| **✦ Ask** | Can I trust it? | Catalog + live feed + web, **with the query shown** |
| **⛑️ Respond** | How do I get to safety? | Google Maps + official safety guides |

**🛰️ Live** — world map of every M2.5+ earthquake in the past 7 days, with
magnitude filters, place search, optional tectonic plate boundaries and tsunami
flagging. Below it: **AI community briefings** for any significant event
(plain language, written for residents, not seismologists), an
**unusual-activity** panel that compares this week's regional activity against
that region's own 50-year weekly average, plus **global media coverage** and
**official updates** from the USGS record and UN agencies.

**📍 My Area** — pick any country and town on Earth and get a risk profile
built from what has actually happened within 300 km since 1975, in **8
languages** (English, Burmese, Thai, Hindi, Bengali, Telugu, Marathi, Tamil),
with charts and a map of every nearby M5+ epicentre.

**✦ Ask** — a chat agent that answers in **whatever language you ask in**, and
routes each question to the right source: historical questions become SQL over
~86,000 verified USGS records (**the generated SQL and the matching rows are
both displayed**, so any figure can be checked); this-week questions come from
the live feed; current events are answered with Google Search grounding and
**cited sources**. Every answer takes a 👍/👎 that is logged for review.

**⛑️ Respond** — find the nearest hospitals, fire stations, police, pharmacies
or shelters **from your own GPS location** (or any place you type), with
addresses, phone numbers, open-now status, a route map with arrival time, and
one-tap links into Google Maps for live navigation. Below it, an **offline
library** of official illustrated safety guides (FEMA, American Red Cross,
USGS, Ready.gov, Earthquake Country Alliance) to download *before* a disaster,
for when networks go down.

A floating **💬 assistant** on every page except Ask answers quick questions
about whatever is currently on screen, naming the specific event or location.

### On trust

QuakeSense **never predicts earthquakes**, and says so on every page. Answers
drawn from the catalog show their SQL and end with a note that events are
verified against the USGS record — we built this after finding that general
chatbots happily invent earthquakes (one gave us an M9.2 in Myanmar that never
happened).

## How it's built

```
USGS live feed + FDSN catalog ──► BigQuery (86k events)
                                        │
        Gemini 2.5 Flash on Vertex AI ──┤  briefings · NL→SQL · risk profiles
        (streaming, Search grounding)   │  anomaly analysis · assistant
                                        ▼
                         Streamlit app on Cloud Run
                  (+ Google Maps Platform, GDELT, ReliefWeb)
```

Design choices that matter:

- **Only `SELECT` reaches BigQuery.** Generated SQL is validated before it runs.
- **Every AI call has a deterministic fallback**, so a service outage degrades
  the answer instead of breaking the page.
- **Content chains never go empty** — if world media is unreachable, the news
  rail falls back to official USGS event cards.
- **Partial reruns** (`st.fragment`) keep the map and tables from re-rendering
  on every interaction.

## Running it yourself

You need a GCP project with **BigQuery** and **Vertex AI** enabled, and access
with BigQuery Data Viewer, BigQuery Job User and Vertex AI User roles.

```bash
pip install -r requirements.txt

# authenticate as yourself (no key file needed)
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

python scripts/load_history.py   # USGS catalog → BigQuery (~10 min, one time)
python scripts/load_towns.py     # GeoNames towns list (~1 min, one time)

streamlit run app.py
```

Project id, region and model are set in `src/config.py`, or via the
`GCP_PROJECT`, `GCP_LOCATION` and `GEMINI_MODEL` environment variables.

**Optional — Google Maps:** set `GOOGLE_MAPS_API_KEY` to enable the Respond
page's live facility finder (place search, phone numbers, open-now status,
route map with ETA). Enable **Maps Embed API** (free) and **Places API (New)**,
then create an API key restricted to those two APIs. Without a key the app
falls back to OpenStreetMap.

New to the project? See **[TEAM_SETUP.md](TEAM_SETUP.md)**.

## Deploying

```bash
gcloud run deploy quakesense --source . --region asia-southeast1 \
  --allow-unauthenticated --memory 2Gi --cpu 2 --cpu-boost \
  --concurrency 40 --min-instances 1 \
  --set-env-vars GOOGLE_MAPS_API_KEY=your_key
```

Give the Cloud Run service account the same roles listed above.
`--min-instances 1` keeps one instance warm so visitors never hit a cold start;
drop it to scale to zero if idle cost matters more than first-load speed.

## Roadmap

Shaped by what people in earthquake-affected areas told us they actually want:

- **Area alerts** — save your town, get an email/push within minutes of a
  significant quake nearby, with the plain-language briefing already written.
- **Aftershock outlook** — surface the official USGS aftershock forecast for an
  event. (Aftershock *forecasting* is established statistics; earthquake
  *prediction* is not, and stays out of scope.)
- **Weather with the response** — rain changes what people need: landslide risk
  on shaken slopes, and shelter for anyone sleeping outside.
- **Works offline** — installable app shell with the last known data cached.
- **Regional agency feeds** (JMA, PHIVOLCS, Thai TMD) alongside USGS.

## Data sources

- **USGS Earthquake Hazards Program** — real-time GeoJSON feeds and the FDSN
  event service (public domain)
- **GeoNames** cities500 database — town names and coordinates (CC-BY 4.0)
- **Tectonic plate boundaries** — Bird (2003), via fraxen/tectonicplates
- **GDELT Project** — global news index · **ReliefWeb (UN OCHA)** — humanitarian
  situation reports · **Google Maps Platform** — places, routing and maps

The offline library links to official publications hosted by their publishers
(FEMA, American Red Cross, USGS, Ready.gov, Earthquake Country Alliance); those
documents remain under their own copyrights and are not redistributed here.

## License

QuakeSense is released under the [MIT License](LICENSE) — free to use, modify
and distribute.

The MIT License covers this project's own source code. Third-party data keeps
its own terms: the bundled GeoNames town data (`data/towns.csv`) is CC-BY 4.0
and requires attribution to GeoNames; USGS data is public domain.

## Team

Team KODA — Paing Thit Htoo, Rushitha Borra, Ardra T J, Mansi Ramesh Pardeshi.
