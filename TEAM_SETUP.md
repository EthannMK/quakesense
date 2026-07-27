# QuakeSense — Team setup guide

Welcome to the team. This doc has two parts: **what has been set up for you**
(access granted on the Google Cloud side) and **what you need to do** to get
running. If anything's off, ping Ethan.

---

## Part A — What you've been granted

Ethan has added your Google account to the Cloud project
**`usar-decision-intel`** with these roles:

| Access granted | What it lets you do |
|---|---|
| **Vertex AI User** | Call Gemini (the AI features) |
| **BigQuery Data Viewer** | Read the 50-year earthquake catalog |
| **BigQuery Job User** | Run queries against it |
| **Service Usage Consumer** *(voice-feature devs only)* | Call the Text-to-Speech API |

Also set up for you:

- The **BigQuery data is already loaded** (`earthquakes_history`) — you only need
  read access, which you have. No data import on your side.
- The **Cloud Text-to-Speech API is enabled** on the project (if you're doing the
  voice feature).
- The **Google Maps API key** is available on request (optional — see Part B,
  step 4).

**You were NOT given a service-account JSON key, and you don't need one.** You
authenticate with your own Google login (Part B, step 3). This is deliberate —
it's more secure and keeps every action tied to you.

---

## Part B — What you need to do

### 1. Install prerequisites

- **Python 3.11+**
- **Google Cloud CLI** — [install guide](https://cloud.google.com/sdk/docs/install)

### 2. Clone and install the project

```bash
git clone https://github.com/EthannMK/quakesense.git
cd quakesense
pip install -r requirements.txt
```

### 3. Log in to Google Cloud (this is your "credential")

Uses your own account — no key file:

```bash
gcloud auth application-default login
gcloud config set project usar-decision-intel
```

A browser opens; sign in with the account Ethan granted access to. This is what
lets the app reach Gemini and BigQuery.

### 4. (Optional) Enable the Maps features

The Response Toolkit's "Find help" map uses a Google Maps key. **The app runs
fine without it** — it falls back to OpenStreetMap. If you want the Maps
features, ask Ethan for the key and set it:

```bash
# Windows (cmd)
set GOOGLE_MAPS_API_KEY=the_key_here
# macOS / Linux
export GOOGLE_MAPS_API_KEY=the_key_here
```

### 5. Run it

```bash
streamlit run app.py
```

Opens at http://localhost:8501. **No data loading needed** — the catalog is in
BigQuery and the town list (`data/towns.csv`) is in the repo. The
`scripts/load_*.py` files are one-time loaders that have already been run; ignore
them.

---

## Project map

| Path | What it is |
|---|---|
| `app.py` | The whole Streamlit UI — pages, layout, the Respond toolkit, news rails |
| `src/ai.py` | The AI layer: Terra (Gemini), question routing, NL→SQL, briefings, SITREP, risk profiles. **Start here for AI-workflow changes.** |
| `src/config.py` | Project ID, region, model name, env vars |
| `src/live_feed.py` | USGS live earthquake feed |
| `src/anomaly.py` | Weekly anomaly / swarm detection |
| `scripts/` | One-time data loaders (already run — don't need to touch) |
| `data/` | Bundled data: towns, baseline, plate boundaries |
| `assets/` | Logos and the Terra avatar |
| `.streamlit/config.toml` | Fixed dark color theme, tuned for low-light readability |

Every AI function in `src/ai.py` follows one pattern: **try the model, fall back
to a deterministic result on failure**, so the app never crashes. Keep that
pattern for anything new.

---

## How we work

- **Branch, don't commit to `main`.** `main` is what's deployed and judged.
  ```bash
  git checkout -b your-feature-name
  ```
- Push your branch and open a **Pull Request** for review before it merges.
- Test locally (`streamlit run app.py`) before pushing — click through the page
  you changed.
- Commit messages: short, present-tense, say what changed and why.

---

## Voice feature (Text-to-Speech) — how it works

Voice is built and live on ✦ Ask: the mic records a question in any language,
Gemini transcribes it natively (no separate speech API), and the reply is
read back with Cloud Text-to-Speech in the best available voice for that
language (`_best_tts_voice` / `text_to_speech` in `src/ai.py`).

- Uses the **Service Usage Consumer** role and the **Cloud Text-to-Speech
  API**, both already enabled on the project — calls it with your normal
  login, no extra keys needed.
- Burmese is a deliberate exception: Cloud TTS has no Burmese voice at all,
  so rather than mispronounce it with an English voice, narration is skipped
  for that language specifically (`_TTS_NO_FALLBACK` in `src/ai.py`) - the
  text answer still shows normally.
- Follows the same fallback pattern as the rest of `src/ai.py` - if TTS
  fails, the app keeps working silently (text answer unaffected), and the
  reason gets printed to server logs rather than raised.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Your default credentials were not found` | Run `gcloud auth application-default login` (Part B, step 3) |
| `403 ... permission denied` on BigQuery | Your IAM roles may not be active yet — confirm with Ethan |
| `Google Places unavailable` in Response Toolkit | Maps key not set or API not enabled — optional, app still works |
| `Google Places unavailable (400: API key not valid)` | Real key wasn't actually passed - check `GOOGLE_MAPS_API_KEY` isn't still the placeholder text from a copy-pasted deploy command |
| Voice replies aren't read aloud (text still works) | Check the Cloud Text-to-Speech API is enabled and the deployed service account has permission to call it - the real reason is printed to Cloud Run logs |
| Historical layer / risk profile fails locally | Almost always missing auth — re-run step 3 |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |

---

*QuakeSense — Team KODA. Earthquakes cannot be predicted; this tool supports
awareness and preparedness, never prediction.*
