# QuakeSense — Demo Video Script (3:00 max)

Live URL: https://quakesense-537926118329.asia-southeast1.run.app
GitHub: https://github.com/EthannMK/quakesense

## Before you hit record

- Close other tabs/notifications. Full-screen the browser.
- Preload the app once so live data is already warm (avoids a slow first load on camera).
- Pick one significant event on Live ahead of time (M5.5+, on land) — reuse it on Ask and Respond so the demo feels continuous.
- Know your interface-language click path (sidebar → Preferences → Language) before recording — don't hunt for it live.
- Have one earthquake question typed and ready to paste into Ask, so you're not typing live.
- Use Win+G (Windows) or QuickTime (Mac) to screen record. Do one silent dry run first.

## Full script — narrate this while you click

### 0:00–0:15 — Hook (15s)
**Say:** "In March 2025, a magnitude 7.7 earthquake hit Mandalay, Myanmar. Millions of people went online searching for answers and found raw numbers — no context, no guidance, nothing in their own language. QuakeSense turns that raw seismic data into decisions people can actually act on."
**Show:** App already open on 🛰️ Live, world map visible.

### 0:15–0:50 — Live (35s)
**Say:** "This is Live — every earthquake from the past 7 days worldwide, straight from USGS. Magnitude sizes the dot, and for any significant event, one click generates a plain-language AI briefing."
**Do:** Click your preselected event → Generate community briefing → read one line aloud.
**Say:** "Below that, Aftershock Outlook shows the official USGS forecast where one exists, plus aftershocks already recorded nearby — it never invents a probability. And this panel flags when a region's activity is way above its own 50-year average, so a swarm or sequence doesn't go unnoticed."
**Do:** Expand the aftershock outlook briefly.

### 0:50–1:15 — My Area (25s)
**Say:** "My Area builds a risk profile for any town on Earth, from what's actually happened within 300 km since 1975."
**Do:** Select a town (e.g. Chiang Mai or Bangkok). Show the scorecards and chart appear.
**Say:** "And it's not just the answers — the whole interface translates into 10 languages, picked for the countries with the heaviest earthquake exposure."
**Do:** Switch interface language in the sidebar, let the nav labels change on screen for 2 seconds.

### 1:15–2:00 — Ask (45s) — the money shot
**Say:** "This is the real differentiator: an agent that answers any earthquake question but never guesses."
**Do:** Paste your preselected question (e.g. "How many M6+ earthquakes hit Myanmar since 1990?") → submit.
**Say:** "It converts that into real SQL, runs it against 86,000 verified USGS records in BigQuery, and shows you both the query and the actual rows — every number here is checkable."
**Do:** Expand the SQL panel briefly.
**Say:** "You can also just ask it out loud."
**Do:** Click the mic, ask a short question by voice, let the spoken reply play for 2–3 seconds.

### 2:00–2:35 — Respond (35s)
**Say:** "Respond is for the hours right after — find the nearest hospitals, fire stations and police from your GPS or any address you type, with a route and arrival time."
**Do:** Show the facility cards and the route map briefly.
**Say:** "It also pulls the weather for that exact spot, because rain after a quake means landslide risk and changes what shelter looks like."
**Do:** Point at the weather advisory box.

### 2:35–2:50 — Close (15s)
**Say:** "Real USGS data, BigQuery, and Gemini, turned into decisions people can act on, in the language they actually speak. QuakeSense is live now — thank you."
**Do:** Cut to the architecture slide, or hold on the live URL in the browser bar.

## Don't forget to mention, somewhere in the 3 minutes

- **Real data only** — 86,000 real USGS events, zero synthetic records.
- **Never predicts** — awareness and preparedness, not forecasting (say this explicitly at least once).
- **Verifiable** — SQL and rows shown, not just an AI's word.
- **10 languages** — accessibility for the communities actually affected.
- **Deployed, not a mockup** — the live Cloud Run URL is the real thing judges can click themselves.

## If you're short on time, cut here first

1. Shorten the My Area language switch to a 1-second flash rather than waiting for full regeneration.
2. Drop the voice demo to a single short question, or skip playback and just show the mic recording.
3. Never cut the 1:15–2:00 Ask section — it's your strongest differentiator and most memorable moment for judges.
