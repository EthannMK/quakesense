
# QuakeSense — Demo Video Script (3:00 max)

Live URL: https://quakesense-537926118329.asia-southeast1.run.app
GitHub: https://github.com/EthannMK/quakesense

## Before you hit record

- Close other tabs/notifications. Full-screen the browser.
- Preload the app once so the first real quake data is already warm (avoids a slow first load on camera).
- Pick one significant event in Live Monitor ahead of time (M5.5+, on land) — you'll reuse it in the Ask the Data and Response Toolkit sections, so continuity reads well.
- Know your Burmese/Hindi toggle location in My Area before recording — don't hunt for it live.
- Use Win+G (Windows) or QuickTime (Mac) to screen record. Do one silent dry run first.

## Full script — narrate this while you click

### 0:00–0:15 — Hook (15s)
**Say:** "In March 2025, a magnitude 7.7 earthquake hit Mandalay, Myanmar. Millions of people went online searching for answers — and found raw numbers. No context, no guidance, nothing in their own language. QuakeSense turns that raw seismic data into decisions people can actually act on."
**Show:** App already open on Live Monitor, world map visible.

### 0:15–0:45 — Live Monitor (30s)
**Say:** "This is Live Monitor — every earthquake from the last 7 days, worldwide, pulled straight from USGS. Magnitude sizes the dot, older events fade out, and tectonic plate boundaries are overlaid so you can see *why* a region is active."
**Do:** Click the M6+ filter. Point at tsunami auto-flagging if a flagged event is visible.
**Say:** "For any significant event, one click generates an AI community briefing — plain language, matched to real severity, not hype."
**Do:** Click your preselected event → Generate briefing → let it render, read one line aloud.

### 0:45–1:15 — My Area, multilingual moment (30s)
**Say:** "Ask the Data starts with My Area — pick any country and town in the world from verified dropdowns, and get a risk profile built from that area's actual 50-year record."
**Do:** Select a town (e.g. Chiang Mai or Bangkok). Show the scorecards and chart appear.
**Say:** "And this works in 8 languages — including Burmese and Hindi, for the communities who need it most."
**Do:** Switch language to Burmese, regenerate — this is your best visual "wow" moment, let it sit on screen for 2–3 seconds.

### 1:15–2:00 — The agent: verified vs. hallucinated (45s) — the money shot
**Say:** "Below that is the real differentiator: an AI agent that answers any earthquake question — but never guesses."
**Do:** Type: *"How many M6+ earthquakes hit Myanmar since 1990?"* → submit.
**Say:** "It converts that into real SQL, runs it against 86,000 verified USGS records in BigQuery, and shows you both the query and the actual rows."
**Do:** Expand "Show the SQL" panel, scroll the returned rows briefly.
**Say:** "We asked a plain chatbot the same question once — it invented an earthquake that never happened. Ours can't do that. Every number here is checkable against usgs.gov."

### 2:00–2:30 — Anomaly Watch (30s)
**Say:** "Anomaly Watch compares this week's activity in every region against that region's own 50-year average — and flags what's statistically unusual."
**Do:** Open a flagged region (e.g. the 221x-normal one), click Generate AI analysis.
**Say:** "The AI explains the pattern, gives historical context, and offers calm, practical advice for nearby communities — it never predicts what happens next, because earthquakes can't be predicted. We say that on every page."

### 2:30–2:50 — Response Toolkit (20s)
**Say:** "For the people actually responding, there's the Response Toolkit."
**Do:** Navigate to Response Toolkit, select your same preselected event.
**Say:** "One click generates a formal situation report in the format emergency operations centers use, do's-and-don'ts safety guidance in 8 languages, and the nearest hospitals, fire stations, and verified hotlines — located from the event's real coordinates, not guesswork."
**Do:** Briefly show the SITREP card and the facilities table.

### 2:50–3:00 — Close (10s)
**Say:** "USGS data, BigQuery, and Gemini — turned into decisions, in the languages people actually speak. QuakeSense is live now at this URL. Thank you."
**Do:** Cut to the architecture slide or just hold on the live URL in the browser bar.

## Don't forget to mention, somewhere in the 3 minutes

- **Real data only** — 86,000 real USGS events, zero synthetic records.
- **Never predicts** — awareness and preparedness, not forecasting (say this explicitly at least once).
- **Verifiable** — SQL and rows shown, not just an AI's word.
- **8 languages** — accessibility for the communities actually affected.
- **Deployed, not a mockup** — the live Cloud Run URL is the real thing judges can click themselves.

## If you're short on time, cut here first

1. Shorten the Anomaly Watch section to 15s (skip reading the AI analysis aloud, just show it appears).
2. Drop the Burmese language switch to a quick 2-second flash rather than waiting for full regeneration.
3. Never cut the 1:15–2:00 agent section — it's your strongest differentiator and most memorable moment for judges.
