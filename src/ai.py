"""Gemini AI layer (Vertex AI): situation briefings, NL->SQL analytics, anomaly explainers.

Every function has a deterministic fallback so the demo never stalls.

Latency notes: Gemini 2.5 Flash spends seconds "thinking" by default, so every
call here disables it (these are formatting/translation tasks, not puzzles).
Routing and SQL generation happen in ONE model call, and chat answers stream.
"""
import json
import re

import pandas as pd

from src.config import GCP_PROJECT, GCP_LOCATION, GEMINI_MODEL, BQ_DATASET, BQ_TABLE


_CLIENT = None
_BQ = None


def _pager_label(ev: dict) -> str:
    """pager_alert is missing as either None (fresh from USGS) or NaN (a float,
    once the event has passed through a pandas DataFrame) - both mean 'no PAGER
    assessment issued', never a real value to display."""
    val = ev.get("pager_alert")
    return val if isinstance(val, str) and val.strip() else "not assigned"


def _client():
    """Singleton Gemini client - recreating per call can hit a closed
    underlying HTTP client in Streamlit's rerun model."""
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        _CLIENT = genai.Client(vertexai=True, project=GCP_PROJECT,
                               location=GCP_LOCATION)
    return _CLIENT


def _config(ground: bool = False, **kwargs):
    """GenerateContentConfig with model thinking disabled for latency.

    ground=True attaches the Google Search tool so Gemini can pull current
    web information and return source links (grounding metadata)."""
    from google.genai import types
    if ground:
        kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    try:
        kwargs.setdefault("thinking_config",
                          types.ThinkingConfig(thinking_budget=0))
        return types.GenerateContentConfig(**kwargs)
    except Exception:
        kwargs.pop("thinking_config", None)
        return types.GenerateContentConfig(**kwargs)


def _collect_sources(resp_or_chunk, out: list):
    """Harvest web sources from grounding metadata into `out` (deduped)."""
    try:
        for cand in resp_or_chunk.candidates or []:
            md = getattr(cand, "grounding_metadata", None)
            for gc in (getattr(md, "grounding_chunks", None) or []):
                web = getattr(gc, "web", None)
                uri = getattr(web, "uri", None)
                if uri and all(s["uri"] != uri for s in out):
                    out.append({"title": getattr(web, "title", None) or uri,
                                "uri": uri})
    except Exception:
        pass


def _stream_text(prompt: str, temperature: float, fallback: str,
                 ground: bool = False, sources: list = None):
    """Yield answer chunks; if the model dies mid-answer, finish with the fallback.

    When ground=True, web sources found by Gemini are appended to `sources`
    (read it after the stream is fully consumed)."""
    produced = False
    try:
        resp = _client().models.generate_content_stream(
            model=GEMINI_MODEL, contents=prompt,
            config=_config(ground=ground, temperature=temperature))
        for chunk in resp:
            if chunk.text:
                produced = True
                yield chunk.text
            if sources is not None:
                _collect_sources(chunk, sources)
    except Exception:
        yield ("\n\n---\n*(Answer interrupted - showing summary instead.)*\n\n"
               + fallback) if produced else fallback
        return
    if not produced:
        yield fallback


def translate_ui(strings: dict, language: str) -> dict:
    """Translate the interface string table into any language Gemini knows -
    including low-resource / regional languages. One batched JSON call.

    Keys come back unchanged; only values are translated. Raises on failure
    so the caller can fall back to English."""
    prompt = (
        f"Translate the VALUES of this JSON object into {language}.\n"
        "Rules:\n"
        "- Return ONLY a JSON object with exactly the same keys.\n"
        "- These are UI labels for an earthquake-safety app: keep them short "
        "and natural, the way a native-speaking app would phrase them.\n"
        "- Preserve markdown syntax, emoji, punctuation like '·', and any "
        "placeholders in curly braces such as {n} exactly as they are.\n"
        "- Do NOT translate: QuakeSense, Terra, USGS, SITREP, PAGER, CSV, "
        "GPS, SQL, FEMA, Gemini, BigQuery, M5+/M6+ magnitude notation, "
        "organization names, or URLs.\n\n"
        + json.dumps(strings, ensure_ascii=False))
    resp = _client().models.generate_content(
        model=GEMINI_MODEL, contents=prompt,
        config=_config(temperature=0.1,
                       response_mime_type="application/json"))
    out = json.loads(resp.text)
    if not isinstance(out, dict) or not out:
        raise ValueError("empty translation")
    return {k: out.get(k) or v for k, v in strings.items()}


# ============================================================ briefings ====
BRIEFING_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "what_happened": {"type": "string"},
        "who_is_affected": {"type": "string"},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "caveats": {"type": "string"},
    },
    "required": ["headline", "what_happened", "who_is_affected",
                 "recommended_actions", "caveats"],
}

BRIEFING_PROMPT = """You are a public-safety communication specialist. Write a calm,
plain-language community situation briefing for this earthquake, suitable for citizens,
local officials, and journalists. No jargon. Do not exaggerate. Base everything ONLY on
the data below; where data is missing, say so.

EARTHQUAKE DATA (USGS real-time feed):
- Magnitude: {mag}
- Location: {place} (lat {lat}, lon {lon})
- Depth: {depth_km} km  (shallow <70 km shakes harder at surface)
- Time (UTC): {time}
- USGS PAGER alert level: {pager} (green=minimal, yellow=local, orange=regional, red=major impact expected)
- Tsunami flag: {tsunami}
- Public 'felt' reports submitted: {felt}
- USGS significance score (0-1000+): {sig}

Style: headline is one factual sentence, no sensational words. what_happened and
who_is_affected are 2-3 sentences each, written for a worried resident reading on
a phone - concrete, specific, reassuring where the data allows it.
recommended_actions: 3-5 short imperative items appropriate to the actual severity.
For a minor deep quake, say monitoring is sufficient - do not cause alarm.
"""


def situation_briefing(ev: dict) -> dict:
    try:
        resp = _client().models.generate_content(
            model=GEMINI_MODEL,
            contents=BRIEFING_PROMPT.format(
                mag=ev["mag"], place=ev["place"], lat=ev["lat"], lon=ev["lon"],
                depth_km=round(ev["depth_km"], 1), time=ev["time"],
                pager=_pager_label(ev),
                tsunami="YES - check official tsunami advisories" if ev.get("tsunami_flag") else "no",
                felt=ev.get("felt_reports", 0), sig=ev.get("significance", 0)),
            config=_config(response_mime_type="application/json",
                           response_schema=BRIEFING_SCHEMA, temperature=0.3),
        )
        out = json.loads(resp.text)
        out["source"] = "gemini"
        return out
    except Exception as e:
        sev = "significant" if ev["mag"] >= 6 else "moderate" if ev["mag"] >= 5 else "minor"
        return {
            "headline": f"M{ev['mag']} {sev} earthquake near {ev['place']}",
            "what_happened": f"A magnitude {ev['mag']} earthquake occurred {ev['place']} at "
                             f"{ev['time']} UTC, at a depth of {round(ev['depth_km'], 1)} km.",
            "who_is_affected": "Populations near the epicenter may have felt shaking; "
                               f"{ev.get('felt_reports', 0)} people filed 'felt' reports with USGS.",
            "recommended_actions": [
                "Check on family and neighbors, especially the elderly.",
                "Inspect your building for visible cracks before re-entering.",
                "Expect possible aftershocks; keep exits clear.",
                "Follow official local emergency channels for updates.",
            ],
            "caveats": "Automated template briefing (Gemini unavailable). Data: USGS real-time feed.",
            "source": f"fallback ({str(e)[:60]})",
        }


# ============================================================== NL -> SQL ===
TABLE_FQN = f"`{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`"

_SCHEMA_BLOCK = """Table: {table}
Columns:
  id STRING, time TIMESTAMP, latitude FLOAT64, longitude FLOAT64,
  depth_km FLOAT64, mag FLOAT64, mag_type STRING,
  place STRING  -- e.g. '112 km NNE of Mandalay, Myanmar' (country usually after last comma),
  type STRING   -- always 'earthquake'

Catalog scope: global events with mag >= 5.0 since 1975 (USGS).

SQL rules:
- BigQuery StandardSQL, ONE SELECT statement only. Never modify data.
- Filter countries/regions with LOWER(place) LIKE '%myanmar%' style matching.
- For "how many / count / list" questions about a specific region or period,
  return the EVENT ROWS themselves (date, magnitude, place, depth_km), ordered
  by magnitude, LIMIT 200 - the count is implied and users can verify each row.
  Use pure aggregates only for questions across many groups (per-country/per-year).
- Always add LIMIT 200 unless aggregating.
- ALWAYS alias every output column with a clear human-readable name
  (COUNT(*) AS event_count, MAX(mag) AS max_magnitude).

Example - "How many M6+ earthquakes hit Myanmar since 1990?":
SELECT DATE(time) AS date, mag AS magnitude, place, depth_km
FROM {table}
WHERE LOWER(place) LIKE '%myanmar%' AND mag >= 6.0
  AND time >= TIMESTAMP('1990-01-01')
ORDER BY mag DESC LIMIT 200

Example - "Which countries had the most M7+ quakes since 2000?":
SELECT TRIM(SPLIT(place, ',')[SAFE_OFFSET(ARRAY_LENGTH(SPLIT(place, ',')) - 1)]) AS region,
       COUNT(*) AS event_count, MAX(mag) AS max_magnitude
FROM {table}
WHERE mag >= 7.0 AND time >= TIMESTAMP('2000-01-01')
GROUP BY region ORDER BY event_count DESC LIMIT 30"""

NL2SQL_PROMPT = """You are a BigQuery analytics engineer. Convert the user's question into
ONE BigQuery StandardSQL SELECT statement over this table:

""" + _SCHEMA_BLOCK + """

Conversation so far (use it to resolve follow-up references like 'and for Japan?'
or 'only the strongest ones'):
{history}

Return ONLY the SQL, no explanation, no markdown fences.

User question: {question}
"""

ROUTE_SQL_PROMPT = """You are the router of an earthquake Q&A agent. Classify the question
AND, when the catalog is needed, write the SQL - in a single JSON response.

Routes:
- "live": about earthquakes in the PAST 7 DAYS ("today", "this week", "right now",
  "the latest quake") - answered from the live USGS feed, no SQL needed
- "data": needs the historical USGS catalog (counts, lists, strongest/when/where
  of past events beyond this week)
- "general": earthquake science, safety, preparedness, definitions, news/details
  about a specific recent event, or questions about the QuakeSense app itself -
  no catalog needed
- "hybrid": needs both catalog numbers AND expert knowledge (e.g. an area's
  history AND what residents should do)

Also detect:
- "fresh": true when the answer benefits from CURRENT web information - details or
  impact of a specific recent earthquake, current advisories/warnings, casualty or
  damage reports, anything after 2024. Otherwise false (timeless science/safety).
- "language": the language the question is written in (e.g. "English", "Thai",
  "Myanmar (Burmese)", "Hindi"). Answer language must match the question.

""" + _SCHEMA_BLOCK + """

Conversation so far (use it to resolve follow-ups like 'and for Japan?'):
{history}

Return JSON only:
{{"route": "live" | "data" | "general" | "hybrid",
  "sql": "<the SELECT statement, or empty string when no catalog needed>",
  "fresh": true | false,
  "language": "<language of the question>"}}

User question: {question}"""

ANSWER_PROMPT = """The user asked: "{question}"
This BigQuery SQL was executed: {sql}
Result (as CSV, possibly truncated):
{result}

You are {bot}, QuakeSense's seismic data analyst. Respond in markdown with:
1. A direct answer with the total count and NAMED specific events from the
   result: always call out the strongest (magnitude, place, date) and the most
   recent, plus any notable cluster in time. Bold the key numbers. The full
   table is shown to the user below your answer, so refer to it
   ("the 19 events listed below") instead of repeating rows.
2. "**Insight:**" - a genuinely valuable observation grounded in the result:
   a trend over time, a comparison, a concentration, or a notable extreme.
3. "**Context:**" - expert interpretation: the tectonic setting behind these
   numbers (e.g. the Sagaing Fault for Myanmar) and what they mean practically.
End with: "All events verified against the official USGS record."
Respond entirely in {language}.
Match length to the question: a simple count or lookup deserves ~80-150 words
total; an analytical question 150-280. Never pad.
Style rules: start directly with the answer - no openers like "Based on the
data" or "According to the query". NEVER mention BigQuery, SQL, databases,
queries, or training data in the prose - when attribution is needed, say
"the official USGS earthquake record (1975-today)".
If the result is empty, say no matching events exist in the official record
(M5+ worldwide since 1975) and suggest how to broaden the question.
Never speculate beyond the data. Do not predict future earthquakes."""


def question_to_sql(question: str, history: str = "(none)") -> str:
    resp = _client().models.generate_content(
        model=GEMINI_MODEL,
        contents=NL2SQL_PROMPT.format(table=TABLE_FQN, question=question,
                                      history=history or "(none)"),
        config=_config(temperature=0.1),
    )
    sql = resp.text.strip()
    sql = re.sub(r"^```(sql)?|```$", "", sql, flags=re.MULTILINE).strip()
    return sql


def is_safe_select(sql: str) -> bool:
    s = re.sub(r"\s+", " ", sql).strip().lower()
    if not (s.startswith("select") or s.startswith("with")):
        return False
    banned = ["insert ", "update ", "delete ", "drop ", "create ", "alter ",
              "truncate ", "merge ", "grant ", ";"]
    return not any(b in s + " " for b in banned)


def _bq_client():
    global _BQ
    if _BQ is None:
        from google.cloud import bigquery
        _BQ = bigquery.Client(project=GCP_PROJECT)
    return _BQ


def run_bigquery(sql: str) -> pd.DataFrame:
    return _bq_client().query(sql).to_dataframe()


# ============================================================== feedback ====
_FEEDBACK_READY = False


def log_feedback(question: str, answer: str, mode: str, rating: str) -> bool:
    """Store a thumbs up/down on an AI answer in BigQuery (local CSV fallback).

    Powers the team's answer-quality review loop; never raises."""
    from datetime import datetime, timezone as tz
    row = {"ts": datetime.now(tz.utc).isoformat(),
           "question": (question or "")[:500],
           "answer_snippet": (answer or "")[:300],
           "mode": mode or "", "rating": rating}
    global _FEEDBACK_READY
    try:
        from google.cloud import bigquery
        table_id = f"{GCP_PROJECT}.{BQ_DATASET}.feedback"
        client = _bq_client()
        if not _FEEDBACK_READY:
            schema = [bigquery.SchemaField("ts", "TIMESTAMP"),
                      bigquery.SchemaField("question", "STRING"),
                      bigquery.SchemaField("answer_snippet", "STRING"),
                      bigquery.SchemaField("mode", "STRING"),
                      bigquery.SchemaField("rating", "STRING")]
            client.create_table(bigquery.Table(table_id, schema=schema),
                                exists_ok=True)
            _FEEDBACK_READY = True
        errors = client.insert_rows_json(table_id, [row])
        if errors:
            raise RuntimeError(str(errors)[:120])
        return True
    except Exception:
        try:
            import csv
            import os
            path = os.path.join("data", "feedback_local.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            return True
        except Exception:
            return False


def explain_result(question: str, sql: str, df: pd.DataFrame,
                   language: str = "English") -> str:
    try:
        resp = _client().models.generate_content(
            model=GEMINI_MODEL,
            contents=ANSWER_PROMPT.format(
                question=question, sql=sql, language=language, bot=BOT_NAME,
                result=df.head(50).to_csv(index=False)[:6000]),
            config=_config(temperature=0.2),
        )
        return resp.text.strip()
    except Exception:
        return f"Query returned {len(df)} rows (see table below)."


def ask_the_data(question: str, history: str = "") -> dict:
    """Agentic NL->SQL->answer pipeline with conversation context."""
    sql = question_to_sql(question, history)
    if not is_safe_select(sql):
        raise ValueError(f"Generated SQL failed the safety check (SELECT-only): {sql}")
    df = run_bigquery(sql)
    return {"sql": sql, "df": df, "answer": explain_result(question, sql, df)}


# ================================================================ SITREP ====
SITREP_PROMPT = """You are the duty officer of an emergency operations center.
Produce a formal SITUATION REPORT (SITREP) in markdown for the event below.
Base every statement ONLY on the data given; where data is missing, write
"No confirmed information at this time." Never predict earthquakes.

EVENT (USGS): M{mag} - {place} - {time} UTC - depth {depth} km
PAGER alert: {pager} | Tsunami flag: {tsunami} | Felt reports: {felt}
REGION'S 50-YEAR RECORD (within ~300 km): {hist}
LIVE ACTIVITY nearby this week: {live_near} events

Use exactly these sections:
# SITUATION REPORT - {place}
*Generated {now} UTC · QuakeSense · Data: USGS*
## 1. Summary
## 2. Earthquake Overview
(include expected shaking character: shallow vs deep, what the PAGER level and
felt reports imply about actual impact)
## 3. Areas Potentially Affected
(reason only from place name, depth and magnitude; be explicit about uncertainty)
## 4. Historical Comparison
(is this event unusual for this region's 50-year record, or routine?)
## 5. Immediate Priorities
(numbered, operationally specific, for local authorities - e.g. what to verify
first, which infrastructure to check, aftershock posture)
## 6. Public Safety Message
(short, calm, quotable for radio/social media)
## 7. Verified External Reports
(ONLY if web search returned relevant reports about THIS event from official
agencies or reputable media: 2-3 one-line bullets, each ending with its source
name. If nothing relevant was found, write "No verified external reports at
this time.")
Keep the whole report under 420 words."""


def sitrep(ev: dict, hist: str, live_near: int) -> str:
    from datetime import datetime, timezone as tz
    now = datetime.now(tz.utc).strftime("%Y-%m-%d %H:%M")
    try:
        srcs = []
        resp = _client().models.generate_content(
            model=GEMINI_MODEL,
            contents=SITREP_PROMPT.format(
                mag=ev["mag"], place=ev["place"], time=ev["time"],
                depth=round(ev["depth_km"], 1), pager=_pager_label(ev),
                tsunami="YES" if ev.get("tsunami_flag") else "no",
                felt=ev.get("felt_reports", 0), hist=hist or "not available",
                live_near=live_near, now=now),
            config=_config(ground=True, temperature=0.2))
        _collect_sources(resp, srcs)
        text = resp.text.strip()
        if srcs:
            text += ("\n\n**Web sources:** "
                     + " · ".join(f"[{s['title']}]({s['uri']})"
                                  for s in srcs[:4]))
        return text
    except Exception as e:
        return (f"# SITUATION REPORT - {ev['place']}\n*Generated {now} UTC (template - "
                f"AI unavailable: {str(e)[:60]})*\n\n## Summary\nM{ev['mag']} earthquake, "
                f"{ev['place']}, depth {ev['depth_km']:.0f} km, at {ev['time']} UTC.\n\n"
                f"## Immediate Priorities\n1. Verify impact through local authorities.\n"
                f"2. Check critical infrastructure.\n3. Prepare for aftershocks.\n\n"
                f"## Public Safety Message\nExpect aftershocks. Drop, cover, hold on. "
                f"Follow official channels.")


# ============================================================= do / don't ===
DO_DONT_PROMPT = """You are a disaster-response educator writing for ONE specific
person in an earthquake-affected area BEFORE professional rescue teams arrive.

Event context: {context}
The reader's situation: {situation}
Write everything in {language}.

Write guidance SPECIFIC to this reader's situation - do not give generic
all-purpose advice. Tailor urgency and content to the event's magnitude and
the reader's role: after a moderate quake (below ~M5.5) lead with reassurance
and simple checks; after a strong quake (M6+) lead with structural caution and
aftershock awareness. Where timing matters, say what to do NOW versus in the
next hours. Use only established international guidance (FEMA, Red Cross,
INSARAG community guidance).

Format in markdown:
### Your situation
One short sentence acknowledging their specific situation and this event.
### Do
5-6 short bullets, most important first, specific to their situation.
### Don't
4-5 bullets specific to their situation.
### {closing}

Calm tone, short sentences, under 280 words total. NEVER advise moving heavy
debris or attempting structural rescue - that is for trained teams only."""

DD_SITUATIONS = {
    "I am trapped or sheltering inside": (
        "trapped under or inside a damaged building, possibly alone, phone may be low",
        "How to signal rescuers"),
    "I am safe and want to help others": (
        "physically safe outside, wants to help neighbors without creating new victims",
        "When professional rescuers arrive"),
    "I am a parent / caring for children or elders": (
        "responsible for children or elderly family members in the affected area",
        "Keeping your family calm"),
    "I am a community leader / volunteer coordinator": (
        "organizing neighbors, shelters or information flow before authorities arrive",
        "Working with authorities when they arrive"),
}


def do_dont(context: str, language: str = "English",
            situation_key: str = "I am safe and want to help others") -> str:
    situation, closing = DD_SITUATIONS.get(
        situation_key, DD_SITUATIONS["I am safe and want to help others"])
    try:
        resp = _client().models.generate_content(
            model=GEMINI_MODEL,
            contents=DO_DONT_PROMPT.format(context=context, language=language,
                                           situation=situation, closing=closing),
            config=_config(temperature=0.4))
        return resp.text.strip()
    except Exception:
        return ("### If you are trapped\n**Do** - cover mouth against dust; tap on pipes "
                "or walls in bursts of three; conserve phone battery.\n**Don't** - don't "
                "shout continuously; never use lighters or matches (gas risk).\n\n"
                "### If you are safe\n**Do** - check on neighbors from outside; keep roads "
                "clear for responders; follow official channels.\n**Don't** - don't "
                "re-enter damaged buildings; don't move heavy debris; don't spread rumors.\n\n"
                "### When rescuers arrive\n- Tell them who is missing and where they were "
                "last seen.\n- Follow their instructions - they are trained for this.")


# =========================================================== smart router ===
BOT_NAME = "Terra"

APP_FACTS = """ABOUT QUAKESENSE (answer questions about the app itself from these facts):
QuakeSense is a free web app by Team KODA, built for the Google Cloud Gen AI
Academy APAC hackathon. Its sections: Live Now (world map of every M2.5+ quake
in the past 7 days from the USGS live feed, updated ~5 minutes, with AI community
briefings and global media headlines); Anomaly Watch (compares this week's
activity in each region against its 50-year weekly average, flags 3x+ elevated);
My Area (a risk profile for any town on Earth from the 50-year USGS catalog
within 300 km, in 8 languages); Ask AI (this chat - historical answers are
verified against ~86,000 USGS records in BigQuery with the SQL shown, this-week
answers come from the live feed, current events use live web search with sources
cited); Response Toolkit (formal SITREP, do's & don'ts in 8 languages, nearby
hospitals/fire/police stations from OpenStreetMap, national emergency hotlines).
Every AI answer takes thumbs up/down feedback. QuakeSense never predicts
earthquakes. Data: USGS (public domain), GeoNames (CC-BY), OpenStreetMap.

OFFLINE LIBRARY (Response Toolkit, right-hand panel - official illustrated
PDFs users should download BEFORE a disaster, for offline reading; you can
answer questions about their contents and always point users there to
download them):
- "Drop, Cover, Hold On" visual poster (FEMA/ShakeOut): the protective
  action during shaking - drop to hands and knees, take cover under a sturdy
  table protecting head and neck, hold on until shaking stops; if no table,
  crouch by an interior wall covering head/neck. Never run outside during
  shaking or stand in a doorway.
- "Earthquake Preparedness Checklist" (American Red Cross): preparing before
  a quake - build an emergency kit (water ~4 liters per person per day for 3
  days, non-perishable food, torch, radio, medications, copies of documents),
  make a family communication and evacuation plan, secure tall/heavy
  furniture and the water heater, know how to shut off gas/water/electricity.
- "Earthquake Safety Checklist" (American Red Cross): what to do during and
  after - drop/cover/hold on; stay clear of windows and heavy fixtures; if
  outdoors move to open ground away from buildings and power lines; after
  shaking check for injuries, smell for gas leaks, expect aftershocks, use
  phone only for emergencies.
- "Seven Steps to Earthquake Safety" (Earthquake Country Alliance):
  1) secure your space, 2) plan to be safe, 3) organize disaster supplies,
  4) minimize financial hardship (documents, insurance, retrofit),
  5) drop-cover-hold on during shaking, 6) improve safety after (check
  injuries and damage, prevent fires), 7) reconnect and restore.
- "Putting Down Roots in Earthquake Country" (USGS, ~5.5 MB): illustrated
  handbook - earthquake science, why quakes happen, regional hazards, and
  the seven steps in depth. The most complete reference in the library."""

GENERAL_PROMPT = """You are """ + BOT_NAME + """, QuakeSense's AI assistant - as capable and
helpful as a top general AI assistant, specialized in earthquakes: science,
safety, preparedness, engineering, history, and current events. You serve
citizens, officials and journalists. If asked who you are, you are """ + BOT_NAME + """
(Google's Gemini 2.5 Flash model on Vertex AI), integrated into QuakeSense by
Team KODA.

""" + APP_FACTS + """

Conversation so far:
{history}

Question: {q}

Respond entirely in {language}.

Give a genuinely informative, well-organized answer in markdown: short paragraphs
and/or a few bold-labelled points, the way a knowledgeable expert would explain it
to an intelligent non-specialist. Cover the why/how, not just the what, with
concrete examples where they help. Match length to the question - a definition
deserves ~80 words, a "why/how" or current-events question 150-320. Never pad.

Accuracy rules:
- Never state precise historical statistics (exact counts, "the Nth strongest",
  full event lists) from memory - famous well-documented facts (e.g. the 2011
  M9.1 Tohoku earthquake) are fine, but for verifiable numbers tell the user the
  verified catalog can answer it and end with one suggested data question
  (prefix "Try asking: ").
- When web search results are available, use them for current facts and recent
  events - prefer official sources (USGS, national agencies) over news.
- Be accurate, calm and practical; use established seismology and official
  safety guidance (drop-cover-hold-on, etc.).
- Never predict earthquakes or give probabilities of future events."""

HYBRID_PROMPT = """You are """ + BOT_NAME + """, QuakeSense's earthquake analyst agent.
The user asked: "{q}"
Catalog query executed: {sql}
Catalog result (CSV, truncated): {result}

Respond entirely in {language}.

Combine the DATA with expert knowledge in markdown (150-280 words):
1. Answer with concrete numbers from the result - bold the key figures.
2. "**Insight:**" - one valuable observation from the data.
3. "**Context:**" - a substantial paragraph of expert interpretation: the
   tectonic setting, why the pattern exists, and practical guidance where
   relevant. Explain like an expert talking to an intelligent non-specialist.
Style: start directly with the answer - no "Based on the data" openers. Never
mention BigQuery, SQL or databases in prose; attribute facts to "the official
USGS earthquake record" when needed.
Never predict future earthquakes."""

LIVE_PROMPT = """You are """ + BOT_NAME + """, QuakeSense's live-monitoring analyst. The user asked: "{q}"

LIVE USGS FEED - every M2.5+ earthquake worldwide in the past 7 days
(UTC times; CSV, possibly truncated):
{feed}

Respond entirely in {language}.

Answer from THIS FEED ONLY, in markdown (80-200 words): give the concrete
numbers and name the specific events that answer the question (magnitude, place,
time), bolding key figures. If the question is about a place with no events in
the feed, say clearly that no M2.5+ events were recorded there in the past
7 days - that is a meaningful, reassuring answer.
Attribute facts to "the live USGS feed (past 7 days)". Do not invent events.
Never predict future earthquakes.
End with: "Source: USGS real-time feed, updated every 5 minutes." """


def route_and_sql(question: str, history: str = "") -> dict:
    """One model call: route + SQL + freshness flag + question language."""
    try:
        resp = _client().models.generate_content(
            model=GEMINI_MODEL,
            contents=ROUTE_SQL_PROMPT.format(table=TABLE_FQN, question=question,
                                             history=history or "(none)"),
            config=_config(response_mime_type="application/json",
                           temperature=0.0))
        out = json.loads(resp.text)
        sql = (out.get("sql") or "").strip()
        sql = re.sub(r"^```(sql)?|```$", "", sql, flags=re.MULTILINE).strip()
        return {"route": out.get("route", "general"), "sql": sql,
                "fresh": bool(out.get("fresh")),
                "language": out.get("language") or "English"}
    except Exception:
        return {"route": "general", "sql": "", "fresh": False,
                "language": "English"}


def smart_ask(question: str, history: str = "", stream: bool = False,
              live_df: pd.DataFrame = None) -> dict:
    """Route a question to the live feed, catalog SQL, general knowledge, or both.

    With stream=True the returned dict carries a "stream" generator of answer
    chunks (feed it to st.write_stream) plus a "sources" list that fills with
    web citations while the stream is consumed; otherwise "answer" holds the
    full text. Answers match the language of the question.
    """
    r = route_and_sql(question, history)
    route, sql, lang = r["route"], r["sql"], r["language"]

    note = ""
    if route == "live" and live_df is not None and not live_df.empty:
        feed = live_df[["time", "mag", "place", "depth_km", "tsunami_flag"]]
        feed_csv = feed.head(300).to_csv(index=False)[:15000]
        prompt = LIVE_PROMPT.format(q=question, feed=feed_csv, language=lang)
        fallback = ("The live feed is loaded on the Live Monitor page - "
                    "the answer service is unavailable right now.")
        if stream:
            return {"mode": "live", "sql": None, "df": None, "sources": [],
                    "stream": _stream_text(prompt, 0.2, fallback)}
        resp = _client().models.generate_content(
            model=GEMINI_MODEL, contents=prompt, config=_config(temperature=0.2))
        return {"mode": "live", "sql": None, "df": None,
                "answer": resp.text.strip()}

    if route in ("data", "hybrid") and sql:
        try:
            if not is_safe_select(sql):
                raise ValueError("unsafe SQL")
            df = run_bigquery(sql)
            fallback = f"Query returned {len(df)} rows (see table below)."
            if route == "data":
                prompt = ANSWER_PROMPT.format(
                    question=question, sql=sql, language=lang, bot=BOT_NAME,
                    result=df.head(50).to_csv(index=False)[:6000])
                temp = 0.2
            else:
                prompt = HYBRID_PROMPT.format(
                    q=question, sql=sql, language=lang,
                    result=df.head(40).to_csv(index=False)[:5000])
                temp = 0.3
            if stream:
                return {"mode": route, "sql": sql, "df": df, "sources": [],
                        "stream": _stream_text(prompt, temp, fallback)}
            resp = _client().models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
                config=_config(temperature=temp))
            return {"mode": route, "sql": sql, "df": df,
                    "answer": resp.text.strip()}
        except Exception as e:
            note = (f"Catalog unavailable ({str(e)[:90]}) - answered from general "
                    f"knowledge instead. Check BigQuery credentials.")

    prompt = GENERAL_PROMPT.format(q=question, history=history or "(none)",
                                   language=lang)
    ground = r["fresh"]
    if stream:
        sources = []
        return {"mode": "general", "sql": None, "df": None, "note": note,
                "sources": sources,
                "stream": _stream_text(prompt, 0.4,
                                       "The knowledge service is unavailable "
                                       "right now - please try again shortly.",
                                       ground=ground, sources=sources)}
    srcs = []
    resp = _client().models.generate_content(
        model=GEMINI_MODEL, contents=prompt,
        config=_config(ground=ground, temperature=0.4))
    _collect_sources(resp, srcs)
    return {"mode": "general", "sql": None, "df": None,
            "answer": resp.text.strip(), "note": note, "sources": srcs}


def prioritize_facilities(context: str, facilities: str) -> str:
    """Terra's short recommendation over a list of nearby facilities."""
    prompt = (
        f"You are {BOT_NAME}, QuakeSense's assistant. Situation: {context}.\n"
        f"Nearby facilities (name | address | phone | distance km | open now):\n"
        f"{facilities}\n\n"
        "In under 100 words of markdown, recommend which facility to head to "
        "first and why (type of facility, distance, open status), name a backup, "
        "and remind the reader to call ahead if phone lines work. Practical "
        "logistics only - no medical advice, no predictions.")
    try:
        resp = _client().models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=_config(temperature=0.2))
        return resp.text.strip()
    except Exception:
        return ("Choose the nearest open hospital for injuries; fire stations "
                "coordinate rescue. Call ahead if lines work - facilities may "
                "be damaged or overloaded after a major quake.")


# ========================================================= area profile ====
AREA_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "seismic_context": {"type": "string",
                            "description": "2-3 sentences: what the 50-year record says about this area"},
        "this_week": {"type": "string", "description": "1-2 sentences on current nearby activity"},
        "preparedness_actions": {"type": "array", "items": {"type": "string"},
                                 "description": "4 practical actions proportionate to actual risk"},
        "caveats": {"type": "string"},
    },
    "required": ["headline", "seismic_context", "this_week",
                 "preparedness_actions", "caveats"],
}

AREA_PROMPT = """You are a seismic risk communicator writing for residents of {place}.
Create a community risk profile based ONLY on this data. Be calm and specific -
if the record shows low local activity, say so plainly. If activity is mostly distant,
mention that tall buildings can still feel long-period shaking from far events.
Never predict earthquakes. Write ALL output in {language}.

DATA (USGS catalog, M5+ since 1975, within 300 km of {place}):
- Total events: {count}
- Strongest: {strongest}
- Most recent M5+ in the area: {latest}
- Average per decade: {per_decade}

LIVE (past 7 days within 500 km): {live_count} events, strongest {live_max}

Quality bar:
- seismic_context: translate the numbers into plain meaning - characterize the
  activity level in words (very low / low / moderate / high for an inhabited
  area), name the general tectonic driver if the location makes it clear (e.g.
  a nearby plate boundary, subduction zone or major fault), and put the
  strongest recorded event in human terms (what shaking of that size feels
  like at that distance).
- this_week: interpret, don't restate - is current activity normal for this
  area or worth noting?
- preparedness_actions: exactly 4, practical, proportionate to this actual risk
  level, and specific enough to act on today (not "be prepared" but what to do).
  For low-risk areas focus on awareness and travel; for active areas focus on
  home safety and family plans.
- caveats: one honest sentence about what this data cannot tell them (local
  soil, building quality), not boilerplate.
"""


def area_profile(place: str, hist: dict, live_near: dict,
                 language: str = "English") -> dict:
    try:
        resp = _client().models.generate_content(
            model=GEMINI_MODEL,
            contents=AREA_PROMPT.format(
                place=place, language=language, count=hist["count"],
                strongest=hist["strongest"], latest=hist["latest"],
                per_decade=hist["per_decade"],
                live_count=live_near["count"], live_max=live_near["max"]),
            config=_config(response_mime_type="application/json",
                           response_schema=AREA_SCHEMA, temperature=0.3),
        )
        out = json.loads(resp.text)
        out["source"] = "gemini"
        return out
    except Exception as e:
        return {
            "headline": f"Seismic profile for {place}",
            "seismic_context": f"The USGS catalog records {hist['count']} M5+ events within "
                               f"300 km since 1975. Strongest: {hist['strongest']}.",
            "this_week": f"{live_near['count']} events within 500 km in the past 7 days.",
            "preparedness_actions": [
                "Identify safe spots at home: under sturdy tables, away from windows.",
                "Keep a basic emergency kit: water, torch, first aid, copies of documents.",
                "Agree a family meeting point and out-of-area contact.",
                "Check your building's condition and secure heavy furniture to walls.",
            ],
            "caveats": "Template profile (Gemini unavailable). Data: USGS catalog.",
            "source": f"fallback ({str(e)[:60]})",
        }


# ======================================================== anomaly explain ===
def explain_anomaly(cell: dict, events: pd.DataFrame,
                    hist_context: str = "") -> str:
    ev_lines = "; ".join(
        f"M{r.mag:.1f} {r.place} ({r.time:%b %d %H:%M} UTC, {r.depth_km:.0f} km deep)"
        for r in events.sort_values('time').head(10).itertuples())
    try:
        prompt = (
            "You are a seismologist-analyst writing for the public. Analyze this week's "
            f"elevated activity in the region around ({cell['cell_lat']}, {cell['cell_lon']}).\n\n"
            f"THIS WEEK: {cell['current']} events M4.5+ vs a 50-year average of "
            f"{cell['weekly_avg']:.2f}/week ({cell['ratio']:.0f}x normal). Strongest: "
            f"M{cell.get('max_mag', 0):.1f}.\n"
            f"EVENT SEQUENCE (chronological): {ev_lines}\n"
            f"REGION'S 50-YEAR RECORD: {hist_context or 'not available'}\n\n"
            "Write ~150 words of markdown with exactly these bold-labelled parts:\n"
            "**Pattern** - classify it (mainshock-aftershock sequence, swarm, or elevated "
            "background) with reasoning from the magnitudes and timing above.\n"
            "**Historical context** - how this week compares to the region's record.\n"
            "**For nearby communities** - 2-3 calm, practical points.\n"
            "Never predict; state clearly that elevated activity does not guarantee a larger event.")
        resp = _client().models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=_config(temperature=0.3))
        return resp.text.strip()
    except Exception:
        places = ", ".join(events["place"].head(5).tolist())
        return (f"**Pattern** - {cell['current']} M4.5+ events this week vs "
                f"{cell['weekly_avg']:.2f}/week normally ({cell['ratio']:.0f}x) — likely an "
                f"aftershock sequence or swarm near: {places}.\n\n**For nearby communities** - "
                f"review preparedness, secure heavy items, follow official guidance. Earthquakes "
                f"cannot be predicted; elevated activity does not guarantee a larger event.")
