"""Gemini AI layer (Vertex AI): situation briefings, NL->SQL analytics, anomaly explainers.

Every function has a deterministic fallback so the demo never stalls.
"""
import json
import re

import pandas as pd

from src.config import GCP_PROJECT, GCP_LOCATION, GEMINI_MODEL, BQ_DATASET, BQ_TABLE


_CLIENT = None


def _client():
    """Singleton Gemini client - recreating per call can hit a closed
    underlying HTTP client in Streamlit's rerun model."""
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        _CLIENT = genai.Client(vertexai=True, project=GCP_PROJECT,
                               location=GCP_LOCATION)
    return _CLIENT


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

recommended_actions: 3-5 short imperative items appropriate to the actual severity.
For a minor deep quake, say monitoring is sufficient - do not cause alarm.
"""


def situation_briefing(ev: dict) -> dict:
    try:
        from google.genai import types
        resp = _client().models.generate_content(
            model=GEMINI_MODEL,
            contents=BRIEFING_PROMPT.format(
                mag=ev["mag"], place=ev["place"], lat=ev["lat"], lon=ev["lon"],
                depth_km=round(ev["depth_km"], 1), time=ev["time"],
                pager=ev.get("pager_alert") or "not assigned",
                tsunami="YES - check official tsunami advisories" if ev.get("tsunami_flag") else "no",
                felt=ev.get("felt_reports", 0), sig=ev.get("significance", 0)),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
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

NL2SQL_PROMPT = """You are a BigQuery analytics engineer. Convert the user's question into
ONE BigQuery StandardSQL SELECT statement over this table:

Table: {table}
Columns:
  id STRING, time TIMESTAMP, latitude FLOAT64, longitude FLOAT64,
  depth_km FLOAT64, mag FLOAT64, mag_type STRING,
  place STRING  -- e.g. '112 km NNE of Mandalay, Myanmar' (country usually after last comma),
  type STRING   -- always 'earthquake'

Catalog scope: global events with mag >= 5.0 since 1975 (USGS).
Rules:
- SELECT statements only. Never modify data.
- Filter countries/regions with LOWER(place) LIKE '%myanmar%' style matching.
- Always add LIMIT 200 unless aggregating.
- Return ONLY the SQL, no explanation, no markdown fences.

User question: {question}
"""

ANSWER_PROMPT = """The user asked: "{question}"
This BigQuery SQL was executed: {sql}
Result (as CSV, possibly truncated):
{result}

Answer the user's question in 2-4 clear sentences based only on this result.
Mention concrete numbers. If the result is empty, say no matching events were found in
the catalog (global M5+ since 1975)."""


def question_to_sql(question: str) -> str:
    from google.genai import types
    resp = _client().models.generate_content(
        model=GEMINI_MODEL,
        contents=NL2SQL_PROMPT.format(table=TABLE_FQN, question=question),
        config=types.GenerateContentConfig(temperature=0.1),
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


def run_bigquery(sql: str) -> pd.DataFrame:
    from google.cloud import bigquery
    client = bigquery.Client(project=GCP_PROJECT)
    return client.query(sql).to_dataframe()


def explain_result(question: str, sql: str, df: pd.DataFrame) -> str:
    try:
        from google.genai import types
        resp = _client().models.generate_content(
            model=GEMINI_MODEL,
            contents=ANSWER_PROMPT.format(
                question=question, sql=sql,
                result=df.head(50).to_csv(index=False)[:6000]),
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return resp.text.strip()
    except Exception:
        return f"Query returned {len(df)} rows (see table below)."


def ask_the_data(question: str) -> dict:
    """Full NL->SQL->answer pipeline. Raises with clear message on failure."""
    sql = question_to_sql(question)
    if not is_safe_select(sql):
        raise ValueError(f"Generated SQL failed the safety check (SELECT-only): {sql}")
    df = run_bigquery(sql)
    return {"sql": sql, "df": df, "answer": explain_result(question, sql, df)}


# ======================================================== anomaly explain ===
def explain_anomaly(cell: dict, events: pd.DataFrame) -> str:
    places = ", ".join(events["place"].head(5).tolist())
    try:
        from google.genai import types
        prompt = (f"You are a seismologist writing for the public. In the last 7 days, the "
                  f"region around ({cell['cell_lat']}, {cell['cell_lon']}) recorded "
                  f"{cell['current']} M4.5+ earthquakes vs a historical average of "
                  f"{cell['weekly_avg']:.2f} per week ({cell['ratio']:.0f}x normal). "
                  f"Recent events: {places}. In 3-4 sentences: what could this pattern mean "
                  f"(aftershock sequence, swarm, or foreshock uncertainty), and what should "
                  f"nearby communities do? Be calm and factual; note forecasting limits.")
        resp = _client().models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3))
        return resp.text.strip()
    except Exception:
        return (f"This area recorded {cell['current']} M4.5+ events this week vs a historical "
                f"average of {cell['weekly_avg']:.2f}/week ({cell['ratio']:.0f}x normal) — "
                f"likely an aftershock sequence or swarm near: {places}. Communities nearby "
                f"should review preparedness and follow official guidance. Earthquakes cannot "
                f"be predicted; elevated activity does not guarantee a larger event.")
