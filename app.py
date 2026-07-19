"""QuakeSense — Global real-time earthquake intelligence.

Live USGS data + Vertex AI Gemini + BigQuery, served by Streamlit.

Run:  streamlit run app.py
"""
import base64
import html
import math
import os
import re
from datetime import datetime, timezone

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
import streamlit.components.v1 as components

from src.ai import (situation_briefing, smart_ask, explain_anomaly,
                    area_profile, sitrep, do_dont, run_bigquery, TABLE_FQN,
                    log_feedback, prioritize_facilities)
from src.config import MAPS_API_KEY
from src.anomaly import detect
from src.live_feed import fetch_live, significant_events, PAGER_LABEL

EMERGENCY_NUMBERS = {
    "Thailand": "Police 191 · Ambulance 1669 · Disaster hotline (DDPM) 1784",
    "Myanmar": "Police 199 · Fire 191 · Ambulance 192",
    "India": "All emergencies 112 · Disaster helpline 1078",
    "Indonesia": "All emergencies 112 · Ambulance 118",
    "Japan": "Police 110 · Fire / Ambulance 119",
    "Philippines": "All emergencies 911",
    "Nepal": "Police 100 · Ambulance 102",
    "Bangladesh": "All emergencies 999",
    "Pakistan": "Rescue 1122",
    "China": "Police 110 · Ambulance 120 · Fire 119",
    "Vietnam": "Police 113 · Ambulance 115 · Fire 114",
    "Laos": "Police 191 · Ambulance 195",
    "Türkiye": "All emergencies 112",
    "United States": "911", "Mexico": "911", "Chile": "Ambulance 131 · Police 133",
    "New Zealand": "111", "Italy": "112", "Greece": "112",
}


st.set_page_config(page_title="QuakeSense - Global real-time earthquake intelligence",
                   page_icon=":material/earthquake:", layout="wide")

# ------------------------------------------------------------------ style --
st.markdown("""
<style>
/* Flag-only emoji font: Windows browsers can't render country-flag emoji
   natively; this webfont covers ONLY the flag codepoints (unicode-range),
   so all other text falls through to the normal font stack. */
@font-face {
  font-family: "Twemoji Country Flags";
  src: url("https://cdn.jsdelivr.net/npm/country-flag-emoji-polyfill@0.1.8/dist/TwemojiCountryFlags.woff2") format("woff2");
  unicode-range: U+1F1E6-1F1FF, U+1F3F4, U+E0062-E007F;
  font-display: swap;
}
[data-baseweb="select"] div, [data-baseweb="popover"] li,
[data-baseweb="popover"] li div, [data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p, [data-testid="stChatMessage"] p {
  font-family: "Twemoji Country Flags", "Source Sans Pro", "Source Sans 3",
               -apple-system, "Segoe UI", sans-serif;
}

#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.0rem; padding-bottom: 1.5rem;}

.qs-header {
  border-bottom: 1px solid #263145;
  padding: 0 0 0.8rem 0; margin-bottom: 0.3rem;
}
.qs-wordmark {
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  font-size: 1.6rem; font-weight: 600; letter-spacing: 0.10em;
  color: #dbe2ec; margin: 0;
}
.qs-wordmark span {color: #e08850;}
.qs-subline {
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  font-size: 0.70rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: #8fa0b5; margin-top: 0.25rem;
}
.qs-live {
  display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: #6fae7f; margin-right: 6px;
}

[data-testid="stMetric"] {
  background: #161e2e; border: 1px solid #263145; border-radius: 6px;
  padding: 12px 14px 9px 14px;
}
[data-testid="stMetricLabel"] p {
  font-size: 0.67rem !important; letter-spacing: 0.12em;
  text-transform: uppercase; color: #8fa0b5 !important;
}
[data-testid="stMetricValue"] {
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  font-variant-numeric: tabular-nums; font-size: 1.65rem !important;
}

h2, h3 {letter-spacing: 0.01em; color: #dbe2ec;}
section[data-testid="stSidebar"] {border-right: 1px solid #263145;}
section[data-testid="stSidebar"] .stRadio label p {font-size: 0.90rem;}

.qs-credit {
  font-size: 0.68rem; letter-spacing: 0.10em; text-transform: uppercase;
  color: #8fa0b5; margin-bottom: 0.15rem;
}
.qs-credit-items {font-size: 0.76rem; color: #c3cede; line-height: 1.5; margin-bottom: 0.55rem;}
.qs-sidebar-bottom {
  position: fixed; bottom: 0; left: 0.9rem; width: 15rem;
  padding: 0.7rem 0 1.0rem 0; background: #161e2e;
  border-top: 1px solid #263145; z-index: 999;
}
.qs-team {font-size: 0.72rem; color: #8fa0b5; margin-top: 0.4rem;}

.stButton button[kind="primary"] {
  letter-spacing: 0.06em; font-weight: 600; border-radius: 4px;
}

/* Floating quick-ask popup (bottom-right on every relevant page) */
div[data-testid="stPopover"] {
  position: fixed !important; bottom: 1.2rem; right: 1.2rem;
  left: auto !important; width: auto !important; z-index: 999;
}
button[data-testid="stPopoverButton"] {
  width: auto !important; border-radius: 999px !important;
  padding: 0.5rem 1.15rem; font-weight: 600;
  background: #e08850 !important; color: #0d1321 !important;
  border: none !important; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
}
button[data-testid="stPopoverButton"]:hover {
  background: #eda06b !important; color: #0d1321 !important;
}
div[data-testid="stPopoverBody"] {min-width: min(400px, 94vw);}
[data-stale="true"] div[data-testid="stPopover"] {display: none !important;}

/* Live event ticker under the header */
.qs-ticker {
  overflow: hidden; white-space: nowrap; border: 1px solid #263145;
  border-radius: 6px; background: #161e2e; padding: 0.35rem 0;
  margin: 0.35rem 0 0.75rem 0; position: relative;
}
.qs-ticker-inner {
  display: inline-block; padding-left: 100%;
  animation: qs-scroll 60s linear infinite;
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  font-size: 0.78rem; color: #c3cede;
}
.qs-ticker:hover .qs-ticker-inner {animation-play-state: paused;}
.qs-ticker .m6 {color: #e08850; font-weight: 600;}
.qs-ticker .alrt {color: #ff6b61; font-weight: 700;}
.qs-ticker .tsu {color: #45b3e6; font-weight: 600;}
@keyframes qs-scroll {
  0% {transform: translateX(0);}
  100% {transform: translateX(-100%);}
}

/* Auto-scrolling news card rail */
.qs-newsrail {overflow: hidden; margin: 0.4rem 0 0.2rem 0;}
.qs-newsrail-inner {
  display: flex; gap: 12px; width: max-content;
  animation: qs-rail 70s linear infinite;
}
.qs-newsrail:hover .qs-newsrail-inner {animation-play-state: paused;}
@keyframes qs-rail {0% {transform: translateX(0);} 100% {transform: translateX(-50%);}}
.qs-newscard {
  flex: 0 0 250px; background: #161e2e; border: 1px solid #263145;
  border-radius: 8px; overflow: hidden; text-decoration: none !important;
  transition: border-color 0.15s;
}
.qs-newscard:hover {border-color: #e08850;}
.qs-newsimg {height: 118px; background-size: cover; background-position: center;
             background-color: #263145;}
.qs-newsmono {height: 118px; display: flex; align-items: center;
              justify-content: center; background: #263145;
              font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
              font-size: 36px; font-weight: 600; color: #e08850;}
.qs-newstxt {display: flex; flex-direction: column; gap: 4px; padding: 8px 10px 10px 10px;}
.qs-newssrc {font-size: 0.66rem; color: #8fa0b5; text-transform: uppercase;
             letter-spacing: 0.06em;}
.qs-newstitle {font-size: 0.8rem; color: #dbe2ec; line-height: 1.35;
               white-space: normal;}

/* Small screens: tighten spacing so phones get a clean layout */
@media (max-width: 640px) {
  .block-container {padding-left: 0.9rem; padding-right: 0.9rem;}
  .qs-wordmark {font-size: 1.2rem;}
  .qs-subline {font-size: 0.58rem; letter-spacing: 0.12em;}
  [data-testid="stMetric"] {padding: 8px 10px 6px 10px;}
  [data-testid="stMetricValue"] {font-size: 1.2rem !important;}
  h2, h3 {font-size: 1.15rem;}
  div[data-testid="stPopover"] {bottom: 0.9rem; right: 0.9rem;}
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers --
@st.cache_data(ttl=300, show_spinner="Contacting USGS feed...")
def get_live():
    return fetch_live()


@st.cache_data(show_spinner="Loading world towns database...")
def towns_db():
    """GeoNames towns database (generated by scripts/load_towns.py)."""
    import os
    path = os.path.join("data", "towns.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


@st.cache_data(ttl=3600, show_spinner=False)
def area_history(lat: float, lon: float):
    """50-year M5+ history within ~300 km, from BigQuery."""
    dlat, dlon = 2.75, 2.75 / max(0.2, math.cos(math.radians(lat)))
    df = run_bigquery(
        f"SELECT time, latitude, longitude, mag, place FROM {TABLE_FQN} "
        f"WHERE latitude BETWEEN {lat - dlat:.2f} AND {lat + dlat:.2f} "
        f"AND longitude BETWEEN {lon - dlon:.2f} AND {lon + dlon:.2f}")
    if df.empty:
        return df
    d = df.apply(lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1)
    return df[d <= 300].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def plate_boundaries():
    """Tectonic plate boundaries (Bird 2003), bundled locally for instant load."""
    import json
    path = os.path.join("data", "plate_boundaries.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        pass
    try:
        url = ("https://raw.githubusercontent.com/fraxen/tectonicplates/"
               "master/GeoJSON/PB2002_boundaries.json")
        return requests.get(url, timeout=15).json()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def logo_b64() -> str:
    with open(os.path.join("assets", "koda.png"), "rb") as f:
        return base64.b64encode(f.read()).decode()


AVATARS = {"user": "🧑", "assistant": os.path.join("assets", "gemini.png")}

BOT_INTRO = (
    "Hi, I'm **Gemini** ✦ — QuakeSense's AI assistant, powered by Google's "
    "Gemini 2.5 Flash on Vertex AI.\n\n"
    "I can help you with:\n"
    "- **What you're seeing** on this page — any event, number, or alert\n"
    "- **Any earthquake question**, in your own language\n"
    "- **Finding help**: nearest hospitals, fire & police stations and national "
    "emergency numbers are in the **Response Toolkit**\n\n"
    "What would you like to know?")


@st.cache_data(ttl=3600, show_spinner=False)
def places_search(query: str, lat: float, lon: float, n: int = 8):
    """Live place results (name, address, phone, open-now) from the Google
    Places API, biased around the affected town."""
    r = requests.post(
        "https://places.googleapis.com/v1/places:searchText",
        json={"textQuery": query,
              "locationBias": {"circle": {"center": {"latitude": lat,
                                                     "longitude": lon},
                               "radius": 40000.0}},
              "maxResultCount": n},
        headers={"X-Goog-Api-Key": MAPS_API_KEY,
                 "X-Goog-FieldMask":
                     "places.displayName,places.formattedAddress,"
                     "places.internationalPhoneNumber,places.location,"
                     "places.currentOpeningHours.openNow"},
        timeout=12)
    r.raise_for_status()
    out = []
    for p in r.json().get("places", []):
        loc = p.get("location", {})
        plat, plon = loc.get("latitude"), loc.get("longitude")
        if plat is None:
            continue
        oh = p.get("currentOpeningHours", {})
        out.append({"name": p.get("displayName", {}).get("text", "(unnamed)"),
                    "addr": p.get("formattedAddress", ""),
                    "phone": p.get("internationalPhoneNumber", ""),
                    "lat": plat, "lon": plon,
                    "open": oh.get("openNow"),
                    "km": haversine_km(lat, lon, plat, plon)})
    out.sort(key=lambda x: x["km"])
    return out


def google_places_section(trow, ev):
    """Google Maps-powered help finder: routes start from the USER's real
    device location (with permission) - or any typed starting point, like
    Grab - never from the epicenter. Shows nearest facilities with contacts,
    Gemini's recommendation, and an embedded route with ETA."""
    from urllib.parse import quote
    st.markdown("**Find help — powered by Google Maps**")

    lc, sc = st.columns([0.22, 0.78])
    with lc:
        st.caption("📍 Use my location")
        loc = None
        try:
            from streamlit_geolocation import streamlit_geolocation
            loc = streamlit_geolocation()
        except Exception:
            pass
    use_me = bool(loc and loc.get("latitude"))
    if use_me:
        lat, lon = float(loc["latitude"]), float(loc["longitude"])
        origin = f"{lat},{lon}"
        origin_label = "your current location"
        with sc:
            st.caption(f"✅ Using your device location — distances and routes "
                       f"start from you.")
    else:
        with sc:
            manual = st.text_input(
                "Starting point", value=f"{trow['name']}, {trow['country']}",
                key=f"gm_org_{trow['name']}",
                help="Tap the location button to use your device position, or "
                     "type any address or place — just like Google Maps.")
        lat, lon = float(trow["latitude"]), float(trow["longitude"])
        origin = (manual.strip() or f"{lat},{lon}") if manual else f"{lat},{lon}"
        origin_label = manual.strip() if manual and manual.strip() else trow["name"]

    k1, k2 = st.columns([0.4, 0.6])
    with k1:
        cat = st.selectbox("What do you need",
                           ["Hospitals", "Fire stations", "Police",
                            "Pharmacies", "Shelters", "Custom search"],
                           key=f"gm_cat_{trow['name']}")
    query = cat.lower()
    if cat == "Custom search":
        with k2:
            query = st.text_input(
                "Search like on Google Maps", value="emergency room",
                key=f"gm_q_{trow['name']}",
                help="Anything works: 'clinic open now', 'evacuation shelter', "
                     "a facility name...")

    if query.strip():
        try:
            places = places_search(query.strip(), lat, lon)
        except Exception as e:
            places = []
            st.warning(f"Google Places unavailable ({str(e)[:80]}).")
        if places:
            for p in places[:6]:
                open_tag = " · 🟢 open now" if p["open"] else (
                    " · 🔴 closed" if p["open"] is False else "")
                phone = f" · 📞 {p['phone']}" if p["phone"] else ""
                dir_url = (f"https://www.google.com/maps/dir/?api=1"
                           f"&destination={p['lat']},{p['lon']}")
                st.markdown(
                    f"**{p['name']}** — {p['km']:.1f} km{open_tag}{phone}  \n"
                    f"{p['addr']} · [🧭 Open in Google Maps]({dir_url})")
            if st.button("✦ Ask Gemini: where should I go first?",
                         key=f"gm_gem_{trow['name']}"):
                fac_lines = "\n".join(
                    f"{p['name']} | {p['addr']} | {p['phone'] or 'no phone'} | "
                    f"{p['km']:.1f} km | {'open' if p['open'] else 'unknown/closed'}"
                    for p in places[:6])
                ctx = (f"M{ev['mag']:.1f} earthquake near {ev['place']}; "
                       f"user is at {origin_label}"
                       if ev else f"user is at {origin_label}")
                with st.spinner("Gemini weighing the options..."):
                    st.markdown(prioritize_facilities(ctx, fac_lines))

            st.markdown("**Route & arrival time**")
            r1, r2 = st.columns([0.62, 0.38])
            with r1:
                dest_ix = st.selectbox(
                    "Destination", range(len(places[:6])),
                    format_func=lambda i: f"{places[i]['name']} "
                                          f"({places[i]['km']:.1f} km)",
                    key=f"gm_dest_{trow['name']}")
            with r2:
                mode = st.selectbox("Travel mode",
                                    ["driving", "walking", "bicycling"],
                                    key=f"gm_mode_{trow['name']}")
            dest = places[dest_ix]
            components.iframe(
                f"https://www.google.com/maps/embed/v1/directions"
                f"?key={MAPS_API_KEY}&origin={quote(origin)}"
                f"&destination={dest['lat']},{dest['lon']}&mode={mode}",
                height=380)
            st.caption("The map shows the route and estimated arrival time. "
                       "For live turn-by-turn navigation, use the "
                       "'Open in Google Maps' link on any result.")
        else:
            components.iframe(
                f"https://www.google.com/maps/embed/v1/search"
                f"?key={MAPS_API_KEY}&q={quote(query.strip())}"
                f"&center={lat},{lon}&zoom=12",
                height=380)


@st.cache_data(show_spinner=False)
def country_flag(name: str) -> str:
    """Emoji flag for a country name (GeoNames names mostly match pycountry)."""
    try:
        import pycountry
        c = pycountry.countries.lookup(name)
        return "".join(chr(0x1F1E6 + ord(ch) - 65) for ch in c.alpha_2)
    except Exception:
        return "🌐"


# Recognized global/regional outlets - ranked above unknown domains because
# content farms republish wire stories verbatim under many domains.
MAJOR_OUTLETS = (
    "reuters.com", "apnews.com", "bbc.co", "cnn.com", "theguardian.com",
    "aljazeera.com", "nytimes.com", "washingtonpost.com", "abc.net.au",
    "npr.org", "france24.com", "dw.com", "nhk.or.jp", "japantimes.co.jp",
    "cbsnews.com", "nbcnews.com", "abcnews.go.com", "news.sky.com",
    "usatoday.com", "latimes.com", "time.com", "straitstimes.com",
    "channelnewsasia.com", "scmp.com", "thehindu.com", "indianexpress.com",
    "bangkokpost.com", "irrawaddy.com", "rappler.com", "usgs.gov")


def _title_key(title: str) -> str:
    """Normalized fingerprint so the same wire story counts once."""
    return re.sub(r"[^a-z0-9]", "", title.lower())[:64]


# Realistic browser UA: bot filters on shared Cloud Run egress IPs reject
# bare python/appname agents.
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/126.0 Safari/537.36"}


def _fetch_gdelt_cards(n: int):
    r = requests.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={"query": "earthquake sourcelang:english", "mode": "ArtList",
                "format": "json", "maxrecords": 75, "sort": "DateDesc"},
        timeout=8, headers=BROWSER_UA)
    r.raise_for_status()
    body = r.content.decode("utf-8", errors="replace").strip()
    if not body.startswith("{"):  # GDELT rate-limit notices are plain text
        raise RuntimeError("GDELT rate limited")
    import json
    arts = json.loads(body).get("articles", [])
    # Major outlets first, keeping GDELT's newest-first order within each tier
    arts.sort(key=lambda a: not any(m in (a.get("domain") or "")
                                    for m in MAJOR_OUTLETS))
    now = datetime.now(timezone.utc)
    seen_dom, seen_title, out = set(), set(), []
    for a in arts:
        img = (a.get("socialimage") or "").strip()
        title = (a.get("title") or "").strip()
        url = (a.get("url") or "").strip()
        dom = (a.get("domain") or "").strip()
        if not (img.startswith("http") and title and url):
            continue
        tkey = _title_key(title)
        if dom in seen_dom or tkey in seen_title:
            continue
        seen_dom.add(dom)
        seen_title.add(tkey)
        ago = ""
        try:
            dt = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc)
            hrs = int((now - dt).total_seconds() // 3600)
            ago = f"{hrs}h ago" if hrs < 48 else f"{hrs // 24}d ago"
        except Exception:
            pass
        out.append({"title": title, "url": url, "img": img,
                    "source": dom, "ago": ago})
        if len(out) >= n:
            break
    return out


def _fetch_relief(n: int):
    """Latest earthquake situation reports/statements from ReliefWeb, the UN
    OCHA humanitarian information service (public RSS)."""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    r = requests.get("https://reliefweb.int/updates/rss.xml?search=earthquake",
                     timeout=6, headers=BROWSER_UA)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    now = datetime.now(timezone.utc)
    out, seen = [], set()
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        url = (it.findtext("link") or "").strip()
        if not title or not url:
            continue
        if any(ch in title for ch in "áéíóúñ¿"):  # skip Spanish duplicates
            continue
        tkey = _title_key(title)
        if tkey in seen:
            continue
        seen.add(tkey)
        ago = ""
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate") or "")
            hrs = int((now - dt).total_seconds() // 3600)
            ago = f"{hrs}h ago" if hrs < 48 else f"{hrs // 24}d ago"
        except Exception:
            pass
        out.append({"title": title, "url": url, "source": "ReliefWeb / UN OCHA",
                    "ago": ago})
        if len(out) >= n:
            break
    return out


def render_news_rail(cards):
    """Auto-scrolling card carousel, pure CSS. Cards without an article photo
    get a monogram tile so the rail stays a card view either way."""
    pieces = []
    for c in cards:
        meta = " · ".join(x for x in (c["source"], c["ago"]) if x)
        if c.get("img"):
            visual = (f'<div class="qs-newsimg" style="background-image:'
                      f'url(\'{html.escape(c["img"])}\')"></div>')
        else:
            initial = (c["source"][:1] or "•").upper()
            visual = f'<div class="qs-newsmono">{html.escape(initial)}</div>'
        pieces.append(
            f'<a class="qs-newscard" href="{html.escape(c["url"])}" '
            f'target="_blank" rel="noopener">{visual}'
            f'<div class="qs-newstxt"><span class="qs-newssrc">{html.escape(meta)}</span>'
            f'<span class="qs-newstitle">{html.escape(c["title"][:110])}</span></div></a>')
    row = "".join(pieces)
    st.markdown(f'<div class="qs-newsrail"><div class="qs-newsrail-inner">'
                f'{row}{row}</div></div>', unsafe_allow_html=True)


def _fetch_gnews(n: int):
    """Top earthquake headlines from global media, via the Google News RSS
    aggregator (carries Reuters, AP, BBC, CNN, etc.)."""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    r = requests.get(
        "https://news.google.com/rss/search?q=earthquake&hl=en-US&gl=US&ceid=US:en",
        timeout=6, headers=BROWSER_UA)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    items, seen = [], set()
    now = datetime.now(timezone.utc)
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        source = (it.findtext("source") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].strip()
        tkey = _title_key(title)
        if not title or not link or tkey in seen:
            continue
        seen.add(tkey)
        ago = ""
        try:
            pub = parsedate_to_datetime(it.findtext("pubDate") or "")
            hrs = int((now - pub).total_seconds() // 3600)
            ago = f"{hrs}h ago" if hrs < 48 else f"{hrs // 24}d ago"
        except Exception:
            pass
        items.append({"title": title, "link": link,
                      "source": source, "ago": ago})
        if len(items) >= n:
            break
    return items


@st.cache_data(ttl=1800, show_spinner=False)
def _photo_cards_cached():
    """Photo cards cache; raises on failure so misses are never cached."""
    out = _fetch_gdelt_cards(8)
    if not out:
        raise RuntimeError("no illustrated articles")
    return out


_PHOTO_BACKOFF = {"until": 0.0}


def photo_cards():
    """Photo cards with a short retry backoff - the auto-refreshing media
    section keeps retrying until photos arrive, then they stick for 30 min."""
    import time
    if time.time() < _PHOTO_BACKOFF["until"]:
        return []
    try:
        return _photo_cards_cached()
    except Exception:
        _PHOTO_BACKOFF["until"] = time.time() + 20
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def _text_feeds_bundle():
    """Headlines + UN reports fetched in parallel; never caches a total miss."""
    from concurrent.futures import ThreadPoolExecutor

    def safe(fn, *a):
        try:
            return fn(*a)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_heads = ex.submit(safe, _fetch_gnews, 10)
        f_reps = ex.submit(safe, _fetch_relief, 5)
        heads, reps = f_heads.result(), f_reps.result()
    if not (heads or reps):
        raise RuntimeError("feeds unavailable")
    return {"headlines": heads, "reports": reps}


_TEXT_BACKOFF = {"until": 0.0}


def text_feeds():
    import time
    if time.time() < _TEXT_BACKOFF["until"]:
        return {"headlines": [], "reports": []}
    try:
        return _text_feeds_bundle()
    except Exception:
        _TEXT_BACKOFF["until"] = time.time() + 60
        return {"headlines": [], "reports": []}


@st.fragment(run_every=30)
def media_section():
    """Auto-refreshing media block: shows headline cards instantly and swaps
    in the photo cards on its own as soon as they're available."""
    cards = photo_cards()
    feeds = text_feeds()
    if not cards:
        cards = [{"title": h["title"], "url": h["link"], "img": "",
                  "source": h["source"], "ago": h["ago"]}
                 for h in feeds["headlines"]]
    if cards:
        render_news_rail(cards)
    else:
        st.caption("Headlines unavailable right now — check back shortly.")

    reports = feeds["reports"]
    if reports:
        st.markdown("**Official statements & humanitarian updates**")
        st.caption("Situation reports and statements from UN agencies, IFRC "
                   "and governments — via ReliefWeb (UN OCHA).")
        for rep in reports:
            meta = " · ".join(x for x in (rep["source"], rep["ago"]) if x)
            st.markdown(f"- [{rep['title']}]({rep['url']})"
                        + (f" — *{meta}*" if meta else ""))


def render_ticker(live_df):
    """CNN-style scrolling strip of this week's significant events."""
    if live_df is None or live_df.empty:
        return
    top = live_df[live_df["mag"] >= 5.0].head(14)
    if top.empty:
        return
    now = pd.Timestamp.now(tz="UTC")
    bits = []
    for r in top.itertuples():
        hrs = int((now - r.time).total_seconds() // 3600)
        ago = f"{hrs}h ago" if hrs < 48 else f"{hrs // 24}d ago"
        if r.tsunami_flag:
            cls, tag = "tsu", " ⚠ tsunami flag"
        elif r.mag >= 6.5:
            cls, tag = "alrt", " ⚠ ALERT"
        elif r.mag >= 6:
            cls, tag = "m6", ""
        else:
            cls, tag = "", ""
        bits.append(f'<span class="{cls}">M{r.mag:.1f}</span> '
                    f'{html.escape(str(r.place))}{tag} · {ago}')
    items = " &nbsp;&nbsp;···&nbsp;&nbsp; ".join(bits)
    st.markdown(f'<div class="qs-ticker"><div class="qs-ticker-inner">'
                f'🛰️ LIVE · THIS WEEK M5+ &nbsp;&nbsp;···&nbsp;&nbsp; {items}'
                f'</div></div>', unsafe_allow_html=True)


@st.fragment
def quick_ask(context: str, live_df):
    """Floating messenger-style chat panel (like LinkedIn messaging).

    Keeps its own mini conversation; the on-screen context (which event /
    which location) travels with every question and is shown in the header,
    and the model is told to name the location it is talking about."""
    hist = st.session_state.setdefault("quick_chat", [])
    with st.popover("💬 Ask QuakeSense"):
        st.markdown("✦ **Gemini** — QuakeSense assistant · Gemini 2.5 Flash")
        st.caption(f"📍 Talking about: {context}")
        box = st.container(height=300)
        with box:
            if not hist:
                with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                    st.markdown(BOT_INTRO)
            for m in hist:
                with st.chat_message(m["role"], avatar=AVATARS.get(m["role"])):
                    st.markdown(m["content"])
                    if m.get("sources"):
                        st.caption("Sources: " + " · ".join(
                            f"[{s['title']}]({s['uri']})" for s in m["sources"][:3]))
        with st.form("quick_form", clear_on_submit=True, border=False):
            c1, c2 = st.columns([0.82, 0.18])
            q = c1.text_input("Message", label_visibility="collapsed",
                              placeholder="Type a message...")
            send = c2.form_submit_button("➤", use_container_width=True)
        if send and q.strip():
            q = q.strip()
            recent = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in hist[-4:])
            ctx_hist = (f"[Floating mini-chat. The user is looking at: {context}. "
                        f"Answer in under 120 words and ALWAYS name the specific "
                        f"location/event you are referring to, so there is no "
                        f"ambiguity.]\n{recent}")
            hist.append({"role": "user", "content": q})
            with box:
                with st.chat_message("user", avatar=AVATARS["user"]):
                    st.markdown(q)
                with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                    try:
                        with st.spinner("Thinking..."):
                            res = smart_ask(q, ctx_hist, live_df=live_df)
                        st.markdown(res["answer"])
                        srcs = res.get("sources") or []
                        if srcs:
                            st.caption("Sources: " + " · ".join(
                                f"[{s['title']}]({s['uri']})" for s in srcs[:3]))
                        hist.append({"role": "assistant", "content": res["answer"],
                                     "sources": srcs})
                    except Exception as e:
                        msg = (f"Unavailable right now ({str(e)[:60]}). "
                               f"Try the Ask AI page.")
                        st.markdown(msg)
                        hist.append({"role": "assistant", "content": msg,
                                     "sources": []})


@st.fragment
def briefing_block(ev: dict, pick: str):
    """Generate + display the community briefing; reruns alone, not the page."""
    if st.button("Generate community briefing", type="primary"):
        with st.spinner("Gemini drafting briefing..."):
            st.session_state.briefing = situation_briefing(ev)
            st.session_state.briefing_event = pick

    if st.session_state.get("briefing") and st.session_state.get("briefing_event") == pick:
        br = st.session_state.briefing
        st.success(f"### {br['headline']}")
        st.markdown(f"**What happened.** {br['what_happened']}")
        st.markdown(f"**Who is affected.** {br['who_is_affected']}")
        st.markdown("**Recommended actions.**")
        for act in br["recommended_actions"]:
            st.markdown(f"- {act}")
        st.info(br["caveats"])
        st.caption(f"Source: {br['source']} · underlying data: USGS · "
                   f"[official event page]({ev['url']})")
        txt = (f"{br['headline']}\n\nWHAT HAPPENED\n{br['what_happened']}\n\n"
               f"WHO IS AFFECTED\n{br['who_is_affected']}\n\nRECOMMENDED ACTIONS\n"
               + "\n".join(f"- {a}" for a in br["recommended_actions"])
               + f"\n\nNOTES\n{br['caveats']}\n\nGenerated by QuakeSense from USGS data.")
        st.download_button("Download briefing (.txt)", txt,
                           "quakesense_briefing.txt", "text/plain")


MODE_BADGE = {"data": "Answered from: USGS catalog (BigQuery)",
              "hybrid": "Answered from: USGS catalog + expert knowledge",
              "live": "Answered from: USGS live feed (past 7 days)",
              "general": "Answered from: expert knowledge (Gemini)"}


def _sources_line(srcs):
    return "Sources: " + " · ".join(
        f"[{s['title']}]({s['uri']})" for s in srcs[:5])


def _rate_answer(i: int, rating: str):
    m = st.session_state.chat[i]
    q = next((c["content"] for c in reversed(st.session_state.chat[:i])
              if c["role"] == "user"), "")
    log_feedback(q, m["content"], m.get("mode") or "", rating)
    m["rated"] = rating


@st.fragment
def chat_agent(live):
    """The full chat agent. As a fragment, every interaction (question,
    feedback click, clear) reruns only this section - the rest of the page,
    including the heavy map/table widgets, stays untouched."""
    if "chat" not in st.session_state:
        st.session_state.chat = []

    examples = [
        "How many M6+ earthquakes hit Myanmar since 1990?",
        "Why does Myanmar get so many big earthquakes?",
        "What should my family do during strong shaking?",
        "Strongest quake ever near Japan - and what made it so deadly?",
    ]
    pending = None
    if not st.session_state.chat:
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            st.markdown(BOT_INTRO)
        st.markdown("<p style='text-align:center;color:#8fa0b5;'>Try one of these:</p>",
                    unsafe_allow_html=True)
        r1 = st.columns(2)
        r2 = st.columns(2)
        for i, ex in enumerate(examples):
            col = (r1 + r2)[i]
            if col.button(ex, key=f"ex{i}", use_container_width=True):
                pending = ex

    for i, m in enumerate(st.session_state.chat):
        with st.chat_message(m["role"], avatar=AVATARS.get(m["role"])):
            st.markdown(m["content"])
            if m.get("mode"):
                st.caption(MODE_BADGE.get(m["mode"], ""))
            if m.get("sources"):
                st.caption(_sources_line(m["sources"]))
            if m.get("note"):
                st.caption(f":orange[{m['note']}]")
            if m.get("sql"):
                with st.expander("Generated SQL (explainable AI)"):
                    st.code(m["sql"], language="sql")
            if m.get("df") is not None and len(m["df"]) and m["df"].size > 1:
                st.dataframe(m["df"].head(30), use_container_width=True, hide_index=True)
            if m["role"] == "assistant":
                if m.get("rated"):
                    st.caption("Feedback recorded — thank you.")
                else:
                    fb1, fb2, _ = st.columns([0.07, 0.07, 0.86])
                    fb1.button("👍", key=f"fb_up_{i}", help="Good answer",
                               on_click=_rate_answer, args=(i, "up"))
                    fb2.button("👎", key=f"fb_down_{i}", help="Poor answer",
                               on_click=_rate_answer, args=(i, "down"))

    if st.session_state.get("area"):
        st.caption(f"The agent can see your current My Area analysis "
                   f"({st.session_state.area['city']}) — ask about it here.")

    typed = st.chat_input("Ask anything about earthquakes — events, science, safety, "
                          "or your area's analysis...")
    question = pending or typed
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(question)
        history = "\n".join(f"{m['role']}: {m['content'][:250]}"
                            for m in st.session_state.chat[-7:-1])
        if st.session_state.get("area"):
            ar = st.session_state.area
            history = (f"[Current 'My Area' analysis shown to user] Location: {ar['city']}. "
                       f"M5+ within 300 km since 1975: {ar['hist']['count']} "
                       f"(~{ar['hist']['per_decade']}/decade). Strongest: {ar['hist']['strongest']}. "
                       f"This week within 500 km: {ar['live']['count']}. "
                       f"Profile headline: {ar['prof']['headline']}\n" + history)
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            try:
                with st.spinner("Checking the USGS record..."):
                    res = smart_ask(question, history, stream=True, live_df=live)
                answer = st.write_stream(res["stream"])
                srcs = res.get("sources") or []
                if srcs:
                    st.caption(_sources_line(srcs))
                st.session_state.chat.append({"role": "assistant", "content": answer,
                                              "sql": res["sql"], "df": res["df"],
                                              "mode": res.get("mode"),
                                              "sources": srcs,
                                              "note": res.get("note", "")})
            except Exception as e:
                st.session_state.chat.append({
                    "role": "assistant",
                    "content": f"I could not answer that: {e}. Try rephrasing, or check that "
                               f"BigQuery and Vertex AI are reachable."})
        st.rerun(scope="fragment")

    if st.session_state.chat and st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun(scope="fragment")


@st.fragment
def my_area_block(tdb, live):
    """Country/town pickers + risk profile - reruns isolated from the page,
    so browsing the (12k-row) town list never re-renders anything else."""
    sel = None
    row = None
    lang = "English"
    if tdb is None:
        st.error("Towns database missing. Run once:  python scripts/load_towns.py")
    else:
        a0, a1, a2 = st.columns([1, 1.4, 1])
        with a0:
            countries = sorted(tdb["country"].dropna().unique().tolist())
            default_ix = countries.index("Thailand") if "Thailand" in countries else 0
            country = st.selectbox("Country", countries, index=default_ix,
                                   format_func=lambda c: f"{country_flag(c)} {c}")
        with a1:
            towns = tdb[tdb["country"] == country]
            labels2 = [f"{r.name_}, {r.admin1}" if pd.notna(r.admin1) and str(r.admin1) != ""
                       else str(r.name_)
                       for r in towns.rename(columns={"name": "name_"}).itertuples()]
            pick_ix = st.selectbox("Town (type to search the list)", range(len(labels2)),
                                   format_func=lambda i: labels2[i],
                                   help="Sorted by population - start typing to jump "
                                        "to your town. Exact coordinates come from the "
                                        "GeoNames database, no guessing.")
        with a2:
            lang = st.selectbox("Language", ["English", "Myanmar (Burmese)", "Thai",
                                             "Hindi", "Bengali", "Telugu",
                                             "Marathi", "Tamil"])
        row = towns.iloc[pick_ix]
        sel = f"{row['name']}, {country}"

    if st.button("Generate risk profile", type="primary", disabled=sel is None) and sel:
        lat, lon, display = float(row["latitude"]), float(row["longitude"]), sel
        try:
            with st.spinner("Reading 50 years of records for your area..."):
                hist_df = area_history(round(lat, 3), round(lon, 3))
            if hist_df.empty:
                hist = {"count": 0, "strongest": "none on record",
                        "latest": "none on record", "per_decade": 0}
            else:
                smax = hist_df.loc[hist_df["mag"].idxmax()]
                latest = hist_df.loc[pd.to_datetime(hist_df["time"]).idxmax()]
                years = max(1, datetime.now(timezone.utc).year - 1975)
                hist = {
                    "count": len(hist_df),
                    "strongest": f"M{smax['mag']:.1f} - {smax['place']} "
                                 f"({pd.to_datetime(smax['time']).year})",
                    "latest": f"M{latest['mag']:.1f} - {latest['place']} "
                              f"({pd.to_datetime(latest['time']).year})",
                    "per_decade": round(len(hist_df) / years * 10, 1),
                }
            near = live.copy() if not live.empty else pd.DataFrame()
            if not near.empty:
                dists = near.apply(lambda r: haversine_km(lat, lon, r["lat"], r["lon"]), axis=1)
                near = near[dists <= 500]
            live_near = {"count": len(near),
                         "max": f"M{near['mag'].max():.1f}" if len(near) else "none"}
            with st.spinner("Gemini writing your community profile..."):
                prof = area_profile(display.split(",")[0], hist, live_near, lang)
            st.session_state.area = {"prof": prof, "hist": hist, "live": live_near,
                                     "df": hist_df, "lat": lat, "lon": lon,
                                     "city": sel, "display": display}
        except Exception as e:
            st.error(f"Historical layer unavailable: {e}")

    if st.session_state.get("area") and st.session_state.area["city"] == sel:
        ar = st.session_state.area
        st.caption(f"Profile for: **{ar.get('display', sel)}** "
                   f"({ar['lat']:.3f}, {ar['lon']:.3f})")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("M5+ since 1975", ar["hist"]["count"],
                  help="All magnitude 5+ earthquakes within 300 km of your town "
                       "in the official USGS record since 1975.")
        m2.metric("Per decade", ar["hist"]["per_decade"],
                  help="Average number of M5+ events per 10 years within 300 km - "
                       "a rough measure of how seismically active your area is.")
        m3.metric("Strongest ever",
                  str(ar["hist"]["strongest"]).split(" - ")[0],
                  help=f"The most powerful event on record near you: {ar['hist']['strongest']}")
        m4.metric("Most recent M5+",
                  str(ar["hist"]["latest"]).rsplit("(", 1)[-1].rstrip(")"),
                  help=f"The last M5+ event within 300 km: {ar['hist']['latest']}")
        m5.metric("This week · 500 km", ar["live"]["count"],
                  help="Live count of M2.5+ events within 500 km in the past 7 days.")
        prof = ar["prof"]
        st.success(f"### {prof['headline']}")
        st.markdown(f"**Seismic history.** {prof['seismic_context']}")
        st.markdown(f"**Right now.** {prof['this_week']}")
        st.markdown("**Be prepared.**")
        for act in prof["preparedness_actions"]:
            st.markdown(f"- {act}")
        st.info(prof["caveats"])
        st.caption(f"Source: {prof['source']} · USGS catalog + live feed · not a prediction")
        if not ar["df"].empty:
            hd = ar["df"].copy()
            hd["year"] = pd.to_datetime(hd["time"]).dt.year
            hd["decade"] = (hd["year"] // 10 * 10).astype(str) + "s"
            hd["strength"] = pd.cut(hd["mag"], [5, 5.5, 6, 6.5, 7, 10],
                                    labels=["M5.0-5.4", "M5.5-5.9", "M6.0-6.4",
                                            "M6.5-6.9", "M7.0+"], right=False)
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Events per decade near you**")
                st.bar_chart(hd.groupby("decade").size(), color="#e08850")
                st.caption("Taller recent bars often reflect better instruments, "
                           "not necessarily more earthquakes.")
            with g2:
                st.markdown("**How strong they were**")
                st.bar_chart(hd.groupby("strength", observed=False).size(), color="#e08850")
                st.caption("Most events cluster at the lower magnitudes - "
                           "the big ones are rare but matter most.")
            with st.expander("Map: every M5+ epicenter within 300 km since 1975",
                             expanded=True):
                hist_map = ar["df"].rename(columns={"latitude": "lat", "longitude": "lon"})
                st.map(hist_map[["lat", "lon"]], zoom=5, color="#e08850", size=8000)


@st.fragment
def anomaly_explain_block(flagged, live_cells):
    """Region explainer - selectbox + AI analysis rerun without the page."""
    idx = st.selectbox(
        "Explain a flagged region",
        range(len(flagged)),
        format_func=lambda i: f"{flagged.iloc[i]['sample_place']}  "
                              f"({flagged.iloc[i]['ratio']:.0f}x normal)",
        help="Pick a region and generate a plain-language explanation of why "
             "its activity is unusual and what nearby communities should know.")
    if st.button("Generate AI analysis", type="primary"):
        cell = flagged.iloc[idx].to_dict()
        evs = live_cells[(live_cells["cell_lat"] == cell["cell_lat"]) &
                         (live_cells["cell_lon"] == cell["cell_lon"])]
        hist_context = ""
        try:
            hdf = run_bigquery(
                f"SELECT COUNT(*) AS n, MAX(mag) AS max_mag, "
                f"MIN(EXTRACT(YEAR FROM time)) AS since "
                f"FROM {TABLE_FQN} WHERE latitude BETWEEN {cell['cell_lat']} "
                f"AND {cell['cell_lat'] + 5} AND longitude BETWEEN "
                f"{cell['cell_lon']} AND {cell['cell_lon'] + 5}")
            r0 = hdf.iloc[0]
            hist_context = (f"{int(r0['n'])} M5+ events since {int(r0['since'])}, "
                            f"strongest ever M{r0['max_mag']:.1f}")
        except Exception:
            pass
        with st.spinner("Gemini analyzing pattern with 50-year context..."):
            st.session_state.anomaly_text = explain_anomaly(cell, evs, hist_context)
            st.session_state.anomaly_idx = idx
    if (st.session_state.get("anomaly_text")
            and st.session_state.get("anomaly_idx") == idx):
        cellm = flagged.iloc[idx]
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Events this week", int(cellm["current"]),
                  help="M4.5+ earthquakes recorded in this region in the past 7 days.")
        n2.metric("Normal week", f"{cellm['weekly_avg']:.2f}",
                  help="This region's average M4.5+ events per week over the last 50 years.")
        n3.metric("Times above normal", f"{cellm['ratio']:.0f}x",
                  help="This week divided by the 50-year weekly average. "
                       "3x or more gets flagged.")
        n4.metric("Strongest this week", f"M {cellm['max_mag']:.1f}",
                  help="The largest event in this region during the past 7 days.")
        st.info(st.session_state.anomaly_text)
        cell = flagged.iloc[idx].to_dict()
        evs = live_cells[(live_cells["cell_lat"] == cell["cell_lat"]) &
                         (live_cells["cell_lon"] == cell["cell_lon"])].copy()
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("**When they struck this week**")
            daily = evs.set_index(evs["time"].dt.floor("D")).groupby(level=0).size()
            daily.index = daily.index.strftime("%b %d")
            st.bar_chart(daily, color="#e08850")
            st.caption("A tight burst suggests an aftershock sequence; "
                       "spread-out days suggest a swarm.")
        with v2:
            st.markdown("**Where they struck**")
            st.map(evs[["lat", "lon"]], zoom=4, color="#e08850")


@st.fragment
def sitrep_block(ev: dict, pick_rt: str, live):
    """SITREP generation reruns alone - the toolkit page stays put."""
    if st.button("Generate SITREP", type="primary"):
        hist_ctx = ""
        try:
            hdf = run_bigquery(
                f"SELECT COUNT(*) AS n, MAX(mag) AS max_mag FROM {TABLE_FQN} "
                f"WHERE latitude BETWEEN {ev['lat'] - 2.7:.2f} AND {ev['lat'] + 2.7:.2f} "
                f"AND longitude BETWEEN {ev['lon'] - 2.8:.2f} AND {ev['lon'] + 2.8:.2f}")
            hist_ctx = (f"{int(hdf.iloc[0]['n'])} M5+ events since 1975, "
                        f"strongest ever M{hdf.iloc[0]['max_mag']:.1f}")
        except Exception:
            pass
        near_ct = int((live.apply(lambda r: haversine_km(ev["lat"], ev["lon"],
                                                         r["lat"], r["lon"]),
                                  axis=1) <= 500).sum())
        with st.spinner("Drafting situation report..."):
            st.session_state.sitrep = sitrep(ev, hist_ctx, near_ct)
            st.session_state.sitrep_event = pick_rt
    if (st.session_state.get("sitrep")
            and st.session_state.get("sitrep_event") == pick_rt):
        st.markdown(st.session_state.sitrep)
        st.download_button("Download SITREP (.txt)", st.session_state.sitrep,
                           "quakesense_sitrep.txt", "text/plain")


@st.fragment
def guidance_block(context: str):
    """Situation/language pickers + guidance, isolated from the page."""
    from src.ai import DD_SITUATIONS
    d1, d2 = st.columns([1.6, 1])
    with d1:
        dd_sit = st.selectbox("Your situation", list(DD_SITUATIONS.keys()),
                              key="rt_sit",
                              help="The advice changes completely depending on who "
                                   "you are and where you are right now.")
    with d2:
        dd_lang = st.selectbox("Language", ["English", "Myanmar (Burmese)", "Thai",
                                            "Hindi", "Bengali", "Telugu",
                                            "Marathi", "Tamil"], key="rt_lang")
    if st.button("Generate guidance", type="primary", key="rt_dd"):
        with st.spinner("Writing guidance for your situation..."):
            st.session_state.dd = do_dont(context, dd_lang, dd_sit)
            st.session_state.dd_key = (dd_lang, dd_sit)
    if st.session_state.get("dd") and st.session_state.get("dd_key") == (dd_lang, dd_sit):
        st.markdown(st.session_state.dd)
        st.download_button("Download guidance (.txt)", st.session_state.dd,
                           "quakesense_guidance.txt", "text/plain")


@st.fragment
def facilities_block(top, ev=None):
    """Town picker + facility finder (Google Maps when a key is configured,
    OpenStreetMap otherwise), isolated from the page."""
    lab3 = [f"{r['name']} ({r['country']}) — {r['km']:.0f} km from epicenter"
            for _, r in top.iterrows()]
    tix = st.selectbox("Affected-area town (nearest first)", range(len(lab3)),
                       format_func=lambda i: lab3[i], key="rt_town")
    trow = top.iloc[tix]
    country2 = trow["country"]
    if country2 in EMERGENCY_NUMBERS:
        st.markdown(f"**Emergency hotlines ({country2}):** "
                    f"{EMERGENCY_NUMBERS[country2]}")
        st.caption("From public sources - verify locally. "
                   "Numbers can differ by region.")

    if MAPS_API_KEY:
        google_places_section(trow, ev)
        return

    if st.button("Find hospitals, fire & police stations within 20 km"):
        try:
            with st.spinner(f"Searching OpenStreetMap around {trow['name']}..."):
                fac = emergency_facilities(round(float(trow["latitude"]), 3),
                                           round(float(trow["longitude"]), 3))
            if fac.empty:
                st.info("OpenStreetMap has no tagged facilities within 20 km of "
                        "this point. Local knowledge may know more.")
            else:
                f1, f2 = st.columns([1.2, 1])
                with f1:
                    st.dataframe(fac[["name", "type", "km away"]].head(25),
                                 use_container_width=True, hide_index=True)
                with f2:
                    st.map(fac[["lat", "lon"]], zoom=10, color="#6fae7f")
                st.caption(f"{len(fac)} facilities from OpenStreetMap (community-"
                           f"maintained - coverage varies by area).")
        except Exception as e:
            st.warning(f"Facility search unavailable right now ({str(e)[:60]}). "
                       f"Try again in a minute.")


@st.cache_data(ttl=86400, show_spinner=False)
def emergency_facilities(lat: float, lon: float, radius_km: int = 20):
    """Hospitals, fire and police stations near a point, from OpenStreetMap."""
    query = (f'[out:json][timeout:25];('
             f'node["amenity"~"hospital|fire_station|police"](around:{radius_km * 1000},{lat},{lon});'
             f'way["amenity"~"hospital|fire_station|police"](around:{radius_km * 1000},{lat},{lon});'
             f');out center 80;')
    headers = {"User-Agent": "QuakeSense/1.0 (hackathon demo; contact: team KODA)"}
    r = None
    for host in ["https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter"]:
        try:
            r = requests.post(host, data={"data": query}, headers=headers, timeout=30)
            r.raise_for_status()
            break
        except Exception:
            r = None
    if r is None:
        raise RuntimeError("all Overpass mirrors unavailable")
    rows = []
    for el in r.json().get("elements", []):
        tags = el.get("tags", {})
        plat = el.get("lat") or el.get("center", {}).get("lat")
        plon = el.get("lon") or el.get("center", {}).get("lon")
        if plat is None:
            continue
        rows.append({"name": tags.get("name", "(unnamed)"),
                     "type": tags.get("amenity", "").replace("_", " "),
                     "lat": plat, "lon": plon})
    df = pd.DataFrame(rows).drop_duplicates(subset=["name", "type"])
    if not df.empty:
        df["km away"] = df.apply(lambda r: round(haversine_km(lat, lon, r["lat"], r["lon"]), 1), axis=1)
        df = df.sort_values("km away").reset_index(drop=True)
    return df


# ----------------------------------------------------------------- header --
st.markdown("""
<div class="qs-header">
  <p class="qs-wordmark">QUAKE<span>SENSE</span></p>
  <p class="qs-subline"><span class="qs-live"></span>Live &nbsp;·&nbsp; Global real-time earthquake intelligence</p>
</div>
""", unsafe_allow_html=True)

try:
    live = get_live()
    feed_ok = True
except Exception as e:
    st.error(f"USGS live feed unreachable: {e}")
    live, feed_ok = pd.DataFrame(), False

render_ticker(live)

# ----------------------------------------------------------------- sidebar --
st.sidebar.markdown("##### MENU")
page = st.sidebar.radio(
    "Navigation",
    ["🛰️ Live Now", "📈 Anomaly Watch", "📍 My Area", "✦ Ask AI",
     "⛑️ Response Toolkit", "📖 Guide"],
    captions=["World map · briefings · media", "Is this week normal?",
              "Your town's risk profile", "Any question, any language",
              "SITREP · guidance · facilities", "How it all fits"],
    label_visibility="collapsed")
page = page.split(" ", 1)[1]

if st.sidebar.button("Refresh live feed", use_container_width=True):
    get_live.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Earthquakes cannot be predicted. This tool supports awareness "
                   "and decision-making, not prediction.")

_koda_b64 = logo_b64()
st.sidebar.markdown(f"""
<div class="qs-sidebar-bottom">
  <p class="qs-credit">Built with</p>
  <p class="qs-credit-items">Google Cloud &nbsp;·&nbsp; BigQuery<br>
  Vertex AI Gemini &nbsp;·&nbsp; Streamlit<br>
  USGS Earthquake Hazards Program</p>
  <img src="data:image/png;base64,{_koda_b64}" width="96">
  <p class="qs-team">Developed by Team KODA</p>
</div>
""", unsafe_allow_html=True)

# ================================================================ LIVE NOW ==
if page == "Live Now":
    if not feed_ok or live.empty:
        st.warning("No live data available right now.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        last24 = live[live["time"] > pd.Timestamp.now(tz="UTC") - pd.Timedelta("24h")]
        c1.metric("Events · 7 days · M2.5+", len(live),
                  help="Every earthquake above magnitude 2.5 recorded worldwide in the past 7 days (USGS live feed).")
        c2.metric("Last 24 hours", len(last24))
        c3.metric("Strongest this week", f"M {live['mag'].max():.1f}")
        c4.metric("M5+ events", int((live["mag"] >= 5).sum()),
                  help="Magnitude 5+ quakes can cause damage near the epicenter.")
        c5.metric("Tsunami-flagged", int(live["tsunami_flag"].sum()),
                  help="Events for which USGS raised a tsunami flag this week. Zero is good news.")
        newest = live.iloc[0]
        mins_ago = int((pd.Timestamp.now(tz="UTC") - newest["time"]).total_seconds() // 60)
        ago = f"{mins_ago} min ago" if mins_ago < 120 else f"{mins_ago // 60} h ago"
        st.caption(f"Most recent event: M{newest['mag']:.1f} — {newest['place']} · {ago}")

        tsu = live[live["tsunami_flag"] == 1]
        if len(tsu):
            worst = tsu.iloc[0]
            st.warning(f"Tsunami flag active this week: {len(tsu)} event(s), including "
                       f"M{worst['mag']:.1f} {worst['place']}. Coastal communities should "
                       f"follow official tsunami advisories.")

        st.write("")
        p1, p2 = st.columns([2, 1])
        with p1:
            preset = st.radio("Quick filters", ["Custom", "M4.5+", "M6+", "Last 24 h"],
                              horizontal=True,
                              help="One-tap views. 'Custom' uses the magnitude slider below.")
        with p2:
            regions = live["place"].str.split(",").str[-1].str.strip()
            region_opts = ["All countries / regions"] + sorted(regions.unique())
            q_region = st.selectbox("Filter by country / region (past 7 days)",
                                    region_opts,
                                    help="Only places with earthquakes this week appear "
                                         "here. For historical events, use Ask the Data.")

        s1, s3 = st.columns([3, 1])
        with s1:
            min_mag = st.slider("Minimum magnitude", 2.5, 8.0, 4.5, 0.1,
                                disabled=(preset != "Custom"))
        with s3:
            show_plates = st.toggle("Plate boundaries", value=False,
                                    help="Overlay tectonic plate boundaries (Bird 2003 dataset) - "
                                         "most earthquakes happen along these lines.")

        if preset == "M4.5+":
            view = live[live["mag"] >= 4.5].copy()
        elif preset == "M6+":
            view = live[live["mag"] >= 6.0].copy()
        elif preset == "Last 24 h":
            view = live[live["time"] > pd.Timestamp.now(tz="UTC") - pd.Timedelta("24h")].copy()
        else:
            view = live[live["mag"] >= min_mag].copy()
        if q_region != "All countries / regions":
            view = view[view["place"].str.split(",").str[-1].str.strip() == q_region]

        if view.empty:
            st.info("No events match this filter in the past 7 days. "
                    "Try a wider preset or 'All countries / regions'.")

        age_h = (pd.Timestamp.now(tz="UTC") - view["time"]).dt.total_seconds() / 3600
        view["alpha"] = (210 - age_h * 0.75).clip(80, 210).astype(int)
        view["color_r"] = (135 + view["mag"] * 14).clip(0, 230).astype(int)
        view["color_g"] = (140 - view["mag"] * 11).clip(45, 200).astype(int)
        view["color_b"] = 70
        is_tsu = view["tsunami_flag"] == 1
        view.loc[is_tsu, ["color_r", "color_g", "color_b"]] = [64, 170, 220]
        view["radius"] = 9000 + view["mag"] * 26000
        map_data = view[["lon", "lat", "color_r", "color_g", "color_b", "alpha",
                         "radius", "mag", "place", "depth_km"]].round(4)
        layers = []
        if show_plates:
            gj = plate_boundaries()
            if gj:
                layers.append(pdk.Layer(
                    "GeoJsonLayer", data=gj, stroked=True, filled=False,
                    get_line_color=[190, 70, 60, 110], line_width_min_pixels=1))
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=map_data,
            get_position=["lon", "lat"],
            get_fill_color="[color_r, color_g, color_b, alpha]",
            get_line_color="[215, 220, 226, 30]",
            stroked=True, line_width_min_pixels=1,
            get_radius="radius", pickable=True))
        st.pydeck_chart(pdk.Deck(
            map_style="dark",
            initial_view_state=pdk.ViewState(latitude=15, longitude=100, zoom=1.6),
            layers=layers,
            tooltip={"text": "M{mag} — {place}\nDepth {depth_km} km"}))

        cap, dl = st.columns([4, 1])
        cap.caption(f"{len(view)} events shown · size and color scale with magnitude · "
                    f"faded markers are older · teal markers carry a tsunami flag")
        dl.download_button("Export CSV",
                           view.drop(columns=["color_r", "color_g", "color_b",
                                              "alpha", "radius"]).to_csv(index=False),
                           "quakesense_events.csv", "text/csv", use_container_width=True)
        st.dataframe(view[["time", "mag", "place", "depth_km", "pager_alert",
                           "felt_reports", "tsunami_flag"]].head(50),
                     use_container_width=True, hide_index=True)

        # ---------------------------------------------- AI briefings section
        st.divider()
        st.subheader("AI Situation Briefings")
        st.caption("Pick any significant event this week - Gemini writes a calm, "
                   "plain-language community briefing from the USGS data.")
        sig = significant_events(live)
        labels = [f"M{r.mag:.1f}  ·  {r.place}  ·  {r.time:%b %d %H:%M} UTC"
                  for r in sig.itertuples()]
        pick = st.selectbox("Event (ranked by USGS significance)", labels,
                            key="sig_event",
                            help="Significance is USGS's newsworthiness score - it combines "
                                 "magnitude, felt reports, and estimated impact. Your pick "
                                 "carries over to the Response Toolkit.")
        ev = sig.iloc[labels.index(pick)].to_dict()

        a, b, c, d = st.columns(4)
        a.metric("Magnitude", f"M {ev['mag']:.1f}")
        b.metric("Depth", f"{ev['depth_km']:.0f} km",
                 help="Shallow quakes (under 70 km) shake the surface harder than deep ones of the same magnitude.")
        c.metric("Felt reports", ev["felt_reports"],
                 help="People who reported feeling this quake via USGS 'Did You Feel It?'.")
        pager_raw = ev.get("pager_alert")
        pager_ok = isinstance(pager_raw, str) and pager_raw.strip() != ""
        d.metric("PAGER alert", pager_raw.upper() if pager_ok else "N/A",
                 help="USGS impact estimate: GREEN minimal, YELLOW local, ORANGE regional, "
                      "RED major. 'N/A' means no assessment was issued.")
        if pager_ok:
            st.caption(PAGER_LABEL.get(pager_raw, ""))

        briefing_block(ev, pick)

        # ------------------------------------------- global media coverage
        st.divider()
        st.subheader("📺 Global media coverage")
        st.caption("Latest earthquake coverage from world media — click a "
                   "card to read the full story.")
        media_section()

        quick_ask(f"Live Now world map (past 7 days); selected event: {pick}", live)

# ================================================================= MY AREA ==
elif page == "My Area":
    st.subheader("My Area — community seismic risk profile")
    st.caption("Select your country and town - the agent combines your area's 50-year record "
               "with this week's live activity into a personal risk profile, in your language.")

    my_area_block(towns_db(), live)

    area_ctx = (f"My Area risk profile for {st.session_state.area['city']}"
                if st.session_state.get("area") else
                "My Area page (no town profiled yet)")
    quick_ask(area_ctx, live)

# ================================================================== ASK AI ==
elif page == "Ask AI":
    st.subheader("Ask about Earthquakes — AI agent")
    st.caption("Ask anything, in any language — it replies in yours. Historical numbers "
               "come from 50 years of USGS data (SQL shown), this week's events from the "
               "live feed, and current news with web sources cited. It remembers "
               "follow-ups and can discuss your My Area analysis (from the My Area page).")

    chat_agent(live)

# =========================================================== ANOMALY WATCH ==
elif page == "Anomaly Watch":
    st.subheader("Anomaly Watch — unusual seismic activity")
    st.caption("Compares this week's M4.5+ activity in every 5-degree region against the "
               "50-year historical baseline. Flags swarms and intense aftershock sequences.")
    if live.empty:
        st.info("Live feed unavailable.")
    else:
        flagged, live_cells = detect(live)
        if flagged.empty:
            st.success("No regions show anomalously elevated activity this week.")
        else:
            st.warning(f"{len(flagged)} region(s) flagged with unusually high activity")
            show = flagged.rename(columns={"sample_place": "region_sample"})
            st.dataframe(show[["cell_lat", "cell_lon", "current", "weekly_avg",
                               "ratio", "max_mag", "region_sample"]].round(2),
                         use_container_width=True, hide_index=True)
            st.caption("current = M4.5+ events this week in that region · weekly_avg = the region's "
                       "50-year average per week · ratio = current ÷ average (3x or more gets flagged)")
            anomaly_explain_block(flagged, live_cells)
        ctx = (f"{len(flagged)} regions flagged with unusual activity"
               if not flagged.empty else "no anomalous regions this week")
        quick_ask(f"Anomaly Watch page; {ctx}", live)

# ========================================================= RESPONSE TOOLKIT ==
elif page == "Response Toolkit":
    st.subheader("Response Toolkit")
    st.caption("Practical tools for the hours after an earthquake - for residents "
               "waiting for help, and for the officials coordinating it.")

    # ---- A: situation report -------------------------------------------
    st.markdown("##### Situation report (SITREP)")
    st.caption("A formal report in the format emergency operations centers use. "
               "Pick an event, generate, download, distribute.")
    if live.empty:
        st.info("Live feed unavailable.")
    else:
        sig = significant_events(live)
        labels_rt = [f"M{r.mag:.1f}  ·  {r.place}  ·  {r.time:%b %d %H:%M} UTC"
                     for r in sig.itertuples()]
        carry = st.session_state.get("sig_event")
        default_rt = labels_rt.index(carry) if carry in labels_rt else 0
        pick_rt = st.selectbox("Event (ranked by USGS significance)", labels_rt,
                               index=default_rt, key="rt_event",
                               help="Defaults to the event you picked on Live Now.")
        ev = sig.iloc[labels_rt.index(pick_rt)].to_dict()
        sitrep_block(ev, pick_rt, live)

    # ---- B: do's and don'ts --------------------------------------------
    st.divider()
    st.markdown("##### Before rescue arrives — do's and don'ts")
    st.caption("Established international guidance (FEMA / Red Cross), written for "
               "your situation and language. Not a substitute for trained rescuers.")
    context = (f"M{ev['mag']:.1f} earthquake near {ev['place']}, depth "
               f"{ev['depth_km']:.0f} km"
               if not live.empty else "a strong earthquake")
    guidance_block(context)

    # ---- C: emergency resources in the affected area --------------------
    st.divider()
    st.markdown("##### Emergency resources in the affected area")
    tdb2 = towns_db()
    if live.empty:
        st.info("Live feed unavailable.")
    elif tdb2 is None:
        st.error("Towns database missing. Run once:  python scripts/load_towns.py")
    else:
        st.caption(f"Towns near the selected event above ({ev['place']}), located from "
                   f"the event's actual USGS coordinates. Pick the affected town, then "
                   f"find its emergency facilities.")
        dlat = 1.5
        dlon = 1.5 / max(0.2, math.cos(math.radians(ev["lat"])))
        near_towns = tdb2[tdb2["latitude"].between(ev["lat"] - dlat, ev["lat"] + dlat)
                          & tdb2["longitude"].between(ev["lon"] - dlon, ev["lon"] + dlon)].copy()
        if not near_towns.empty:
            near_towns["km"] = near_towns.apply(
                lambda r: haversine_km(ev["lat"], ev["lon"],
                                       r["latitude"], r["longitude"]), axis=1)
            near_towns = near_towns[near_towns["km"] <= 150].sort_values("km")
        if near_towns.empty:
            st.info("No towns within 150 km of this epicenter - it is likely offshore "
                    "or in a remote area. Select a different event above.")
        else:
            facilities_block(near_towns.head(15).reset_index(drop=True), ev)

    quick_ask(f"Response Toolkit; selected event: {pick_rt}" if not live.empty
              else "Response Toolkit page", live)

# ============================================================== HOW TO USE ==
else:
    st.subheader("How to use QuakeSense")
    st.markdown("""
QuakeSense answers three questions after an earthquake: **what just happened,
what does it mean for my community, and is this pattern normal?** The menu is
organized around those moments:

| Section | Purpose | Data behind it |
|---|---|---|
| **Live Now** | What's happening right now, worldwide | USGS live feed (7 days, M2.5+), every 5 min |
| **Anomaly Watch** | Is this week normal for each region? | Live feed vs 50-year baseline |
| **My Area** | What's the risk where *I* live? | 50-year USGS catalog (BigQuery) |
| **Ask AI** | Any earthquake question, any language | Catalog + live feed + web search |
| **Response Toolkit** | The hours after a quake | All of the above + OpenStreetMap |

The scrolling strip under the header shows this week's M5+ events everywhere in
the app — orange for M6+, **red for M6.5+ alerts**, blue for tsunami-flagged.
On most pages a **💬 Ask QuakeSense** chat bubble floats bottom-right: quick
questions about what's on screen, answered with the exact event/location named.

#### Live Now
The world map of every earthquake in the last 7 days: magnitude presets and
slider, location filter, tectonic plate boundaries (red lines — that's where
quakes happen), tsunami auto-flagging, CSV export. Below the map: **AI Situation
Briefings** for any significant event, and **global media coverage** — top
earthquake headlines from world media.

#### Anomaly Watch
Compares this week's activity in every region against its 50-year average and
flags what's unusual — swarms, aftershock sequences — with calm AI explanations.

#### My Area
Pick your country and town from verified dropdowns, choose from 8 languages
(English, Burmese, Thai, Hindi, Bengali, Telugu, Marathi, Tamil), and get a
community risk profile grounded in your area's real 50-year record — with
charts and a map of every M5+ epicenter near you.

#### Ask AI
Ask anything about earthquakes, in any language — it answers in yours.
Historical questions are answered from 50 years of USGS records with the SQL
shown; this-week questions from the live feed; current events with live web
search, **sources cited**. Every answer takes a 👍/👎 so we keep improving.
It remembers follow-ups (*"and for Japan?"*) and can discuss your My Area profile.

#### Response Toolkit
For the hours after a quake: a formal **situation report (SITREP)** with
web-verified external reports, **do's and don'ts** for people waiting for
rescue (8 languages, FEMA/Red Cross guidance), and **hospitals, fire and
police stations** near any affected town, with national emergency hotlines.
The event you picked on Live Now carries over automatically.

---
*Earthquakes cannot be predicted. QuakeSense supports awareness, communication,
and preparedness decisions — never prediction.*
""")

st.divider()
st.caption("Global real-time earthquake intelligence · USGS live feed & FDSN catalog · "
           "Vertex AI Gemini · Google BigQuery")
