"""QuakeSense — AI Earthquake Situation Room for Communities.

Live USGS data + Vertex AI Gemini + BigQuery, served by Streamlit.

Run:  streamlit run app.py
"""
from datetime import datetime, timezone

import pandas as pd
import pydeck as pdk
import streamlit as st

from src.ai import situation_briefing, ask_the_data, explain_anomaly
from src.anomaly import detect
from src.live_feed import fetch_live, significant_events, PAGER_LABEL

st.set_page_config(page_title="QuakeSense | Seismic Decision Intelligence",
                   page_icon=":material/earthquake:", layout="wide")

# ------------------------------------------------------------------ style --
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.0rem; padding-bottom: 1.5rem;}

.qs-header {
  border-bottom: 1px solid #29313c;
  padding: 0 0 0.8rem 0; margin-bottom: 0.3rem;
}
.qs-wordmark {
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  font-size: 1.6rem; font-weight: 600; letter-spacing: 0.10em;
  color: #d7dce2; margin: 0;
}
.qs-wordmark span {color: #d97e4a;}
.qs-subline {
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  font-size: 0.70rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: #93a0af; margin-top: 0.25rem;
}
.qs-live {
  display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: #6fae7f; margin-right: 6px;
}

[data-testid="stMetric"] {
  background: #1a2029; border: 1px solid #29313c; border-radius: 6px;
  padding: 12px 14px 9px 14px;
}
[data-testid="stMetricLabel"] p {
  font-size: 0.67rem !important; letter-spacing: 0.12em;
  text-transform: uppercase; color: #93a0af !important;
}
[data-testid="stMetricValue"] {
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  font-variant-numeric: tabular-nums; font-size: 1.65rem !important;
}

h2, h3 {letter-spacing: 0.01em; color: #d7dce2;}
section[data-testid="stSidebar"] {border-right: 1px solid #29313c;}
section[data-testid="stSidebar"] .stRadio label p {font-size: 0.90rem;}

.qs-credit {
  font-size: 0.68rem; letter-spacing: 0.10em; text-transform: uppercase;
  color: #93a0af; margin-bottom: 0.15rem;
}
.qs-credit-items {font-size: 0.76rem; color: #b6c0cc; line-height: 1.5; margin-bottom: 0.55rem;}
.qs-sidebar-bottom {
  position: fixed; bottom: 0; left: 0.9rem; width: 15rem;
  padding: 0.7rem 0 1.0rem 0; background: #1a2029;
  border-top: 1px solid #29313c; z-index: 999;
}
.qs-team {font-size: 0.72rem; color: #93a0af; margin-top: 0.4rem;}

.stButton button[kind="primary"] {
  letter-spacing: 0.06em; font-weight: 600; border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner="Contacting USGS feed...")
def get_live():
    return fetch_live()


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

# ----------------------------------------------------------------- sidebar --
st.sidebar.markdown("##### OPERATIONS")
page = st.sidebar.radio(
    "Navigation",
    ["Live Monitor", "AI Briefings", "Ask the Data", "Anomaly Watch", "How to Use"],
    label_visibility="collapsed")

if st.sidebar.button("Refresh live feed", use_container_width=True):
    get_live.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Earthquakes cannot be predicted. This tool supports awareness "
                   "and decision-making, not prediction.")

import base64  # noqa: E402
with open("assets/koda.png", "rb") as _f:
    _koda_b64 = base64.b64encode(_f.read()).decode()
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

# ============================================================ LIVE MONITOR ==
if page == "Live Monitor":
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

        st.write("")
        f1, f2 = st.columns([2, 1])
        with f1:
            min_mag = st.slider("Minimum magnitude", 2.5, 8.0, 4.5, 0.1)
        with f2:
            q = st.text_input("Filter by location (past 7 days only)",
                              placeholder="e.g. Japan, Myanmar, Chile",
                              help="This page shows the live 7-day feed. For historical "
                                   "events, use Ask the Data.")

        view = live[live["mag"] >= min_mag].copy()
        if q.strip():
            view = view[view["place"].str.contains(q.strip(), case=False, na=False)]

        if view.empty:
            st.info(f"No events match — M{min_mag:.1f}+ "
                    f"{('near ' + chr(39) + q.strip() + chr(39) + ' ') if q.strip() else ''}"
                    f"in the past 7 days. Lower the magnitude or clear the filter.")

        view["color_r"] = (135 + view["mag"] * 14).clip(0, 230).astype(int)
        view["color_g"] = (140 - view["mag"] * 11).clip(45, 200).astype(int)
        view["radius"] = 9000 + view["mag"] * 26000
        map_data = view[["lon", "lat", "color_r", "color_g", "radius",
                         "mag", "place", "depth_km"]].round(4)
        st.pydeck_chart(pdk.Deck(
            map_style="dark",
            initial_view_state=pdk.ViewState(latitude=15, longitude=100, zoom=1.6),
            layers=[pdk.Layer(
                "ScatterplotLayer", data=map_data,
                get_position=["lon", "lat"],
                get_fill_color="[color_r, color_g, 70, 165]",
                get_line_color="[215, 220, 226, 30]",
                stroked=True, line_width_min_pixels=1,
                get_radius="radius", pickable=True)],
            tooltip={"text": "M{mag} — {place}\nDepth {depth_km} km"}))

        cap, dl = st.columns([4, 1])
        cap.caption(f"{len(view)} events shown · marker size and color scale with magnitude "
                    f"· feed refreshes every 5 minutes")
        dl.download_button("Export CSV",
                           view.drop(columns=["color_r", "color_g", "radius"]).to_csv(index=False),
                           "quakesense_events.csv", "text/csv", use_container_width=True)
        st.dataframe(view[["time", "mag", "place", "depth_km", "pager_alert",
                           "felt_reports", "tsunami_flag"]].head(50),
                     use_container_width=True, hide_index=True)

# ============================================================ AI BRIEFINGS ==
elif page == "AI Briefings":
    st.subheader("AI Situation Briefings")
    st.caption("Gemini converts raw USGS seismic data into calm, plain-language community briefings.")
    if live.empty:
        st.info("Live feed unavailable.")
    else:
        sig = significant_events(live)
        labels = [f"M{r.mag:.1f}  ·  {r.place}  ·  {r.time:%b %d %H:%M} UTC"
                  for r in sig.itertuples()]
        pick = st.selectbox("Event (ranked by USGS significance)", labels,
                            help="Significance is USGS's newsworthiness score - it combines "
                                 "magnitude, felt reports, and estimated impact. "
                                 "Top of the list = most important this week.")
        ev = sig.iloc[labels.index(pick)].to_dict()

        a, b, c, d = st.columns(4)
        a.metric("Magnitude", f"M {ev['mag']:.1f}")
        b.metric("Depth", f"{ev['depth_km']:.0f} km",
                 help="Shallow quakes (under 70 km) shake the surface harder than deep ones of the same magnitude.")
        c.metric("Felt reports", ev["felt_reports"],
                 help="People who reported feeling this quake via USGS 'Did You Feel It?'.")
        d.metric("PAGER alert", (ev.get("pager_alert") or "n/a").upper(),
                 help="USGS impact estimate: GREEN minimal, YELLOW local, ORANGE regional, "
                      "RED major. 'N/A' means no assessment was issued.")
        if ev.get("pager_alert"):
            st.caption(PAGER_LABEL.get(ev["pager_alert"], ""))

        st.write("")
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

# ============================================================ ASK THE DATA ==
elif page == "Ask the Data":
    st.subheader("Ask the Data — conversational analytics")
    st.caption("Ask anything about 50 years of global M5+ earthquakes (USGS catalog in BigQuery). "
               "Gemini writes the SQL, BigQuery executes it, Gemini explains the answer.")
    examples = [
        "How many M6+ earthquakes hit Myanmar since 1990?",
        "Which 10 countries had the most M7+ earthquakes since 2000?",
        "What was the strongest earthquake ever recorded near Japan?",
        "Show the yearly count of M6+ quakes in Southeast Asia for the last 15 years",
    ]
    ex = st.pills("Examples", examples) if hasattr(st, "pills") else st.selectbox(
        "Examples", [""] + examples)
    question = st.text_input("Your question", value=ex or "",
                             help="The archive covers 1975 to today, magnitude 5+ worldwide. "
                                  "Smaller quakes are not in this catalog. Ask in plain English - "
                                  "no SQL needed.")
    if st.button("Run analysis", type="primary") and question.strip():
        try:
            with st.spinner("Gemini → SQL → BigQuery → answer..."):
                st.session_state.ask_result = ask_the_data(question)
                st.session_state.ask_question = question
        except Exception as e:
            st.session_state.ask_result = None
            st.error(f"Could not answer: {e}")
            st.caption("Check that BigQuery is loaded (scripts/load_history.py) and "
                       "Vertex AI credentials are configured.")

    if st.session_state.get("ask_result"):
        res = st.session_state.ask_result
        st.markdown(f"#### {res['answer']}")
        with st.expander("Generated SQL (explainable AI)"):
            st.code(res["sql"], language="sql")
        if not res["df"].empty:
            st.dataframe(res["df"], use_container_width=True, hide_index=True)
            st.download_button("Export result (.csv)", res["df"].to_csv(index=False),
                               "quakesense_analysis.csv", "text/csv")
            num_cols = res["df"].select_dtypes("number").columns
            if len(res["df"]) > 1 and len(num_cols) >= 1 and len(res["df"].columns) >= 2:
                first = res["df"].columns[0]
                if first not in num_cols or len(res["df"].columns) == 2:
                    try:
                        st.bar_chart(res["df"].set_index(first)[
                            num_cols[0] if num_cols[0] != first else num_cols[-1]],
                            color="#d97e4a")
                    except Exception:
                        pass

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
                with st.spinner("Gemini analyzing pattern..."):
                    st.session_state.anomaly_text = explain_anomaly(cell, evs)
                    st.session_state.anomaly_idx = idx
            if (st.session_state.get("anomaly_text")
                    and st.session_state.get("anomaly_idx") == idx):
                st.info(st.session_state.anomaly_text)
                cell = flagged.iloc[idx].to_dict()
                evs = live_cells[(live_cells["cell_lat"] == cell["cell_lat"]) &
                                 (live_cells["cell_lon"] == cell["cell_lon"])]
                st.map(evs[["lat", "lon"]], zoom=4, color="#d97e4a")

# ============================================================== HOW TO USE ==
else:
    st.subheader("How to use QuakeSense")
    st.markdown("""
QuakeSense answers three questions after an earthquake: **what just happened,
what does it mean for my community, and is this pattern normal?**

It works with two datasets, and knowing which page uses which will save you confusion:

| | Data scope | Updated |
|---|---|---|
| **Live Monitor · AI Briefings · Anomaly Watch** | Past **7 days**, magnitude 2.5+ (USGS live feed) | Every 5 minutes |
| **Ask the Data** | Past **50 years**, magnitude 5+, worldwide (USGS catalog in BigQuery) | Historical archive |

#### Live Monitor
The world map of every earthquake in the last 7 days. Use the magnitude slider
(0.1 steps) and the location box (type *Japan*, press **Enter**) to narrow it down.
Export the filtered list as CSV. If the map goes empty, your filter is stricter
than this week's reality — lower it.

#### AI Briefings
Pick any significant event from this week and generate a plain-language community
briefing: what happened, who is affected, what to do. Written by Gemini strictly
from USGS data, with a download button for sharing.

#### Ask the Data
The history desk. Ask questions in plain English about 50 years of earthquakes —
*"List the largest earthquakes in Myanmar in 2025"*, *"Which countries had the most
M7+ quakes since 2000?"* Gemini writes SQL, BigQuery runs it, and you can inspect
the generated SQL and export results. **Looking for a past event? Ask here, not
in Live Monitor.**

#### Anomaly Watch
Compares this week's activity in every region against its 50-year average and
flags what's unusual — swarms, aftershock sequences. Generate an AI explanation
of any flagged region.

---
*Earthquakes cannot be predicted. QuakeSense supports awareness, communication,
and preparedness decisions — never prediction.*
""")

st.divider()
st.caption("Global real-time earthquake intelligence · USGS live feed & FDSN catalog · "
           "Vertex AI Gemini · Google BigQuery")
