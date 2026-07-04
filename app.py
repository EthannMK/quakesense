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

st.set_page_config(page_title="QuakeSense | AI Earthquake Situation Room",
                   page_icon="🌍", layout="wide")


@st.cache_data(ttl=300, show_spinner="Fetching live USGS feed...")
def get_live():
    return fetch_live()


st.title("🌍 QuakeSense — AI Earthquake Situation Room")
st.caption("Live USGS data · Vertex AI Gemini · BigQuery · built for communities, officials, and journalists")

try:
    live = get_live()
    feed_ok = True
except Exception as e:
    st.error(f"USGS live feed unreachable: {e}")
    live, feed_ok = pd.DataFrame(), False

page = st.sidebar.radio(
    "Navigation",
    ["🗺️ Live Monitor", "📰 AI Briefings", "💬 Ask the Data", "📈 Anomaly Watch"])

# ============================================================ LIVE MONITOR ==
if page == "🗺️ Live Monitor":
    if not feed_ok or live.empty:
        st.warning("No live data available right now.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        last24 = live[live["time"] > pd.Timestamp.now(tz="UTC") - pd.Timedelta("24h")]
        c1.metric("Quakes, last 7 days (M2.5+)", len(live))
        c2.metric("Last 24 hours", len(last24))
        c3.metric("Strongest this week", f"M{live['mag'].max():.1f}")
        c4.metric("M5+ events", int((live["mag"] >= 5).sum()))
        c5.metric("Tsunami-flagged", int(live["tsunami_flag"].sum()))

        min_mag = st.slider("Minimum magnitude", 2.5, 8.0, 4.5, 0.5)
        view = live[live["mag"] >= min_mag].copy()
        view["color_r"] = (view["mag"] * 36).clip(0, 255).astype(int)
        view["color_b"] = (255 - view["mag"] * 36).clip(0, 255).astype(int)
        view["radius"] = 10000 + view["mag"] * 28000
        map_data = view[["lon", "lat", "color_r", "color_b", "radius",
                         "mag", "place", "depth_km"]].round(4)
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(latitude=15, longitude=100, zoom=1.6),
            layers=[pdk.Layer(
                "ScatterplotLayer", data=map_data,
                get_position=["lon", "lat"],
                get_fill_color="[color_r, 80, color_b, 170]",
                get_radius="radius", pickable=True)],
            tooltip={"text": "M{mag} — {place}\nDepth {depth_km} km"}))
        st.caption(f"{len(view)} events shown · color/size scale with magnitude · data refreshes every 5 min")
        st.dataframe(view[["time", "mag", "place", "depth_km", "pager_alert",
                           "felt_reports", "tsunami_flag"]].head(50),
                     use_container_width=True, hide_index=True)

# ============================================================ AI BRIEFINGS ==
elif page == "📰 AI Briefings":
    st.subheader("📰 AI Situation Briefings")
    st.caption("Gemini converts raw USGS seismic data into calm, plain-language community briefings.")
    if live.empty:
        st.info("Live feed unavailable.")
    else:
        sig = significant_events(live)
        labels = [f"M{r.mag:.1f} — {r.place} — {r.time:%b %d %H:%M} UTC"
                  for r in sig.itertuples()]
        pick = st.selectbox("Select an event (ranked by USGS significance)", labels)
        ev = sig.iloc[labels.index(pick)].to_dict()

        a, b, c, d = st.columns(4)
        a.metric("Magnitude", f"M{ev['mag']:.1f}")
        b.metric("Depth", f"{ev['depth_km']:.0f} km")
        c.metric("Felt reports", ev["felt_reports"])
        d.metric("PAGER", (ev.get("pager_alert") or "n/a").upper())
        if ev.get("pager_alert"):
            st.caption(PAGER_LABEL.get(ev["pager_alert"], ""))

        if st.button("🧠 Generate community briefing", type="primary"):
            with st.spinner("Gemini drafting briefing..."):
                br = situation_briefing(ev)
            st.success(f"### {br['headline']}")
            st.markdown(f"**What happened:** {br['what_happened']}")
            st.markdown(f"**Who is affected:** {br['who_is_affected']}")
            st.markdown("**Recommended actions:**")
            for act in br["recommended_actions"]:
                st.markdown(f"- {act}")
            st.info(f"⚠️ {br['caveats']}")
            st.caption(f"source: {br['source']} · underlying data: USGS · [event page]({ev['url']})")

# ============================================================ ASK THE DATA ==
elif page == "💬 Ask the Data":
    st.subheader("💬 Ask the Data — conversational analytics")
    st.caption("Ask anything about 50 years of global M5+ earthquakes (USGS catalog in BigQuery). "
               "Gemini writes the SQL, BigQuery runs it, Gemini explains the answer.")
    examples = [
        "How many M6+ earthquakes hit Myanmar since 1990?",
        "Which 10 countries had the most M7+ earthquakes since 2000?",
        "What was the strongest earthquake ever recorded near Japan?",
        "Show the yearly count of M6+ quakes in Southeast Asia for the last 15 years",
    ]
    ex = st.pills("Try an example", examples) if hasattr(st, "pills") else st.selectbox(
        "Try an example", [""] + examples)
    question = st.text_input("Your question", value=ex or "")
    if st.button("🔎 Ask", type="primary") and question.strip():
        try:
            with st.spinner("Gemini → SQL → BigQuery → answer..."):
                res = ask_the_data(question)
            st.markdown(f"#### 💡 {res['answer']}")
            with st.expander("🔍 Show the SQL Gemini generated (explainable AI)"):
                st.code(res["sql"], language="sql")
            if not res["df"].empty:
                st.dataframe(res["df"], use_container_width=True, hide_index=True)
                num_cols = res["df"].select_dtypes("number").columns
                if len(res["df"]) > 1 and len(num_cols) >= 1 and len(res["df"].columns) >= 2:
                    first = res["df"].columns[0]
                    if first not in num_cols or len(res["df"].columns) == 2:
                        try:
                            st.bar_chart(res["df"].set_index(first)[num_cols[0] if num_cols[0] != first else num_cols[-1]])
                        except Exception:
                            pass
        except Exception as e:
            st.error(f"Could not answer: {e}")
            st.caption("Check that BigQuery is loaded (scripts/load_history.py) and "
                       "Vertex AI credentials are configured.")

# =========================================================== ANOMALY WATCH ==
else:
    st.subheader("📈 Anomaly Watch — unusual seismic activity")
    st.caption("Compares this week's M4.5+ activity in every 5° region against the 50-year "
               "historical baseline. Flags swarms and intense aftershock sequences.")
    if live.empty:
        st.info("Live feed unavailable.")
    else:
        flagged, live_cells = detect(live)
        if flagged.empty:
            st.success("✅ No regions show anomalously elevated activity this week.")
        else:
            st.warning(f"⚠️ {len(flagged)} region(s) flagged with unusually high activity")
            show = flagged.copy()
            show["region_sample"] = show["sample_place"]
            st.dataframe(show[["cell_lat", "cell_lon", "current", "weekly_avg",
                               "ratio", "max_mag", "region_sample"]].round(2),
                         use_container_width=True, hide_index=True)
            idx = st.selectbox(
                "Explain a flagged region",
                range(len(flagged)),
                format_func=lambda i: f"{flagged.iloc[i]['sample_place']} "
                                      f"({flagged.iloc[i]['ratio']:.0f}x normal)")
            if st.button("🧠 AI explanation", type="primary"):
                cell = flagged.iloc[idx].to_dict()
                evs = live_cells[(live_cells["cell_lat"] == cell["cell_lat"]) &
                                 (live_cells["cell_lon"] == cell["cell_lon"])]
                with st.spinner("Gemini analyzing pattern..."):
                    st.info(explain_anomaly(cell, evs))
                st.map(evs[["lat", "lon"]], zoom=4)

st.divider()
st.caption("Data: U.S. Geological Survey (USGS) real-time feeds & FDSN catalog · AI: Vertex AI Gemini · "
           "Analytics: Google BigQuery · Earthquakes cannot be predicted — this tool supports awareness "
           "and decision-making, not prediction.")
