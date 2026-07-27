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
                    log_feedback, prioritize_facilities, BOT_NAME, APP_FACTS,
                    transcribe_audio, text_to_speech)
from src.config import MAPS_API_KEY, GEMINI_MODEL
from src.anomaly import detect
from src.aftershock import official_forecast, observed_aftershocks, GUIDANCE
from src.weather import conditions as weather_conditions, advisory as weather_advisory
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

# UI chrome language - translates the app's own interface text (nav, headers,
# sidebar). Chosen for the countries with the heaviest earthquake exposure,
# English default.
UI_LANGUAGES = ["English", "Bahasa Indonesia", "日本語", "Filipino", "Türkçe",
                "नेपाली", "हिन्दी", "Español", "မြန်မာ", "ไทย"]

# AI-content language - the language an answer/profile is WRITTEN in on My
# Area / Respond. A different purpose from UI_LANGUAGES above (that one
# translates interface chrome, this one steers Gemini's output), but kept
# aligned to the same set of languages so the two pickers make sense together.
APP_LANGUAGES = UI_LANGUAGES

UI_STRINGS = {
    "nav_live":        {"English": "🛰️ Live", "Bahasa Indonesia": "🛰️ Langsung", "日本語": "🛰️ ライブ", "Filipino": "🛰️ Live", "Türkçe": "🛰️ Canlı", "नेपाली": "🛰️ प्रत्यक्ष", "हिन्दी": "🛰️ लाइव", "Español": "🛰️ En vivo", "မြန်မာ": "🛰️ တိုက်ရိုက်", "ไทย": "🛰️ สด"},
    "nav_my_area":     {"English": "📍 My Area", "Bahasa Indonesia": "📍 Wilayah Saya", "日本語": "📍 マイエリア", "Filipino": "📍 Aking Lugar", "Türkçe": "📍 Bölgem", "नेपाली": "📍 मेरो क्षेत्र", "हिन्दी": "📍 मेरा क्षेत्र", "Español": "📍 Mi Zona", "မြန်မာ": "📍 ကျွန်ုပ်၏ဒေသ", "ไทย": "📍 พื้นที่ของฉัน"},
    "nav_ask":         {"English": "✦ Ask", "Bahasa Indonesia": "✦ Tanya", "日本語": "✦ 質問", "Filipino": "✦ Magtanong", "Türkçe": "✦ Sor", "नेपाली": "✦ सोध्नुहोस्", "हिन्दी": "✦ पूछें", "Español": "✦ Preguntar", "မြန်မာ": "✦ မေးရန်", "ไทย": "✦ ถาม"},
    "nav_respond":     {"English": "⛑️ Respond", "Bahasa Indonesia": "⛑️ Respons", "日本語": "⛑️ 対応", "Filipino": "⛑️ Tumugon", "Türkçe": "⛑️ Müdahale", "नेपाली": "⛑️ प्रतिक्रिया", "हिन्दी": "⛑️ प्रतिक्रिया", "Español": "⛑️ Responder", "မြန်မာ": "⛑️ တုန့်ပြန်ရန်", "ไทย": "⛑️ ตอบสนอง"},
    "tagline":         {"English": "Live · Global real-time earthquake intelligence", "Bahasa Indonesia": "Langsung · Intelijen gempa global waktu-nyata", "日本語": "ライブ・世界のリアルタイム地震インテリジェンス", "Filipino": "Live · Pandaigdigang real-time na impormasyon sa lindol", "Türkçe": "Canlı · Küresel gerçek zamanlı deprem istihbaratı", "नेपाली": "प्रत्यक्ष · विश्वव्यापी वास्तविक-समय भूकम्प जानकारी", "हिन्दी": "लाइव · वैश्विक रीयल-टाइम भूकंप जानकारी", "Español": "En vivo · Inteligencia sísmica global en tiempo real", "မြန်မာ": "တိုက်ရိုက် · ကမ္ဘာလုံးဆိုင်ရာ အချိန်နှင့်တပြေးညီ ငလျင်သတင်းအချက်အလက်", "ไทย": "สด · ข้อมูลแผ่นดินไหวทั่วโลกแบบเรียลไทม์"},
    "sidebar_data":    {"English": "Data", "Bahasa Indonesia": "Data", "日本語": "データ", "Filipino": "Data", "Türkçe": "Veri", "नेपाली": "डाटा", "हिन्दी": "डेटा", "Español": "Datos", "မြန်မာ": "ဒေတာ", "ไทย": "ข้อมูล"},
    "sidebar_refresh": {"English": "Refresh live feed", "Bahasa Indonesia": "Segarkan umpan langsung", "日本語": "ライブフィードを更新", "Filipino": "I-refresh ang live feed", "Türkçe": "Canlı akışı yenile", "नेपाली": "प्रत्यक्ष फिड ताजा गर्नुहोस्", "हिन्दी": "लाइव फ़ीड रीफ़्रेश करें", "Español": "Actualizar feed en vivo", "မြန်မာ": "တိုက်ရိုက်ဖိဒ်ကို ပြန်လည်စတင်ပါ", "ไทย": "รีเฟรชฟีดสด"},
    "sidebar_prefs":   {"English": "Preferences", "Bahasa Indonesia": "Preferensi", "日本語": "設定", "Filipino": "Mga Kagustuhan", "Türkçe": "Tercihler", "नेपाली": "प्राथमिकताहरू", "हिन्दी": "प्राथमिकताएँ", "Español": "Preferencias", "မြန်မာ": "စိတ်ကြိုက်ရွေးချယ်မှုများ", "ไทย": "การตั้งค่า"},
    "sidebar_language":{"English": "Language", "Bahasa Indonesia": "Bahasa", "日本語": "言語", "Filipino": "Wika", "Türkçe": "Dil", "नेपाली": "भाषा", "हिन्दी": "भाषा", "Español": "Idioma", "မြန်မာ": "ဘာသာစကား", "ไทย": "ภาษา"},
    "sidebar_disclaimer": {"English": "Earthquakes cannot be predicted. This tool supports awareness and decision-making, not prediction.", "Bahasa Indonesia": "Gempa bumi tidak dapat diprediksi. Alat ini mendukung kesadaran dan pengambilan keputusan, bukan prediksi.", "日本語": "地震は予知できません。本ツールは予測ではなく、認識と意思決定を支援するものです。", "Filipino": "Hindi mahuhulaan ang mga lindol. Sinusuportahan ng tool na ito ang kamalayan at paggawa ng desisyon, hindi ang panghuhula.", "Türkçe": "Depremler tahmin edilemez. Bu araç tahmin değil, farkındalık ve karar almayı destekler.", "नेपाली": "भूकम्पको पूर्वानुमान गर्न सकिँदैन। यो उपकरणले पूर्वानुमान होइन, सचेतना र निर्णय लिने कार्यलाई समर्थन गर्छ।", "हिन्दी": "भूकंप की भविष्यवाणी नहीं की जा सकती। यह टूल जागरूकता और निर्णय लेने में मदद करता है, भविष्यवाणी में नहीं।", "Español": "Los terremotos no se pueden predecir. Esta herramienta apoya la conciencia y la toma de decisiones, no la predicción.", "မြန်မာ": "ငလျင်ကို ကြိုတင်ခန့်မှန်း၍မရပါ။ ဤကိရိယာသည် ခန့်မှန်းခြင်းမဟုတ်ဘဲ အသိပညာနှင့် ဆုံးဖြတ်ချက်ချမှုကို ပံ့ပိုးပေးသည်။", "ไทย": "ไม่สามารถพยากรณ์แผ่นดินไหวได้ เครื่องมือนี้ช่วยสร้างความตระหนักรู้และสนับสนุนการตัดสินใจ ไม่ใช่การพยากรณ์"},
    "sidebar_built_with": {"English": "Built with", "Bahasa Indonesia": "Dibangun dengan", "日本語": "使用技術", "Filipino": "Ginawa gamit ang", "Türkçe": "Şununla oluşturuldu", "नेपाली": "यसद्वारा निर्मित", "हिन्दी": "इनसे निर्मित", "Español": "Creado con", "မြန်မာ": "ဖြင့်တည်ဆောက်ထားသည်", "ไทย": "สร้างด้วย"},
    "sidebar_team":    {"English": "Developed by Team KODA", "Bahasa Indonesia": "Dikembangkan oleh Tim KODA", "日本語": "Team KODA 開発", "Filipino": "Binuo ng Team KODA", "Türkçe": "Team KODA tarafından geliştirildi", "नेपाली": "Team KODA द्वारा विकसित", "हिन्दी": "Team KODA द्वारा विकसित", "Español": "Desarrollado por Team KODA", "မြန်မာ": "Team KODA မှ ဖန်တီးသည်", "ไทย": "พัฒนาโดย Team KODA"},
    "live_subheader":  {"English": "AI Situation Briefings", "Bahasa Indonesia": "Ringkasan Situasi AI", "日本語": "AI状況ブリーフィング", "Filipino": "AI Situation Briefing", "Türkçe": "Yapay Zeka Durum Raporları", "नेपाली": "एआई अवस्था विवरण", "हिन्दी": "एआई स्थिति संक्षिप्त विवरण", "Español": "Informes de Situación con IA", "မြန်မာ": "AI အခြေအနေရှင်းလင်းချက်", "ไทย": "สรุปสถานการณ์โดย AI"},
    "myarea_subheader":{"English": "My Area — community seismic risk profile", "Bahasa Indonesia": "Wilayah Saya — profil risiko seismik komunitas", "日本語": "マイエリア — 地域の地震リスクプロファイル", "Filipino": "Aking Lugar — profile ng panganib sa lindol ng komunidad", "Türkçe": "Bölgem — toplum deprem riski profili", "नेपाली": "मेरो क्षेत्र — सामुदायिक भूकम्प जोखिम प्रोफाइल", "हिन्दी": "मेरा क्षेत्र — सामुदायिक भूकंप जोखिम प्रोफ़ाइल", "Español": "Mi Zona — perfil de riesgo sísmico comunitario", "မြန်မာ": "ကျွန်ုပ်၏ဒေသ — ရပ်ရွာငလျင်အန္တရာယ်ပရိုဖိုင်", "ไทย": "พื้นที่ของฉัน — ข้อมูลความเสี่ยงแผ่นดินไหวในชุมชน"},
    "ask_subheader":   {"English": "✦ Ask — full answers with the evidence", "Bahasa Indonesia": "✦ Tanya — jawaban lengkap dengan bukti", "日本語": "✦ 質問 — 根拠付きの完全な回答", "Filipino": "✦ Magtanong — buong sagot na may ebidensya", "Türkçe": "✦ Sor — kanıtlarla birlikte tam yanıtlar", "नेपाली": "✦ सोध्नुहोस् — प्रमाणसहितको पूर्ण जवाफ", "हिन्दी": "✦ पूछें — प्रमाण सहित पूर्ण उत्तर", "Español": "✦ Preguntar — respuestas completas con evidencia", "မြန်မာ": "✦ မေးရန် — အထောက်အထားနှင့်တကွ အပြည့်အစုံဖြေကြားချက်", "ไทย": "✦ ถาม — คำตอบฉบับเต็มพร้อมหลักฐาน"},
    "respond_subheader": {"English": "⛑️ Respond", "Bahasa Indonesia": "⛑️ Respons", "日本語": "⛑️ 対応", "Filipino": "⛑️ Tumugon", "Türkçe": "⛑️ Müdahale", "नेपाली": "⛑️ प्रतिक्रिया", "हिन्दी": "⛑️ प्रतिक्रिया", "Español": "⛑️ Responder", "မြန်မာ": "⛑️ တုန့်ပြန်ရန်", "ไทย": "⛑️ ตอบสนอง"},
    "footer":          {"English": "Global real-time earthquake intelligence · USGS live feed & FDSN catalog · Vertex AI Gemini · Google BigQuery", "Bahasa Indonesia": "Intelijen gempa global waktu-nyata · Umpan langsung USGS & katalog FDSN · Vertex AI Gemini · Google BigQuery", "日本語": "世界のリアルタイム地震インテリジェンス · USGSライブフィード & FDSNカタログ · Vertex AI Gemini · Google BigQuery", "Filipino": "Pandaigdigang real-time na impormasyon sa lindol · USGS live feed at FDSN catalog · Vertex AI Gemini · Google BigQuery", "Türkçe": "Küresel gerçek zamanlı deprem istihbaratı · USGS canlı akışı ve FDSN kataloğu · Vertex AI Gemini · Google BigQuery", "नेपाली": "विश्वव्यापी वास्तविक-समय भूकम्प जानकारी · USGS प्रत्यक्ष फिड र FDSN क्याटलग · Vertex AI Gemini · Google BigQuery", "हिन्दी": "वैश्विक रीयल-टाइम भूकंप जानकारी · USGS लाइव फ़ीड और FDSN कैटलॉग · Vertex AI Gemini · Google BigQuery", "Español": "Inteligencia sísmica global en tiempo real · Feed en vivo de USGS y catálogo FDSN · Vertex AI Gemini · Google BigQuery", "မြန်မာ": "ကမ္ဘာလုံးဆိုင်ရာ အချိန်နှင့်တပြေးညီ ငလျင်သတင်းအချက်အလက် · USGS တိုက်ရိုက်ဖိဒ်နှင့် FDSN စာရင်း · Vertex AI Gemini · Google BigQuery", "ไทย": "ข้อมูลแผ่นดินไหวทั่วโลกแบบเรียลไทม์ · ฟีดสด USGS และแคตตาล็อก FDSN · Vertex AI Gemini · Google BigQuery"},
}


def t(key: str) -> str:
    """UI chrome translation - separate from APP_LANGUAGES (which only sets
    the language an AI answer is WRITTEN in). Falls back to English if the
    current UI language or key is missing, so a partial translation never
    breaks the page."""
    lang = st.session_state.get("ui_language", "English")
    entry = UI_STRINGS.get(key, {})
    return entry.get(lang) or entry.get("English") or key

# Single dark palette - chosen and tuned for eye comfort in low light (the
# use case this app is actually built for: checking on an earthquake at
# night). A theme switcher was tried and pulled after it kept surfacing
# half-themed native Streamlit widgets across every page; one well-tuned
# dark palette is safer than three inconsistently-themed ones.
_pal = {
    "bg": "#0d1321", "bg2": "#0d1321", "panel": "#161e2e",
    "panel2": "#1a2333", "panel3": "#141b28",
    "deep1": "#141b28", "deep2": "#141b28", "deep3": "#141b28",
    "deep4": "#141b28", "deep5": "#141b28", "nav-active": "#1a2333",
    "border": "#263145", "border2": "#263145", "border3": "#34435c",
    "text": "#dbe2ec", "muted": "#8fa0b5", "muted2": "#8fa0b5",
    "accent": "#e08850", "accent-light": "#eda06b", "accent-dark": "#5c3a1f",
    "blue": "#45b3e6", "blue2": "#45b3e6", "blue3": "#45b3e6",
    "red": "#ff6b61", "green": "#6fae7f",
}

if "ui_language" not in st.session_state:
    st.session_state.ui_language = st.query_params.get("lang", "English")
    if st.session_state.ui_language not in UI_LANGUAGES:
        st.session_state.ui_language = "English"

st.markdown(
    "<style>:root {" +
    "".join(f"--qs-{k}: {v};" for k, v in _pal.items()) +
    "}</style>", unsafe_allow_html=True)

# Short haptic buzz on every button/tab/chip tap, app-wide - phones otherwise
# give zero physical confirmation that a tap registered, especially on
# anything that takes a moment to respond (GPS, AI calls). Silently does
# nothing on devices/browsers without the Vibration API (all of iOS Safari).
components.html("""
<script>
try {
  const doc = window.parent.document;
  if (!doc.__qsHapticsAttached) {
    doc.__qsHapticsAttached = true;
    doc.addEventListener('pointerdown', function (e) {
      const el = e.target.closest('button, [role="radio"], [role="tab"], a[data-testid^="stBaseLinkButton"]');
      if (el) { try { doc.defaultView.navigator.vibrate && doc.defaultView.navigator.vibrate(12); } catch (err) {} }
    }, true);
  }
} catch (e) {}
</script>""", height=0)

# ------------------------------------------------------------------ style --
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ---------------------------------------------------- theme canvas ----
   .streamlit/config.toml bakes in one static dark theme at startup - these
   rules repaint the base app canvas and native widget chrome to follow our
   --qs-* variables instead, so Dark/Light/Warm actually change the whole
   app, not just the custom cards defined further down this stylesheet. */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background-color: var(--qs-bg);
  color: var(--qs-text);
}
/* Streamlit's own header bar sits fixed across the full width at the very
   top with a huge z-index, invisible (transparent) but still solid for
   touch/click - once our sticky nav below scrolls up to meet it, it sat
   underneath that invisible bar and stopped responding to taps (mobile-only
   symptom, since desktop clicks land below the bar's short height anyway
   at most scroll positions). Letting clicks pass through it, then
   re-enabling them only on its own real controls (sidebar toggle), fixes
   that without touching anything else. */
[data-testid="stHeader"], [data-testid="stToolbar"] {
  background: transparent; pointer-events: none;
}
[data-testid="stHeader"] button, [data-testid="stHeader"] a,
[data-testid="stToolbar"] button, [data-testid="stToolbar"] a {
  pointer-events: auto;
}
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] {
  background-color: var(--qs-bg2) !important;
}
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] span,
label p, .stMarkdown p, h1, h2, h3, h4 {color: var(--qs-text);}
[data-testid="stCaptionContainer"] p {color: var(--qs-muted);}
[data-testid="stMetric"], [data-testid="stExpander"] details,
[data-testid="stExpander"] summary, [data-testid="stDataFrame"],
[data-baseweb="select"] > div:first-child, [data-baseweb="popover"],
[data-baseweb="menu"] {
  background-color: var(--qs-panel) !important; border-color: var(--qs-border) !important;
}
[data-baseweb="select"] *, [data-baseweb="menu"] *, [data-baseweb="popover"] *,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] p {
  color: var(--qs-text) !important;
}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-testid="stChatInput"] textarea, [data-testid="stNumberInput"] input {
  background-color: var(--qs-panel) !important; color: var(--qs-text) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div {background-color: var(--qs-border) !important;}
[data-testid="stSlider"] [role="slider"] {background-color: var(--qs-accent) !important;}
input[type="checkbox"], input[type="radio"] {accent-color: var(--qs-accent) !important;}
[data-testid="stCheckbox"] [data-baseweb="checkbox"] div:first-child,
[data-testid="stToggle"] [data-baseweb="checkbox"] div:first-child {
  background-color: var(--qs-panel) !important; border-color: var(--qs-border) !important;
}
[data-testid="stCheckbox"] [aria-checked="true"] div:first-child,
[data-testid="stToggle"] [aria-checked="true"] div:first-child {
  background-color: var(--qs-accent) !important; border-color: var(--qs-accent) !important;
}

/* ============ Enterprise design system ============ */
/* One professional typeface everywhere (flags-only font first - its
   unicode-range means it only ever renders flag glyphs) */
html, body, .stApp, [data-testid="stAppViewContainer"] *,
section[data-testid="stSidebar"] * {
  font-family: "Twemoji Country Flags", "Inter", "Segoe UI", system-ui, sans-serif;
}
/* Protect glyph fonts from the global override */
code, pre, kbd, code *, pre * {
  font-family: "Source Code Pro", Consolas, monospace !important;
}
span[class*="material-symbols"], [class*="material-icons"],
[data-testid="stIconMaterial"] {
  font-family: "Material Symbols Rounded" !important;
  font-weight: normal !important;
}
.qs-wordmark, .qs-subline, .qs-ticker-inner, [data-testid="stMetricValue"] {
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace !important;
}

/* Type scale */
h1, h2, h3 {font-weight: 600; letter-spacing: -0.01em;}
.block-container h2 {font-size: 1.3rem;}
.block-container h3 {font-size: 1.08rem;}
[data-testid="stCaptionContainer"], .stCaption {line-height: 1.45;}

/* Controls: uniform radius, weight, focus - every clickable button filled
   with the same accent orange as the floating chat launcher, instead of
   Streamlit's own static secondaryBackgroundColor from config.toml (which
   is where the inconsistent "some buttons show black" came from - not
   every native button type was covered by the old override, so whichever
   ones weren't kept Streamlit's default dark styling instead). */
.stButton button, [data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"],
[data-testid="stBaseButton-secondaryFormSubmit"],
a[data-testid="stBaseLinkButton-secondary"], a[data-testid="stBaseLinkButton-primary"],
[data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {
  border-radius: 8px; font-weight: 700;
  background-color: var(--qs-accent) !important;
  color: var(--qs-bg2) !important;
  border-color: var(--qs-accent) !important;
}
.stButton button:hover, [data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover,
[data-testid="stBaseButton-secondaryFormSubmit"]:hover,
a[data-testid="stBaseLinkButton-secondary"]:hover, a[data-testid="stBaseLinkButton-primary"]:hover,
[data-testid="stDownloadButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover {
  background-color: var(--qs-accent-light) !important;
  border-color: var(--qs-accent-light) !important;
}
[data-testid="stButtonGroup"] button {border-radius: 999px; font-weight: 600;}
hr {margin: 1.1rem 0 0.9rem 0;}

/* Chat: card-style message bubbles */
[data-testid="stChatMessage"] {
  background: var(--qs-panel3); border: 1px solid var(--qs-border2); border-radius: 12px;
  padding: 0.85rem 1rem; margin-bottom: 0.45rem;
}

/* Sidebar nav — modern menu with active accent, hover, no radio circles */
section[data-testid="stSidebar"] [role="radiogroup"] {gap: 3px;}
section[data-testid="stSidebar"] [role="radiogroup"] > label {
  padding: 9px 12px; border-radius: 10px; margin: 0;
  border-left: 3px solid transparent;
  transition: background 0.12s, border-color 0.12s; cursor: pointer;
}
section[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
  background: var(--qs-panel);
}
section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
  background: var(--qs-nav-active); border-left: 3px solid var(--qs-accent);
}
/* hide the round radio marker for a clean nav look */
section[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {
  display: none !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {
  color: var(--qs-accent);
}

/* Freeze the WHOLE top bar — wordmark, tagline, nav and live ticker — so the
   menu area is always visible and obviously navigation, not page content */
/* Sample prompt chips inside the chat popup */
.qs-sug-label {
  font-size: 0.6rem; letter-spacing: 0.13em; text-transform: uppercase;
  color: var(--qs-muted); margin: 0.6rem 0 0.3rem 0;
}
div[data-testid="stPopoverBody"] [class*="st-key-qx_"] button {
  background: var(--qs-deep1) !important; border: 1px solid var(--qs-border3) !important;
  color: var(--qs-blue3) !important; font-size: 0.76rem !important;
  font-weight: 500 !important; border-radius: 999px !important;
  padding: 4px 12px !important; min-height: 0 !important;
  text-align: left; line-height: 1.3;
}
div[data-testid="stPopoverBody"] [class*="st-key-qx_"] button:hover {
  border-color: var(--qs-blue) !important; color: var(--qs-text) !important;
  background: var(--qs-deep3) !important;
}

/* Sidebar toggle shows a gear instead of chevrons (Material Symbols is a
   ligature font, so hiding the original text and injecting "settings" as
   pseudo-content swaps the glyph) */
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"] {
  font-size: 0 !important; line-height: 1 !important;
}
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]::after,
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"]::after,
[data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"]::after {
  content: "settings";
  font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
  font-size: 21px !important; color: var(--qs-muted);
}
[data-testid="stSidebarCollapseButton"]:hover [data-testid="stIconMaterial"]::after,
[data-testid="stExpandSidebarButton"]:hover [data-testid="stIconMaterial"]::after,
[data-testid="stSidebarCollapsedControl"]:hover [data-testid="stIconMaterial"]::after {
  color: var(--qs-accent);
}

/* Sidebar as a compact settings panel — small enough to need no scrollbar,
   so the gear (collapse) control at the top stays reachable at a glance. */
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  gap: 0.35rem !important;
}
section[data-testid="stSidebar"] [data-testid="stElementContainer"] {
  margin-left: 0 !important; margin-right: 0 !important; margin-bottom: 0 !important;
  width: 100% !important;
}
/* margin-top reset excludes the last child, which gets margin-top:auto below
   to push "Built with" to the bottom of the sidebar. */
section[data-testid="stSidebar"] [data-testid="stElementContainer"]:not(:last-child) {
  margin-top: 0 !important;
}
section[data-testid="stSidebar"] .stButton {margin: 0 !important; width: 100% !important;}
section[data-testid="stSidebar"] .stButton button {width: 100% !important;}
section[data-testid="stSidebar"] .stButton button {
  padding: 0.3rem 0.8rem !important; min-height: 0 !important;
  font-size: 0.8rem !important;
}
.qs-set-sec {
  font-size: 0.58rem; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--qs-muted); margin: 1rem 0 0.7rem 0;
  border-top: 1px solid var(--qs-border); padding-top: 0.7rem;
}
.qs-set-sec:first-of-type {margin-top: 0.2rem; border-top: none; padding-top: 0;}

/* Answer rating buttons — blue thumb up, red thumb down (icon-only, compact) */
[class*="st-key-fb_up_"] button, [class*="st-key-fb_down_"] button {
  width: 40px !important; min-width: 40px !important; height: 34px;
  padding: 0 !important; border-radius: 9px !important;
  background: var(--qs-panel2) !important; border: 1px solid var(--qs-border) !important;
}
[class*="st-key-fb_up_"] button [data-testid="stIconMaterial"] {
  color: var(--qs-blue) !important; font-size: 19px !important;
}
[class*="st-key-fb_down_"] button [data-testid="stIconMaterial"] {
  color: var(--qs-red) !important; font-size: 19px !important;
}
[class*="st-key-fb_up_"] button:hover {border-color: var(--qs-blue) !important;}
[class*="st-key-fb_down_"] button:hover {border-color: var(--qs-red) !important;}

/* Freeze just the nav buttons at the top of the scroll area — nothing else
   moves, so the rest of the layout is untouched. */
.st-key-topnav {
  position: sticky; top: 0; z-index: 500;
  background: var(--qs-bg2); padding: 0.35rem 0 0.15rem 0;
}

/* Top navigation — a distinct segmented-control BAR (the app menu), so it
   reads as navigation, clearly separate from the page controls below it. */
.st-key-topnav [role="radiogroup"] {
  gap: 4px; flex-wrap: wrap; background: var(--qs-deep2);
  border: 1px solid var(--qs-border); border-radius: 12px;
  padding: 6px; margin-bottom: 16px;
}
.st-key-topnav [role="radiogroup"] > label {
  padding: 7px 18px; border-radius: 8px; margin: 0; cursor: pointer;
  border: none; background: transparent; transition: background 0.12s;
}
.st-key-topnav [role="radiogroup"] > label:hover {background: var(--qs-deep4);}
.st-key-topnav [role="radiogroup"] > label:has(input:checked) {
  background: var(--qs-accent);
}
.st-key-topnav [role="radiogroup"] > label:has(input:checked) p {
  color: var(--qs-bg2) !important;
}
.st-key-topnav [role="radiogroup"] > label > div:first-child {
  display: none !important;  /* hide the radio circle */
}
.st-key-topnav [role="radiogroup"] > label p {
  font-size: 0.9rem !important; font-weight: 600;
}

/* Chat composer — the mic sits inline next to the auto-growing chat input.
   The chat input already reads clearly as an input (rounded, with a send
   arrow), distinct from the rectangular suggestion buttons. */
.st-key-stt_ask, .st-key-quick_stt {
  display: flex !important; justify-content: center; align-items: flex-end;
}
/* Composer input — accent border so it clearly reads as the place to type,
   with breathing room above it separating it from the suggestion buttons */
.st-key-ask_chat_input, .st-key-quick_chat_input {margin-top: 1.4rem;}
.st-key-stt_ask, .st-key-quick_stt {margin-top: 1.4rem;}
.st-key-ask_chat_input [data-testid="stChatInput"],
.st-key-quick_chat_input [data-testid="stChatInput"] {
  border: 1.5px solid var(--qs-blue) !important; border-radius: 12px;
  background: var(--qs-bg) !important;
}
.st-key-ask_chat_input [data-testid="stChatInput"]:focus-within,
.st-key-quick_chat_input [data-testid="stChatInput"]:focus-within {
  border-color: var(--qs-accent) !important;
  box-shadow: 0 0 0 2px rgba(224, 136, 80, 0.18) !important;
}

/* Messenger panel send button - same accent orange as every other button */
div[data-testid="stPopoverBody"] [data-testid="stBaseButton-secondaryFormSubmit"] {
  background: var(--qs-accent) !important; color: var(--qs-bg2) !important;
  border-color: var(--qs-accent) !important;
  border-radius: 10px; font-weight: 700; font-size: 1.05rem;
}
div[data-testid="stPopoverBody"] [data-testid="stBaseButton-secondaryFormSubmit"]:hover {
  background: var(--qs-accent-light) !important; border-color: var(--qs-accent-light) !important;
}
/* ================================================== */

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
  border-bottom: 1px solid var(--qs-border);
  padding: 0 0 0.8rem 0; margin-bottom: 0.3rem;
}
.qs-wordmark {
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  font-size: 1.6rem; font-weight: 600; letter-spacing: 0.10em;
  color: var(--qs-text); margin: 0;
}
.qs-wordmark span {color: var(--qs-accent);}
.qs-subline {
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  font-size: 0.70rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--qs-muted); margin-top: 0.25rem;
}
.qs-live {
  display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: var(--qs-green); margin-right: 6px;
}

[data-testid="stMetric"] {
  background: var(--qs-panel); border: 1px solid var(--qs-border); border-radius: 6px;
  padding: 12px 14px 9px 14px;
}
[data-testid="stMetricLabel"] p {
  font-size: 0.67rem !important; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--qs-muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  font-variant-numeric: tabular-nums; font-size: 1.65rem !important;
}

h2, h3 {letter-spacing: 0.01em; color: var(--qs-text);}
section[data-testid="stSidebar"] {border-right: 1px solid var(--qs-border);}
section[data-testid="stSidebar"] .stRadio label p {font-size: 0.95rem; font-weight: 600;}

/* Narrower sidebar — but ONLY when expanded, so collapsing fully hides it
   (forcing the width unconditionally left a black strip when collapsed). */
section[data-testid="stSidebar"][aria-expanded="true"] {
  min-width: 15rem !important; max-width: 15rem !important;
}
/* Push "Built with / Team KODA" to the bottom of the sidebar instead of
   trailing right after whatever content happens to be above it. */
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  display: flex; flex-direction: column; min-height: 100%;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div {
  flex: 1; display: flex; flex-direction: column; min-height: 100%;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]
  > div > [data-testid="stVerticalBlock"] {
  flex: 1; display: flex; flex-direction: column;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]
  > div > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:last-child {
  margin-top: auto !important;
}
.qs-sidebar-bottom {
  margin-top: 0; padding: 0.6rem 0 0.1rem 0;
  border-top: 1px solid var(--qs-border);
}
.qs-sidebar-bottom p, .qs-sidebar-bottom img {margin: 0 !important;}
.qs-credit {
  font-size: 0.52rem !important; letter-spacing: 0.10em; text-transform: uppercase;
  color: var(--qs-muted) !important; margin-bottom: 0.15rem !important;
}
.qs-credit-items {
  font-size: 0.56rem !important; color: var(--qs-muted2) !important;
  line-height: 1.3; margin-bottom: 0.4rem !important;
}
.qs-sidebar-bottom img {display: block; margin: 0.2rem 0 !important;}
.qs-team {
  font-size: 0.54rem !important; color: var(--qs-muted) !important;
  margin-top: 0.25rem !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  font-size: 0.62rem !important; line-height: 1.3;
}

.stButton button[kind="primary"] {
  letter-spacing: 0.06em; font-weight: 600; border-radius: 4px;
}

/* Floating chat launcher — circular FAB, bottom-right */
div[data-testid="stPopover"] {
  position: fixed !important; bottom: 1.3rem; right: 1.3rem;
  left: auto !important; width: auto !important; z-index: 1000;
}
button[data-testid="stPopoverButton"] {
  width: 58px !important; min-width: 58px !important; height: 58px !important;
  border-radius: 50% !important; padding: 0 !important;
  font-size: 0 !important; color: transparent !important;
  background-color: var(--qs-accent) !important;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M5 4h14a1.6 1.6 0 0 1 1.6 1.6v8.8A1.6 1.6 0 0 1 19 16H9l-4 3.6V5.6A1.6 1.6 0 0 1 6.6 4z' fill='%23ffffff'/%3E%3Ccircle cx='9' cy='10' r='1.15' fill='%23e08850'/%3E%3Ccircle cx='12.5' cy='10' r='1.15' fill='%23e08850'/%3E%3Ccircle cx='16' cy='10' r='1.15' fill='%23e08850'/%3E%3C/svg%3E");
  background-repeat: no-repeat; background-position: center;
  background-size: 30px 30px;
  border: none !important; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
  transition: transform 0.12s, background-color 0.12s;
}
button[data-testid="stPopoverButton"]:hover {
  background-color: var(--qs-accent-light) !important; transform: scale(1.06);
}
button[data-testid="stPopoverButton"] * {
  display: none !important;  /* hide emoji label + chevron; the SVG bg is the icon */
}
/* Remove Streamlit's "Press Enter to submit form" hint inside the chat */
div[data-testid="stPopoverBody"] [data-testid="InputInstructions"] {
  display: none !important;
}

/* Chat panel — fits its content; only the conversation scrolls, never the panel */
div[data-testid="stPopoverBody"] {
  width: 384px !important; max-width: 92vw !important;
  max-height: 86vh; overflow: hidden !important;
  padding: 0.85rem !important;
  background-color: var(--qs-bg) !important; color: var(--qs-text) !important;
  border-color: var(--qs-border) !important;
}
/* Coloured header bar that bleeds to the panel edges */
.qs-chat-head {
  display: flex; align-items: center; gap: 10px;
  background: var(--qs-accent); margin: -0.85rem -0.85rem 0.6rem -0.85rem;
  padding: 0.7rem 0.9rem;
}
.qs-chat-av {
  width: 32px; height: 32px; border-radius: 50%; background: var(--qs-bg2);
  color: var(--qs-accent); display: flex; align-items: center; justify-content: center;
  font-size: 1rem; flex: 0 0 32px;
}
.qs-chat-title {font-weight: 700; font-size: 0.98rem; color: var(--qs-bg2); line-height: 1.1;}
.qs-chat-sub {font-size: 0.68rem; color: var(--qs-accent-dark); line-height: 1.2;}
/* Compact chat text inside the popup */
div[data-testid="stPopoverBody"] [data-testid="stChatMessage"] {
  padding: 0.5rem 0.55rem; margin-bottom: 0.3rem;
}
div[data-testid="stPopoverBody"] [data-testid="stChatMessage"] p,
div[data-testid="stPopoverBody"] [data-testid="stChatMessage"] li {
  font-size: 0.85rem !important; line-height: 1.45 !important;
}
div[data-testid="stPopoverBody"] [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"],
div[data-testid="stPopoverBody"] [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] {
  width: 26px; height: 26px;
}
[data-stale="true"] div[data-testid="stPopover"] {display: none !important;}

/* Live event ticker under the header */
.qs-ticker {
  overflow: hidden; white-space: nowrap; border: 1px solid var(--qs-border);
  border-radius: 6px; background: var(--qs-panel); padding: 0.35rem 0;
  margin: 0.35rem 0 0.75rem 0; position: relative;
}
.qs-ticker-inner {
  display: inline-block; padding-left: 100%;
  animation: qs-scroll 60s linear infinite;
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  font-size: 0.78rem; color: var(--qs-muted2);
}
.qs-ticker:hover .qs-ticker-inner {animation-play-state: paused;}
.qs-ticker .m6 {color: var(--qs-accent); font-weight: 600;}
.qs-ticker .alrt {color: var(--qs-red); font-weight: 700;}
.qs-ticker .tsu {color: var(--qs-blue); font-weight: 600;}
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
  flex: 0 0 250px; background: var(--qs-panel); border: 1px solid var(--qs-border);
  border-radius: 8px; overflow: hidden; text-decoration: none !important;
  transition: border-color 0.15s;
}
.qs-newscard:hover {border-color: var(--qs-accent);}
.qs-newsimg {height: 118px; background-size: cover; background-position: center;
             background-color: var(--qs-border);}
.qs-newsmono {height: 118px; display: flex; align-items: center;
              justify-content: center; background: var(--qs-border);
              font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
              font-size: 36px; font-weight: 600; color: var(--qs-accent);}
.qs-newstxt {display: flex; flex-direction: column; gap: 4px; padding: 8px 10px 10px 10px;}
.qs-newssrc {font-size: 0.66rem; color: var(--qs-muted); text-transform: uppercase;
             letter-spacing: 0.06em;}
.qs-newstitle {font-size: 0.8rem; color: var(--qs-text); line-height: 1.35;
               white-space: normal;}

/* Chat input: light-blue send icon */
[data-testid="stChatInput"] button {color: var(--qs-blue) !important;}
[data-testid="stChatInput"] button svg {fill: var(--qs-blue) !important;}
[data-testid="stChatInput"] button:hover {color: var(--qs-accent) !important;}
[data-testid="stChatInput"] button:hover svg {fill: var(--qs-accent) !important;}

/* GPS button: sized here; its internal styling is injected directly into
   the component (same-origin) by _style_gps_component() */
iframe[title="streamlit_geolocation.streamlit_geolocation"] {
  width: 48px !important; height: 48px !important;
}

/* Messenger panel: keep the input + send button side by side on every
   screen size (Streamlit stacks columns on phones, hiding the send) */
div[data-testid="stPopoverBody"] [data-testid="stHorizontalBlock"] {
  flex-direction: row !important; flex-wrap: nowrap !important;
}
div[data-testid="stPopoverBody"] [data-testid="stColumn"] {
  width: auto !important; min-width: 0 !important;
}

/* Step-by-step help finder */
.qs-fac {
  display: flex; gap: 12px; align-items: center; background: var(--qs-panel);
  border: 1px solid var(--qs-border); border-radius: 12px; padding: 10px 14px;
  margin-bottom: 8px; transition: border-color 0.15s;
}
.qs-fac:hover {border-color: var(--qs-accent);}
.qs-fac-ic {
  width: 42px; height: 42px; border-radius: 50%; background: var(--qs-border);
  display: flex; align-items: center; justify-content: center;
  font-size: 20px; flex: 0 0 42px;
}
.qs-fac-main {flex: 1; min-width: 0;}
.qs-fac-name {color: var(--qs-text); font-weight: 600; font-size: 0.92rem;}
.qs-fac-addr {color: var(--qs-muted); font-size: 0.78rem; margin-top: 1px;}
.qs-fac-actions {display: flex; gap: 14px; margin-top: 5px;}
.qs-fac-actions a {
  font-size: 0.8rem; color: var(--qs-blue); text-decoration: none; font-weight: 600;
}
.qs-fac-actions a:hover {color: var(--qs-accent);}
.qs-fac-meta {
  display: flex; flex-direction: column; align-items: flex-end; gap: 5px;
  flex: 0 0 auto;
}
.qs-km {
  background: var(--qs-border); color: var(--qs-accent); font-weight: 600;
  border-radius: 999px; padding: 2px 10px; font-size: 0.76rem;
  white-space: nowrap;
}
.qs-open {color: var(--qs-green); font-size: 0.72rem; white-space: nowrap;}
.qs-closed {color: var(--qs-red); font-size: 0.72rem; white-space: nowrap;}
.qs-trip {font-size: 0.85rem; color: var(--qs-text); line-height: 1.9;}
.qs-trip .dotA {color: var(--qs-green);} .qs-trip .dotB {color: var(--qs-red);}
.qs-trip .leg {color: var(--qs-muted); padding-left: 0.32rem;}
.qs-maphelp {
  display: flex; flex-direction: column; gap: 5px; margin-top: 10px;
  padding: 10px 12px; background: var(--qs-panel); border: 1px solid var(--qs-border);
  border-radius: 10px; font-size: 0.8rem; color: var(--qs-muted2); line-height: 1.4;
}
.qs-maphelp b {color: var(--qs-text);}
.qs-maphelp .dotA {color: var(--qs-green);} .qs-maphelp .dotB {color: var(--qs-red);}

/* Small screens: tighten spacing so phones get a clean layout */
@media (max-width: 640px) {
  .block-container {padding-left: 0.9rem; padding-right: 0.9rem;}
  .qs-wordmark {font-size: 1.2rem;}
  .qs-subline {font-size: 0.58rem; letter-spacing: 0.12em;}
  [data-testid="stMetric"] {padding: 8px 10px 6px 10px;}
  [data-testid="stMetricValue"] {font-size: 1.2rem !important;}
  h2, h3 {font-size: 1.1rem;}
  div[data-testid="stPopover"] {bottom: 0.9rem; right: 0.9rem;}
  /* Chat box fills the phone width; fits content, never overflows */
  div[data-testid="stPopoverBody"] {
    width: 94vw !important; max-width: 94vw !important;
    max-height: 88vh !important;
  }
  .stButton button {min-height: 44px;}
  [data-testid="stChatMessage"] {padding: 0.7rem 0.75rem;}
  .qs-fac {flex-wrap: wrap; padding: 10px 12px;}
  .qs-fac-meta {
    flex-direction: row; width: 100%; order: 3;
    justify-content: flex-start; gap: 10px; margin-top: 2px;
  }
  .qs-fac-actions a {padding: 4px 0; display: inline-block;}
  .qs-newscard {flex: 0 0 210px;}
  .qs-newsimg, .qs-newsmono {height: 100px;}
  iframe {max-width: 100%;}
  /* On phones, Streamlit's own header strip (holding the sidebar toggle)
     sits fixed across the very top of the screen. Our nav sticks to top:0
     too, so once scrolled it lands in that same band and visually collides
     with the sidebar icon living there. Sticking a little lower instead
     clears it - desktop doesn't need this, the two never overlap there. */
  .st-key-topnav {top: 3.75rem;}
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


@st.cache_data(show_spinner=False)
def asset_b64(name: str) -> str:
    """Base64 for an asset PNG, so it can be inlined in card HTML."""
    try:
        with open(os.path.join("assets", name), "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


AVATARS = {"user": os.path.join("assets", "user.png"),
           "assistant": os.path.join("assets", "gemini.png")}

# The small popup's old mic (streamlit_mic_recorder's built-in browser speech
# recognition, hard-coded to English) stays off - kept only in case it's
# useful again later. Flip to re-enable.
VOICE_INPUT_ENABLED = False    # popup mic (browser speech-to-text, English only)

# The full ✦ Ask page's voice pipeline: same 🎙️ mic icon, but records raw audio
# and transcribes it with Gemini (transcribe_audio) - understands any language,
# unlike the browser recognition above - then narrates the reply with Cloud
# Text-to-Speech (text_to_speech) in the language it was asked in.
ASK_VOICE_ENABLED = True

# Where the sidebar feedback form sends to. Never rendered as text in
# the UI - it only ever appears inside a mailto link's href.
FEEDBACK_TO = os.environ.get("FEEDBACK_EMAIL", "ethanmk2205@gmail.com")

# This popup ONLY explains the app itself - it deliberately does not answer
# earthquake content (that's the job of ✦ Ask, which has the catalog, live
# feed and web search). Two different chat surfaces, two different jobs.
QUICK_EXAMPLES = [
    "What does this page show me?",
    "How do I use My Area?",
    "What can this app do?",
]

POPUP_PROMPT = """You are """ + BOT_NAME + """, the small quick-help assistant that
appears on every QuakeSense page except ✦ Ask. Your ONLY job is to help people
understand and use the app itself. You do not answer questions about
earthquakes as a topic (data, science, safety, current events, risk,
aftershocks, weather) - a separate, more capable agent on the ✦ Ask page
handles all of that, with the full USGS catalog, live feed and web search.

""" + APP_FACTS + """

The user is currently on: {context}

Conversation so far:
{history}

Decide which kind of question this is:
- ABOUT THE APP (what a page/feature does, how to use something, general
  greetings/small talk) -> answer helpfully and specifically from the facts
  above, in the same language as the question, under 100 words.
- ABOUT EARTHQUAKES as a topic (a specific event, safety guidance, science,
  current news, risk, aftershocks, weather, or anything needing the catalog
  or live feed) -> do NOT answer the question itself. Write exactly one short,
  friendly sentence in the same language as the question, telling them the
  ✦ Ask page can give a full, verified answer. Then end your reply with the
  exact token [[GOTO_ASK]] on its own line, nothing after it.

Question: {question}"""


def popup_answer(question: str, context: str, history: str) -> dict:
    """Answers app-usage questions directly; flags earthquake-content
    questions so the UI can offer a one-click link to ✦ Ask. Deliberately
    self-contained (own Gemini call) rather than sharing smart_ask's routing,
    so the two chat surfaces stay cleanly separated."""
    from src.ai import _client, _config  # read-only reuse of the low-level client
    prompt = POPUP_PROMPT.format(context=context, history=history or "(none)",
                                 question=question)
    try:
        resp = _client().models.generate_content(
            model=GEMINI_MODEL, contents=prompt, config=_config(temperature=0.3))
        text = (resp.text or "").strip()
    except Exception as e:
        return {"answer": f"Unavailable right now ({str(e)[:60]}). "
                          f"Try the ✦ Ask page instead.",
                "goto_ask": True}
    goto = "[[GOTO_ASK]]" in text
    text = text.replace("[[GOTO_ASK]]", "").strip()
    if not text:
        text = "That sounds like a great question for our full earthquake agent."
        goto = True
    return {"answer": text, "goto_ask": goto}


BOT_INTRO = (
    "Hi, I'm **Terra** ✦.\n\n"
    "Ask me about **this app** — what a page does, how to use a feature, "
    "anything about QuakeSense itself. For earthquake questions (data, "
    "safety, science), I'll point you to **✦ Ask**, our full research agent.\n\n"
    "What would you like to know?")


@st.cache_data(ttl=3600, show_spinner=False)
@st.cache_data(ttl=600, show_spinner=False)
def place_suggestions(text: str, lat: float, lon: float, n: int = 4):
    """Google-Maps-style place hints (Places Autocomplete)."""
    if not text or len(text.strip()) < 3:
        return []
    try:
        r = requests.post(
            "https://places.googleapis.com/v1/places:autocomplete",
            json={"input": text.strip(),
                  "locationBias": {"circle": {"center": {"latitude": lat,
                                                         "longitude": lon},
                                   "radius": 50000.0}}},
            headers={"X-Goog-Api-Key": MAPS_API_KEY,
                     "Content-Type": "application/json"}, timeout=8)
        r.raise_for_status()
        out = []
        for s in r.json().get("suggestions", [])[:n]:
            t = s.get("placePrediction", {}).get("text", {}).get("text")
            if t:
                out.append(t)
        return out
    except Exception:
        return []


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
    if not r.ok:
        # Google's actual reason (bad key, API not enabled, billing, quota...)
        # lives in the response body, not in requests' generic HTTPError text.
        try:
            detail = r.json().get("error", {}).get("message", r.text[:200])
        except Exception:
            detail = r.text[:200]
        raise RuntimeError(f"{r.status_code}: {detail}")
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


OFFLINE_DOCS = [
    {"icon": "🧎", "title": "Drop, Cover, Hold On — visual poster",
     "src": "FEMA · publication P-529", "size": "0.3 MB",
     "url": "https://www.shakeout.org/downloads/fema-529.pdf"},
    {"icon": "📋", "title": "Earthquake Preparedness Checklist",
     "src": "American Red Cross", "size": "PDF",
     "url": "https://www.redcross.org/content/dam/redcross/get-help/pdfs/"
            "earthquake/EN_Earthquake-Preparedness-Checklist.pdf"},
    {"icon": "✅", "title": "Earthquake Safety Checklist",
     "src": "American Red Cross", "size": "PDF",
     "url": "https://www.redcross.org/content/dam/redcross/lp/"
            "hfc-reporting-portal/Earthquake-Safety-Checklist.pdf"},
    {"icon": "🚨", "title": "Earthquake Hazard Information Sheet",
     "src": "Ready.gov · U.S. Dept. of Homeland Security", "size": "PDF",
     "url": "https://www.ready.gov/sites/default/files/2024-03/"
            "ready.gov_earthquake_hazard-info-sheet.pdf"},
    {"icon": "🪜", "title": "Seven Steps to Earthquake Safety",
     "src": "Earthquake Country Alliance · SCEC", "size": "0.3 MB",
     "url": "https://www.earthquakecountry.org/library/"
            "ShakeOut_Recommended_Earthquake_Safety_Actions.pdf"},
    {"icon": "📖", "title": "Putting Down Roots in Earthquake Country",
     "src": "USGS · U.S. Geological Survey", "size": "5.5 MB",
     "url": "https://pubs.usgs.gov/gip/2005/15/gip-15.pdf"},
]


def offline_library():
    """Curated official PDFs (visual guides: protection, first aid,
    preparedness) - meant to be downloaded BEFORE disaster strikes, when
    networks still work, then read offline. Two-column responsive grid."""
    st.markdown("##### 📥 Offline library")
    st.caption("Official safety guides from the American Red Cross, FEMA, "
               "USGS, Ready.gov and the Earthquake Country Alliance. "
               "Download for offline use.")
    rows = []
    for d in OFFLINE_DOCS:
        rows.append(
            f'<a class="qs-fac" style="text-decoration:none;margin-bottom:0" '
            f'href="{d["url"]}" target="_blank" rel="noopener">'
            f'<span class="qs-fac-ic">{d["icon"]}</span>'
            f'<span class="qs-fac-main" style="display:block">'
            f'<span class="qs-fac-name" style="display:block">'
            f'{html.escape(d["title"])}</span>'
            f'<span class="qs-fac-addr" style="display:block">'
            f'{html.escape(d["src"])} · {d["size"]}</span></span>'
            f'<span class="qs-km">⬇ PDF</span></a>')
    st.markdown('<div style="display:grid;grid-template-columns:repeat('
                'auto-fit,minmax(320px,1fr));gap:10px">'
                + "".join(rows) + '</div>', unsafe_allow_html=True)


CAT_ICONS = {"Hospitals": "🏥", "Fire stations": "🚒", "Police": "👮",
             "Pharmacies": "💊", "Shelters": "🏠", "Custom search": "🔍"}

# categories with custom artwork used on the result cards
CAT_IMAGES = {"Shelters": "shelter.png"}


def cat_icon_html(cat_name: str, fallback: str) -> str:
    """Card icon: inline PNG artwork when the category has one, else emoji."""
    img = CAT_IMAGES.get(cat_name)
    if img:
        b64 = asset_b64(img)
        if b64:
            return (f'<img src="data:image/png;base64,{b64}" '
                    f'style="width:26px;height:26px;border-radius:7px">')
    return fallback


def _chip_pick(label, options, key, default=None):
    """Compact chip picker (st.pills), selectbox fallback on old Streamlit."""
    if hasattr(st, "pills"):
        val = st.pills(label, options, key=key,
                       default=default or options[0])
        return val or default or options[0]
    return st.selectbox(label, options, key=key)


def _style_mic_component():
    """Give the mic button the same accent border as the chat input. The
    component iframe is served by our own app (same origin), so a zero-height
    helper frame can inject styling into its document - reading the current
    theme's colors straight off the parent document so it follows Dark/Light/
    Warm without needing its own copy of the palette."""
    components.html("""
<script>
function qsVar(name, fallback) {
  try {
    const v = getComputedStyle(window.parent.document.documentElement)
      .getPropertyValue(name).trim();
    return v || fallback;
  } catch (e) { return fallback; }
}
function injectMic() {
  try {
    const bg = qsVar('--qs-bg', '#0b1220');
    const blue = qsVar('--qs-blue', '#45b3e6');
    const accent = qsVar('--qs-accent', '#e08850');
    const panel2 = qsVar('--qs-panel2', '#131c2c');
    const micCss = `
      html, body {margin:0; padding:0; background:transparent; overflow:hidden;
                  display:flex; align-items:center; justify-content:center;}
      html body button, html body button:not([disabled]), button {
        width: 44px !important; height: 42px !important;
        background-color: ${bg} !important;
        border: 1.5px solid ${blue} !important;
        border-color: ${blue} !important; border-style: solid !important;
        border-width: 1.5px !important;
        border-radius: 12px !important; cursor: pointer !important;
        box-shadow: none !important; outline: none !important;
        display: flex !important; align-items: center !important;
        justify-content: center !important; padding: 0 !important;
        font-size: 17px !important; line-height: 1 !important;
        transition: border-color 0.15s, background-color 0.15s !important;
      }
      html body button:hover {
        border-color: ${accent} !important; background-color: ${panel2} !important;
      }
    `;
    const frames = window.parent.document.querySelectorAll('iframe');
    for (const f of frames) {
      const t = f.getAttribute('title') || '';
      if (!/mic_recorder|speech_to_text/i.test(t)) continue;
      const d = f.contentDocument;
      if (d && d.head) {
        let s = d.getElementById('qs-mic-style');
        if (!s) {
          s = d.createElement('style'); s.id = 'qs-mic-style'; d.head.appendChild(s);
        }
        s.textContent = micCss;
      }
      // the component sets inline styles that beat stylesheet rules, so write
      // the accent border straight onto the element with top priority
      const btn = d ? d.querySelector('button') : null;
      if (btn) {
        btn.style.setProperty('border', '1.5px solid ' + blue, 'important');
        btn.style.setProperty('background-color', bg, 'important');
        btn.style.setProperty('border-radius', '12px', 'important');
        btn.style.setProperty('width', '44px', 'important');
        btn.style.setProperty('height', '42px', 'important');
        btn.style.setProperty('box-shadow', 'none', 'important');
        btn.onmouseenter = function(){
          btn.style.setProperty('border-color', accent, 'important'); };
        btn.onmouseleave = function(){
          btn.style.setProperty('border-color', blue, 'important'); };
        if (!btn.dataset.qsHaptic) {
          btn.dataset.qsHaptic = "1";
          btn.addEventListener('pointerdown', function () {
            try { window.parent.navigator.vibrate && window.parent.navigator.vibrate(12); }
            catch (e) {}
          });
        }
      }
    }
  } catch (e) {}
}
injectMic();
setInterval(injectMic, 600);
</script>""", height=0)


def _style_gps_component():
    """Restyle the third-party GPS button from the inside: the component
    iframe is served by our own app (same origin), so a zero-height helper
    frame can inject our design system into its document - themed card,
    accent crosshair, hover state - instead of its stock white box. Colors
    are read live off the parent document so it follows Dark/Light/Warm."""
    components.html("""
<script>
function qsVar(name, fallback) {
  try {
    const v = getComputedStyle(window.parent.document.documentElement)
      .getPropertyValue(name).trim();
    return v || fallback;
  } catch (e) { return fallback; }
}
function inject() {
  try {
    const panel = qsVar('--qs-panel', '#161e2e');
    const border = qsVar('--qs-border', '#263145');
    const accent = qsVar('--qs-accent', '#e08850');
    const deep = qsVar('--qs-deep4', '#1b2434');
    const css = `
      html, body {margin:0; padding:0; background:transparent; overflow:hidden;
                  display:flex; align-items:center; justify-content:center;}
      button {
        width: 44px !important; height: 44px !important;
        background: ${panel} !important; border: 1px solid ${border} !important;
        border-radius: 12px !important; cursor: pointer !important;
        display: flex !important; align-items: center !important;
        justify-content: center !important; padding: 0 !important;
        transition: border-color 0.15s !important;
      }
      button:hover {border-color: ${accent} !important; background: ${deep} !important;}
      button:active {
        background: ${accent} !important; transform: scale(0.92) !important;
        transition: transform 0.05s !important;
      }
      button svg, button svg * {stroke: ${accent} !important; fill: ${accent} !important;}
      button span, button div {color: ${accent} !important;}
    `;
    const frames = window.parent.document.querySelectorAll(
      'iframe[title="streamlit_geolocation.streamlit_geolocation"]');
    for (const f of frames) {
      const d = f.contentDocument;
      if (d && d.head) {
        let s = d.getElementById('qs-gps-style');
        if (!s) {
          s = d.createElement('style'); s.id = 'qs-gps-style'; d.head.appendChild(s);
        }
        s.textContent = css;
      }
      // Immediate haptic buzz on tap - GPS acquisition can take a second or
      // two with no visual change otherwise, which is why it can feel like
      // the tap didn't register and needs pressing again.
      const btn = d ? d.querySelector('button') : null;
      if (btn && !btn.dataset.qsHaptic) {
        btn.dataset.qsHaptic = "1";
        btn.addEventListener('pointerdown', function () {
          try { window.parent.navigator.vibrate && window.parent.navigator.vibrate(12); }
          catch (e) {}
        });
      }
    }
  } catch (e) {}
}
inject();
setInterval(inject, 700);
</script>""", height=0)


def google_places_section(trow, ev):
    """Step-by-step help finder. Top to bottom, one decision per row:
    where you are -> what you need -> pick from cards -> route with ETA."""
    from urllib.parse import quote
    st.markdown("##### ⛑️ Find help")
    st.caption("Powered by Google Maps.")
    _style_gps_component()

    # -- 1. STARTING POINT — editable like Google Maps; the GPS button
    #       auto-selects the device location.
    GPS_TXT = "📍 My location (GPS)"
    okey = f"gm_org_{trow['name']}"
    pending_key = f"_pending_{okey}"

    # Any value queued by a "Did you mean" click or the GPS fix below MUST be
    # applied before the text_input with this key is instantiated - Streamlit
    # forbids writing to a widget's session_state key after it's created in
    # the same run (that was the StreamlitAPIException on the suggestion click).
    if pending_key in st.session_state:
        st.session_state[okey] = st.session_state.pop(pending_key)

    with st.container(border=True):
        st.markdown("🟢 **Starting point** — type a place, or tap the button "
                    "for your GPS location")
        tc, ic = st.columns([0.87, 0.13], gap="small",
                            vertical_alignment="center")
        with ic:
            loc = None
            try:
                from streamlit_geolocation import streamlit_geolocation
                loc = streamlit_geolocation()
            except Exception:
                pass
        use_me = bool(loc and loc.get("latitude"))
        glat = glon = None
        if use_me:
            glat = round(float(loc["latitude"]), 5)
            glon = round(float(loc["longitude"]), 5)
            # Auto-fill the field the moment the device location arrives.
            if st.session_state.get("_gps_fix") != (glat, glon):
                st.session_state["_gps_fix"] = (glat, glon)
                st.session_state[pending_key] = GPS_TXT
                st.rerun(scope="fragment")
        with tc:
            typed = st.text_input(
                "Starting point",
                value=f"{trow['name']}, {trow['country']}",
                placeholder="Type an address or place, like Google Maps",
                key=okey, label_visibility="collapsed")
        t = (typed or "").strip()
        default_origin = f"{trow['name']}, {trow['country']}"
        if use_me and (not t or t == GPS_TXT):
            lat, lon = glat, glon
            origin, origin_label = f"{glat},{glon}", "your current location"
        elif t and t != default_origin:
            # Custom typed location - geocode it instead of silently staying
            # on the selected town's coordinates (that was the bug: weather,
            # the facility search bias, and the fallback map center never
            # actually moved to what was typed here).
            try:
                hit = places_search(t, float(trow["latitude"]),
                                    float(trow["longitude"]), n=1)
            except Exception:
                hit = []
            if hit:
                lat, lon = hit[0]["lat"], hit[0]["lon"]
            else:
                lat, lon = float(trow["latitude"]), float(trow["longitude"])
            origin, origin_label = t, t
        else:
            lat, lon = float(trow["latitude"]), float(trow["longitude"])
            origin, origin_label = default_origin, trow["name"]

        # Google-Maps-style hints while typing
        if t and t != GPS_TXT:
            hints = [h for h in place_suggestions(t, lat, lon, 3)
                     if h.lower() != t.lower()]
            if hints:
                st.caption("Did you mean:")
                hcols = st.columns(len(hints))
                for j, h in enumerate(hints):
                    label = h if len(h) <= 36 else h[:34] + "…"
                    if hcols[j].button(label, key=f"sg_{trow['name']}_{j}",
                                       help=h, use_container_width=True):
                        st.session_state[pending_key] = h
                        st.rerun(scope="fragment")

    # -- 2. WHAT YOU NEED (service chips) ---------------------------------
    cat = _chip_pick("What do you need?",
                     [f"{v} {k}" for k, v in CAT_ICONS.items()],
                     key=f"gm_cat_{trow['name']}")
    cat_name = cat.split(" ", 1)[1]
    cat_icon = CAT_ICONS.get(cat_name, "📍")
    cat_art = cat_icon_html(cat_name, cat_icon)
    query = cat_name.lower()
    if cat_name == "Custom search":
        query = st.text_input(
            "Search like on Google Maps", value="emergency room",
            key=f"gm_q_{trow['name']}",
            help="Anything works: 'clinic open now', 'evacuation shelter', "
                 "a facility name...")

    if not query.strip():
        return lat, lon, origin_label
    try:
        places = places_search(query.strip(), lat, lon)
    except Exception as e:
        places = []
        st.warning(f"Google Places unavailable ({str(e)[:200]}).")

    # -- 3. NEAREST OPTIONS (ride-option cards) ---------------------------
    if places:
        rows = []
        for p in places[:6]:
            status = ('<span class="qs-open">● open now</span>' if p["open"]
                      else ('<span class="qs-closed">● closed</span>'
                            if p["open"] is False else ""))
            tel = re.sub(r"[^\d+]", "", p["phone"] or "")
            call = (f'<a href="tel:{tel}">📞 Call</a>' if tel else "")
            nav = (f'<a href="https://www.google.com/maps/dir/?api=1'
                   f'&destination={p["lat"]},{p["lon"]}" target="_blank" '
                   f'rel="noopener">🧭 Navigate</a>')
            rows.append(
                f'<div class="qs-fac"><div class="qs-fac-ic">{cat_art}</div>'
                f'<div class="qs-fac-main">'
                f'<div class="qs-fac-name">{html.escape(p["name"])}</div>'
                f'<div class="qs-fac-addr">{html.escape(p["addr"])}</div>'
                f'<div class="qs-fac-actions">{call}{nav}</div></div>'
                f'<div class="qs-fac-meta">'
                f'<span class="qs-km">{p["km"]:.1f} km</span>{status}'
                f'</div></div>')
        st.markdown("".join(rows), unsafe_allow_html=True)

        # -- 4. YOUR ROUTE (trip card) ------------------------------------
        with st.container(border=True):
            r1, r2 = st.columns([0.62, 0.38])
            with r1:
                dest_ix = st.selectbox(
                    "Destination", range(len(places[:6])),
                    format_func=lambda i: f"{places[i]['name']} "
                                          f"({places[i]['km']:.1f} km)",
                    key=f"gm_dest_{trow['name']}")
            with r2:
                mode_pick = _chip_pick("Travel mode",
                                       ["🚗 Drive", "🚶 Walk", "🚴 Bike"],
                                       key=f"gm_mode_{trow['name']}")
            mode = {"🚗 Drive": "driving", "🚶 Walk": "walking",
                    "🚴 Bike": "bicycling"}.get(mode_pick, "driving")
            dest = places[dest_ix]
            st.markdown(
                f'<div class="qs-trip">'
                f'<span class="dotA">●</span> {html.escape(origin_label.title())}'
                f'<br><span class="leg">┆</span><br>'
                f'<span class="dotB">●</span> {html.escape(dest["name"])}'
                f'</div>', unsafe_allow_html=True)
            components.iframe(
                f"https://www.google.com/maps/embed/v1/directions"
                f"?key={MAPS_API_KEY}&origin={quote(origin)}"
                f"&destination={dest['lat']},{dest['lon']}&mode={mode}",
                height=360)
            st.markdown(
                '<div class="qs-maphelp">'
                '<b>How to use this map</b>'
                '<span>The map shows the route from your starting point to the '
                'facility, with the estimated arrival time.</span>'
                '<span>🧭 <b>Navigate</b> opens live turn-by-turn in Google Maps.'
                ' &nbsp; 📞 <b>Call</b> dials the facility.</span>'
                '<span>Switch <b>Drive / Walk / Bike</b> to update the route '
                'and arrival time.</span>'
                '</div>', unsafe_allow_html=True)
    else:
        components.iframe(
            f"https://www.google.com/maps/embed/v1/search"
            f"?key={MAPS_API_KEY}&q={quote(query.strip())}"
            f"&center={lat},{lon}&zoom=12",
            height=360)
    return lat, lon, origin_label


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


# Different hosts filter differently: some reject bare python agents,
# others flag browser agents coming from datacenter IPs. Try both.
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/126.0 Safari/537.36"}
FEED_UA = {"User-Agent": "QuakeSense/1.0 (earthquake information service; "
                         "+https://github.com/EthannMK/quakesense)"}


def _resilient_get(url, params=None, timeout=8):
    """GET with one UA, retry once with the other on any failure."""
    try:
        r = requests.get(url, params=params, timeout=timeout, headers=FEED_UA)
        r.raise_for_status()
        return r
    except Exception as e:
        print(f"[feeds] {url.split('/')[2]} failed with feed UA ({e}); "
              f"retrying with browser UA")
        r = requests.get(url, params=params, timeout=timeout, headers=BROWSER_UA)
        r.raise_for_status()
        return r


def _fetch_relief(n: int):
    """Latest earthquake situation reports/statements from ReliefWeb, the UN
    OCHA humanitarian information service (public RSS)."""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    r = _resilient_get("https://reliefweb.int/updates/rss.xml?search=earthquake",
                       timeout=6)
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


def _fetch_gdacs(n: int):
    """Earthquake alerts from GDACS - the UN/EC Global Disaster Alert and
    Coordination System (backup source for official statements)."""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    r = _resilient_get("https://www.gdacs.org/xml/rss.xml", timeout=6)
    root = ET.fromstring(r.content)
    now = datetime.now(timezone.utc)
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        url = (it.findtext("link") or "").strip()
        if "earthquake" not in title.lower() or not url:
            continue
        ago = ""
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate") or "")
            hrs = int((now - dt).total_seconds() // 3600)
            ago = f"{hrs}h ago" if hrs < 48 else f"{hrs // 24}d ago"
        except Exception:
            pass
        out.append({"title": title, "url": url, "source": "GDACS · UN/EC",
                    "ago": ago})
        if len(out) >= n:
            break
    return out


def news_rail_component(server_cards, fetch_photos=True):
    """Card rail rendered by the USER's browser. With fetch_photos=True it
    fetches GDELT directly (residential IPs aren't bot-filtered like cloud
    egress IPs are), falling back to the server-fetched cards; with False it
    renders the given cards as-is (static rails like Official updates).
    Self-contained CSS - the component renders in its own iframe."""
    import json as _json
    payload = _json.dumps(server_cards)
    majors = _json.dumps(list(MAJOR_OUTLETS))
    fetch_flag = "true" if fetch_photos else "false"
    rail_css = f"""
<style>
body {{margin:0; background:transparent; font-family:-apple-system,"Segoe UI",Roboto,sans-serif;}}
.qs-newsrail {{position:relative;}}
.qs-newsrail-scroll {{overflow-x:auto; overflow-y:hidden; scroll-behavior:smooth;
  scrollbar-width:none; -ms-overflow-style:none;}}
.qs-newsrail-scroll::-webkit-scrollbar {{display:none;}}
.qs-newsrail-inner {{display:flex; gap:12px; width:max-content;}}
.qs-railbtn {{position:absolute; top:44%; transform:translateY(-50%); z-index:2;
  width:30px; height:30px; border-radius:50%; border:1px solid {_pal['border']};
  background:{_pal['panel']}; color:{_pal['text']}; display:flex; align-items:center;
  justify-content:center; cursor:pointer; opacity:0.85; font-size:16px;
  line-height:1; user-select:none; box-shadow:0 1px 4px rgba(0,0,0,0.4);}}
.qs-railbtn:hover {{opacity:1; border-color:{_pal['accent']}; color:{_pal['accent']};}}
.qs-railbtn.qs-rail-left {{left:2px;}}
.qs-railbtn.qs-rail-right {{right:2px;}}
.qs-newscard {{flex:0 0 250px; background:{_pal['panel']}; border:1px solid {_pal['border']};
  border-radius:8px; overflow:hidden; text-decoration:none;
  transition:border-color 0.15s;}}
.qs-newscard:hover {{border-color:{_pal['accent']};}}
.qs-newsimg {{height:118px; background-size:cover; background-position:center;
  background-color:{_pal['border']};}}
.qs-newsmono {{height:118px; display:flex; align-items:center;
  justify-content:center; background:{_pal['border']}; font-size:30px;
  font-weight:600; color:{_pal['accent']}; letter-spacing:0.02em;}}
.qs-newstxt {{display:flex; flex-direction:column; gap:4px; padding:8px 10px 10px;}}
.qs-newssrc {{font-size:11px; color:{_pal['muted']}; text-transform:uppercase;
  letter-spacing:0.06em;}}
.qs-newstitle {{font-size:13px; color:{_pal['text']}; line-height:1.35;}}
.qs-empty {{color:{_pal['muted']}; font-size:13px;}}
</style>
<div id="rail"><span class="qs-empty">Loading headlines...</span></div>
"""
    components.html(rail_css + """
<script>
const FALLBACK = """ + payload + """;
const MAJORS = """ + majors + """;
const FETCH = """ + fetch_flag + """;
function esc(s) {const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML;}
function agoFrom(sd) {
  try {
    const dt = new Date(sd.slice(0,4) + '-' + sd.slice(4,6) + '-' + sd.slice(6,8) + 'T'
                        + sd.slice(9,11) + ':' + sd.slice(11,13) + ':' + sd.slice(13,15) + 'Z');
    const h = Math.floor((Date.now() - dt.getTime()) / 3600000);
    return h < 48 ? h + 'h ago' : Math.floor(h / 24) + 'd ago';
  } catch (e) {return '';}
}
function card(c) {
  const vis = c.img
    ? '<div class="qs-newsimg" style="background-image:url(\\'' + esc(c.img) + '\\')"></div>'
    : '<div class="qs-newsmono">' + esc(c.mono || (c.source || '•')[0].toUpperCase()) + '</div>';
  const meta = esc(c.source || '') + (c.ago ? ' · ' + esc(c.ago) : '');
  return '<a class="qs-newscard" href="' + esc(c.url) + '" target="_blank" rel="noopener">'
       + vis + '<div class="qs-newstxt"><span class="qs-newssrc">' + meta
       + '</span><span class="qs-newstitle">' + esc((c.title || '').slice(0, 110))
       + '</span></div></a>';
}
let railPauseUntil = 0;
function railScroll(dir) {
  const sc = document.getElementById('railScroll');
  if (!sc) return;
  sc.scrollBy({left: dir * 270, behavior: 'smooth'});
  railPauseUntil = Date.now() + 2500;
}
function railAutoStep() {
  const sc = document.getElementById('railScroll');
  if (!sc || Date.now() < railPauseUntil || sc.matches(':hover')) return;
  sc.scrollLeft += 1;
  const half = sc.scrollWidth / 2;
  if (sc.scrollLeft >= half) {sc.scrollLeft -= half;}
}
setInterval(railAutoStep, 40);
function render(cards) {
  const el = document.getElementById('rail');
  if (!cards || !cards.length) {
    el.innerHTML = '<span class="qs-empty">Headlines unavailable right now — check back shortly.</span>';
    return;
  }
  const row = cards.map(card).join('');
  el.innerHTML = '<div class="qs-newsrail">'
    + '<div class="qs-railbtn qs-rail-left" onclick="railScroll(-1)">&#8249;</div>'
    + '<div class="qs-railbtn qs-rail-right" onclick="railScroll(1)">&#8250;</div>'
    + '<div class="qs-newsrail-scroll" id="railScroll">'
    + '<div class="qs-newsrail-inner">' + row + row + '</div></div></div>';
}
render(FALLBACK);
if (FETCH) {
fetch('https://api.gdeltproject.org/api/v2/doc/doc?query=earthquake%20sourcelang:english&mode=ArtList&format=json&maxrecords=50&sort=DateDesc',
      {signal: AbortSignal.timeout(8000)})
  .then(function(r) {return r.json();})
  .then(function(d) {
    const arts = d.articles || [];
    arts.sort(function(a, b) {
      const am = MAJORS.some(function(m) {return (a.domain || '').includes(m);}) ? 0 : 1;
      const bm = MAJORS.some(function(m) {return (b.domain || '').includes(m);}) ? 0 : 1;
      return am - bm;
    });
    const seenD = new Set(), seenT = new Set(), out = [];
    for (const a of arts) {
      const img = (a.socialimage || '').trim(), t = (a.title || '').trim();
      const u = a.url, dom = a.domain || '';
      if (!img.startsWith('http') || !t || !u) continue;
      const tk = t.toLowerCase().replace(/[^a-z0-9]/g, '').slice(0, 64);
      if (seenD.has(dom) || seenT.has(tk)) continue;
      seenD.add(dom); seenT.add(tk);
      out.push({title: t, url: u, img: img, source: dom, ago: agoFrom(a.seendate || '')});
      if (out.length >= 8) break;
    }
    if (out.length) {render(out);}
  })
  .catch(function(e) {});
}
</script>""", height=215)


def _fetch_gnews(n: int):
    """Top earthquake headlines from global media, via the Google News RSS
    aggregator (carries Reuters, AP, BBC, CNN, etc.)."""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    r = _resilient_get(
        "https://news.google.com/rss/search?q=earthquake&hl=en-US&gl=US&ceid=US:en",
        timeout=6)
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
def _text_feeds_bundle():
    """Headlines + UN reports fetched in parallel; never caches a total miss."""
    from concurrent.futures import ThreadPoolExecutor

    def safe(fn, *a):
        try:
            return fn(*a)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_heads = ex.submit(safe, _fetch_gnews, 10)
        f_reps = ex.submit(safe, _fetch_relief, 5)
        f_gdacs = ex.submit(safe, _fetch_gdacs, 5)
        heads, reps = f_heads.result(), f_reps.result()
        gdacs = f_gdacs.result()
    if not reps:
        reps = gdacs
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


def usgs_event_cards(live_df, n: int = 8):
    """Cards built from the USGS live feed itself - the guaranteed fallback.
    The feed always loads (it has a bundled offline snapshot), so the media
    rail can never be empty."""
    if live_df is None or live_df.empty:
        return []
    now = pd.Timestamp.now(tz="UTC")
    out = []
    for r in significant_events(live_df, n).itertuples():
        hrs = int((now - r.time).total_seconds() // 3600)
        ago = f"{hrs}h ago" if hrs < 48 else f"{hrs // 24}d ago"
        out.append({"title": f"M{r.mag:.1f} earthquake — {r.place}",
                    "url": r.url or "https://earthquake.usgs.gov/",
                    "img": "", "mono": f"M{r.mag:.1f}",
                    "source": "USGS · official record", "ago": ago})
    return out


def media_search_fallback_cards(live_df, n: int = 6):
    """Search-link cards for when live headlines can't be fetched (Google
    News RSS runs server-side from Cloud Run, whose egress IPs get
    rate-limited by news aggregators far more than a regular visitor's
    browser does - this happens more often than it should).

    Deliberately NOT the same content as Official Updates below (that used
    to be the fallback here, and looked like a bug when both rails showed
    identical USGS cards) - these link out to a live news search per event
    instead, so the rail is never just stuck on "unavailable" for a long
    time, but also never silently duplicates the other rail."""
    if live_df is None or live_df.empty:
        return []
    from urllib.parse import quote as _urlquote
    out = []
    for r in significant_events(live_df, n).itertuples():
        q = _urlquote(f"M{r.mag:.1f} earthquake {r.place}")
        out.append({"title": f"Search coverage: M{r.mag:.1f} — {r.place}",
                    "url": f"https://news.google.com/search?q={q}",
                    "img": "", "mono": "🔎",
                    "source": "Search the news", "ago": ""})
    return out


@st.fragment(run_every=60)
def media_section(live_df):
    """Two card rails. (1) World media: photos fetched by the visitor's own
    browser, instant headline cards as fallback. (2) Official updates:
    magnitude-tile cards from the USGS record plus UN situation reports -
    replaces the old plain link list. Neither rail can be empty or slow."""
    feeds = text_feeds()
    news_cards = [{"title": h["title"], "url": h["link"], "img": "",
                   "source": h["source"], "ago": h["ago"]}
                  for h in feeds["headlines"]]
    if not news_cards:
        # Search-link cards, NOT the same USGS event cards as Official
        # Updates below (that duplication looked like a bug). The
        # component's own client-side GDELT fetch still runs after this and
        # replaces these with real headlines if it succeeds.
        news_cards = media_search_fallback_cards(live_df)
    news_rail_component(news_cards)

    official = usgs_event_cards(live_df)
    for rep in feeds["reports"][:4]:
        official.append({"title": rep["title"], "url": rep["url"], "img": "",
                         "mono": "UN", "source": rep["source"],
                         "ago": rep["ago"]})
    if official:
        st.subheader("🌐 Official updates")
        st.caption("Significant events from the official USGS record and "
                   "situation reports from UN agencies — tap a card to open "
                   "the source.")
        news_rail_component(official, fetch_photos=False)


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
    with st.popover("💬", help="Ask QuakeSense"):
        st.markdown(
            '<div class="qs-chat-head"><div class="qs-chat-av">✦</div>'
            '<div><div class="qs-chat-title">Terra</div>'
            '<div class="qs-chat-sub">QuakeSense assistant'
            '</div></div></div>', unsafe_allow_html=True)
        # inject an ✕ close button into the header (clicks the launcher to close)
        components.html("""
<script>
function qsAddClose(){
 try{
  const doc=window.parent.document;
  const head=doc.querySelector('.qs-chat-head');
  if(head && !head.querySelector('.qs-x')){
    const x=doc.createElement('div');
    x.className='qs-x'; x.textContent='\\u2715';
    x.style.cssText='margin-left:auto;cursor:pointer;color:var(--qs-accent-dark);'
      +'font-weight:700;font-size:17px;line-height:1;padding:2px 4px';
    x.onclick=function(){const t=doc.querySelector('button[data-testid=\\"stPopoverButton\\"]'); if(t) t.click();};
    head.appendChild(x);
  }
 }catch(e){}
}
qsAddClose(); setInterval(qsAddClose,400);
</script>""", height=0)
        st.caption(f"📍 Quick answers about: {context}")
        pending_q = None
        box = st.container(height=300)
        with box:
            if not hist:
                with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                    st.markdown(BOT_INTRO)
                st.markdown('<p class="qs-sug-label">Try asking</p>',
                            unsafe_allow_html=True)
                for _j, _ex in enumerate(QUICK_EXAMPLES):
                    if st.button(_ex, key=f"qx_{_j}", use_container_width=True):
                        pending_q = _ex
            for _i, m in enumerate(hist):
                with st.chat_message(m["role"], avatar=AVATARS.get(m["role"])):
                    st.markdown(m["content"])
                    if m.get("sources"):
                        st.caption("Sources: " + " · ".join(
                            f"[{s['title']}]({s['uri']})" for s in m["sources"][:3]))
                    if m.get("goto_ask") and st.button(
                            "✦ Open Ask page", key=f"goto_ask_{_i}",
                            use_container_width=True):
                        st.session_state["_pending_topnav"] = "Ask"
                        st.rerun()
        # Composer. (Voice input is available via VOICE_INPUT_ENABLED - kept
        # off for now: browser speech recognition only handled English.)
        voice_q = None
        if VOICE_INPUT_ENABLED:
            _style_mic_component()
            ic, tc = st.columns([0.14, 0.86], vertical_alignment="bottom")
            with ic:
                try:
                    from streamlit_mic_recorder import speech_to_text
                    voice_q = speech_to_text(
                        language="en", start_prompt="🎙️", stop_prompt="🔴",
                        just_once=True, use_container_width=False,
                        key="quick_stt")
                except Exception:
                    pass
            with tc:
                q_sub = st.chat_input("Type a message…", key="quick_chat_input")
        else:
            q_sub = st.chat_input("Type a message…", key="quick_chat_input")
        the_q = (pending_q or voice_q
                 or (q_sub.strip() if q_sub and q_sub.strip() else None))
        if the_q:
            q = the_q
            recent = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in hist[-4:])
            hist.append({"role": "user", "content": q})
            with box:
                with st.chat_message("user", avatar=AVATARS["user"]):
                    st.markdown(q)
                with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                    try:
                        with st.spinner("Thinking..."):
                            res = popup_answer(q, context, recent)
                        st.markdown(res["answer"])
                        if res.get("goto_ask"):
                            _gk = f"goto_ask_live_{len(hist)}"
                            if st.button("✦ Open Ask page", key=_gk,
                                        use_container_width=True):
                                st.session_state["_pending_topnav"] = "Ask"
                                st.rerun()
                        hist.append({"role": "assistant", "content": res["answer"],
                                     "goto_ask": res.get("goto_ask", False)})
                    except Exception as e:
                        msg = (f"Unavailable right now ({str(e)[:60]}). "
                               f"Try the ✦ Ask page.")
                        st.markdown(msg)
                        if st.button("✦ Open Ask page",
                                    key=f"goto_ask_err_{len(hist)}",
                                    use_container_width=True):
                            st.session_state["_pending_topnav"] = "Ask"
                            st.rerun()
                        hist.append({"role": "assistant", "content": msg,
                                     "goto_ask": True})


@st.cache_data(ttl=1800, show_spinner=False)
def _oaf_cached(event_id: str):
    return official_forecast(event_id)


@st.cache_data(ttl=1800, show_spinner=False)
def _weather_cached(lat: float, lon: float):
    return weather_conditions(lat, lon)


def aftershock_block(ev: dict, live_df):
    """Aftershock outlook: the OFFICIAL USGS forecast when one exists, plus
    aftershocks already recorded. Never our own prediction.

    Always rendered (same expander, every event) so its presence doesn't look
    like a bug when USGS hasn't issued a forecast for a particular event."""
    fc = _oaf_cached(ev.get("id", ""))
    obs = observed_aftershocks(live_df, ev)
    with st.expander("🔄 Aftershock outlook", expanded=False):
        if obs and obs.get("count"):
            c1, c2 = st.columns(2)
            c1.metric("Aftershocks recorded nearby", obs["count"],
                      help="M2.5+ events within 150 km since this earthquake "
                           "(USGS live feed).")
            if obs.get("max_mag"):
                c2.metric("Largest so far", f"M {obs['max_mag']:.1f}")
        else:
            st.caption("No aftershocks recorded near this event yet (USGS "
                       "live feed, past 14 days).")

        if fc:
            st.markdown("**Official USGS aftershock forecast**")
            rows = [{"Window": w["label"], "Magnitude": f"M{w['mag']:.0f}+",
                     "Chance of at least one": f"{w['probability']}%"}
                    for w in fc["windows"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)
            st.caption("Published by the USGS Operational Aftershock Forecast "
                       "for this event. Aftershock forecasting is statistical; "
                       "earthquakes themselves cannot be predicted.")
        else:
            st.caption("USGS has not issued an aftershock forecast for this "
                       "event — it only publishes one for some events. The "
                       "counts above are earthquakes already recorded, not a "
                       "forecast.")
        st.markdown(GUIDANCE)


def weather_block(lat: float, lon: float, label: str):
    """Weather where people actually are - the resolved Starting point (GPS
    or a custom typed location), not necessarily the affected-area town
    dropdown, so it follows whichever location the user is really at."""
    w = _weather_cached(round(float(lat), 2), round(float(lon), 2))
    if not w:
        return
    st.markdown(f"##### 🌦️ Weather in {label}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Now", f"{w['icon']} {w['temp']:.0f}°C" if w.get("temp") is not None
              else w["icon"], help=w["label"])
    c2.metric("Rain chance · 24 h",
              f"{w['rain_chance']}%" if w.get("rain_chance") is not None else "—")
    c3.metric("Expected rain · 24 h", f"{w['rain_mm']} mm")
    st.info(weather_advisory(w))


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


def _play_then_reveal(audio_bytes: bytes, key: str):
    """Autoplay `audio_bytes` with no visible controls; once playback ends
    (or fails / is blocked by the browser), swap in a normal player so the
    answer can be replayed on demand.

    Deliberately does NOT rely on the HTML `autoplay` attribute alone: mobile
    browsers routinely block audio that isn't started synchronously within a
    user gesture (this narration arrives asynchronously, after the AI answer
    finishes streaming, well after the original tap) - and a *blocked*
    autoplay fires neither "ended" nor "error", so relying on those alone
    left the player invisible for a full 15 seconds with no sound and no
    visible control, which looked like total silence/failure. Calling
    play() ourselves lets us catch that rejection and reveal a normal,
    tappable player immediately instead."""
    b64 = base64.b64encode(audio_bytes).decode()
    components.html(f"""
      <audio id="aud_{key}" style="display:none;width:100%;height:34px;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mpeg">
      </audio>
      <script>
        const a = document.getElementById("aud_{key}");
        const reveal = () => {{ a.style.display = "block"; a.controls = true; }};
        a.addEventListener("ended", reveal);
        a.addEventListener("error", reveal);
        const p = a.play();
        if (p && p.catch) {{ p.catch(reveal); }}
        setTimeout(reveal, 15000);
      </script>
    """, height=40)


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
        st.markdown("<p style='text-align:center;color:var(--qs-muted);'>Try one of these:</p>",
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
                a1, a2, _ = st.columns([0.09, 0.09, 0.82])
                if m.get("rated"):
                    a1.caption("✓ noted")
                else:
                    a1.button(":material/thumb_up:", key=f"fb_up_{i}",
                              help="Good answer", on_click=_rate_answer,
                              args=(i, "up"))
                    a2.button(":material/thumb_down:", key=f"fb_down_{i}",
                              help="Poor answer", on_click=_rate_answer,
                              args=(i, "down"))
                if m.get("audio"):
                    st.audio(m["audio"], format="audio/mp3")

    if st.session_state.get("area"):
        st.caption(f"The agent can see your current My Area analysis "
                   f"({st.session_state.area['city']}) — ask about it here.")

    # Composer. st.chat_input grows as you type, scrolls at its max height and
    # carries its own send arrow. (Voice input behind ASK_VOICE_ENABLED - same
    # 🎙️ mic icon as before, but now records raw audio and transcribes it with
    # Gemini instead of the browser's English-only speech recognition.)
    voice_q = None
    if ASK_VOICE_ENABLED:
        _style_mic_component()
        # Mic first, chat input second: side by side on desktop, but on a
        # narrow phone screen Streamlit stacks these columns top-to-bottom in
        # the order they're declared, so the mic ends up ABOVE the text
        # field rather than squeezed right next to it. That's deliberate,
        # not a layout accident - a thumb reaching for the send arrow at the
        # edge of a chat_input is exactly the kind of slip-touch that would
        # otherwise land on a mic button crammed right beside it, arming the
        # recorder by mistake. Stacked, there's real space between "type" and
        # "record" instead of two tiny targets touching each other.
        ic, tc = st.columns([0.07, 0.93], vertical_alignment="bottom")
        with ic:
            try:
                from streamlit_mic_recorder import mic_recorder
                clip = mic_recorder(start_prompt="🎙️", stop_prompt="🔴",
                                    just_once=True, use_container_width=False,
                                    format="webm", key="mic_ask")
            except Exception:
                clip = None
        with tc:
            typed = st.chat_input("Ask anything about earthquakes…",
                                  key="ask_chat_input")
        if clip and clip.get("bytes"):
            with st.spinner("Transcribing..."):
                voice_q = transcribe_audio(clip["bytes"], mime_type="audio/webm")
            if not voice_q:
                st.warning("Didn't catch any speech in that clip — try again, "
                          "or type your question instead.")
    else:
        typed = st.chat_input("Ask anything about earthquakes…",
                              key="ask_chat_input")
    question = pending or voice_q or typed
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
                # Narrate once the full answer has streamed - same pipeline
                # whether the question was typed or spoken.
                audio_bytes = None
                if ASK_VOICE_ENABLED:
                    try:
                        audio_bytes = text_to_speech(answer, res.get("lang_code", "en"))
                    except Exception:
                        audio_bytes = None
                    if audio_bytes:
                        _play_then_reveal(audio_bytes,
                                          key=f"voice_new_{len(st.session_state.chat)}")
                    else:
                        st.caption("🔇 Voice narration isn't available in this "
                                  "language yet.")
                st.session_state.chat.append({"role": "assistant", "content": answer,
                                              "sql": res["sql"], "df": res["df"],
                                              "mode": res.get("mode"),
                                              "sources": srcs,
                                              "note": res.get("note", ""),
                                              "audio": audio_bytes})
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
            lang = st.selectbox("Language", APP_LANGUAGES)
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
                st.bar_chart(hd.groupby("decade").size(), color=_pal["accent"])
                st.caption("Taller recent bars often reflect better instruments, "
                           "not necessarily more earthquakes.")
            with g2:
                st.markdown("**How strong they were**")
                st.bar_chart(hd.groupby("strength", observed=False).size(), color=_pal["accent"])
                st.caption("Most events cluster at the lower magnitudes - "
                           "the big ones are rare but matter most.")
            with st.expander("Map: every M5+ epicenter within 300 km since 1975",
                             expanded=True):
                hist_map = ar["df"].rename(columns={"latitude": "lat", "longitude": "lon"})
                st.map(hist_map[["lat", "lon"]], zoom=5, color=_pal["accent"], size=8000)


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
            st.bar_chart(daily, color=_pal["accent"])
            st.caption("A tight burst suggests an aftershock sequence; "
                       "spread-out days suggest a swarm.")
        with v2:
            st.markdown("**Where they struck**")
            st.map(evs[["lat", "lon"]], zoom=4, color=_pal["accent"])


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
        dd_lang = st.selectbox("Language", APP_LANGUAGES, key="rt_lang")
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
        resolved = google_places_section(trow, ev)
        if resolved:
            weather_block(*resolved)
        else:
            weather_block(trow["latitude"], trow["longitude"], trow["name"])
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

    weather_block(trow["latitude"], trow["longitude"], trow["name"])


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
st.markdown(f"""
<div class="qs-header">
  <p class="qs-wordmark">QUAKE<span>SENSE</span></p>
  <p class="qs-subline"><span class="qs-live"></span>{t("tagline")}</p>
</div>
""", unsafe_allow_html=True)

# ---- top navigation — segmented tabs (mobile-friendly, no hamburger) -------
# Apply any page-switch queued by the popup's "Open Ask" button BEFORE the
# widget below is instantiated (Streamlit forbids writing to a widget's
# session_state key after it's created in the same run).
if "_pending_topnav" in st.session_state:
    st.session_state["topnav"] = st.session_state.pop("_pending_topnav")

# Stable internal keys (never translated) drive routing; format_func supplies
# the translated label the user actually sees, so the UI language switch
# never breaks navigation or the popup's "open Ask page" redirect.
NAV_KEYS = ["Live", "My Area", "Ask", "Respond"]
NAV_LABEL_KEYS = {"Live": "nav_live", "My Area": "nav_my_area",
                  "Ask": "nav_ask", "Respond": "nav_respond"}
page = st.radio(
    "Navigation", NAV_KEYS, format_func=lambda k: t(NAV_LABEL_KEYS[k]),
    horizontal=True, label_visibility="collapsed", key="topnav")

try:
    live = get_live()
    feed_ok = True
except Exception as e:
    st.error(f"USGS live feed unreachable: {e}")
    live, feed_ok = pd.DataFrame(), False

render_ticker(live)

# -------------------------------------------------- sidebar = settings panel --
st.sidebar.markdown(f'<p class="qs-set-sec">{t("sidebar_data")}</p>', unsafe_allow_html=True)
if st.sidebar.button(t("sidebar_refresh"), use_container_width=True):
    get_live.clear()
    st.rerun()

st.sidebar.markdown(f'<p class="qs-set-sec">{t("sidebar_prefs")}</p>',
                    unsafe_allow_html=True)
_lang_choice = st.sidebar.selectbox(
    t("sidebar_language"), UI_LANGUAGES,
    index=UI_LANGUAGES.index(st.session_state.ui_language),
    key="lang_pick",
    help="Translates the app's own interface. Doesn't change the language "
         "Terra writes answers in - that always matches your question.")
if _lang_choice != st.session_state.ui_language:
    st.session_state.ui_language = _lang_choice
    st.query_params["lang"] = _lang_choice
    st.rerun()

st.sidebar.caption(t("sidebar_disclaimer"))

_koda_b64 = logo_b64()
st.sidebar.markdown(f"""
<div class="qs-sidebar-bottom">
  <p class="qs-credit">{t("sidebar_built_with")}</p>
  <p class="qs-credit-items">Google Cloud &nbsp;·&nbsp; BigQuery &nbsp;·&nbsp;
  Vertex AI Gemini &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; USGS</p>
  <img src="data:image/png;base64,{_koda_b64}" width="56">
  <p class="qs-team">{t("sidebar_team")}</p>
</div>
""", unsafe_allow_html=True)

# ==================================================================== LIVE ==
if page == "Live":
    st.caption(t("sidebar_disclaimer"))
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
        st.subheader(t("live_subheader"))
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
        aftershock_block(ev, live)

        # ------------------------------------------- unusual activity (anomaly)
        st.divider()
        st.subheader("🚨 Unusual activity this week")
        st.caption("Regions where this week's M4.5+ activity is far above their "
                   "own 50-year weekly average — swarms and aftershock sequences, "
                   "with an AI explanation of the pattern.")
        flagged, live_cells = detect(live)
        if flagged.empty:
            st.success("No regions show anomalously elevated activity this week.")
        else:
            st.warning(f"{len(flagged)} region(s) flagged with unusually high activity")
            show = flagged.rename(columns={"sample_place": "region",
                                           "weekly_avg": "normal_week",
                                           "ratio": "times_normal"})
            st.dataframe(show[["region", "current", "normal_week",
                               "times_normal", "max_mag"]].round(2),
                         use_container_width=True, hide_index=True)
            anomaly_explain_block(flagged, live_cells)

        # ------------------------------------------- global media coverage
        st.divider()
        st.subheader("📺 Global media coverage")
        st.caption("Latest earthquake coverage from world media — click a "
                   "card to read the full story.")
        media_section(live)

        with st.expander("ℹ️  How QuakeSense works"):
            st.markdown("""
QuakeSense turns raw USGS seismic data into decisions communities can act on —
it **never predicts** earthquakes.

- **🛰️ Live** — every M2.5+ quake worldwide in the past 7 days: map, filters,
  plain-language **AI briefings** for significant events, **unusual-activity**
  flags vs the 50-year record, and **global media** coverage.
- **📍 My Area** — a risk profile for any town on Earth, in 8 languages, from
  that area's real 50-year record.
- **✦ Ask** — ask anything in any language. Historical numbers come from
  86,000 verified USGS records (the **SQL is shown**), this week from the live
  feed, current events from the web with sources cited.
- **⛑️ Respond** — find the nearest hospitals / fire / police from your live
  location via Google Maps, and download official offline safety guides.

Data: USGS (public domain) · Gemini on Vertex AI · BigQuery · Google Maps.
""")

        quick_ask(f"Live world map (past 7 days); selected event: {pick}", live)

# ================================================================= MY AREA ==
elif page == "My Area":
    st.subheader(t("myarea_subheader"))
    st.caption("Select your country and town - the agent combines your area's 50-year record "
               "with this week's live activity into a personal risk profile, in your language.")
    st.caption(t("sidebar_disclaimer"))

    my_area_block(towns_db(), live)

    _area_ctx = (f"My Area risk profile for {st.session_state.area['city']}"
                 if st.session_state.get("area")
                 else "My Area page (no town profiled yet)")
    quick_ask(_area_ctx, live)

# ====================================================================== ASK ==
elif page == "Ask":
    st.subheader(t("ask_subheader"))
    st.caption("Ask anything in any language. Historical numbers come from 50 years "
               "of USGS records — **the query and the matching rows are shown**, so "
               "every figure can be checked. This week's events come from the live "
               "feed, current news with sources cited. For quick questions about a "
               "page you're on, use the 💬 button instead.")
    st.caption(t("sidebar_disclaimer"))

    chat_agent(live)

# ==================================================================== RESPOND ==
elif page == "Respond":
    st.subheader(t("respond_subheader"))
    st.caption("Find nearby hospitals, fire and police, and download official "
               "safety guides for offline use.")
    st.caption(t("sidebar_disclaimer"))

    # ---- Find help in the affected area --------------------------------
    st.markdown("##### Emergency resources in the affected area")
    pick_rt = None
    tdb2 = towns_db()
    if live.empty:
        st.info("Live feed unavailable.")
    elif tdb2 is None:
        st.error("Towns database missing. Run once:  python scripts/load_towns.py")
    else:
        sig = significant_events(live)
        labels_rt = [f"M{r.mag:.1f}  ·  {r.place}  ·  {r.time:%b %d %H:%M} UTC"
                     for r in sig.itertuples()]
        carry = st.session_state.get("sig_event")
        default_rt = labels_rt.index(carry) if carry in labels_rt else 0
        pick_rt = st.selectbox("Which event are you responding to?", labels_rt,
                               index=default_rt, key="rt_event",
                               help="Defaults to the event you picked on Live.")
        ev = sig.iloc[labels_rt.index(pick_rt)].to_dict()
        st.caption(f"Towns near this event ({ev['place']}), located from its actual "
                   f"USGS coordinates. Pick the affected town, then find help.")
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

    # ---- Offline library ------------------------------------------------
    st.divider()
    offline_library()

    quick_ask(f"Respond page; selected event: {pick_rt}" if pick_rt
              else "Respond page", live)

st.divider()
st.caption(t("footer"))
