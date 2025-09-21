# streamlit/pages/02_Weatherwatch.py
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import date, timedelta

from api_client import (
    get_weather_accuracy,
    get_weather_data,
    get_weather_logs,
    delete_weather_logs,
)

st.set_page_config(page_title="sehbmaster – Weatherwatch", page_icon="⛅", layout="wide")
st.title("⛅ Weatherwatch")
st.caption("Vorhersage-Güte je Vorlauf (Lead-Tage) – Vergleich von Vorhersagen (Lead 1..N) mit Beobachtung (Lead 0)")

# -----------------------------
# Caching-Loader
# -----------------------------
@st.cache_data(ttl=30)
def load_accuracy(frm: date, to: date, model: str, city: str, max_lead: int):
    return get_weather_accuracy(frm.isoformat(), to.isoformat(), model=model, city=city, max_lead=max_lead)

@st.cache_data(ttl=30)
def load_latest_data(days_back: int = 30, model: str = "default", city: str = "berlin"):
    to = date.today()
    frm = to - timedelta(days=days_back)
    return get_weather_data(frm.isoformat(), to.isoformat(), model=model, city=city, lead_days=None, limit=10000)

# -----------------------------
# Controls
# -----------------------------
colA, colB, colC, colD = st.columns([1, 1, 1, 1])
with colA:
    days_back = st.slider("Zeitraum (Tage)", 7, 180, 35)
with colB:
    max_lead = st.slider("Max Lead", 0, 7, 7)
with colC:
    model = st.text_input("Modell", "default")
with colD:
    city = st.text_input("Stadt", "berlin")

frm = date.today() - timedelta(days=days_back)
to = date.today()

# Reload-Button
if st.button("🔄 Neu laden"):
    load_accuracy.clear()
    load_latest_data.clear()
    st.rerun()

# -----------------------------
# Accuracy (MAE + Wetterstring-Treffer)
# -----------------------------
try:
    acc = load_accuracy(frm, to, model=model, city=city, max_lead=max_lead)
    if acc and acc.get("buckets"):
        df = pd.DataFrame(acc["buckets"]).sort_values("lead_days")

        col1, col2, col3 = st.columns(3)
        with col1:
            fig_t = px.line(
                df, x="lead_days", y="temp_mae", markers=True,
                title=f"Temp MAE (°C) – {model} @ {city}"
            )
            fig_t.update_layout(xaxis_title="Lead (Tage)", yaxis_title="MAE (°C)")
            st.plotly_chart(fig_t, use_container_width=True)

        with col2:
            fig_w = px.line(
                df, x="lead_days", y="wind_mae", markers=True,
                title=f"Wind MAE (m/s) – {model} @ {city}"
            )
            fig_w.update_layout(xaxis_title="Lead (Tage)", yaxis_title="MAE (m/s)")
            st.plotly_chart(fig_w, use_container_width=True)

        with col3:
            fig_r = px.line(
                df, x="lead_days", y="rain_mae", markers=True,
                title=f"Regen MAE (mm) – {model} @ {city}"
            )
            fig_r.update_layout(xaxis_title="Lead (Tage)", yaxis_title="MAE (mm)")
            st.plotly_chart(fig_r, use_container_width=True)

        fig_m = px.bar(
            df, x="lead_days", y="weather_match_pct",
            title=f"Wetter-String: exakte Treffer (%) – {model} @ {city}"
        )
        fig_m.update_layout(xaxis_title="Lead (Tage)", yaxis_title="Trefferquote (%)")
        st.plotly_chart(fig_m, use_container_width=True)

        with st.expander("Details (Buckets)"):
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Zeitraum: {acc.get('from_date')} bis {acc.get('to_date')} – Modell: {acc.get('model')} – Stadt: {acc.get('city')}")
    else:
        st.info("Keine Accuracy-Daten im gewählten Zeitraum.")
except Exception as e:
    st.error(f"Fehler beim Laden der Accuracy: {e}")

st.divider()

# -----------------------------
# Rohdaten (optional zur Kontrolle)
# -----------------------------
with st.expander("Rohdaten (letzte 30 Tage)"):
    try:
        rows = load_latest_data(30, model=model, city=city)
        dfr = pd.DataFrame(rows)
        if not dfr.empty:
            # Sortierung: Datum aufsteigend, Lead absteigend (erst weit in der Zukunft, dann näher dran)
            dfr = dfr.sort_values(["target_date", "lead_days"], ascending=[True, False])
            st.dataframe(dfr, use_container_width=True, hide_index=True)
            st.caption(f"{len(dfr)} Zeilen")
        else:
            st.info("Keine Daten vorhanden.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Rohdaten: {e}")

# -----------------------------
# Logs
# -----------------------------
st.divider()
st.subheader("Logs (weather.log)")
colx, coly = st.columns([1, 1])

if "show_weather_logs" not in st.session_state:
    st.session_state.show_weather_logs = False

with colx:
    if st.button("📜 Load logs"):
        st.session_state.show_weather_logs = True
        st.rerun()

with coly:
    if st.button("🗑️ Delete all logs"):
        try:
            delete_weather_logs()
            st.session_state.show_weather_logs = False
            st.success("Alle Logs gelöscht.")
            st.rerun()
        except Exception as e:
            st.error(f"Fehler beim Löschen der Logs: {e}")

if st.session_state.show_weather_logs:
    try:
        logs = get_weather_logs(limit=2000, asc=True)
        dfl = pd.DataFrame(logs)
        if not dfl.empty:
            # Timestamps als naive UTC anzeigen (oder nach Bedarf lokalisieren)
            dfl["timestamp"] = pd.to_datetime(dfl["timestamp"], utc=True, errors="coerce")
            st.dataframe(dfl, use_container_width=True, hide_index=True)
            st.caption(f"{len(dfl)} Log-Einträge geladen.")
        else:
            st.info("Keine Logs vorhanden.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Logs: {e}")
