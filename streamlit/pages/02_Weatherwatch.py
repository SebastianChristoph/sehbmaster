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
st.caption("Vorhersage-Güte je Vorlauf (Lead-Tage)")

@st.cache_data(ttl=30)
def load_accuracy(frm: date, to: date, model: str, max_lead: int):
    return get_weather_accuracy(frm.isoformat(), to.isoformat(), model=model, max_lead=max_lead)

@st.cache_data(ttl=30)
def load_latest_data(days_back: int = 30, model: str = "default"):
    to = date.today()
    frm = to - timedelta(days=days_back)
    return get_weather_data(frm.isoformat(), to.isoformat(), model=model, lead_days=None, limit=10000)

# Controls
colA, colB, colC = st.columns([1,1,1])
with colA:
    days_back = st.slider("Zeitraum (Tage)", 7, 120, 35)
with colB:
    max_lead = st.slider("Max Lead", 0, 7, 7)
with colC:
    model = st.text_input("Modell", "default")

frm = date.today() - timedelta(days=days_back)
to  = date.today()

# Accuracy
try:
    acc = load_accuracy(frm, to, model=model, max_lead=max_lead)
    if acc and acc.get("buckets"):
        df = pd.DataFrame(acc["buckets"]).sort_values("lead_days")
        # MAE Liniendiagramme
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = px.line(df, x="lead_days", y="temp_mae", markers=True, title="Temp MAE (°C) nach Lead")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.line(df, x="lead_days", y="wind_mae", markers=True, title="Wind MAE (m/s) nach Lead")
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            fig = px.line(df, x="lead_days", y="rain_mae", markers=True, title="Regen MAE (mm) nach Lead")
            st.plotly_chart(fig, use_container_width=True)

        # Wetterstring-Accuracy
        fig2 = px.bar(df, x="lead_days", y="weather_match_pct",
                      title="Wetter-String: exakte Treffer (%) nach Lead")
        fig2.update_layout(yaxis_title="Trefferquote (%)")
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Details (Buckets)"):
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Keine Accuracy-Daten im gewählten Zeitraum.")
except Exception as e:
    st.error(f"Fehler beim Laden der Accuracy: {e}")

st.divider()

# Rohdaten (optional zur Kontrolle)
with st.expander("Rohdaten (letzte 30 Tage)"):
    try:
        rows = load_latest_data(30, model=model)
        dfr = pd.DataFrame(rows)
        if not dfr.empty:
            st.dataframe(
                dfr.sort_values(["target_date", "lead_days"], ascending=[True, False]),
                use_container_width=True, hide_index=True
            )
            st.caption(f"{len(dfr)} Zeilen")
        else:
            st.info("Keine Daten vorhanden.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Rohdaten: {e}")

# Logs
st.divider()
st.subheader("Logs (weather.log)")
colx, coly = st.columns([1,1])
with colx:
    if st.button("📜 Load logs"):
        try:
            logs = get_weather_logs(limit=2000, asc=True)
            dfl = pd.DataFrame(logs)
            if not dfl.empty:
                dfl["timestamp"] = pd.to_datetime(dfl["timestamp"], utc=True, errors="coerce")
                st.dataframe(dfl, use_container_width=True, hide_index=True)
            else:
                st.info("Keine Logs vorhanden.")
        except Exception as e:
            st.error(f"Fehler beim Laden der Logs: {e}")
with coly:
    if st.button("🗑️ Delete all logs"):
        try:
            delete_weather_logs()
            st.success("Alle Logs gelöscht.")
        except Exception as e:
            st.error(f"Fehler beim Löschen: {e}")
