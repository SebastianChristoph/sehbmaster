import os
import pandas as pd
import streamlit as st
from api_client import get_status

st.set_page_config(page_title="sehbmaster – Home", page_icon="🏠", layout="wide")
st.title("🏠 Home")
st.caption("Alle Einträge aus **status.status**")

@st.cache_data(ttl=10)
def load_status():
    return get_status()

col_btn, _ = st.columns([1, 5])
if col_btn.button("Neu laden"):
    load_status.clear()
    st.rerun()

try:
    rows = load_status()
    if rows:
        df = pd.DataFrame(rows)
        # optionale Spaltenreihenfolge
        cols = [c for c in ["raspberry", "status", "message"] if c in df.columns]
        if cols:
            df = df[cols]
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.success(f"{len(df)} Einträge geladen.")
    else:
        st.info("Keine Einträge vorhanden.")
except Exception as e:
    st.error(f"Fehler beim Laden: {e}")
