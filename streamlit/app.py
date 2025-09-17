import pandas as pd
import streamlit as st
from api_client import get_status

st.set_page_config(page_title="Sehbmaster – Home", page_icon="🏠", layout="wide")
st.title("👋 Hallo Sehb")

@st.cache_data(ttl=10)
def load_status():
    return get_status()

try:
    rows = load_status()
    if rows:
        df = pd.DataFrame(rows)

        # Reihenfolge der Spalten (falls vorhanden)
        cols = [c for c in ["raspberry", "status", "message"] if c in df.columns]
        if cols:
            df = df[cols]

        # --- NEU: Icon-Spalte basierend auf status ---
        def status_to_icon(val: str) -> str:
            if not isinstance(val, str):
                return "⚪️"
            v = val.strip().lower()
            if v == "idle":
                return "🟡"
            if v == "working":
                return "🟢"
            if v == "error":
                return "🔴"
            return "⚪️"  # Fallback

        icon_col = df["status"].map(status_to_icon) if "status" in df.columns else "⚪️"
        df.insert(0, " ", icon_col)  # Icon-Spalte ganz links

        # Optional: hübsche Spaltenüberschriften
        df = df.rename(columns={
            "raspberry": "Raspberry",
            "status": "Status",
            "message": "Message",
        })

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                " ": st.column_config.TextColumn(" ", width="small"),  # Icon schmal
            },
        )
    else:
        st.info("Keine Einträge vorhanden.")
except Exception as e:
    st.error(f"Fehler beim Laden: {e}")


col_btn, _ = st.columns([1, 5])
if col_btn.button("Neu laden"):
    load_status.clear()
    st.rerun()
