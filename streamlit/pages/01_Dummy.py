import pandas as pd
import streamlit as st
from api_client import get_dummy, create_dummy

st.set_page_config(page_title="sehbmaster – Dummy", page_icon="🧪", layout="wide")
st.title("🧪 Dummy")
st.caption("Alle Einträge aus **dummy.dummy_table** und neue hinzufügen")

@st.cache_data(ttl=10)
def load_dummy():
    return get_dummy()

with st.form("dummy_form", clear_on_submit=True):
    msg = st.text_input("Message", "")
    submitted = st.form_submit_button("Hinzufügen")
    if submitted:
        if not msg.strip():
            st.warning("Bitte eine Nachricht eingeben.")
        else:
            try:
                create_dummy(msg.strip())
                load_dummy.clear()
                st.success("Gespeichert.")
                st.rerun()
            except Exception as e:
                st.error(f"Fehler beim Speichern: {e}")

col_btn, _ = st.columns([1, 5])
if col_btn.button("Neu laden"):
    load_dummy.clear()
    st.rerun()

try:
    rows = load_dummy()
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.success(f"{len(df)} Einträge geladen.")
    else:
        st.info("Keine Einträge vorhanden.")
except Exception as e:
    st.error(f"Fehler beim Laden: {e}")
