import os
import requests
import pandas as pd
import streamlit as st

# ===== Konfiguration =====
API_BASE = (os.getenv("API_BASE", "http://backend:8000/api")).rstrip("/")

st.set_page_config(page_title="sehbmaster", page_icon="🧰", layout="wide")

# ===== API-Helper =====
def json_or_raise(resp: requests.Response):
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Versuche, Text/JSON hilfreich zu zeigen
        txt = ""
        try:
            txt = resp.text[:500]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {resp.status_code}: {txt or e}") from e
    try:
        return resp.json()
    except Exception:
        raise RuntimeError("Antwort war kein JSON")

def api_get(path: str):
    r = requests.get(f"{API_BASE}{path}", timeout=10)
    return json_or_raise(r)

def api_post(path: str, payload: dict):
    r = requests.post(f"{API_BASE}{path}", json=payload, timeout=10)
    return json_or_raise(r)

# ===== Datenabfragen (mit Cache) =====
@st.cache_data(ttl=10)
def load_status():
    return api_get("/status")  # -> [{raspberry, status}, ...]

@st.cache_data(ttl=10)
def load_dummy():
    return api_get("/dummy")   # -> [{message}, ...]

def add_dummy(message: str):
    return api_post("/dummy", {"message": message})

# ===== Sidebar / Navigation =====
st.sidebar.title("sehbmaster")
page = st.sidebar.radio("Menü", ["Home", "Dummy"], index=0)

# ===== Seiten =====
if page == "Home":
    st.title("Home")
    st.caption("Alle Einträge aus **status.status**")

    col_btn, _ = st.columns([1, 5])
    if col_btn.button("Neu laden"):
        load_status.clear()  # Cache leeren
        st.experimental_rerun()

    try:
        rows = load_status()
        if rows:
            df = pd.DataFrame(rows)
            # Sortiere optional nach raspberry
            if "raspberry" in df.columns:
                df = df.sort_values("raspberry")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.success(f"{len(df)} Einträge geladen.")
        else:
            st.info("Keine Einträge vorhanden.")
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")

elif page == "Dummy":
    st.title("Dummy")
    st.caption("Alle Einträge aus **dummy.dummy_table** und neues Hinzufügen")

    # Formular
    with st.form("dummy_form", clear_on_submit=True):
        msg = st.text_input("Message", "")
        submitted = st.form_submit_button("Hinzufügen")
        if submitted:
            if not msg.strip():
                st.warning("Bitte eine Nachricht eingeben.")
            else:
                try:
                    add_dummy(msg.strip())
                    # Nach erfolgreichem POST neu laden
                    load_dummy.clear()
                    st.success("Gespeichert.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Fehler beim Speichern: {e}")

    col_btn, _ = st.columns([1, 5])
    if col_btn.button("Neu laden"):
        load_dummy.clear()
        st.experimental_rerun()

    # Tabelle
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
