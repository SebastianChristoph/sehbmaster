# streamlit/pages/03_Regierungsflieger-Tracking.py
import re
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

from api_client import (
    get_gov_incidents, get_gov_incident_detail, post_gov_incident,
    post_gov_article, patch_gov_incident_seen, delete_gov_incident,
    delete_gov_incident_article, wipe_gov, ApiError
)

st.set_page_config(page_title="Regierungsflieger-Tracking", page_icon="✈️", layout="wide")
st.title("✈️ Regierungsflieger-Tracking")

st.caption("Liste aller Pannen-Cluster (Incidents) aus der Datenbank. Markiere als „gesichtet“, lösche Vorfälle oder entferne einzelne Links. Unten kannst du manuell ein neues Pannen-Cluster hinzufügen.")

# ----------------------------- Helper -----------------------------
@st.cache_data(ttl=10)
def load_incidents(seen: bool | None):
    rows = get_gov_incidents(seen=seen, limit=1000, offset=0)
    # sortiere: occurred_at DESC, fallback created_at
    def sort_key(r):
        oa = r.get("occurred_at")
        ca = r.get("created_at")
        return (oa or ca or "")
    rows = sorted(rows, key=sort_key, reverse=True)
    return rows

def iso_or_none(date_str: str, time_str: str | None) -> str | None:
    date_str = (date_str or "").strip()
    if not date_str:
        return None
    t = (time_str or "00:00").strip()
    try:
        # als naive Local annehmen und in UTC umwandeln? Simpler: ISO lassen.
        return f"{date_str}T{t}:00Z" if len(t) == 5 else date_str
    except Exception:
        return None

def fetch_title(url: str) -> str:
    """Sehr leichter Auto-Titel-Fetch (optional)."""
    try:
        resp = requests.get(url, timeout=6, headers={"User-Agent": "GovWatch/streamlit"})
        resp.raise_for_status()
        html = resp.text
        m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
        if m:
            title = re.sub(r"\s+", " ", m.group(1)).strip()
            return title[:200] or url
    except Exception:
        pass
    return url

# --------------------------- Filterleiste -------------------------
col_a, col_b, col_c = st.columns([1,1,2])
with col_a:
    seen_filter = st.selectbox("Filter", options=["Alle", "Ungesichtet", "Gesichtet"], index=1)
    seen_value = {"Alle": None, "Ungesichtet": False, "Gesichtet": True}[seen_filter]
with col_b:
    reload_clicked = st.button("Neu laden")

if reload_clicked:
    load_incidents.clear()

rows = load_incidents(seen_value)

st.subheader(f"Vorfälle: {len(rows)}")

# --------------------------- Liste/Expander -----------------------
if not rows:
    st.info("Keine Incidents vorhanden.")
else:
    for r in rows:
        inc_id = r["id"]
        subtitle = f'{r["headline"]}'
        meta_bits = []
        if r.get("occurred_at"):
            meta_bits.append(r["occurred_at"])
        if "articles_count" in r:
            meta_bits.append(f'Quellen: {r["articles_count"]}')
        with st.expander(" • ".join([subtitle] + meta_bits)):
            try:
                det = get_gov_incident_detail(inc_id)
            except Exception as e:
                st.error(f"Fehler beim Laden der Details: {e}")
                continue

            articles = det.get("articles", [])
            if articles:
                df = pd.DataFrame([{
                    "ID": a.get("id"),
                    "Quelle": a.get("source"),
                    "Titel": a.get("title"),
                    "Link": a.get("link"),
                    "Published": a.get("published_at"),
                } for a in articles])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.write("Keine Artikel verknüpft.")

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Als gesichtet markieren", key=f"seen_{inc_id}"):
                    try:
                        patch_gov_incident_seen(inc_id, True)
                        load_incidents.clear()
                        st.success("Incident auf gesichtet gesetzt.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fehler: {e}")
            with c2:
                # einzelnes Entfernen
                if articles:
                    to_del = st.selectbox("Artikel entfernen", options=[a["id"] for a in articles], key=f"sel_{inc_id}",
                                          format_func=lambda aid: next((a["title"] for a in articles if a["id"]==aid), f"#{aid}"))
                    if st.button("Entfernen", key=f"rm_{inc_id}"):
                        try:
                            delete_gov_incident_article(inc_id, int(to_del))
                            load_incidents.clear()
                            st.success("Artikel entfernt.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fehler: {e}")
            with c3:
                if st.button("Vorfall löschen", key=f"del_{inc_id}", type="secondary"):
                    try:
                        delete_gov_incident(inc_id)
                        load_incidents.clear()
                        st.success("Incident gelöscht.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fehler: {e}")

st.divider()

# ----------------------- Manuell hinzufügen -----------------------
st.subheader("Manuell neues Pannen-Cluster hinzufügen")

with st.form("form_manual_incident"):
    headline = st.text_input("Überschrift (Headline)", placeholder="Regierungsflieger-Panne in XYZ")
    c1, c2 = st.columns(2)
    with c1:
        date_str = st.text_input("Pannen-Datum (YYYY-MM-DD)", placeholder="2025-11-08")
    with c2:
        time_str = st.text_input("Zeit (HH:MM, optional)", placeholder="10:00")

    st.caption("Links (eine URL pro Zeile). Optional: „Quelle | Titel | Published-ISO“ pro Zeile.")
    links_raw = st.text_area("Links", height=140, placeholder="https://beispiel.de/artikel-1\nhttps://beispiel.de/artikel-2")

    autofetch = st.checkbox("Titel automatisch abrufen (best effort)")
    submitted = st.form_submit_button("Cluster anlegen")

    if submitted:
        if not headline.strip():
            st.error("Bitte eine Headline angeben.")
        else:
            occurred_at = iso_or_none(date_str, time_str)
            articles_payload: list[dict] = []
            for line in links_raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Format: URL  ODER  "Quelle | Titel | Published"
                if " | " in line:
                    parts = [p.strip() for p in line.split(" | ")]
                    url = parts[0]
                    source = parts[1] if len(parts) > 1 else "Manuell"
                    title = parts[2] if len(parts) > 2 else (fetch_title(url) if autofetch else url)
                    published = parts[3] if len(parts) > 3 else None
                else:
                    url = line
                    source = "Manuell"
                    title = fetch_title(url) if autofetch else url
                    published = None

                articles_payload.append({
                    "title": title[:500],
                    "source": source[:200],
                    "link": url,
                    "published_at": published,
                })

            if not articles_payload:
                st.error("Bitte mindestens einen Link angeben.")
            else:
                try:
                    res = post_gov_incident(headline=headline.strip(), occurred_at=occurred_at, articles=articles_payload)
                    st.success(f"Incident angelegt (ID {res.get('id')}).")
                    load_incidents.clear()
                    st.rerun()
                except ApiError as e:
                    msg = str(e)
                    # Falls Backend den „selber Tag existiert schon“-Check macht:
                    if "existiert bereits" in msg or "already exists" in msg:
                        st.warning(msg)
                    else:
                        st.error(f"Fehler beim Anlegen: {msg}")

st.divider()

# ----------------------- Danger Zone (Wipe) -----------------------
with st.expander("🧨 Danger Zone: Datenbank-Inhalte für GOV löschen"):
    st.caption("Löscht ALLE gov_incidents und zugehörige Artikel. Erfordert API-Key.")
    colw1, colw2 = st.columns([1,2])
    with colw1:
        confirm1 = st.checkbox("Ich weiß, was ich tue.")
    with colw2:
        confirm2 = st.text_input("Tippe 'WIPE' zur Bestätigung").strip().upper() == "WIPE"

    if st.button("Alles löschen (Wipe)"):
        if not (confirm1 and confirm2):
            st.error("Bitte beide Bestätigungen ausführen.")
        else:
            try:
                res = wipe_gov(confirm=True)
                load_incidents.clear()
                st.success("Wipe ausgeführt.")
                st.rerun()
            except Exception as e:
                st.error(f"Wipe fehlgeschlagen: {e}")
