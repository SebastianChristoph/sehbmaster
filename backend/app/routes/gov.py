# streamlit/pages/03_Regierungsflieger-Tracking.py
import os
import pandas as pd
import streamlit as st
import requests
from datetime import datetime
from typing import List, Dict, Any

# ---- vorhandene API-Client-Calls nutzen ----
from api_client import (
    get_gov_incidents,
    get_gov_incident_detail,
    patch_gov_incident_seen,
    delete_gov_incident,
    delete_gov_incident_article,
)

API_BASE = os.getenv("API_BASE", "http://backend:8000/api").rstrip("/")
API_KEY  = os.getenv("INGEST_API_KEY", "dev-secret")

st.set_page_config(page_title="Regierungsflieger-Tracking", page_icon="✈️", layout="wide")
st.title("✈️ Regierungsflieger-Tracking")
st.caption("Liste aller Pannen-Cluster (Incidents) aus der Datenbank. Oben: nur gesichtete. "
           "Unten: ungesichtete prüfen, als gesichtet markieren, Links löschen. "
           "Ganz unten: manuell neues Pannen-Cluster anlegen (ohne Datum).")

# ---------- kleine Helpers ----------
def _fmt_date(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        # "2025-11-08T10:00:00Z" / mit Offset
        iso = iso.replace("Z", "+00:00")
        d = datetime.fromisoformat(iso)
        return d.strftime("%d.%m.%Y")
    except Exception:
        return ""

def _post_new_incident(headline: str, links: List[str]) -> tuple[int, Any]:
    """
    Lokaler POST-Helper (falls du noch keinen create_* in api_client hast).
    articles: wir setzen Titel=Link, Source='Manuell', published_at=None.
    """
    url = f"{API_BASE}/gov/incidents"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }
    payload = {
        "headline": headline,
        # occurred_at weglassen (None) -> Backend kann selbst entscheiden
        "articles": [{"title": u, "source": "Manuell", "link": u} for u in links if u.strip()],
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        return r.status_code, (r.json() if r.headers.get("content-type","").startswith("application/json") else r.text)
    except Exception as e:
        return 0, str(e)

# ============== Abschnitt 1: nur GESICHTETE Vorfälle (Tabelle) ==============
st.subheader("Gesichtete Vorfälle")

try:
    seen_incidents = get_gov_incidents(seen=True, limit=500)  # nur seen=true
except Exception as e:
    seen_incidents = []
    st.error(f"Fehler beim Laden (gesehen): {e}")

rows_seen: List[Dict[str, Any]] = []
for inc in seen_incidents or []:
    # Detail laden, um Quellen/Links zu zeigen
    det = {}
    try:
        det = get_gov_incident_detail(inc["id"])
    except Exception:
        det = {}

    links = [a.get("link","") for a in det.get("articles", [])]
    date_str = _fmt_date(inc.get("occurred_at") or inc.get("created_at"))
    rows_seen.append({
        "Datum": date_str,
        "Titel": inc.get("headline",""),
        "Quellen": "\n".join(links) if links else "",
    })

if rows_seen:
    df_seen = pd.DataFrame(rows_seen)
    st.dataframe(
        df_seen,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Datum": st.column_config.TextColumn(width="small"),
            "Titel": st.column_config.TextColumn(width="large"),
            "Quellen": st.column_config.TextColumn(width="large"),
        },
    )
else:
    st.info("Keine gesichteten Vorfälle vorhanden.")

st.markdown("---")

# ============== Abschnitt 2: UNGESICHTETE sichten/aufräumen ==============
st.subheader("Ungesichtete sichten")

try:
    unseen_incidents = get_gov_incidents(seen=False, limit=500)
except Exception as e:
    unseen_incidents = []
    st.error(f"Fehler beim Laden (ungesichtet): {e}")

st.caption(f"Ungesichtet: {len(unseen_incidents or [])}")

if unseen_incidents:
    for inc in unseen_incidents:
        det = {}
        try:
            det = get_gov_incident_detail(inc["id"])
        except Exception as e:
            st.warning(f"Details konnten nicht geladen werden (ID {inc['id']}): {e}")
            continue

        date_str = _fmt_date(inc.get("occurred_at") or inc.get("created_at"))
        header = f"{inc.get('headline','(ohne Titel)')} • {date_str or 'ohne Datum'} • Quellen: {len(det.get('articles',[]))}"
        with st.expander(header, expanded=False):
            # Liste der Quellen
            if det.get("articles"):
                for art in det["articles"]:
                    cols = st.columns([0.75, 0.15, 0.10])
                    with cols[0]:
                        st.write(f"- [{art.get('title') or art.get('link')}]({art.get('link')})")
                    with cols[1]:
                        st.caption(art.get("source") or "")
                    with cols[2]:
                        if st.button("Link entfernen", key=f"rm-{inc['id']}-{art['id']}", help="Entfernt diesen Artikel-Link aus dem Vorfall."):
                            try:
                                delete_gov_incident_article(inc["id"], art["id"])
                                st.success("Link entfernt.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Fehler beim Entfernen: {e}")
            else:
                st.write("Keine Artikel verknüpft.")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Als gesichtet markieren", key=f"seen-{inc['id']}"):
                    try:
                        patch_gov_incident_seen(inc["id"], True)
                        st.success("Vorfall als gesichtet markiert.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fehler: {e}")
            with c2:
                if st.button("Vorfall löschen", key=f"del-{inc['id']}", type="secondary"):
                    try:
                        delete_gov_incident(inc["id"])
                        st.success("Vorfall gelöscht.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fehler: {e}")
else:
    st.info("Keine ungesichteten Vorfälle vorhanden.")

st.markdown("---")

# ============== Abschnitt 3: Manuell neues Pannen-Cluster hinzufügen ==============
st.subheader("Manuell neues Pannen-Cluster hinzufügen")

with st.form("manual_add"):
    title = st.text_input("Titel des Pannen-Clusters", placeholder="z. B. Regierungsflieger mit Defekt – Außenminister muss umsteigen")
    links_text = st.text_area("Quellen-Links (je Zeile ein Link)", height=120, placeholder="https://…\nhttps://…")
    submitted = st.form_submit_button("Anlegen")

    if submitted:
        links = [ln.strip() for ln in (links_text or "").splitlines() if ln.strip()]
        if not title or not links:
            st.error("Bitte Titel und mindestens einen Link angeben.")
        else:
            code, data = _post_new_incident(title, links)
            if code in (200, 201):
                st.success("Vorfall angelegt.")
                st.rerun()
            else:
                # Backend kann z. B. zurückgeben: „Es existiert bereits eine Panne an diesem Tag“
                msg = data if isinstance(data, str) else str(data)
                st.error(f"Fehler ({code}): {msg[:400]}")
