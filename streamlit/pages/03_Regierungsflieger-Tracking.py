import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from api_client import (
    get_gov_incidents,
    get_gov_incident_detail,
    post_gov_incident,
    post_gov_article,
    patch_gov_incident_seen,
    delete_gov_incident,
    delete_gov_incident_article,
    ApiError,
)

st.set_page_config(page_title="Regierungsflieger-Tracking", page_icon="✈️", layout="wide")
st.title("✈️ Regierungsflieger-Tracking")

st.caption(
    "Liste der Pannen-Cluster (Incidents) aus der Datenbank. "
    "Oben: **Gesichtete** tabellarisch. Unten: **Ungesichtete sichten** (prüfen, als gesichtet markieren, Links entfernen, löschen). "
    "Ganz unten kannst du **manuell** ein Pannen-Cluster hinzufügen."
)

# -------------------- Helpers --------------------

def _fmt_date_iso_to_de(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        # accept "YYYY-MM-DDTHH:MM:SSZ" or with offset
        s = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.strftime("%d.%m.%Y")
    except Exception:
        return ""

def _iso_from_ddmmyyyy(s: str) -> str | None:
    """
    expects dd-mm-yyyy (with dashes) and returns ISO 'YYYY-MM-DDT00:00:00Z'
    """
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    m = re.match(r"^(\d{2})-(\d{2})-(\d{4})$", s)
    if not m:
        return None
    dd, mm, yyyy = m.groups()
    try:
        dt = datetime(int(yyyy), int(mm), int(dd), 0, 0, 0, tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return None

def _hostname(u: str) -> str:
    try:
        return urlparse(u).netloc or u
    except Exception:
        return u

def _links_from_detail(detail: dict) -> list[str]:
    links: list[str] = []
    for a in detail.get("articles", []):
        u = a.get("link")
        if isinstance(u, str) and u:
            links.append(u)
    return links

# Cache to avoid hammering API on each rerun
@st.cache_data(ttl=10)
def _load_incident_ids(seen: bool | None) -> list[dict]:
    return get_gov_incidents(seen=seen, limit=500, offset=0)

@st.cache_data(ttl=10)
def _load_incident_detail(incident_id: int) -> dict:
    return get_gov_incident_detail(incident_id)

def _refresh_all():
    _load_incident_ids.clear()
    _load_incident_detail.clear()

# ====================== A) Gesichtete: Tabelle ======================

st.subheader("Gesichtete Vorfälle")

seen_rows = _load_incident_ids(seen=True)

# Build table: we need links, so fetch detail for each id
table_records: list[dict] = []
for r in seen_rows:
    det = _load_incident_detail(r["id"])
    links = _links_from_detail(det)
    table_records.append({
        "Datum": _fmt_date_iso_to_de(r.get("occurred_at")),
        "Titel": r.get("headline", ""),
        # DataFrame shows single-line text better; join with • to avoid newline issues in cells
        "Quellen": " • ".join(links) if links else "",
    })

if table_records:
    df_seen = pd.DataFrame(table_records, columns=["Datum", "Titel", "Quellen"])
    st.dataframe(df_seen, use_container_width=True, hide_index=True)
else:
    st.info("Keine **gesichteten** Vorfälle vorhanden.")

st.divider()

# ====================== B) Ungesichtete sichten ======================

st.subheader("Ungesichtete sichten")

unseen_rows = _load_incident_ids(seen=False)
st.caption(f"{len(unseen_rows)} ungesichtete Vorfälle")

if not unseen_rows:
    st.success("Es gibt aktuell keine ungesichteten Vorfälle.")
else:
    for r in unseen_rows:
        det = _load_incident_detail(r["id"])
        links = _links_from_detail(det)
        with st.expander(f"{r['headline']} • { _fmt_date_iso_to_de(r.get('occurred_at')) } • Quellen: {len(links)}", expanded=True):
            # Links als Liste
            if links:
                for u in links:
                    st.markdown(f"- {u}")
            else:
                st.write("Keine Quellen vorhanden.")

            # Auswahl einzelner Artikel zum Entfernen
            if det.get("articles"):
                id_to_title = {a["id"]: a["title"] for a in det["articles"]}
                article_choices = list(id_to_title.keys())
                rm_ids = st.multiselect(
                    "Links aus diesem Vorfall entfernen",
                    options=article_choices,
                    format_func=lambda aid: f"{id_to_title.get(aid, aid)}",
                    key=f"rm_{r['id']}",
                )
            else:
                rm_ids = []

            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                if st.button("Als gesichtet markieren", key=f"s_{r['id']}"):
                    try:
                        patch_gov_incident_seen(r["id"], True)
                        st.success("Vorfall als gesichtet markiert.")
                        _refresh_all()
                        st.rerun()
                    except ApiError as e:
                        st.error(f"Fehler: {e}")
            with col2:
                if st.button("Vorfall löschen", key=f"d_{r['id']}"):
                    try:
                        delete_gov_incident(r["id"])
                        st.warning("Vorfall gelöscht.")
                        _refresh_all()
                        st.rerun()
                    except ApiError as e:
                        st.error(f"Fehler: {e}")
            with col3:
                if st.button("Ausgewählte Links entfernen", key=f"rm_btn_{r['id']}", disabled=(len(rm_ids)==0)):
                    try:
                        for aid in rm_ids:
                            delete_gov_incident_article(r["id"], int(aid))
                        st.success(f"{len(rm_ids)} Link(s) entfernt.")
                        _refresh_all()
                        st.rerun()
                    except ApiError as e:
                        st.error(f"Fehler: {e}")

st.divider()

# ====================== C) Manuell hinzufügen ======================

st.subheader("Manuell neues Pannen-Cluster hinzufügen")

with st.form("manual_add"):
    headline = st.text_input("Titel / Headline", placeholder="z. B. Panne auf Langstrecke A350")
    date_str = st.text_input("Datum (dd-mm-yyyy) – ohne Uhrzeit", placeholder="z. B. 08-11-2025")
    links_text = st.text_area(
        "Quellen-Links (je Zeile)",
        placeholder="https://beispiel.de/artikel-1\nhttps://andere-quelle.de/meldung",
        height=140,
    )
    submitted = st.form_submit_button("Anlegen")

if submitted:
    if not headline.strip():
        st.error("Bitte einen **Titel** angeben.")
    else:
        iso = _iso_from_ddmmyyyy(date_str)
        if not iso:
            st.error("Bitte ein gültiges Datum im Format **dd-mm-yyyy** angeben.")
        else:
            # Artikel aus den Linkzeilen bauen
            rows: list[dict] = []
            for line in links_text.splitlines():
                u = line.strip()
                if not u:
                    continue
                rows.append({
                    "title": u,                  # minimal: Titel = Link (später gern upgraden)
                    "source": _hostname(u),      # Quelle = Hostname
                    "link": u,
                    "published_at": None,        # optional
                })
            if not rows:
                st.error("Bitte mindestens **einen Link** angeben.")
            else:
                try:
                    # nutzt deinen API-Client (mit occurred_at)
                    created = post_gov_incident(headline=headline.strip(), occurred_at=iso, articles=rows)
                    # Bei Erfolg: Cache leeren und UI refreshen
                    _refresh_all()
                    st.success(f"Vorfall angelegt (ID {created.get('id')}).")
                    st.rerun()
                except ApiError as e:
                    # Falls Backend den Tages-Duplikat-Check hat, kommt hier die passende Meldung an
                    st.error(f"Anlegen fehlgeschlagen: {e}")
st.divider()

# ---------- 4) ⚠️ Datenbank leeren ----------
st.subheader("⚠️ Datenbank leeren")
st.caption("Löscht **alle** Regierungsflieger-Daten (Incidents + Artikel). Nicht rückgängig zu machen!")

left, right = st.columns([1, 3])
with left:
    really = st.checkbox("Ja, alles löschen", value=False)
with right:
    if st.button("Datenbank löschen", type="secondary", disabled=not really):
        try:
            res = wipe_gov(confirm=True)
            st.warning("Alle Daten gelöscht.")
            st.json(res)
            st.rerun()
        except ApiError as e:
            st.error(f"Wipe fehlgeschlagen: {e}")