# streamlit/pages/03_Regierungsflieger-Tracking.py
import pandas as pd
import streamlit as st
from datetime import datetime
from api_client import (
    get_gov_incidents,
    get_gov_incident_detail,
    patch_gov_incident_seen,
    delete_gov_incident,
    delete_gov_incident_article,
    create_gov_incident,
    wipe_gov,
    ApiError,
)

st.set_page_config(page_title="Regierungsflieger-Tracking", page_icon="✈️", layout="wide")
st.title("✈️ Regierungsflieger-Tracking")
st.caption(
    "Liste aller Pannen-Cluster (Incidents) aus der Datenbank. "
    "Oben: bereits **gesichtete** Vorfälle (lesend). "
    "Darunter: **ungesichtete** prüfen → als gesichtet markieren, Links entfernen oder Vorfall löschen. "
    "Unten: **manuell** ein neues Pannen-Cluster anlegen (ohne Datum)."
)

# ---------- Helpers ----------
def _fmt_date_iso_to_de(iso_str: str | None) -> str:
    if not iso_str:
        return ""
    try:
        # ISO mit/ohne Z/Offset -> nur Datum
        s = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.strftime("%d.%m.%Y")
    except Exception:
        return iso_str

def _reload():
    st.session_state.pop("seen_rows", None)
    st.session_state.pop("unseen_rows", None)
    st.session_state.pop("details_cache", None)
    st.rerun()

@st.cache_data(ttl=10)
def _load_seen():
    try:
        return get_gov_incidents(seen=True, limit=1000, offset=0)
    except Exception as e:
        raise e

@st.cache_data(ttl=10)
def _load_unseen():
    try:
        return get_gov_incidents(seen=False, limit=1000, offset=0)
    except Exception as e:
        raise e

@st.cache_data(ttl=10)
def _load_detail(incident_id: int):
    return get_gov_incident_detail(incident_id)

# ---------- Abschnitt A: Gesichtete Vorfälle (nur Anzeige, tabellarisch) ----------
st.subheader("Gesichtete Vorfälle")

try:
    seen_rows = _load_seen()
except Exception as e:
    st.error(f"Fehler beim Laden der gesichteten Vorfälle: {e}")
    seen_rows = []

if seen_rows:
    # Wir holen Detaildaten für die Links (Quellen-Anzahl & Liste)
    table_records = []
    for r in seen_rows:
        try:
            det = _load_detail(r["id"])
            links = [a.get("link", "") for a in det.get("articles", [])]
        except Exception:
            links = []
        table_records.append({
            "Datum": _fmt_date_iso_to_de(r.get("occurred_at")),
            "Titel": r.get("headline", ""),
            "Quellen": "\n".join(links) if links else "",
        })

    df_seen = pd.DataFrame(table_records, columns=["Datum", "Titel", "Quellen"])
    st.dataframe(
        df_seen,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Datum": st.column_config.TextColumn("Datum", width="small"),
            "Titel": st.column_config.TextColumn("Titel"),
            "Quellen": st.column_config.TextColumn("Quellen"),
        },
    )
else:
    st.info("Keine gesichteten Vorfälle vorhanden.")

st.divider()

# ---------- Abschnitt B: Ungesichtete sichten ----------
st.subheader("Ungesichtete sichten")

reload_col, _ = st.columns([1, 5])
if reload_col.button("Neu laden"):
    _reload()

try:
    unseen_rows = _load_unseen()
except Exception as e:
    st.error(f"Fehler beim Laden der ungesichteten Vorfälle: {e}")
    unseen_rows = []

st.caption(f"Vorfälle: {len(unseen_rows)}")

for r in unseen_rows:
    # Detaildaten holen (Artikel/Links)
    try:
        det = _load_detail(r["id"])
        arts = det.get("articles", [])
    except Exception as e:
        st.error(f"Fehler beim Laden der Details für ID {r['id']}: {e}")
        arts = []

    # Kopfzeile
    title_line = f"{r.get('headline','')}"
    meta_line = f"{_fmt_date_iso_to_de(r.get('occurred_at'))} • Quellen: {len(arts)}"
    with st.expander(f"{title_line} • {meta_line}", expanded=False):
        # Tabelle der Artikel/Quellen
        art_rows = [{
            "ID": a.get("id"),
            "Quelle": a.get("source"),
            "Titel": a.get("title"),
            "Link": a.get("link"),
            "Published": a.get("published_at"),
        } for a in arts]

        if art_rows:
            df = pd.DataFrame(art_rows, columns=["ID", "Quelle", "Titel", "Link", "Published"])
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ID": st.column_config.NumberColumn("ID", width="small"),
                    "Quelle": st.column_config.TextColumn("Quelle", width="small"),
                    "Titel": st.column_config.TextColumn("Titel"),
                    "Link": st.column_config.TextColumn("Link"),
                    "Published": st.column_config.TextColumn("Published", width="small"),
                },
            )
        else:
            st.info("Keine Quellen eingetragen.")

        # Aktionen
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Als gesichtet markieren", key=f"seen_{r['id']}"):
                try:
                    patch_gov_incident_seen(r["id"], True)
                    _reload()
                except ApiError as e:
                    st.error(f"Fehler: {e}")
        with c2:
            if st.button("Vorfall löschen", type="primary", key=f"del_{r['id']}"):
                try:
                    delete_gov_incident(r["id"])
                    _reload()
                except ApiError as e:
                    st.error(f"Fehler: {e}")
        with c3:
            # Einzelnen Artikel entfernen
            if arts:
                remove_id = st.selectbox(
                    "Link entfernen (Artikel-ID)",
                    options=[a["id"] for a in arts],
                    format_func=lambda aid: next((a["title"] for a in arts if a["id"] == aid), str(aid)),
                    key=f"rm_sel_{r['id']}",
                )
                if st.button("Entfernen", key=f"rm_btn_{r['id']}"):
                    try:
                        delete_gov_incident_article(r["id"], int(remove_id))
                        _reload()
                    except ApiError as e:
                        st.error(f"Fehler: {e}")

st.divider()

# ---------- Abschnitt C: Manuell neues Pannen-Cluster hinzufügen ----------
st.subheader("Manuell neues Pannen-Cluster hinzufügen")

with st.form("manual_add"):
    headline = st.text_input("Titel / Headline", placeholder="z. B. Regierungsflieger-Panne bei Reise X")
    st.caption("Links der Quellen (je Zeile ein Link). Titel/Quelle werden automatisch vorbelegt.")
    links_multiline = st.text_area("Quellen-Links (je Zeile)", height=120, placeholder="https://…\nhttps://…")
    submit = st.form_submit_button("Anlegen")

    if submit:
        links = [ln.strip() for ln in (links_multiline or "").splitlines() if ln.strip()]
        if not headline or not links:
            st.error("Bitte Headline und mindestens einen Link angeben.")
        else:
            # Artikel-Payloads minimal erzeugen
            articles = [{"title": ln, "source": "Manuell", "link": ln, "published_at": None} for ln in links]
            try:
                created = create_gov_incident(headline=headline, articles=articles)
                st.success(f"Vorfall angelegt (ID {created.get('id')}).")
                _reload()
            except ApiError as e:
                st.error(f"Fehler beim Anlegen: {e}")

st.divider()

# ---------- Abschnitt D: ⚠️ Datenbank leeren ----------
with st.expander("Datenbank-Inhalt löschen (gefährlich!)"):
    st.caption("Löscht **alle** gov-Incidents inkl. Artikel.")
    if st.button("Alles löschen (Wipe)", type="primary"):
        try:
            res = wipe_gov(confirm=True)
            st.success(f"Wipe: {res}")
            _reload()
        except ApiError as e:
            st.error(f"Fehler beim Wipe: {e}")
