# streamlit/pages/03_Govwatch.py
import re
from datetime import datetime, timezone
from urllib.parse import urlparse
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px

from api_client import (
    get_gov_incidents,
    get_gov_incident_detail,
    post_gov_incident,
    patch_gov_incident_seen,
    delete_gov_incident,
    delete_gov_incident_article,
    wipe_gov,
    ApiError,
)

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(page_title="Government Aircraft – Incident Tracking", page_icon="✈️", layout="wide")
st.title("✈️ Government Aircraft – Incident Tracking")
st.caption(
    "Top: table of **seen** incidents. Below: **Review unreviewed** (mark as seen, delete incident, or remove links). "
    "At the bottom you can **manually add** an incident and **wipe** the database if needed."
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _fmt_date_iso_to_de(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        s = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.strftime("%d.%m.%Y")
    except Exception:
        return ""

def _iso_from_ddmmyyyy(s: str) -> str | None:
    """
    Convert 'dd-mm-yyyy' to ISO 'YYYY-MM-DDT00:00:00Z'
    """
    if not isinstance(s, str):
        return None
    s = s.strip()
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

def _year_from_iso(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        s = iso.replace("Z", "+00:00")
        return datetime.fromisoformat(s).year
    except Exception:
        return None

# ---------------------------------------------------------------------
# Small caching to avoid hammering the API on each rerun
# ---------------------------------------------------------------------
@st.cache_data(ttl=10)
def _load_incident_ids(seen: bool | None) -> list[dict]:
    return get_gov_incidents(seen=seen, limit=500, offset=0)

@st.cache_data(ttl=10)
def _load_incident_detail(incident_id: int) -> dict:
    return get_gov_incident_detail(incident_id)

def _refresh_all():
    _load_incident_ids.clear()
    _load_incident_detail.clear()

# ---------------------------------------------------------------------
# A) Seen incidents (table) + Bar chart “incidents per year”
# ---------------------------------------------------------------------
try:
    seen_rows = _load_incident_ids(seen=True)
except ApiError as e:
    st.error(f"api_client.ApiError: {e}")
    st.stop()

st.subheader(f"Seen incidents ({len(seen_rows)})")

table_records: list[dict] = []
years: list[int] = []

for r in seen_rows:
    det = _load_incident_detail(r["id"])
    links = _links_from_detail(det)
    table_records.append(
        {
            "Date": _fmt_date_iso_to_de(r.get("occurred_at")),
            "Title": r.get("headline", ""),
            # join to keep single-line cells (newlines render poorly in DataFrame)
            "Sources": " • ".join(links) if links else "",
        }
    )
    y = _year_from_iso(r.get("occurred_at"))
    if y is not None:
        years.append(y)

if table_records:
    df_seen = pd.DataFrame(table_records, columns=["Date", "Title", "Sources"])
    st.dataframe(df_seen, use_container_width=True, hide_index=True)
else:
    st.info("No **seen** incidents yet.")

# ---- Bar chart: incidents per year (Plotly) ----
st.markdown("#### Incidents per year")
if years:
    counts = Counter(years)
    df_counts = pd.DataFrame({"Year": sorted(counts.keys()), "Incidents": [counts[y] for y in sorted(counts.keys())]})
    fig = px.bar(df_counts, x="Year", y="Incidents", text="Incidents")
    fig.update_traces(textposition="outside")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No data available for the chart yet.")

st.divider()

# ---------------------------------------------------------------------
# B) Review unreviewed incidents
# ---------------------------------------------------------------------
st.subheader("Review unreviewed")

try:
    unseen_rows = _load_incident_ids(seen=False)
except ApiError as e:
    st.error(f"api_client.ApiError: {e}")
    unseen_rows = []

st.caption(f"{len(unseen_rows)} unreviewed incident(s)")

if not unseen_rows:
    st.success("No unreviewed incidents at the moment.")
else:
    for r in unseen_rows:
        det = _load_incident_detail(r["id"])
        links = _links_from_detail(det)
        with st.expander(
            f"{r['headline']} • { _fmt_date_iso_to_de(r.get('occurred_at')) } • Sources: {len(links)}",
            expanded=True,
        ):
            # Show source links
            if links:
                for u in links:
                    st.markdown(f"- {u}")
            else:
                st.write("No sources yet.")

            # Select articles you want to remove
            if det.get("articles"):
                id_to_title = {a["id"]: a.get("title") or a.get("link") for a in det["articles"]}
                article_choices = list(id_to_title.keys())
                rm_ids = st.multiselect(
                    "Remove links from this incident",
                    options=article_choices,
                    format_func=lambda aid: f"{id_to_title.get(aid, aid)}",
                    key=f"rm_{r['id']}",
                )
            else:
                rm_ids = []

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("Mark as seen", key=f"s_{r['id']}"):
                    try:
                        patch_gov_incident_seen(r["id"], True)
                        st.success("Incident marked as seen.")
                        _refresh_all()
                        st.rerun()
                    except ApiError as e:
                        st.error(f"Error: {e}")
            with col2:
                if st.button("Delete incident", key=f"d_{r['id']}"):
                    try:
                        delete_gov_incident(r["id"])
                        st.warning("Incident deleted.")
                        _refresh_all()
                        st.rerun()
                    except ApiError as e:
                        st.error(f"Error: {e}")
            with col3:
                if st.button("Remove selected links", key=f"rm_btn_{r['id']}", disabled=(len(rm_ids) == 0)):
                    try:
                        for aid in rm_ids:
                            delete_gov_incident_article(r["id"], int(aid))
                        st.success(f"Removed {len(rm_ids)} link(s).")
                        _refresh_all()
                        st.rerun()
                    except ApiError as e:
                        st.error(f"Error: {e}")

st.divider()

# ---------------------------------------------------------------------
# C) Manually add a new incident
# ---------------------------------------------------------------------
st.subheader("Manually add a new incident")

with st.form("manual_add"):
    headline = st.text_input("Title / headline", placeholder="e.g., Long-haul A350 technical issue")
    date_str = st.text_input("Date (dd-mm-yyyy) – no time", placeholder="e.g., 08-11-2025")
    links_text = st.text_area(
        "Source links (one per line)",
        placeholder="https://example.com/article-1\nhttps://another-site.com/story",
        height=140,
    )
    submitted = st.form_submit_button("Create")

if submitted:
    if not headline.strip():
        st.error("Please provide a **title**.")
    else:
        iso = _iso_from_ddmmyyyy(date_str)
        if not iso:
            st.error("Please enter a valid date in **dd-mm-yyyy** format.")
        else:
            # Prepare minimal article rows from given links
            rows: list[dict] = []
            for line in links_text.splitlines():
                u = line.strip()
                if not u:
                    continue
                rows.append(
                    {
                        "title": u,             # minimal: title = link
                        "source": _hostname(u), # source = hostname
                        "link": u,
                        "published_at": None,   # optional
                    }
                )
            if not rows:
                st.error("Please provide at least **one link**.")
            else:
                try:
                    created = post_gov_incident(headline=headline.strip(), occurred_at=iso, articles=rows)
                    _refresh_all()
                    st.success(f"Incident created (ID {created.get('id')}).")
                    st.rerun()
                except ApiError as e:
                    st.error(f"Creation failed: {e}")

st.divider()

# ---------------------------------------------------------------------
# D) Danger zone – wipe database
# ---------------------------------------------------------------------
st.subheader("⚠️ Danger zone – wipe database")
st.caption("Deletes **all** government-aircraft data (incidents + articles). This cannot be undone.")

left, right = st.columns([1, 3])
with left:
    really = st.checkbox("Yes, delete everything", value=False)
with right:
    if st.button("Wipe database", type="secondary", disabled=not really):
        try:
            res = wipe_gov(confirm=True)
            st.warning("All data wiped.")
            st.json(res)
            st.rerun()
        except ApiError as e:
            st.error(f"Wipe failed: {e}")
