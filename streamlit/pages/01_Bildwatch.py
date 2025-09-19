# streamlit/pages/01_Bildwatch.py
import pandas as pd
import streamlit as st
import plotly.express as px
from zoneinfo import ZoneInfo
from datetime import datetime, timezone

from api_client import (
    get_bild_articles,
    delete_bild_articles,
    get_bild_category_counts,     # jetzt backend-berechnet; supports premium_only
    get_bild_hourly,              # neu
    get_bild_daily_conversions,   # neu
    get_bild_logs,
    delete_bild_logs,
)

TZ = ZoneInfo("Europe/Berlin")

st.set_page_config(page_title="sehbmaster – Bildwatch", page_icon="📰", layout="wide")
st.title("📰 Bildwatch")
st.caption("Übersicht & Metriken. Zeiten in Europe/Berlin.")

# -----------------------------
# Caches / Loader
# -----------------------------
@st.cache_data(ttl=15)
def load_category_counts(premium_only: bool = False):
    return get_bild_category_counts(premium_only=premium_only)

@st.cache_data(ttl=15)
def load_hourly(days: int = 60):
    return get_bild_hourly(days=days)

@st.cache_data(ttl=15)
def load_daily_conversions(days: int = 60):
    return get_bild_daily_conversions(days=days)

@st.cache_data(ttl=15)
def load_articles(limit: int = 20000, offset: int = 0):
    return get_bild_articles(limit=limit, offset=offset)

# Reload-Button
if st.button("🔄 Neu laden"):
    load_category_counts.clear()
    load_hourly.clear()
    load_daily_conversions.clear()
    load_articles.clear()
    st.rerun()

# -----------------------------
# a) Kreisdiagramm Kategorien (alle)
# -----------------------------
try:
    cat_counts = load_category_counts(premium_only=False)
    if cat_counts:
        labels = list(cat_counts.keys())
        values = list(cat_counts.values())
        fig_pie = px.pie(
            names=labels,
            values=values,
            title="Verteilung der Kategorien (alle Artikel)",
            hole=0.3,
        )
        fig_pie.update_traces(textinfo="percent+label", textposition="inside", showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Keine Kategorien-Daten verfügbar.")
except Exception as e:
    st.error(f"Fehler beim Laden der Kategorien: {e}")

# -----------------------------
# Kategorien – nur Premium-Artikel (Backend-berechnet)
# -----------------------------
st.subheader("Kategorien – nur Premium-Artikel")
try:
    cat_counts_prem = load_category_counts(premium_only=True)
    if cat_counts_prem:
        labels = list(cat_counts_prem.keys())
        values = list(cat_counts_prem.values())
        fig_prem_pie = px.pie(
            names=labels,
            values=values,
            title="Verteilung der Kategorien (nur Premium)",
            hole=0.3,
        )
        fig_prem_pie.update_traces(textinfo="percent+label", textposition="inside", showlegend=False)
        st.plotly_chart(fig_prem_pie, use_container_width=True)
    else:
        st.info("Aktuell keine Premium-Artikel vorhanden.")
except Exception as e:
    st.error(f"Fehler beim Erstellen des Premium-Kreisdiagramms: {e}")

# -----------------------------
# b) & c) Stündliche Charts (lokale Zeit) – Backend-berechnet
# -----------------------------
try:
    hourly = load_hourly(days=60)  # {"snapshot_avg": [...], "new_avg": [...]}
    if hourly and (hourly.get("snapshot_avg") or hourly.get("new_avg")):
        # Snapshot
        if hourly.get("snapshot_avg"):
            df_snap = pd.DataFrame(hourly["snapshot_avg"])
            df_snap = df_snap.sort_values("hour")
            snap_long = df_snap.melt(id_vars="hour", var_name="Typ", value_name="Ø Bestand")

            fig_snap = px.bar(
                snap_long, x="hour", y="Ø Bestand", color="Typ",
                title="Ø Artikel gesamt pro Stunde (Europe/Berlin)",
                barmode="stack", category_orders={"hour": list(range(24))},
            )
            fig_snap.update_layout(xaxis_title="Stunde (0–23)", yaxis_title="Ø Artikel")
            st.plotly_chart(fig_snap, use_container_width=True)
        else:
            st.info("Keine Snapshot-Daten vorhanden.")

        # New
        if hourly.get("new_avg"):
            df_new = pd.DataFrame(hourly["new_avg"]).sort_values("hour")
            new_long = df_new.melt(id_vars="hour", var_name="Typ", value_name="Ø Neu")

            fig_new = px.bar(
                new_long, x="hour", y="Ø Neu", color="Typ",
                title="Ø neue Artikel pro Stunde (Europe/Berlin)",
                barmode="stack", category_orders={"hour": list(range(24))},
            )
            fig_new.update_layout(xaxis_title="Stunde (0–23)", yaxis_title="Ø neue Artikel")
            st.plotly_chart(fig_new, use_container_width=True)
        else:
            st.info("Keine New-Count-Daten vorhanden.")
    else:
        st.info("Keine Metrik-Daten vorhanden. Läuft der Scraper schon stündlich?")
except Exception as e:
    st.error(f"Fehler beim Laden der Metriken: {e}")

# -----------------------------
# d) Umstellungen Premium→frei pro Tag
# -----------------------------
st.subheader("Premium → frei pro Tag (Europe/Berlin)")
try:
    conv = load_daily_conversions(days=60)  # [{"day":"YYYY-MM-DD","count":N}, ...]
    if conv:
        dfc = pd.DataFrame(conv).sort_values("day")
        fig_conv = px.bar(
            dfc, x="day", y="count",
            title="Umstellungen Premium→frei pro Tag",
            labels={"day": "Tag", "count": "Anzahl Umstellungen"},
        )
        fig_conv.update_xaxes(type="category")
        st.plotly_chart(fig_conv, use_container_width=True)
    else:
        st.info("Es liegen noch keine Umstellungen (converted_time) vor.")
except Exception as e:
    st.error(f"Fehler beim Auswerten der Umstellungen: {e}")

# -----------------------------
# Tabelle der Artikel
# -----------------------------
st.subheader("Alle Artikel (neueste zuerst)")
try:
    rows = load_articles(limit=20000, offset=0)
    if rows:
        df = pd.DataFrame(rows)
        for col in ("published", "converted_time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert(TZ)

        want = [
            "id", "title", "category",
            "is_premium", "converted",
            "published", "converted_time", "converted_duration_hours",
            "url",
        ]
        cols = [c for c in want if c in df.columns] + [c for c in df.columns if c not in want]
        df = df[cols]

        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "url": st.column_config.LinkColumn("URL"),
                "is_premium": st.column_config.CheckboxColumn("Premium", disabled=True),
                "converted": st.column_config.CheckboxColumn("Converted", disabled=True),
                "converted_duration_hours": st.column_config.NumberColumn("Converted (h)", format="%.3f"),
                "published": st.column_config.DatetimeColumn("Published"),
                "converted_time": st.column_config.DatetimeColumn("Converted time"),
            },
        )
        st.caption(f"{len(df)} Einträge geladen.")
    else:
        st.info("Keine Artikel vorhanden.")
except Exception as e:
    st.error(f"Fehler beim Laden der Artikel: {e}")



# -----------------------------
# LOGS (Lazy Load)
# -----------------------------
st.divider()
st.subheader("Logs (bild.log)")

if "show_logs" not in st.session_state:
    st.session_state.show_logs = False

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("📜 Load logs"):
        st.session_state.show_logs = True
        st.rerun()
with col_b:
    if st.button("🗑️ Delete all logs"):
        try:
            delete_bild_logs()
            st.session_state.show_logs = False
            st.success("Alle Logs gelöscht.")
            st.rerun()
        except Exception as e:
            st.error(f"Fehler beim Löschen: {e}")

if st.session_state.show_logs:
    try:
        logs = get_bild_logs(limit=5000, asc=True)
        dfl = pd.DataFrame(logs)
        if not dfl.empty:
            dfl["timestamp"] = pd.to_datetime(dfl["timestamp"], utc=True, errors="coerce").dt.tz_convert(TZ)
            subset = ["timestamp", "message", "id"]
            dfl = dfl[subset] if set(subset).issubset(dfl.columns) else dfl
            st.dataframe(
                dfl.sort_values("timestamp"),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Zeit"),
                    "message": st.column_config.TextColumn("Nachricht") if hasattr(st.column_config, "TextColumn") else None,
                },
            )
            st.caption(f"{len(dfl)} Log-Einträge geladen.")
        else:
            st.info("Keine Logs vorhanden.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Logs: {e}")
