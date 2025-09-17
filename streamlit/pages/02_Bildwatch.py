import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta, timezone

from api_client import (
    get_bild_articles,
    delete_bild_articles,
    get_bild_category_counts,
    get_bild_metrics,
)

st.set_page_config(page_title="sehbmaster – Bildwatch", page_icon="📰", layout="wide")
st.title("📰 Bildwatch")
st.caption("Übersicht & Metriken. Zeiten in UTC.")

# -----------------------------
# Daten-Lader (Caching)
# -----------------------------
@st.cache_data(ttl=15)
def load_category_counts():
    return get_bild_category_counts()

@st.cache_data(ttl=15)
def load_metrics():
    # optional: Filterzeitraum (hier: letzte 60 Tage)
    to_dt = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=60)
    return get_bild_metrics(time_from=from_dt.isoformat(), time_to=to_dt.isoformat(), limit=20000)

@st.cache_data(ttl=15)
def load_articles():
    # ohne Sidebar-Controls: fixe, großzügige Grenze
    return get_bild_articles(limit=2000, offset=0)

# Reload-Button
if st.button("🔄 Neu laden"):
    load_category_counts.clear()
    load_metrics.clear()
    load_articles.clear()
    st.rerun()

# -----------------------------
# a) Kreisdiagramm Kategorien
# -----------------------------
try:
    cat_counts = load_category_counts()
    if cat_counts:
        labels = list(cat_counts.keys())
        values = list(cat_counts.values())
        fig_pie = px.pie(
            names=labels,
            values=values,
            title="Verteilung der Kategorien",
            hole=0.3,
        )
        fig_pie.update_traces(
            textinfo="percent+label",
            textposition="inside",
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Keine Kategorien-Daten verfügbar.")
except Exception as e:
    st.error(f"Fehler beim Laden der Kategorien: {e}")

# -----------------------------
# b) & c) Balkendiagramme (0–23 Uhr)
# -----------------------------
try:
    metrics = load_metrics()
    dfm = pd.DataFrame(metrics)

    if not dfm.empty:
        dfm["ts_hour"] = pd.to_datetime(dfm["ts_hour"], utc=True, errors="coerce")
        dfm["hour"] = dfm["ts_hour"].dt.hour  # 0..23 (UTC)

        # abgeleitet
        dfm["snapshot_non_premium"] = dfm["snapshot_total"] - dfm["snapshot_premium"]
        dfm["new_non_premium"] = dfm["new_count"] - dfm["new_premium_count"]

        # Ø pro Stunde (über alle Tage hinweg)
        snap_avg = (
            dfm.groupby("hour", as_index=False)[["snapshot_premium", "snapshot_non_premium"]]
              .mean(numeric_only=True)
              .round(2)
              .rename(columns={"snapshot_premium": "Premium", "snapshot_non_premium": "Nicht-Premium"})
        )
        new_avg = (
            dfm.groupby("hour", as_index=False)[["new_premium_count", "new_non_premium"]]
              .mean(numeric_only=True)
              .round(3)
              .rename(columns={"new_premium_count": "Premium", "new_non_premium": "Nicht-Premium"})
        )

        # Long-Format für Stack
        snap_long = snap_avg.melt(id_vars="hour", var_name="Typ", value_name="Ø Bestand")
        new_long  = new_avg.melt(id_vars="hour", var_name="Typ", value_name="Ø Neu")

        # b) Ø Gesamtartikel pro Stunde
        fig_snap = px.bar(
            snap_long,
            x="hour", y="Ø Bestand", color="Typ",
            title="Ø Artikel gesamt pro Stunde (UTC)",
            barmode="stack",
            category_orders={"hour": list(range(24))},
        )
        fig_snap.update_layout(xaxis_title="Stunde (0–23)", yaxis_title="Ø Artikel")

        # c) Ø neue Artikel pro Stunde
        fig_new = px.bar(
            new_long,
            x="hour", y="Ø Neu", color="Typ",
            title="Ø neue Artikel pro Stunde (UTC)",
            barmode="stack",
            category_orders={"hour": list(range(24))},
        )
        fig_new.update_layout(xaxis_title="Stunde (0–23)", yaxis_title="Ø neue Artikel")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_snap, use_container_width=True)
        with col2:
            st.plotly_chart(fig_new, use_container_width=True)

    else:
        st.info("Keine Metrik-Daten vorhanden. Läuft der Scraper schon stündlich?")
except Exception as e:
    st.error(f"Fehler beim Laden der Metriken: {e}")

# -----------------------------
# Tabelle der Artikel
# -----------------------------
st.subheader("Alle Artikel (neueste zuerst)")

try:
    rows = load_articles()
    if rows:
        df = pd.DataFrame(rows)

        # Zeitspalten
        for col in ("published", "converted_time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Reihenfolge der Spalten
        want = [
            "id", "title", "category",
            "is_premium", "converted",
            "published", "converted_time", "converted_duration_hours",
            "url",
        ]
        cols = [c for c in want if c in df.columns] + [c for c in df.columns if c not in want]
        df = df[cols]

        # Tabelle
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
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
# Danger-Zone: Alles löschen
# -----------------------------
st.divider()
if st.button("🗑️ Alle Bildwatch-Einträge löschen", type="primary"):
    try:
        delete_bild_articles()
        load_articles.clear()
        load_category_counts.clear()
        load_metrics.clear()
        st.success("Alle Einträge wurden gelöscht.")
        st.rerun()
    except Exception as e:
        st.error(str(e))
