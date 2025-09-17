# pages/02_Bildwatch.py
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from api_client import (
    get_bild_articles,
    delete_bild_articles,
    get_bild_category_counts,
    get_bild_metrics,
)

TZ = ZoneInfo("Europe/Berlin")

st.set_page_config(page_title="sehbmaster – Bildwatch", page_icon="📰", layout="wide")
st.title("📰 Bildwatch")
st.caption("Übersicht & Metriken. Zeiten in Europe/Berlin.")

# -----------------------------
# Caches / Loader
# -----------------------------
@st.cache_data(ttl=15)
def load_category_counts():
    return get_bild_category_counts()

@st.cache_data(ttl=15)
def load_metrics(days: int = 60, limit: int = 20000):
    to_dt = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=days)
    return get_bild_metrics(time_from=from_dt.isoformat(), time_to=to_dt.isoformat(), limit=limit)

@st.cache_data(ttl=15)
def load_articles(limit: int = 20000, offset: int = 0):
    return get_bild_articles(limit=limit, offset=offset)

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
        fig_pie.update_traces(textinfo="percent+label", textposition="inside", showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Keine Kategorien-Daten verfügbar.")
except Exception as e:
    st.error(f"Fehler beim Laden der Kategorien: {e}")

# -----------------------------
# b) & c) Stündliche Charts (lokale Zeit)
# -----------------------------
try:
    metrics = load_metrics()
    dfm = pd.DataFrame(metrics)

    if not dfm.empty:
        # UTC -> Europe/Berlin
        dfm["ts_hour"] = pd.to_datetime(dfm["ts_hour"], utc=True, errors="coerce")
        dfm["ts_hour_local"] = dfm["ts_hour"].dt.tz_convert(TZ)
        dfm["hour"] = dfm["ts_hour_local"].dt.hour  # 0..23 lokal

        # abgeleitet
        dfm["snapshot_non_premium"] = dfm["snapshot_total"] - dfm["snapshot_premium"]
        dfm["new_non_premium"] = dfm["new_count"] - dfm["new_premium_count"]

        # Ø pro Stunde (über alle Tage)
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

        snap_long = snap_avg.melt(id_vars="hour", var_name="Typ", value_name="Ø Bestand")
        new_long  = new_avg.melt(id_vars="hour", var_name="Typ", value_name="Ø Neu")

        col1, col2 = st.columns(2)

        with col1:
            fig_snap = px.bar(
                snap_long, x="hour", y="Ø Bestand", color="Typ",
                title="Ø Artikel gesamt pro Stunde (Europe/Berlin)",
                barmode="stack", category_orders={"hour": list(range(24))},
            )
            fig_snap.update_layout(xaxis_title="Stunde (0–23)", yaxis_title="Ø Artikel")
            st.plotly_chart(fig_snap, use_container_width=True)

        with col2:
            fig_new = px.bar(
                new_long, x="hour", y="Ø Neu", color="Typ",
                title="Ø neue Artikel pro Stunde (Europe/Berlin)",
                barmode="stack", category_orders={"hour": list(range(24))},
            )
            fig_new.update_layout(xaxis_title="Stunde (0–23)", yaxis_title="Ø neue Artikel")
            st.plotly_chart(fig_new, use_container_width=True)
    else:
        st.info("Keine Metrik-Daten vorhanden. Läuft der Scraper schon stündlich?")
except Exception as e:
    st.error(f"Fehler beim Laden der Metriken: {e}")

# -----------------------------
# d) Umstellungen Premium→frei pro Tag (Europe/Berlin)
# -----------------------------
st.subheader("Premium → frei pro Tag (Europe/Berlin)")
try:
    rows_all = load_articles(limit=20000, offset=0)
    dfa = pd.DataFrame(rows_all)

    if not dfa.empty and "converted_time" in dfa.columns:
        # nur Einträge mit gesetzter converted_time
        dfa = dfa[dfa["converted_time"].notna()].copy()
        if not dfa.empty:
            # UTC -> Europe/Berlin und dann Tages-STRING bauen (saubere Kategorie-Achse)
            local_ct = pd.to_datetime(dfa["converted_time"], utc=True, errors="coerce").dt.tz_convert(TZ)
            dfa["day"] = local_ct.dt.strftime("%Y-%m-%d")   # z.B. "2025-09-17"

            conv_daily = (
                dfa.groupby("day", as_index=False)
                   .size()
                   .rename(columns={"size": "count"})
                   .sort_values("day")
            )

            fig_conv = px.bar(
                conv_daily, x="day", y="count",
                title="Umstellungen Premium→frei pro Tag",
                labels={"day": "Tag", "count": "Anzahl Umstellungen"},
            )
            # explizit Kategorie-Achse -> genau ein Balken je Tag
            fig_conv.update_xaxes(type="category")
            st.plotly_chart(fig_conv, use_container_width=True)
        else:
            st.info("Es liegen noch keine Umstellungen (converted_time) vor.")
    else:
        st.info("Keine Artikel mit converted_time gefunden.")
except Exception as e:
    st.error(f"Fehler beim Auswerten der Umstellungen: {e}")


# -----------------------------
# Tabelle der Artikel
# -----------------------------
st.subheader("Alle Artikel (neueste zuerst)")
try:
    rows = rows_all if 'rows_all' in locals() else load_articles()
    if rows:
        df = pd.DataFrame(rows)

        # Zeitspalten lokal anzeigen
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
