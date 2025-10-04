# streamlit/pages/01_Bildwatch.py
import pandas as pd
import streamlit as st
import plotly.express as px
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
from datetime import datetime, timezone

from api_client import (
    get_bild_articles,
    delete_bild_articles,
    get_bild_category_counts,
    get_bild_hourly,
    get_bild_daily_conversions,
    get_bild_logs,
    delete_bild_logs,
    get_bild_corrections,        # <- NEU
    delete_bild_corrections,     # <- NEU
)

TZ = ZoneInfo("Europe/Berlin")

# -----------------------------
# English UI + category translation
# -----------------------------
st.set_page_config(page_title="sehbmaster – Bildwatch", page_icon="📰", layout="wide")
st.title("📰 Bildwatch")
st.caption("Overview & metrics. Times in Europe/Berlin.")

CATEGORY_MAP = {
    "Politik": "Politics",
    "politik": "Politics",
    "Sport": "Sports",
    "sport": "Sports",
    "Fußball": "Football",
    "Bundesliga": "Bundesliga",
    "Wirtschaft": "Business",
    "Geld": "Money",
    "Finanzen": "Finance",
    "Panorama": "General",
    "Blaulicht": "Crime",
    "Wissen": "Science",
    "leben-wissen": "Life & Knowledge",
    "Technik": "Tech",
    "Digital": "Tech",
    "Unterhaltung": "Entertainment",
    "unterhaltung": "Entertainment",
    "VIP": "Celebrities",
    "Kultur": "Culture",
    "Reise": "Travel",
    "Leben": "Lifestyle",
    "Ratgeber": "Advice",
    "Gesundheit": "Health",
    "Auto": "Cars",
    "Motorsport": "Motorsport",
    "Meinung": "Opinion",
    "International": "World",
    "Region": "Local",
    "regional": "Local",
    "Berlin": "Berlin",
    "Hamburg": "Hamburg",
    "news": "News",
    "Unbekannt": "Unknown",
}

def translate_category(cat: str) -> str:
    if not isinstance(cat, str):
        return cat
    return CATEGORY_MAP.get(cat.strip(), cat.strip())

def translate_cat_counts(d: dict) -> dict:
    return {translate_category(k): v for k, v in (d or {}).items()}

def extract_bild_category(url: str) -> str | None:
    """
    Holt den ersten Path-Slug hinter bild.de/, z.B.
      https://www.bild.de/unterhaltung/stars... -> 'unterhaltung'
      https://bild.de/sport/mehr-sport/...      -> 'sport'
    Fällt zurück auf 'Unknown', wenn nichts passt.
    """
    if not isinstance(url, str) or not url:
        return "Unknown"
    try:
        p = urlparse(url)
        # path beginnt mit /foo/bar... -> nimm erstes Segment
        segment = (p.path or "/").strip("/").split("/")[0] or "Unknown"
        # manche Artikel haben volle Domain ohne www., egal – wir nehmen nur das Segment
        return segment.lower() if segment else "Unknown"
    except Exception:
        return "Unknown"

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

@st.cache_data(ttl=15)
def load_corrections():
    return get_bild_corrections()

# Reload button
if st.button("🔄 Reload"):
    load_category_counts.clear()
    load_hourly.clear()
    load_daily_conversions.clear()
    load_articles.clear()
    load_corrections.clear()
    st.rerun()

# -----------------------------
# a) Category pies (all) + premium
# -----------------------------
st.subheader("Category distribution")

# 1) Fetch once & build shared color map over the union of categories (in EN)
try:
    cat_counts_raw = load_category_counts(premium_only=False) or {}
    cat_counts_prem_raw = load_category_counts(premium_only=True) or {}

    cat_counts = translate_cat_counts(cat_counts_raw)
    cat_counts_prem = translate_cat_counts(cat_counts_prem_raw)

    all_cats = sorted(set(cat_counts.keys()) | set(cat_counts_prem.keys()))
    palette = px.colors.qualitative.D3
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(all_cats)}
except Exception as e:
    st.error(f"Error while loading categories: {e}")
    cat_counts, cat_counts_prem, color_map = {}, {}, {}

col1, col2 = st.columns(2)

# All articles
with col1:
    try:
        if cat_counts:
            labels = list(cat_counts.keys())
            values = list(cat_counts.values())
            fig_pie = px.pie(
                names=labels,
                values=values,
                title="All articles",
                hole=0.3,
                color=labels,
                color_discrete_map=color_map,
            )
            fig_pie.update_traces(textinfo="percent+label", textposition="inside", showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No category data available.")
    except Exception as e:
        st.error(f"Error when creating the pie chart: {e}")

# Premium only
with col2:
    try:
        if cat_counts_prem:
            labels = list(cat_counts_prem.keys())
            values = list(cat_counts_prem.values())
            fig_prem_pie = px.pie(
                names=labels,
                values=values,
                title="Premium only",
                hole=0.3,
                color=labels,
                color_discrete_map=color_map,
            )
            fig_prem_pie.update_traces(textinfo="percent+label", textposition="inside", showlegend=False)
            st.plotly_chart(fig_prem_pie, use_container_width=True)
        else:
            st.info("No premium articles at the moment.")
    except Exception as e:
        st.error(f"Error when creating the premium pie chart: {e}")

# -----------------------------
# b) & c) Hourly charts – backend-calculated
# -----------------------------
try:
    hourly = load_hourly(days=60)  # {"snapshot_avg": [...], "new_avg": [...]}
    if hourly and (hourly.get("snapshot_avg") or hourly.get("new_avg")):
        # Snapshot
        if hourly.get("snapshot_avg"):
            df_snap = pd.DataFrame(hourly["snapshot_avg"]).sort_values("hour")
            snap_long = df_snap.melt(id_vars="hour", var_name="Type", value_name="Ø inventory")

            fig_snap = px.bar(
                snap_long, x="hour", y="Ø inventory", color="Type",
                title="Avg. total articles per hour (Europe/Berlin)",
                barmode="stack", category_orders={"hour": list(range(24))},
            )
            fig_snap.update_layout(xaxis_title="Hour (0–23)", yaxis_title="Avg. articles")
            st.plotly_chart(fig_snap, use_container_width=True)
        else:
            st.info("No snapshot data available.")

        # New
        if hourly.get("new_avg"):
            df_new = pd.DataFrame(hourly["new_avg"]).sort_values("hour")
            new_long = df_new.melt(id_vars="hour", var_name="Type", value_name="Ø new")

            fig_new = px.bar(
                new_long, x="hour", y="Ø new", color="Type",
                title="Avg. new articles per hour (Europe/Berlin)",
                barmode="stack", category_orders={"hour": list(range(24))},
            )
            fig_new.update_layout(xaxis_title="Hour (0–23)", yaxis_title="Avg. new articles")
            st.plotly_chart(fig_new, use_container_width=True)
        else:
            st.info("No new-count data available.")
    else:
        st.info("No metric data available. Is the scraper already running hourly?")
except Exception as e:
    st.error(f"Error while loading metrics: {e}")

# -----------------------------
# d) Premium → free per day
# -----------------------------
st.subheader("Premium → free per day (Europe/Berlin)")
try:
    conv = load_daily_conversions(days=60)  # [{"day":"YYYY-MM-DD","count":N}, ...]
    if conv:
        df_conv = pd.DataFrame(conv).sort_values("day")
        fig_conv = px.bar(
            df_conv, x="day", y="count",
            title="Premium → free switches per day",
            labels={"day": "Day", "count": "Switches"},
        )
        fig_conv.update_xaxes(type="category")
        st.plotly_chart(fig_conv, use_container_width=True)
    else:
        st.info("No conversions (converted_time) yet.")
except Exception as e:
    st.error(f"Error while evaluating conversions: {e}")

# -----------------------------
# Articles table
# -----------------------------
st.subheader("All articles (newest first)")
try:
    rows = load_articles(limit=20000, offset=0)
    if rows:
        df = pd.DataFrame(rows)

        # Timestamps to local tz
        for col in ("published", "converted_time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert(TZ)

        # Category to English (display)
        if "category" in df.columns:
            df["category_en"] = df["category"].map(translate_category)
            df.drop(columns=["category"], inplace=True)
            df.rename(columns={"category_en": "category"}, inplace=True)

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
                "category": st.column_config.TextColumn("Category") if hasattr(st.column_config, "TextColumn") else None,
                "title": st.column_config.TextColumn("Title") if hasattr(st.column_config, "TextColumn") else None,
            },
        )
        st.caption(f"{len(df)} entries loaded.")
    else:
        st.info("No articles available.")
except Exception as e:
    st.error(f"Error while loading articles: {e}")

# -----------------------------
# Corrections (neu)
# -----------------------------
st.divider()
st.subheader("Bild corrections")

c1, c2 = st.columns([1, 1])
with c1:
    if st.button("🔄 Reload corrections"):
        load_corrections.clear()
        st.rerun()
with c2:
    if st.button("🗑️ Delete ALL corrections"):
        try:
            delete_bild_corrections()
            load_corrections.clear()
            st.success("All corrections deleted.")
            st.rerun()
        except Exception as e:
            st.error(f"Delete failed: {e}")

try:
    corr = load_corrections() or []
    if not corr:
        st.info("No corrections yet.")
    else:
        dfc = pd.DataFrame(corr)

        # Timestamps → Europe/Berlin
        for col in ("published", "created_at"):
            if col in dfc.columns:
                dfc[col] = pd.to_datetime(dfc[col], utc=True, errors="coerce").dt.tz_convert(TZ)

        # Kategorie aus article_url extrahieren und in EN übertragen
        dfc["corr_category_raw"] = dfc["article_url"].map(extract_bild_category)
        dfc["corr_category"] = dfc["corr_category_raw"].map(translate_category)

        # Tabelle
        table_cols = ["published", "created_at", "corr_category", "title", "article_url", "source_url", "id"]
        cols = [c for c in table_cols if c in dfc.columns] + [c for c in dfc.columns if c not in table_cols]
        st.dataframe(
            dfc[cols].sort_values(["published", "created_at"], ascending=[False, False]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "published": st.column_config.DatetimeColumn("Published"),
                "created_at": st.column_config.DatetimeColumn("Ingested"),
                "article_url": st.column_config.LinkColumn("Article URL"),
                "source_url": st.column_config.LinkColumn("Source URL"),
                "corr_category": st.column_config.TextColumn("Category") if hasattr(st.column_config, "TextColumn") else None,
                "title": st.column_config.TextColumn("Title") if hasattr(st.column_config, "TextColumn") else None,
            },
        )
        st.caption(f"{len(dfc)} corrections loaded.")

        # ---- Pie: "In welchen Kategorien wurde wie oft korrigiert?"
        st.subheader("Corrections by category")
        counts = (
            dfc.groupby("corr_category", dropna=False)["id"]
               .count()
               .reset_index()
               .rename(columns={"id": "count"})
               .sort_values("count", ascending=False)
        )

        # Farbe konsistent mit den Artikel-Charts, ggf. Categories ergänzen
        existing = set(color_map.keys())
        extra_cats = [c for c in counts["corr_category"].tolist() if c not in existing]
        if extra_cats:
            palette = px.colors.qualitative.D3
            start_idx = len(color_map)
            for i, cat in enumerate(extra_cats, start=start_idx):
                color_map[cat] = palette[i % len(palette)]

        fig_corr_pie = px.pie(
            counts,
            names="corr_category",
            values="count",
            title="Corrections per category",
            hole=0.3,
            color="corr_category",
            color_discrete_map=color_map,
        )
        fig_corr_pie.update_traces(textinfo="percent+label", textposition="inside", showlegend=False)
        st.plotly_chart(fig_corr_pie, use_container_width=True)

except Exception as e:
    st.error(f"Error while loading corrections: {e}")

# -----------------------------
# LOGS (lazy load)
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
            st.success("All logs deleted.")
            st.rerun()
        except Exception as e:
            st.error(f"Error while deleting: {e}")

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
                    "timestamp": st.column_config.DatetimeColumn("Time"),
                    "message": st.column_config.TextColumn("Message") if hasattr(st.column_config, "TextColumn") else None,
                },
            )
            st.caption(f"{len(dfl)} log entries loaded.")
        else:
            st.info("No logs available.")
    except Exception as e:
        st.error(f"Error while loading logs: {e}")
