# streamlit/pages/01_Bildwatch.py
import pandas as pd
import streamlit as st
import plotly.express as px
from zoneinfo import ZoneInfo
from datetime import datetime, timezone, date
from typing import Dict, Any, List, Optional

from api_client import (
    # Bildwatch – articles & charts
    get_bild_articles,
    delete_bild_articles,
    get_bild_category_counts,
    get_bild_hourly,
    get_bild_daily_conversions,
    get_bild_logs,
    delete_bild_logs,
    # Bild corrections
    get_bild_corrections,
    delete_bild_corrections,
)

TZ = ZoneInfo("Europe/Berlin")

# -----------------------------
# Page & header
# -----------------------------
st.set_page_config(page_title="Bildwatch – Metrics & Transparency", page_icon="📰", layout="wide")
st.title("📰 Bildwatch (Metrics & Transparency)")
st.caption(
    "All times displayed in Europe/Berlin. Below every chart/table you’ll find a short, reproducible note on "
    "what the data shows, how it’s calculated, and how many data points were used."
)

# -----------------------------
# Category translation (DE → EN)
# -----------------------------
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

# -----------------------------
# Helpers: captions
# -----------------------------
def _fmt_timerange(d1: Optional[date], d2: Optional[date]) -> str:
    if not d1 or not d2:
        return "n/a"
    return f"{d1} → {d2}"

def caption(text: str):
    st.caption(text)

# -----------------------------
# Caches / Loaders
# -----------------------------
@st.cache_data(ttl=30)
def load_category_counts(premium_only: bool = False):
    return get_bild_category_counts(premium_only=premium_only)

@st.cache_data(ttl=30)
def load_hourly(days: int = 60):
    return get_bild_hourly(days=days)

@st.cache_data(ttl=30)
def load_daily_conversions(days: int = 60):
    return get_bild_daily_conversions(days=days)

@st.cache_data(ttl=30)
def load_articles(limit: int = 20000, offset: int = 0):
    return get_bild_articles(limit=limit, offset=offset)

@st.cache_data(ttl=30)
def load_corrections():
    return get_bild_corrections()

# Reload button
if st.button("🔄 Refresh all"):
    load_category_counts.clear()
    load_hourly.clear()
    load_daily_conversions.clear()
    load_articles.clear()
    load_corrections.clear()
    st.rerun()

# ===================================================
# Corrections
# ===================================================
st.subheader("Corrections")

c1, c2 = st.columns([1, 1])
with c1:
    if st.button("🔁 Reload corrections"):
        load_corrections.clear(); st.rerun()
with c2:
    if st.button("🗑️ Delete ALL corrections"):
        try:
            delete_bild_corrections()
            load_corrections.clear()
            st.success("All corrections deleted.")
            st.rerun()
        except Exception as e:
            st.error(f"Error while deleting: {e}")

try:
    corr = load_corrections() or []
    if corr:
        dfc = pd.DataFrame(corr)

        # Convert timestamps to local time
        for col in ("published", "created_at"):
            if col in dfc.columns:
                dfc[col] = pd.to_datetime(dfc[col], utc=True, errors="coerce").dt.tz_convert(TZ)

        # Infer category from first path segment of article_url
        def infer_cat(url: str) -> str | None:
            try:
                if not isinstance(url, str):
                    return None
                part = url.split("://", 1)[-1]  # remove scheme
                path = part.split("/", 1)[-1]   # remove host
                seg0 = path.split("/", 1)[0]    # first path segment
                return seg0 or None
            except Exception:
                return None

        dfc["category_raw"] = dfc.get("article_url", pd.Series(dtype=str)).map(infer_cat)
        dfc["category"] = dfc["category_raw"].map(translate_category).fillna("Unknown")

        show_cols = [c for c in ["published", "created_at", "category", "title", "message", "article_url", "source_url", "id"] if c in dfc.columns]

        st.dataframe(
            dfc[show_cols].sort_values("published", ascending=False) if "published" in dfc.columns else dfc[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "published": st.column_config.DatetimeColumn("Published"),
                "created_at": st.column_config.DatetimeColumn("Ingested"),
                "category": st.column_config.TextColumn("Category") if hasattr(st.column_config, "TextColumn") else None,
                "title": st.column_config.TextColumn("Title") if hasattr(st.column_config, "TextColumn") else None,
                "article_url": st.column_config.LinkColumn("Article URL"),
                "source_url": st.column_config.LinkColumn("Source URL"),
            },
        )
        caption(f"{len(dfc)} corrections listed. Local timezone: Europe/Berlin. Source: `/api/bild/corrections`.")

        # ----- Corrections by category (donut) -----
        st.markdown("**Corrections by category**")
        cat_counts = (
            dfc["category"]
            .value_counts(dropna=False)
            .rename_axis("Category")
            .reset_index(name="count")
            .sort_values("Category")
        )
        if not cat_counts.empty:
            palette = px.colors.qualitative.D3
            all_cats = cat_counts["Category"].tolist()
            color_map = {c: palette[i % len(palette)] for i, c in enumerate(all_cats)}
            fig_corr_pie = px.pie(
                cat_counts,
                names="Category",
                values="count",
                title="Share of corrections by category",
                hole=0.45,
                color="Category",
                color_discrete_map=color_map,
            )
            fig_corr_pie.update_traces(textinfo="percent+label", textposition="inside", showlegend=False)
            st.plotly_chart(fig_corr_pie, use_container_width=True)
            caption("Distribution of all currently loaded corrections by category. Percentages are relative shares of the total corrections set above.")
        else:
            st.info("No category data available for corrections.")

        # ----- Corrections per day (absolute) + averages -----
        st.markdown("**Corrections per day (Europe/Berlin)**")

        def _ensure_local_time_cols(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if "published_local" not in out.columns and "published" in out.columns:
                out["published_local"] = pd.to_datetime(out["published"], utc=True, errors="coerce").dt.tz_convert(TZ)
            if "ingested_local" not in out.columns and "created_at" in out.columns:
                out["ingested_local"] = pd.to_datetime(out["created_at"], utc=True, errors="coerce").dt.tz_convert(TZ)
            return out

        try:
            dfc = _ensure_local_time_cols(dfc)
            basis = st.radio(
                "Time basis",
                ["Published (converted to Europe/Berlin)", "Ingested (created_at → Europe/Berlin)"],
                horizontal=True,
                index=0,
                key="corrections_daily_basis",
            )
            use_col = "published_local" if basis.startswith("Published") else "ingested_local"

            tmp = dfc[[use_col, "id"]].dropna(subset=[use_col]).copy() if use_col in dfc.columns else pd.DataFrame(columns=["id"])
            if tmp.empty:
                st.info("No timestamps available for this basis.")
            else:
                tmp["day"] = tmp[use_col].dt.date
                counts = tmp.groupby("day", as_index=False)["id"].count().rename(columns={"id": "count"})
                day_min = pd.to_datetime(min(tmp["day"]))
                day_max = pd.to_datetime(max(tmp["day"]))
                all_days = pd.date_range(day_min, day_max, freq="D")
                counts["day"] = pd.to_datetime(counts["day"])
                counts_full = pd.DataFrame({"day": all_days}).merge(counts, on="day", how="left").fillna({"count": 0})
                counts_full["count"] = counts_full["count"].astype(int)
                counts_full["day"] = counts_full["day"].dt.date

                fig_day = px.bar(
                    counts_full,
                    x="day",
                    y="count",
                    title="Absolute count per local day",
                    labels={"day": "Day", "count": "Corrections"},
                )
                fig_day.update_xaxes(type="category")
                st.plotly_chart(fig_day, use_container_width=True)

                total = int(tmp.shape[0])
                n_days = len(all_days)
                avg_per_day = total / n_days if n_days else 0.0
                avg_per_hour_overall = total / (n_days * 24) if n_days else 0.0

                k1, k2 = st.columns(2)
                k1.metric("Avg. corrections per day", f"{avg_per_day:.2f}")
                k2.metric("Avg. corrections per hour (overall)", f"{avg_per_hour_overall:.3f}")

                caption(
                    f"Counts are based on local calendar days ({_fmt_timerange(all_days.min().date(), all_days.max().date())}). "
                    f"Total corrections considered: {total}. Days without corrections are shown as zero."
                )
        except Exception as e:
            st.error(f"Error while loading corrections per day: {e}")

    else:
        st.info("No corrections available.")
except Exception as e:
    st.error(f"Error while loading corrections: {e}")

st.divider()

# ===================================================
# Category distribution (articles)
# ===================================================
st.subheader("Category distribution (articles)")

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
            caption(
                f"Share of article categories across all items currently in the database view. "
                f"Total articles counted here: {sum(values)}."
            )
        else:
            st.info("No category data available.")
    except Exception as e:
        st.error(f"Error when creating the pie chart: {e}")

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
            caption(
                f"Same method as left chart but restricted to items marked as Premium. "
                f"Total premium articles counted: {sum(values)}."
            )
        else:
            st.info("No premium articles at the moment.")
    except Exception as e:
        st.error(f"Error when creating the premium pie chart: {e}")

# ===================================================
# Hourly charts (articles)
# ===================================================
st.subheader("Hourly article metrics (rolling averages)")

try:
    hourly = load_hourly(days=60)  # {"snapshot_avg": [...], "new_avg": [...]}
    if hourly and (hourly.get("snapshot_avg") or hourly.get("new_avg")):
        # Snapshot: average inventory by hour
        if hourly.get("snapshot_avg"):
            df_snap = pd.DataFrame(hourly["snapshot_avg"]).sort_values("hour")
            snap_long = df_snap.melt(id_vars="hour", var_name="Type", value_name="Avg inventory")
            fig_snap = px.bar(
                snap_long, x="hour", y="Avg inventory", color="Type",
                title="Avg. total articles on site per hour (Europe/Berlin)",
                barmode="stack", category_orders={"hour": list(range(24))},
            )
            fig_snap.update_layout(xaxis_title="Local hour (0–23)", yaxis_title="Avg. articles")
            st.plotly_chart(fig_snap, use_container_width=True)
            caption(
                "Mean number of articles visible on the site per local hour (stacked by type). "
                "Computed from the last N days (see backend window), then averaged by hour of day."
            )
        else:
            st.info("No snapshot data available.")

        # New: average new articles per hour
        if hourly.get("new_avg"):
            df_new = pd.DataFrame(hourly["new_avg"]).sort_values("hour")
            new_long = df_new.melt(id_vars="hour", var_name="Type", value_name="Avg new")
            fig_new = px.bar(
                new_long, x="hour", y="Avg new", color="Type",
                title="Avg. new articles per local hour (Europe/Berlin)",
                barmode="stack", category_orders={"hour": list(range(24))},
            )
            fig_new.update_layout(xaxis_title="Local hour (0–23)", yaxis_title="Avg. new articles")
            st.plotly_chart(fig_new, use_container_width=True)
            caption(
                "Mean number of newly published articles per local hour. "
                "Computed across the rolling window in the backend."
            )
        else:
            st.info("No new-count data available.")
    else:
        st.info("No hourly metrics yet. Is the scraper running hourly?")
except Exception as e:
    st.error(f"Error while loading metrics: {e}")

# ===================================================
# Premium → free per day
# ===================================================
st.subheader("Premium → free per day")

try:
    conv = load_daily_conversions(days=60)  # [{"day":"YYYY-MM-DD","count":N}, ...]
    if conv:
        dfconv = pd.DataFrame(conv).sort_values("day")
        fig_conv = px.bar(
            dfconv, x="day", y="count",
            title="Number of premium→free switches per local day",
            labels={"day": "Day", "count": "Switches"},
        )
        fig_conv.update_xaxes(type="category")
        st.plotly_chart(fig_conv, use_container_width=True)

        try:
            d1 = pd.to_datetime(dfconv["day"].min()).date()
            d2 = pd.to_datetime(dfconv["day"].max()).date()
        except Exception:
            d1 = d2 = None
        caption(
            f"Counts of articles that switched from paywalled to free on each local calendar day "
            f"({_fmt_timerange(d1, d2)}). Source: `/api/bild/charts/daily_conversions`."
        )
    else:
        st.info("No conversions (converted_time) yet.")
except Exception as e:
    st.error(f"Error while evaluating conversions: {e}")

# ===================================================
# All articles table
# ===================================================
st.subheader("All articles (newest first)")

try:
    rows = load_articles(limit=20000, offset=0)
    if rows:
        df = pd.DataFrame(rows)

        for col in ("published", "converted_time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert(TZ)

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
        caption(
            f"{len(df)} rows shown. Columns reflect the current backend schema. "
            "Timestamps are converted to Europe/Berlin for display."
        )
    else:
        st.info("No articles available.")
except Exception as e:
    st.error(f"Error while loading articles: {e}")

# ===================================================
# LOGS (lazy load)
# ===================================================
st.divider()
st.subheader("Logs (bild.log)")

if "show_logs" not in st.session_state:
    st.session_state.show_logs = False

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("📜 Load logs"):
        st.session_state.show_logs = True; st.rerun()
with col_b:
    if st.button("🗑️ Delete all logs"):
        try:
            delete_bild_logs()
            st.session_state.show_logs = False
            st.success("All logs deleted."); st.rerun()
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
            caption(f"{len(dfl)} log entries loaded. Source: `/api/bild/logs`.")
        else:
            st.info("No logs available.")
    except Exception as e:
        st.error(f"Error while loading logs: {e}")

# ===================================================
# Methodology & Data Notes
# ===================================================
st.divider()
st.subheader("Methodology & Data Notes")

with st.container():
    try:
        # derive some simple counts / time windows for transparency
        articles_total = len(rows) if "rows" in locals() and rows else None
        corrections_total = len(corr) if "corr" in locals() and corr else None
        # conversions window
        if "dfconv" in locals() and not dfconv.empty:
            conv_d1 = pd.to_datetime(dfconv["day"].min()).date()
            conv_d2 = pd.to_datetime(dfconv["day"].max()).date()
        else:
            conv_d1 = conv_d2 = None
    except Exception:
        articles_total = corrections_total = None
        conv_d1 = conv_d2 = None

    bullets = [
        "**Scope:** This dashboard summarizes scraped article metadata, category distributions, hourly activity, conversions from premium to free, and recorded corrections.",
        f"**Timezone:** All timestamps are displayed in Europe/Berlin; underlying storage uses UTC.",
        f"**Articles in this view:** {articles_total if articles_total is not None else 'n/a'} (table above).",
        f"**Corrections loaded:** {corrections_total if corrections_total is not None else 'n/a'}.",
        f"**Conversions window:** {_fmt_timerange(conv_d1, conv_d2)} (per-day counts).",
        "**Category charts:** Shares are computed over current counts returned by the backend; premium-only chart filters by `is_premium=true`.",
        "**Hourly metrics:** Rolling averages computed server-side (snapshot inventory and new items per hour), then grouped by local hour 0–23.",
        "**Corrections per day:** Local-day aggregation; days without corrections are shown as zero to avoid survivorship bias.",
        "**Data sources:** Content is scraped from the public BILD website for research/monitoring purposes; please respect their terms of service and robots.txt.",
        "**Limitations:** Category labels are inferred and translated; site structure can change; scraping gaps or rate limits may introduce missing data.",
    ]
    for b in bullets:
        st.markdown(f"- {b}")
