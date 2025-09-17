import pandas as pd
import streamlit as st
from api_client import get_bild_articles, delete_bild_articles, get_bild_category_counts
import plotly.express as px

st.set_page_config(page_title="sehbmaster – Bildwatch", page_icon="📰", layout="wide")
st.title("📰 Bildwatch")
st.caption("Alle Einträge aus **bild.bildwatch** (nur lesen)")

# --- Kreisdiagramm ---
@st.cache_data(ttl=10)
def load_category_counts():
    return get_bild_category_counts()

try:
    cat_counts = load_category_counts()
    if cat_counts:
        labels = list(cat_counts.keys())
        values = list(cat_counts.values())
        fig = px.pie(
            names=labels,
            values=values,
            title="Verteilung der Kategorien",
            hole=0.3,
        )
        fig.update_traces(
            textinfo="percent+label",  # Prozent und Label direkt am Chart
            textposition="inside",     # Text ins Segment
            showlegend=False           # Keine Legende
        )
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Fehler beim Laden der Kategorien: {e}")

# --- Controls ---
with st.sidebar:
    st.subheader("Anzeige")
    limit = st.slider("Limit", 100, 5000, 1000, 100)
    offset = st.number_input("Offset", min_value=0, value=0, step=100)

@st.cache_data(ttl=10)
def load_articles(limit: int, offset: int):
    return get_bild_articles(limit=limit, offset=offset)

col_btn, col_del = st.columns([1, 1])
if col_btn.button("Neu laden"):
    load_articles.clear()
    load_category_counts.clear()
    st.rerun()
with col_del:
    if st.button("Alle Bildwatch-Einträge löschen", type="primary", use_container_width=True, help="Löscht alle Einträge unwiderruflich!", key="delete_bildwatch",):
        try:
            delete_bild_articles()
            load_articles.clear()
            load_category_counts.clear()
            st.success("Alle Einträge wurden gelöscht.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

# --- Tabelle ---
try:
    rows = load_articles(limit, offset)
    if rows:
        df = pd.DataFrame(rows)

        # Datumsfelder hübsch parsen
        for col in ("published", "converted_time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Spaltenreihenfolge
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
        st.success(f"{len(df)} Einträge geladen.")
    else:
        st.info("Keine Einträge vorhanden.")
except Exception as e:
    st.error(f"Fehler beim Laden: {e}")

