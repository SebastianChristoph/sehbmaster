# streamlit/pages/02_Weatherwatch.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import date, timedelta

from api_client import (
    get_weather_accuracy,
    get_weather_data,
    get_weather_logs,
    delete_weather_logs,
)

st.set_page_config(page_title="sehbmaster – Weatherwatch", page_icon="⛅", layout="wide")
st.title("⛅ Weatherwatch")
st.caption("Vorhersage-Güte (Lead 1..N) vs. Beobachtung (Lead 0). Temperaturen: Min/Max separat. Zusätzlich Bias und Pivot-Tabellen je Variable.")

# ---------------------------------
# Konstanten
# ---------------------------------
KNOWN_MODELS = ["open-meteo", "metno", "wettercom", "default"]
KNOWN_CITIES = [
    "berlin", "hamburg", "muenchen", "koeln", "frankfurt", "stuttgart",
    "duesseldorf", "dortmund", "essen", "leipzig", "bremen", "dresden",
    "hannover", "nuernberg", "duisburg", "bochum", "wuppertal", "bielefeld",
    "bonn", "muenster", "rostock"
]
NUM_VARS = {
    "temp_min_c": {"title": "Min Temperatur (°C)", "unit": "°C"},
    "temp_max_c": {"title": "Max Temperatur (°C)", "unit": "°C"},
    "wind_mps":   {"title": "Wind (m/s)", "unit": "m/s"},
    "rain_mm":    {"title": "Regen (mm)", "unit": "mm"},
}
WEATHER_VAR = "weather"

# Default-Ampelschwellen für absolute Fehler |Forecast - Obs|
DEFAULT_THRESHOLDS = {
    "temp_min_c": (1.0, 2.0),  # <=1 grün, <=2 orange, sonst rot
    "temp_max_c": (1.0, 2.0),
    "wind_mps":   (0.6, 1.2),
    "rain_mm":    (0.5, 1.5),
}

# ---------------------------------
# Caches / Loader
# ---------------------------------
@st.cache_data(ttl=30)
def load_accuracy(frm: date, to: date, model: str, city: str, max_lead: int):
    return get_weather_accuracy(frm.isoformat(), to.isoformat(), model=model, city=city, max_lead=max_lead)

@st.cache_data(ttl=30)
def load_data_window(days_back: int, days_forward: int, model: str, city: str):
    # Zukunft inkl., damit neue Forecasts sofort sichtbar sind
    window_to = date.today() + timedelta(days=days_forward)
    window_from = date.today() - timedelta(days=days_back)
    return get_weather_data(
        window_from.isoformat(), window_to.isoformat(),
        model=model, city=city, lead_days=None, limit=10000
    )

# ---------------------------------
# UI Controls
# ---------------------------------
colA, colB, colC, colD = st.columns([1,1,1,1])
with colA:
    days_back = st.slider("Zeitraum (Tage zurück)", 7, 180, 35)
with colB:
    max_lead = st.slider("Max Lead", 0, 7, 7)
with colC:
    model = st.selectbox("Modell", options=KNOWN_MODELS, index=0)
with colD:
    city = st.selectbox("Stadt", options=KNOWN_CITIES, index=0)

frm = date.today() - timedelta(days=days_back)
to = date.today()

with st.sidebar:
    st.header("Farbschwellen")
    st.caption("Grenzen für die Tabellen-Ampel (absoluter Fehler vs. Beobachtung).")
    thr_temp_g, thr_temp_o = st.slider("Temp (°C): grün/orange", 0.0, 5.0, DEFAULT_THRESHOLDS["temp_min_c"][0]), \
                             st.slider("Temp (°C): orange/rot", 0.0, 6.0, DEFAULT_THRESHOLDS["temp_min_c"][1])
    thr_wind_g, thr_wind_o = st.slider("Wind (m/s): grün/orange", 0.0, 3.0, DEFAULT_THRESHOLDS["wind_mps"][0]), \
                             st.slider("Wind (m/s): orange/rot", 0.0, 4.0, DEFAULT_THRESHOLDS["wind_mps"][1])
    thr_rain_g, thr_rain_o = st.slider("Regen (mm): grün/orange", 0.0, 5.0, DEFAULT_THRESHOLDS["rain_mm"][0]), \
                             st.slider("Regen (mm): orange/rot", 0.0, 8.0, DEFAULT_THRESHOLDS["rain_mm"][1])

    thresholds = {
        "temp_min_c": (thr_temp_g, thr_temp_o),
        "temp_max_c": (thr_temp_g, thr_temp_o),
        "wind_mps":   (thr_wind_g, thr_wind_o),
        "rain_mm":    (thr_rain_g, thr_rain_o),
    }

if st.button("🔄 Neu laden"):
    load_accuracy.clear(); load_data_window.clear(); st.rerun()

# ---------------------------------
# Helper: Captions unter den Charts
# ---------------------------------
def list_leads_used(series: pd.Series) -> str:
    leads = sorted([int(x) for x in series.tolist()])
    if not leads:
        return "–"
    return ", ".join(str(x) for x in leads)

def caption_mae(kind: str, df_acc: pd.DataFrame, model: str, city: str, frm: date, to: date):
    n_total = int(df_acc["n"].fillna(0).sum())
    leads_ok = df_acc.loc[df_acc["n"].fillna(0) > 0, "lead_days"]
    st.caption(
        f"{kind} – MAE = mittlerer absoluter Fehler |Vorhersage − Beobachtung|. "
        f"Zeitraum: {frm} bis {to}. Leads: {list_leads_used(leads_ok)}. "
        f"Datenpunkte (Paare): {n_total}. Quelle: Backend /api/weather/accuracy – Modell: {model}, Stadt: {city}."
    )

def caption_weather_string(df_acc: pd.DataFrame, model: str, city: str, frm: date, to: date):
    n_total = int(df_acc["n"].fillna(0).sum())
    leads_ok = df_acc.loc[df_acc["n"].fillna(0) > 0, "lead_days"]
    st.caption(
        f"Wetter-String (exakte Übereinstimmung in %) – Anteil der Fälle, in denen der Text exakt identisch war. "
        f"Zeitraum: {frm} bis {to}. Leads: {list_leads_used(leads_ok)}. "
        f"Datenpunkte (Paare): {n_total}. Quelle: Backend /api/weather/accuracy – Modell: {model}, Stadt: {city}."
    )

def caption_bias(kind: str, df_bias: pd.DataFrame, model: str, city: str, frm: date, to: date):
    n_total = int(df_bias["n"].fillna(0).sum())
    leads_ok = df_bias.loc[df_bias["n"].fillna(0) > 0, "lead_days"]
    st.caption(
        f"{kind} – Bias = Mittel der Signed Errors (Vorhersage − Beobachtung). "
        f"> 0 = Überschätzung, < 0 = Unterschätzung. Zeitraum: {frm} bis {to}. "
        f"Leads: {list_leads_used(leads_ok)}. Datenpunkte (Paare): {n_total}. "
        f"Quelle: Backend /api/weather/data (clientseitig berechnet) – Modell: {model}, Stadt: {city}."
    )

# ---------------------------------
# Helper: Bias aus Rohdaten berechnen
# ---------------------------------
def compute_bias_buckets(df: pd.DataFrame, max_lead: int) -> pd.DataFrame:
    """
    Liefert pro Lead (1..max_lead) den Mittelwert der signed errors (Forecast - Observation)
    für temp_min_c, temp_max_c, wind_mps, rain_mm + n.
    """
    if df.empty:
        return pd.DataFrame({"lead_days": [], "n": []})

    # Beobachtungen (Lead 0) je Target-Date
    obs = df[df["lead_days"] == 0].set_index("target_date")

    rows = []
    for d in range(1, max_lead + 1):
        fc = df[df["lead_days"] == d].copy()
        if fc.empty:
            rows.append({"lead_days": d, "n": 0}); continue

        merged = fc.merge(
            obs[["temp_min_c", "temp_max_c", "wind_mps", "rain_mm", "weather"]],
            left_on="target_date", right_index=True, suffixes=("", "_obs")
        )

        def signed_mean(a, b):
            mask = a.notna() & b.notna()
            if not mask.any(): return None
            return float((a[mask] - b[mask]).mean())

        row = {
            "lead_days": d,
            "n": int(len(merged)),
            "temp_min_bias": signed_mean(merged["temp_min_c"], merged["temp_min_c_obs"]),
            "temp_max_bias": signed_mean(merged["temp_max_c"], merged["temp_max_c_obs"]),
            "wind_bias":     signed_mean(merged["wind_mps"], merged["wind_mps_obs"]),
            "rain_bias":     signed_mean(merged["rain_mm"], merged["rain_mm_obs"]),
            "weather_match_pct": float(100.0 * (merged["weather"].str.strip().str.lower()
                                               == merged["weather_obs"].str.strip().str.lower()).mean()) if len(merged) else None
        }
        rows.append(row)

    return pd.DataFrame(rows)

# ---------------------------------
# Accuracy (MAE)
# ---------------------------------
st.subheader("Accuracy (MAE)")
try:
    acc = load_accuracy(frm, to, model=model, city=city, max_lead=max_lead)
    buckets = (acc or {}).get("buckets", [])
    if buckets:
        df_acc = pd.DataFrame(buckets).sort_values("lead_days")
        if df_acc["n"].fillna(0).sum() == 0:
            st.info("Noch keine Vergleichspaare (lead ≥ 1 vs. lead = 0) im gewählten Zeitraum.")

        c1, c2 = st.columns(2)
        with c1:
            fig_min = px.line(df_acc, x="lead_days", y="temp_min_mae", markers=True,
                              title=f"Temp MIN MAE (°C) – {model} @ {city}")
            fig_min.update_layout(xaxis_title="Lead (Tage)", yaxis_title="MAE (°C)")
            st.plotly_chart(fig_min, use_container_width=True)
            caption_mae("Temp MIN", df_acc, model, city, frm, to)

        with c2:
            fig_max = px.line(df_acc, x="lead_days", y="temp_max_mae", markers=True,
                              title=f"Temp MAX MAE (°C) – {model} @ {city}")
            fig_max.update_layout(xaxis_title="Lead (Tage)", yaxis_title="MAE (°C)")
            st.plotly_chart(fig_max, use_container_width=True)
            caption_mae("Temp MAX", df_acc, model, city, frm, to)

        c3, c4 = st.columns(2)
        with c3:
            fig_w = px.line(df_acc, x="lead_days", y="wind_mae", markers=True,
                            title=f"Wind MAE (m/s) – {model} @ {city}")
            fig_w.update_layout(xaxis_title="Lead (Tage)", yaxis_title="MAE (m/s)")
            st.plotly_chart(fig_w, use_container_width=True)
            caption_mae("Wind", df_acc, model, city, frm, to)

        with c4:
            fig_r = px.line(df_acc, x="lead_days", y="rain_mae", markers=True,
                            title=f"Regen MAE (mm) – {model} @ {city}")
            fig_r.update_layout(xaxis_title="Lead (Tage)", yaxis_title="MAE (mm)")
            st.plotly_chart(fig_r, use_container_width=True)
            caption_mae("Regen", df_acc, model, city, frm, to)

        fig_m = px.bar(df_acc, x="lead_days", y="weather_match_pct",
                       title=f"Wetter-String: exakte Treffer (%) – {model} @ {city}")
        fig_m.update_layout(xaxis_title="Lead (Tage)", yaxis_title="Trefferquote (%)")
        st.plotly_chart(fig_m, use_container_width=True)
        caption_weather_string(df_acc, model, city, frm, to)

        with st.expander("Details (Buckets – MAE)"):
            st.dataframe(df_acc, use_container_width=True, hide_index=True)
            st.caption(f"Zeitraum: {acc.get('from_date')} bis {acc.get('to_date')} – Modell: {acc.get('model')} – Stadt: {acc.get('city')}")
    else:
        st.info("Keine Accuracy-Daten im Zeitraum.")
except Exception as e:
    st.error(f"Fehler beim Laden der Accuracy: {e}")

st.divider()

# ---------------------------------
# Bias (signed error)
# ---------------------------------
st.subheader("Bias (signed error: Forecast − Observation)")
try:
    rows = load_data_window(days_back, 7, model=model, city=city)
    df_all = pd.DataFrame(rows)
    if df_all.empty:
        st.info("Keine Rohdaten verfügbar.")
    else:
        df_bias = compute_bias_buckets(df_all, max_lead=max_lead).sort_values("lead_days")

        c1, c2 = st.columns(2)
        with c1:
            fig_bmin = px.line(df_bias, x="lead_days", y="temp_min_bias", markers=True,
                               title=f"Bias Temp MIN (°C) – {model} @ {city}")
            fig_bmin.update_layout(xaxis_title="Lead (Tage)", yaxis_title="Bias (°C)")
            st.plotly_chart(fig_bmin, use_container_width=True)
            caption_bias("Temp MIN", df_bias, model, city, frm, to)

        with c2:
            fig_bmax = px.line(df_bias, x="lead_days", y="temp_max_bias", markers=True,
                               title=f"Bias Temp MAX (°C) – {model} @ {city}")
            fig_bmax.update_layout(xaxis_title="Lead (Tage)", yaxis_title="Bias (°C)")
            st.plotly_chart(fig_bmax, use_container_width=True)
            caption_bias("Temp MAX", df_bias, model, city, frm, to)

        c3, c4 = st.columns(2)
        with c3:
            fig_bwind = px.line(df_bias, x="lead_days", y="wind_bias", markers=True,
                                title=f"Bias Wind (m/s) – {model} @ {city}")
            fig_bwind.update_layout(xaxis_title="Lead (Tage)", yaxis_title="Bias (m/s)")
            st.plotly_chart(fig_bwind, use_container_width=True)
            caption_bias("Wind", df_bias, model, city, frm, to)

        with c4:
            fig_brain = px.line(df_bias, x="lead_days", y="rain_bias", markers=True,
                                title=f"Bias Regen (mm) – {model} @ {city}")
            fig_brain.update_layout(xaxis_title="Lead (Tage)", yaxis_title="Bias (mm)")
            st.plotly_chart(fig_brain, use_container_width=True)
            caption_bias("Regen", df_bias, model, city, frm, to)

        with st.expander("Details (Buckets – Bias)"):
            st.dataframe(df_bias, use_container_width=True, hide_index=True)
except Exception as e:
    st.error(f"Fehler beim Berechnen des Bias: {e}")

st.divider()

# ---------------------------------
# Pivot-Tabellen im Excel-Stil
# ---------------------------------
st.subheader("Pivot-Tabellen je Variable (Zeilen = Tage, Spalten = Leads −7…0)")

def _lead_to_negcol(lead: int) -> int:
    return -lead  # Darstellung wie im Screenshot: -7 … -1, 0

def _build_pivot(df_all: pd.DataFrame, var: str, thresholds: tuple[float,float]) -> pd.io.formats.style.Styler:
    """
    Baut eine Pivot-Tabelle je Variable:
    Index: target_date (aufsteigend)
    Columns: -7..0 (negativ dargestellte Leads)
    Values: Vorhersage je Lead, für 0 die Beobachtung.
    Zellenfärbung (nicht bei Spalte 0): anhand |Forecast - Obs| und thresholds.
    """
    if df_all.empty:
        return pd.DataFrame().style

    cols = ["target_date", "lead_days", var]
    df = df_all[cols].copy()
    df["neg_lead"] = df["lead_days"].apply(_lead_to_negcol)

    pv = df.pivot_table(index="target_date", columns="neg_lead", values=var, aggfunc="first").sort_index()
    for col in range(-7, 1):
        if col not in pv.columns:
            pv[col] = np.nan
    pv = pv[sorted(pv.columns)]

    # Fehler vs. Beobachtung
    pv_err = pv.copy()
    base = pv[0] if 0 in pv.columns else pd.Series(index=pv.index, dtype=float)
    for c in pv.columns:
        pv_err[c] = np.nan if c == 0 else (pv[c] - base).abs()

    thr_g, thr_o = thresholds
    def colorize(val, err):
        if pd.isna(val) or pd.isna(err): return ""
        if err <= thr_g: return "background-color: #e6f4ea"   # grün
        if err <= thr_o: return "background-color: #fff4e5"   # orange
        return "background-color: #fde8e8"                    # rot

    styles = pv.copy()
    for r in pv.index:
        for c in pv.columns:
            styles.loc[r, c] = colorize(pv.loc[r, c], pv_err.loc[r, c])

    pv_with_date = pv.copy()
    pv_with_date.insert(0, "date", pv_with_date.index.astype(str))
    styled = pv_with_date.style.format(precision=2).apply(lambda _: styles, axis=None)
    return styled

try:
    rows = load_data_window(days_back, 7, model=model, city=city)
    dfr = pd.DataFrame(rows)
    if dfr.empty:
        st.info("Keine Daten im aktuellen Fenster.")
    else:
        dfr = dfr.sort_values(["target_date", "lead_days"], ascending=[True, False])

        for var in NUM_VARS.keys():
            st.markdown(f"**{NUM_VARS[var]['title']}**")
            styler = _build_pivot(dfr, var, thresholds[var])
            st.dataframe(styler, use_container_width=True)
        st.markdown("**Aussichten (Wetter-String)**")
        dfw = dfr[["target_date", "lead_days", "weather"]].copy()
        dfw["neg_lead"] = dfw["lead_days"].apply(_lead_to_negcol)
        pvw = dfw.pivot_table(index="target_date", columns="neg_lead", values="weather", aggfunc="first").sort_index()
        for col in range(-7, 1):
            if col not in pvw.columns: pvw[col] = np.nan
        pvw = pvw[sorted(pvw.columns)]
        pvw.insert(0, "date", pvw.index.astype(str))
        st.dataframe(pvw, use_container_width=True)
except Exception as e:
    st.error(f"Fehler beim Erstellen der Tabellen: {e}")

# ---------------------------------
# Rohdaten (zur Kontrolle)
# ---------------------------------
st.divider()
with st.expander("Rohdaten (zur Kontrolle)"):
    try:
        rows = load_data_window(days_back, 7, model=model, city=city)
        dfr = pd.DataFrame(rows)
        if not dfr.empty:
            dfr = dfr.sort_values(["target_date", "lead_days"], ascending=[True, False])
            st.dataframe(dfr, use_container_width=True, hide_index=True)
            st.caption(f"{len(dfr)} Zeilen")
        else:
            st.info("Keine Daten im aktuellen Fenster. Prüfe Modell/Stadt oder Zeitfenster.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Rohdaten: {e}")

# ---------------------------------
# Logs
# ---------------------------------
st.divider()
st.subheader("Logs (weather.log)")
if "show_weather_logs" not in st.session_state:
    st.session_state.show_weather_logs = False
colx, coly = st.columns([1,1])
with colx:
    if st.button("📜 Load logs"):
        st.session_state.show_weather_logs = True; st.rerun()
with coly:
    if st.button("🗑️ Delete all logs"):
        try:
            delete_weather_logs()
            st.session_state.show_weather_logs = False
            st.success("Alle Logs gelöscht."); st.rerun()
        except Exception as e:
            st.error(f"Fehler beim Löschen der Logs: {e}")

if st.session_state.show_weather_logs:
    try:
        logs = get_weather_logs(limit=2000, asc=True)
        dfl = pd.DataFrame(logs)
        if not dfl.empty:
            dfl["timestamp"] = pd.to_datetime(dfl["timestamp"], utc=True, errors="coerce")
            st.dataframe(dfl, use_container_width=True, hide_index=True)
            st.caption(f"{len(dfl)} Log-Einträge geladen.")
        else:
            st.info("Keine Logs vorhanden.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Logs: {e}")
