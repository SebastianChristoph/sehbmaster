# streamlit/pages/02_Weatherwatch.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import date, timedelta
from typing import List, Dict

from api_client import (
    get_weather_accuracy,
    get_weather_data,
    get_weather_logs,
    delete_weather_logs,
)

st.set_page_config(page_title="sehbmaster – Weatherwatch", page_icon="⛅", layout="wide")
st.title("⛅ Weatherwatch")
st.caption("Vorhersage-Güte (Lead 1..N) vs. Beobachtung (Lead 0). Temperaturen: Min/Max separat. Zusätzlich Bias und Pivot-Tabellen je Variable. Mit Brier/Directional Accuracy für Regenwahrscheinlichkeit.")

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
CITY_ALL_LABEL = "ALL (alle Städte)"

NUM_VARS = {
    "temp_min_c": {"title": "Min Temperatur (°C)", "unit": "°C"},
    "temp_max_c": {"title": "Max Temperatur (°C)", "unit": "°C"},
    "wind_mps":   {"title": "Wind (m/s)", "unit": "m/s"},
    "rain_mm":    {"title": "Regen (mm)", "unit": "mm"},
}

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
    city = st.selectbox("Stadt", options=[CITY_ALL_LABEL] + KNOWN_CITIES, index=1)

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
    return ", ".join(str(x) for x in leads) if leads else "–"

def caption_mae(kind: str, df_acc: pd.DataFrame, model: str, city_lbl: str, frm: date, to: date):
    n_total = int(df_acc["n"].fillna(0).sum())
    leads_ok = df_acc.loc[df_acc["n"].fillna(0) > 0, "lead_days"]
    st.caption(
        f"{kind} – MAE = mittlerer absoluter Fehler |Vorhersage − Beobachtung|. "
        f"Zeitraum: {frm} bis {to}. Leads: {list_leads_used(leads_ok)}. "
        f"Datenpunkte (Paare): {n_total}. Quelle: Backend /api/weather/accuracy – Modell: {model}, Stadt: {city_lbl}."
    )

def caption_weather_string(df_acc: pd.DataFrame, model: str, city_lbl: str, frm: date, to: date):
    n_total = int(df_acc["n"].fillna(0).sum())
    leads_ok = df_acc.loc[df_acc["n"].fillna(0) > 0, "lead_days"]
    st.caption(
        f"Wetter-String (exakte Übereinstimmung in %) – Anteil exakt identischer Texte. "
        f"Zeitraum: {frm} bis {to}. Leads: {list_leads_used(leads_ok)}. "
        f"Datenpunkte (Paare): {n_total}. Quelle: Backend /api/weather/accuracy – Modell: {model}, Stadt: {city_lbl}."
    )

def caption_bias(kind: str, df_bias: pd.DataFrame, model: str, city_lbl: str, frm: date, to: date):
    n_total = int(df_bias["n"].fillna(0).sum())
    leads_ok = df_bias.loc[df_bias["n"].fillna(0) > 0, "lead_days"]
    st.caption(
        f"{kind} – Bias = Mittel der Signed Errors (Vorhersage − Beobachtung). "
        f"> 0 = Überschätzung, < 0 = Unterschätzung. Zeitraum: {frm} bis {to}. "
        f"Leads: {list_leads_used(leads_ok)}. Datenpunkte (Paare): {n_total}. "
        f"Quelle: Backend /api/weather/data (clientseitig berechnet) – Modell: {model}, Stadt: {city_lbl}."
    )

# ---------------------------------
# Helper: Aggregation über Städte
# ---------------------------------
def weighted_merge_accuracy(acc_list: List[Dict]) -> pd.DataFrame:
    """Führt mehrere Accuracy-Bucket-Listen (verschiedener Städte) korrekt gewichtet zusammen."""
    if not acc_list:
        return pd.DataFrame()
    frames = []
    for acc in acc_list:
        b = (acc or {}).get("buckets", [])
        if not b:
            continue
        df = pd.DataFrame(b)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)

    # Gruppieren nach Lead und gewichtet mitteln
    def wavg(series, weights):
        m = series.notna() & weights.notna() & (weights > 0)
        return (series[m] * weights[m]).sum() / weights[m].sum() if m.any() else np.nan

    g = df_all.groupby("lead_days", as_index=False).apply(
        lambda d: pd.Series({
            "n": int(d["n"].fillna(0).sum()),
            "temp_min_mae": wavg(d.get("temp_min_mae"), d["n"]),
            "temp_max_mae": wavg(d.get("temp_max_mae"), d["n"]),
            "wind_mae":     wavg(d.get("wind_mae"),     d["n"]),
            "rain_mae":     wavg(d.get("rain_mae"),     d["n"]),
            "weather_match_pct": wavg(d.get("weather_match_pct"), d["n"]),
            # NEU: Wahrscheinlichkeitsmetriken (gewichtete Mittel; ohne separates prob_n approximieren wir mit n)
            "rain_prob_brier":        wavg(d.get("rain_prob_brier"), d["n"]),
            "rain_prob_diracc_pct":   wavg(d.get("rain_prob_diracc_pct"), d["n"]),
            "rain_prob_mae_pctpts":   wavg(d.get("rain_prob_mae_pctpts"), d["n"]),
        })
    ).reset_index(drop=True).sort_values("lead_days")
    return g

def compute_bias_buckets_grouped(df: pd.DataFrame, max_lead: int) -> pd.DataFrame:
    """
    Bias korrekt über Städte aggregieren:
    1) je Stadt Observationen joinen, signed errors berechnen,
    2) je Lead Mittel und n,
    3) über Städte nach n gewichten.
    """
    if df.empty:
        return pd.DataFrame({"lead_days": [], "n": []})

    rows = []
    for city_name, dcity in df.groupby("city"):
        obs = dcity[dcity["lead_days"] == 0].set_index("target_date")
        for d in range(1, max_lead + 1):
            fc = dcity[dcity["lead_days"] == d].copy()
            if fc.empty:
                rows.append({"city": city_name, "lead_days": d, "n": 0}); continue
            merged = fc.merge(
                obs[["temp_min_c", "temp_max_c", "wind_mps", "rain_mm", "weather"]],
                left_on="target_date", right_index=True, suffixes=("", "_obs")
            )
            def signed_mean(a, b):
                mask = a.notna() & b.notna()
                return float((a[mask] - b[mask]).mean()) if mask.any() else np.nan
            rows.append({
                "city": city_name,
                "lead_days": d,
                "n": int(len(merged)),
                "temp_min_bias": signed_mean(merged["temp_min_c"], merged["temp_min_c_obs"]),
                "temp_max_bias": signed_mean(merged["temp_max_c"], merged["temp_max_c_obs"]),
                "wind_bias":     signed_mean(merged["wind_mps"], merged["wind_mps_obs"]),
                "rain_bias":     signed_mean(merged["rain_mm"], merged["rain_mm_obs"]),
            })
    df_city = pd.DataFrame(rows)
    if df_city.empty:
        return pd.DataFrame({"lead_days": [], "n": []})

    def wavg(series, weights):
        m = series.notna() & weights.notna() & (weights > 0)
        return (series[m] * weights[m]).sum() / weights[m].sum() if m.any() else np.nan

    g = df_city.groupby("lead_days", as_index=False).apply(
        lambda d: pd.Series({
            "n": int(d["n"].fillna(0).sum()),
            "temp_min_bias": wavg(d["temp_min_bias"], d["n"]),
            "temp_max_bias": wavg(d["temp_max_bias"], d["n"]),
            "wind_bias":     wavg(d["wind_bias"],     d["n"]),
            "rain_bias":     wavg(d["rain_bias"],     d["n"]),
        })
    ).reset_index(drop=True).sort_values("lead_days")
    return g

# ---------------------------------
# Accuracy (MAE + Probabilistik)
# ---------------------------------
st.subheader("Accuracy (MAE & Probabilistik)")
try:
    if city == CITY_ALL_LABEL:
        acc_list = []
        for c in KNOWN_CITIES:
            acc_list.append(load_accuracy(frm, to, model=model, city=c, max_lead=max_lead))
        df_acc = weighted_merge_accuracy(acc_list)
        city_label = "ALL"
    else:
        acc = load_accuracy(frm, to, model=model, city=city, max_lead=max_lead)
        df_acc = pd.DataFrame((acc or {}).get("buckets", [])).sort_values("lead_days")
        city_label = city

    if not df_acc.empty:
        if df_acc["n"].fillna(0).sum() == 0:
            st.info("Noch keine Vergleichspaare (lead ≥ 1 vs. lead = 0) im gewählten Zeitraum.")

        c1, c2 = st.columns(2)
        with c1:
            fig_min = px.line(df_acc, x="lead_days", y="temp_min_mae", markers=True,
                              title=f"Temp MIN MAE (°C) – {model} @ {city_label}")
            fig_min.update_layout(xaxis_title="Lead (Tage)", yaxis_title="MAE (°C)")
            st.plotly_chart(fig_min, width="stretch")
            caption_mae("Temp MIN", df_acc, model, city_label, frm, to)

        with c2:
            fig_max = px.line(df_acc, x="lead_days", y="temp_max_mae", markers=True,
                              title=f"Temp MAX MAE (°C) – {model} @ {city_label}")
            fig_max.update_layout(xaxis_title="Lead (Tage)", yaxis_title="MAE (°C)")
            st.plotly_chart(fig_max, width="stretch")
            caption_mae("Temp MAX", df_acc, model, city_label, frm, to)

        c3, c4 = st.columns(2)
        with c3:
            fig_w = px.line(df_acc, x="lead_days", y="wind_mae", markers=True,
                            title=f"Wind MAE (m/s) – {model} @ {city_label}")
            fig_w.update_layout(xaxis_title="Lead (Tage)", yaxis_title="MAE (m/s)")
            st.plotly_chart(fig_w, width="stretch")
            caption_mae("Wind", df_acc, model, city_label, frm, to)

        with c4:
            fig_r = px.line(df_acc, x="lead_days", y="rain_mae", markers=True,
                            title=f"Regen MAE (mm) – {model} @ {city_label}")
            fig_r.update_layout(xaxis_title="Lead (Tage)", yaxis_title="MAE (mm)")
            st.plotly_chart(fig_r, width="stretch")
            caption_mae("Regen", df_acc, model, city_label, frm, to)

        # Wetter-String (exakte Übereinstimmung)
        fig_m = px.bar(df_acc, x="lead_days", y="weather_match_pct",
                       title=f"Wetter-String: exakte Treffer (%) – {model} @ {city_label}")
        fig_m.update_layout(xaxis_title="Lead (Tage)", yaxis_title="Trefferquote (%)")
        st.plotly_chart(fig_m, width="stretch")
        caption_weather_string(df_acc, model, city_label, frm, to)

        # ---- NEU: Probabilistische Metriken für Regenwahrscheinlichkeit ----
        if "rain_prob_brier" in df_acc.columns:
            fig_bs = px.line(df_acc, x="lead_days", y="rain_prob_brier", markers=True,
                             title=f"Brier Score Regenwahrscheinlichkeit – {model} @ {city_label}")
            fig_bs.update_layout(xaxis_title="Lead (Tage)", yaxis_title="Brier (niedriger besser)")
            st.plotly_chart(fig_bs, width="stretch")
            st.caption(
                "Brier Score: mittl. quadratischer Fehler zwischen Vorhersage-p (0..1) und Ereignis (0/1, Regen ≥ 0.1 mm). "
                f"Zeitraum: {frm} bis {to}. Leads: {list_leads_used(df_acc.loc[df_acc['n'].fillna(0) > 0, 'lead_days'])}."
            )

        if "rain_prob_diracc_pct" in df_acc.columns:
            fig_da = px.bar(df_acc, x="lead_days", y="rain_prob_diracc_pct",
                            title=f"Regen-Wahrscheinlichkeit: Directional Accuracy @50% – {model} @ {city_label}")
            fig_da.update_layout(xaxis_title="Lead (Tage)", yaxis_title="Trefferquote (%)")
            st.plotly_chart(fig_da, width="stretch")
            st.caption(
                "Directional Accuracy @50%: Anteil der Fälle, in denen p≥50% korrekt ein Regen-Ereignis (≥0.1 mm) vorhersagt "
                "bzw. p<50% korrekt kein Ereignis. "
                f"Zeitraum: {frm} bis {to}."
            )

        if "rain_prob_mae_pctpts" in df_acc.columns:
            fig_pm = px.line(df_acc, x="lead_days", y="rain_prob_mae_pctpts", markers=True,
                             title=f"MAE (%-Punkte) Regenwahrscheinlichkeit – {model} @ {city_label}")
            fig_pm.update_layout(xaxis_title="Lead (Tage)", yaxis_title="MAE (%-Punkte)")
            st.plotly_chart(fig_pm, width="stretch")
            st.caption(
                "Mittlerer absoluter Fehler in %-Punkten zwischen p und beobachtetem Ereignis (0/100)."
            )
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
    if city == CITY_ALL_LABEL:
        frames = []
        for c in KNOWN_CITIES:
            rows = load_data_window(days_back, 7, model=model, city=c)
            if rows:
                frames.append(pd.DataFrame(rows))
        df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        city_label = "ALL"
    else:
        rows = load_data_window(days_back, 7, model=model, city=city)
        df_all = pd.DataFrame(rows)
        city_label = city

    if df_all.empty:
        st.info("Keine Rohdaten verfügbar.")
    else:
        df_bias = compute_bias_buckets_grouped(df_all, max_lead=max_lead).sort_values("lead_days")

        c1, c2 = st.columns(2)
        with c1:
            fig_bmin = px.line(df_bias, x="lead_days", y="temp_min_bias", markers=True,
                               title=f"Bias Temp MIN (°C) – {model} @ {city_label}")
            fig_bmin.update_layout(xaxis_title="Lead (Tage)", yaxis_title="Bias (°C)")
            st.plotly_chart(fig_bmin, width="stretch")
            caption_bias("Temp MIN", df_bias, model, city_label, frm, to)

        with c2:
            fig_bmax = px.line(df_bias, x="lead_days", y="temp_max_bias", markers=True,
                               title=f"Bias Temp MAX (°C) – {model} @ {city_label}")
            fig_bmax.update_layout(xaxis_title="Lead (Tage)", yaxis_title="Bias (°C)")
            st.plotly_chart(fig_bmax, width="stretch")
            caption_bias("Temp MAX", df_bias, model, city_label, frm, to)

        c3, c4 = st.columns(2)
        with c3:
            fig_bwind = px.line(df_bias, x="lead_days", y="wind_bias", markers=True,
                                title=f"Bias Wind (m/s) – {model} @ {city_label}")
            fig_bwind.update_layout(xaxis_title="Lead (Tage)", yaxis_title="Bias (m/s)")
            st.plotly_chart(fig_bwind, width="stretch")
            caption_bias("Wind", df_bias, model, city_label, frm, to)

        with c4:
            fig_brain = px.line(df_bias, x="lead_days", y="rain_bias", markers=True,
                                title=f"Bias Regen (mm) – {model} @ {city_label}")
            fig_brain.update_layout(xaxis_title="Lead (Tage)", yaxis_title="Bias (mm)")
            st.plotly_chart(fig_brain, width="stretch")
            caption_bias("Regen", df_bias, model, city_label, frm, to)
except Exception as e:
    st.error(f"Fehler beim Berechnen des Bias: {e}")

st.divider()

# ---------------------------------
# Pivot-Tabellen im Excel-Stil
# ---------------------------------
st.subheader("Pivot-Tabellen je Variable (Zeilen = Tage, Spalten = Leads −7…0)")

def _lead_to_negcol(lead: int) -> int:
    return -lead  # Darstellung: -7 … -1, 0

def _build_pivot(df_all: pd.DataFrame, var: str, thresholds: tuple[float,float]):
    """
    Für city = Einzelstadt: Index = target_date.
    Für city = ALL: Index = (city, target_date), damit Beobachtungen nicht gekreuzt werden.
    Spalten: -7..0 (negativ dargestellte Leads). Zellenfärbung vs. Spalte 0 (Beobachtung).
    """
    if df_all.empty:
        return pd.DataFrame()

    cols = ["target_date", "lead_days", var, "city"]
    df = df_all[cols].copy()
    df["neg_lead"] = df["lead_days"].apply(_lead_to_negcol)

    index_cols = ["target_date"] if df["city"].nunique() == 1 else ["city", "target_date"]
    pv = df.pivot_table(index=index_cols, columns="neg_lead", values=var, aggfunc="first").sort_index()

    # alle Spalten sicher anlegen
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
    def colorize(err):
        if pd.isna(err): return ""
        if err <= thr_g: return "background-color: #e6f4ea"   # grün
        if err <= thr_o: return "background-color: #fff4e5"   # orange
        return "background-color: #fde8e8"                    # rot

    # Build a Styler by applying per-cell styles
    styled = pv.copy()
    for c in pv.columns:
        if c == 0:
            styled[c] = ""  # Beobachtung nicht färben
        else:
            styled[c] = pv_err[c].apply(colorize)

    # für Anzeige: Indexspalten als echte Spalten ausgeben
    pv_show = pv.copy()
    if isinstance(pv_show.index, pd.MultiIndex):
        pv_show.insert(0, "city", [idx[0] for idx in pv_show.index])
        pv_show.insert(1, "date", [str(idx[1]) for idx in pv_show.index])
    else:
        pv_show.insert(0, "date", pv_show.index.astype(str))

    styler = pv_show.style.format(precision=2).apply(lambda _: styled, axis=None)
    return styler

try:
    if city == CITY_ALL_LABEL:
        frames = []
        for c in KNOWN_CITIES:
            rows = load_data_window(days_back, 7, model=model, city=c)
            if rows:
                frames.append(pd.DataFrame(rows))
        dfr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        city_label = "ALL"
    else:
        rows = load_data_window(days_back, 7, model=model, city=city)
        dfr = pd.DataFrame(rows)
        city_label = city

    if dfr.empty:
        st.info("Keine Daten im aktuellen Fenster.")
    else:
        dfr = dfr.sort_values(["city", "target_date", "lead_days"], ascending=[True, True, False])

        for var in NUM_VARS.keys():
            st.markdown(f"**{NUM_VARS[var]['title']} – {model} @ {city_label}**")
            styler = _build_pivot(dfr, var, thresholds[var])
            if isinstance(styler, pd.io.formats.style.Styler) or hasattr(styler, "to_html"):
                st.dataframe(styler, width="stretch")
            else:
                st.dataframe(styler, width="stretch")

        st.markdown(f"**Aussichten (Wetter-String) – {model} @ {city_label}**")
        dfw = dfr[["target_date", "lead_days", "weather", "city"]].copy()
        dfw["neg_lead"] = dfw["lead_days"].apply(_lead_to_negcol)
        index_cols = ["target_date"] if dfw["city"].nunique() == 1 else ["city", "target_date"]
        pvw = dfw.pivot_table(index=index_cols, columns="neg_lead", values="weather", aggfunc="first").sort_index()
        for col in range(-7, 1):
            if col not in pvw.columns: pvw[col] = np.nan
        pvw = pvw[sorted(pvw.columns)]
        if isinstance(pvw.index, pd.MultiIndex):
            pvw.insert(0, "city", [idx[0] for idx in pvw.index])
            pvw.insert(1, "date", [str(idx[1]) for idx in pvw.index])
        else:
            pvw.insert(0, "date", pvw.index.astype(str))
        st.dataframe(pvw, width="stretch")
except Exception as e:
    st.error(f"Fehler beim Erstellen der Tabellen: {e}")

# ---------------------------------
# Rohdaten (zur Kontrolle)
# ---------------------------------
st.divider()
with st.expander("Rohdaten (zur Kontrolle)"):
    try:
        if city == CITY_ALL_LABEL:
            frames = []
            for c in KNOWN_CITIES:
                rows = load_data_window(days_back, 7, model=model, city=c)
                if rows:
                    frames.append(pd.DataFrame(rows))
            dfr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            rows = load_data_window(days_back, 7, model=model, city=city)
            dfr = pd.DataFrame(rows)

        if not dfr.empty:
            dfr = dfr.sort_values(["city", "target_date", "lead_days"], ascending=[True, True, False])
            st.dataframe(dfr, width="stretch", hide_index=True)
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
            st.dataframe(dfl, width="stretch", hide_index=True)
            st.caption(f"{len(dfl)} Log-Einträge geladen.")
        else:
            st.info("Keine Logs vorhanden.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Logs: {e}")
