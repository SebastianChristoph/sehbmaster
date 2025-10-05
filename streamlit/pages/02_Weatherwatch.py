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

st.set_page_config(page_title="Weatherwatch – Model Verification", page_icon="⛅", layout="wide")
st.title("⛅ Weatherwatch (Model Verification)")
st.caption(
    "Daily verification of forecast models against observations. "
    "Temperatures use separate min/max. We also evaluate wind, precipitation, and probability of precipitation (PoP) with probabilistic metrics."
)

# ---------------------------------
# Constants
# ---------------------------------
KNOWN_MODELS = ["open-meteo", "metno", "wettercom", "default"]
KNOWN_CITIES = [
    "berlin", "hamburg", "muenchen", "koeln", "frankfurt", "stuttgart",
    "duesseldorf", "dortmund", "essen", "leipzig", "bremen", "dresden",
    "hannover", "nuernberg", "duisburg", "bochum", "wuppertal", "bielefeld",
    "bonn", "muenster", "rostock"
]
CITY_ALL_LABEL = "ALL (all cities)"

NUM_VARS = {
    "temp_min_c": {"title": "Min temperature (°C)", "unit": "°C"},
    "temp_max_c": {"title": "Max temperature (°C)", "unit": "°C"},
    "wind_mps":   {"title": "Wind (m/s)", "unit": "m/s"},
    "rain_mm":    {"title": "Precipitation (mm)", "unit": "mm"},
}

# Traffic-light thresholds for absolute error |forecast − observation|
DEFAULT_THRESHOLDS = {
    "temp_min_c": (1.0, 2.0),   # <= green, <= orange, else red
    "temp_max_c": (1.0, 2.0),
    "wind_mps":   (0.6, 1.2),
    "rain_mm":    (0.5, 1.5),
}

# Probability of precipitation (PoP): thresholds for error in %-points vs. observed event 0/100
DEFAULT_THRESHOLDS_PROB = (10.0, 25.0)  # <=10 green, <=25 orange, else red
# Event definition for “rain observed”
RAIN_EVENT_THRESHOLD_MM = 0.1

# ---------------------------------
# Cached loaders
# ---------------------------------
@st.cache_data(ttl=30)
def load_accuracy(frm: date, to: date, model: str, city: str, max_lead: int):
    return get_weather_accuracy(frm.isoformat(), to.isoformat(), model=model, city=city, max_lead=max_lead)

@st.cache_data(ttl=30)
def load_data_window(days_back: int, days_forward: int, model: str, city: str):
    # include future so new forecasts appear immediately
    window_to = date.today() + timedelta(days=days_forward)
    window_from = date.today() - timedelta(days=days_back)
    return get_weather_data(
        window_from.isoformat(), window_to.isoformat(),
        model=model, city=city, lead_days=None, limit=10000
    )

# ---------------------------------
# Controls
# ---------------------------------
colA, colB, colC, colD = st.columns([1,1,1,1])
with colA:
    days_back = st.slider("Time window (days back)", 7, 180, 35)
with colB:
    max_lead = st.slider("Max lead (days ahead)", 0, 7, 7)
with colC:
    model = st.selectbox("Model", options=KNOWN_MODELS, index=0)
with colD:
    city = st.selectbox("City", options=[CITY_ALL_LABEL] + KNOWN_CITIES, index=1)

frm = date.today() - timedelta(days=days_back)
to = date.today()

with st.sidebar:
    st.header("Color thresholds")
    st.caption("Thresholds used for cell colors in the pivot tables (absolute error vs. observation).")
    thr_temp_g, thr_temp_o = st.slider("Temp (°C): green/orange", 0.0, 5.0, DEFAULT_THRESHOLDS["temp_min_c"][0]), \
                             st.slider("Temp (°C): orange/red",   0.0, 6.0, DEFAULT_THRESHOLDS["temp_min_c"][1])
    thr_wind_g, thr_wind_o = st.slider("Wind (m/s): green/orange", 0.0, 3.0, DEFAULT_THRESHOLDS["wind_mps"][0]), \
                             st.slider("Wind (m/s): orange/red",   0.0, 4.0, DEFAULT_THRESHOLDS["wind_mps"][1])
    thr_rain_g, thr_rain_o = st.slider("Precip (mm): green/orange", 0.0, 5.0, DEFAULT_THRESHOLDS["rain_mm"][0]), \
                             st.slider("Precip (mm): orange/red",   0.0, 8.0, DEFAULT_THRESHOLDS["rain_mm"][1])

    thresholds = {
        "temp_min_c": (thr_temp_g, thr_temp_o),
        "temp_max_c": (thr_temp_g, thr_temp_o),
        "wind_mps":   (thr_wind_g, thr_wind_o),
        "rain_mm":    (thr_rain_g, thr_rain_o),
    }

    # PoP thresholds in %-points
    thr_prob_g = st.slider("PoP error (%-points): green/orange", 0.0, 50.0, DEFAULT_THRESHOLDS_PROB[0])
    thr_prob_o = st.slider("PoP error (%-points): orange/red",   0.0, 60.0, DEFAULT_THRESHOLDS_PROB[1])
    thresholds_prob = (thr_prob_g, thr_prob_o)

if st.button("🔄 Refresh"):
    load_accuracy.clear(); load_data_window.clear(); st.rerun()

# ---------------------------------
# Caption helpers (transparent reporting)
# ---------------------------------
def list_leads_used(series: pd.Series) -> str:
    leads = sorted([int(x) for x in series.tolist()])
    return ", ".join(str(x) for x in leads) if leads else "–"

def caption_mae(kind: str, df_acc: pd.DataFrame, model: str, city_lbl: str, frm: date, to: date):
    n_total = int(df_acc.get("n", pd.Series(dtype=float)).fillna(0).sum())
    leads_ok = df_acc.loc[df_acc.get("n", pd.Series(dtype=float)).fillna(0) > 0, "lead_days"]
    st.caption(
        f"{kind} – MAE = mean absolute error |forecast − observation|. "
        f"Window: {frm} to {to}. Leads used: {list_leads_used(leads_ok)}. "
        f"Data points (paired forecast/obs): {n_total}. Source: backend /api/weather/accuracy – model: {model}, city: {city_lbl}."
    )

def caption_weather_string(df_acc: pd.DataFrame, model: str, city_lbl: str, frm: date, to: date):
    n_total = int(df_acc.get("n", pd.Series(dtype=float)).fillna(0).sum())
    leads_ok = df_acc.loc[df_acc.get("n", pd.Series(dtype=float)).fillna(0) > 0, "lead_days"]
    st.caption(
        f"Weather text exact match (%). "
        f"Window: {frm} to {to}. Leads used: {list_leads_used(leads_ok)}. "
        f"Pairs: {n_total}. Source: /api/weather/accuracy – model: {model}, city: {city_lbl}."
    )

def caption_bias(kind: str, df_bias: pd.DataFrame, model: str, city_lbl: str, frm: date, to: date):
    n_total = int(df_bias.get("n", pd.Series(dtype=float)).fillna(0).sum())
    leads_ok = df_bias.loc[df_bias.get("n", pd.Series(dtype=float)).fillna(0) > 0, "lead_days"]
    st.caption(
        f"{kind} – Bias = mean of signed errors (forecast − observation). "
        f"> 0 = overestimation, < 0 = underestimation. Window: {frm} to {to}. "
        f"Leads used: {list_leads_used(leads_ok)}. Pairs: {n_total}. "
        f"Source: /api/weather/data (client-side computation) – model: {model}, city: {city_lbl}."
    )

def caption_pop_pivot(city_lbl: str):
    st.caption(
        f"Cells show forecast probability of precipitation (PoP, %). Column 0 shows the observed event as 0/100 (rain ≥ {RAIN_EVENT_THRESHOLD_MM} mm). "
        "Cell colors reflect absolute error in %-points vs. the observed 0/100. "
        "This is a per-day, per-lead view (higher = wetter forecast)."
    )

# ---------------------------------
# Aggregation across cities
# ---------------------------------
def weighted_merge_accuracy(acc_list: List[Dict]) -> pd.DataFrame:
    """Merge multiple Accuracy bucket lists across cities with n-weighted averages."""
    if not acc_list:
        return pd.DataFrame()
    frames = []
    for acc in acc_list:
        b = (acc or {}).get("buckets", [])
        if not b:
            continue
        frames.append(pd.DataFrame(b))
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)

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
            # probabilistic metrics (n-weighted)
            "rain_prob_brier":        wavg(d.get("rain_prob_brier"), d["n"]),
            "rain_prob_diracc_pct":   wavg(d.get("rain_prob_diracc_pct"), d["n"]),
            "rain_prob_mae_pctpts":   wavg(d.get("rain_prob_mae_pctpts"), d["n"]),
        })
    ).reset_index(drop=True).sort_values("lead_days")
    return g

def compute_bias_buckets_grouped(df: pd.DataFrame, max_lead: int) -> pd.DataFrame:
    """
    Bias aggregation over cities:
      1) per city: join forecasts with same-day observation (lead 0),
      2) per lead: compute mean signed error and n,
      3) across cities: n-weighted mean per lead.
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
# Accuracy (MAE + probabilistic)
# ---------------------------------
st.subheader("Accuracy (MAE & Probabilistic)")
try:
    if city == CITY_ALL_LABEL:
        acc_list = [load_accuracy(frm, to, model=model, city=c, max_lead=max_lead) for c in KNOWN_CITIES]
        df_acc = weighted_merge_accuracy(acc_list)
        city_label = "ALL"
    else:
        acc = load_accuracy(frm, to, model=model, city=city, max_lead=max_lead)
        df_acc = pd.DataFrame((acc or {}).get("buckets", [])).sort_values("lead_days")
        city_label = city

    if not df_acc.empty:
        if df_acc.get("n", pd.Series(dtype=float)).fillna(0).sum() == 0:
            st.info("No comparison pairs (lead ≥ 1 vs. lead = 0) in the selected window yet.")

        c1, c2 = st.columns(2)
        with c1:
            fig_min = px.line(df_acc, x="lead_days", y="temp_min_mae", markers=True,
                              title=f"Temp MIN – MAE (°C) • {model} @ {city_label}")
            fig_min.update_layout(xaxis_title="Lead (days)", yaxis_title="MAE (°C)")
            st.plotly_chart(fig_min, use_container_width=True)
            caption_mae("Temp MIN", df_acc, model, city_label, frm, to)

        with c2:
            fig_max = px.line(df_acc, x="lead_days", y="temp_max_mae", markers=True,
                              title=f"Temp MAX – MAE (°C) • {model} @ {city_label}")
            fig_max.update_layout(xaxis_title="Lead (days)", yaxis_title="MAE (°C)")
            st.plotly_chart(fig_max, use_container_width=True)
            caption_mae("Temp MAX", df_acc, model, city_label, frm, to)

        c3, c4 = st.columns(2)
        with c3:
            fig_w = px.line(df_acc, x="lead_days", y="wind_mae", markers=True,
                            title=f"Wind – MAE (m/s) • {model} @ {city_label}")
            fig_w.update_layout(xaxis_title="Lead (days)", yaxis_title="MAE (m/s)")
            st.plotly_chart(fig_w, use_container_width=True)
            caption_mae("Wind", df_acc, model, city_label, frm, to)

        with c4:
            fig_r = px.line(df_acc, x="lead_days", y="rain_mae", markers=True,
                            title=f"Precipitation – MAE (mm) • {model} @ {city_label}")
            fig_r.update_layout(xaxis_title="Lead (days)", yaxis_title="MAE (mm)")
            st.plotly_chart(fig_r, use_container_width=True)
            caption_mae("Precipitation", df_acc, model, city_label, frm, to)

        fig_m = px.bar(df_acc, x="lead_days", y="weather_match_pct",
                       title=f"Weather text: exact matches (%) • {model} @ {city_label}")
        fig_m.update_layout(xaxis_title="Lead (days)", yaxis_title="Match rate (%)")
        st.plotly_chart(fig_m, use_container_width=True)
        caption_weather_string(df_acc, model, city_label, frm, to)

        # Probabilistic PoP metrics (if available from backend)
        if "rain_prob_brier" in df_acc.columns:
            fig_bs = px.line(df_acc, x="lead_days", y="rain_prob_brier", markers=True,
                             title=f"Brier score (PoP) • {model} @ {city_label}")
            fig_bs.update_layout(xaxis_title="Lead (days)", yaxis_title="Brier (lower is better)")
            st.plotly_chart(fig_bs, use_container_width=True)
            st.caption(
                "Brier score = mean squared error between forecast probability (0..1) and observed event (0/1, rain ≥ 0.1 mm)."
            )
        if "rain_prob_diracc_pct" in df_acc.columns:
            fig_da = px.bar(df_acc, x="lead_days", y="rain_prob_diracc_pct",
                            title=f"Directional accuracy @50% (PoP) • {model} @ {city_label}")
            fig_da.update_layout(xaxis_title="Lead (days)", yaxis_title="Hit rate (%)")
            st.plotly_chart(fig_da, use_container_width=True)
            st.caption("Share of cases where p≥50% correctly predicts an event, or p<50% correctly predicts no event.")
        if "rain_prob_mae_pctpts" in df_acc.columns:
            fig_pm = px.line(df_acc, x="lead_days", y="rain_prob_mae_pctpts", markers=True,
                             title=f"MAE in %-points (PoP vs. 0/100) • {model} @ {city_label}")
            fig_pm.update_layout(xaxis_title="Lead (days)", yaxis_title="MAE (percentage points)")
            st.plotly_chart(fig_pm, use_container_width=True)
            st.caption("Mean absolute error in percentage points between forecast p and observed 0/100 event.")
    else:
        st.info("No accuracy data in the selected window.")
except Exception as e:
    st.error(f"Error loading accuracy: {e}")

st.divider()

# ---------------------------------
# Bias (signed error)
# ---------------------------------
st.subheader("Bias (signed error: forecast − observation)")
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
        st.info("No raw data available.")
    else:
        df_bias = compute_bias_buckets_grouped(df_all, max_lead=max_lead).sort_values("lead_days")

        c1, c2 = st.columns(2)
        with c1:
            fig_bmin = px.line(df_bias, x="lead_days", y="temp_min_bias", markers=True,
                               title=f"Bias – Temp MIN (°C) • {model} @ {city_label}")
            fig_bmin.update_layout(xaxis_title="Lead (days)", yaxis_title="Bias (°C)")
            st.plotly_chart(fig_bmin, use_container_width=True)
            caption_bias("Temp MIN", df_bias, model, city_label, frm, to)

        with c2:
            fig_bmax = px.line(df_bias, x="lead_days", y="temp_max_bias", markers=True,
                               title=f"Bias – Temp MAX (°C) • {model} @ {city_label}")
            fig_bmax.update_layout(xaxis_title="Lead (days)", yaxis_title="Bias (°C)")
            st.plotly_chart(fig_bmax, use_container_width=True)
            caption_bias("Temp MAX", df_bias, model, city_label, frm, to)

        c3, c4 = st.columns(2)
        with c3:
            fig_bwind = px.line(df_bias, x="lead_days", y="wind_bias", markers=True,
                                title=f"Bias – Wind (m/s) • {model} @ {city_label}")
            fig_bwind.update_layout(xaxis_title="Lead (days)", yaxis_title="Bias (m/s)")
            st.plotly_chart(fig_bwind, use_container_width=True)
            caption_bias("Wind", df_bias, model, city_label, frm, to)

        with c4:
            fig_brain = px.line(df_bias, x="lead_days", y="rain_bias", markers=True,
                                title=f"Bias – Precipitation (mm) • {model} @ {city_label}")
            fig_brain.update_layout(xaxis_title="Lead (days)", yaxis_title="Bias (mm)")
            st.plotly_chart(fig_brain, use_container_width=True)
            caption_bias("Precipitation", df_bias, model, city_label, frm, to)
except Exception as e:
    st.error(f"Error computing bias: {e}")

st.divider()

# ---------------------------------
# Pivot tables (Excel-style)
# ---------------------------------
st.subheader("Pivot tables per variable (rows = dates, columns = leads −7…0)")

def _lead_to_negcol(lead: int) -> int:
    return -lead  # display as −7 … −1, 0

def _build_pivot(df_all: pd.DataFrame, var: str, thresholds: tuple[float,float]):
    """
    City = single city: index = target_date.
    City = ALL: index = (city, target_date) to avoid cross-mixing observations.
    Columns: −7..0 (negative leads) – colored by absolute error vs. column 0 (observation).
    """
    if df_all.empty:
        return pd.DataFrame()

    cols = ["target_date", "lead_days", var, "city"]
    df = df_all[cols].copy()
    df["neg_lead"] = df["lead_days"].apply(_lead_to_negcol)

    index_cols = ["target_date"] if df["city"].nunique() == 1 else ["city", "target_date"]
    pv = df.pivot_table(index=index_cols, columns="neg_lead", values=var, aggfunc="first").sort_index()

    # ensure columns −7..0 exist
    for col in range(-7, 1):
        if col not in pv.columns:
            pv[col] = np.nan
    pv = pv[sorted(pv.columns)]

    # absolute error vs. observation
    pv_err = pv.copy()
    base = pv[0] if 0 in pv.columns else pd.Series(index=pv.index, dtype=float)
    for c in pv.columns:
        pv_err[c] = np.nan if c == 0 else (pv[c] - base).abs()

    thr_g, thr_o = thresholds
    def colorize(err):
        if pd.isna(err): return ""
        if err <= thr_g: return "background-color: #e6f4ea"   # green
        if err <= thr_o: return "background-color: #fff4e5"   # orange
        return "background-color: #fde8e8"                    # red

    styled = pv.copy()
    for c in pv.columns:
        styled[c] = "" if c == 0 else pv_err[c].apply(colorize)

    # display index columns
    pv_show = pv.copy()
    if isinstance(pv_show.index, pd.MultiIndex):
        pv_show.insert(0, "city", [idx[0] for idx in pv_show.index])
        pv_show.insert(1, "date", [str(idx[1]) for idx in pv_show.index])
    else:
        pv_show.insert(0, "date", pv_show.index.astype(str))

    styler = pv_show.style.format(precision=2).apply(lambda _: styled, axis=None)
    return styler

# Pivot for probability of precipitation (%)
def _build_pivot_prob(df_all: pd.DataFrame, thresholds_pp: tuple[float, float], event_threshold_mm: float = RAIN_EVENT_THRESHOLD_MM):
    """
    Pivot for probability of precipitation (PoP, %):
      - cells: forecast PoP (%) by lead
      - column 0: observed event as 0/100 (rain ≥ threshold)
      - cell color: absolute error in %-points vs. observed 0/100
    """
    if df_all.empty:
        return pd.DataFrame()

    cols = ["target_date", "lead_days", "rain_probability_pct", "rain_mm", "city"]
    have = [c for c in cols if c in df_all.columns]
    if not {"target_date", "lead_days", "city"}.issubset(have):
        return pd.DataFrame()

    d = df_all[have].copy()
    d["neg_lead"] = d["lead_days"].apply(lambda x: -int(x))

    # observed event (0/100) from lead 0
    obs = d[d["lead_days"] == 0].copy()
    obs["event_0_100"] = np.where(obs["rain_mm"].fillna(0) >= event_threshold_mm, 100.0, 0.0)
    obs_base = obs[["target_date", "city", "event_0_100"]].drop_duplicates()

    index_cols = ["target_date"] if d["city"].nunique() == 1 else ["city", "target_date"]
    pv = d.pivot_table(index=index_cols, columns="neg_lead", values="rain_probability_pct", aggfunc="first").sort_index()

    for col in range(-7, 1):
        if col not in pv.columns:
            pv[col] = np.nan
    pv = pv[sorted(pv.columns)]

    # set column 0 to observed 0/100
    if isinstance(pv.index, pd.MultiIndex):
        base_join = obs_base.set_index(["city", "target_date"])
    else:
        base_join = obs_base.set_index("target_date")
    base_series = base_join["event_0_100"].reindex(pv.index)
    pv[0] = base_series

    # error in %-points vs. base
    pv_err = pv.copy()
    for c in pv.columns:
        pv_err[c] = np.nan if c == 0 else (pv[c] - base_series).abs()

    thr_g, thr_o = thresholds_pp
    def colorize(err):
        if pd.isna(err): return ""
        if err <= thr_g: return "background-color: #e6f4ea"
        if err <= thr_o: return "background-color: #fff4e5"
        return "background-color: #fde8e8"

    styled = pv.copy()
    for c in pv.columns:
        styled[c] = "" if c == 0 else pv_err[c].apply(colorize)

    pv_show = pv.copy()
    if isinstance(pv_show.index, pd.MultiIndex):
        pv_show.insert(0, "city", [idx[0] for idx in pv_show.index])
        pv_show.insert(1, "date", [str(idx[1]) for idx in pv_show.index])
    else:
        pv_show.insert(0, "date", pv_show.index.astype(str))

    styler = pv_show.style.format(precision=1).apply(lambda _: styled, axis=None)
    return styler

# ---- Render pivots ----
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
        st.info("No data in the current window.")
    else:
        dfr = dfr.sort_values(["city", "target_date", "lead_days"], ascending=[True, True, False])

        for var in NUM_VARS.keys():
            st.markdown(f"**{NUM_VARS[var]['title']} • {model} @ {city_label}**")
            styler = _build_pivot(dfr, var, thresholds[var])
            st.dataframe(styler, use_container_width=True)

        st.markdown(f"**Outlook (weather text) • {model} @ {city_label}**")
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
        st.dataframe(pvw, use_container_width=True)
        st.caption("Cells show the forecast weather string at each lead; column 0 is the observation text of the day.")

        st.markdown(f"**Probability of precipitation (PoP, %) • {model} @ {city_label}**")
        styler_prob = _build_pivot_prob(dfr, thresholds_prob)
        st.dataframe(styler_prob, use_container_width=True)
        caption_pop_pivot(city_label)
except Exception as e:
    st.error(f"Error creating tables: {e}")

# ---------------------------------
# Raw data (for inspection)
# ---------------------------------
st.divider()
with st.expander("Raw data (for inspection)"):
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
            st.dataframe(dfr, use_container_width=True, hide_index=True)
            st.caption(f"{len(dfr)} rows")
        else:
            st.info("No data in the current window. Check model/city or the time window.")
    except Exception as e:
        st.error(f"Error loading raw data: {e}")

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
            st.success("All logs deleted."); st.rerun()
        except Exception as e:
            st.error(f"Error deleting logs: {e}")

if st.session_state.show_weather_logs:
    try:
        logs = get_weather_logs(limit=2000, asc=True)
        dfl = pd.DataFrame(logs)
        if not dfl.empty:
            dfl["timestamp"] = pd.to_datetime(dfl["timestamp"], utc=True, errors="coerce")
            st.dataframe(dfl, use_container_width=True, hide_index=True)
            st.caption(f"{len(dfl)} log entries loaded.")
        else:
            st.info("No logs available.")
    except Exception as e:
        st.error(f"Error loading logs: {e}")

# ---------------------------------
# Methodology & Data Notes (for publication)
# ---------------------------------
st.divider()
st.subheader("Methodology & Data Notes")

with st.container():
    # Try to compute simple counts for transparency
    try:
        # reuse df_acc / df_bias / dfr if present
        acc_pairs = int(df_acc.get("n", pd.Series(dtype=float)).fillna(0).sum()) if "df_acc" in locals() else None
    except Exception:
        acc_pairs = None
    try:
        bias_pairs = int(df_bias.get("n", pd.Series(dtype=float)).fillna(0).sum()) if "df_bias" in locals() else None
    except Exception:
        bias_pairs = None
    try:
        raw_rows = int(len(dfr)) if "dfr" in locals() else None
    except Exception:
        raw_rows = None

    bullets = [
        f"**Time window:** {frm} to {to} (local dates).",
        f"**Cities:** {'ALL' if city == CITY_ALL_LABEL else city}.",
        f"**Model:** {model}.",
        f"**Accuracy pairs (lead≥1 vs. lead=0):** {acc_pairs if acc_pairs is not None else 'n/a'}.",
        f"**Bias pairs (per-lead n, n-weighted across cities):** {bias_pairs if bias_pairs is not None else 'n/a'}.",
        f"**Raw rows shown in pivots:** {raw_rows if raw_rows is not None else 'n/a'}.",
        f"**Observation definition for rain events:** rain ≥ {RAIN_EVENT_THRESHOLD_MM} mm ⇒ event = 1 (100%), else 0.",
        "**PoP evaluation:** Brier score (lower is better), directional accuracy @50%, and MAE in percentage points vs. observed 0/100.",
        "**Pivot coloring:** By absolute error vs. observation (for PoP: vs. 0/100 event). Thresholds can be adjusted in the sidebar.",
        "**Data sources:** Open-Meteo API (daily aggregates), MET Norway / Yr API (hourly → daily aggregates), and wetter.com (HTML, 7-day page). "
        "Scraping of wetter.com is for testing/private purposes only; please respect their ToS and robots.txt.",
        "**ETL cadence:** Forecasts (lead 0..7) are upserted daily per model and city. When multiple runs occur in a day, the latest run overwrites earlier ones for the same target date/lead.",
        "**Limitations:** Different providers define daily maxima/minima and PoP differently; wind is max 10-m wind (m/s) as provided/approximated by each source; precipitation sums are daily totals.",
    ]
    for b in bullets:
        st.markdown(f"- {b}")
