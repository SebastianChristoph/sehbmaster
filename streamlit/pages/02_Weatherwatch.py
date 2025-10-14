# streamlit/pages/02_Weatherwatch.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import date, timedelta
from typing import List, Dict, Tuple

import io, base64, calendar, shutil
from textwrap import dedent

from api_client import (
    get_weather_accuracy,
    get_weather_data,
    get_weather_logs,
    delete_weather_logs,
)

# ────────────────────────────────────────────────────────────────────────────────
# Page header
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Weatherwatch – Model Verification", page_icon="⛅", layout="wide")
st.title("⛅ Weatherwatch (Model Verification)")
st.caption(
    "Daily verification of forecast models against observations. "
    "Temperatures use separate min/max. We also evaluate wind, precipitation, and probability of precipitation (PoP) with probabilistic metrics."
)

# ────────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────────
KNOWN_MODELS = ["open-meteo", "metno", "wettercom", "default"]


# Lead-day mapping per model
LEAD_MAP = {
    "open-meteo": list(range(0, 8)),
    "metno":      list(range(0, 8)),
    "wettercom":  list(range(0, 7)),
    "default":    list(range(0, 8)),
}
def allowed_leads_for_model(model: str) -> list[int]:
    return LEAD_MAP.get(model, LEAD_MAP["default"])
def allowed_neg_leads_for_model(model: str) -> list[int]:
    # negative column labels used in the pivots
    return sorted([-l for l in allowed_leads_for_model(model)], reverse=False)

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

# default traffic-light thresholds for |forecast − observation|
DEFAULT_THRESHOLDS = {
    "temp_min_c": (1.0, 2.0),
    "temp_max_c": (1.0, 2.0),
    "wind_mps":   (0.6, 1.2),
    "rain_mm":    (0.5, 1.5),
}
# PoP thresholds in %-points vs. 0/100 observed event
DEFAULT_THRESHOLDS_PROB = (10.0, 25.0)
RAIN_EVENT_THRESHOLD_MM = 0.1

# ────────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_accuracy(frm: date, to: date, model: str, city: str, max_lead: int):
    return get_weather_accuracy(frm.isoformat(), to.isoformat(), model=model, city=city, max_lead=max_lead)

@st.cache_data(ttl=30)
def load_data_window(days_back: int, days_forward: int, model: str, city: str):
    window_to = date.today() + timedelta(days=days_forward)
    window_from = date.today() - timedelta(days=days_back)
    return get_weather_data(
        window_from.isoformat(), window_to.isoformat(),
        model=model, city=city, lead_days=None, limit=10000
    )

@st.cache_data(ttl=300, show_spinner=False)
def load_available_year_months(model: str, city: str) -> list[tuple[int,int]]:
    """Return unique (year, month) pairs present for model/city. For ALL, union across cities."""
    window_from = date(2020, 1, 1)
    window_to   = date.today() + timedelta(days=7)
    frames = []
    if city == CITY_ALL_LABEL:
        for c in KNOWN_CITIES:
            rows = get_weather_data(window_from.isoformat(), window_to.isoformat(),
                                    model=model, city=c, lead_days=None, limit=100000)
            if rows:
                frames.append(pd.DataFrame(rows))
    else:
        rows = get_weather_data(window_from.isoformat(), window_to.isoformat(),
                                model=model, city=city, lead_days=None, limit=100000)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    if "target_date" not in df:
        return []
    dt = pd.to_datetime(df["target_date"], errors="coerce")
    ym = pd.DataFrame({"y": dt.dt.year, "m": dt.dt.month})
    pairs = (
        ym.dropna().astype({"y":"int32","m":"int16"})
          .drop_duplicates()
          .sort_values(["y","m"])
    )
    return list(pairs.itertuples(index=False, name=None))

# ────────────────────────────────────────────────────────────────────────────────
# Controls (top bar)
# ────────────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar: Color thresholds + Integer points for overall score
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Color thresholds")
    st.caption("Cell colors in the pivots (absolute error vs. observation).")

    thr_temp_g = st.slider("Temp (°C): green/orange", 0.0, 5.0, DEFAULT_THRESHOLDS["temp_min_c"][0])
    thr_temp_o = st.slider("Temp (°C): orange/red",   0.0, 6.0, DEFAULT_THRESHOLDS["temp_min_c"][1])
    thr_wind_g = st.slider("Wind (m/s): green/orange", 0.0, 3.0, DEFAULT_THRESHOLDS["wind_mps"][0])
    thr_wind_o = st.slider("Wind (m/s): orange/red",   0.0, 4.0, DEFAULT_THRESHOLDS["wind_mps"][1])
    thr_rain_g = st.slider("Precip (mm): green/orange", 0.0, 5.0, DEFAULT_THRESHOLDS["rain_mm"][0])
    thr_rain_o = st.slider("Precip (mm): orange/red",   0.0, 8.0, DEFAULT_THRESHOLDS["rain_mm"][1])

    thresholds: Dict[str, Tuple[float,float]] = {
        "temp_min_c": (thr_temp_g, thr_temp_o),
        "temp_max_c": (thr_temp_g, thr_temp_o),
        "wind_mps":   (thr_wind_g, thr_wind_o),
        "rain_mm":    (thr_rain_g, thr_rain_o),
    }

    st.markdown("---")
    st.header("Overall score – weights (points)")
    st.caption("Give **points** per component (integers). Points are normalized to sum to 1 per row.")

    POINTS_MAX = 10
    pts_max  = st.slider("Points: Temp MAX", 0, POINTS_MAX, 3, step=1)
    pts_min  = st.slider("Points: Temp MIN", 0, POINTS_MAX, 3, step=1)
    pts_pop  = st.slider("Points: PoP (rain prob.)", 0, POINTS_MAX, 3, step=1)
    pts_rain = st.slider("Points: Precip (mm)", 0, POINTS_MAX, 1, step=1)
    pts_wind = st.slider("Points: Wind (m/s)", 0, POINTS_MAX, 1, step=1)

    # store points for other parts (PDF uses them too)
    st.session_state["overall_weights_points"] = {
        "temp_max_c": pts_max,
        "temp_min_c": pts_min,
        "rain_probability_pct": pts_pop,
        "rain_mm": pts_rain,
        "wind_mps": pts_wind,
    }

    def _norm(d: dict) -> dict:
        s = sum(d.values())
        if s <= 0:
            return {k: 0.0 for k in d}
        return {k: v / s for k, v in d.items()}

    norm_shares = _norm(st.session_state["overall_weights_points"])
    st.caption(
        "Normalized shares (sum = 1):  "
        f"MAX {norm_shares['temp_max_c']:.2f} • "
        f"MIN {norm_shares['temp_min_c']:.2f} • "
        f"PoP {norm_shares['rain_probability_pct']:.2f} • "
        f"Precip {norm_shares['rain_mm']:.2f} • "
        f"Wind {norm_shares['wind_mps']:.2f}"
    )

    st.markdown("---")
    score_thr_g = st.slider("Score (%): green/orange", 0, 100, 60, step=1)
    score_thr_o = st.slider("Score (%): orange/red",   0, 100, 80, step=1)
    st.session_state["overall_score_thresholds"] = (float(score_thr_g), float(score_thr_o))

    st.markdown("---")
    thr_prob_g = st.slider("PoP error (%-points): green/orange", 0.0, 50.0, DEFAULT_THRESHOLDS_PROB[0])
    thr_prob_o = st.slider("PoP error (%-points): orange/red",   0.0, 60.0, DEFAULT_THRESHOLDS_PROB[1])
    thresholds_prob: Tuple[float,float] = (thr_prob_g, thr_prob_o)

# pull score thresholds for the rest of the app
score_thresholds: Tuple[float,float] = tuple(st.session_state.get("overall_score_thresholds", (60.0, 80.0)))

if st.button("🔄 Refresh"):
    load_accuracy.clear(); load_data_window.clear(); st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# Caption helpers
# ────────────────────────────────────────────────────────────────────────────────
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
        "Cell colors reflect absolute error in %-points vs. the observed 0/100."
    )

def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in w.values())
    if total <= 0:
        return {k: 0.0 for k in w}
    return {k: float(v)/total for k, v in w.items()}

# Convert integer points -> weight dict expected by scoring function
_points = st.session_state.get("overall_weights_points", {})
weights_cfg = {
    "w_tmax": float(_points.get("temp_max_c", 0)),
    "w_tmin": float(_points.get("temp_min_c", 0)),
    "w_pop":  float(_points.get("rain_probability_pct", 0)),
    "w_rain": float(_points.get("rain_mm", 0)),
    "w_wind": float(_points.get("wind_mps", 0)),
}
weights_norm = _normalize_weights(weights_cfg)


def _filter_allowed_leads(df: pd.DataFrame, model: str, col: str = "lead_days") -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    allowed = set(allowed_leads_for_model(model))
    return df[df[col].isin(allowed)].copy()

# ────────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ────────────────────────────────────────────────────────────────────────────────
def weighted_merge_accuracy(acc_list: List[Dict]) -> pd.DataFrame:
    """Merge Accuracy buckets across cities with n-weighted averages."""
    if not acc_list:
        return pd.DataFrame()
    frames = []
    for acc in acc_list:
        b = (acc or {}).get("buckets", [])
        if b: frames.append(pd.DataFrame(b))
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
            "rain_prob_brier":        wavg(d.get("rain_prob_brier"), d["n"]),
            "rain_prob_diracc_pct":   wavg(d.get("rain_prob_diracc_pct"), d["n"]),
            "rain_prob_mae_pctpts":   wavg(d.get("rain_prob_mae_pctpts"), d["n"]),
        })
    ).reset_index(drop=True).sort_values("lead_days")
    return g

def compute_bias_buckets_grouped(df: pd.DataFrame, max_lead: int) -> pd.DataFrame:
    """Bias aggregation over cities (n-weighted)."""
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

# ────────────────────────────────────────────────────────────────────────────────
# Overall score computation
# ────────────────────────────────────────────────────────────────────────────────
def _subscore(err: float, orange_thr: float) -> float:
    """Map an absolute error to 0..100 using the orange threshold as the scale."""
    if pd.isna(err) or orange_thr is None or orange_thr <= 0:
        return np.nan
    e = float(err)
    n = min(1.0, max(0.0, e / float(orange_thr)))
    return (1.0 - n) * 100.0

def compute_overall_scores(df_all: pd.DataFrame,
                           thresholds_in: Dict[str, Tuple[float, float]],
                           thresholds_prob_in: Tuple[float, float],
                           weights_in: Dict[str, float]) -> pd.DataFrame:
    """
    For each (city, target_date, lead_days) compute an overall score 0..100.
    Subscores:
      - Temp min/max, Wind, Rain(mm): |Forecast-Obs| → 0..100 (scaled with orange threshold).
      - PoP: |p% − (Obs 0/100)| → 0..100 (scaled with PoP orange threshold).
    Weights are normalized globally and re-normalized per row to available components.
    """
    if df_all.empty:
        return pd.DataFrame(columns=["city", "target_date", "lead_days", "score_overall"])

    have_prob = "rain_probability_pct" in df_all.columns
    out_rows: list[dict] = []
    w_norm = _normalize_weights(weights_in)

    keep_obs = ["temp_min_c", "temp_max_c", "wind_mps", "rain_mm"]

    for city_name, dcity in df_all.groupby("city"):
        obs = dcity[dcity["lead_days"] == 0].set_index("target_date")
        m = dcity.merge(
            obs[keep_obs],
            left_on="target_date", right_index=True, suffixes=("", "_obs")
        ).reset_index(drop=True)
        if m.empty:
            continue

        event_base = np.where(m["rain_mm_obs"].fillna(0) >= RAIN_EVENT_THRESHOLD_MM, 100.0, 0.0)

        err_tmin = (m["temp_min_c"] - m["temp_min_c_obs"]).abs()
        err_tmax = (m["temp_max_c"] - m["temp_max_c_obs"]).abs()
        err_wind = (m["wind_mps"]   - m["wind_mps_obs"]).abs()
        err_rain = (m["rain_mm"]    - m["rain_mm_obs"]).abs()
        err_pop  = (m["rain_probability_pct"] - event_base).abs() if have_prob else pd.Series(np.nan, index=m.index)

        s_tmin = err_tmin.apply(lambda e: _subscore(e, thresholds_in["temp_min_c"][1]))
        s_tmax = err_tmax.apply(lambda e: _subscore(e, thresholds_in["temp_max_c"][1]))
        s_wind = err_wind.apply(lambda e: _subscore(e, thresholds_in["wind_mps"][1]))
        s_rain = err_rain.apply(lambda e: _subscore(e, thresholds_in["rain_mm"][1]))
        s_pop  = err_pop.apply(lambda e: _subscore(e, thresholds_prob_in[1]))

        a_tmin = s_tmin.to_numpy()
        a_tmax = s_tmax.to_numpy()
        a_wind = s_wind.to_numpy()
        a_rain = s_rain.to_numpy()
        a_pop  = s_pop.to_numpy()

        for pos, row in enumerate(m.itertuples(index=False)):
            subs = {
                "w_tmax": a_tmax[pos],
                "w_tmin": a_tmin[pos],
                "w_pop":  a_pop[pos],
                "w_rain": a_rain[pos],
                "w_wind": a_wind[pos],
            }
            avail = {k: v for k, v in subs.items() if not pd.isna(v)}
            if not avail:
                score = np.nan
            else:
                w_av = {k: v for k, v in w_norm.items() if k in avail}
                s = sum(w_av.values())
                if s <= 0:
                    score = float(np.mean(list(avail.values())))
                else:
                    score = float(sum(w_av[k] * avail[k] for k in avail) / s)
            out_rows.append({
                "city": city_name,
                "target_date": getattr(row, "target_date"),
                "lead_days": int(getattr(row, "lead_days")),
                "score_overall": score,
            })

    return pd.DataFrame(out_rows)

# ────────────────────────────────────────────────────────────────────────────────
# Accuracy
# ────────────────────────────────────────────────────────────────────────────────
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
            fig_min.update_xaxes(tickmode='array', tickvals=allowed_ticks)
            st.plotly_chart(fig_min, width='stretch')
            caption_mae("Temp MIN", df_acc, model, city_label, frm, to)

        with c2:
            fig_max = px.line(df_acc, x="lead_days", y="temp_max_mae", markers=True,
                              title=f"Temp MAX – MAE (°C) • {model} @ {city_label}")
            fig_max.update_layout(xaxis_title="Lead (days)", yaxis_title="MAE (°C)")
            fig_max.update_xaxes(tickmode='array', tickvals=allowed_ticks)
            st.plotly_chart(fig_max, width='stretch')
            caption_mae("Temp MAX", df_acc, model, city_label, frm, to)

        c3, c4 = st.columns(2)
        with c3:
            fig_w = px.line(df_acc, x="lead_days", y="wind_mae", markers=True,
                            title=f"Wind – MAE (m/s) • {model} @ {city_label}")
            fig_w.update_layout(xaxis_title="Lead (days)", yaxis_title="MAE (m/s)")
            st.plotly_chart(fig_w, width='stretch')
            caption_mae("Wind", df_acc, model, city_label, frm, to)

        with c4:
            fig_r = px.line(df_acc, x="lead_days", y="rain_mae", markers=True,
                            title=f"Precipitation – MAE (mm) • {model} @ {city_label}")
            fig_r.update_layout(xaxis_title="Lead (days)", yaxis_title="MAE (mm)")
            st.plotly_chart(fig_r, width='stretch')
            caption_mae("Precipitation", df_acc, model, city_label, frm, to)

        fig_m = px.bar(df_acc, x="lead_days", y="weather_match_pct",
                       title=f"Weather text: exact matches (%) • {model} @ {city_label}")
        fig_m.update_layout(xaxis_title="Lead (days)", yaxis_title="Match rate (%)")
        st.plotly_chart(fig_m, width='stretch')
        caption_weather_string(df_acc, model, city_label, frm, to)

        if "rain_prob_brier" in df_acc.columns:
            fig_bs = px.line(df_acc, x="lead_days", y="rain_prob_brier", markers=True,
                             title=f"Brier score (PoP) • {model} @ {city_label}")
            fig_bs.update_layout(xaxis_title="Lead (days)", yaxis_title="Brier (lower is better)")
            st.plotly_chart(fig_bs, width='stretch')
            st.caption("Brier score = mean squared error between forecast probability (0..1) and observed event (0/1, rain ≥ 0.1 mm).")

        if "rain_prob_diracc_pct" in df_acc.columns:
            fig_da = px.bar(df_acc, x="lead_days", y="rain_prob_diracc_pct",
                            title=f"Directional accuracy @50% (PoP) • {model} @ {city_label}")
            fig_da.update_layout(xaxis_title="Lead (days)", yaxis_title="Hit rate (%)")
            st.plotly_chart(fig_da, width='stretch')
            st.caption("Share of cases where p≥50% correctly predicts an event, or p<50% correctly predicts no event.")

        if "rain_prob_mae_pctpts" in df_acc.columns:
            fig_pm = px.line(df_acc, x="lead_days", y="rain_prob_mae_pctpts", markers=True,
                             title=f"MAE in %-points (PoP vs. 0/100) • {model} @ {city_label}")
            fig_pm.update_layout(xaxis_title="Lead (days)", yaxis_title="MAE (percentage points)")
            st.plotly_chart(fig_pm, width='stretch')
            st.caption("Mean absolute error in percentage points between forecast p and observed 0/100 event.")
    else:
        st.info("No accuracy data in the selected window.")
except Exception as e:
    st.error(f"Error loading accuracy: {e}")

st.divider()

# ────────────────────────────────────────────────────────────────────────────────
# Bias
# ────────────────────────────────────────────────────────────────────────────────
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
            _dfb = _filter_allowed_leads(df_bias, model)
            fig_bmin = px.line(df_bias, x="lead_days", y="temp_min_bias", markers=True,
                               title=f"Bias – Temp MIN (°C) • {model} @ {city_label}")
            fig_bmin.update_layout(xaxis_title="Lead (days)", yaxis_title="Bias (°C)")
            fig_bmin.update_xaxes(tickmode='array', tickvals=allowed_ticks)
            st.plotly_chart(fig_bmin, width='stretch')
            caption_bias("Temp MIN", df_bias, model, city_label, frm, to)

        with c2:
            fig_bmax = px.line(df_bias, x="lead_days", y="temp_max_bias", markers=True,
                               title=f"Bias – Temp MAX (°C) • {model} @ {city_label}")
            fig_bmax.update_layout(xaxis_title="Lead (days)", yaxis_title="Bias (°C)")
            fig_bmax.update_xaxes(tickmode='array', tickvals=allowed_ticks)
            st.plotly_chart(fig_bmax, width='stretch')
            caption_bias("Temp MAX", df_bias, model, city_label, frm, to)

        c3, c4 = st.columns(2)
        with c3:
            fig_bwind = px.line(df_bias, x="lead_days", y="wind_bias", markers=True,
                                title=f"Bias – Wind (m/s) • {model} @ {city_label}")
            fig_bwind.update_layout(xaxis_title="Lead (days)", yaxis_title="Bias (m/s)")
            st.plotly_chart(fig_bwind, width='stretch')
            caption_bias("Wind", df_bias, model, city_label, frm, to)

        with c4:
            fig_brain = px.line(df_bias, x="lead_days", y="rain_bias", markers=True,
                                title=f"Bias – Precipitation (mm) • {model} @ {city_label}")
            fig_brain.update_layout(xaxis_title="Lead (days)", yaxis_title="Bias (mm)")
            st.plotly_chart(fig_brain, width='stretch')
            caption_bias("Precipitation", df_bias, model, city_label, frm, to)
except Exception as e:
    st.error(f"Error computing bias: {e}")

st.divider()

# ────────────────────────────────────────────────────────────────────────────────
# Pivot helpers (incl. overall score)
# ────────────────────────────────────────────────────────────────────────────────

def aggregate_all_cities_pivot(df_all: pd.DataFrame, model: str, var: str, thresholds_v: Tuple[float,float]) -> 'pd.io.formats.style.Styler':
    if df_all.empty:
        return pd.DataFrame()
    d = df_all[['target_date','lead_days',var,'city']].copy()
    d['target_date'] = pd.to_datetime(d['target_date'], errors='coerce').dt.date
    d = d.dropna(subset=['target_date'])
    d = d.dropna(subset=[var])
    d = _filter_allowed_leads(d, model, 'lead_days')
    grp = d.groupby(['target_date','lead_days'], as_index=False)[var].mean()
    grp['neg_lead'] = -grp['lead_days'].astype(int)
    pv = grp.pivot(index='target_date', columns='neg_lead', values=var).sort_index()
    allowed_cols = set(allowed_neg_leads_for_model(model))
    pv = pv[[c for c in sorted(pv.columns) if c in allowed_cols]]
    base = pv[0] if 0 in pv.columns else pd.Series(index=pv.index, dtype=float)
    pv_err = pv.copy()
    for c in pv.columns:
        pv_err[c] = np.nan if c == 0 else (pv[c] - base).abs()
    thr_g, thr_o = thresholds_v
    def colorize(err):
        if pd.isna(err): return ''
        if err <= thr_g: return 'background-color: #e6f4ea'
        if err <= thr_o: return 'background-color: #fff4e5'
        return 'background-color: #fde8e8'
    styled = pv.copy()
    for c in pv.columns:
        styled[c] = '' if c == 0 else pv_err[c].apply(colorize)
    show = pv.copy()
    show.insert(0, 'date', show.index.astype(str))
    return show.style.format(precision=1).apply(lambda _: styled, axis=None)

st.subheader("Pivot tables per variable (rows = dates, columns = leads −7…0)")

def _lead_to_negcol(lead: int) -> int:
    return -lead

def _build_colored_table_from_values(pv: pd.DataFrame, style_func) -> "pd.io.formats.style.Styler":
    pv_show = pv.copy()
    if isinstance(pv_show.index, pd.MultiIndex):
        pv_show.insert(0, "city", [idx[0] for idx in pv_show.index])
        pv_show.insert(1, "date", [str(idx[1]) for idx in pv_show.index])
    else:
        pv_show.insert(0, "date", pv_show.index.astype(str))
    return pv_show.style.format(precision=2).apply(lambda _: pv.applymap(style_func), axis=None)

def _build_pivot(df_all: pd.DataFrame, var: str, thresholds_v: Tuple[float,float], model: str):
    if df_all.empty:
        return pd.DataFrame()
    cols = ["target_date", "lead_days", var, "city"]
    df = df_all[cols].copy()
    df["neg_lead"] = df["lead_days"].apply(_lead_to_negcol)
    index_cols = ["target_date"] if df["city"].nunique() == 1 else ["city", "target_date"]
    pv = df.pivot_table(index=index_cols, columns="neg_lead", values=var, aggfunc="first").sort_index()
    allowed_cols = set(allowed_neg_leads_for_model(model))
    pv = pv[[c for c in sorted(pv.columns) if c in allowed_cols]]

    base = pv[0] if 0 in pv.columns else pd.Series(index=pv.index, dtype=float)
    pv_err = pv.copy()
    for c in pv.columns:
        pv_err[c] = np.nan if c == 0 else (pv[c] - base).abs()

    thr_g, thr_o = thresholds_v
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
    return pv_show.style.format(precision=2).apply(lambda _: styled, axis=None)

def _build_pivot_prob(df_all: pd.DataFrame, thresholds_pp: Tuple[float,float], model: str):
    if df_all.empty:
        return pd.DataFrame()
    cols = ["target_date", "lead_days", "rain_probability_pct", "rain_mm", "city"]
    have = [c for c in cols if c in df_all.columns]
    if not {"target_date","lead_days","city"}.issubset(have):
        return pd.DataFrame()

    d = df_all[have].copy()
    d["neg_lead"] = -d["lead_days"].astype(int)

    obs = d[d["lead_days"] == 0].copy()
    obs["event_0_100"] = np.where(obs["rain_mm"].fillna(0) >= RAIN_EVENT_THRESHOLD_MM, 100.0, 0.0)
    obs_base = obs[["target_date", "city", "event_0_100"]].drop_duplicates()

    index_cols = ["target_date"] if d["city"].nunique() == 1 else ["city", "target_date"]
    pv = d.pivot_table(index=index_cols, columns="neg_lead", values="rain_probability_pct", aggfunc="first").sort_index()
    for col in range(-7, 1):
        if col not in pv.columns: pv[col] = np.nan
    pv = pv[sorted(pv.columns)]

    if isinstance(pv.index, pd.MultiIndex):
        base_join = obs_base.set_index(["city", "target_date"])
    else:
        base_join = obs_base.set_index("target_date")
    base_series = base_join["event_0_100"].reindex(pv.index)
    pv[0] = base_series

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
    return pv_show.style.format(precision=1).apply(lambda _: styled, axis=None)

def _build_pivot_prob_ALL_aggregate(
    df_all: pd.DataFrame,
    thresholds_pp: tuple[float, float],
    event_threshold_mm: float = RAIN_EVENT_THRESHOLD_MM,
    model: str = 'default',
):
    """PoP pivot aggregated across cities for ALL."""
    need = {"target_date", "lead_days", "city", "rain_probability_pct", "rain_mm"}
    if df_all.empty or not need.issubset(df_all.columns):
        return pd.DataFrame()

    d = df_all[list(need)].copy()
    d["target_date"] = pd.to_datetime(d["target_date"], errors="coerce").dt.date

    grp = (
        d.groupby(["target_date", "lead_days"], as_index=False)["rain_probability_pct"]
         .mean()
         .rename(columns={"rain_probability_pct": "pop_mean"})
    )
    grp["neg_lead"] = -grp["lead_days"].astype(int)
    pv = grp.pivot(index="target_date", columns="neg_lead", values="pop_mean").sort_index()
    allowed_cols = set(allowed_neg_leads_for_model(model))
    pv = pv[[c for c in sorted(pv.columns) if c in allowed_cols]]

    obs = d[d["lead_days"] == 0].copy()
    obs["event_0_100"] = (obs["rain_mm"].fillna(0) >= event_threshold_mm).astype(float) * 100.0
    base = obs.groupby("target_date", as_index=True)["event_0_100"].mean()
    pv[0] = base.reindex(pv.index)

    base_series = pv[0]
    pv_err = pv.copy()
    for c in pv.columns:
        pv_err[c] = np.nan if c == 0 else (pv[c] - base_series).abs()

    thr_g, thr_o = thresholds_pp
    def colorize(err):
        if pd.isna(err): return ""
        if err <= thr_g: return "background-color:#e6f4ea"
        if err <= thr_o: return "background-color:#fff4e5"
        return "background-color:#fde8e8"

    styled = pv.copy()
    for c in pv.columns:
        styled[c] = "" if c == 0 else pv_err[c].apply(colorize)

    show = pv.copy()
    show.insert(0, "date", show.index.astype(str))
    styler = show.style.format(precision=1).apply(lambda _: styled, axis=None)
    return styler

def _build_score_pivot(df_scores: pd.DataFrame, score_thr: Tuple[float, float]):
    if df_scores.empty:
        return pd.DataFrame()
    d = df_scores[["city","target_date","lead_days","score_overall"]].copy()
    d["neg_lead"] = -d["lead_days"].astype(int)
    index_cols = ["target_date"] if d["city"].nunique() == 1 else ["city","target_date"]
    pv = d.pivot_table(index=index_cols, columns="neg_lead", values="score_overall", aggfunc="mean").sort_index()
    for col in range(-7, 1):
        if col not in pv.columns: pv[col] = np.nan
    pv = pv[sorted(pv.columns)]
    g_thr, o_thr = score_thr
    def style_cell(v):
        if pd.isna(v): return ""
        if v >= g_thr: return "background-color: #e6f4ea"
        if v >= o_thr: return "background-color: #fff4e5"
        return "background-color: #fde8e8"
    return _build_colored_table_from_values(pv, style_cell)

def _build_score_pivot_ALL_aggregate(df_scores: pd.DataFrame, score_thr: Tuple[float, float]):
    if df_scores.empty:
        return pd.DataFrame()
    d = df_scores[["target_date","lead_days","score_overall","city"]].copy()
    g = d.groupby(["target_date","lead_days"], as_index=False)["score_overall"].mean()
    g["neg_lead"] = -g["lead_days"].astype(int)
    pv = g.pivot_table(index="target_date", columns="neg_lead", values="score_overall", aggfunc="first").sort_index()
    for col in range(-7, 1):
        if col not in pv.columns: pv[col] = np.nan
    pv = pv[sorted(pv.columns)]
    g_thr, o_thr = score_thr
    def style_cell(v):
        if pd.isna(v): return ""
        if v >= g_thr: return "background-color: #e6f4ea"
        if v >= o_thr: return "background-color: #fff4e5"
        return "background-color: #fde8e8"
    pv.insert(0, "date", pv.index.astype(str))
    return pv.style.format(precision=1).apply(lambda _: pv.applymap(style_cell), axis=None)

# ── Render pivot tables in the UI ───────────────────────────────────────────────
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
            if city == CITY_ALL_LABEL:
                styler = aggregate_all_cities_pivot(dfr, model, var, thresholds[var])
            else:
                styler = _build_pivot(dfr, var, thresholds[var], model)
            st.dataframe(styler, width='stretch')

        st.markdown(f"**Outlook (weather text) • {model} @ {city_label}**")
        dfw = dfr[["target_date", "lead_days", "weather", "city"]].copy()
        dfw["neg_lead"] = dfw["lead_days"].apply(_lead_to_negcol)
        index_cols = ["target_date"] if dfw["city"].nunique() == 1 else ["city", "target_date"]
        pvw = dfw.pivot_table(index=index_cols, columns="neg_lead", values="weather", aggfunc="first").sort_index()
        allowed_cols = set(allowed_neg_leads_for_model(model))
        pvw = pvw[[c for c in sorted(pvw.columns) if c in allowed_cols]]
        if isinstance(pvw.index, pd.MultiIndex):
            pvw.insert(0, "city", [idx[0] for idx in pvw.index])
            pvw.insert(1, "date", [str(idx[1]) for idx in pvw.index])
        else:
            pvw.insert(0, "date", pvw.index.astype(str))
        st.dataframe(pvw, width='stretch')
        st.caption("Cells show the forecast weather string at each lead; column 0 is the observation text of the day.")

        st.markdown(f"**Probability of precipitation (PoP, %) • {model} @ {city_label}**")
        styler_prob = _build_pivot_prob(dfr, thresholds_prob, model)
        st.dataframe(styler_prob, width='stretch')
        caption_pop_pivot(city_label)

        # NEW: overall model score (weighted)
        st.markdown(f"**Overall model score (0–100, weighted) • {model} @ {city_label}**")
        df_scores = compute_overall_scores(dfr, thresholds, thresholds_prob, weights_norm)
        styler_score = _build_score_pivot(df_scores, score_thresholds)
        st.dataframe(styler_score, width='stretch')
        st.caption(
            f"Score = weighted combination of subscores (Temp MAX/MIN, PoP, Precip mm, Wind). "
            f"Each subscore maps absolute error to 0..100 using the orange thresholds as scale. "
            f"Weights (normalized): Tmax {weights_norm['w_tmax']:.2f}, Tmin {weights_norm['w_tmin']:.2f}, "
            f"PoP {weights_norm['w_pop']:.2f}, Rain {weights_norm['w_rain']:.2f}, Wind {weights_norm['w_wind']:.2f}. "
            f"Green ≥ {score_thresholds[0]:.0f}, Orange ≥ {score_thresholds[1]:.0f}, else red."
        )
except Exception as e:
    st.error(f"Error creating tables: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# Raw data (inspection)
# ────────────────────────────────────────────────────────────────────────────────
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
            st.dataframe(dfr, width='stretch', hide_index=True)
            st.caption(f"{len(dfr)} rows")
        else:
            st.info("No data in the current window. Check model/city or the time window.")
    except Exception as e:
        st.error(f"Error loading raw data: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# Logs
# ────────────────────────────────────────────────────────────────────────────────
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
            st.dataframe(dfl, width='stretch', hide_index=True)
            st.caption(f"{len(dfl)} log entries loaded.")
        else:
            st.info("No logs available.")
    except Exception as e:
        st.error(f"Error loading logs: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# Methodology & Data Notes
# ────────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Methodology & Data Notes")

with st.container():
    try:
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
        f"**Rain event definition:** rain ≥ {RAIN_EVENT_THRESHOLD_MM} mm ⇒ event = 1 (100%), else 0.",
        "**PoP evaluation:** Brier score (lower is better), directional accuracy @50%, and MAE in percentage points vs. observed 0/100.",
        "**Pivot coloring:** By absolute error vs. observation (for PoP: vs. 0/100 event). Thresholds can be adjusted in the sidebar and are applied in the PDF.",
        f"**Overall score:** subscores map absolute errors to 0..100 using orange thresholds; weighted with normalized points "
        f"(Tmax {weights_norm['w_tmax']:.2f}, Tmin {weights_norm['w_tmin']:.2f}, PoP {weights_norm['w_pop']:.2f}, Rain {weights_norm['w_rain']:.2f}, Wind {weights_norm['w_wind']:.2f}). "
        f"Score colors: green ≥ {score_thresholds[0]:.0f}, orange ≥ {score_thresholds[1]:.0f}, else red.",
        "**Data sources:** Open-Meteo API (daily aggregates), MET Norway / Yr API (hourly → daily aggregates), and wetter.com (HTML, 7-day page). "
        "Scraping of wetter.com is for testing/private purposes only; please respect their ToS and robots.txt.",
        "**ETL cadence:** Forecasts (lead 0..7) are upserted daily per model and city; later runs overwrite earlier ones for the same target date/lead.",
    ]
    for b in bullets:
        st.markdown(f"- {b}")

# ────────────────────────────────────────────────────────────────────────────────
# Monthly PDF report (pivots + overall score)
# ────────────────────────────────────────────────────────────────────────────────
try:
    import pdfkit  # needs wkhtmltopdf in the container
    _WKHTMLTOPDF = shutil.which("wkhtmltopdf") is not None
except Exception:
    pdfkit = None
    _WKHTMLTOPDF = False

st.divider()
st.subheader("Monthly PDF report (pivot tables)")

col_m1, col_m2 = st.columns([1,1])
with col_m1:
    rep_model = st.selectbox("Model", options=KNOWN_MODELS,
                             index=KNOWN_MODELS.index(model) if model in KNOWN_MODELS else 0,
                             key="pdf_model")
with col_m2:
    rep_city = st.selectbox("City", options=[CITY_ALL_LABEL] + KNOWN_CITIES,
                            index=( [CITY_ALL_LABEL] + KNOWN_CITIES ).index(city)
                                   if city in KNOWN_CITIES or city==CITY_ALL_LABEL else 1,
                            key="pdf_city")

agg_mode = "Aggregate (mean across cities)" if rep_city == CITY_ALL_LABEL else "Single city"
if rep_city == CITY_ALL_LABEL:
    agg_mode = st.radio(
        "Report mode for ALL cities",
        ["Aggregate (mean across cities)", "Per-city breakdown"],
        horizontal=True,
        index=0,
        key="pdf_mode"
    )

pairs = load_available_year_months(rep_model, rep_city)
if not pairs:
    st.info("No target dates available for this model/city. Adjust your selection.")
    st.stop()

years = sorted({y for y, _ in pairs})
default_year = years[-1]

col_y, col_m = st.columns([1,1])
with col_y:
    rep_year = st.selectbox("Year", options=years, index=years.index(default_year), key="pdf_year")
months_for_year = sorted([m for (y, m) in pairs if y == rep_year])
month_label_map = {m: calendar.month_name[m] for m in months_for_year}
with col_m:
    rep_month = st.selectbox(
        "Month",
        options=months_for_year,
        format_func=lambda m: month_label_map.get(m, str(m)),
        index=len(months_for_year)-1,
        key="pdf_month"
    )

st.caption(
    f"Available months are derived from existing target dates in the database for **{rep_model}** / **{rep_city}**."
)

def _month_window(y: int, m: int):
    first = date(y, m, 1)
    last = date(y, m, calendar.monthrange(y, m)[1])
    return first, last

def _df_for_month(model_: str, city_: str, y: int, m: int) -> pd.DataFrame:
    frm_m, to_m = _month_window(y, m)
    rows = []
    if city_ == CITY_ALL_LABEL:
        for c in KNOWN_CITIES:
            r = get_weather_data(frm_m.isoformat(), to_m.isoformat(), model=model_, city=c, lead_days=None, limit=10000)
            if r: rows.extend(r)
    else:
        rows = get_weather_data(frm_m.isoformat(), to_m.isoformat(), model=model_, city=city_, lead_days=None, limit=10000)
    df = pd.DataFrame(rows or [])
    if df.empty:
        return df
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.date
    return df

def _styler_html(styler: "pd.io.formats.style.Styler") -> str:
    try:
        return styler.to_html()
    except Exception:
        df = getattr(styler, "data", None)
        return (df if isinstance(df, pd.DataFrame) else pd.DataFrame()).to_html(index=True)

def _build_pivot_ALL_aggregate(df_all: pd.DataFrame, var: str, thresholds_v: Tuple[float,float]):
    if df_all.empty:
        return pd.DataFrame()
    d = df_all[["target_date","lead_days","city",var]].copy()
    g = d.groupby(["target_date","lead_days"], as_index=False)[var].mean()
    g["neg_lead"] = -g["lead_days"].astype(int)
    pv = g.pivot_table(index="target_date", columns="neg_lead", values=var, aggfunc="first").sort_index()
    for col in range(-7, 1):
        if col not in pv.columns: pv[col] = np.nan
    pv = pv[sorted(pv.columns)]
    base = pv[0] if 0 in pv.columns else pd.Series(index=pv.index, dtype=float)
    thr_g, thr_o = thresholds_v
    def colorize(err):
        if pd.isna(err): return ""
        if err <= thr_g: return "background-color: #e6f4ea"
        if err <= thr_o: return "background-color: #fff4e5"
        return "background-color: #fde8e8"
    pv_err = pv.copy()
    for c in pv.columns:
        pv_err[c] = np.nan if c == 0 else (pv[c] - base).abs()
    styled = pv.copy()
    for c in pv.columns:
        styled[c] = "" if c == 0 else pv_err[c].apply(colorize)
    pv.insert(0, "date", pv.index.astype(str))
    return pv.style.format(precision=2).apply(lambda _: styled, axis=None)

def _pivot_html(df_all: pd.DataFrame, var: str, title: str, thresholds_local: Tuple[float,float], aggregate_all: bool) -> str:
    if aggregate_all:
        styler = _build_pivot_ALL_aggregate(df_all, var, thresholds_local)
    else:
        styler = _build_pivot(df_all, var, thresholds_local)
    html = _styler_html(styler)
    return f"<h3>{title}</h3>\n{html}"

def _pivot_weather_text_html(df_all: pd.DataFrame) -> str:
    dfw = df_all[["target_date", "lead_days", "weather", "city"]].copy()
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
    return "<h3>Outlook (weather text)</h3>\n" + pvw.to_html(index=False)

def _pivot_rain_prob_html(df_all: pd.DataFrame, aggregate_all: bool) -> str:
    if "rain_probability_pct" not in df_all.columns:
        return ""
    if aggregate_all:
        sty = _build_pivot_prob_ALL_aggregate(df_all, thresholds_prob)
    else:
        sty = _build_pivot_prob(df_all, thresholds_prob)
    return "<h3>Rain probability (%)</h3>\n" + _styler_html(sty)

def _pivot_overall_score_html(df_all: pd.DataFrame, aggregate_all: bool) -> str:
    df_scores = compute_overall_scores(df_all, thresholds, thresholds_prob, weights_norm)
    if aggregate_all:
        sty = _build_score_pivot_ALL_aggregate(df_scores, score_thresholds)
    else:
        sty = _build_score_pivot(df_scores, score_thresholds)
    weights_txt = f"(weights: Tmax {weights_norm['w_tmax']:.2f}, Tmin {weights_norm['w_tmin']:.2f}, PoP {weights_norm['w_pop']:.2f}, Rain {weights_norm['w_rain']:.2f}, Wind {weights_norm['w_wind']:.2f})"
    return f"<h3>Overall model score {weights_txt}</h3>\n" + _styler_html(sty)

def _compose_report_html(df_month: pd.DataFrame, y: int, m: int, model_: str, city_: str, aggregate_all: bool) -> str:
    css = dedent("""
    <style>
      body { font-family: Arial, sans-serif; margin: 18px; }
      h1 { font-size: 20px; margin: 0 0 6px 0; }
      h2 { font-size: 16px; margin: 18px 0 6px 0; }
      h3 { font-size: 14px; margin: 14px 0 6px 0; }
      table { border-collapse: collapse; width: 100%; font-size: 11px; }
      th, td { border: 1px solid #ddd; padding: 6px; }
      th { background: #f6f8fa; }
    </style>
    """)
    title = f"Weatherwatch – Monthly pivot report ({calendar.month_name[m]} {y})"
    scope = f"Model: {model_} | City: {city_}" + (" (AGGREGATE)" if aggregate_all and city_ == CITY_ALL_LABEL else "")

    blocks = []
    # numeric pivots
    blocks.append(_pivot_html(df_month, "temp_max_c", "Max temperature (°C)", thresholds["temp_max_c"], aggregate_all))
    blocks.append(_pivot_html(df_month, "temp_min_c", "Min temperature (°C)", thresholds["temp_min_c"], aggregate_all))
    blocks.append(_pivot_html(df_month, "wind_mps",   "Wind (m/s)",            thresholds["wind_mps"],   aggregate_all))
    blocks.append(_pivot_html(df_month, "rain_mm",    "Precipitation (mm)",    thresholds["rain_mm"],    aggregate_all))

    # PoP + overall
    blocks.append(_pivot_rain_prob_html(df_month, aggregate_all))
    blocks.append(_pivot_overall_score_html(df_month, aggregate_all))

    n_rows = len(df_month) if isinstance(df_month, pd.DataFrame) else 0
    notes = dedent(f"""
    <h2>Methodology & Notes</h2>
    <ul>
      <li>Target dates limited to {calendar.month_name[m]} {y}. Columns show leads −7…0; column 0 are observations (for probability: observed 0/100).</li>
      <li>Color thresholds applied in this PDF:
        <ul>
          <li>Temp: {thresholds['temp_min_c'][0]}/{thresholds['temp_min_c'][1]} °C (green/orange • orange/red)</li>
          <li>Wind: {thresholds['wind_mps'][0]}/{thresholds['wind_mps'][1]} m/s</li>
          <li>Precip: {thresholds['rain_mm'][0]}/{thresholds['rain_mm'][1]} mm</li>
          <li>PoP error: {thresholds_prob[0]}/{thresholds_prob[1]} %-points</li>
          <li>Overall score colors: green ≥ {score_thresholds[0]:.0f}, orange ≥ {score_thresholds[1]:.0f}</li>
        </ul>
      </li>
      <li>Overall score weights (normalized): Tmax {weights_norm['w_tmax']:.2f}, Tmin {weights_norm['w_tmin']:.2f}, PoP {weights_norm['w_pop']:.2f}, Rain {weights_norm['w_rain']:.2f}, Wind {weights_norm['w_wind']:.2f}. Missing components are skipped and remaining weights re-normalized per row.</li>
      <li>PoP base column uses rain ≥ {RAIN_EVENT_THRESHOLD_MM} mm as event ⇒ 100 else 0; for ALL-aggregate the base is the share of cities with an event (0..100).</li>
      <li>Data source: your FastAPI endpoint <code>/api/weather/data</code>. Models as labeled above.</li>
      <li>Total rows in this month window: {n_rows}.</li>
    </ul>
    """)
    html = f"<!doctype html><html><head><meta charset='utf-8'>{css}</head><body>"
    html += f"<h1>{title}</h1><h2>{scope}</h2>"
    html += "\n".join(blocks)
    html += notes
    html += "</body></html>"
    return html

def _make_pdf(html: str) -> bytes:
    if not (pdfkit and _WKHTMLTOPDF):
        raise RuntimeError("wkhtmltopdf is not available in the container.")
    options = {
        "page-size": "A4",
        "margin-top": "10mm",
        "margin-bottom": "10mm",
        "margin-left": "10mm",
        "margin-right": "10mm",
        "encoding": "UTF-8",
        "quiet": "",
    }
    return pdfkit.from_string(html, False, options=options)

gen_col1, gen_col2 = st.columns([1,2])
with gen_col1:
    btn = st.button("📄 Generate monthly PDF (pivots)", type="primary", key="btn_gen_pdf")

if btn:
    try:
        df_month = _df_for_month(rep_model, rep_city, rep_year, rep_month)
        if df_month.empty:
            st.warning("No data for the selected month/model/city.")
        else:
            aggregate_flag = (rep_city == CITY_ALL_LABEL and agg_mode.startswith("Aggregate"))
            html = _compose_report_html(df_month, rep_year, rep_month, rep_model, rep_city, aggregate_flag)
            if pdfkit and _WKHTMLTOPDF:
                pdf_bytes = _make_pdf(html)
                fname_pdf = f"Weatherwatch_{rep_model}_{rep_city}_{rep_year}-{rep_month:02d}.pdf".replace(" ", "_")
                st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name=fname_pdf,
                                   mime="application/pdf", key="dl_pdf_main")
                b64 = base64.b64encode(pdf_bytes).decode("ascii")
                pdf_url = f"data:application/pdf;base64,{b64}"
                st.markdown(f"[🔎 Open PDF in a new tab]({pdf_url})", unsafe_allow_html=True)
                with st.expander("Preview PDF (inline)"):
                    st.markdown(f"<iframe src='{pdf_url}' width='100%' height='1000px' frameborder='0'></iframe>", unsafe_allow_html=True)
                st.success("PDF generated.")
            else:
                st.info("wkhtmltopdf is not installed in the container. Offering HTML instead (you can print to PDF from your browser).")
                fname_html = f"Weatherwatch_{rep_model}_{rep_city}_{rep_year}-{rep_month:02d}.html".replace(" ", "_")
                st.download_button("⬇️ Download HTML", data=html.encode("utf-8"), file_name=fname_html,
                                   mime="text/html", key="dl_html_fallback")
                with st.expander("Preview (HTML)"):
                    st.components.v1.html(html, height=900, scrolling=True)
    except Exception as e:
        st.error(f"PDF creation failed: {e}")
