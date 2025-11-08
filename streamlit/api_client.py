# streamlit/api_client.py
import os
import requests
from typing import Optional, Dict, Any, List

API_BASE = os.getenv("API_BASE", "http://backend:8000/api").rstrip("/")


class ApiError(RuntimeError):
    pass


def _json_or_raise(resp: requests.Response):
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        text = ""
        try:
            text = resp.text[:500]
        except Exception:
            pass
        raise ApiError(f"HTTP {resp.status_code}: {text or e}") from e
    try:
        return resp.json()
    except Exception as e:
        raise ApiError("Antwort war kein JSON") from e


def get_status():
    return _json_or_raise(requests.get(f"{API_BASE}/status", timeout=10))


def upsert_status(raspberry: str, status: str, message: Optional[str] = None, api_key: Optional[str] = None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload = {"raspberry": raspberry, "status": status, "message": message}
    return _json_or_raise(requests.post(f"{API_BASE}/status", json=payload, headers=headers, timeout=10))


def patch_status(raspberry: str, *, status: Optional[str] = None, message: Optional[str] = None, api_key: Optional[str] = None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload: Dict[str, Any] = {}
    if status is not None:
        payload["status"] = status
    if message is not None:
        payload["message"] = message
    return _json_or_raise(requests.patch(f"{API_BASE}/status/{raspberry}", json=payload, headers=headers, timeout=10))


# ==================== BILDWATCH ====================

# Articles
def get_bild_articles(limit: int = 500, offset: int = 0):
    r = requests.get(f"{API_BASE}/bild/articles", params={"limit": limit, "offset": offset}, timeout=10)
    return _json_or_raise(r)


def delete_bild_articles():
    api_key = os.getenv("INGEST_API_KEY", "dev-secret")
    resp = requests.delete(
        f"{API_BASE}/bild/articles",
        headers={"X-API-Key": api_key},
        timeout=10,
    )
    if resp.status_code not in (204, 200):
        try:
            msg = resp.json()
        except Exception:
            msg = resp.text
        raise ApiError(f"Fehler beim Löschen: {msg}")


# Charts
def get_bild_category_counts(premium_only: bool = False):
    """Backend berechnet die Kategorie-Verteilung."""
    r = requests.get(
        f"{API_BASE}/bild/charts/category_counts",
        params={"premium_only": str(premium_only).lower()},
        timeout=10,
    )
    if r.status_code != 200:
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise ApiError(f"Fehler beim Abrufen der Kategorien: {msg}")
    return r.json()


def get_bild_hourly(days: int = 60):
    """Liefert {'snapshot_avg': [...], 'new_avg': [...]} mit Stunden 0..23."""
    r = requests.get(f"{API_BASE}/bild/charts/hourly", params={"days": days}, timeout=10)
    return _json_or_raise(r)


def get_bild_daily_conversions(days: int = 60):
    """Liefert [{'day':'YYYY-MM-DD','count':N}, ...]"""
    r = requests.get(f"{API_BASE}/bild/charts/daily_conversions", params={"days": days}, timeout=10)
    return _json_or_raise(r)


# Raw Metrics (falls woanders benötigt)
def get_bild_metrics(time_from: Optional[str] = None, time_to: Optional[str] = None, limit: int = 5000):
    params: Dict[str, Any] = {}
    if time_from:
        params["time_from"] = time_from
    if time_to:
        params["time_to"] = time_to
    params["limit"] = limit
    r = requests.get(f"{API_BASE}/bild/metrics", params=params, timeout=10)
    return _json_or_raise(r)


# Logs
def get_bild_logs(limit: int = 1000, offset: int = 0, asc: bool = False):
    params = {"limit": limit, "offset": offset, "asc": str(asc).lower()}
    r = requests.get(f"{API_BASE}/bild/logs", params=params, timeout=10)
    return _json_or_raise(r)


def delete_bild_logs():
    api_key = os.getenv("INGEST_API_KEY", "dev-secret")
    r = requests.delete(f"{API_BASE}/bild/logs", headers={"X-API-Key": api_key}, timeout=10)
    if r.status_code not in (200, 204):
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise ApiError(f"Fehler beim Löschen der Logs: {msg}")


# Corrections (NEU)
def get_bild_corrections():
    """Liest alle gespeicherten Corrections (neueste zuerst per Backend-Order)."""
    r = requests.get(f"{API_BASE}/bild/corrections", timeout=10)
    return _json_or_raise(r)


def post_bild_correction(payload: Dict[str, Any], api_key: Optional[str] = None):
    """
    Optional: Manuelles Anlegen (zum Testen).
    payload keys: id, title, published (ISO-UTC), source_url, article_url
    """
    headers = {"Content-Type": "application/json"}
    headers["X-API-Key"] = api_key or os.getenv("INGEST_API_KEY", "dev-secret")
    r = requests.post(f"{API_BASE}/bild/corrections", json=payload, headers=headers, timeout=10)
    return _json_or_raise(r)


def delete_bild_corrections():
    """Löscht alle Corrections."""
    api_key = os.getenv("INGEST_API_KEY", "dev-secret")
    r = requests.delete(f"{API_BASE}/bild/corrections", headers={"X-API-Key": api_key}, timeout=10)
    if r.status_code not in (200, 204):
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise ApiError(f"Fehler beim Löschen der Corrections: {msg}")


# ==================== WEATHERWATCH ====================

def upsert_weather_data(payload: dict, api_key: str | None = None):
    # payload MUSS jetzt city enthalten
    headers = {"Content-Type": "application/json"}
    if api_key is None:
        api_key = os.getenv("INGEST_API_KEY", "dev-secret")
    headers["X-API-Key"] = api_key
    r = requests.post(f"{API_BASE}/weather/data", json=payload, headers=headers, timeout=10)
    return _json_or_raise(r)

def get_weather_data(date_from: str | None = None, date_to: str | None = None, model: str = "default", city: str = "berlin", lead_days: int | None = None, limit: int = 5000):
    params = {"model": model, "city": city, "limit": limit}
    if date_from: params["date_from"] = date_from
    if date_to:   params["date_to"]   = date_to
    if lead_days is not None: params["lead_days"] = lead_days
    r = requests.get(f"{API_BASE}/weather/data", params=params, timeout=10)
    return _json_or_raise(r)

def get_weather_accuracy(date_from: str, date_to: str, model: str = "default", city: str = "berlin", max_lead: int = 7):
    params = {"date_from": date_from, "date_to": date_to, "model": model, "city": city, "max_lead": max_lead}
    r = requests.get(f"{API_BASE}/weather/accuracy", params=params, timeout=10)
    return _json_or_raise(r)

def get_weather_logs(limit: int = 1000, offset: int = 0, asc: bool = False):
    params = {"limit": limit, "offset": offset, "asc": str(asc).lower()}
    r = requests.get(f"{API_BASE}/weather/logs", params=params, timeout=10)
    return _json_or_raise(r)

def delete_weather_logs():
    api_key = os.getenv("INGEST_API_KEY", "dev-secret")
    r = requests.delete(f"{API_BASE}/weather/logs", headers={"X-API-Key": api_key}, timeout=10)
    if r.status_code not in (200, 204):
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise ApiError(f"Fehler beim Löschen der Weather-Logs: {msg}")

# ==================== GOVWATCH ====================

def get_gov_incidents(seen: bool | None = None, limit: int = 500, offset: int = 0):
    params = {"limit": limit, "offset": offset}
    if seen is not None:
        params["seen"] = str(seen).lower()
    r = requests.get(f"{API_BASE}/gov/incidents", params=params, timeout=10)
    return _json_or_raise(r)

def get_gov_incident_detail(incident_id: int):
    r = requests.get(f"{API_BASE}/gov/incidents/{incident_id}", timeout=10)
    return _json_or_raise(r)

def post_gov_incident(headline: str, occurred_at: str | None, articles: list[dict], api_key: str | None = None):
    headers = {"Content-Type": "application/json"}
    headers["X-API-Key"] = api_key or os.getenv("INGEST_API_KEY", "dev-secret")
    payload = {"headline": headline, "occurred_at": occurred_at, "articles": articles}
    r = requests.post(f"{API_BASE}/gov/incidents", json=payload, headers=headers, timeout=15)
    return _json_or_raise(r)

def post_gov_article(incident_id: int, title: str, source: str, link: str, published_at: str | None = None, api_key: str | None = None):
    headers = {"Content-Type": "application/json"}
    headers["X-API-Key"] = api_key or os.getenv("INGEST_API_KEY", "dev-secret")
    payload = {"title": title, "source": source, "link": link, "published_at": published_at}
    r = requests.post(f"{API_BASE}/gov/incidents/{incident_id}/articles", json=payload, headers=headers, timeout=10)
    return _json_or_raise(r)

def create_gov_incident(headline: str, articles: list[dict], api_key: str | None = None):
    headers = {"Accept":"application/json","Content-Type":"application/json"}
    headers["X-API-Key"] = api_key or os.getenv("INGEST_API_KEY","dev-secret")
    payload = {"headline": headline, "articles": articles}
    r = requests.post(f"{API_BASE}/gov/incidents", json=payload, headers=headers, timeout=15)
    if r.status_code not in (200,201):
        try: msg = r.json()
        except Exception: msg = r.text
        raise ApiError(f"Fehler beim Anlegen: {r.status_code} {msg}")
    return r.json()


def patch_gov_incident_seen(incident_id: int, seen: bool, api_key: str | None = None):
    headers = {"Content-Type": "application/json"}
    headers["X-API-Key"] = api_key or os.getenv("INGEST_API_KEY", "dev-secret")
    r = requests.patch(f"{API_BASE}/gov/incidents/{incident_id}/seen", params={"seen": str(seen).lower()}, headers=headers, timeout=10)
    return r.json() if r.content else {"updated": 0}

def delete_gov_incident(incident_id: int, api_key: str | None = None):
    headers = {}
    headers["X-API-Key"] = api_key or os.getenv("INGEST_API_KEY", "dev-secret")
    r = requests.delete(f"{API_BASE}/gov/incidents/{incident_id}", headers=headers, timeout=10)
    if r.status_code not in (200, 204):
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise ApiError(f"Löschen fehlgeschlagen: {msg}")

def delete_gov_incident_article(incident_id: int, article_id: int, api_key: str | None = None):
    headers = {}
    headers["X-API-Key"] = api_key or os.getenv("INGEST_API_KEY", "dev-secret")
    r = requests.delete(f"{API_BASE}/gov/incidents/{incident_id}/articles/{article_id}", headers=headers, timeout=10)
    if r.status_code not in (200, 204):
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise ApiError(f"Artikel entfernen fehlgeschlagen: {msg}")

def wipe_gov(confirm: bool, api_key: str | None = None):
    headers = {}
    headers["X-API-Key"] = api_key or os.getenv("INGEST_API_KEY", "dev-secret")
    r = requests.delete(f"{API_BASE}/gov/wipe", params={"confirm": "yes" if confirm else "no"}, headers=headers, timeout=10)
    return _json_or_raise(r)
