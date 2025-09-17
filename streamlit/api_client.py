import os
import requests

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

def upsert_status(raspberry: str, status: str, message: str | None = None, api_key: str | None = None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload = {"raspberry": raspberry, "status": status, "message": message}
    return _json_or_raise(requests.post(f"{API_BASE}/status", json=payload, headers=headers, timeout=10))

def patch_status(raspberry: str, *, status: str | None = None, message: str | None = None, api_key: str | None = None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload = {}
    if status is not None:
        payload["status"] = status
    if message is not None:
        payload["message"] = message
    return _json_or_raise(requests.patch(f"{API_BASE}/status/{raspberry}", json=payload, headers=headers, timeout=10))

def get_dummy():
    return _json_or_raise(requests.get(f"{API_BASE}/dummy", timeout=10))

def create_dummy(message: str):
    return _json_or_raise(requests.post(f"{API_BASE}/dummy", json={"message": message}, timeout=10))

# BILDWATCH
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

def get_bild_category_counts():
    api_base = os.getenv("API_BASE", "http://backend:8000/api").rstrip("/")
    r = requests.get(f"{api_base}/bild/articles/category_counts", timeout=10)
    if r.status_code != 200:
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise ApiError(f"Fehler beim Abrufen der Kategorien: {msg}")
    return r.json()

def get_bild_metrics(time_from: str | None = None, time_to: str | None = None, limit: int = 5000):
    params = {}
    if time_from: params["time_from"] = time_from
    if time_to:   params["time_to"]   = time_to
    params["limit"] = limit
    r = requests.get(f"{API_BASE}/bild/metrics", params=params, timeout=10)
    return _json_or_raise(r)
