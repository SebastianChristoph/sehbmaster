from bs4 import BeautifulSoup
import requests
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")

# ========= Konfiguration =========
URL = "https://www.bild.de/"
HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 12_3_1 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 "
        "Instagram 105.0.0.11.118 (iPhone11,8; iOS 12_3_1; en_US; en-US; scale=2.00; 828x1792; 165586599)"
    )
}

API_BASE   = os.getenv("SEHBMASTER_API",     "http://localhost:8000/api").rstrip("/")
API_KEY    = os.getenv("SEHBMASTER_API_KEY", "dev-secret")
SCRAPER_ID = os.getenv("SCRAPER_ID",         "server-bildwatch")
TIMEOUT    = 15
FETCH_RETRY = 3

# ========= HTTP-Helper =========
class ApiError(RuntimeError):
    pass

class DuplicateError(ApiError):
    pass

def _req(method: str, path: str, json_body: Optional[dict] = None) -> Any:
    url = f"{API_BASE}{path}"
    headers = {"Accept": "application/json"}
    if method in ("POST", "PATCH", "PUT", "DELETE"):
        headers["Content-Type"] = "application/json"
        headers["X-API-Key"] = API_KEY
    resp = None
    try:
        resp = requests.request(method, url, json=json_body, headers=headers, timeout=TIMEOUT)
        if resp.status_code == 409:
            raise DuplicateError(f"{method} {url} -> 409 Conflict")
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()
    except DuplicateError:
        raise
    except requests.HTTPError as e:
        text = resp.text[:500] if resp is not None else ""
        raise ApiError(f"{method} {url} -> HTTP {resp.status_code}: {text}") from e
    except Exception as e:
        raise ApiError(f"{method} {url} failed: {e}") from e

# ========= Zeit/Parsing-Helper =========
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def iso_now_utc() -> str:
    return now_utc().replace(microsecond=0).isoformat()

def parse_iso_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except Exception:
        return None

def hours_between(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
    if not start or not end:
        return None
    return (end - start).total_seconds() / 3600.0

def fmt_berlin(dt_utc: datetime) -> tuple[str, str]:
    local = dt_utc.astimezone(BERLIN)
    return local.strftime("%H:%M"), local.strftime("%d.%m.%Y")

# ========= Backend-Calls =========
def send_log(message: str, ts_iso: Optional[str] = None):
    payload: Dict[str, Any] = {"message": message}
    if ts_iso:
        payload["timestamp"] = ts_iso
    print(f"[LOG] {ts_iso or ''} {message}")
    try:
        return _req("POST", "/bild/logs", payload)
    except Exception as e:
        print(f"[log-fallback] {e}")

def status_upsert(status: str, message: Optional[str] = None):
    payload = {"raspberry": SCRAPER_ID, "status": status, "message": message}
    return _req("POST", "/status", payload)

def get_all_articles_from_db() -> List[Dict[str, Any]]:
    """Paginiert durch alle Artikel – kein hartes Limit."""
    all_rows: List[Dict[str, Any]] = []
    limit, offset = 1000, 0
    while True:
        batch = _req("GET", f"/bild/articles?limit={limit}&offset={offset}") or []
        all_rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return all_rows

def create_article(scraped: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "id":                       scraped["id"],
        "title":                    scraped["title"],
        "url":                      scraped["url"],
        "category":                 scraped.get("category"),
        "is_premium":               bool(scraped.get("isPremium", False)),
        "converted":                False,
        "published":                iso_now_utc(),
        "converted_time":           None,
        "converted_duration_hours": None,
    }
    return _req("POST", "/bild/articles", payload)

def patch_article(article_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    clean = {k: v for k, v in updates.items() if v is not None or k in ("converted", "is_premium")}
    return _req("PATCH", f"/bild/articles/{article_id}", clean)

def post_metrics(ts_hour_iso: str, snapshot_total: int, snapshot_premium: int,
                 new_count: int, new_premium_count: int):
    pct = round((snapshot_premium / snapshot_total * 100.0), 2) if snapshot_total else 0.0
    return _req("POST", "/bild/metrics", {
        "ts_hour":             ts_hour_iso,
        "snapshot_total":      snapshot_total,
        "snapshot_premium":    snapshot_premium,
        "snapshot_premium_pct": pct,
        "new_count":           new_count,
        "new_premium_count":   new_premium_count,
    })

# ========= Scraping =========
def fetch_bild_html() -> str:
    """Fetch bild.de mit Retry + Backoff."""
    last_err: Exception = RuntimeError("unknown")
    for attempt in range(1, FETCH_RETRY + 1):
        try:
            resp = requests.get(URL, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_err = e
            if attempt < FETCH_RETRY:
                wait = 5 * attempt
                print(f"[WARN] bild.de fetch Versuch {attempt} fehlgeschlagen ({e}), warte {wait}s …")
                time.sleep(wait)
    raise RuntimeError(f"bild.de nicht erreichbar nach {FETCH_RETRY} Versuchen: {last_err}")

def get_articles_from_bild() -> List[Dict[str, Any]]:
    source = fetch_bild_html()
    soup = BeautifulSoup(source, "html.parser")
    script_tag = soup.find("script", {"id": "pageContext", "type": "application/json"})
    articles: List[Dict[str, Any]] = []

    if not script_tag:
        send_log("WARN: Kein <script id='pageContext'> gefunden.", iso_now_utc())
        return articles

    try:
        data = json.loads(script_tag.string or "{}")
        blocks = data["CLIENT_STORE_INITIAL_STATE"]["pageAggregation"]["curation"]["page"]["blocks"]
        for block in blocks:
            for layout in block.get("children", []):
                for child in layout.get("children", []):
                    if child.get("type") == "ARTICLE":
                        props = child.get("props", {}) or {}
                        url = props.get("url") or ""
                        category = url.split("/")[1] if url else None
                        articles.append({
                            "id":       props.get("id"),
                            "title":    props.get("title"),
                            "url":      url,
                            "category": category,
                            "isPremium": bool(props.get("isPremium", False)),
                        })
    except Exception as e:
        send_log(f"Fehler beim Parsen: {e}", iso_now_utc())

    valid = [a for a in articles if a.get("id") and a.get("title") and a.get("url")]
    send_log(f"Bild-Artikel gefunden: {len(valid)} (roh: {len(articles)})", iso_now_utc())
    return valid

# ========= Sync-Logik =========
def sync_bildwatch():
    start_ts = now_utc()
    time_str, date_str = fmt_berlin(start_ts)
    msg_start = f"Start scraping BILD at {time_str}, {date_str}"

    try:
        status_upsert("working", msg_start)
    except Exception as e:
        print(f"[WARN] Status-Start fehlgeschlagen: {e}")

    send_log(msg_start, start_ts.replace(microsecond=0).isoformat())

    try:
        # 1) Alle DB-Artikel laden (paginiert)
        db_rows = get_all_articles_from_db()
        db_by_id = {row["id"]: row for row in db_rows}
        send_log(f"DB-Artikel geladen: {len(db_rows)}", iso_now_utc())

        # 2) Bild.de scrapen
        scraped = get_articles_from_bild()
        scraped_by_id = {row["id"]: row for row in scraped}

        created = created_premium = skipped = patched = 0

        # 3) Neue Artikel anlegen
        for sid, srow in scraped_by_id.items():
            if sid not in db_by_id:
                try:
                    create_article(srow)
                    created += 1
                    if srow.get("isPremium"):
                        created_premium += 1
                    send_log(f"CREATE {sid} – {srow.get('title')!r}", iso_now_utc())
                except DuplicateError:
                    skipped += 1  # Race condition / Paginierungslücke – ignorieren
                except Exception as e:
                    send_log(f"ERR create {sid}: {e}", iso_now_utc())

        # 4) Premium → Frei Conversions
        now = now_utc()
        for sid, srow in scraped_by_id.items():
            db_row = db_by_id.get(sid)
            if not db_row:
                continue
            if bool(db_row.get("is_premium")) and not bool(srow.get("isPremium", False)):
                published_dt = parse_iso_dt(db_row.get("published"))
                duration = hours_between(published_dt, now)
                try:
                    patch_article(sid, {
                        "is_premium":               False,
                        "converted":                True,
                        "converted_time":           iso_now_utc(),
                        "converted_duration_hours": round(duration, 4) if duration is not None else None,
                    })
                    patched += 1
                    dur_str = f" nach {duration:.1f}h" if duration else ""
                    send_log(f"CONVERT {sid} premium→frei{dur_str}", iso_now_utc())
                except Exception as e:
                    send_log(f"ERR patch {sid}: {e}", iso_now_utc())

        # 5) Metrics
        snapshot_total   = len(scraped)
        snapshot_premium = sum(1 for a in scraped if a.get("isPremium"))
        ts_hour = now.replace(minute=0, second=0, microsecond=0)
        try:
            post_metrics(
                ts_hour_iso       = ts_hour.isoformat(),
                snapshot_total    = snapshot_total,
                snapshot_premium  = snapshot_premium,
                new_count         = created,
                new_premium_count = created_premium,
            )
        except Exception as e:
            send_log(f"ERR metrics: {e}", iso_now_utc())

        # 6) Abschluss
        end_ts = now_utc()
        mins = round((end_ts - start_ts).total_seconds() / 60.0, 2)
        time_str, date_str = fmt_berlin(end_ts)
        mins_str = f"{mins:.2f}".rstrip("0").rstrip(".")

        summary = (
            f"finished bild scraping at {time_str}, {date_str} after {mins_str}min"
            f" | neu={created} (premium={created_premium})"
            f" | conversions={patched}"
            f" | snapshot={snapshot_total} ({snapshot_premium} premium)"
        )
        send_log(summary, end_ts.replace(microsecond=0).isoformat())
        try:
            status_upsert("idle", f"finished bild scraping at {time_str}, {date_str} after {mins_str}min")
        except Exception as e:
            print(f"[WARN] Status-Ende fehlgeschlagen: {e}")

        print(f"[OK] {summary}")

    except Exception as e:
        end_ts = now_utc()
        mins = round((end_ts - start_ts).total_seconds() / 60.0, 2)
        time_str, date_str = fmt_berlin(end_ts)
        mins_str = f"{mins:.2f}".rstrip("0").rstrip(".")

        err_msg = f"SCRAPER ERROR at {time_str}, {date_str} after {mins_str}min: {e}"
        send_log(err_msg, end_ts.replace(microsecond=0).isoformat())
        try:
            status_upsert("error", err_msg)
        except Exception as e2:
            print(f"[FATAL] Status-Update fehlgeschlagen: {e2}")

        print(f"[FAIL] {err_msg}", file=sys.stderr)
        sys.exit(1)

# ========= Run =========
if __name__ == "__main__":
    sync_bildwatch()
