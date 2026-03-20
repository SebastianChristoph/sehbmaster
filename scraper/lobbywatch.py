"""
Lobbywatch Scraper
Trackt Änderungen im Lobbyregister des Deutschen Bundestags.
API: https://www.lobbyregister.bundestag.de/sucheDetailJson
"""
import requests
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")
API_BASE   = os.getenv("SEHBMASTER_API",     "http://localhost:8000/api").rstrip("/")
API_KEY    = os.getenv("SEHBMASTER_API_KEY", "dev-secret")
SCRAPER_ID = os.getenv("SCRAPER_ID",         "server-lobbywatch")
TIMEOUT    = 30
LOBBY_API  = "https://www.lobbyregister.bundestag.de"

# ======= HTTP-Helper =======
class ApiError(RuntimeError): pass

def _req(method: str, path: str, json_body: Optional[dict] = None) -> Any:
    url = f"{API_BASE}{path}"
    headers = {"Accept": "application/json"}
    if method in ("POST", "PATCH", "PUT", "DELETE"):
        headers["Content-Type"] = "application/json"
        headers["X-API-Key"] = API_KEY
    try:
        resp = requests.request(method, url, json=json_body, headers=headers, timeout=15)
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()
    except Exception as e:
        raise ApiError(f"{method} {url} failed: {e}") from e

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def iso_now() -> str:
    return now_utc().replace(microsecond=0).isoformat()

def fmt_berlin(dt: datetime) -> str:
    return dt.astimezone(BERLIN).strftime("%H:%M %d.%m.%Y")

def send_log(msg: str):
    print(f"[LOG] {msg}")
    try:
        _req("POST", "/lobby/logs", {"message": msg, "timestamp": iso_now()})
    except Exception as e:
        print(f"[log-fallback] {e}")

def status_upsert(status: str, message: Optional[str] = None):
    try:
        _req("POST", "/status", {"raspberry": SCRAPER_ID, "status": status, "message": message})
    except Exception as e:
        print(f"[WARN] status upsert failed: {e}")

# ======= Lobbyregister API =======
def fetch_all_entries(query: str = "", sort: str = "REGISTRATION_DESC") -> List[Dict]:
    """Fetches all entries matching query. Without query: all 6000+ entries."""
    url = f"{LOBBY_API}/sucheDetailJson"
    params = {"q": query, "sort": sort}
    headers = {"Accept": "application/json", "User-Agent": "lobbywatch-research/1.0"}
    resp = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])

def fetch_entry_detail(register_number: str, entry_id: int) -> Optional[Dict]:
    """Fetches a specific version of a lobby entry."""
    url = f"{LOBBY_API}/sucheJson/{register_number}/{entry_id}"
    headers = {"Accept": "application/json", "User-Agent": "lobbywatch-research/1.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[WARN] Failed to fetch {register_number}/{entry_id}: {e}")
        return None

# ======= DB-Calls =======
def get_db_entries() -> Dict[str, Dict]:
    """Returns all known entries keyed by register_number."""
    try:
        rows = _req("GET", "/lobby/entries?limit=10000&active_only=false") or []
        return {r["register_number"]: r for r in rows}
    except Exception as e:
        send_log(f"ERR loading DB entries: {e}")
        return {}

def upsert_entry(data: Dict):
    _req("POST", "/lobby/entries", data)

def post_change(data: Dict):
    _req("POST", "/lobby/changes", data)

def post_alert(data: Dict):
    _req("POST", "/lobby/alerts", data)

# ======= Entry Parsing =======
def parse_entry(raw: Dict) -> Dict:
    """Extracts flat fields from a lobbyregister API result entry."""
    acc = raw.get("accountDetails", {})
    details = raw.get("registerEntryDetails", {})
    org = raw.get("organisation", {})
    activity = raw.get("activityDetails", {})

    # Financial
    fin = raw.get("financialDetails", {})
    exp = fin.get("financialExpensesEuro") or {}
    exp_from = exp.get("from")
    exp_to = exp.get("to")
    refuse_fin = fin.get("refuseFinancialExpensesInformation", False)

    # Fields of interest
    fois = raw.get("fieldsOfInterest", []) or []
    foi_codes = [f.get("code") for f in fois if f.get("code")]

    # Clients
    client_orgs = raw.get("clientOrganizations", []) or []
    client_persons = raw.get("clientPersons", []) or []

    # Legislative projects
    leg_projects = activity.get("legislativeProjects", []) or []

    # Codex
    codex_violation = raw.get("codexViolation", False) or False

    return {
        "register_number": raw.get("registerNumber"),
        "name": org.get("name") or raw.get("name"),
        "legal_form": (org.get("legalForm") or {}).get("descriptionDe"),
        "first_publication_date": acc.get("firstPublicationDate"),
        "last_update_date": acc.get("lastUpdateDate"),
        "active": acc.get("activeLobbyist", True),
        "current_entry_id": details.get("registerEntryId"),
        "financial_expenses_from": float(exp_from) if exp_from is not None else None,
        "financial_expenses_to": float(exp_to) if exp_to is not None else None,
        "refuse_financial_info": bool(refuse_fin),
        "codex_violation": bool(codex_violation),
        "fields_of_interest": foi_codes,
        "client_orgs": [{"name": c.get("name"), "country": (c.get("address") or {}).get("country")} for c in client_orgs],
        "client_persons": [{"firstName": p.get("firstName"), "lastName": p.get("lastName")} for p in client_persons],
        "legislative_projects": [{"name": lp.get("name"), "printingNumber": lp.get("printingNumber")} for lp in leg_projects],
        "raw_json": raw,
    }

# ======= Diff Logic =======
TRACKED_FIELDS = [
    "name", "active", "financial_expenses_from", "financial_expenses_to",
    "refuse_financial_info", "codex_violation", "fields_of_interest",
    "client_orgs", "client_persons", "legislative_projects",
]

def compute_diff(old: Dict, new: Dict) -> Dict:
    diff = {}
    for field in TRACKED_FIELDS:
        old_val = old.get(field)
        new_val = new.get(field)
        if old_val != new_val:
            diff[field] = {"old": old_val, "new": new_val}
    return diff

def classify_change(diff: Dict) -> str:
    if "codex_violation" in diff and diff["codex_violation"]["new"]:
        return "CODEX_VIOLATION"
    if "client_orgs" in diff or "client_persons" in diff:
        return "CLIENT_CHANGE"
    if "financial_expenses_from" in diff or "refuse_financial_info" in diff:
        return "FINANCIAL_CHANGE"
    if "active" in diff:
        return "ACTIVE_CHANGE"
    if "legislative_projects" in diff:
        return "LEGISLATIVE_CHANGE"
    return "FIELD_CHANGE"

def check_alerts(parsed: Dict, is_new: bool, diff: Dict, register_number: str):
    """Post alerts for noteworthy findings."""
    if parsed.get("codex_violation"):
        post_alert({
            "register_number": register_number,
            "alert_type": "CODEX_VIOLATION",
            "description": f"{parsed.get('name')} hat einen Ethik-Code-Verstoß",
            "evidence": {"name": parsed.get("name")},
        })
    if parsed.get("refuse_financial_info"):
        post_alert({
            "register_number": register_number,
            "alert_type": "FINANCIAL_NONDISCLOSURE",
            "description": f"{parsed.get('name')} verweigert Finanzangaben",
            "evidence": {"name": parsed.get("name")},
        })
    if is_new:
        post_alert({
            "register_number": register_number,
            "alert_type": "NEW_ENTRY",
            "description": f"Neu registriert: {parsed.get('name')}",
            "evidence": {"name": parsed.get("name"), "fields_of_interest": parsed.get("fields_of_interest")},
        })
    if not is_new and "client_orgs" in diff:
        old_names = {c.get("name") for c in (diff["client_orgs"]["old"] or [])}
        new_names = {c.get("name") for c in (diff["client_orgs"]["new"] or [])}
        added = new_names - old_names
        if added:
            post_alert({
                "register_number": register_number,
                "alert_type": "NEW_CLIENT",
                "description": f"{parsed.get('name')} hat neue Auftraggeber: {', '.join(added)}",
                "evidence": {"added_clients": list(added)},
            })

# ======= Main Sync =======
def sync_lobbywatch():
    start = now_utc()
    status_upsert("working", f"Start lobbywatch scraping at {fmt_berlin(start)}")
    send_log(f"Start lobbywatch scraping at {fmt_berlin(start)}")

    try:
        # Load DB state
        db_entries = get_db_entries()
        send_log(f"DB-Eintraege geladen: {len(db_entries)}")

        # Fetch recent changes from Lobbyregister (sorted by last update)
        send_log("Fetche Lobbyregister (letzte Aenderungen)...")
        recent = fetch_all_entries(query="", sort="REGISTRATION_DESC")
        send_log(f"Lobbyregister geliefert: {len(recent)} Eintraege")

        new_count = changed_count = unchanged_count = 0

        for raw in recent:
            reg_num = raw.get("registerNumber")
            if not reg_num:
                continue

            acc = raw.get("accountDetails", {})
            api_last_update = acc.get("lastUpdateDate")
            db_row = db_entries.get(reg_num)

            # Skip if not changed
            if db_row and db_row.get("last_update_date") and api_last_update:
                # Compare timestamps
                api_ts = api_last_update[:19]  # truncate to seconds for comparison
                db_ts  = db_row["last_update_date"][:19]
                if api_ts == db_ts:
                    unchanged_count += 1
                    continue

            # Fetch full detail for changed/new entries
            details = raw.get("registerEntryDetails", {})
            entry_id = details.get("registerEntryId")
            if not entry_id:
                continue

            full = fetch_entry_detail(reg_num, entry_id)
            if not full:
                continue

            parsed = parse_entry(full)
            is_new = reg_num not in db_entries

            if is_new:
                new_count += 1
                send_log(f"NEW {reg_num} – {parsed.get('name')}")
            else:
                diff = compute_diff(db_entries[reg_num], parsed)
                if diff:
                    changed_count += 1
                    change_type = classify_change(diff)
                    post_change({
                        "register_number": reg_num,
                        "old_entry_id": db_entries[reg_num].get("current_entry_id"),
                        "new_entry_id": entry_id,
                        "change_type": change_type,
                        "diff": diff,
                        "notes": parsed.get("name"),
                    })
                    send_log(f"CHANGE {reg_num} [{change_type}] – {parsed.get('name')}")

            check_alerts(parsed, is_new, {} if is_new else diff, reg_num)
            upsert_entry(parsed)
            time.sleep(0.1)  # polite crawl

        end = now_utc()
        mins = round((end - start).total_seconds() / 60, 2)
        summary = (
            f"finished lobbywatch scraping at {fmt_berlin(end)} after {mins:.1f}min"
            f" | neu={new_count} | geaendert={changed_count} | unveraendert={unchanged_count}"
        )
        send_log(summary)
        status_upsert("idle", summary)
        print(f"[OK] {summary}")

    except Exception as e:
        end = now_utc()
        mins = round((end - start).total_seconds() / 60, 2)
        err = f"SCRAPER ERROR lobbywatch at {fmt_berlin(end)} after {mins:.1f}min: {e}"
        send_log(err)
        status_upsert("error", err)
        print(f"[FAIL] {err}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    sync_lobbywatch()
