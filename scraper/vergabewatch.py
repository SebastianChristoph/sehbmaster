"""
Vergabewatch Scraper
Trackt EU-Vergabebekanntmachungen (Contract Award Notices) via TED API.
API: https://api.ted.europa.eu/v3/notices/search (kein API-Key noetig)
XML pro Notice: https://ted.europa.eu/en/notice/{pub_number}/xml
"""
import requests
import xml.etree.ElementTree as ET
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")
API_BASE   = os.getenv("SEHBMASTER_API",     "http://localhost:8000/api").rstrip("/")
API_KEY    = os.getenv("SEHBMASTER_API_KEY", "dev-secret")
SCRAPER_ID = os.getenv("SCRAPER_ID",         "server-vergabewatch")
TIMEOUT    = 30
TED_SEARCH = "https://api.ted.europa.eu/v3/notices/search"
TED_XML    = "https://ted.europa.eu/en/notice/{pub}/xml"

# CPV top-level descriptions (simplified)
CPV_LABELS = {
    "03": "Land-/Forstwirtschaft", "09": "Mineralölerzeugnisse", "14": "Bergbau",
    "15": "Nahrungsmittel", "16": "Landmaschinen", "18": "Bekleidung",
    "19": "Leder", "22": "Druckerzeugnisse", "24": "Chemikalien",
    "30": "Büromaschinen/IT", "31": "Elektrische Ausrüstung", "32": "Kommunikation",
    "33": "Medizinische Ausrüstung", "34": "Fahrzeuge", "35": "Sicherheit",
    "37": "Musikinstrumente/Sport", "38": "Laborgeräte", "39": "Möbel",
    "41": "Wasser", "42": "Maschinen", "43": "Bergbaumaschinen",
    "44": "Baumaterialien", "45": "Bauarbeiten", "48": "Software",
    "50": "Reparatur", "51": "Installation", "55": "Gaststätten/Beherbergung",
    "60": "Transport", "63": "Hilfsleistungen Transport", "64": "Post/Telekommunikation",
    "65": "Versorgung", "66": "Finanzdienstleistungen", "70": "Immobilien",
    "71": "Architektur/Ingenieurwesen", "72": "IT-Dienstleistungen",
    "73": "Forschung/Entwicklung", "75": "Verwaltung/Verteidigung",
    "76": "Öldienstleistungen", "77": "Forstwirtschaft", "79": "Unternehmensberatung",
    "80": "Bildung", "85": "Gesundheit/Soziales", "90": "Abwasser/Abfall",
    "92": "Erholung/Kultur", "98": "Sonstige Dienstleistungen",
}

def cpv_label(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    prefix = code[:2]
    return CPV_LABELS.get(prefix, f"CPV {prefix}xxx")

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
        _req("POST", "/vergabe/logs", {"message": msg, "timestamp": iso_now()})
    except Exception as e:
        print(f"[log-fallback] {e}")

def status_upsert(status: str, message: Optional[str] = None):
    try:
        _req("POST", "/status", {"raspberry": SCRAPER_ID, "status": status, "message": message})
    except Exception as e:
        print(f"[WARN] status upsert failed: {e}")

# ======= TED API =======
def fetch_ted_notices(days_back: int = 1) -> List[str]:
    """Returns publication numbers of DE contract award notices from the last N days."""
    since = (now_utc() - timedelta(days=days_back)).strftime("%Y%m%d")
    pub_numbers = []
    page = 1

    while True:
        payload = {
            "query": f"place-of-performance IN (DEU) AND PD >= {since}",
            "fields": ["publication-number"],
            "limit": 250,
            "page": page,
            "scope": "ACTIVE",
            "paginationMode": "PAGE_NUMBER",
        }
        try:
            resp = requests.post(TED_SEARCH, json=payload,
                                 headers={"Content-Type": "application/json"},
                                 timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            send_log(f"WARN TED search page {page} failed: {e}")
            break

        notices = data.get("notices", [])
        for n in notices:
            pub_numbers.append(n["publication-number"])

        total = data.get("totalNoticeCount", 0)
        if len(pub_numbers) >= total or len(notices) == 0:
            break
        page += 1
        time.sleep(0.3)

    return pub_numbers

def parse_ted_xml(pub_number: str) -> Optional[Dict]:
    """Fetches and parses TED XML for a notice. Returns structured dict or None."""
    url = TED_XML.format(pub=pub_number)
    try:
        resp = requests.get(url, timeout=TIMEOUT,
                            headers={"Accept": "application/xml, text/xml, */*"})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        xml_text = resp.text
    except Exception as e:
        print(f"[WARN] XML fetch {pub_number}: {e}")
        return None

    try:
        root = ET.fromstring(xml_text)

        # Collect all tag->values for quick lookup (eForms UBL format)
        tag_values: Dict[str, List[str]] = {}
        for el in root.iter():
            tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
            text = (el.text or "").strip()
            if text:
                tag_values.setdefault(tag, []).append(text)

        def first(tag: str) -> Optional[str]:
            return tag_values.get(tag, [None])[0]

        def first_float(tag: str) -> Optional[float]:
            v = first(tag)
            if v:
                try:
                    return float(v.replace(",", "."))
                except ValueError:
                    pass
            return None

        # eForms fields (new format, post-2021)
        # Contracting authority: first Name occurrence is the buyer
        # Contractor: FamilyName (org name in eForms winner section) or later Name
        all_names = tag_values.get("Name", [])
        contracting_authority = all_names[0] if all_names else None
        contractor_name = (
            first("FamilyName")          # eForms winner org name
            or (all_names[1] if len(all_names) > 1 else None)
        )

        # Value: TotalAmount > PayableAmount > PriceAmount
        contract_value = (
            first_float("TotalAmount")
            or first_float("PayableAmount")
            or first_float("PriceAmount")
        )

        # CPV code: ItemClassificationCode (eForms) or CPV_CODE (legacy)
        cpv_code = first("ItemClassificationCode") or first("CPV_CODE")
        if cpv_code and "@" in cpv_code:  # sometimes "45442110@cpv" format
            cpv_code = cpv_code.split("@")[0]

        # Date
        pub_raw = first("PublicationDate") or first("IssueDate")
        published_date = pub_raw[:10] if pub_raw else None  # keep YYYY-MM-DD

        notice_type = first("NoticeTypeCode") or first("TD_DOCUMENT_TYPE") or "CAN"
        procedure_type = first("ProcedureCode") or first("PR_TYPE")
        description = first("Description") or first("Title")
        if description:
            description = description[:500]

        return {
            "publication_number": pub_number,
            "notice_type": notice_type,
            "published_date": published_date,
            "contracting_authority": contracting_authority,
            "contracting_country": "DEU",
            "contractor_name": contractor_name,
            "contract_value_eur": contract_value,
            "cpv_code": cpv_code,
            "cpv_description": cpv_label(cpv_code),
            "procedure_type": procedure_type,
            "description": description,
            "ted_url": f"https://ted.europa.eu/en/notice/-/detail/{pub_number}",
            "raw_xml_url": url,
        }

    except Exception as e:
        print(f"[WARN] XML parse {pub_number}: {e}")
        return None

def get_known_pub_numbers() -> set:
    try:
        rows = _req("GET", "/vergabe/notices?limit=5000") or []
        return {r["publication_number"] for r in rows}
    except Exception:
        return set()

# ======= Pattern Detection =======
def run_pattern_alerts():
    """Detect repeat-winner patterns from DB."""
    try:
        stats = _req("GET", "/vergabe/notices/stats") or {}
        top = stats.get("top_contractors", [])
        for c in top:
            if c["count"] >= 3:
                _req("POST", "/vergabe/alerts", {
                    "alert_type": "REPEAT_WINNER",
                    "contractor": c["name"],
                    "evidence": {"count": c["count"]},
                })
    except Exception as e:
        send_log(f"WARN pattern detection: {e}")

# ======= Main =======
def sync_vergabewatch():
    start = now_utc()
    status_upsert("working", f"Start vergabewatch scraping at {fmt_berlin(start)}")
    send_log(f"Start vergabewatch scraping at {fmt_berlin(start)}")

    try:
        known = get_known_pub_numbers()
        send_log(f"Bekannte Notices in DB: {len(known)}")

        pub_numbers = fetch_ted_notices(days_back=2)
        send_log(f"TED API geliefert: {len(pub_numbers)} Notices (letzte 2 Tage, DEU)")

        new_count = skip_count = err_count = 0

        for pub in pub_numbers:
            if pub in known:
                skip_count += 1
                continue

            parsed = parse_ted_xml(pub)
            if not parsed:
                err_count += 1
                continue

            try:
                _req("POST", "/vergabe/notices", parsed)
                new_count += 1
                send_log(f"NEW {pub} | {parsed.get('contracting_authority', '?')} → {parsed.get('contractor_name', '?')} | {parsed.get('contract_value_eur', '?')} EUR")
            except Exception as e:
                send_log(f"ERR upsert {pub}: {e}")
                err_count += 1

            time.sleep(0.2)

        if new_count > 0:
            run_pattern_alerts()

        end = now_utc()
        mins = round((end - start).total_seconds() / 60, 2)
        summary = (
            f"finished vergabewatch scraping at {fmt_berlin(end)} after {mins:.1f}min"
            f" | neu={new_count} | skip={skip_count} | fehler={err_count}"
        )
        send_log(summary)
        status_upsert("idle", summary)
        print(f"[OK] {summary}")

    except Exception as e:
        end = now_utc()
        mins = round((end - start).total_seconds() / 60, 2)
        err = f"SCRAPER ERROR vergabewatch at {fmt_berlin(end)} after {mins:.1f}min: {e}"
        send_log(err)
        status_upsert("error", err)
        print(f"[FAIL] {err}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    sync_vergabewatch()
