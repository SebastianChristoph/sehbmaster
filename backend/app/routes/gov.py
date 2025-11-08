# backend/app/routes/gov.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Header, status
from typing import Optional, List, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy import text
from sqlalchemy.orm import Session
import os

from ..db import get_session, get_db_session, engine

router = APIRouter(prefix="/api/gov", tags=["gov"])

API_KEY = os.getenv("INGEST_API_KEY", "dev-secret")
TZ = ZoneInfo("Europe/Berlin")  # nur für evtl. spätere Umrechnungen

# -------------------------------------------------------------------
# Schema & Tabellen idempotent anlegen (ohne Alembic)
# -------------------------------------------------------------------
def ensure_gov_schema() -> None:
    """
    Legt Schema und Tabellen an (idempotent). Keine Datenverluste.
    Wird beim App-Start aus main.py aufgerufen.
    """
    ddl = """
    CREATE SCHEMA IF NOT EXISTS "gov";

    CREATE TABLE IF NOT EXISTS "gov"."incidents" (
        id           SERIAL PRIMARY KEY,
        headline     TEXT NOT NULL,
        occurred_at  TIMESTAMPTZ NULL,
        seen         BOOLEAN NOT NULL DEFAULT FALSE,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS "gov"."incident_articles" (
        id            SERIAL PRIMARY KEY,
        incident_id   INTEGER NOT NULL REFERENCES "gov"."incidents"(id) ON DELETE CASCADE,
        title         TEXT NOT NULL,
        source        TEXT NOT NULL,
        link          TEXT NOT NULL UNIQUE,
        published_at  TIMESTAMPTZ NULL
    );

    CREATE INDEX IF NOT EXISTS idx_gov_incidents_seen_created
      ON "gov"."incidents"(seen, created_at DESC);

    CREATE INDEX IF NOT EXISTS idx_gov_incident_articles_incident
      ON "gov"."incident_articles"(incident_id);
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()


def _require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


# -------------------------------------------------------------------
# Pydantic-ähnliche leichte Validierung (wir benutzen Raw-SQL)
# -------------------------------------------------------------------
def _coerce_iso(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str or not isinstance(dt_str, str):
        return None
    try:
        # akzeptiert "YYYY-MM-DDTHH:MM:SSZ" oder mit Offset
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


# ============================ LISTE ===============================
@router.get("/incidents")
def list_incidents(
    seen: Optional[bool] = Query(None),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    """
    Liefert eine Liste der Incidents (ohne Artikel), neueste zuerst.
    Enthält `articles_count`.
    """
    with get_session() as s:
        where = []
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if seen is not None:
            where.append('i.seen = :seen')
            params["seen"] = seen
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        sql = f"""
        SELECT
            i.id,
            i.headline,
            i.occurred_at,
            i.seen,
            i.created_at,
            COALESCE((
                SELECT COUNT(1) FROM gov.incident_articles a WHERE a.incident_id = i.id
            ), 0) AS articles_count
        FROM gov.incidents i
        {where_sql}
        ORDER BY i.created_at DESC
        LIMIT :limit OFFSET :offset
        """
        rows = s.execute(text(sql), params).mappings().all()
        # JSON-ready
        return [
            {
                "id": r["id"],
                "headline": r["headline"],
                "occurred_at": (r["occurred_at"].isoformat() if r["occurred_at"] else None),
                "seen": r["seen"],
                "created_at": r["created_at"].isoformat(),
                "articles_count": int(r["articles_count"]),
            }
            for r in rows
        ]


# ============================ DETAIL ==============================
@router.get("/incidents/{incident_id}")
def get_incident_detail(incident_id: int):
    with get_session() as s:
        inc = s.execute(
            text("""
                SELECT id, headline, occurred_at, seen, created_at
                FROM gov.incidents WHERE id = :id
            """),
            {"id": incident_id},
        ).mappings().first()
        if not inc:
            raise HTTPException(status_code=404, detail="incident not found")

        arts = s.execute(
            text("""
                SELECT id, title, source, link, published_at
                FROM gov.incident_articles
                WHERE incident_id = :id
                ORDER BY COALESCE(published_at, '1970-01-01') DESC, id DESC
            """),
            {"id": incident_id},
        ).mappings().all()

        return {
            "id": inc["id"],
            "headline": inc["headline"],
            "occurred_at": (inc["occurred_at"].isoformat() if inc["occurred_at"] else None),
            "seen": inc["seen"],
            "created_at": inc["created_at"].isoformat(),
            "articles": [
                {
                    "id": a["id"],
                    "title": a["title"],
                    "source": a["source"],
                    "link": a["link"],
                    "published_at": (a["published_at"].isoformat() if a["published_at"] else None),
                }
                for a in arts
            ],
        }


# ============================ CREATE ==============================
@router.post("/incidents")
def create_incident(
    payload: Dict[str, Any],
    _=Depends(_require_api_key),
):
    """
    Body:
      {
        "headline": "…",               (required)
        "occurred_at": "ISO" | null,   (optional – bei dir oft null)
        "articles": [
          {"title":"…","source":"…","link":"…","published_at":"ISO"|null}, ...
        ]
      }

    Besonderheit: Duplikat-Check "an diesem Tag schon vorhanden".
    - Wenn occurred_at angegeben -> vergleiche occurred_at::date.
    - Wenn occurred_at fehlt -> vergleiche created_at::date = heute (Serverzeit UTC).
    Bei Treffer -> 409 mit Meldung.
    """
    headline = (payload.get("headline") or "").strip()
    occurred_at = _coerce_iso(payload.get("occurred_at"))
    articles = payload.get("articles") or []

    if not headline:
        raise HTTPException(status_code=422, detail="headline required")

    with get_session() as s:
        # Tag ableiten
        if occurred_at:
            sql_day_check = text("""
                SELECT id FROM gov.incidents
                WHERE DATE(occurred_at AT TIME ZONE 'UTC') = DATE(:occ AT TIME ZONE 'UTC')
                LIMIT 1
            """)
            dup = s.execute(sql_day_check, {"occ": occurred_at}).mappings().first()
        else:
            sql_day_check = text("""
                SELECT id FROM gov.incidents
                WHERE DATE(created_at AT TIME ZONE 'UTC') = DATE(NOW() AT TIME ZONE 'UTC')
                LIMIT 1
            """)
            dup = s.execute(sql_day_check).mappings().first()

        if dup:
            raise HTTPException(status_code=409, detail="Es existiert bereits ein Vorfall für diesen Tag.")

        # Incident anlegen
        inc = s.execute(
            text("""
                INSERT INTO gov.incidents (headline, occurred_at)
                VALUES (:h, :occ)
                RETURNING id, created_at, seen
            """),
            {"h": headline, "occ": occurred_at},
        ).mappings().first()

        inc_id = inc["id"]

        # Artikel anhängen (Duplikate per UNIQUE(link) werden abgefangen)
        created = 0
        for a in articles:
            title = (a.get("title") or "").strip()
            source = (a.get("source") or "").strip()
            link = (a.get("link") or "").strip()
            pub = _coerce_iso(a.get("published_at"))
            if not (title and source and link):
                continue
            try:
                s.execute(
                    text("""
                        INSERT INTO gov.incident_articles
                          (incident_id, title, source, link, published_at)
                        VALUES (:iid, :t, :s, :l, :p)
                    """),
                    {"iid": inc_id, "t": title, "s": source, "l": link, "p": pub},
                )
                created += 1
            except Exception:
                # Duplicate link -> ignorieren
                s.rollback()
                # damit die Transaktion offen bleibt, unmittelbar neuen Begin:
                pass
        s.commit()

        # Detail zurückgeben
        detail = get_incident_detail(inc_id)
        return detail


# ============================ ADD ARTICLE =========================
@router.post("/incidents/{incident_id}/articles")
def add_article(
    incident_id: int,
    payload: Dict[str, Any],
    _=Depends(_require_api_key),
):
    title = (payload.get("title") or "").strip()
    source = (payload.get("source") or "").strip()
    link = (payload.get("link") or "").strip()
    published_at = _coerce_iso(payload.get("published_at"))

    if not (title and source and link):
        raise HTTPException(status_code=422, detail="title, source, link required")

    with get_session() as s:
        # existiert Incident?
        inc = s.execute(text("SELECT 1 FROM gov.incidents WHERE id=:id"), {"id": incident_id}).first()
        if not inc:
            raise HTTPException(status_code=404, detail="incident not found")

        try:
            s.execute(
                text("""
                    INSERT INTO gov.incident_articles
                      (incident_id, title, source, link, published_at)
                    VALUES (:iid, :t, :s, :l, :p)
                """),
                {"iid": incident_id, "t": title, "s": source, "l": link, "p": published_at},
            )
            s.commit()
        except Exception:
            s.rollback()
            # UNIQUE(link) verletzt -> 409
            raise HTTPException(status_code=409, detail="duplicate link")

        return {"ok": True}


# ============================ SEEN PATCH ==========================
@router.patch("/incidents/{incident_id}/seen")
def patch_incident_seen(
    incident_id: int,
    seen: bool = Query(..., description="true|false"),
    _=Depends(_require_api_key),
):
    with get_session() as s:
        res = s.execute(
            text("""UPDATE gov.incidents SET seen=:seen WHERE id=:id"""),
            {"seen": seen, "id": incident_id},
        )
        s.commit()
        return {"updated": res.rowcount}


# ============================ DELETE ==============================
@router.delete("/incidents/{incident_id}")
def delete_incident(incident_id: int, _=Depends(_require_api_key)):
    with get_session() as s:
        s.execute(text("""DELETE FROM gov.incidents WHERE id=:id"""), {"id": incident_id})
        s.commit()
    return {"deleted": 1}


@router.delete("/incidents/{incident_id}/articles/{article_id}")
def delete_article(incident_id: int, article_id: int, _=Depends(_require_api_key)):
    with get_session() as s:
        s.execute(
            text("""DELETE FROM gov.incident_articles WHERE id=:aid AND incident_id=:iid"""),
            {"aid": article_id, "iid": incident_id},
        )
        s.commit()
    return {"deleted": 1}


# ============================ WIPE (Danger!) ======================
@router.delete("/wipe")
def wipe_everything(confirm: Optional[str] = Query(None, description='bestätige mit "yes"'), _=Depends(_require_api_key)):
    if confirm != "yes":
        raise HTTPException(status_code=400, detail='Missing confirm="yes"')
    with get_session() as s:
        # Reihenfolge wegen FK
        s.execute(text('TRUNCATE TABLE "gov"."incident_articles" RESTART IDENTITY CASCADE;'))
        s.execute(text('TRUNCATE TABLE "gov"."incidents" RESTART IDENTITY CASCADE;'))
        s.commit()
    return {"status": "wiped"}
