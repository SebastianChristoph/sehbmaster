# backend/app/routes/gov.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from typing import Optional, List, Dict, Any
from datetime import datetime
import os

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db import get_db_session, engine

router = APIRouter(prefix="/api/gov", tags=["gov"])

API_KEY = os.getenv("INGEST_API_KEY", "dev-secret")

def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True

# -----------------------------------------------------------------------------
# Schema (ohne Alembic) – idempotent
# -----------------------------------------------------------------------------
SCHEMA_SQL = """
-- Schema
CREATE SCHEMA IF NOT EXISTS "gov";

-- Incidents
CREATE TABLE IF NOT EXISTS "gov"."incidents" (
  id           SERIAL PRIMARY KEY,
  headline     TEXT NOT NULL,
  occurred_at  TIMESTAMPTZ,
  seen         BOOLEAN NOT NULL DEFAULT FALSE,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Für spätere, idempotente Erweiterungen
ALTER TABLE "gov"."incidents" ADD COLUMN IF NOT EXISTS occurred_at TIMESTAMPTZ;
ALTER TABLE "gov"."incidents" ADD COLUMN IF NOT EXISTS seen BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE "gov"."incidents" ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

-- Articles
CREATE TABLE IF NOT EXISTS "gov"."incident_articles" (
  id           SERIAL PRIMARY KEY,
  incident_id  INTEGER NOT NULL REFERENCES "gov"."incidents"(id) ON DELETE CASCADE,
  title        TEXT NOT NULL,
  source       TEXT NOT NULL,
  link         TEXT NOT NULL,
  published_at TIMESTAMPTZ
);

-- Eindeutigkeit der Links global erzwingen (über Unique-Index, idempotent)
CREATE UNIQUE INDEX IF NOT EXISTS "idx_gov_articles_link_unique"
  ON "gov"."incident_articles"(link);

-- Häufige Abfragen
CREATE INDEX IF NOT EXISTS "idx_gov_incidents_seen_created"
  ON "gov"."incidents"(seen, created_at DESC);

CREATE INDEX IF NOT EXISTS "idx_gov_articles_incident"
  ON "gov"."incident_articles"(incident_id);
"""

def ensure_gov_schema() -> None:
    with engine.begin() as conn:
        conn.execute(text(SCHEMA_SQL))

# -----------------------------------------------------------------------------
# Schemas (DTOs) – leichtgewichtig, ohne Pydantic-Model-Datei
# -----------------------------------------------------------------------------
def _row_to_incident(row) -> Dict[str, Any]:
    return {
        "id": row.id,
        "headline": row.headline,
        "occurred_at": row.occurred_at,
        "seen": row.seen,
        "created_at": row.created_at,
        "articles_count": row.articles_count,
    }

def _row_to_article(row) -> Dict[str, Any]:
    return {
        "id": row.id,
        "incident_id": row.incident_id,
        "title": row.title,
        "source": row.source,
        "link": row.link,
        "published_at": row.published_at,
    }

# -----------------------------------------------------------------------------
# Endpunkte
# -----------------------------------------------------------------------------

@router.get("/incidents")
def list_incidents(
    seen: Optional[bool] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_db_session),
):
    """
    Liste der Incidents (mit articles_count).
    """
    base_sql = """
        SELECT i.id, i.headline, i.occurred_at, i.seen, i.created_at,
               COALESCE(a.cnt, 0) AS articles_count
        FROM "gov"."incidents" i
        LEFT JOIN (
          SELECT incident_id, COUNT(*) AS cnt
          FROM "gov"."incident_articles"
          GROUP BY incident_id
        ) a ON a.incident_id = i.id
    """
    where = []
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    if seen is not None:
        where.append("i.seen = :seen")
        params["seen"] = seen

    if where:
        base_sql += " WHERE " + " AND ".join(where)

    base_sql += " ORDER BY i.created_at DESC LIMIT :limit OFFSET :offset"

    rows = session.execute(text(base_sql), params).mappings().all()
    return [_row_to_incident(r) for r in rows]


@router.get("/incidents/{incident_id}")
def get_incident_detail(
    incident_id: int,
    session: Session = Depends(get_db_session),
):
    """
    Detail: Incident + Artikel.
    """
    inc = session.execute(
        text('SELECT id, headline, occurred_at, seen, created_at FROM "gov"."incidents" WHERE id = :id'),
        {"id": incident_id},
    ).mappings().first()
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")

    arts = session.execute(
        text("""
            SELECT id, incident_id, title, source, link, published_at
            FROM "gov"."incident_articles"
            WHERE incident_id = :id
            ORDER BY published_at NULLS LAST, id ASC
        """),
        {"id": incident_id},
    ).mappings().all()

    return {
        "incident": {
            "id": inc.id,
            "headline": inc.headline,
            "occurred_at": inc.occurred_at,
            "seen": inc.seen,
            "created_at": inc.created_at,
            "articles_count": len(arts),
        },
        "articles": [_row_to_article(a) for a in arts],
    }


@router.patch("/incidents/{incident_id}/seen", dependencies=[Depends(require_api_key)])
def set_incident_seen(
    incident_id: int,
    payload: Dict[str, Any],
    session: Session = Depends(get_db_session),
):
    """
    { "seen": true|false }
    """
    if "seen" not in payload or not isinstance(payload["seen"], bool):
        raise HTTPException(status_code=400, detail="Body must contain boolean 'seen'")
    res = session.execute(
        text('UPDATE "gov"."incidents" SET seen = :seen WHERE id = :id'),
        {"seen": payload["seen"], "id": incident_id},
    )
    session.commit()
    if res.rowcount == 0:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"updated": res.rowcount}


@router.delete("/incidents/{incident_id}", status_code=204, dependencies=[Depends(require_api_key)])
def delete_incident(
    incident_id: int,
    session: Session = Depends(get_db_session),
):
    """
    Löscht den Incident (Artikel werden per ON DELETE CASCADE gelöscht).
    """
    res = session.execute(
        text('DELETE FROM "gov"."incidents" WHERE id = :id'),
        {"id": incident_id},
    )
    session.commit()
    if res.rowcount == 0:
        raise HTTPException(status_code=404, detail="Incident not found")
    return


@router.delete("/incidents/{incident_id}/articles/{article_id}", status_code=204, dependencies=[Depends(require_api_key)])
def delete_article(
    incident_id: int,
    article_id: int,
    session: Session = Depends(get_db_session),
):
    """
    Entfernt einen Artikel aus dem Incident.
    """
    res = session.execute(
        text('DELETE FROM "gov"."incident_articles" WHERE id = :aid AND incident_id = :iid'),
        {"aid": article_id, "iid": incident_id},
    )
    session.commit()
    if res.rowcount == 0:
        raise HTTPException(status_code=404, detail="Article not found")
    return


@router.post("/incidents/{incident_id}/articles", dependencies=[Depends(require_api_key)])
def add_article(
    incident_id: int,
    payload: Dict[str, Any],
    session: Session = Depends(get_db_session),
):
    """
    Artikel manuell hinzufügen.
    Body: { "title": str, "source": str, "link": str, "published_at": ISO8601|null }
    """
    for key in ("title", "source", "link"):
        if key not in payload or not isinstance(payload[key], str) or not payload[key].strip():
            raise HTTPException(status_code=400, detail=f"Missing or invalid field: {key}")

    published_at = payload.get("published_at")
    # Insert (unique link via unique index -> 409)
    try:
        row = session.execute(
            text("""
                INSERT INTO "gov"."incident_articles" (incident_id, title, source, link, published_at)
                VALUES (:iid, :title, :source, :link, :published_at)
                RETURNING id, incident_id, title, source, link, published_at
            """),
            {
                "iid": incident_id,
                "title": payload["title"].strip(),
                "source": payload["source"].strip(),
                "link": payload["link"].strip(),
                "published_at": published_at,
            },
        ).mappings().first()
        session.commit()
        return _row_to_article(row)
    except Exception as e:
        # Prüfen auf unique violation (SQLSTATE 23505)
        try:
            if getattr(e, "orig", None) and getattr(e.orig, "pgcode", "") == "23505":
                raise HTTPException(status_code=409, detail="duplicate link") from e
        except Exception:
            pass
        raise


# -------------------- ✳️ Gefährlich: Wipe -----------------------
@router.delete("/wipe", dependencies=[Depends(require_api_key)])
def wipe_all(confirm: Optional[str] = Query(None, description='Must be "yes" to proceed'),
             session: Session = Depends(get_db_session)):
    """
    ⚠️ Löscht ALLE gov-Daten. Nutzung:
      DELETE /api/gov/wipe?confirm=yes   (mit X-API-Key)

    Macht: TRUNCATE beider Tabellen + Identity-Reset.
    """
    if confirm != "yes":
        raise HTTPException(status_code=400, detail='Add query param confirm=yes to proceed')
    session.execute(text('TRUNCATE TABLE "gov"."incident_articles", "gov"."incidents" RESTART IDENTITY CASCADE;'))
    session.commit()
    return {"status": "wiped"}
