# backend/app/routes/gov.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, AnyUrl
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.orm import Session
import os

from ..db import get_session, get_db_session  # wie in bild.py genutzt

router = APIRouter(prefix="/api/gov", tags=["gov"])

API_KEY = os.getenv("INGEST_API_KEY", "dev-secret")

# ==========================
# Init (ohne Alembic)
# ==========================
INIT_SQL = """
CREATE SCHEMA IF NOT EXISTS "gov";

CREATE TABLE IF NOT EXISTS "gov"."incidents" (
  id           SERIAL PRIMARY KEY,
  headline     TEXT NOT NULL,
  occurred_at  TIMESTAMPTZ NULL,
  seen         BOOLEAN NOT NULL DEFAULT FALSE,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "gov"."articles" (
  id           SERIAL PRIMARY KEY,
  incident_id  INTEGER NOT NULL REFERENCES "gov"."incidents"(id) ON DELETE CASCADE,
  title        TEXT NOT NULL,
  source       TEXT NOT NULL,
  link         TEXT NOT NULL UNIQUE,
  published_at TIMESTAMPTZ NULL
);

-- Häufige Abfragen
CREATE INDEX IF NOT EXISTS idx_gov_incidents_seen_created
  ON "gov"."incidents"(seen, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gov_articles_incident
  ON "gov"."articles"(incident_id);
"""

def _ensure_gov_schema():
    with get_session() as s:
        s.execute(text(INIT_SQL))
        s.commit()

# beim Import einmal sicherstellen
_ensure_gov_schema()

# ==========================
# Schemas (DTOs)
# ==========================
class GovArticleIn(BaseModel):
    title: str
    source: str
    link: AnyUrl | str
    published_at: Optional[datetime] = None

class GovIncidentCreateIn(BaseModel):
    headline: str
    occurred_at: Optional[datetime] = None
    articles: List[GovArticleIn] = []

class GovIncidentListOut(BaseModel):
    id: int
    headline: str
    occurred_at: Optional[datetime] = None
    seen: bool
    created_at: datetime
    articles_count: int

class GovArticleOut(BaseModel):
    id: int
    title: str
    source: str
    link: str
    published_at: Optional[datetime] = None

class GovIncidentDetailOut(BaseModel):
    id: int
    headline: str
    occurred_at: Optional[datetime] = None
    seen: bool
    created_at: datetime
    articles: List[GovArticleOut]

class SeenPatchIn(BaseModel):
    seen: bool

# ==========================
# Helpers
# ==========================
def _require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True

# ==========================
# Endpoints
# ==========================

@router.get("/incidents", response_model=List[GovIncidentListOut])
def list_incidents(
    seen: Optional[bool] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Liefert Vorfälle (ohne Artikel) inkl. articles_count.
    """
    _ensure_gov_schema()
    with get_session() as s:
        where = ""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if seen is not None:
            where = "WHERE i.seen = :seen"
            params["seen"] = seen

        q = text(f"""
            SELECT
              i.id, i.headline, i.occurred_at, i.seen, i.created_at,
              COALESCE(a.cnt, 0) AS articles_count
            FROM "gov"."incidents" i
            LEFT JOIN (
              SELECT incident_id, COUNT(*) AS cnt
              FROM "gov"."articles"
              GROUP BY incident_id
            ) a ON a.incident_id = i.id
            {where}
            ORDER BY i.created_at DESC
            LIMIT :limit OFFSET :offset
        """)
        rows = s.execute(q, params).mappings().all()
        return [GovIncidentListOut(**row) for row in rows]

@router.get("/incidents/{incident_id}", response_model=GovIncidentDetailOut)
def get_incident(incident_id: int):
    _ensure_gov_schema()
    with get_session() as s:
        inc_q = text("""
            SELECT id, headline, occurred_at, seen, created_at
            FROM "gov"."incidents"
            WHERE id = :id
        """)
        inc = s.execute(inc_q, {"id": incident_id}).mappings().first()
        if not inc:
            raise HTTPException(status_code=404, detail="Incident not found")

        art_q = text("""
            SELECT id, title, source, link, published_at
            FROM "gov"."articles"
            WHERE incident_id = :id
            ORDER BY published_at NULLS LAST, id ASC
        """)
        arts = s.execute(art_q, {"id": incident_id}).mappings().all()

        return GovIncidentDetailOut(
            id=inc["id"],
            headline=inc["headline"],
            occurred_at=inc["occurred_at"],
            seen=inc["seen"],
            created_at=inc["created_at"],
            articles=[GovArticleOut(**a) for a in arts],
        )

@router.post("/incidents", response_model=GovIncidentDetailOut, status_code=201, dependencies=[Depends(_require_api_key)])
def create_incident(payload: GovIncidentCreateIn):
    """
    Legt einen Vorfall + zugehörige Artikel an.
    - duplicate Artikel-Links (UNIQUE) werden still übersprungen.
    """
    _ensure_gov_schema()
    with get_session() as s:
        inc_q = text("""
            INSERT INTO "gov"."incidents"(headline, occurred_at)
            VALUES (:headline, :occurred_at)
            RETURNING id, headline, occurred_at, seen, created_at
        """)
        inc = s.execute(inc_q, {"headline": payload.headline, "occurred_at": payload.occurred_at}).mappings().first()

        # Artikel einfügen
        if payload.articles:
            ins = text("""
                INSERT INTO "gov"."articles"(incident_id, title, source, link, published_at)
                VALUES (:incident_id, :title, :source, :link, :published_at)
                ON CONFLICT (link) DO NOTHING
            """)
            for a in payload.articles:
                s.execute(ins, {
                    "incident_id": inc["id"],
                    "title": a.title,
                    "source": a.source,
                    "link": str(a.link),
                    "published_at": a.published_at
                })

        s.commit()

        # Detail zurückgeben
        arts = s.execute(
            text('SELECT id, title, source, link, published_at FROM "gov"."articles" WHERE incident_id = :id ORDER BY id ASC'),
            {"id": inc["id"]}
        ).mappings().all()

        return GovIncidentDetailOut(
            id=inc["id"],
            headline=inc["headline"],
            occurred_at=inc["occurred_at"],
            seen=inc["seen"],
            created_at=inc["created_at"],
            articles=[GovArticleOut(**a) for a in arts],
        )

@router.patch("/incidents/{incident_id}/seen", dependencies=[Depends(_require_api_key)])
def patch_incident_seen(incident_id: int, payload: SeenPatchIn):
    _ensure_gov_schema()
    with get_session() as s:
        upd = s.execute(
            text('UPDATE "gov"."incidents" SET seen = :seen WHERE id = :id'),
            {"seen": payload.seen, "id": incident_id}
        )
        s.commit()
        if upd.rowcount == 0:
            raise HTTPException(status_code=404, detail="Incident not found")
        return {"updated": upd.rowcount}

@router.delete("/incidents/{incident_id}", status_code=204, dependencies=[Depends(_require_api_key)])
def delete_incident(incident_id: int):
    _ensure_gov_schema()
    with get_session() as s:
        res = s.execute(text('DELETE FROM "gov"."incidents" WHERE id = :id'), {"id": incident_id})
        s.commit()
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Incident not found")
    return

@router.delete("/incidents/{incident_id}/articles/{article_id}", status_code=204, dependencies=[Depends(_require_api_key)])
def delete_article_from_incident(incident_id: int, article_id: int):
    _ensure_gov_schema()
    with get_session() as s:
        res = s.execute(
            text('DELETE FROM "gov"."articles" WHERE id = :aid AND incident_id = :iid'),
            {"aid": article_id, "iid": incident_id}
        )
        s.commit()
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Article not found")
    return

@router.delete("/wipe", dependencies=[Depends(_require_api_key)])
def wipe_everything(confirm: str = Query("no")):
    """
    ⚠️ Löscht ALLES im gov-Schema. Nur wenn confirm=yes.
    """
    if confirm.lower() != "yes":
        raise HTTPException(status_code=400, detail='Setze confirm=yes, wenn du wirklich alles löschen willst.')
    _ensure_gov_schema()
    with get_session() as s:
        s.execute(text('TRUNCATE TABLE "gov"."articles" RESTART IDENTITY CASCADE;'))
        s.execute(text('TRUNCATE TABLE "gov"."incidents" RESTART IDENTITY CASCADE;'))
        s.commit()
    return {"status": "wiped"}
