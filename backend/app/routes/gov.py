# backend/app/routes/gov.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import os

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db import get_session, get_db_session, engine

router = APIRouter(prefix="/api/gov", tags=["gov"])

API_KEY = os.getenv("INGEST_API_KEY", "dev-secret")

# --------------------------------------------------------------------
# Schema / Tabellen ohne Alembic (idempotent, keine Datenverluste)
# --------------------------------------------------------------------
def ensure_gov_schema() -> None:
    DDL = """
    CREATE SCHEMA IF NOT EXISTS "gov";

    CREATE TABLE IF NOT EXISTS "gov"."incidents" (
      id           SERIAL PRIMARY KEY,
      headline     TEXT NOT NULL,
      occurred_at  TIMESTAMPTZ NULL,
      seen         BOOLEAN NOT NULL DEFAULT FALSE,
      created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS "gov"."incident_articles" (
      id           SERIAL PRIMARY KEY,
      incident_id  INTEGER NOT NULL REFERENCES "gov"."incidents"(id) ON DELETE CASCADE,
      title        TEXT NOT NULL,
      source       TEXT NOT NULL,
      link         TEXT NOT NULL UNIQUE,
      published_at TIMESTAMPTZ NULL
    );

    CREATE INDEX IF NOT EXISTS idx_gov_incidents_seen_created
      ON "gov"."incidents"(seen, created_at DESC);

    CREATE INDEX IF NOT EXISTS idx_gov_incident_articles_incident
      ON "gov"."incident_articles"(incident_id);
    """
    with engine.connect() as conn:
        conn.execute(text(DDL))
        conn.commit()

def _require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

# --------------------------------------------------------------------
# Pydantic Schemas
# --------------------------------------------------------------------
class ArticleIn(BaseModel):
    title: str
    source: str
    link: str
    published_at: Optional[datetime] = None

class ArticleOut(ArticleIn):
    id: int

class IncidentCreate(BaseModel):
    headline: str
    occurred_at: Optional[datetime] = None
    articles: List[ArticleIn] = []

class IncidentOut(BaseModel):
    id: int
    headline: str
    occurred_at: Optional[datetime] = None
    seen: bool
    created_at: datetime
    articles_count: int

class IncidentDetailOut(BaseModel):
    id: int
    headline: str
    occurred_at: Optional[datetime] = None
    seen: bool
    created_at: datetime
    articles: List[ArticleOut]

# --------------------------------------------------------------------
# Endpunkte
# --------------------------------------------------------------------
@router.get("/incidents", response_model=List[IncidentOut])
def list_incidents(
    seen: Optional[bool] = Query(default=None),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Liste aller Incidents, optional gefiltert nach seen, inkl. articles_count.
    """
    with get_session() as s:
        base = """
            SELECT i.id, i.headline, i.occurred_at, i.seen, i.created_at,
                   COALESCE(a.cnt, 0) AS articles_count
            FROM "gov"."incidents" AS i
            LEFT JOIN (
              SELECT incident_id, COUNT(*) AS cnt
              FROM "gov"."incident_articles"
              GROUP BY incident_id
            ) AS a ON a.incident_id = i.id
        """
        cond = []
        if seen is not None:
            cond.append("i.seen = :seen")
        if cond:
            base += " WHERE " + " AND ".join(cond)
        base += " ORDER BY i.created_at DESC LIMIT :limit OFFSET :offset"

        rows = s.execute(
            text(base),
            {"seen": seen, "limit": limit, "offset": offset}
        ).mappings().all()

        return [
            IncidentOut(
                id=r["id"],
                headline=r["headline"],
                occurred_at=r["occurred_at"],
                seen=r["seen"],
                created_at=r["created_at"],
                articles_count=r["articles_count"],
            )
            for r in rows
        ]

@router.get("/incidents/{incident_id}", response_model=IncidentDetailOut)
def get_incident_detail(incident_id: int):
    with get_session() as s:
        head = s.execute(
            text('SELECT id, headline, occurred_at, seen, created_at FROM "gov"."incidents" WHERE id=:id'),
            {"id": incident_id}
        ).mappings().first()
        if not head:
            raise HTTPException(status_code=404, detail="Incident not found")

        arts = s.execute(
            text('''
                SELECT id, title, source, link, published_at
                FROM "gov"."incident_articles"
                WHERE incident_id=:id
                ORDER BY COALESCE(published_at, 'epoch'::timestamptz) DESC, id DESC
            '''),
            {"id": incident_id}
        ).mappings().all()

        return IncidentDetailOut(
            id=head["id"],
            headline=head["headline"],
            occurred_at=head["occurred_at"],
            seen=head["seen"],
            created_at=head["created_at"],
            articles=[
                ArticleOut(
                    id=a["id"],
                    title=a["title"],
                    source=a["source"],
                    link=a["link"],
                    published_at=a["published_at"],
                ) for a in arts
            ],
        )

@router.post("/incidents", response_model=IncidentDetailOut, status_code=201, dependencies=[Depends(_require_api_key)])
def create_incident(payload: IncidentCreate):
    """
    Legt einen Incident an und fügt (optionale) Artikel hinzu.
    Duplicate-Links werden via UNIQUE(link) ignoriert.
    """
    with get_session() as s:
        # Incident anlegen
        row = s.execute(
            text('''
                INSERT INTO "gov"."incidents"(headline, occurred_at)
                VALUES (:headline, :occurred_at)
                RETURNING id, headline, occurred_at, seen, created_at
            '''),
            {"headline": payload.headline, "occurred_at": payload.occurred_at}
        ).mappings().first()
        incident_id = row["id"]

        # Artikel einfügen (on conflict do nothing)
        created_articles: List[ArticleOut] = []
        for a in payload.articles:
            ins = text('''
                INSERT INTO "gov"."incident_articles"(incident_id, title, source, link, published_at)
                VALUES (:incident_id, :title, :source, :link, :published_at)
                ON CONFLICT (link) DO NOTHING
                RETURNING id, title, source, link, published_at
            ''')
            res = s.execute(ins, {
                "incident_id": incident_id,
                "title": a.title.strip(),
                "source": a.source.strip(),
                "link": a.link.strip(),
                "published_at": a.published_at,
            }).mappings().first()
            if res:
                created_articles.append(ArticleOut(**res))
        s.commit()

        return IncidentDetailOut(
            id=row["id"],
            headline=row["headline"],
            occurred_at=row["occurred_at"],
            seen=row["seen"],
            created_at=row["created_at"],
            articles=created_articles,
        )

@router.patch("/incidents/{incident_id}/seen", dependencies=[Depends(_require_api_key)])
def patch_incident_seen(incident_id: int, seen: bool):
    with get_session() as s:
        res = s.execute(
            text('UPDATE "gov"."incidents" SET seen=:seen WHERE id=:id'),
            {"seen": seen, "id": incident_id}
        )
        s.commit()
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Incident not found")
    return {"updated": int(res.rowcount)}

@router.delete("/incidents/{incident_id}", status_code=204, dependencies=[Depends(_require_api_key)])
def delete_incident(incident_id: int):
    with get_session() as s:
        res = s.execute(text('DELETE FROM "gov"."incidents" WHERE id=:id'), {"id": incident_id})
        s.commit()
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Incident not found")
    return

@router.delete("/incidents/{incident_id}/articles/{article_id}", status_code=204, dependencies=[Depends(_require_api_key)])
def delete_incident_article(incident_id: int, article_id: int):
    with get_session() as s:
        res = s.execute(
            text('DELETE FROM "gov"."incident_articles" WHERE id=:aid AND incident_id=:iid'),
            {"aid": article_id, "iid": incident_id}
        )
        s.commit()
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Article not found")
    return

@router.delete("/wipe", dependencies=[Depends(_require_api_key)])
def wipe_everything(confirm: str = Query(..., description='Muss "yes" sein')):
    """
    ⚠️ Vorsicht: löscht ALLE gov-Daten.
    """
    if confirm.lower() != "yes":
        raise HTTPException(status_code=400, detail='Pass "confirm=yes" to wipe.')
    with get_session() as s:
        s.execute(text('TRUNCATE TABLE "gov"."incident_articles" RESTART IDENTITY CASCADE;'))
        s.execute(text('TRUNCATE TABLE "gov"."incidents" RESTART IDENTITY CASCADE;'))
        s.commit()
    return {"status": "wiped"}
