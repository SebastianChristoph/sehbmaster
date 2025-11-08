# backend/app/routes/gov.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import os

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db import get_session

router = APIRouter(prefix="/api/gov", tags=["gov"])

API_KEY = os.getenv("INGEST_API_KEY", "dev-secret")
SCHEMA = 'gov'  # eigenes Schema für diesen Scraper

# -----------------------------------------------------------------------------
# Schema / Tabellen idempotent erstellen (ohne Alembic, Daten bleiben erhalten)
# -----------------------------------------------------------------------------
def ensure_gov_schema() -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS "{SCHEMA}";

    CREATE TABLE IF NOT EXISTS "{SCHEMA}"."incidents" (
        id           SERIAL PRIMARY KEY,
        headline     TEXT NOT NULL,
        occurred_at  TIMESTAMPTZ,
        seen         BOOLEAN NOT NULL DEFAULT FALSE,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS "{SCHEMA}"."incident_articles" (
        id           SERIAL PRIMARY KEY,
        incident_id  INTEGER NOT NULL REFERENCES "{SCHEMA}"."incidents"(id) ON DELETE CASCADE,
        title        TEXT NOT NULL,
        source       TEXT NOT NULL,
        link         TEXT NOT NULL UNIQUE,
        published_at TIMESTAMPTZ
    );

    CREATE INDEX IF NOT EXISTS idx_gov_incidents_seen_created
      ON "{SCHEMA}"."incidents"(seen, created_at DESC);

    CREATE INDEX IF NOT EXISTS idx_gov_articles_incident
      ON "{SCHEMA}"."incident_articles"(incident_id);
    """
    with get_session() as s:
        s.execute(text(ddl))
        s.commit()


# -----------------------------------------------------------------------------
# Security helper
# -----------------------------------------------------------------------------
def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True


# -----------------------------------------------------------------------------
# DTOs (leichtgewichtig, ohne Pydantic-Klassen)
# -----------------------------------------------------------------------------
def _row_to_incident(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "headline": row["headline"],
        "occurred_at": (row["occurred_at"].isoformat() if row["occurred_at"] else None),
        "seen": row["seen"],
        "created_at": (row["created_at"].isoformat() if row["created_at"] else None),
    }

def _row_to_article(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "source": row["source"],
        "link": row["link"],
        "published_at": (row["published_at"].isoformat() if row["published_at"] else None),
    }


# -----------------------------------------------------------------------------
# GET /incidents?seen=...
#   Liste mit Article-Count (für kompakte Übersicht)
# -----------------------------------------------------------------------------
@router.get("/incidents")
def list_incidents(
    seen: Optional[bool] = Query(None),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    sql = f"""
    SELECT i.id, i.headline, i.occurred_at, i.seen, i.created_at,
           COALESCE(a.cnt, 0) AS articles_count
    FROM "{SCHEMA}"."incidents" i
    LEFT JOIN (
        SELECT incident_id, COUNT(*) AS cnt
        FROM "{SCHEMA}"."incident_articles"
        GROUP BY incident_id
    ) a ON a.incident_id = i.id
    {{where}}
    ORDER BY i.created_at DESC
    LIMIT :limit OFFSET :offset
    """
    where = ""
    params = {"limit": limit, "offset": offset}
    if seen is not None:
        where = "WHERE i.seen = :seen"
        params["seen"] = seen

    with get_session() as s:
        rows = s.execute(text(sql.replace("{where}", where)), params).mappings().all()
        return [
            {
                **_row_to_incident(r),
                "articles_count": r["articles_count"],
            }
            for r in rows
        ]


# -----------------------------------------------------------------------------
# GET /incidents/{id}  (Detail inkl. Artikel)
# -----------------------------------------------------------------------------
@router.get("/incidents/{incident_id}")
def get_incident_detail(incident_id: int):
    with get_session() as s:
        row = s.execute(
            text(f'SELECT * FROM "{SCHEMA}"."incidents" WHERE id = :id'),
            {"id": incident_id},
        ).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="not found")

        arts = s.execute(
            text(f'''
                SELECT * FROM "{SCHEMA}"."incident_articles"
                WHERE incident_id = :id
                ORDER BY COALESCE(published_at, 'epoch'::timestamptz) DESC, id DESC
            '''),
            {"id": incident_id},
        ).mappings().all()

        return {
            **_row_to_incident(row),
            "articles": [_row_to_article(a) for a in arts],
        }


# -----------------------------------------------------------------------------
# POST /incidents    (mit Duplicate-Day-Check)
#   Body: { headline, occurred_at?, articles?[] }
#   - Wenn occurred_at gesetzt ist, prüfe: existiert an diesem Kalendertag bereits ein Incident?
#     -> ja: 409 + Meldung
# -----------------------------------------------------------------------------
@router.post("/incidents", dependencies=[Depends(require_api_key)], status_code=201)
def create_incident(payload: Dict[str, Any]):
    headline = (payload.get("headline") or "").strip()
    occurred_at = payload.get("occurred_at")
    articles = payload.get("articles") or []

    if not headline:
        raise HTTPException(status_code=400, detail="headline required")

    with get_session() as s:
        # Duplicate-Day-Check (UTC-Tag; falls du Berlin-Tag willst, siehe Kommentar unten)
        if occurred_at:
            # Hinweis: erfolgt in UTC. Für Europe/Berlin stattdessen:
            # WHERE (occurred_at AT TIME ZONE 'Europe/Berlin')::date = ( :occurred_at::timestamptz AT TIME ZONE 'Europe/Berlin')::date
            exists = s.execute(
                text(f'''
                    SELECT 1
                    FROM "{SCHEMA}"."incidents"
                    WHERE occurred_at::date = (:ts)::date
                    LIMIT 1
                '''),
                {"ts": occurred_at},
            ).first()
            if exists:
                raise HTTPException(status_code=409, detail="Es existiert bereits eine Panne an diesem Tag")

        ins = s.execute(
            text(f'''
                INSERT INTO "{SCHEMA}"."incidents"(headline, occurred_at)
                VALUES (:headline, :occurred_at)
                RETURNING *
            '''),
            {"headline": headline, "occurred_at": occurred_at},
        ).mappings().first()
        incident_id = ins["id"]

        # optionale Artikel direkt anlegen
        created_articles: List[Dict[str, Any]] = []
        for a in articles:
            title = (a.get("title") or "").strip()
            source = (a.get("source") or "").strip()
            link = (a.get("link") or "").strip()
            pub = a.get("published_at")

            if not (title and source and link):
                continue

            try:
                row = s.execute(
                    text(f'''
                        INSERT INTO "{SCHEMA}"."incident_articles"
                        (incident_id, title, source, link, published_at)
                        VALUES (:iid, :title, :source, :link, :pub)
                        RETURNING *
                    '''),
                    {"iid": incident_id, "title": title, "source": source, "link": link, "pub": pub},
                ).mappings().first()
                created_articles.append(_row_to_article(row))
            except Exception:
                # z.B. Unique-Verstoß auf link -> ignorieren
                pass

        s.commit()

        return {
            **_row_to_incident(ins),
            "articles": created_articles,
        }


# -----------------------------------------------------------------------------
# POST /incidents/{id}/articles
# -----------------------------------------------------------------------------
@router.post("/incidents/{incident_id}/articles", dependencies=[Depends(require_api_key)], status_code=201)
def add_article(incident_id: int, payload: Dict[str, Any]):
    title = (payload.get("title") or "").strip()
    source = (payload.get("source") or "").strip()
    link = (payload.get("link") or "").strip()
    pub = payload.get("published_at")

    if not (title and source and link):
        raise HTTPException(status_code=400, detail="title, source, link required")

    with get_session() as s:
        inc = s.execute(
            text(f'SELECT 1 FROM "{SCHEMA}"."incidents" WHERE id=:id'),
            {"id": incident_id},
        ).first()
        if not inc:
            raise HTTPException(status_code=404, detail="incident not found")

        try:
            row = s.execute(
                text(f'''
                    INSERT INTO "{SCHEMA}"."incident_articles"
                    (incident_id, title, source, link, published_at)
                    VALUES (:iid, :title, :source, :link, :pub)
                    RETURNING *
                '''),
                {"iid": incident_id, "title": title, "source": source, "link": link, "pub": pub},
            ).mappings().first()
            s.commit()
            return _row_to_article(row)
        except Exception as e:
            # Prüfe Unique-Verletzung (link)
            msg = str(e).lower()
            if "unique" in msg and "link" in msg:
                raise HTTPException(status_code=409, detail="duplicate link")
            raise


# -----------------------------------------------------------------------------
# PATCH /incidents/{id}/seen?seen=true|false
# -----------------------------------------------------------------------------
@router.patch("/incidents/{incident_id}/seen")
def set_seen(
    incident_id: int,
    seen: bool = Query(...),
    _=Depends(require_api_key),
):
    with get_session() as s:
        res = s.execute(
            text(f'UPDATE "{SCHEMA}"."incidents" SET seen=:seen WHERE id=:id'),
            {"seen": seen, "id": incident_id},
        )
        s.commit()
        return {"updated": res.rowcount}


# -----------------------------------------------------------------------------
# DELETE /incidents/{id}
# -----------------------------------------------------------------------------
@router.delete("/incidents/{incident_id}", status_code=204, dependencies=[Depends(require_api_key)])
def delete_incident(incident_id: int):
    with get_session() as s:
        s.execute(text(f'DELETE FROM "{SCHEMA}"."incidents" WHERE id=:id'), {"id": incident_id})
        s.commit()
    return


# -----------------------------------------------------------------------------
# DELETE /incidents/{id}/articles/{article_id}
# -----------------------------------------------------------------------------
@router.delete("/incidents/{incident_id}/articles/{article_id}", status_code=204, dependencies=[Depends(require_api_key)])
def delete_article(incident_id: int, article_id: int):
    with get_session() as s:
        s.execute(
            text(f'''
                DELETE FROM "{SCHEMA}"."incident_articles"
                WHERE id=:aid AND incident_id=:iid
            '''),
            {"aid": article_id, "iid": incident_id},
        )
        s.commit()
    return


# -----------------------------------------------------------------------------
# ⚠️ DELETE /wipe?confirm=yes
# -----------------------------------------------------------------------------
@router.delete("/wipe")
def wipe_all(confirm: str = Query(...), _=Depends(require_api_key)):
    if confirm.lower() != "yes":
        raise HTTPException(status_code=400, detail="set confirm=yes to wipe")
    with get_session() as s:
        s.execute(text(f'TRUNCATE TABLE "{SCHEMA}"."incident_articles" RESTART IDENTITY CASCADE'))
        s.execute(text(f'TRUNCATE TABLE "{SCHEMA}"."incidents" RESTART IDENTITY CASCADE'))
        s.commit()
    return {"status": "wiped"}
