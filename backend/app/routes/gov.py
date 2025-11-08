# backend/app/routes/gov.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timezone
import re
import os

from sqlalchemy import text, select
from sqlalchemy.orm import Session

from ..db import get_db_session, engine

router = APIRouter(prefix="/api/gov", tags=["govwatch"])

API_KEY = os.getenv("INGEST_API_KEY", "dev-secret")


# ---------- Helpers ----------
def _require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True


def ensure_gov_schema() -> None:
    """Idempotent: legt Schema + Tabellen + Indizes an (ohne Datenverlust)."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE SCHEMA IF NOT EXISTS gov;

            CREATE TABLE IF NOT EXISTS gov.incidents (
              id           SERIAL PRIMARY KEY,
              headline     TEXT NOT NULL,
              occurred_at  TIMESTAMPTZ NULL,
              seen         BOOLEAN NOT NULL DEFAULT FALSE,
              created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS gov.incident_articles (
              id           SERIAL PRIMARY KEY,
              incident_id  INTEGER NOT NULL REFERENCES gov.incidents(id) ON DELETE CASCADE,
              title        TEXT NOT NULL,
              source       TEXT NOT NULL,
              link         TEXT NOT NULL,
              published_at TIMESTAMPTZ NULL
            );

            CREATE INDEX IF NOT EXISTS idx_gov_incidents_seen_created
              ON gov.incidents (seen, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_gov_articles_incident
              ON gov.incident_articles (incident_id);
        """))
        # Constraint-Umstellung: unique (incident_id, link)
        conn.execute(text("""
            DO $$
            BEGIN
              IF EXISTS (
                SELECT 1 FROM pg_constraint c
                JOIN pg_namespace n ON n.oid = c.connamespace
                JOIN pg_class t ON t.oid = c.conrelid
                WHERE n.nspname='gov' AND t.relname='incident_articles'
                  AND c.conname='idx_gov_articles_link_unique'
              ) THEN
                ALTER TABLE gov.incident_articles
                  DROP CONSTRAINT idx_gov_articles_link_unique;
              END IF;
            END$$;

            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM pg_constraint c
                JOIN pg_namespace n ON n.oid = c.connamespace
                JOIN pg_class t ON t.oid = c.conrelid
                WHERE n.nspname='gov' AND t.relname='incident_articles'
                  AND c.conname='uq_gov_article_incident_link'
              ) THEN
                ALTER TABLE gov.incident_articles
                  ADD CONSTRAINT uq_gov_article_incident_link UNIQUE (incident_id, link);
              END IF;
            END$$;
        """))


def _parse_date_any(s: Optional[str]) -> Optional[datetime]:
    """Akzeptiert:
       - ISO (z. B. '2025-11-08T10:00:00Z' / '+01:00')
       - reines Datum 'YYYY-MM-DD'
       - deutsches Datum 'dd.mm.yyyy' oder mit Bindestrich 'dd-mm-yyyy'
       Ergebnis immer als aware-UTC (00:00Z bei Datum ohne Uhrzeit).
    """
    if not s or not isinstance(s, str) or not s.strip():
        return None
    s = s.strip()

    # dd.mm.yyyy / dd-mm-yyyy
    m = re.fullmatch(r"(\d{2})[.\-](\d{2})[.\-](\d{4})", s)
    if m:
        d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

    # YYYY-MM-DD
    m2 = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m2:
        d = date(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

    # ISO 8601
    try:
        s_iso = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _date_only(dt: Optional[datetime]) -> Optional[date]:
    return dt.date() if isinstance(dt, datetime) else None


# ---------- Schemas (leichtgewichtig) ----------
from pydantic import BaseModel, Field, AnyUrl
from typing import List as TList, Optional as TOpt

class ArticleIn(BaseModel):
    title: str
    source: str
    link: str
    published_at: TOpt[str] = None

class IncidentCreateIn(BaseModel):
    headline: str
    occurred_at: TOpt[str] = None
    articles: TList[ArticleIn] = Field(default_factory=list)

class IncidentOut(BaseModel):
    id: int
    headline: str
    occurred_at: TOpt[str] = None
    seen: bool
    created_at: str
    articles_count: int

class IncidentDetailOut(BaseModel):
    id: int
    headline: str
    occurred_at: TOpt[str] = None
    seen: bool
    created_at: str
    articles: TList[Dict[str, Any]]


# ---------- Routes ----------
@router.get("/incidents", response_model=List[IncidentOut])
def list_incidents(
    seen: Optional[bool] = Query(None),
    limit: int = Query(500, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_db_session),
):
    """Chronologisch (occurred_at ASC, NULLS LAST)"""
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    where = ""
    if seen is not None:
        where = "WHERE i.seen = :seen"
        params["seen"] = seen

    rows = session.execute(text(f"""
        SELECT
          i.id, i.headline, i.occurred_at, i.seen, i.created_at,
          COUNT(a.id)::int AS articles_count
        FROM gov.incidents i
        LEFT JOIN gov.incident_articles a ON a.incident_id = i.id
        {where}
        GROUP BY i.id
        ORDER BY i.occurred_at ASC NULLS LAST, i.created_at ASC
        LIMIT :limit OFFSET :offset
    """), params).mappings().all()

    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "headline": r["headline"],
            "occurred_at": r["occurred_at"].isoformat() if r["occurred_at"] else None,
            "seen": r["seen"],
            "created_at": r["created_at"].isoformat(),
            "articles_count": r["articles_count"],
        })
    return out


@router.get("/incidents/{incident_id}", response_model=IncidentDetailOut)
def get_incident_detail(incident_id: int, session: Session = Depends(get_db_session)):
    row = session.execute(text("""
        SELECT id, headline, occurred_at, seen, created_at
        FROM gov.incidents WHERE id = :id
    """), {"id": incident_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="incident not found")

    arts = session.execute(text("""
        SELECT id, title, source, link, published_at
        FROM gov.incident_articles
        WHERE incident_id = :id
        ORDER BY published_at ASC NULLS LAST, id ASC
    """), {"id": incident_id}).mappings().all()

    return {
        "id": row["id"],
        "headline": row["headline"],
        "occurred_at": row["occurred_at"].isoformat() if row["occurred_at"] else None,
        "seen": row["seen"],
        "created_at": row["created_at"].isoformat(),
        "articles": [
            {
                "id": a["id"],
                "title": a["title"],
                "source": a["source"],
                "link": a["link"],
                "published_at": a["published_at"].isoformat() if a["published_at"] else None,
            } for a in arts
        ]
    }


@router.post("/incidents", response_model=IncidentDetailOut, status_code=201, dependencies=[Depends(_require_api_key)])
def create_incident(payload: IncidentCreateIn, session: Session = Depends(get_db_session)):
    # occurred_at parsen (Datum ohne Uhrzeit erlaubt)
    occurred_dt = _parse_date_any(payload.occurred_at)

    # **Neue Regel**: Wenn occurred_at gesetzt ist → Prüfe, ob an diesem Tag schon ein Incident existiert.
    if occurred_dt is not None:
        exists = session.execute(text("""
            SELECT id FROM gov.incidents
            WHERE occurred_at::date = :d::date
            LIMIT 1
        """), {"d": occurred_dt.date().isoformat()}).mappings().first()
        if exists:
            raise HTTPException(status_code=409, detail="Es existiert bereits eine Panne an diesem Tag.")

    # Incident anlegen
    row = session.execute(text("""
        INSERT INTO gov.incidents (headline, occurred_at)
        VALUES (:h, :o)
        RETURNING id, headline, occurred_at, seen, created_at
    """), {"h": payload.headline.strip(), "o": occurred_dt}).mappings().first()
    inc_id = row["id"]

    # Artikel anhängen (Duplikate nur innerhalb desselben Incidents verhindern)
    for a in (payload.articles or []):
        try:
            session.execute(text("""
                INSERT INTO gov.incident_articles (incident_id, title, source, link, published_at)
                VALUES (:iid, :t, :s, :l, :p)
                ON CONFLICT ON CONSTRAINT uq_gov_article_incident_link DO NOTHING
            """), {
                "iid": inc_id,
                "t": a.title.strip(),
                "s": a.source.strip(),
                "l": a.link.strip(),
                "p": _parse_date_any(a.published_at)
            })
        except Exception:
            # defensive, sollte durch ON CONFLICT kaum passieren
            pass

    session.commit()
    return get_incident_detail(inc_id, session)


@router.post("/incidents/{incident_id}/articles", status_code=201, dependencies=[Depends(_require_api_key)])
def add_article(incident_id: int, a: ArticleIn, session: Session = Depends(get_db_session)):
    # existiert Incident?
    chk = session.execute(text("SELECT 1 FROM gov.incidents WHERE id=:id"), {"id": incident_id}).first()
    if not chk:
        raise HTTPException(status_code=404, detail="incident not found")

    # Einfügen (nur innerhalb desselben Incident einzigartige Links)
    res = session.execute(text("""
        INSERT INTO gov.incident_articles (incident_id, title, source, link, published_at)
        VALUES (:iid, :t, :s, :l, :p)
        ON CONFLICT ON CONSTRAINT uq_gov_article_incident_link DO NOTHING
        RETURNING id
    """), {
        "iid": incident_id,
        "t": a.title.strip(),
        "s": a.source.strip(),
        "l": a.link.strip(),
        "p": _parse_date_any(a.published_at)
    }).first()

    session.commit()
    if res is None:
        # bereits vorhanden – als 409 signalisieren
        raise HTTPException(status_code=409, detail="article already exists in this incident")
    return {"id": res[0]}


@router.patch("/incidents/{incident_id}/seen")
def patch_incident_seen(
    incident_id: int,
    seen: bool = Query(...),
    _=Depends(_require_api_key),
    session: Session = Depends(get_db_session),
):
    res = session.execute(text("""
        UPDATE gov.incidents SET seen=:seen WHERE id=:id
    """), {"seen": seen, "id": incident_id})
    session.commit()
    return {"updated": res.rowcount}


@router.delete("/incidents/{incident_id}", status_code=204, dependencies=[Depends(_require_api_key)])
def delete_incident(incident_id: int, session: Session = Depends(get_db_session)):
    session.execute(text("DELETE FROM gov.incidents WHERE id=:id"), {"id": incident_id})
    session.commit()
    return


@router.delete("/incidents/{incident_id}/articles/{article_id}", status_code=204, dependencies=[Depends(_require_api_key)])
def delete_article(incident_id: int, article_id: int, session: Session = Depends(get_db_session)):
    session.execute(text("""
        DELETE FROM gov.incident_articles
        WHERE id=:aid AND incident_id=:iid
    """), {"aid": article_id, "iid": incident_id})
    session.commit()
    return


@router.delete("/wipe", dependencies=[Depends(_require_api_key)])
def wipe_all(confirm: str = Query(...)):
    if confirm.lower() != "yes":
        raise HTTPException(status_code=400, detail="confirmation required")
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE gov.incident_articles RESTART IDENTITY CASCADE;"))
        conn.execute(text("TRUNCATE TABLE gov.incidents RESTART IDENTITY CASCADE;"))
    return {"status": "wiped"}
