# backend/app/routes/gov.py
from __future__ import annotations

from datetime import datetime, date
from typing import List, Optional

import os
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column, Integer, Text, DateTime, Boolean, ForeignKey,
    select, func, delete, asc, desc
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, declarative_base, relationship

from ..db import get_db_session

router = APIRouter(prefix="/api/gov", tags=["gov"])

API_KEY = os.getenv("INGEST_API_KEY", "dev-secret")
Base = declarative_base()

# ==========================
# ORM-Modelle (Schema: gov)
# ==========================

class GovIncident(Base):
    __tablename__ = "incidents"
    __table_args__ = {"schema": "gov"}

    id = Column(Integer, primary_key=True)
    headline = Column(Text, nullable=False)
    occurred_at = Column(DateTime(timezone=True), nullable=True)
    seen = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    articles = relationship("GovIncidentArticle", back_populates="incident", cascade="all, delete-orphan")


class GovIncidentArticle(Base):
    __tablename__ = "incident_articles"
    __table_args__ = {"schema": "gov"}

    id = Column(Integer, primary_key=True)
    incident_id = Column(Integer, ForeignKey("gov.incidents.id", ondelete="CASCADE"), nullable=False)
    title = Column(Text, nullable=False)
    source = Column(Text, nullable=False)
    link = Column(Text, nullable=False)  # DB-Constraint: UNIQUE(incident_id, link)
    published_at = Column(DateTime(timezone=True), nullable=True)

    incident = relationship("GovIncident", back_populates="articles")


# ==========================
# Schemas
# ==========================

class GovArticleIn(BaseModel):
    title: str
    source: str
    link: str
    published_at: Optional[datetime] = None


class GovArticleOut(GovArticleIn):
    id: int


class GovIncidentCreate(BaseModel):
    headline: str = Field(..., min_length=1)
    occurred_at: Optional[datetime] = None
    articles: List[GovArticleIn] = Field(default_factory=list)


class GovIncidentOut(BaseModel):
    id: int
    headline: str
    occurred_at: Optional[datetime]
    seen: bool
    created_at: datetime
    articles: List[GovArticleOut]


class GovIncidentListItem(BaseModel):
    id: int
    headline: str
    occurred_at: Optional[datetime]
    seen: bool
    created_at: datetime
    articles_count: int


# ==========================
# Helpers
# ==========================

def _require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True


# ==========================
# Endpunkte
# ==========================

@router.get("/incidents", response_model=List[GovIncidentListItem])
def list_incidents(
    seen: Optional[bool] = Query(None),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_db_session),
):
    q = (
        select(
            GovIncident.id,
            GovIncident.headline,
            GovIncident.occurred_at,
            GovIncident.seen,
            GovIncident.created_at,
            func.count(GovIncidentArticle.id).label("articles_count"),
        )
        .join(GovIncident.articles, isouter=True)
        .group_by(GovIncident.id)
    )

    if seen is not None:
        q = q.where(GovIncident.seen == seen)

    # Chronologisch: occurred_at ASC (NULLS LAST), dann created_at ASC
    q = q.order_by(asc(GovIncident.occurred_at).nulls_last(), asc(GovIncident.created_at))
    q = q.offset(offset).limit(limit)

    rows = session.execute(q).all()
    return [
        GovIncidentListItem(
            id=r.id,
            headline=r.headline,
            occurred_at=r.occurred_at,
            seen=r.seen,
            created_at=r.created_at,
            articles_count=r.articles_count or 0,
        )
        for r in rows
    ]


@router.get("/incidents/{incident_id}", response_model=GovIncidentOut)
def get_incident_detail(incident_id: int, session: Session = Depends(get_db_session)):
    inc = session.get(GovIncident, incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail="incident not found")
    arts = (
        session.execute(
            select(GovIncidentArticle)
            .where(GovIncidentArticle.incident_id == incident_id)
            .order_by(desc(GovIncidentArticle.published_at.nulls_last()), desc(GovIncidentArticle.id))
        )
        .scalars()
        .all()
    )
    return GovIncidentOut(
        id=inc.id,
        headline=inc.headline,
        occurred_at=inc.occurred_at,
        seen=inc.seen,
        created_at=inc.created_at,
        articles=[
            GovArticleOut(
                id=a.id,
                title=a.title,
                source=a.source,
                link=a.link,
                published_at=a.published_at,
            )
            for a in arts
        ],
    )


@router.post("/incidents", response_model=GovIncidentOut, status_code=201)
def create_incident(
    payload: GovIncidentCreate,
    manual: bool = Query(False, description="Wenn true: prüfe auf vorhandenen Vorfall mit gleichem Kalendertag"),
    _=Depends(_require_api_key),
    session: Session = Depends(get_db_session),
):
    # Optionaler Check: Manual → Datum darf nicht bereits existieren (Kalendertag)
    if manual and payload.occurred_at is not None:
        day: date = payload.occurred_at.date()
        exists_id = (
            session.execute(
                select(GovIncident.id).where(func.date(GovIncident.occurred_at) == day).limit(1)
            )
            .scalars()
            .first()
        )
        if exists_id is not None:
            raise HTTPException(
                status_code=409,
                detail="Es existiert bereits ein Vorfall an diesem Tag",
            )

    inc = GovIncident(
        headline=payload.headline.strip(),
        occurred_at=payload.occurred_at,
        seen=False,
    )
    session.add(inc)
    session.flush()  # inc.id verfügbar

    for a in payload.articles:
        row = GovIncidentArticle(
            incident_id=inc.id,
            title=a.title.strip(),
            source=a.source.strip(),
            link=a.link.strip(),
            published_at=a.published_at,
        )
        session.add(row)
        try:
            session.flush()
        except IntegrityError:
            # UNIQUE(incident_id, link) → innerhalb desselben Incidents doppelte Links ignorieren
            session.rollback()

    session.commit()

    arts = (
        session.execute(
            select(GovIncidentArticle).where(GovIncidentArticle.incident_id == inc.id).order_by(GovIncidentArticle.id.asc())
        )
        .scalars()
        .all()
    )
    return GovIncidentOut(
        id=inc.id,
        headline=inc.headline,
        occurred_at=inc.occurred_at,
        seen=inc.seen,
        created_at=inc.created_at,
        articles=[
            GovArticleOut(
                id=r.id,
                title=r.title,
                source=r.source,
                link=r.link,
                published_at=r.published_at,
            )
            for r in arts
        ],
    )


@router.patch("/incidents/{incident_id}/seen")
def set_seen(
    incident_id: int,
    seen: bool = Query(...),
    _=Depends(_require_api_key),
    session: Session = Depends(get_db_session),
):
    inc = session.get(GovIncident, incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail="incident not found")
    inc.seen = bool(seen)
    session.commit()
    return {"updated": 1}


@router.post("/incidents/{incident_id}/articles", response_model=GovArticleOut, status_code=201)
def add_article(
    incident_id: int,
    payload: GovArticleIn,
    _=Depends(_require_api_key),
    session: Session = Depends(get_db_session),
):
    inc = session.get(GovIncident, incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail="incident not found")

    row = GovIncidentArticle(
        incident_id=incident_id,
        title=payload.title.strip(),
        source=payload.source.strip(),
        link=payload.link.strip(),
        published_at=payload.published_at,
    )
    session.add(row)
    try:
        session.commit()
        session.refresh(row)
    except IntegrityError:
        session.rollback()
        # Duplikat innerhalb desselben Incidents
        raise HTTPException(status_code=409, detail="article already exists in this incident")

    return GovArticleOut(
        id=row.id,
        title=row.title,
        source=row.source,
        link=row.link,
        published_at=row.published_at,
    )


@router.delete("/incidents/{incident_id}/articles/{article_id}", status_code=204)
def delete_article(
    incident_id: int,
    article_id: int,
    _=Depends(_require_api_key),
    session: Session = Depends(get_db_session),
):
    row = session.get(GovIncidentArticle, article_id)
    if not row or row.incident_id != incident_id:
        raise HTTPException(status_code=404, detail="article not found")
    session.delete(row)
    session.commit()
    return


@router.delete("/incidents/{incident_id}", status_code=204)
def delete_incident(
    incident_id: int,
    _=Depends(_require_api_key),
    session: Session = Depends(get_db_session),
):
    row = session.get(GovIncident, incident_id)
    if not row:
        raise HTTPException(status_code=404, detail="incident not found")
    session.delete(row)
    session.commit()
    return


@router.delete("/wipe")
def wipe_all(
    confirm: str = Query(..., description='Gib "yes" ein, um wirklich alles zu löschen.'),
    _=Depends(_require_api_key),
    session: Session = Depends(get_db_session),
):
    if confirm.lower() != "yes":
        raise HTTPException(status_code=400, detail="confirmation required")
    session.execute(delete(GovIncidentArticle))
    session.execute(delete(GovIncident))
    session.commit()
    return {"status": "wiped"}
