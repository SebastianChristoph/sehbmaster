# backend/app/routes/bild.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, field_validator
from sqlalchemy import func, select, delete, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional
import os

from ..models import BildWatch, BildWatchMetrics, BildLog, BildCorrection
from ..schemas import (
    BildWatchIn, BildWatchOut,
    BildWatchMetricsIn, BildWatchMetricsOut,
    BildLogIn, BildLogOut, BildCorrectionIn, BildCorrectionOut
)
from ..services.bild_charts import (
    compute_category_counts,
    compute_hourly_charts,
    compute_daily_conversions,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from ..db import get_session, get_db_session 

router = APIRouter(prefix="/api/bild", tags=["bild"])

API_KEY = os.getenv("INGEST_API_KEY", "dev-secret")
TZ = ZoneInfo("Europe/Berlin")

def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True

@router.get("/articles", response_model=List[BildWatchOut])
def get_all_articles(limit: int = 500, offset: int = 0):
    with get_session() as s:
        q = (
            select(BildWatch)
            .order_by(BildWatch.published.desc())
            .offset(offset)
            .limit(limit)
        )
        rows = s.execute(q).scalars().all()
        return [
            BildWatchOut(
                id=r.id, title=r.title, url=r.url, category=r.category,
                is_premium=r.is_premium, converted=r.converted,
                published=r.published, converted_time=r.converted_time,
                converted_duration_hours=r.converted_duration_hours,
            )
            for r in rows
        ]

@router.post("/articles", response_model=BildWatchOut, status_code=201)
def add_article(payload: BildWatchIn, _=Depends(require_api_key)):
    with get_session() as s:
        exists = s.get(BildWatch, payload.id)
        if exists:
            raise HTTPException(status_code=409, detail="Article with this id already exists")
        obj = BildWatch(**payload.model_dump())
        s.add(obj)
        s.commit()
        s.refresh(obj)
        return BildWatchOut(
            id=obj.id, title=obj.title, url=obj.url, category=obj.category,
            is_premium=obj.is_premium, converted=obj.converted,
            published=obj.published, converted_time=obj.converted_time,
            converted_duration_hours=obj.converted_duration_hours,
        )

# ---- PATCH-Schema: jetzt mit title/url/category ----
class BildWatchPatch(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    category: Optional[str] = None

    is_premium: Optional[bool] = None
    converted: Optional[bool] = None
    converted_time: Optional[datetime] = None
    converted_duration_hours: Optional[float] = None

    @field_validator("title")
    @classmethod
    def _norm_title(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if isinstance(v, str) else v

    @field_validator("url")
    @classmethod
    def _norm_url(cls, v: Optional[str]) -> Optional[str]:
        if not isinstance(v, str):
            return v
        v = v.strip()
        if len(v) > 1 and v.endswith("/"):
            v = v[:-1]
        return v

    @field_validator("category")
    @classmethod
    def _norm_category(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if isinstance(v, str) else v

@router.patch("/articles/{article_id}", response_model=BildWatchOut)
def update_article(article_id: str, patch: BildWatchPatch, _=Depends(require_api_key)):
    updates = patch.model_dump(exclude_unset=True)
    with get_session() as s:
        obj = s.get(BildWatch, article_id)
        if not obj:
            raise HTTPException(status_code=404, detail="Article not found")

        if not updates:
            # idempotent: einfach aktuellen Zustand zurückgeben
            return BildWatchOut(
                id=obj.id, title=obj.title, url=obj.url, category=obj.category,
                is_premium=obj.is_premium, converted=obj.converted,
                published=obj.published, converted_time=obj.converted_time,
                converted_duration_hours=obj.converted_duration_hours,
            )

        # Nur bei tatsächlicher Änderung committen
        changed = False
        for k, v in updates.items():
            if getattr(obj, k) != v:
                setattr(obj, k, v)
                changed = True

        if changed:
            s.commit()
            s.refresh(obj)

        return BildWatchOut(
            id=obj.id, title=obj.title, url=obj.url, category=obj.category,
            is_premium=obj.is_premium, converted=obj.converted,
            published=obj.published, converted_time=obj.converted_time,
            converted_duration_hours=obj.converted_duration_hours,
        )

@router.delete("/articles", status_code=204, dependencies=[Depends(require_api_key)])
def delete_all_articles_and_metrics():
    with get_session() as s:
        s.execute(delete(BildWatchMetrics))
        s.execute(delete(BildWatch))
        s.commit()
    return

# ---- Charts und Logs bleiben unverändert ----
@router.get("/articles/category_counts", response_model=dict)
def get_category_counts_legacy():
    with get_session() as s:
        q = (
            s.query(BildWatch.category, func.count(BildWatch.id))
            .group_by(BildWatch.category)
            .all()
        )
        result: Dict[str, int] = {}
        for cat, count in q:
            label = cat if cat is not None else "Unbekannt"
            result[label] = count
        return result

@router.get("/charts/category_counts", response_model=Dict[str, int])
def chart_category_counts(premium_only: bool = Query(False, description="Nur Premium-Artikel zählen")):
    with get_session() as s:
        return compute_category_counts(s, premium_only=premium_only)

class HourlyPoint(BaseModel):
    hour: int
    Premium: float
    Nicht_Premium: float

class HourlyCharts(BaseModel):
    snapshot_avg: List[HourlyPoint]
    new_avg: List[HourlyPoint]

@router.get("/charts/hourly", response_model=HourlyCharts)
def chart_hourly(
    days: int = Query(60, ge=1, le=365),
    limit: int = Query(20000, ge=1, le=200000),
):
    time_to = datetime.now(timezone.utc)
    time_from = time_to - timedelta(days=days)
    with get_session() as s:
        return compute_hourly_charts(s, time_from=time_from, time_to=time_to, limit=limit)

class DailyCount(BaseModel):
    day: str
    count: int

@router.get("/charts/daily_conversions", response_model=List[DailyCount])
def chart_daily_conversions(
    days: int = Query(60, ge=1, le=365),
    limit: int = Query(200000, ge=1, le=2000000),
):
    time_to = datetime.now(timezone.utc)
    time_from = time_to - timedelta(days=days)
    with get_session() as s:
        return compute_daily_conversions(s, time_from=time_from, time_to=time_to, limit=limit)

@router.post("/metrics", response_model=BildWatchMetricsOut, status_code=201)
def upsert_metrics(payload: BildWatchMetricsIn, _=Depends(require_api_key)):
    table = BildWatchMetrics.__table__
    with get_session() as s:
        stmt = (
            pg_insert(table)
            .values(
                ts_hour=payload.ts_hour,
                snapshot_total=payload.snapshot_total,
                snapshot_premium=payload.snapshot_premium,
                snapshot_premium_pct=payload.snapshot_premium_pct,
                new_count=payload.new_count,
                new_premium_count=payload.new_premium_count,
            )
            .on_conflict_do_update(
                index_elements=[table.c.ts_hour],
                set_={
                    "snapshot_total": payload.snapshot_total,
                    "snapshot_premium": payload.snapshot_premium,
                    "snapshot_premium_pct": payload.snapshot_premium_pct,
                    "new_count": table.c.new_count + payload.new_count,
                    "new_premium_count": table.c.new_premium_count + payload.new_premium_count,
                },
            )
            .returning(*table.c)
        )
        row = s.execute(stmt).mappings().first()
        s.commit()
        return BildWatchMetricsOut(**row)  # type: ignore[arg-type]

@router.get("/metrics", response_model=List[BildWatchMetricsOut])
def get_metrics(
    time_from: Optional[datetime] = None,
    time_to: Optional[datetime] = None,
    limit: int = 1000,
):
    with get_session() as s:
        q = select(BildWatchMetrics).order_by(BildWatchMetrics.ts_hour.asc())
        if time_from:
            q = q.where(BildWatchMetrics.ts_hour >= time_from)
        if time_to:
            q = q.where(BildWatchMetrics.ts_hour < time_to)
        q = q.limit(limit)
        rows = s.execute(q).scalars().all()
        return [
            BildWatchMetricsOut(
                id=r.id,
                ts_hour=r.ts_hour,
                snapshot_total=r.snapshot_total,
                snapshot_premium=r.snapshot_premium,
                snapshot_premium_pct=r.snapshot_premium_pct,
                new_count=r.new_count,
                new_premium_count=r.new_premium_count,
                created_at=r.created_at,
            )
            for r in rows
        ]

@router.post("/logs", response_model=BildLogOut, status_code=201)
def add_log(payload: BildLogIn, _=Depends(require_api_key)):
    ts = payload.timestamp or datetime.now(timezone.utc)
    with get_session() as s:
        obj = BildLog(timestamp=ts, message=payload.message)
        s.add(obj)
        s.commit()
        s.refresh(obj)
        return BildLogOut(id=obj.id, timestamp=obj.timestamp, message=obj.message)

@router.get("/logs", response_model=List[BildLogOut])
def list_logs(limit: int = 1000, offset: int = 0, asc: bool = False):
    with get_session() as s:
        q = select(BildLog)
        q = q.order_by(BildLog.timestamp.asc() if asc else BildLog.timestamp.desc())
        q = q.limit(limit).offset(offset)
        rows = s.execute(q).scalars().all()
        return [BildLogOut(id=r.id, timestamp=r.timestamp, message=r.message) for r in rows]

@router.delete("/logs", status_code=204)
def delete_all_logs(_=Depends(require_api_key)):
    with get_session() as s:
        s.execute(text('TRUNCATE TABLE "bild"."log";'))
        s.commit()
    return



@router.get("/corrections", response_model=list[BildCorrectionOut])
def list_corrections(session: Session = Depends(get_db_session)):
    rows = (
        session.execute(
            select(BildCorrection).order_by(BildCorrection.published.desc())
        )
        .scalars()
        .all()
    )
    return [
        BildCorrectionOut(
            id=r.id,
            title=r.title,
            published=r.published,
            source_url=r.source_url,
            article_url=r.article_url,
            created_at=r.created_at,   # <- wichtig
        )
        for r in rows
    ]


@router.post("/corrections", response_model=BildCorrectionOut, status_code=201)
def create_correction(payload: BildCorrectionIn, session: Session = Depends(get_db_session)):
    row = BildCorrection(
        id=payload.id,
        title=payload.title,
        published=payload.published,
        source_url=payload.source_url,
        article_url=payload.article_url,
    )
    session.add(row)
    try:
        session.commit()
    except Exception as e:
        from sqlalchemy.exc import IntegrityError
        session.rollback()
        if isinstance(e, IntegrityError):
            # schon vorhanden
            raise HTTPException(status_code=409, detail="duplicate")
        raise
    session.refresh(row)
    return BildCorrectionOut(
        id=row.id,
        title=row.title,
        published=row.published,
        source_url=row.source_url,
        article_url=row.article_url,
        created_at=row.created_at,   # <- wichtig
    )

@router.delete("/corrections", status_code=204, dependencies=[Depends(require_api_key)])
def delete_all_corrections(session: Session = Depends(get_db_session)) -> None:
    """Löscht alle Bild-Corrections."""
    session.execute(delete(BildCorrection))
    session.commit()
    return