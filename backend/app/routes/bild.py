# backend/app/routes/bild.py
from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from ..db import get_session
from ..models import BildWatch, BildWatchMetrics
from ..schemas import (
    BildWatchIn, BildWatchOut,
    BildWatchMetricsIn, BildWatchMetricsOut,
)
from pydantic import BaseModel
from datetime import datetime
import os

router = APIRouter(prefix="/api/bild", tags=["bild"])

API_KEY = os.getenv("INGEST_API_KEY", "dev-secret")

def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True

# -------- GET: alle Articles --------
@router.get("/articles", response_model=list[BildWatchOut])
def get_all_articles(limit: int = 500, offset: int = 0):
    with get_session() as s:
        q = select(BildWatch).order_by(BildWatch.published.desc()) \
                             .offset(offset).limit(limit)
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

# -------- POST: neuen Article hinzufügen --------
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

# -------- PATCH: Article per ID teilweise aktualisieren --------
class BildWatchPatch(BaseModel):
    is_premium: bool | None = None           # <— NEU
    converted: bool | None = None
    converted_time: datetime | None = None
    converted_duration_hours: float | None = None
    
@router.patch("/articles/{article_id}", response_model=BildWatchOut)
def update_article(article_id: str, patch: BildWatchPatch, _=Depends(require_api_key)):
    updates = patch.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    with get_session() as s:
        obj = s.get(BildWatch, article_id)
        if not obj:
            raise HTTPException(status_code=404, detail="Article not found")

        for k, v in updates.items():
            setattr(obj, k, v)

        s.commit()
        s.refresh(obj)

        return BildWatchOut(
            id=obj.id, title=obj.title, url=obj.url, category=obj.category,
            is_premium=obj.is_premium, converted=obj.converted,
            published=obj.published, converted_time=obj.converted_time,
            converted_duration_hours=obj.converted_duration_hours,
        )

# -------- DELETE: Alle Articles löschen --------
@router.delete("/articles", status_code=204, dependencies=[Depends(require_api_key)])
def delete_all_articles():
    with get_session() as s:
        s.execute(delete(BildWatch))
        s.commit()
    return

# -------- GET: Kategorien-Counts für Kreisdiagramm --------
@router.get("/articles/category_counts", response_model=dict)
def get_category_counts():
    with get_session() as s:
        q = (
            s.query(BildWatch.category, func.count(BildWatch.id))
            .group_by(BildWatch.category)
            .all()
        )
        # category kann None sein, das ggf. als "Unbekannt" labeln
        result = {}
        for cat, count in q:
            label = cat if cat is not None else "Unbekannt"
            result[label] = count
        return JSONResponse(result)


@router.post("/metrics", response_model=BildWatchMetricsOut, status_code=201)
def upsert_metrics(payload: BildWatchMetricsIn, _=Depends(require_api_key)):
    """
    Upsert pro Stunde:
    - Snapshot-Felder werden *überschrieben*
    - new_count / new_premium_count werden *aufaddiert*
    """
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
            .returning(*table.c)  # wir wollen die Zeile zurück
        )
        row = s.execute(stmt).mappings().first()
        s.commit()
        return BildWatchMetricsOut(**row)  # type: ignore[arg-type]

@router.get("/metrics", response_model=list[BildWatchMetricsOut])
def get_metrics(
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    limit: int = 1000,
):
    """
    Liefert Metriken, optional gefiltert nach Zeitfenster.
    Standard: letzte 1000 Buckets.
    """
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