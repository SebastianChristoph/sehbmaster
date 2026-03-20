from __future__ import annotations
from fastapi import APIRouter, Depends, Header, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy import select, text, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from datetime import datetime, date, timezone
from typing import Optional
import os

from ..models_vergabe import VergabeNotice, VergabeAlert, VergabeLog
from ..db import get_session

router = APIRouter(prefix="/api/vergabe", tags=["vergabe"])
API_KEY = os.getenv("INGEST_API_KEY", "dev-secret")

def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

class NoticeIn(BaseModel):
    publication_number: str
    notice_type: Optional[str] = None
    published_date: Optional[date] = None
    contracting_authority: Optional[str] = None
    contracting_country: Optional[str] = "DEU"
    contractor_name: Optional[str] = None
    contract_value_eur: Optional[float] = None
    cpv_code: Optional[str] = None
    cpv_description: Optional[str] = None
    procedure_type: Optional[str] = None
    description: Optional[str] = None
    ted_url: Optional[str] = None
    raw_xml_url: Optional[str] = None

class NoticeOut(NoticeIn):
    created_at: datetime

class AlertIn(BaseModel):
    alert_type: str
    authority: Optional[str] = None
    contractor: Optional[str] = None
    evidence: Optional[dict] = None

class AlertOut(AlertIn):
    id: int
    created_at: datetime

class LogIn(BaseModel):
    message: str
    timestamp: Optional[datetime] = None

class LogOut(BaseModel):
    id: int
    timestamp: datetime
    message: str

# ---- Notices ----
@router.get("/notices", response_model=list[NoticeOut])
def list_notices(limit: int = 100, offset: int = 0, procedure_type: Optional[str] = None):
    with get_session() as s:
        q = select(VergabeNotice).order_by(VergabeNotice.published_date.desc().nullslast())
        if procedure_type is not None:
            if procedure_type == '__null__':
                q = q.where(VergabeNotice.procedure_type.is_(None))
            else:
                q = q.where(VergabeNotice.procedure_type == procedure_type)
        rows = s.execute(q.offset(offset).limit(limit)).scalars().all()
        return [NoticeOut.model_validate(r, from_attributes=True) for r in rows]

@router.get("/notices/stats")
def notices_stats():
    with get_session() as s:
        total = s.execute(select(func.count()).select_from(VergabeNotice)).scalar()
        total_value = s.execute(select(func.sum(VergabeNotice.contract_value_eur))).scalar()
        top_contractors = s.execute(
            select(VergabeNotice.contractor_name, func.count().label("cnt"))
            .where(VergabeNotice.contractor_name.isnot(None))
            .group_by(VergabeNotice.contractor_name)
            .order_by(func.count().desc())
            .limit(10)
        ).all()
        top_authorities = s.execute(
            select(VergabeNotice.contracting_authority, func.count().label("cnt"))
            .where(VergabeNotice.contracting_authority.isnot(None))
            .group_by(VergabeNotice.contracting_authority)
            .order_by(func.count().desc())
            .limit(10)
        ).all()
        return {
            "total_notices": total,
            "total_value_eur": float(total_value) if total_value else 0,
            "top_contractors": [{"name": r[0], "count": r[1]} for r in top_contractors],
            "top_authorities": [{"name": r[0], "count": r[1]} for r in top_authorities],
        }

@router.get("/notices/procedure_types")
def procedure_types():
    with get_session() as s:
        rows = s.execute(
            select(VergabeNotice.procedure_type, func.count().label("cnt"))
            .group_by(VergabeNotice.procedure_type)
            .order_by(func.count().desc())
        ).all()
        return [{"code": r[0] if r[0] is not None else "__null__", "count": r[1]} for r in rows]

@router.post("/notices", status_code=201)
def upsert_notice(payload: NoticeIn, _=Depends(require_api_key)):
    data = payload.model_dump()
    with get_session() as s:
        stmt = pg_insert(VergabeNotice).values(**data).on_conflict_do_update(
            index_elements=["publication_number"],
            set_={k: v for k, v in data.items() if k != "publication_number"}
        )
        s.execute(stmt)
        s.commit()
    return {"ok": True}

# ---- Alerts ----
@router.get("/alerts", response_model=list[AlertOut])
def list_alerts(limit: int = 100):
    with get_session() as s:
        rows = s.execute(
            select(VergabeAlert).order_by(VergabeAlert.created_at.desc()).limit(limit)
        ).scalars().all()
        return [AlertOut.model_validate(r, from_attributes=True) for r in rows]

@router.post("/alerts", status_code=201)
def add_alert(payload: AlertIn, _=Depends(require_api_key)):
    with get_session() as s:
        s.add(VergabeAlert(**payload.model_dump()))
        s.commit()
    return {"ok": True}

# ---- Logs ----
@router.get("/logs", response_model=list[LogOut])
def list_logs(limit: int = 200):
    with get_session() as s:
        rows = s.execute(
            select(VergabeLog).order_by(VergabeLog.timestamp.desc()).limit(limit)
        ).scalars().all()
        return [LogOut.model_validate(r, from_attributes=True) for r in rows]

@router.post("/logs", status_code=201)
def add_log(payload: LogIn, _=Depends(require_api_key)):
    with get_session() as s:
        obj = VergabeLog(
            message=payload.message,
            timestamp=payload.timestamp or datetime.now(timezone.utc)
        )
        s.add(obj)
        s.commit()
    return {"ok": True}
