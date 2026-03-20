from __future__ import annotations
from fastapi import APIRouter, Depends, Header, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy import select, update, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from datetime import datetime, timezone
from typing import Optional
import os

from ..models_lobby import LobbyEntry, LobbyChange, LobbyAlert, LobbyLog
from ..db import get_session

router = APIRouter(prefix="/api/lobby", tags=["lobby"])
API_KEY = os.getenv("INGEST_API_KEY", "dev-secret")

def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

# ---- Pydantic ----
class EntryIn(BaseModel):
    register_number: str
    name: Optional[str] = None
    legal_form: Optional[str] = None
    first_publication_date: Optional[datetime] = None
    last_update_date: Optional[datetime] = None
    active: bool = True
    current_entry_id: Optional[int] = None
    financial_expenses_from: Optional[float] = None
    financial_expenses_to: Optional[float] = None
    refuse_financial_info: bool = False
    codex_violation: bool = False
    fields_of_interest: Optional[list[str]] = None
    client_orgs: Optional[list] = None
    client_persons: Optional[list] = None
    legislative_projects: Optional[list] = None
    raw_json: Optional[dict] = None

class EntryOut(EntryIn):
    created_at: datetime
    updated_at: datetime

class ChangeIn(BaseModel):
    register_number: str
    old_entry_id: Optional[int] = None
    new_entry_id: Optional[int] = None
    change_type: Optional[str] = None
    diff: Optional[dict] = None
    notes: Optional[str] = None

class ChangeOut(ChangeIn):
    id: int
    detected_at: datetime

class AlertIn(BaseModel):
    register_number: str
    alert_type: str
    description: Optional[str] = None
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

# ---- Entries ----
@router.get("/entries", response_model=list[EntryOut])
def list_entries(limit: int = 100, offset: int = 0, active_only: bool = True):
    with get_session() as s:
        q = select(LobbyEntry)
        if active_only:
            q = q.where(LobbyEntry.active == True)
        q = q.order_by(LobbyEntry.last_update_date.desc().nullslast()).offset(offset).limit(limit)
        rows = s.execute(q).scalars().all()
        return [EntryOut.model_validate(r, from_attributes=True) for r in rows]

@router.get("/entries/count")
def count_entries(active_only: bool = True):
    with get_session() as s:
        q = select(text("count(*)")).select_from(LobbyEntry)
        if active_only:
            q = q.where(LobbyEntry.active == True)
        return {"count": s.execute(q).scalar()}

@router.post("/entries", status_code=201)
def upsert_entry(payload: EntryIn, _=Depends(require_api_key)):
    data = payload.model_dump()
    data["updated_at"] = datetime.now(timezone.utc)
    with get_session() as s:
        stmt = pg_insert(LobbyEntry).values(**data).on_conflict_do_update(
            index_elements=["register_number"],
            set_={k: v for k, v in data.items() if k != "register_number"}
        )
        s.execute(stmt)
        s.commit()
    return {"ok": True}

# ---- Changes ----
@router.get("/changes", response_model=list[ChangeOut])
def list_changes(limit: int = 100, offset: int = 0):
    with get_session() as s:
        rows = s.execute(
            select(LobbyChange).order_by(LobbyChange.detected_at.desc()).offset(offset).limit(limit)
        ).scalars().all()
        return [ChangeOut.model_validate(r, from_attributes=True) for r in rows]

@router.post("/changes", status_code=201)
def add_change(payload: ChangeIn, _=Depends(require_api_key)):
    with get_session() as s:
        s.add(LobbyChange(**payload.model_dump()))
        s.commit()
    return {"ok": True}

# ---- Alerts ----
@router.get("/alerts", response_model=list[AlertOut])
def list_alerts(limit: int = 100, offset: int = 0):
    with get_session() as s:
        rows = s.execute(
            select(LobbyAlert).order_by(LobbyAlert.created_at.desc()).offset(offset).limit(limit)
        ).scalars().all()
        return [AlertOut.model_validate(r, from_attributes=True) for r in rows]

@router.post("/alerts", status_code=201)
def add_alert(payload: AlertIn, _=Depends(require_api_key)):
    with get_session() as s:
        s.add(LobbyAlert(**payload.model_dump()))
        s.commit()
    return {"ok": True}

# ---- Logs ----
@router.get("/logs", response_model=list[LogOut])
def list_logs(limit: int = 200):
    with get_session() as s:
        rows = s.execute(
            select(LobbyLog).order_by(LobbyLog.timestamp.desc()).limit(limit)
        ).scalars().all()
        return [LogOut.model_validate(r, from_attributes=True) for r in rows]

@router.post("/logs", status_code=201)
def add_log(payload: LogIn, _=Depends(require_api_key)):
    with get_session() as s:
        obj = LobbyLog(
            message=payload.message,
            timestamp=payload.timestamp or datetime.now(timezone.utc)
        )
        s.add(obj)
        s.commit()
    return {"ok": True}
