from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import select
from ..db import get_session
from ..models import Status
from ..schemas import StatusIn, StatusOut
import os

router = APIRouter(prefix="/api/status", tags=["status"])

API_KEY = os.getenv("INGEST_API_KEY", "dev-secret")

def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True

@router.get("", response_model=list[StatusOut])
def list_status():
    with get_session() as session:
        rows = session.execute(select(Status)).scalars().all()
        return [StatusOut(raspberry=r.raspberry, status=r.status) for r in rows]

@router.post("", response_model=StatusOut)
def upsert_status(payload: StatusIn, _ok=Depends(require_api_key)):
    with get_session() as session:
        stmt = pg_insert(Status.__table__).values(
            raspberry=payload.raspberry,
            status=payload.status,
        ).on_conflict_do_update(
            index_elements=[Status.__table__.c.raspberry],
            set_={"status": payload.status},
        )
        session.execute(stmt)
        session.commit()
        return StatusOut(**payload.model_dump())
