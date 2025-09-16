from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import select, update
from ..db import get_session
from ..models import Status
from ..schemas import StatusIn, StatusOut, StatusPatch
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
        return [StatusOut(raspberry=r.raspberry, status=r.status, message=r.message) for r in rows]

@router.post("", response_model=StatusOut)
def upsert_status(payload: StatusIn, _ok=Depends(require_api_key)):
    with get_session() as session:
        stmt = pg_insert(Status.__table__).values(
            raspberry=payload.raspberry,
            status=payload.status,
            message=payload.message,
        ).on_conflict_do_update(
            index_elements=[Status.__table__.c.raspberry],
            set_={"status": payload.status, "message": payload.message},
        )
        session.execute(stmt)
        session.commit()
        return StatusOut(**payload.model_dump())

# NEU: Teil-Update anhand "raspberry" (dein Identifier)
@router.patch("/{raspberry}", response_model=StatusOut)
def patch_status(raspberry: str, payload: StatusPatch, _ok=Depends(require_api_key)):
    # Nur gesetzte Felder übernehmen
    updates = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    with get_session() as session:
        # Gibt es den Datensatz?
        existing = session.execute(select(Status).where(Status.raspberry == raspberry)).scalar_one_or_none()
        if existing is None:
            raise HTTPException(status_code=404, detail="Status not found")

        # Update ausführen
        session.execute(
            update(Status)
            .where(Status.raspberry == raspberry)
            .values(**updates)
        )
        session.commit()

        # Reload für Antwort
        refreshed = session.execute(select(Status).where(Status.raspberry == raspberry)).scalar_one()
        return StatusOut(
            raspberry=refreshed.raspberry,
            status=refreshed.status,
            message=refreshed.message,
        )
