from fastapi import APIRouter
from sqlalchemy import select
from ..db import get_session
from ..models import DummyTable
from ..schemas import DummyIn, DummyOut

router = APIRouter(prefix="/api/dummy", tags=["dummy"])

@router.get("", response_model=list[DummyOut])
def list_dummy():
    with get_session() as session:
        rows = session.execute(select(DummyTable)).scalars().all()
        return [DummyOut(message=r.message) for r in rows]

@router.post("", response_model=DummyOut)
def create_dummy(payload: DummyIn):
    with get_session() as session:
        row = DummyTable(message=payload.message)
        session.add(row)
        session.commit()
        return DummyOut(message=row.message)
