from fastapi import APIRouter
from sqlalchemy import text
from ..db import get_session

router = APIRouter(prefix="/api/info", tags=["info"])

SYSTEM_SCHEMAS = {"public", "information_schema", "pg_catalog", "pg_toast",
                  "pg_temp_1", "pg_toast_temp_1", "dummy"}

@router.get("")
def get_info():
    with get_session() as session:
        rows = session.execute(
            text("SELECT schema_name FROM information_schema.schemata ORDER BY schema_name")
        ).fetchall()
    schemas = [r[0] for r in rows if r[0] not in SYSTEM_SCHEMAS and not r[0].startswith("pg_")]
    return {"schemas": schemas}
