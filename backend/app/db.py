from contextlib import contextmanager
import os
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

# --------------------------------------------------------------------
# Konfiguration
# --------------------------------------------------------------------
# Hinweis: In Docker/Prod kommt DATABASE_URL aus der ENV, z. B.
# postgresql+psycopg://sehb:sehbmaster2025!@db:5432/sehbmaster
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://sehb:pass@localhost:5432/sehbmaster",
)

# Engine mit soliden Defaults
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,          # invalid connections werden erkannt
    future=True,                 # moderne SQLAlchemy-APIs
)

# Session Factory
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session, future=True)

# --------------------------------------------------------------------
# 1) Contextmanager für Skripte / einmalige Jobs
#    Nutzung: with get_session() as s: ...
# --------------------------------------------------------------------
@contextmanager
def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# --------------------------------------------------------------------
# 2) FastAPI-Dependency (ohne @contextmanager!)
#    Nutzung in Routes: def endpoint(session: Session = Depends(get_db_session)):
# --------------------------------------------------------------------
def get_db_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --------------------------------------------------------------------
# Optionales Helper: Schemas sicherstellen (idempotent)
# Kann beim App-Start einmalig aufgerufen werden.
# --------------------------------------------------------------------
def ensure_schemas() -> None:
    """Stellt sicher, dass die Schemas existieren."""
    with engine.connect() as conn:
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS "status";'))
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS "dummy";'))
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS "bild";'))
        conn.commit()
