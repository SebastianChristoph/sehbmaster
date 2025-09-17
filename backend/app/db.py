from contextlib import contextmanager
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://sehb:pass@localhost:5432/sehbmaster")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.close()
    except Exception:
        session.close()
        raise

def ensure_schemas():
    """Stellt sicher, dass die beiden Schemas existieren."""
    with engine.connect() as conn:
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS "status";'))
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS "dummy";'))
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS "bild";')) 
        conn.commit()
