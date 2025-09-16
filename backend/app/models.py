from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text
from typing import Optional

class Base(DeclarativeBase):
    pass

# Schema: status, Tabelle: status
class Status(Base):
    __tablename__ = "status"
    __table_args__ = {"schema": "status"}

    # raspberry als Primärschlüssel (einfach und eindeutig)
    raspberry: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True) 

# Schema: dummy, Tabelle: dummy_table
class DummyTable(Base):
    __tablename__ = "dummy_table"
    __table_args__ = {"schema": "dummy"}

    # nur eine Spalte - wir machen sie zum Primärschlüssel
    message: Mapped[str] = mapped_column(Text, primary_key=True)
