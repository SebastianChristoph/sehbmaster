from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String, Text, Boolean, DateTime, Numeric, ARRAY, text
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from .models import Base

class LobbyEntry(Base):
    __tablename__ = "entries"
    __table_args__ = {"schema": "lobby"}

    register_number: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    legal_form: Mapped[str | None] = mapped_column(String(128), nullable=True)
    first_publication_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_update_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    current_entry_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    financial_expenses_from: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    financial_expenses_to: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    refuse_financial_info: Mapped[bool] = mapped_column(Boolean, default=False)
    codex_violation: Mapped[bool] = mapped_column(Boolean, default=False)
    fields_of_interest: Mapped[list | None] = mapped_column(ARRAY(Text), nullable=True)
    client_orgs: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    client_persons: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    legislative_projects: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

class LobbyChange(Base):
    __tablename__ = "changes"
    __table_args__ = {"schema": "lobby"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    register_number: Mapped[str] = mapped_column(String(32), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    old_entry_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    new_entry_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    change_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    diff: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

class LobbyAlert(Base):
    __tablename__ = "alerts"
    __table_args__ = {"schema": "lobby"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    register_number: Mapped[str] = mapped_column(String(32), nullable=False)
    alert_type: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

class LobbyLog(Base):
    __tablename__ = "log"
    __table_args__ = {"schema": "lobby"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    message: Mapped[str] = mapped_column(Text, nullable=False)
