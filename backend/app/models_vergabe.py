from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String, Text, Boolean, DateTime, Numeric, Date, text
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime, date
from .models import Base

class VergabeNotice(Base):
    __tablename__ = "notices"
    __table_args__ = {"schema": "vergabe"}

    publication_number: Mapped[str] = mapped_column(String(64), primary_key=True)
    notice_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    published_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    contracting_authority: Mapped[str | None] = mapped_column(Text, nullable=True)
    contracting_country: Mapped[str | None] = mapped_column(String(8), nullable=True, default="DEU")
    contractor_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    contract_value_eur: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    cpv_code: Mapped[str | None] = mapped_column(String(16), nullable=True)
    cpv_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    procedure_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    ted_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_xml_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

class VergabeAlert(Base):
    __tablename__ = "alerts"
    __table_args__ = {"schema": "vergabe"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_type: Mapped[str] = mapped_column(String(64), nullable=False)
    authority: Mapped[str | None] = mapped_column(Text, nullable=True)
    contractor: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

class VergabeLog(Base):
    __tablename__ = "log"
    __table_args__ = {"schema": "vergabe"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    message: Mapped[str] = mapped_column(Text, nullable=False)
