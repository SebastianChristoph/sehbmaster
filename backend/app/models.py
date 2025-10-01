from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import (
    Integer, String, Text, Boolean, DateTime, Float, UniqueConstraint, text
)
from datetime import datetime

class Base(DeclarativeBase):
    pass

class Status(Base):
    __tablename__ = "status"
    __table_args__ = {"schema": "status"}

    raspberry: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True) 

class DummyTable(Base):
    __tablename__ = "dummy_table"
    __table_args__ = {"schema": "dummy"}

    message: Mapped[str] = mapped_column(Text, primary_key=True)

class BildWatch(Base):
    __tablename__ = "bildwatch"
    __table_args__ = ({"schema": "bild"},)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_premium: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    converted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    published: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    converted_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    converted_duration_hours: Mapped[float | None] = mapped_column(Float, nullable=True)

class BildWatchMetrics(Base):
    __tablename__ = "bildwatch_metrics"
    __table_args__ = (
        UniqueConstraint("ts_hour", name="uq_bildwatch_metrics_ts_hour"),
        {"schema": "bild"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts_hour: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    snapshot_total: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_premium: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_premium_pct: Mapped[float] = mapped_column(Float, nullable=False)
    new_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    new_premium_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )

class BildLog(Base):
    __tablename__ = "log"
    __table_args__ = ({"schema": "bild"},)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
