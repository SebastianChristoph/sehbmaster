# backend/app/models_weather.py
from __future__ import annotations
from datetime import datetime, date
from typing import Optional

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import BigInteger, SmallInteger, String, Date, Float, Text, DateTime

class BaseWeather(DeclarativeBase):
    pass

class WeatherData(BaseWeather):
    __tablename__ = "data"
    __table_args__ = {"schema": "weather"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    target_date: Mapped[date] = mapped_column(Date, nullable=False)
    lead_days: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False, default="default")
    city: Mapped[str] = mapped_column(String, nullable=False)
    run_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    weather: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # NEU: getrennte Temperaturen (+ Alt-Kompatibilität über temp_avg_c)
    temp_avg_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temp_min_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temp_max_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    wind_mps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rain_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

class WeatherLog(BaseWeather):
    __tablename__ = "log"
    __table_args__ = {"schema": "weather"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
