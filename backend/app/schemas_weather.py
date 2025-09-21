# backend/app/schemas_weather.py
from __future__ import annotations
from datetime import datetime, date
from pydantic import BaseModel, Field
from typing import Optional, List

# --- CRUD ---
class WeatherDataIn(BaseModel):
    target_date: date
    lead_days: int = Field(ge=0, le=7)
    model: str = "default"
    city: str                                # NEU
    run_time: datetime

    weather: Optional[str] = None
    temp_c: Optional[float] = None
    wind_mps: Optional[float] = None
    rain_mm: Optional[float] = None

class WeatherDataOut(WeatherDataIn):
    id: int
    created_at: datetime

# --- Logs ---
class WeatherLogIn(BaseModel):
    timestamp: Optional[datetime] = None
    message: str

class WeatherLogOut(BaseModel):
    id: int
    timestamp: datetime
    message: str

# --- Accuracy response ---
class LeadBucketAccuracy(BaseModel):
    lead_days: int
    n: int
    temp_mae: Optional[float] = None
    wind_mae: Optional[float] = None
    rain_mae: Optional[float] = None
    weather_match_pct: Optional[float] = None

class AccuracySummary(BaseModel):
    model: str
    city: str                                # NEU
    from_date: date
    to_date: date
    buckets: List[LeadBucketAccuracy]
