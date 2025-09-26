# backend/app/routes/weather.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy import select, text, delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import List, Optional, Dict, Any
import os, math

from ..db import get_session
from ..models_weather import WeatherData, WeatherLog
from ..schemas_weather import (
    WeatherDataIn, WeatherDataOut,
    WeatherLogIn, WeatherLogOut,
    LeadBucketAccuracy, AccuracySummary,
)

router = APIRouter(prefix="/api/weather", tags=["weather"])
API_KEY = os.getenv("INGEST_API_KEY", "dev-secret")
TZ = ZoneInfo("Europe/Berlin")

def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True

# ---------- UPSERT ----------
@router.post("/data", response_model=WeatherDataOut, status_code=201)
def upsert_weather_data(payload: WeatherDataIn, _=Depends(require_api_key)):
    table = WeatherData.__table__
    with get_session() as s:
        stmt = (
            pg_insert(table)
            .values(
                target_date=payload.target_date,
                lead_days=payload.lead_days,
                model=payload.model,
                city=payload.city,
                run_time=payload.run_time,
                weather=payload.weather,
                temp_avg_c=payload.temp_avg_c,
                temp_min_c=payload.temp_min_c,
                temp_max_c=payload.temp_max_c,
                wind_mps=payload.wind_mps,
                rain_mm=payload.rain_mm,
            )
            .on_conflict_do_update(
                index_elements=[table.c.target_date, table.c.model, table.c.city, table.c.lead_days],
                set_={
                    "run_time": payload.run_time,
                    "weather": payload.weather,
                    "temp_avg_c": payload.temp_avg_c,
                    "temp_min_c": payload.temp_min_c,
                    "temp_max_c": payload.temp_max_c,
                    "wind_mps": payload.wind_mps,
                    "rain_mm": payload.rain_mm,
                },
            )
            .returning(*table.c)
        )
        row = s.execute(stmt).mappings().first()
        s.commit()
        return WeatherDataOut(**row)  # type: ignore[arg-type]

# ---------- SELECT ----------
@router.get("/data", response_model=List[WeatherDataOut])
def list_weather_data(
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    model: str = Query("default"),
    city: str = Query(...),
    lead_days: Optional[int] = Query(None, ge=0, le=7),
    limit: int = Query(5000, ge=1, le=100000),
    offset: int = Query(0, ge=0),
):
    with get_session() as s:
        q = select(WeatherData).where(WeatherData.model == model, WeatherData.city == city)
        if date_from:
            q = q.where(WeatherData.target_date >= date_from)
        if date_to:
            q = q.where(WeatherData.target_date <= date_to)
        if lead_days is not None:
            q = q.where(WeatherData.lead_days == lead_days)
        q = q.order_by(WeatherData.target_date.asc(), WeatherData.lead_days.desc())
        q = q.offset(offset).limit(limit)
        rows = s.execute(q).scalars().all()
        return [
            WeatherDataOut(
                id=r.id,
                target_date=r.target_date,
                lead_days=r.lead_days,
                model=r.model,
                city=r.city,
                run_time=r.run_time,
                weather=r.weather,
                temp_avg_c=r.temp_avg_c,
                temp_min_c=r.temp_min_c,
                temp_max_c=r.temp_max_c,
                wind_mps=r.wind_mps,
                rain_mm=r.rain_mm,
                created_at=r.created_at,
            ) for r in rows
        ]

# ---------- DELETE ALL ----------
@router.delete("/data", status_code=204)
def delete_all_weather_data(_=Depends(require_api_key)):
    with get_session() as s:
        s.execute(delete(WeatherData))
        s.commit()
    return

# ---------- runs/today ----------
@router.get("/runs/today")
def runs_today(model: str = Query("default"), city: str = Query(...), tz: str = Query("Europe/Berlin")) -> Dict[str, Any]:
    local_tz = ZoneInfo(tz)
    now_local = datetime.now(local_tz)
    start_local = datetime.combine(now_local.date(), datetime.min.time(), tzinfo=local_tz)
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)
    with get_session() as s:
        rows = s.execute(
            select(WeatherData).where(
                WeatherData.model == model,
                WeatherData.city == city,
                WeatherData.run_time >= start_utc,
                WeatherData.run_time < end_utc,
            ).order_by(WeatherData.run_time.asc())
        ).scalars().all()
    return {
        "already_scraped": len(rows) > 0,
        "count": len(rows),
        "first_run_time": rows[0].run_time.isoformat() if rows else None,
        "last_run_time": rows[-1].run_time.isoformat() if rows else None,
        "model": model, "city": city, "day_local": now_local.date().isoformat(), "tz": tz,
    }

# ---------- ACCURACY ----------
@router.get("/accuracy", response_model=AccuracySummary)
def get_accuracy(
    date_from: date = Query(...),
    date_to: date = Query(...),
    model: str = Query("default"),
    city: str = Query(...),
    max_lead: int = Query(7, ge=0, le=7),
):
    with get_session() as s:
        obs_rows = s.execute(
            select(WeatherData).where(
                WeatherData.model == model,
                WeatherData.city == city,
                WeatherData.lead_days == 0,
                WeatherData.target_date >= date_from,
                WeatherData.target_date <= date_to,
            )
        ).scalars().all()
        obs_by_day: Dict[date, WeatherData] = {r.target_date: r for r in obs_rows}

        buckets: List[LeadBucketAccuracy] = []
        for d in range(1, max_lead + 1):
            fc_rows = s.execute(
                select(WeatherData).where(
                    WeatherData.model == model,
                    WeatherData.city == city,
                    WeatherData.lead_days == d,
                    WeatherData.target_date >= date_from,
                    WeatherData.target_date <= date_to,
                )
            ).scalars().all()

            n = 0
            temp_min_err = 0.0
            temp_max_err = 0.0
            wind_err = 0.0
            rain_err = 0.0
            weather_match = 0

            for fc in fc_rows:
                obs = obs_by_day.get(fc.target_date)
                if not obs:
                    continue
                n += 1
                if fc.temp_min_c is not None and obs.temp_min_c is not None:
                    temp_min_err += abs(fc.temp_min_c - obs.temp_min_c)
                if fc.temp_max_c is not None and obs.temp_max_c is not None:
                    temp_max_err += abs(fc.temp_max_c - obs.temp_max_c)
                if fc.wind_mps is not None and obs.wind_mps is not None:
                    wind_err += abs(fc.wind_mps - obs.wind_mps)
                if fc.rain_mm is not None and obs.rain_mm is not None:
                    rain_err += abs(fc.rain_mm - obs.rain_mm)
                if fc.weather and obs.weather:
                    weather_match += int(fc.weather.strip().lower() == obs.weather.strip().lower())

            buckets.append(
                LeadBucketAccuracy(
                    lead_days=d,
                    n=n,
                    temp_min_mae=round(temp_min_err / n, 3) if n else None,
                    temp_max_mae=round(temp_max_err / n, 3) if n else None,
                    wind_mae=round(wind_err / n, 3) if n else None,
                    rain_mae=round(rain_err / n, 3) if n else None,
                    weather_match_pct=round((weather_match / n) * 100.0, 2) if n else None,
                )
            )

        return AccuracySummary(model=model, city=city, from_date=date_from, to_date=date_to, buckets=buckets)

# ---------- LOGS ----------
@router.post("/logs", response_model=WeatherLogOut, status_code=201)
def add_log(payload: WeatherLogIn, _=Depends(require_api_key)):
    ts = payload.timestamp or datetime.now(timezone.utc)
    with get_session() as s:
        obj = WeatherLog(timestamp=ts, message=payload.message)
        s.add(obj); s.commit(); s.refresh(obj)
        return WeatherLogOut(id=obj.id, timestamp=obj.timestamp, message=obj.message)

@router.get("/logs", response_model=List[WeatherLogOut])
def list_logs(limit: int = 1000, offset: int = 0, asc: bool = False):
    with get_session() as s:
        q = select(WeatherLog).order_by(WeatherLog.timestamp.asc() if asc else WeatherLog.timestamp.desc())
        q = q.limit(limit).offset(offset)
        rows = s.execute(q).scalars().all()
        return [WeatherLogOut(id=r.id, timestamp=r.timestamp, message=r.message) for r in rows]

@router.delete("/logs", status_code=204)
def delete_all_logs(_=Depends(require_api_key)):
    with get_session() as s:
        s.execute(text('TRUNCATE TABLE "weather"."log";')); s.commit()
    return
