# backend/app/services/bild_charts.py
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple

from sqlalchemy import select, func

from ..models import BildWatch, BildWatchMetrics

TZ = ZoneInfo("Europe/Berlin")

def _utc_to_local_berlin(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TZ)

# ---------- a) Kategorie-Verteilung ----------
def compute_category_counts(session, *, premium_only: bool = False) -> Dict[str, int]:
    q = session.query(BildWatch.category, func.count(BildWatch.id))
    if premium_only:
        q = q.filter(BildWatch.is_premium.is_(True))
    rows: List[Tuple[str | None, int]] = q.group_by(BildWatch.category).all()

    result: Dict[str, int] = {}
    for cat, count in rows:
        label = cat if (cat is not None and str(cat).strip() != "") else "Unbekannt"
        result[label] = count
    return result

# ---------- b/c) Stündliche Ø-Werte ----------
def compute_hourly_charts(
    session,
    *,
    time_from: datetime,
    time_to: datetime,
    limit: int = 20000,
) -> dict:
    q = (
        select(BildWatchMetrics)
        .where(BildWatchMetrics.ts_hour >= time_from)
        .where(BildWatchMetrics.ts_hour < time_to)
        .order_by(BildWatchMetrics.ts_hour.asc())
        .limit(limit)
    )
    rows = session.execute(q).scalars().all()

    if not rows:
        zeros = [{"hour": h, "Premium": 0.0, "Nicht_Premium": 0.0} for h in range(24)]
        return {"snapshot_avg": zeros, "new_avg": zeros}

    bucket_snap = {h: {"p": [], "np": []} for h in range(24)}
    bucket_new  = {h: {"p": [], "np": []} for h in range(24)}

    for r in rows:
        local = _utc_to_local_berlin(r.ts_hour)
        h = local.hour

        snapshot_p = float(r.snapshot_premium or 0)
        snapshot_np = float((r.snapshot_total or 0) - (r.snapshot_premium or 0))
        new_p = float(r.new_premium_count or 0.0)
        new_np = float((r.new_count or 0.0) - (r.new_premium_count or 0.0))

        bucket_snap[h]["p"].append(snapshot_p)
        bucket_snap[h]["np"].append(snapshot_np)
        bucket_new[h]["p"].append(new_p)
        bucket_new[h]["np"].append(new_np)

    def avg(vals: List[float]) -> float:
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    snapshot_avg = [
        {"hour": h, "Premium": avg(bucket_snap[h]["p"]), "Nicht_Premium": avg(bucket_snap[h]["np"])}
        for h in range(24)
    ]
    new_avg = [
        {"hour": h, "Premium": avg(bucket_new[h]["p"]), "Nicht_Premium": avg(bucket_new[h]["np"])}
        for h in range(24)
    ]
    return {"snapshot_avg": snapshot_avg, "new_avg": new_avg}

# ---------- d) Umstellungen Premium→frei pro Tag ----------
def compute_daily_conversions(
    session,
    *,
    time_from: datetime,
    time_to: datetime,
    limit: int = 200000,
) -> List[dict]:
    q = (
        select(BildWatch.id, BildWatch.converted_time)
        .where(BildWatch.converted_time.isnot(None))
        .where(BildWatch.converted_time >= time_from)
        .where(BildWatch.converted_time < time_to)
        .order_by(BildWatch.converted_time.asc())
        .limit(limit)
    )
    rows = session.execute(q).all()

    per_day: Dict[str, int] = {}
    for _id, ct in rows:
        if not ct:
            continue
        local = _utc_to_local_berlin(ct)
        key = local.strftime("%Y-%m-%d")
        per_day[key] = per_day.get(key, 0) + 1

    return [{"day": d, "count": per_day[d]} for d in sorted(per_day.keys())]
