from pydantic import BaseModel, Field
from datetime import datetime

# ---- Status ----
class StatusIn(BaseModel):
    raspberry: str = Field(..., min_length=1, max_length=64)
    status: str = Field(..., min_length=1, max_length=64)
    message: str | None = None

class StatusOut(StatusIn):
    pass

class StatusPatch(BaseModel):
    status: str | None = Field(default=None, max_length=64)
    message: str | None = None

# ---- Dummy ----
class DummyIn(BaseModel):
    message: str = Field(..., min_length=1)

class DummyOut(DummyIn):
    pass


class BildWatchIn(BaseModel):
    id: str
    title: str
    url: str
    category: str | None = None
    is_premium: bool = False
    converted: bool = False
    published: datetime | None = None
    converted_time: datetime | None = None
    converted_duration_hours: float | None = None  # manuell gesetzt

class BildWatchOut(BildWatchIn):
    pass

class BildWatchMetricsIn(BaseModel):
    ts_hour: datetime  # ISO 8601 (UTC empfohlen)
    snapshot_total: int
    snapshot_premium: int
    snapshot_premium_pct: float
    new_count: int = 0
    new_premium_count: int = 0

class BildWatchMetricsOut(BildWatchMetricsIn):
    id: int
    created_at: datetime

class BildLogIn(BaseModel):
    message: str = Field(..., min_length=1)
    timestamp: datetime | None = None  # optional; wenn leer -> now()

class BildLogOut(BaseModel):
    id: int
    timestamp: datetime
    message: str

# ---- Bild Corrections (Live-Ticker) ----
class BildCorrectionIn(BaseModel):
    id: str
    title: str
    published: datetime
    source_url: str
    article_url: str | None = None
    message: str | None = None    

class BildCorrectionOut(BildCorrectionIn):
    created_at: datetime
    