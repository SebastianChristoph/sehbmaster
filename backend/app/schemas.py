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