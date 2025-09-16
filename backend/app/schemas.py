from pydantic import BaseModel, Field

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
