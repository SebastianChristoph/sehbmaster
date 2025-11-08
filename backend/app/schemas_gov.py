from pydantic import BaseModel, AnyHttpUrl
from typing import List, Optional
from datetime import datetime

# ---- Artikel ----
class GovArticleIn(BaseModel):
    title: str
    source: str
    link: AnyHttpUrl | str
    published_at: Optional[datetime] = None

class GovArticleOut(GovArticleIn):
    id: int

# ---- Incident ----
class GovIncidentCreate(BaseModel):
    headline: str
    occurred_at: Optional[datetime] = None
    articles: List[GovArticleIn] = []

class GovIncidentListOut(BaseModel):
    id: int
    headline: str
    occurred_at: Optional[datetime] = None
    seen: bool
    created_at: datetime
    articles_count: int

class GovIncidentDetailOut(BaseModel):
    id: int
    headline: str
    occurred_at: Optional[datetime] = None
    seen: bool
    created_at: datetime
    articles: List[GovArticleOut]

class SeenPatch(BaseModel):
    seen: bool
