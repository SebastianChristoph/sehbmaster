from sqlalchemy import Column, Integer, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .db import SessionLocal  # nur falls woanders benötigt
from .models import Base      # du hast bereits ein gemeinsames Base

SCHEMA = "gov"

class GovIncident(Base):
    __tablename__ = "incidents"
    __table_args__ = {"schema": SCHEMA}

    id = Column(Integer, primary_key=True)
    headline = Column(Text, nullable=False)
    occurred_at = Column(DateTime(timezone=True), nullable=True)
    seen = Column(Boolean, nullable=False, server_default="false")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    articles = relationship("GovArticle", back_populates="incident", cascade="all,delete-orphan")

class GovArticle(Base):
    __tablename__ = "articles"
    __table_args__ = {"schema": SCHEMA}

    id = Column(Integer, primary_key=True)
    incident_id = Column(Integer, ForeignKey(f'{SCHEMA}.incidents.id', ondelete="CASCADE"), nullable=False)
    title = Column(Text, nullable=False)
    source = Column(Text, nullable=False)
    link = Column(Text, nullable=False, unique=True)
    published_at = Column(DateTime(timezone=True), nullable=True)

    incident = relationship("GovIncident", back_populates="articles")
