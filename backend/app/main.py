from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from .db import ensure_schemas, engine
from .models import Base
from .routes import status as status_routes
from .routes import dummy as dummy_routes

app = FastAPI(title="sehbmaster backend", version="0.1.0")

# --- CORS: ALLES ERLAUBEN ---
# Hinweis: Das ist nur für offene Dev/Tests sinnvoll.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # <= jeder Origin darf
    allow_credentials=False, # "*" + Credentials ist nicht erlaubt
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    ensure_schemas()
    Base.metadata.create_all(bind=engine)

app.include_router(status_routes.router)
app.include_router(dummy_routes.router)

@app.get("/healthz")
def health():
    return {"status": "ok"}
