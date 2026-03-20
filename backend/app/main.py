from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from .db import ensure_schemas, engine
from .models import Base
from .routes import status as status_routes
from .routes import bild as bild_routes
from .routes import weather as weather_routes
from .routes import auth as auth_routes


app = FastAPI(title="sehbmaster backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    ensure_schemas()
    Base.metadata.create_all(bind=engine)


app.include_router(status_routes.router)
app.include_router(bild_routes.router)
app.include_router(weather_routes.router)
app.include_router(auth_routes.router)
