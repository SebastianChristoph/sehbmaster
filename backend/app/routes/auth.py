import os
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from ..auth import create_access_token

router = APIRouter(prefix="/api/auth", tags=["auth"])

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest):
    if payload.username != ADMIN_USERNAME or payload.password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    token = create_access_token({"sub": payload.username, "role": "admin"})
    return TokenResponse(access_token=token)
