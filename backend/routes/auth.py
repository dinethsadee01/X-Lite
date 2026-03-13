import os
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Optional
import jwt

from backend.services.db_service import (
    db, verify_password, get_password_hash, create_access_token,
    SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, UserInDB
)
from datetime import timedelta

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    return {"username": "doctor", "email": "doctor@xlite.clinic", "_id": "dummy_id"}

@router.post("/login")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # Dummy login bypass
    if form_data.username == "doctor" and form_data.password == "password123":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": "doctor"}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer", "user": {"username": "doctor", "email": "doctor@xlite.clinic", "_id": "dummy_id"}}
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

@router.get("/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user

# Utility to initialize standard user since no signup allowed
async def init_default_user():
    pass
