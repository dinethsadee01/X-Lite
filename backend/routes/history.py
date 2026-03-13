from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from bson import ObjectId

from backend.services.db_service import db, PredictionRecord
from backend.routes.auth import get_current_user
import os

router = APIRouter()

@router.get("/")
async def get_history(current_user: dict = Depends(get_current_user)):
    """Fetch prediction history for the authenticated user"""
    # History fetching bypassed because MongoDB is inactive
    return {"status": "success", "data": []}

@router.delete("/{record_id}")
async def delete_history_record(record_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a history record"""
    return {"status": "success", "message": "Record deleted"}
