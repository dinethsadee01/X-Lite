"""
Health Check Endpoint
Simple endpoint to check if API is running
"""

from fastapi import APIRouter
from datetime import datetime
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import Config

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    Returns API status and system information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
        "system": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "config": {
            "num_classes": Config.NUM_CLASSES,
            "image_size": Config.IMAGE_SIZE,
            "model_checkpoint": str(Config.CHECKPOINT_DIR)
        }
    }


@router.get("/status")
async def get_status():
    """
    Detailed status endpoint
    Returns model loading status and capabilities
    """
    # Check if model checkpoint exists
    checkpoint_exists = Config.CHECKPOINT_DIR.exists()
    
    # List available models
    available_models = []
    if checkpoint_exists:
        available_models = [
            f.stem for f in Config.CHECKPOINT_DIR.glob("*.pth")
        ]
    
    return {
        "api_status": "running",
        "model_ready": len(available_models) > 0,
        "available_models": available_models,
        "supported_formats": list(Config.ALLOWED_EXTENSIONS),
        "max_upload_size_mb": Config.MAX_UPLOAD_SIZE / (1024 * 1024),
        "timestamp": datetime.now().isoformat()
    }
