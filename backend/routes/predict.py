"""
Prediction Endpoint
Handles inference requests for chest X-ray classification
"""

from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import Config, DISEASE_LABELS, get_risk_level, get_risk_color
from backend.services.prediction_service import PredictionService

router = APIRouter()

# Initialize prediction service (lazy loading)
prediction_service = None


def get_prediction_service():
    """Get or create prediction service instance"""
    global prediction_service
    if prediction_service is None:
        prediction_service = PredictionService()
    return prediction_service


class PredictionRequest(BaseModel):
    """Request model for prediction"""
    filename: str
    return_heatmap: bool = True
    confidence_threshold: float = 0.5


class PredictionResult(BaseModel):
    """Single disease prediction result"""
    disease: str
    probability: float
    risk_level: str
    color: str
    description: Optional[str] = None


class PredictionResponse(BaseModel):
    """Complete prediction response"""
    success: bool
    predictions: List[PredictionResult]
    positive_findings: List[str]
    num_positive: int
    heatmap_path: Optional[str] = None
    processing_time_ms: float
    model_name: str


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict diseases from uploaded chest X-ray
    
    Args:
        request: Prediction request with filename
    
    Returns:
        PredictionResponse: Predictions with probabilities and heatmap
    """
    try:
        # Get prediction service
        service = get_prediction_service()
        
        # Get full file path
        image_path = Config.UPLOAD_FOLDER / request.filename
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Run prediction
        result = service.predict(
            image_path=str(image_path),
            return_heatmap=request.return_heatmap,
            threshold=request.confidence_threshold
        )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/batch")
async def predict_batch(filenames: List[str], return_heatmap: bool = False):
    """
    Batch prediction for multiple images
    
    Args:
        filenames: List of uploaded filenames
        return_heatmap: Whether to generate heatmaps
    
    Returns:
        List of predictions
    """
    try:
        service = get_prediction_service()
        results = []
        
        for filename in filenames:
            image_path = Config.UPLOAD_FOLDER / filename
            
            if not image_path.exists():
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": "File not found"
                })
                continue
            
            try:
                result = service.predict(
                    image_path=str(image_path),
                    return_heatmap=return_heatmap
                )
                result["filename"] = filename
                results.append(result)
            except Exception as e:
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "total": len(filenames),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/models")
async def list_available_models():
    """
    List available trained models
    
    Returns:
        List of model names and metadata
    """
    try:
        models = []
        
        if Config.CHECKPOINT_DIR.exists():
            for model_path in Config.CHECKPOINT_DIR.glob("*.pth"):
                size_mb = model_path.stat().st_size / (1024 * 1024)
                models.append({
                    "name": model_path.stem,
                    "filename": model_path.name,
                    "size_mb": round(size_mb, 2),
                    "path": str(model_path)
                })
        
        return {
            "success": True,
            "models": models,
            "total": len(models)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )
