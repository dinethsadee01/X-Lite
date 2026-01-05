"""
Image Upload Endpoint
Handles chest X-ray image uploads with validation
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from pathlib import Path
import sys
import uuid
from datetime import datetime
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import Config
from backend.services.image_service import ImageService

router = APIRouter()
image_service = ImageService()


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload chest X-ray image for prediction
    
    Args:
        file: Image file (PNG, JPG, JPEG, DCM)
    
    Returns:
        dict: Upload status and file information
    """
    try:
        # Validate file
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in Config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {Config.ALLOWED_EXTENSIONS}"
            )
        
        # Generate unique filename
        unique_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_{unique_id}{file_ext}"
        
        # Save file
        save_path = Config.UPLOAD_FOLDER / new_filename
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file info
        file_size = save_path.stat().st_size
        
        # Validate image (check if it's a valid image)
        is_valid, error_msg = image_service.validate_image(save_path)
        if not is_valid:
            save_path.unlink()  # Delete invalid file
            raise HTTPException(status_code=400, detail=error_msg)
        
        return {
            "success": True,
            "message": "File uploaded successfully",
            "data": {
                "file_id": unique_id,
                "filename": new_filename,
                "original_filename": file.filename,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "upload_time": datetime.now().isoformat(),
                "file_path": f"/static/{new_filename}"
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.delete("/upload/{file_id}")
async def delete_uploaded_file(file_id: str):
    """
    Delete uploaded file
    
    Args:
        file_id: Unique file identifier
    
    Returns:
        dict: Deletion status
    """
    try:
        # Find file with matching ID
        files = list(Config.UPLOAD_FOLDER.glob(f"*_{file_id}.*"))
        
        if not files:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete file
        for file_path in files:
            file_path.unlink()
        
        return {
            "success": True,
            "message": "File deleted successfully",
            "file_id": file_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")
