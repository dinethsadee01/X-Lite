"""
Report Generation Endpoint
Generate PDF reports for predictions
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import Config
from backend.services.report_service import ReportService

router = APIRouter()
report_service = ReportService()


class ReportRequest(BaseModel):
    """Request model for report generation"""
    patient_id: str = "Anonymous"
    predictions: List[Dict]
    image_filename: str
    additional_notes: str = ""


@router.post("/report/generate")
async def generate_report(request: ReportRequest):
    """
    Generate PDF report from prediction results
    
    Args:
        request: Report generation request
    
    Returns:
        dict: Report information with download link
    """
    try:
        # Generate report
        report_path = report_service.generate_pdf_report(
            patient_id=request.patient_id,
            predictions=request.predictions,
            image_filename=request.image_filename,
            notes=request.additional_notes
        )
        
        return {
            "success": True,
            "message": "Report generated successfully",
            "report_path": f"/static/reports/{report_path.name}",
            "download_url": f"/api/report/download/{report_path.name}"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}"
        )


@router.get("/report/download/{filename}")
async def download_report(filename: str):
    """
    Download generated PDF report
    
    Args:
        filename: Report filename
    
    Returns:
        FileResponse: PDF file download
    """
    try:
        report_path = Config.UPLOAD_FOLDER / "reports" / filename
        
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")
        
        return FileResponse(
            path=str(report_path),
            media_type="application/pdf",
            filename=filename
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Download failed: {str(e)}"
        )
