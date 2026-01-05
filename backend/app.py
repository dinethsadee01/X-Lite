"""
Main FastAPI Application
Chest X-ray Multi-Label Classification API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from backend.routes import predict, upload, health, report

# Create FastAPI app
app = FastAPI(
    title="X-Lite API",
    description="Lightweight Hybrid CNN-Transformer for Chest X-Ray Classification",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(predict.router, prefix="/api", tags=["prediction"])
app.include_router(report.router, prefix="/api", tags=["report"])

# Serve static files (uploaded images, reports)
app.mount("/static", StaticFiles(directory=str(Config.UPLOAD_FOLDER)), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize app on startup"""
    # Create necessary directories
    Config.create_directories()
    print("✓ X-Lite API initialized")
    print(f"✓ Upload folder: {Config.UPLOAD_FOLDER}")
    print(f"✓ Checkpoint folder: {Config.CHECKPOINT_DIR}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("✓ X-Lite API shutdown")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "X-Lite API - Chest X-Ray Multi-Label Classification",
        "version": "0.1.0",
        "docs": "/api/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    # Create directories
    Config.create_directories()
    
    # Run server
    uvicorn.run(
        "app:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )
