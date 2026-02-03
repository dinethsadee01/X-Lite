# X-Lite Frontend & Backend - Quick Start Guide

## Overview
This is a basic implementation of X-Lite showing core functionalities:
- **Backend**: FastAPI-based REST API for model inference
- **Frontend**: React-based UI for image upload and prediction display

## Project Structure
```
├── backend/
│   ├── app.py           # Main FastAPI application
│   ├── routes/          # API endpoints
│   ├── services/        # Business logic (prediction, image handling)
│   └── uploads/         # Uploaded images storage
├── frontend/
│   ├── public/          # Static assets
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── App.js       # Main app component
│   │   └── index.js     # Entry point
│   └── package.json     # Dependencies
└── config/              # Configuration and constants
```

## Prerequisites
- Python 3.8+
- Node.js 14+
- Virtual environment (recommended)

## Setup Instructions

### 1. Backend Setup

#### Option A: Quick Start (Windows)
```bash
# Run the startup script
start.bat
```

#### Option B: Manual Setup
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install/update backend dependencies
pip install -r requirements.txt

# Start backend API
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

**Backend will be available at:** `http://localhost:8000`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (if not already done)
npm install

# Start development server
npm start
```

**Frontend will be available at:** `http://localhost:3000`

### 3. Verify Setup
```bash
# Run verification script
python verify_setup.py
```

## API Endpoints

### Health Check
```
GET /api/health
```
Returns system status and model information.

### Upload Image
```
POST /api/upload
Content-Type: multipart/form-data
- file: <chest X-ray image>

Response:
{
  "success": true,
  "data": {
    "filename": "20240203_120000_abc123def456.jpg",
    "file_id": "abc123def456",
    "file_size_bytes": 245872,
    "file_path": "/static/20240203_120000_abc123def456.jpg"
  }
}
```

### Get Predictions
```
POST /api/predict
Content-Type: application/json

Request:
{
  "filename": "20240203_120000_abc123def456.jpg",
  "return_heatmap": true,
  "confidence_threshold": 0.5
}

Response:
{
  "success": true,
  "predictions": [
    {
      "disease": "Pneumonia",
      "probability": 0.87,
      "risk_level": "high",
      "color": "#f57c00",
      "description": "Infection causing inflammation in air sacs"
    },
    ...
  ],
  "positive_findings": ["Pneumonia", "Infiltration"],
  "num_positive": 2,
  "processing_time_ms": 245.3,
  "model_name": "convnext_tiny_mhsa"
}
```

## Features Implemented

### Frontend
- ✅ Drag-and-drop image upload
- ✅ Image preview
- ✅ Real-time predictions
- ✅ Disease probability display with progress bars
- ✅ Risk level color-coding
- ✅ Loading states
- ✅ Error handling
- ✅ Responsive Material-UI design

### Backend
- ✅ Image validation and storage
- ✅ Multi-label disease prediction
- ✅ CORS support for frontend integration
- ✅ Structured API responses
- ✅ Health check endpoint
- ✅ Error handling with proper HTTP status codes
- ✅ Swagger/OpenAPI documentation

## Usage Workflow

1. Open `http://localhost:3000` in your browser
2. Drag and drop a chest X-ray image or click to select
3. Wait for processing (typically 200-500ms)
4. View predictions with:
   - Disease name
   - Probability score
   - Risk level (Low/Moderate/High/Critical)
   - Detailed description
5. Click "Analyze Another Image" to process another X-ray

## API Documentation
Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## Testing

### Test with Sample Image
```bash
# Using curl
curl -X POST -F "file=@/path/to/xray.jpg" http://localhost:8000/api/upload

# Then predict
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"filename": "your_uploaded_filename.jpg"}'
```

### Test from Python
```python
import requests

# Upload
with open('xray.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/api/upload', files=files)
    filename = response.json()['data']['filename']

# Predict
response = requests.post('http://localhost:8000/api/predict', json={
    'filename': filename,
    'return_heatmap': True,
    'confidence_threshold': 0.5
})
print(response.json())
```

## Directory Structure for Uploads
Uploaded images are stored in: `backend/uploads/`

Files are named with timestamp and unique ID to avoid conflicts:
```
backend/uploads/
├── 20240203_120000_abc123def456.jpg
├── 20240203_120515_xyz789abc123.jpg
└── ...
```

## Configuration

### Backend Config
Edit `config/config.py`:
- `UPLOAD_FOLDER`: Where uploaded images are stored
- `MAX_UPLOAD_SIZE`: Maximum file size (default: 10MB)
- `ALLOWED_EXTENSIONS`: Supported image formats
- `CONFIDENCE_THRESHOLD`: Default prediction threshold
- `API_HOST` and `API_PORT`: Backend server settings

### Frontend Config
Edit `frontend/.env`:
- `REACT_APP_API_URL`: Backend API URL
- `REACT_APP_NAME`: Application name
- `REACT_APP_VERSION`: App version

## Troubleshooting

### Backend Won't Start
```
Error: Port 8000 already in use
Solution: Kill the process or use a different port
  Windows: netstat -ano | findstr :8000
  macOS/Linux: lsof -i :8000
```

### Frontend Dependencies Missing
```
Error: npm packages not installed
Solution: cd frontend && npm install
```

### CORS Errors
```
Error: Access to XMLHttpRequest has been blocked by CORS policy
Solution: Backend CORS is already configured in app.py
  Check that REACT_APP_API_URL matches backend URL
```

### No Predictions
```
Ensure:
1. Model checkpoint is loaded (check backend logs)
2. Image format is supported (JPG, PNG)
3. Image is valid chest X-ray
```

## Performance

### Typical Response Times
- **Image Upload**: < 100ms
- **Prediction**: 200-500ms (CPU)
- **Prediction**: 50-150ms (GPU)
- **Total**: ~300-600ms for upload + prediction

### Optimization Tips
- Use GPU if available (much faster)
- Batch process multiple images with `/api/predict/batch`
- Enable image caching for repeated images

## Next Steps for Enhancement

### Phase 2 Features (Future)
- Report generation with disease summaries
- Image history and comparison
- Model fine-tuning interface
- Advanced visualizations (Grad-CAM heatmaps)
- User authentication
- Database for storing results
- Mobile app

### Performance Improvements
- Model quantization
- Batch prediction optimization
- Image caching
- Frontend optimization

## Support
For issues or questions, check:
1. Backend logs in terminal
2. Browser console for frontend errors
3. API documentation at `/api/docs`

## License
This project is part of X-Lite research initiative.
