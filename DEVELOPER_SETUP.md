# X-Lite Backend & Frontend - Developer Setup Guide

## ✅ What's Ready

### Backend (FastAPI)
- [x] Main FastAPI application configured with CORS
- [x] Health check endpoint (`/api/health`)
- [x] Image upload endpoint (`/api/upload`)
- [x] Prediction endpoint (`/api/predict`)
- [x] Report endpoint (`/api/report`)
- [x] Error handling and validation
- [x] Static file serving for uploads

### Frontend (React)
- [x] Modern React 18 with Material-UI components
- [x] Drag-and-drop image upload
- [x] Image preview with display
- [x] Prediction results visualization
- [x] Disease probability charts with risk levels
- [x] Loading states and error handling
- [x] Responsive design

### Core Features Implemented
- [x] Image upload validation
- [x] Real-time predictions
- [x] Multi-label disease classification display
- [x] Risk level color-coding (Low/Moderate/High/Critical)
- [x] Processing time display
- [x] Clean, professional UI

---

## 🚀 Quick Start (Recommended)

### Windows Users
```bash
# Simply run the startup script
start.bat
```

This will:
1. Activate Python virtual environment
2. Start backend API on port 8000
3. Start frontend on port 3000
4. Open both in new terminal windows

### macOS/Linux Users
```bash
chmod +x start.sh
./start.sh
```

---

## 📋 Manual Setup

### Step 1: Backend Setup
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies (if needed)
pip install -r requirements.txt

# Start backend API
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

✅ Backend ready at: `http://localhost:8000`  
✅ API docs at: `http://localhost:8000/api/docs`

### Step 2: Frontend Setup (In a new terminal)
```bash
# Navigate to frontend
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm start
```

✅ Frontend ready at: `http://localhost:3000`

---

## 🧪 Verify Installation

### Run Verification Script
```bash
python verify_setup.py
```

Should show:
- ✓ Directories ready
- ✓ Backend is running
- ✓ Frontend dependencies installed

### Quick Test
1. Open `http://localhost:3000`
2. Upload a chest X-ray image
3. View predictions

---

## 📁 Key Files

### Backend
- `backend/app.py` - Main FastAPI application
- `backend/routes/*.py` - API endpoints
- `backend/services/*.py` - Business logic
- `config/config.py` - Configuration

### Frontend
- `frontend/src/App.js` - Main app component
- `frontend/src/components/` - React components
  - `ImageUpload.js` - Image upload interface
  - `PredictionResults.js` - Results display
  - `Header.js` - App header

---

## 🔧 Configuration

### Backend Ports & Settings
Edit `config/config.py`:
```python
API_HOST = '0.0.0.0'
API_PORT = 8000
UPLOAD_FOLDER = 'backend/uploads'
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
```

### Frontend API Connection
Edit `frontend/.env`:
```
REACT_APP_API_URL=http://localhost:8000/api
```

---

## 📡 API Endpoints

### POST /api/upload
Upload a chest X-ray image
```bash
curl -F "file=@image.jpg" http://localhost:8000/api/upload
```

### POST /api/predict
Get predictions for uploaded image
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"filename": "image.jpg", "confidence_threshold": 0.5}'
```

### GET /api/health
Check API status
```bash
curl http://localhost:8000/api/health
```

---

## 🎨 UI Components

### ImageUpload
- Drag & drop interface
- File validation
- Click to select fallback

### PredictionResults
- Image preview
- Disease list with probabilities
- Risk level indicators
- Processing time display

### LoadingSpinner
- Animated loading state
- User feedback during processing

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Windows: Find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :8000
kill -9 <PID>
```

### npm Dependencies Issue
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Backend Not Responding
- Check Python virtual environment is activated
- Check no errors in backend terminal
- Verify port 8000 is available
- Check `config/config.py` for correct paths

### Frontend Can't Connect to Backend
- Ensure backend is running: `http://localhost:8000/api/health`
- Check `frontend/.env` has correct `REACT_APP_API_URL`
- Browser console may show CORS errors (check backend logs)

---

## 📊 Architecture

```
User Browser (localhost:3000)
         ↓
React Frontend (App.js)
         ↓ (HTTP Requests)
FastAPI Backend (localhost:8000)
         ↓
ML Model (PyTorch)
         ↓
Predictions
         ↓
JSON Response
         ↓
React Components (Display)
```

---

## 🎯 Core Workflow

1. **User uploads image** → `ImageUpload` component
2. **Frontend sends** → `POST /api/upload`
3. **Backend validates** → `image_service.validate_image()`
4. **Frontend sends** → `POST /api/predict`
5. **Backend runs inference** → `prediction_service.predict()`
6. **Frontend receives** → Predictions JSON
7. **Display results** → `PredictionResults` component

---

## 📈 Performance

Expected response times:
- Image upload: ~50-100ms
- Model inference: ~200-500ms (CPU) / ~50-150ms (GPU)
- UI render: ~50-100ms
- **Total**: ~300-700ms

---

## 🔐 Security Notes (For Production)

Currently configured for development with:
- CORS: Open to all origins
- No authentication
- No input rate limiting
- Uploaded files not encrypted

For production, implement:
- ✓ Authentication
- ✓ CORS restrictions
- ✓ Rate limiting
- ✓ Input validation
- ✓ HTTPS
- ✓ File upload restrictions

---

## 📝 Next Steps

### For IPD Submission ✅
- Core functionality complete
- Basic UI showing predictions
- Ready for demo

### Future Enhancements
- [ ] Add disease descriptions and recommendations
- [ ] Implement heatmap visualization
- [ ] Add report generation
- [ ] User authentication
- [ ] Result history/database
- [ ] Batch image processing
- [ ] Advanced analytics

---

## 📚 Documentation

- `FRONTEND_BACKEND_SETUP.md` - Detailed setup guide
- `http://localhost:8000/api/docs` - Interactive API docs
- Frontend components have JSDoc comments

---

## ✨ Ready to Go!

The application is ready for:
- ✅ Local development
- ✅ Testing with sample X-rays
- ✅ IPD submission/demo
- ✅ Training integration (runs in parallel)

**Start now with:** `start.bat` (Windows) or `./start.sh` (macOS/Linux)
