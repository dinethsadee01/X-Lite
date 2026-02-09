# X-Lite High-Level System Architecture

## Overview

X-Lite is a full-stack web application for automated chest X-ray analysis using lightweight hybrid CNN-Transformer models trained via knowledge distillation. The system follows a layered architecture pattern with clear separation of concerns.

---

## Architecture Layers

### 🔵 Client Layer
**Purpose**: User interaction and presentation

**Components**:
- **End User**: Medical professionals, radiologists, clinicians
- **Web Browser**: Modern browsers (Chrome, Firefox, Safari, Edge)

**Responsibilities**:
- Render user interface
- Handle user interactions
- Display diagnostic results
- Download PDF reports

---

### 🟣 Frontend Layer (React)
**Purpose**: Client-side application logic and UI rendering

**Technology Stack**:
- **React 18**: Component-based UI framework
- **Material-UI (MUI)**: Design system and pre-built components
- **React Hooks**: State management (useState, useEffect)
- **Fetch API**: HTTP requests to backend

**Components**:

1. **Upload Component**
   - Drag-and-drop file upload
   - File type validation
   - Image preview
   - Upload progress indicator

2. **Results Component**
   - X-ray image display
   - Grad-CAM heatmap visualization
   - Disease probability list
   - Risk level indicators
   - Positive findings highlights

3. **Report Component**
   - PDF download button
   - Report generation trigger

4. **Loading Component**
   - Animated spinner
   - Progress messages
   - Upload/processing status

**State Management**:
- `uploadedImage`: Current X-ray image data
- `predictions`: Model inference results
- `loading`: Loading state flag
- `error`: Error message state

**Communication**:
- REST API calls to FastAPI backend
- JSON request/response format
- FormData for file uploads

---

### 🟠 API Gateway Layer
**Purpose**: HTTP request routing and middleware

**Technology Stack**:
- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server (production)
- **Pydantic**: Request/response validation
- **Python 3.10+**: Runtime environment

**Components**:

1. **FastAPI Server**
   - Runs on port 8000
   - Async request handling
   - Automatic OpenAPI documentation
   - Request validation

2. **CORS Middleware**
   - Cross-Origin Resource Sharing
   - Allows frontend (port 3000) to access API
   - Configure allowed origins, methods, headers

3. **API Router**
   - Route registration and management
   - Path parameters and query parameters
   - Request body parsing

**API Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/upload` | POST | Upload X-ray image |
| `/api/predict` | POST | Run model inference |
| `/api/report` | POST | Generate PDF report |
| `/api/health` | GET | Health check |

---

### 🟢 Service Layer
**Purpose**: Business logic and core functionality

**Components**:

#### 1. Image Service
**File**: `backend/services/image_service.py`

**Responsibilities**:
- **Validation**: Check file type, size, format
- **Storage**: Save uploaded images with unique filenames
- **Preprocessing**: CLAHE enhancement, resizing, normalization
- **Caching**: Lookup preprocessed images in cache

**Key Methods**:
```python
validate_image(file) -> bool
save_image(file) -> str  # Returns filename
load_and_preprocess(filename) -> Tensor
apply_clahe(image) -> np.ndarray
```

#### 2. Prediction Service
**File**: `backend/services/prediction_service.py`

**Responsibilities**:
- **Model Loading**: Lazy initialization of ML model
- **Inference**: Forward pass through neural network
- **Post-Processing**: Apply threshold, compute risk levels
- **Explainability**: Generate Grad-CAM heatmaps

**Key Methods**:
```python
predict(image_tensor, return_heatmap=True) -> dict
generate_gradcam(image_tensor, target_class) -> np.ndarray
assign_risk_level(probability) -> str
```

#### 3. Report Service
**File**: `backend/services/report_service.py`

**Responsibilities**:
- **PDF Generation**: Create diagnostic report
- **Formatting**: Layout images, tables, text
- **Metadata**: Add timestamps, patient info
- **Storage**: Save PDFs to reports directory

**Key Methods**:
```python
generate_report(filename, predictions) -> bytes
format_findings(predictions) -> str
add_images(pdf, xray_image, heatmap) -> None
```

---

### 🔴 ML/AI Layer
**Purpose**: Deep learning model inference and explainability

**Components**:

#### 1. Student Model
**Architecture**: Hybrid CNN-Transformer

**Variants**:
- **EfficientNet-B0 + MHSA**: Balanced efficiency and accuracy
- **ConvNeXt-Tiny + Performer**: Modern CNN with linear attention
- **MobileNetV3 + Linear Attention**: Maximum efficiency

**Structure**:
```
Input (224x224x3)
    ↓
CNN Backbone (Feature Extraction)
    ↓
Attention Module (Global Context)
    ↓
Classification Head (14-class output)
    ↓
Sigmoid Activation
    ↓
Disease Probabilities [p1, p2, ..., p14]
```

**Model Characteristics**:
- Parameters: 5-25M (vs. 7M for DenseNet121)
- FLOPs: 0.5-2G (vs. 3G for DenseNet121)
- Inference Time: 50-200ms on CPU

#### 2. Teacher Model
**Architecture**: DenseNet121 (CheXNet)

**Purpose**:
- Used during training for knowledge distillation
- Not loaded during inference (only student used)
- Pre-trained on ChestX-ray14 dataset

#### 3. Grad-CAM
**Purpose**: Visual explainability

**Process**:
1. Forward pass to get predictions
2. Backward pass to compute gradients
3. Weight feature maps by gradients
4. Generate heatmap
5. Overlay on original image

**Output**: Highlights regions influencing predictions

---

### 🟡 Data Layer
**Purpose**: Persistent storage and data management

**Components**:

#### 1. File Storage
**Location**: `backend/uploads/`

**Structure**:
```
uploads/
├── [timestamp]_[uuid].jpg     # Uploaded X-rays
└── reports/
    └── report_[timestamp].pdf  # Generated reports
```

**File Naming Convention**:
- Format: `YYYYMMDD_HHMMSS_[12-char-uuid].[ext]`
- Example: `20260209_143025_a1b2c3d4e5f6.jpg`

#### 2. Model Checkpoints
**Location**: `ml/models/checkpoints/`

**Structure**:
```
checkpoints/
├── kd/                         # Knowledge distillation models
│   ├── convnext_tiny_mhsa/
│   │   └── best_model.pth
│   ├── efficientnet_b0_mhsa/
│   └── mobilenetv3_linear/
└── baseline/                   # Baseline models
    └── densenet121/
```

**Checkpoint Contents**:
- Model state dict (weights and biases)
- Optimizer state (for resuming training)
- Training metadata (epoch, loss, metrics)

#### 3. CLAHE Cache
**Location**: `data/clahe_cache/`

**Purpose**:
- Pre-computed CLAHE-enhanced images
- Speeds up inference by ~100ms per image
- Reduces redundant preprocessing

**Format**: PNG files with same naming as original dataset

---

### 🔷 Training Pipeline (Offline)
**Purpose**: Model development and training

**Components**:

#### 1. Dataset
**Name**: ChestX-ray14 (NIH)

**Specifications**:
- **Size**: ~112,000 frontal-view chest X-rays
- **Format**: 1024x1024 PNG images
- **Labels**: 14 disease classes (multi-label)
- **Source**: NIH Clinical Center

**Data Splits**:
- Training: 70% (~78,400 images)
- Validation: 15% (~16,800 images)
- Test: 15% (~16,800 images)

**Split Strategy**: Stratified by disease prevalence

#### 2. Data Loader
**File**: `ml/data/loader.py`

**Features**:
- Batch loading with PyTorch DataLoader
- Weighted random sampling for class balance
- Multi-worker data loading
- Pin memory for GPU efficiency

#### 3. KD Trainer
**File**: `ml/training/kd_trainer.py`

**Training Process**:
1. Load teacher and student models
2. Forward pass through both models
3. Compute distillation loss + task loss
4. Backpropagation and parameter update
5. Validation and checkpointing

**Loss Function**:
```
L_total = α * L_distillation + (1 - α) * L_task

L_distillation = KL(σ(z_student/T), σ(z_teacher/T))
L_task = BCE(σ(z_student), y_true)
```

**Hyperparameters**:
- Temperature (T): 4.0
- Alpha (α): 0.7
- Learning rate: 1e-4
- Weight decay: 1e-5
- Batch size: 32

#### 4. Metrics & Evaluation
**Metrics Tracked**:
- AUC-ROC (macro-averaged)
- F1 Score (macro-averaged)
- PR-AUC (macro-averaged)
- Precision (per-class and macro)
- Recall (per-class and macro)

**Evaluation**: Computed on validation set every epoch

---

### ⚙️ Configuration Layer
**Purpose**: Centralized settings and constants

**Files**:
- `config/config.py`: Application settings
- `config/disease_labels.py`: Disease class definitions

**Configuration Items**:

1. **Disease Labels**
   ```python
   DISEASE_LABELS = [
       "Atelectasis", "Cardiomegaly", "Effusion", 
       "Infiltration", "Mass", "Nodule", "Pneumonia", 
       "Pneumothorax", "Consolidation", "Edema", 
       "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"
   ]
   ```

2. **Application Settings**
   ```python
   ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.dcm'}
   MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
   UPLOAD_DIR = 'backend/uploads/'
   MODEL_PATH = 'ml/models/checkpoints/kd/convnext_tiny_mhsa/best_model.pth'
   ```

3. **Path Configurations**
   ```python
   DATA_DIR = 'data/'
   CLAHE_CACHE_DIR = 'data/clahe_cache/'
   CHECKPOINT_DIR = 'ml/models/checkpoints/'
   ```

---

## Data Flow

### 1. Upload Flow
```
User → Browser → React Upload Component → 
POST /api/upload → Image Service → File Storage → 
Response {filename} → React State
```

### 2. Prediction Flow
```
React → POST /api/predict → Prediction Service →
Image Service (preprocess) → Student Model (inference) →
Post-Processing → Grad-CAM → Response → React Results
```

### 3. Report Flow
```
React → POST /api/report → Report Service →
Image Service (load image) → PDF Generator →
File Storage → Response (PDF bytes) → Browser Download
```

---

## Technology Stack Summary

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.x | UI framework |
| Material-UI | 5.x | Component library |
| JavaScript | ES6+ | Programming language |
| npm | 9.x | Package manager |

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Programming language |
| FastAPI | 0.104+ | Web framework |
| Uvicorn | 0.24+ | ASGI server |
| Pydantic | 2.x | Data validation |

### ML/AI
| Technology | Version | Purpose |
|------------|---------|---------|
| PyTorch | 2.1+ | Deep learning |
| timm | 0.9+ | Model architectures |
| torchvision | 0.16+ | Vision utilities |
| numpy | 1.24+ | Numerical computing |
| Pillow | 10.x | Image processing |
| opencv-python | 4.8+ | Computer vision |

### Data & Utilities
| Technology | Version | Purpose |
|------------|---------|---------|
| pandas | 2.1+ | Data manipulation |
| scikit-learn | 1.3+ | Metrics |
| albumentations | 1.3+ | Augmentation |
| reportlab | 4.0+ | PDF generation |

---

## Deployment Architecture

### Development
```
React Dev Server (Port 3000) ← → FastAPI Dev Server (Port 8000)
```

### Production
```
Nginx (Port 80/443)
    ├── /           → React Build (Static Files)
    └── /api/*      → FastAPI (Reverse Proxy to Port 8000)
```

### Docker Deployment
```yaml
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./backend/uploads:/app/backend/uploads
      - ./ml/models/checkpoints:/app/ml/models/checkpoints
```

---

## Security Considerations

### 1. File Upload Security
- File type validation (whitelist approach)
- File size limits (max 10MB)
- Virus scanning (optional)
- Unique filename generation (prevent overwrites)

### 2. API Security
- CORS configuration (restrict origins)
- Rate limiting (prevent abuse)
- Input validation (Pydantic models)
- Error handling (don't leak sensitive info)

### 3. Data Privacy
- No patient identifiers stored
- Image filenames randomized
- Reports stored temporarily
- HIPAA compliance considerations

---

## Scalability Considerations

### 1. Horizontal Scaling
- **Frontend**: Serve static files from CDN
- **Backend**: Load balance multiple FastAPI instances
- **ML Model**: Model serving with TorchServe or TensorFlow Serving

### 2. Caching Strategies
- **CLAHE Cache**: Pre-computed preprocessed images
- **Model Cache**: Keep model in memory (lazy loading)
- **Redis**: Cache API responses (future enhancement)

### 3. Asynchronous Processing
- **Background Jobs**: Queue inference requests with Celery
- **WebSockets**: Real-time prediction updates
- **Batch Processing**: Process multiple images in parallel

---

## Monitoring & Logging

### 1. Application Logs
- Request/response logging
- Error tracking and stack traces
- Performance metrics (inference time)

### 2. Model Monitoring
- Prediction distribution monitoring
- Confidence score analysis
- Grad-CAM generation success rate

### 3. Health Checks
- `/api/health` endpoint
- Model loaded status
- Disk space monitoring

---

## System Requirements

### Development Environment
- **CPU**: 4+ cores
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB (dataset + models)
- **OS**: Windows 10+, macOS 11+, Ubuntu 20.04+

### Production Environment
- **CPU**: 8+ cores (inference without GPU)
- **RAM**: 16GB minimum
- **Storage**: 100GB (images, models, reports)
- **Network**: 100Mbps+

---

## Performance Metrics

### Inference Performance
- **Model Loading**: One-time ~2-5 seconds
- **Preprocessing**: ~100ms per image
- **Inference**: ~50-200ms per image (CPU)
- **Grad-CAM**: ~50ms per image
- **Total Response Time**: <1 second

### Web Application
- **Page Load**: <2 seconds
- **Upload Time**: Network dependent (~100KB/s for X-ray)
- **Report Generation**: ~500ms

---

## Future Enhancements

### 1. Architecture Improvements
- Microservices architecture (separate services)
- Message queue for async processing (RabbitMQ/Kafka)
- Database for patient records (PostgreSQL)
- Object storage for images (S3/MinIO)

### 2. ML Improvements
- Ensemble models for better accuracy
- Uncertainty quantification (Bayesian methods)
- Active learning for continuous improvement
- Multi-modal inputs (patient metadata)

### 3. Features
- User authentication and authorization
- Patient management system
- PACS integration (DICOM support)
- Mobile application (React Native)
- Real-time collaboration (WebRTC)

---

## References

- FastAPI Documentation: https://fastapi.tiangolo.com/
- React Documentation: https://react.dev/
- PyTorch Documentation: https://pytorch.org/docs/
- Material-UI: https://mui.com/
- ChestX-ray14 Dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC
