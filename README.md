# X-Lite: Lightweight Hybrid CNN-Transformer for Chest X-Ray Classification

A lightweight hybrid CNN-Transformer framework for multi-label chest X-ray classification via knowledge distillation, designed for resource-constrained clinical environments.

## 🎯 Project Overview

**X-Lite** addresses the challenge of deploying accurate deep learning models for chest X-ray diagnosis in resource-limited settings by combining:

- **Hybrid CNN-Transformer Architecture**: Leveraging both local feature extraction and global context
- **Knowledge Distillation**: Transferring knowledge from a high-performance teacher to efficient student models
- **Multi-Label Classification**: Simultaneous detection of 14 thoracic diseases
- **Web Application**: User-friendly interface for clinical deployment

## 📊 Dataset

**ChestX-ray14** (NIH Clinical Center)

- ~112,000 frontal-view X-ray images
- 14 disease labels: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia
- Multi-label annotations (images may have multiple findings)

## 🏗️ Architecture

### Teacher Model

- **Backbone**: DenseNet121 (CheXNet-inspired)
- **Purpose**: High-performance reference model for knowledge transfer

### Student Models (Experimental)

Lightweight architectures combining efficient CNNs with transformer attention:

- **CNN Backbones**: EfficientNet-B0, ConvNeXt-Tiny, MobileNetV3-Large, etc.
- **Attention Modules**: Multi-Head Self-Attention (MHSA), Performer, Linear Attention
- **Goal**: Maintain accuracy while reducing parameters and FLOPs

## 🚀 Features

- ✅ One-click chest X-ray upload
- ✅ Multi-label disease prediction with confidence scores
- ✅ Grad-CAM visualization for model explainability
- ✅ PDF report generation
- ✅ CPU-optimized inference for deployment
- ✅ RESTful API for integration

## 📁 Project Structure

```
X-Lite/
├── config/              # Configuration files (Python)
├── ml/                  # Machine learning pipeline
│   ├── data/           # Data loading & preprocessing
│   ├── models/          # Model architectures (student & teacher)
│   └── training/        # Training & knowledge distillation
├── backend/            # FastAPI backend server
│   ├── app.py         # Main FastAPI application
│   ├── routes/        # API endpoints
│   └── services/      # Business logic
├── frontend/           # React frontend
├── scripts/            # Utility scripts for training, eval, visualization
├── data/              # Dataset folder (images, splits, cache)
├── experiments/       # Logs, results, checkpoints
└── docs/              # Documentation
```

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/dinethsadee01/X-Lite.git
cd X-Lite

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Web App Setup (Clone-Ready)

### 1. Backend Environment File

Copy `.env.example` to `.env`, then update values as needed:

```bash
cp .env.example .env
# Windows PowerShell:
Copy-Item .env.example .env
```

Or create a `.env` file in the project root:

```env
MONGODB_URL=mongodb+srv://<username>:<password>@<cluster-url>/?retryWrites=true&w=majority
MONGODB_DB_NAME=xlite_db
SECRET_KEY=replace_with_a_long_random_secret
REACT_APP_API_URL=http://localhost:8000/api
```

### 2. Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 3. Start the App

```bash
# Terminal 1 (backend)
uvicorn backend.app:app --reload

# Terminal 2 (frontend)
cd frontend
npm start
```

### 4. Demo Login (DB-Bypass Mode)

Current repository state supports demo login flow with:

- Username: `doctor`
- Password: `password123`

This allows core flow testing:

- Upload chest X-ray
- Run prediction
- View result page
- Download PDF report

## ⚡ Quick Start

### Run the Web Application

```bash
# Terminal 1: Start the backend API
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start the frontend (in a new terminal)
cd frontend
npm install
npm start
```

Then open `http://localhost:3000` in your browser to upload and analyze chest X-rays.

### Current Model Entry Point

The repository snapshot includes the improved training entry point below. The older verification helper is not part of this checkout.

## 📈 Training Pipeline

### 1. Dataset Preparation

Dataset download and preprocessing helpers are not included in this snapshot. Use the prebuilt data artifacts in `data/` or your own preparation workflow.

### 2. Train the Current Improved Model

```bash
python scripts/train_improved.py
```

### 3. Archived Script References

The following commands were used in earlier project iterations and are not present in this repository snapshot:

- `python scripts/verify_predictions.py`
- `python scripts/download_chestxray14.py`
- `python scripts/precompute_clahe.py`
- `python scripts/train_baseline.py --student_model efficientnet_b0_performer`
- `python scripts/train_kd_with_xrv.py --student_model efficientnet_b0_performer`
- `python scripts/evaluate_test_set.py --student_model efficientnet_b0_performer`
- `python scripts/cross_validation_analysis.py`
- `python scripts/generate_kd_visualizations.py`

Use `python scripts/train_improved.py` for the currently available training entry point.

## 🌐 Web Application

### Backend (FastAPI)

```bash
# Option 1: Using uvicorn directly
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

# Option 2: Running app.py directly (uses built-in uvicorn)
python backend/app.py
```

The API will be available at:
- Main endpoint: `http://localhost:8000`
- API docs (Swagger): `http://localhost:8000/api/docs`
- Health check: `http://localhost:8000/api/health`

### Frontend (React)

```bash
cd frontend
npm install
npm start
```

The frontend will open at `http://localhost:3000` and automatically proxy API calls to `http://localhost:8000`

## 📊 Evaluation Metrics

- **AUC-ROC**: Area Under ROC Curve (primary metric)
- **Precision, Recall, F1-Score**: Per-disease and macro-averaged
- **Model Size**: Parameters and file size (MB)
- **Inference Time**: CPU inference latency (ms)
- **FLOPs**: Computational complexity

## 🎓 Research Goals

1. Achieve competitive AUC-ROC (>0.75 per disease) with <50% model size
2. CPU inference time <500ms per image
3. Maintain interpretability through attention visualization
4. Enable deployment in resource-limited clinical settings

## 📄 License

This project is for academic research purposes.

## 👥 Contributors

- Dineth Sadee (Computer Science Final Year Project)

## 🙏 Acknowledgments

- NIH Clinical Center for ChestX-ray14 dataset
- CheXNet paper for teacher model inspiration
- Open-source deep learning community

---

**Status**: ✅ Phase 4 Complete (February 2026)

**Final Model**: EfficientNet-B0 with Performer Attention + Knowledge Distillation
- **Test AUC**: 0.8390 (on 16,818 unseen images)
- **Validation AUC**: 0.8446
- **Baseline AUC**: 0.8351 (no distillation)
- **Training Efficiency**: 3× faster convergence (15 vs 50 epochs)

**Deliverables**:
- ✅ Trained student model with knowledge distillation
- ✅ Test set evaluation on unseen data
- ✅ Cross-validation analysis
- ✅ Calibration curves for reliability assessment
- ✅ Web application (FastAPI + React)
- ✅ Complete documentation and visualizations
