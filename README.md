# X-Lite — Lightweight Hybrid CNN-Attention for Chest X-Ray Classification

> A CPU-deployable multi-label chest X-ray classification system detecting 14 thoracic diseases using a hybrid EfficientNet-B0 + Performer Attention architecture with Grad-CAM explainability.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)
![AUC](https://img.shields.io/badge/Macro--AUC-0.866-brightgreen?style=flat)
![Inference](https://img.shields.io/badge/CPU%20Inference-58.5ms-blue?style=flat)
![Params](https://img.shields.io/badge/Parameters-4.06M-orange?style=flat)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat)

---

## Overview

**X-Lite** addresses the challenge of deploying accurate chest X-ray AI in resource-constrained healthcare settings — district hospitals, clinics, and medical training institutions — where GPU servers and cloud subscriptions are not feasible.

Most existing solutions require GPU hardware or cloud connectivity. X-Lite achieves a **macro-AUC of 0.866** on the ChestX-ray14 benchmark, outperforming CheXNet (0.841 AUC), while running entirely on CPU with **58.5ms inference latency** and only **4.06M parameters**.

### Key Contributions

- **Novel Per-class Alpha Focal Loss** — per-disease alpha weights computed as `1/frequency`, clamped to [0.5, 0.95], addressing the severe 200:1 class imbalance in ChestX-ray14
- **Hybrid CNN-Attention Architecture** — EfficientNet-B0 backbone + Performer attention (FAVOR+) for O(N) linear complexity, enabling CPU-first deployment
- **Per-class Threshold Optimisation** — post-training threshold tuning per disease (mean 0.237, range 0.10–0.35), improving macro-F1 by 46% over the default 0.5 threshold
- **End-to-end Clinical Web Application** — upload → predict → Grad-CAM → PDF report in under 2 seconds on CPU

---

## Results

### Final Model Performance (EfficientNet-B0 + Performer, Test Set — 21,845 images)

| Disease | AUC-ROC | Recall | Precision |
|---|---|---|---|
| Hernia | **0.999** | 0.813 | 0.481 |
| Cardiomegaly | **0.921** | 0.497 | 0.432 |
| Pneumothorax | **0.911** | 0.417 | 0.494 |
| Effusion | **0.900** | 0.627 | 0.493 |
| Emphysema | **0.898** | 0.512 | 0.476 |
| Edema | **0.893** | 0.567 | 0.384 |
| Fibrosis | **0.867** | 0.370 | 0.302 |
| Mass | **0.862** | 0.395 | 0.290 |
| Consolidation | **0.849** | 0.467 | 0.218 |
| Pleural Thickening | **0.838** | 0.363 | 0.237 |
| Atelectasis | **0.818** | 0.467 | 0.345 |
| Nodule | **0.790** | 0.324 | 0.188 |
| Pneumonia | **0.752** | 0.292 | 0.138 |
| Infiltration | **0.734** | 0.548 | 0.332 |
| **Macro Average** | **0.866** | **0.476** | **0.347** |

### Benchmark Comparison

| Model | Macro-AUC | Parameters | Hardware |
|---|---|---|---|
| CheXNet (DenseNet-121) | 0.841 | 7.5M | GPU required |
| **X-Lite (EfficientNet-B0 + Performer)** | **0.866** | **4.06M** | **CPU ✅** |
| LungMaxViT | 0.932 | Much larger | GPU required |

### Deployment Metrics

| Metric | Value |
|---|---|
| Model Parameters | 4.06M |
| Inference Time (CPU) | 58.5ms |
| End-to-End Response | < 2 seconds |
| Hamming Loss | 0.090 |
| Macro-F1 (optimised thresholds) | 0.333 |
| Macro-F1 (default threshold 0.5) | 0.046 |

---

## Architecture

```
Input X-ray (224×224×3)
        │
   CLAHE Preprocessing
        │
   EfficientNet-B0 Backbone
   (Pretrained, ImageNet)
        │
   Feature Maps (B × 1280 × 7 × 7)
        │
   Reshape → Sequence (B × 49 × 1280)
        │
   Performer Attention (FAVOR+)
   Linear O(N) complexity
        │
   Global Average Pooling
        │
   FC Classification Head
   (14 sigmoid outputs)
        │
   Per-class Threshold Application
   (mean 0.237, range 0.10–0.35)
        │
   14 Disease Probabilities + Grad-CAM
```

---

## Web Application

The full-stack clinical web application enables real-time chest X-ray analysis with explainability and PDF reporting.

### Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Material-UI v5 |
| Backend | FastAPI, Python 3.10+ |
| ML Inference | PyTorch, TorchVision |
| Explainability | Grad-CAM (last conv layer hooks) |
| Reports | ReportLab PDF |
| Auth | JWT tokens |

### REST API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | System status check |
| `/api/upload` | POST | Upload chest X-ray image |
| `/api/predict` | POST | Run inference + Grad-CAM |
| `/api/report/generate` | POST | Generate PDF clinical report |
| `/api/report/download/{filename}` | GET | Download PDF report |
| `/api/auth/login` | POST | User authentication |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- 4GB+ RAM (no GPU required)

### Installation

```bash
# Clone repository
git clone https://github.com/dinethsadee01/X-Lite.git
cd X-Lite

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Environment Setup

```bash
# Copy the example env file
cp .env.example .env   # Linux/macOS
Copy-Item .env.example .env   # Windows PowerShell
```

Edit `.env` with your values:

```env
MONGODB_URL=mongodb+srv://<username>:<password>@<cluster>/?retryWrites=true&w=majority
MONGODB_DB_NAME=xlite_db
SECRET_KEY=your_long_random_secret_key
REACT_APP_API_URL=http://localhost:8000/api
```

### Run the Application

```bash
# Option 1 — Quick start scripts
./start.sh          # Linux/macOS
start.bat           # Windows

# Option 2 — Manual (two terminals)
# Terminal 1: Backend
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend && npm start
```

Open **http://localhost:3000** in your browser.

**Demo login** (no database required):
- Username: `doctor`
- Password: `password123`

### API Documentation

Swagger UI available at: **http://localhost:8000/api/docs**

---

## Training Pipeline

### Train the Final Model

```bash
python scripts/train_improved.py
```

This runs the refined training pipeline with:
- Per-class Alpha Focal Loss (`γ=2.0`, per-disease alpha)
- Sqrt-dampened WeightedRandomSampler
- CosineAnnealingWarmRestarts scheduler
- 2-epoch LR warmup
- Per-class threshold optimisation on validation set

### Training Configuration

| Setting | Value |
|---|---|
| Loss Function | Per-class Alpha Focal Loss |
| Optimizer | AdamW (weight_decay=1e-5) |
| Learning Rate | 3e-5 with cosine schedule |
| Batch Size | 128 |
| Max Epochs | 70 (early stopping patience=5) |
| Hardware | NVIDIA RTX 4070 Ti SUPER 16GB |

---

## Project Structure

```
X-Lite/
├── config/                  # Hyperparameters, disease mappings, thresholds
├── ml/
│   ├── data/                # DataLoader, CLAHE preprocessing, augmentation
│   ├── models/              # EfficientNet-B0 + Performer, DenseNet-121 teacher
│   └── training/            # Training loops, KD pipeline, evaluation
├── backend/
│   ├── app.py               # FastAPI application entry point
│   ├── routes/              # API route definitions
│   └── services/            # PredictionService, GradCAMService, ReportService
├── frontend/                # React SPA (Material-UI)
├── scripts/                 # train_improved.py and utility scripts
├── data/                    # Dataset splits, CLAHE cache
├── requirements.txt
├── setup.py
├── start.sh                 # Linux/macOS quick-start
└── start.bat                # Windows quick-start
```

---

## Dataset

**ChestX-ray14** (NIH Clinical Center)
- 112,120 frontal-view chest X-ray images
- 30,805 unique patients
- 14 thoracic disease labels (multi-label, NLP-extracted)
- Severe class imbalance: No Finding 60.4% → Hernia 0.14%

The dataset is not included in this repository. Download from the [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC).

---

## Known Limitations

- Trained and evaluated on ChestX-ray14 only — external validation on CheXpert or MIMIC-CXR not yet performed
- Labels are NLP-extracted from radiology reports (~10% label noise)
- Knowledge Distillation (DenseNet-121 → EfficientNet-B0 Performer) degraded performance due to architecture mismatch — documented as a negative result; direct training with Per-class Alpha Focal Loss was used for the final model
- Not validated for clinical deployment — requires radiologist confirmation for all outputs

---

## Acknowledgements

- [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) — ChestX-ray14 dataset
- [CheXNet (Rajpurkar et al., 2017)](https://arxiv.org/abs/1711.05225) — teacher model inspiration
- [Performer (Choromanski et al., 2021)](https://arxiv.org/abs/2009.14794) — FAVOR+ linear attention
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision) — pretrained teacher weights

---

## Citation

If you use this work, please cite:

```
Edirisinghe, D.S. (2026). X-Lite: Lightweight Hybrid CNN-Attention Framework
for Multi-Label Chest X-Ray Classification in Resource-Constrained Environments.
BSc Computer Science Final Year Project, University of Westminster.
```

---

## License

This project is released for **academic and research purposes only**.  
Clinical use requires radiologist validation. Not approved for medical diagnosis.

---

*Built as a BSc Computer Science Final Year Project — University of Westminster, 2026*
