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
x-lite-chest-xray/
├── config/              # Configuration files
├── ml/                  # Machine learning pipeline
│   ├── data/           # Data loading & preprocessing
│   ├── models/         # Model architectures
│   ├── training/       # Training & distillation
│   ├── inference/      # Prediction & explainability
│   └── evaluation/     # Metrics & validation
├── backend/            # Flask/FastAPI backend
├── frontend/           # React frontend
├── notebooks/          # Jupyter notebooks for experiments
├── scripts/            # Utility scripts
├── tests/              # Unit & integration tests
└── docs/               # Documentation
```

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/dinethsadee01/X-Lite.git
cd x-lite-chest-xray

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📈 Training Pipeline

### 1. Download Dataset

```bash
python scripts/download_chestxray14.py
```

### 2. Train Teacher Model

```bash
python scripts/train_teacher.py --config config/teacher_config.yaml
```

### 3. Knowledge Distillation

```bash
python scripts/distill_student.py --config config/student_config.yaml
```

### 4. Evaluation

```bash
python scripts/evaluate.py --model checkpoints/student_best.pth
```

## 🌐 Web Application

### Backend

```bash
cd backend
python app.py
```

### Frontend

```bash
cd frontend
npm install
npm start
```

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

**Status**: 🚧 In Development (Final Year Project 2026)
