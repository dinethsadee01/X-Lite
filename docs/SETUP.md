# X-Lite Project Setup Guide

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- At least 50GB free disk space (for dataset)
- Git

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/dinethsadee01/X-Lite.git
cd x-lite-chest-xray
```

### 2. Create Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# For GPU support (CUDA 11.8):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 4. Verify Installation

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 📊 Download Dataset

### Option 1: Automated Download (Recommended)

```bash
# Download metadata only (quick start)
python scripts/download_chestxray14.py --metadata-only

# Download full dataset (~45GB, takes hours)
python scripts/download_chestxray14.py
```

### Option 2: Manual Download

1. Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC
2. Download all 12 image archives (images_001.tar.gz to images_012.tar.gz)
3. Download Data_Entry_2017.csv
4. Extract archives to `data/raw/`
5. Move CSV to `data/`

## 📁 Project Structure Setup

The folder structure should look like:

```
x-lite-chest-xray/
├── data/
│   ├── raw/
│   │   ├── images/               # Extracted X-ray images
│   │   │   ├── 00000001_000.png
│   │   │   ├── 00000001_001.png
│   │   │   └── ...
│   ├── Data_Entry_2017.csv      # Metadata file
│   └── processed/                # Preprocessed data (auto-generated)
├── ml/
│   └── models/
│       └── checkpoints/          # Model weights (auto-generated)
└── ...
```

## 🧪 Test Your Setup

### Run Data Exploration Notebook

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Quick Test Script

Create `test_setup.py`:

```python
import sys
from pathlib import Path

# Test imports
try:
    import torch
    import torchvision
    import pandas as pd
    import albumentations
    from config import Config, DISEASE_LABELS
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test config
Config.create_directories()
print(f"✓ Directories created")

# Test CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test dataset path
if Config.METADATA_CSV.exists():
    print(f"✓ Dataset metadata found: {Config.METADATA_CSV}")
else:
    print(f"⚠ Dataset metadata not found. Run: python scripts/download_chestxray14.py --metadata-only")

print("\n✓ Setup complete! Ready to start training.")
```

Run it:

```bash
python test_setup.py
```

## 🎯 Next Steps

1. **Explore Data**: Run `notebooks/01_data_exploration.ipynb`
2. **Train Teacher Model**: `python scripts/train_teacher.py`
3. **Knowledge Distillation**: `python scripts/distill_student.py`
4. **Start Backend**: `cd backend && python app.py`
5. **Start Frontend**: `cd frontend && npm install && npm start`

## ⚙️ Configuration

Edit `config/config.py` to customize:

- Image size
- Batch size
- Learning rates
- Model architectures
- Data augmentation

## 🐛 Troubleshooting

### CUDA Out of Memory

- Reduce batch size in `config/config.py`
- Use gradient accumulation
- Enable mixed precision training

### Import Errors

```bash
pip install -r requirements.txt --upgrade
```

### Dataset Not Found

- Verify `data/Data_Entry_2017.csv` exists
- Check image path in `data/raw/images/`

## 📚 Documentation

- [API Documentation](docs/API.md)
- [Model Architecture](docs/MODEL.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## 💡 Tips

- Start with a small subset of data for testing
- Use TensorBoard for monitoring training: `tensorboard --logdir logs/tensorboard`
- Enable Weights & Biases for experiment tracking (optional)

## 🤝 Getting Help

- Create an issue on GitHub
- Check project documentation in `docs/`
- Review example notebooks in `notebooks/`

---

**Ready to build X-Lite! 🚀**
