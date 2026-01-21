# Friend's PC Setup Instructions

**RTX 4070 GPU Training Setup - Day 2**

---

## System Requirements

- **OS**: Windows 10/11
- **GPU**: RTX 4070 (16GB VRAM) ✓
- **RAM**: 64GB ✓
- **Storage**: ~50GB free space for dataset + models

---

## Step 1: Install Python 3.13.1 (EXACT VERSION)

Download and install Python 3.13.1 from official website:

- URL: https://www.python.org/downloads/release/python-3131/
- File: **Windows installer (64-bit)**

**Installation Settings:**

- ✓ Add Python 3.13 to PATH
- ✓ Install pip
- ✓ Install for all users (optional)

**Verify Installation:**

```powershell
python --version
# Expected output: Python 3.13.1
```

---

## Step 2: Install Git (if not installed)

Download Git for Windows:

- URL: https://git-scm.com/download/win
- Use default settings during installation

**Verify:**

```powershell
git --version
```

---

## Step 3: Clone Repository

```powershell
# Navigate to desired location (e.g., Documents)
cd C:\Users\YourUsername\Documents

# Clone repository
git clone https://github.com/yourusername/x-lite-chest-xray.git

# Navigate to project
cd x-lite-chest-xray
```

---

## Step 4: Create Virtual Environment

```powershell
# Create virtual environment (must be named .venv)
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Verify Python version in venv
python --version
# Expected: Python 3.13.1
```

---

## Step 5: Install PyTorch with CUDA Support

**CRITICAL**: Install GPU-enabled PyTorch for RTX 4070

```powershell
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify GPU is detected:**

```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Expected output:
# CUDA Available: True
# GPU: NVIDIA GeForce RTX 4070
```

---

## Step 6: Install Project Dependencies

```powershell
# Install all dependencies from requirements.txt
pip install -r requirements.txt

# Verify key packages
pip list | findstr "torch albumentations pandas scikit-learn"
```

---

## Step 7: Prepare Dataset

### Option A: Copy from Your Laptop (RECOMMENDED)

1. **Transfer dataset using external drive or network:**
   - Copy `data/raw/images/` folder (~45GB, 112,120 images)
   - Copy `data/splits/` folder (train.csv, val.csv, test.csv)
   - Copy `data/Data_Entry_2017.csv` metadata file

2. **Directory structure after copy:**
   ```
   x-lite-chest-xray/
   ├── data/
   │   ├── Data_Entry_2017.csv
   │   ├── raw/
   │   │   └── images/        # 112,120 .png files
   │   └── splits/
   │       ├── train.csv      # 78,484 samples
   │       ├── val.csv        # 16,818 samples
   │       └── test.csv       # 16,818 samples
   ```

### Option B: Download Dataset (if needed)

```powershell
python scripts/download_chestxray14.py
```

**Note:** This will take ~2-3 hours depending on internet speed.

---

## Step 8: Validate Setup

**Run validation script:**

```powershell
python scripts/validate_data_splits.py
```

**Expected output:**

```
✓ Test 1: Missing values check
✓ Test 2: Label preservation check
✓ Test 3: Class imbalance check
✓ Test 4: Data leakage check (image-level)
! Test 5: Patient-level leakage detected (6,582 patients overlap) - ACCEPTABLE
✓ Test 6: Image corruption check

5/6 tests passed
```

---

## Step 9: Run Baseline Training

**Start baseline training (all 6 models):**

```powershell
# Make sure .venv is activated
.venv\Scripts\activate

# Run baseline training script
python scripts/train_baseline.py
```

**Training Details:**

- **Models**: 6 hybrid CNN-Transformer variants
- **Dataset**: 20% of training data (~15,696 images)
- **Epochs**: 30 per model (early stopping patience=10)
- **Batch size**: 32
- **Time estimate**: ~4-6 hours total (all 6 models)
- **Checkpoints**: Saved in `ml/models/checkpoints/{model_name}/`
- **Results**: Saved in `experiments/baseline_results.csv`

**Monitor Training:**

- Watch console output for epoch-by-epoch metrics
- Check `ml/models/checkpoints/` for saved models
- Checkpoints saved every improvement in validation AUC

---

## Step 10: Verify Training Results

**After training completes:**

```powershell
# View results
python -c "import pandas as pd; df = pd.read_csv('experiments/baseline_results.csv'); print(df[['model_name', 'best_val_auc', 'best_val_f1', 'num_parameters', 'training_time_minutes']].to_string(index=False))"
```

**Expected outputs:**

- Validation AUC: 0.60-0.75 (baseline without KD)
- F1 Score: 0.20-0.40 (multi-label is challenging)
- Training time: 30-60 minutes per model

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**

```powershell
# Reduce batch size in scripts/train_baseline.py
# Line 249: batch_size=32 → batch_size=16
```

### Issue: "ModuleNotFoundError"

**Solution:**

```powershell
# Verify venv is activated
.venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "No module named 'torch.cuda'"

**Solution:**

```powershell
# Uninstall CPU version
pip uninstall torch torchvision torchaudio

# Reinstall GPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Training very slow

**Check:**

```powershell
# Verify GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU utilization during training
# Open Task Manager > Performance > GPU
```

---

## Quick Reference: Daily Workflow

**Start of day:**

```powershell
cd C:\Users\YourUsername\Documents\x-lite-chest-xray
.venv\Scripts\activate
git pull  # Get latest changes
```

**Run training:**

```powershell
python scripts/train_baseline.py
```

**Check results:**

```powershell
cat experiments\baseline_results.csv
```

**End of day:**

```powershell
git add .
git commit -m "Day 2: Baseline training results"
git push
```

---

## Contact

If any issues, message me with:

1. Error message (full traceback)
2. Command you ran
3. Output of `python --version` and `pip list`

---

## Timeline (7-Day Plan)

- **Day 1 (Jan 21)**: ✓ Setup complete on your laptop
- **Day 2 (Jan 22)**: ← **YOU ARE HERE** - Setup friend's PC + baseline training
- **Day 3 (Jan 23)**: Analyze baseline results, select best model
- **Day 4 (Jan 24)**: Download CheXNet teacher, setup KD training
- **Day 5-6 (Jan 25-26)**: Knowledge distillation experiments (8 configs)
- **Day 7 (Jan 27)**: Final model training + deployment + testing

---

**Good luck! 🚀**
