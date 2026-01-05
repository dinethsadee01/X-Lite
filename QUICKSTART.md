# 🚀 Quick Start Checklist

Before you begin development, complete these steps:

## ✅ Initial Setup (One-time)

### 1. Environment Setup

```powershell
# Navigate to project
cd c:\Users\Asus\x-lite-chest-xray

# Verify Python installation
python --version  # Should be 3.8+

# Dependencies already installed ✓
# (You completed this step!)
```

### 2. Initialize Git (If not done)

```powershell
# Initialize repository
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: X-Lite project structure"
```

### 3. Create GitHub Repository

```powershell
# Go to https://github.com/dinethsadee01
# Click "New Repository"
# Name: X-Lite
# Description: Lightweight Hybrid CNN-Transformer for Chest X-Ray Classification
# Public or Private: Your choice
# DO NOT initialize with README (we have one)

# Link local repo to GitHub
git remote add origin https://github.com/dinethsadee01/X-Lite.git
git branch -M main
git push -u origin main
```

### 4. Set Up Google Drive Structure

```
Create folders in Google Drive:
My Drive/
└── X-Lite/
    ├── data/              # Upload dataset here
    ├── checkpoints/       # Model weights saved here
    └── results/           # Training results
```

---

## 📊 Development Workflow

### VS Code (Local) - Daily Work

#### For Code Development:

```powershell
# 1. Open project in VS Code
code .

# 2. Work on files (examples):
code ml/models/teacher_model.py
code backend/app.py
code frontend/src/App.jsx

# 3. Test locally (small data, CPU)
jupyter notebook notebooks/local/00_quick_start.ipynb

# 4. Commit changes
git add .
git commit -m "Add teacher model architecture"
git push origin main
```

### Google Colab - For Training

#### For GPU Training:

1. **Open Colab**: https://colab.research.google.com/
2. **Upload Notebook**:
   - File → Upload → `notebooks/colab/00_colab_setup.ipynb`
   - Or connect to GitHub (easier for updates)
3. **Run Setup**: Execute all cells in `00_colab_setup.ipynb`
4. **Start Training**: Open next notebook (`01_train_teacher.ipynb`)

---

## 🎯 Your First Tasks

### This Week:

- [x] ✅ Install dependencies (DONE!)
- [ ] 📝 Initialize Git repository
- [ ] 🌐 Create GitHub repository and push code
- [ ] ☁️ Set up Google Drive folders
- [ ] 📓 Run `notebooks/local/00_quick_start.ipynb`
- [ ] 🔬 Run `notebooks/colab/00_colab_setup.ipynb` (in Colab)

### Next Week:

- [ ] 📊 Download ChestX-ray14 dataset (metadata)
- [ ] 🔍 Explore data with visualization
- [ ] 🏗️ Implement teacher model (`ml/models/teacher_model.py`)
- [ ] 🎓 Write teacher training loop (`ml/training/teacher_trainer.py`)
- [ ] 📝 Test teacher model locally (small dataset)

### Week 3+:

- [ ] 🚀 Train teacher model in Colab (full dataset, GPU)
- [ ] 💾 Download teacher checkpoint
- [ ] 🎯 Implement student models
- [ ] 🔥 Implement knowledge distillation
- [ ] 📈 Hyperparameter optimization

---

## 📂 Important File Locations

**Configuration:**

- `config/config.py` - All hyperparameters
- `config/disease_labels.py` - Disease classes

**Data:**

- `ml/data/loader.py` - Dataset loader
- `ml/data/preprocessing.py` - Image transforms
- `ml/data/augmentation.py` - Data augmentation

**Models:** (To implement)

- `ml/models/teacher_model.py`
- `ml/models/student_model.py`

**Backend:**

- `backend/app.py` - FastAPI application
- `backend/routes/` - API endpoints

**Documentation:**

- `docs/SETUP.md` - Detailed setup guide
- `docs/WORKFLOW.md` - VS Code + Colab workflow
- `notebooks/README.md` - Notebook usage guide

---

## 🆘 Quick Commands Reference

### Git Commands

```bash
git status                    # Check status
git add .                     # Stage all changes
git commit -m "message"       # Commit changes
git push origin main          # Push to GitHub
git pull origin main          # Pull latest changes
```

### Python/Jupyter

```bash
jupyter notebook              # Start Jupyter
python backend/app.py         # Run backend
python scripts/train_teacher.py  # Train teacher (local)
```

### Package Management

```bash
pip install -r requirements.txt   # Install dependencies
pip list                          # List installed packages
pip freeze > requirements.txt     # Update requirements
```

---

## 💡 Remember

1. **Code in VS Code**, train in Colab
2. **Commit often** to GitHub
3. **Save checkpoints** to Google Drive
4. **Test locally first** before Colab
5. **Document experiments** as you go

---

## 📞 Need Help?

Check these resources:

- `GETTING_STARTED.md` - Comprehensive guide
- `docs/SETUP.md` - Installation details
- `docs/WORKFLOW.md` - Development workflow
- `notebooks/README.md` - Notebook usage

---

**Ready to build X-Lite! 🎉**

**Next Step**: Run `notebooks/local/00_quick_start.ipynb` to verify everything works!
