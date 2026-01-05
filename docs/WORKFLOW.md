# Hybrid Development Workflow: VS Code + Google Colab

## 🎯 Strategy Overview

**Your approach is perfect!** Here's why this hybrid workflow is ideal:

### ✅ VS Code (Local Development)

**Best for:**

- 💻 **All Code Development**: Models, data loaders, utilities
- 🏗️ **Project Structure**: Organizing files and modules
- 🔧 **Configuration**: Managing config files
- 🌐 **Backend Development**: FastAPI/Flask development
- ⚛️ **Frontend**: React/Vue.js development
- 📝 **Documentation**: Writing docs, README files
- 🔄 **Git Workflow**: Version control, commits, branches
- 🧪 **Unit Testing**: Quick tests without GPU
- 🐛 **Debugging**: Better debugging tools than Colab

**With GitHub Copilot:**

- Auto-complete code
- Generate boilerplate
- Fix bugs quickly
- Write tests faster

### ✅ Google Colab (Cloud Training)

**Best for:**

- 🚀 **Teacher Model Training**: Free Tesla T4 GPU (~15GB VRAM)
- 🔥 **Knowledge Distillation**: Heavy training workloads
- 🎨 **Grad-CAM at Scale**: Generate heatmaps for entire dataset
- 📊 **Large-scale Inference**: Batch predictions
- 🧪 **Hyperparameter Tuning**: Try multiple configurations
- 📈 **Experiments**: Quick iteration with GPU access

**Colab Benefits:**

- Free GPU access (up to 12 hours continuous)
- No local GPU needed
- Easy to share results
- Pre-installed PyTorch/TensorFlow

### ❌ Avoid (As You Noted)

- ❌ **All-in Colab**: Hard to manage, no proper IDE, messy files
- ❌ **Local CPU Training**: Teacher training would take days/weeks
- ❌ **No Version Control**: Risk losing work, can't collaborate
- ❌ **No Code Organization**: Everything in one giant notebook

---

## 🔄 Recommended Workflow

### 1. Development Cycle (VS Code)

```
┌─────────────────────────────────────┐
│  VS Code (Local)                    │
├─────────────────────────────────────┤
│  1. Write model code                │
│  2. Write data loaders              │
│  3. Write training utilities        │
│  4. Test on small dataset (CPU)     │
│  5. Commit to GitHub                │
│  6. Create Colab-ready notebooks    │
└─────────────────────────────────────┘
         │
         ▼ (Push to GitHub)
┌─────────────────────────────────────┐
│  GitHub Repository                  │
│  • Version control                  │
│  • Code backup                      │
│  • Collaboration                    │
└─────────────────────────────────────┘
         │
         ▼ (Pull from GitHub)
┌─────────────────────────────────────┐
│  Google Colab (Cloud GPU)           │
├─────────────────────────────────────┤
│  1. Clone repo from GitHub          │
│  2. Install requirements            │
│  3. Train teacher model (GPU)       │
│  4. Save checkpoints to Drive       │
│  5. Download results                │
└─────────────────────────────────────┘
         │
         ▼ (Download checkpoints)
┌─────────────────────────────────────┐
│  VS Code (Local)                    │
│  • Integrate trained models         │
│  • Build backend/frontend           │
│  • Test inference                   │
│  • Deploy application               │
└─────────────────────────────────────┘
```

---

## 📋 Step-by-Step Setup

### Phase 1: Set Up GitHub (Do This First!)

#### 1.1 Initialize Git (if not done)

```bash
cd c:\Users\Asus\x-lite-chest-xray
git init
git add .
git commit -m "Initial commit: X-Lite project structure"
```

#### 1.2 Create GitHub Repository

```bash
# Go to github.com/dinethsadee01 and create new repo: X-Lite

# Link local to remote
git remote add origin https://github.com/dinethsadee01/X-Lite.git
git branch -M main
git push -u origin main
```

#### 1.3 Add .gitignore for Large Files

Already configured! The `.gitignore` excludes:

- ✅ Model checkpoints (_.pth, _.pt)
- ✅ Dataset images
- ✅ Logs and cache
- ✅ Virtual environment

---

### Phase 2: Create Colab-Ready Notebooks

We'll create special notebooks that:

1. Clone your GitHub repo
2. Install dependencies
3. Load data from Google Drive
4. Train models with GPU
5. Save results back to Drive

**Structure:**

```
notebooks/
├── colab/                          # Colab-specific notebooks
│   ├── 00_colab_setup.ipynb       # Setup Colab environment
│   ├── 01_train_teacher.ipynb     # Train teacher model
│   ├── 02_train_student.ipynb     # Train student models
│   ├── 03_knowledge_distillation.ipynb
│   └── 04_gradcam_generation.ipynb
└── local/                          # Local development notebooks
    ├── 00_quick_start.ipynb       # Your current notebook
    └── 01_data_exploration.ipynb
```

---

### Phase 3: Development Workflow

#### In VS Code (Daily Work)

```bash
# 1. Work on code
code ml/models/teacher_model.py
code ml/training/teacher_trainer.py

# 2. Test locally (small dataset, CPU)
python -c "from ml.models.teacher_model import TeacherModel; print('✓')"

# 3. Commit changes
git add .
git commit -m "Add teacher model architecture"
git push origin main
```

#### In Google Colab (Training)

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone your repo
!git clone https://github.com/dinethsadee01/X-Lite.git
%cd X-Lite

# 3. Install dependencies
!pip install -r requirements.txt

# 4. Train model
!python scripts/train_teacher.py --epochs 30 --gpu

# 5. Save checkpoint to Drive
!cp ml/models/checkpoints/teacher_best.pth /content/drive/MyDrive/X-Lite/
```

---

## 🚀 Specific Use Cases

### Use Case 1: Train Teacher Model

**VS Code (Preparation):**

1. Write `ml/models/teacher_model.py`
2. Write `ml/training/teacher_trainer.py`
3. Test on 100 images locally
4. Push to GitHub

**Colab (Training):**

1. Open `notebooks/colab/01_train_teacher.ipynb`
2. Clone repo, install deps
3. Upload dataset to Google Drive (one-time)
4. Train for 30 epochs (~3-4 hours with GPU)
5. Save checkpoint to Drive
6. Download results

**VS Code (Integration):**

1. Download checkpoint from Drive
2. Place in `ml/models/checkpoints/`
3. Test inference locally
4. Integrate with backend

### Use Case 2: Generate Grad-CAM Heatmaps

**VS Code:**

1. Write `ml/inference/explainability.py`
2. Test on 1 image
3. Push to GitHub

**Colab:**

1. Clone repo
2. Load trained model
3. Generate heatmaps for 1000s of images (GPU accelerated)
4. Save to Drive
5. Download selected heatmaps

**VS Code:**

1. Integrate heatmap generation in backend
2. Test with frontend

### Use Case 3: Hyperparameter Tuning

**Colab:**

1. Try different temperatures [2, 4, 6, 8]
2. Try different alphas [0.5, 0.7, 0.9]
3. Log results to TensorBoard
4. Download best configuration

**VS Code:**

1. Update `config/config.py` with best values
2. Commit to GitHub

---

## 📦 File Sync Strategy

### Option A: Google Drive (Recommended)

```
Google Drive/
└── X-Lite/
    ├── checkpoints/           # Model weights
    │   ├── teacher_best.pth
    │   └── student_best.pth
    ├── data/                  # Dataset (upload once)
    │   ├── Data_Entry_2017.csv
    │   └── images/
    └── results/               # Training results
        ├── tensorboard_logs/
        └── metrics/
```

### Option B: Direct Download

```python
# In Colab, after training
from google.colab import files
files.download('ml/models/checkpoints/teacher_best.pth')
```

---

## 🛠️ Tools Setup

### VS Code Extensions (Already Recommended)

- ✅ Python
- ✅ Pylance
- ✅ GitHub Copilot
- ✅ Jupyter
- ✅ GitLens

### Colab Settings

- Enable GPU: Runtime → Change runtime type → GPU → T4
- Increase RAM: Runtime → Change runtime type → High-RAM (if needed)

---

## 📊 Resource Allocation

| Task             | Environment | Time | GPU |
| ---------------- | ----------- | ---- | --- |
| Code Development | VS Code     | 60%  | ❌  |
| Teacher Training | Colab       | 20%  | ✅  |
| Student Training | Colab       | 10%  | ✅  |
| Backend/Frontend | VS Code     | 10%  | ❌  |

---

## 💡 Pro Tips

### 1. Dataset Management

```bash
# Upload dataset to Google Drive once
# In Colab, create symlink
!ln -s /content/drive/MyDrive/ChestX-ray14 /content/X-Lite/data/raw
```

### 2. Checkpoint Management

```python
# Save checkpoints periodically
torch.save(model.state_dict(),
           f'/content/drive/MyDrive/X-Lite/checkpoints/teacher_epoch_{epoch}.pth')
```

### 3. Experiment Tracking

- Use Weights & Biases (free, integrates with Colab)
- Access results from anywhere
- Compare experiments easily

### 4. Code Sync

```bash
# In Colab, pull latest changes
!git pull origin main

# After local changes
git push origin main
```

---

## 🚨 Important Notes

### Colab Limitations

- **Session Timeout**: 12 hours max (90 min idle)
- **Solution**: Save checkpoints every epoch
- **GPU Quota**: Limited daily usage
- **Solution**: Use efficiently, train overnight

### Data Privacy

- Don't commit large files to GitHub
- Don't commit API keys or secrets
- Use `.env` files (already in `.gitignore`)

---

## ✅ Next Actions

1. **Initialize Git** (if not done)

   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Create GitHub Repo**

   - Go to github.com
   - Create "X-Lite" repository
   - Push code

3. **Set Up Google Drive Folder**

   - Create `X-Lite` folder
   - Create subfolders: `checkpoints`, `data`, `results`

4. **Create First Colab Notebook**
   - I'll help you create `colab/00_colab_setup.ipynb`

Ready to proceed? Should I create the Colab setup notebook now?
