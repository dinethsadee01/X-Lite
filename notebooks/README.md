# X-Lite Notebooks

This directory contains Jupyter notebooks for different environments and purposes.

## 📁 Directory Structure

```
notebooks/
├── local/                    # Run on your local machine (VS Code)
│   └── 00_quick_start.ipynb # Project overview and setup verification
│
└── colab/                    # Run on Google Colab (GPU training)
    └── 00_colab_setup.ipynb # Colab environment setup
```

## 🎯 Usage Guide

### Local Notebooks (VS Code)

**Purpose**: Development, testing, and exploration on your local machine

**When to use:**
- ✅ Quick prototyping and code testing
- ✅ Data exploration and visualization
- ✅ Debugging models and utilities
- ✅ Testing with small datasets
- ✅ Backend/frontend integration

**Notebooks:**
- `00_quick_start.ipynb` - Verify installation, explore configuration

**To run:**
```bash
cd c:\Users\Asus\x-lite-chest-xray
jupyter notebook notebooks/local/
```

---

### Colab Notebooks (Google Colab)

**Purpose**: GPU-accelerated training and large-scale computations

**When to use:**
- ✅ Training teacher model (requires GPU)
- ✅ Training student models
- ✅ Knowledge distillation experiments
- ✅ Generating Grad-CAM heatmaps at scale
- ✅ Hyperparameter tuning
- ✅ Large-scale inference

**Notebooks:**
- `00_colab_setup.ipynb` - Set up Colab environment (run this first!)
- `01_train_teacher.ipynb` - Train DenseNet121 teacher (coming soon)
- `02_train_student.ipynb` - Train student models (coming soon)
- `03_knowledge_distillation.ipynb` - Knowledge distillation (coming soon)
- `04_gradcam_generation.ipynb` - Generate heatmaps (coming soon)

**To run:**
1. Push your code to GitHub
2. Open [Google Colab](https://colab.research.google.com/)
3. File → Upload notebook → Browse to `notebooks/colab/`
4. Or use direct link (once uploaded to GitHub)

---

## 🔄 Typical Workflow

### 1. Local Development (VS Code)
```
1. Write model code in ml/models/
2. Test with local/00_quick_start.ipynb
3. Verify on small dataset (CPU)
4. Commit and push to GitHub
```

### 2. Cloud Training (Colab)
```
1. Open colab/00_colab_setup.ipynb
2. Mount Drive, clone repo, install deps
3. Open colab/01_train_teacher.ipynb
4. Train with GPU
5. Save checkpoints to Google Drive
```

### 3. Integration (VS Code)
```
1. Download checkpoints from Drive
2. Test inference locally
3. Integrate with backend API
4. Deploy application
```

---

## 📊 Notebook Comparison

| Feature | Local Notebooks | Colab Notebooks |
|---------|----------------|-----------------|
| **Environment** | VS Code | Google Colab |
| **GPU Access** | If available | Free T4/P100/V100 |
| **Session Length** | Unlimited | 12 hours max |
| **Best For** | Development | Training |
| **Data Storage** | Local disk | Google Drive |
| **Code Editing** | Excellent (Copilot) | Basic |
| **Debugging** | Excellent | Limited |
| **Collaboration** | Git | Share link |

---

## 💡 Tips

### Local Notebooks
- Use for rapid iteration and testing
- Keep datasets small for faster loading
- Leverage GitHub Copilot for code generation
- Use for backend/frontend development

### Colab Notebooks
- Always run `00_colab_setup.ipynb` first
- Save checkpoints every epoch (to Drive)
- Monitor session time (12 hour limit)
- Use for computationally intensive tasks
- Enable GPU: Runtime → Change runtime type → GPU

---

## 🚀 Getting Started

**First time setup:**

1. **Local**: Run `notebooks/local/00_quick_start.ipynb`
   - Verify installation
   - Check configuration
   - Understand project structure

2. **Colab**: Run `notebooks/colab/00_colab_setup.ipynb`
   - Set up GPU environment
   - Mount Google Drive
   - Clone repository
   - Install dependencies

**Then proceed to:**
- Data exploration and preprocessing
- Model training and evaluation
- Application development

---

## 📚 Resources

- **Jupyter Documentation**: https://jupyter.org/documentation
- **Google Colab Guide**: https://colab.research.google.com/notebooks/intro.ipynb
- **X-Lite Docs**: Check `docs/` folder for detailed guides

---

**Happy coding! 🚀**
