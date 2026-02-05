# X-Lite Model Inference Testing Guide

## 🎯 Overview

You now have **two ways** to test your trained model:

### 1. **Direct Notebook Testing** (NEW)
- **File**: `notebooks/local/02_model_inference_testing.ipynb`
- **Use case**: Quick model testing without frontend/backend
- **No dependencies**: Just Jupyter + your trained checkpoint
- **Features**:
  - Load any trained checkpoint
  - Test on single or batch images
  - Compare WITH CLAHE vs WITHOUT CLAHE preprocessing
  - Full per-class metrics evaluation
  - Visualizations included

### 2. **Frontend/Backend Testing** (Original)
- **Setup**: Run both backend API and React frontend
- **More realistic**: Tests the full inference pipeline
- **Now fixed**: Inference applies **CLAHE preprocessing** (matches training)

---

## 🚀 How to Use the Inference Notebook

### Prerequisites
Make sure you have trained a model first:
```bash
python scripts/train_baseline.py
```

This creates checkpoints in `ml/models/checkpoints/`

### Open & Run the Notebook

1. **Open VS Code**
2. **Navigate to**: `notebooks/local/02_model_inference_testing.ipynb`
3. **Run cells in order** (Jupyter will detect kernel automatically)

### What Each Section Does

| Section | Purpose |
|---------|---------|
| **1. Find & Load Model** | Auto-discovers and loads your latest trained checkpoint |
| **2. Prepare Test Data** | Sets up CLAHE transforms matching your training pipeline |
| **3. Make Predictions** | Runs inference on a single test image with CLAHE |
| **4. Visualize Results** | Shows prediction bars and risk levels |
| **5. Compare CLAHE Impact** | Shows predictions WITH vs WITHOUT CLAHE side-by-side |
| **6. Batch Test** | Tests multiple images in one go |
| **7. Full Test Set Eval** | Calculates per-class AUC, F1, Precision, Recall on all 16,818 test images |

---

## 📊 Example Output

**With CLAHE (training-matched):**
```
Disease                  Probability    Risk Level    
────────────────────────────────────────────────────
Pneumonia                0.8742         High          
Infiltration             0.6521         Moderate      
Atelectasis              0.4321         Low           
Effusion                 0.2891         Low           
```

**Per-Class Metrics:**
```
Disease             AUC      F1      Precision  Recall
──────────────────────────────────────────────────────
Atelectasis         0.8234   0.7621  0.7854     0.7402
Cardiomegaly        0.8901   0.8234  0.8456     0.8015
...
Mean AUC: 0.8234
```

---

## ⚙️ CLAHE Preprocessing (Now Applied at Inference)

### What Changed?
- **Training**: Used CLAHE-cached images for better contrast
- **Inference (Before)**: Raw images with only resize/normalize → domain mismatch
- **Inference (Now)**: Applied **CLAHE on-the-fly** → matches training exactly

### Backend Configuration
The file `backend/services/prediction_service.py` now uses:
```python
self.transform = get_medical_transforms(use_clahe=True, use_denoising=False)
```

This ensures **frontend/backend predictions** match **notebook predictions**.

---

## 🔍 Why This Matters for Your Issue

Your original problem:
> "All predictions are wrong. Why?"

**Root cause**: Preprocessing mismatch
- ✅ **Training**: CLAHE-enhanced images
- ❌ **Inference**: Raw images without CLAHE
- Result: Model gets different input → different (wrong) predictions

**Fix applied**:
- ✅ **Notebook**: Uses CLAHE transforms
- ✅ **Backend**: Now applies CLAHE at inference
- Result: Consistent predictions across all interfaces

---

## 🧪 Testing Checklist

Before going to production:

- [ ] Run notebook section 1-4 on a single image → verify reasonable predictions
- [ ] Run notebook section 5 → compare WITH vs WITHOUT CLAHE impact
- [ ] Run notebook section 6 → batch test 5+ images
- [ ] Run notebook section 7 → evaluate on full test set, check mean AUC > 0.80
- [ ] Test frontend upload → verify predictions match notebook

---

## 💡 Troubleshooting

### "No checkpoints found"
- Train a model first: `python scripts/train_baseline.py`
- Checkpoints should appear in `ml/models/checkpoints/`

### "No test images available"
- CLAHE cache missing? Run: `python scripts/precompute_clahe.py`
- Or use raw images from: `data/raw/images/`

### "Predictions still look wrong"
- Check mean AUC in section 7
- If mean AUC < 0.70, model may be undertrained
- Re-train with more epochs or full dataset

### "Different predictions in notebook vs backend"
- Both now use CLAHE, should match exactly
- If not: check model path is the same checkpoint

---

## 📁 Key Files Reference

```
notebooks/
  └─ local/
     ├─ 02_model_inference_testing.ipynb  ← NEW: Direct inference testing
     ├─ 01_data_exploration.ipynb          (existing)
     └─ 00_quick_start.ipynb               (existing)

backend/
  └─ services/
     └─ prediction_service.py              (updated with CLAHE)

ml/
  └─ data/
     └─ preprocessing.py                   (CLAHE implementation)
```

---

## 🎓 Next Steps

1. **Test immediately**: Run the notebook on your trained model
2. **Verify results**: Check if predictions now make sense
3. **Frontend testing**: Upload same test images to UI, compare results
4. **Production**: Once tests pass, deploy with confidence!

---

*Created: 2026-02-05*
*X-Lite Project | Direct Inference Testing Guide*
