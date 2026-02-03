# Phase 2: Knowledge Distillation Training

**Status**: Ready to begin  
**Date Started**: February 3, 2026  
**Objective**: Distill knowledge from CheXNet teacher to lightweight student models

---

## Overview

**Knowledge Distillation (KD)** trains lightweight student models to mimic a larger, more accurate teacher model. This allows us to:
- ✅ Create faster, smaller models (inference: <100ms)
- ✅ Retain high accuracy (target: 0.80+ AUC)
- ✅ Reduce memory footprint (50-70% parameter reduction)

---

## Setup Summary

### 1. Teacher Model
- **Architecture**: DenseNet121 (medical imaging adapted)
- **Purpose**: Distill knowledge to students
- **Weights**: Frozen during training
- **Location**: `ml/models/teacher_model.py`

### 2. Student Models (to be distilled)
- **ConvNext Tiny MHSA** (baseline - 0.8351 AUC)
- **EfficientNet-B0** (lightweight, fast)
- **MobileNet V3** (ultra-lightweight)

### 3. Distillation Loss
- **Components**:
  - **KD Loss**: KL divergence on soft targets (with temperature T)
  - **CE Loss**: Binary cross-entropy on hard targets (ground truth)
  - **Weighted**: alpha * KD_loss + (1-alpha) * CE_loss

- **Hyperparameters** (configurable):
  - `temperature`: 2, 4, 6, 8 (controls softness)
  - `alpha`: 0.5, 0.7, 0.9 (KD vs CE weight)

### 4. Training Infrastructure
- **Trainer**: `ml/training/kd_trainer.py`
- **Loss Functions**: `ml/training/kd_losses.py`
- **Teacher**: `ml/models/teacher_model.py`

---

## Implementation Checklist

### Phase 2A: Teacher Model Preparation (Today)

- [ ] **Load pretrained CheXNet weights** (if available)
  ```bash
  # Option 1: Download from public source
  # Option 2: Use DenseNet121 from torchvision (already implemented)
  ```

- [ ] **Verify teacher model loads correctly**
  ```python
  from ml.models.teacher_model import create_teacher_model
  teacher = create_teacher_model(device=device)
  print(f"Teacher params: {teacher.get_num_params():,}")  # Should be ~7M
  ```

- [ ] **Test teacher inference** (sanity check)
  ```python
  # Forward pass on dummy input
  dummy_input = torch.randn(4, 3, 224, 224).to(device)
  output = teacher(dummy_input)
  assert output.shape == (4, 14), f"Expected (4, 14), got {output.shape}"
  ```

### Phase 2B: Student Distillation Training (Days 2-3)

- [ ] **Create training script** (`scripts/train_kd.py`)
- [ ] **Run baseline KD** (ConvNext Tiny → smaller network)
- [ ] **Track distillation metrics**:
  - Student Val AUC
  - Knowledge transfer efficiency
  - Training time
  - Model size

- [ ] **Experiment with hyperparameters**:
  - Temperature: 2, 4, 6, 8
  - Alpha: 0.5, 0.7, 0.9
  - Different student architectures

### Phase 2C: Analysis & Selection (Day 4)

- [ ] **Compare student models**
- [ ] **Select best trade-off** (AUC vs speed vs size)
- [ ] **Document results** in experiment log

---

## Key Files Created

```
✅ ml/models/teacher_model.py          - Teacher (DenseNet121 based)
✅ ml/training/kd_losses.py            - KD + Focal KD losses
✅ ml/training/kd_trainer.py           - KD training loop
⏳ scripts/train_kd.py                 - Main KD training script (TODO)
⏳ experiments/kd_results.csv           - Results tracker (TODO)
```

---

## Next Steps

1. **Load baseline teacher** and verify it works
2. **Prepare KD training script** with sample configs
3. **Run first experiment**: ConvNext Tiny MHSA as student
4. **Track & compare** distillation effectiveness

---

## Configuration Example

```python
# KD Hyperparameters
KD_CONFIG = {
    'teacher_model': 'densenet121',
    'student_model': 'convnext_tiny_mhsa',
    'temperature': 4.0,        # Soft target softness
    'alpha': 0.7,              # KD vs CE weight
    'student_epochs': 40,
    'student_batch_size': 64,
    'student_lr': 1e-3,
    'early_stopping_patience': 10,
    'target_auc': 0.80
}
```

---

**Ready to implement KD training script?** (scripts/train_kd.py)

I can create this immediately with:
- Integrated teacher loading
- KD loss computation
- Student training loop
- Results tracking
