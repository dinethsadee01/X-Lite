# X-Lite Design & Implementation Document

**Project**: Lightweight Hybrid CNN-Transformer for Chest X-Ray Classification  
**Date**: January 21, 2026  
**Status**: Phase 1 - EDA & Design  
**Version**: 1.0

---

## Executive Summary

X-Lite is a research project to develop lightweight, interpretable deep learning models for multi-label chest X-ray disease classification. Due to strict 7-day deadline and computational constraints, this document outlines the streamlined approach using pre-trained teacher model and local GPU training.

### Project Goals

1. ✅ **Accuracy**: Per-class AUC-ROC > 0.80 on ChestX-ray14 dataset
2. ✅ **Efficiency**: Model size < 50% of baseline (EfficientNet-B0 baseline)
3. ✅ **Speed**: CPU inference < 500ms per image
4. ✅ **Interpretability**: Grad-CAM heatmaps for explainability
5. ✅ **Reproducibility**: Complete documentation and code versioning

---

## 1. Problem Statement

### Background

- **Dataset**: NIH ChestX-ray14 (112,120 frontal-view X-rays)
- **Task**: Multi-label classification of 14 disease classes
- **Challenge**: Class imbalance, computational constraints, 7-day deadline

### Class Imbalance Issue

From EDA analysis:

- **Imbalance Ratio**: ~20:1 (max:min)
- **"No Finding" prevalence**: ~60% of dataset
- **Rare diseases**: Hernia, Edema (<1% each)
- **Multi-label complexity**: 40% of images have multiple pathologies

**Impact**: Standard cross-entropy loss biased toward majority class; rare diseases underrepresented.

---

## 2. Proposed Solution Architecture

### 2.1 Overall Approach

```
Input X-ray (224×224)
    ↓
[Data Augmentation] (stratified sampling, balanced batches)
    ↓
[Student CNN Backbone] (EfficientNet-B0, MobileNetV3)
    ↓
[Optional: Transformer Attention] (MHSA or None)
    ↓
[Multi-label Head] (14 sigmoid outputs)
    ↓
[Knowledge Distillation] (from pre-trained CheXNet teacher)
    ↓
Output: [P₁, P₂, ..., P₁₄] (probability per disease)
         + Grad-CAM heatmaps
```

### 2.2 Key Decisions & Rationale

| Component                  | Choice                           | Rationale                                 |
| -------------------------- | -------------------------------- | ----------------------------------------- |
| **Teacher Model**          | Pre-trained CheXNet              | ✅ Skip 8-12h training, baseline AUC 0.84 |
| **Training Environment**   | Local (RTX 4070)                 | ✅ No session limits, faster I/O          |
| **Student Backbones**      | EfficientNet-B0, MobileNetV3     | ✅ Lightweight, proven on medical imaging |
| **Attention Modules**      | MHSA + None                      | ✅ Transformer + CNN baseline comparison  |
| **Loss Function**          | BCE + Class Weights + Focal Loss | ✅ Address class imbalance                |
| **Optimizer**              | AdamW                            | ✅ Better regularization than Adam        |
| **Learning Rate Schedule** | Cosine Annealing                 | ✅ Smooth convergence                     |
| **Data Split**             | Stratified (70/15/15)            | ✅ Preserve disease distribution          |

---

## 3. Data Preprocessing Strategy

### 3.1 Class Imbalance Mitigation

**Multi-pronged approach:**

1. **Loss Weighting**

   ```python
   weight_i = N / (C × n_i)  # Inverse frequency
   ```

   - Applied per class in BCEWithLogitsLoss
   - Rescales gradient contribution based on class frequency

2. **Balanced Sampling**
   - WeightedRandomSampler: ensure each batch contains rare classes
   - Prevents skipping minority diseases during training

3. **Focal Loss (Optional)**

   ```python
   FL(p_t) = -α_t(1 - p_t)^γ * log(p_t)
   ```

   - Reduces easy negatives' contribution
   - Focuses on hard examples

4. **Augmentation Strategy**
   - **Safe for medical imaging**: Horizontal flip, ±15° rotation
   - **Avoided**: Vertical flip, 90° rotation (destroys anatomy)
   - **Strength**: Medium (light for majority, normal for minority)

### 3.2 Train/Val/Test Split

- **Strategy**: Stratified split by number of diseases per image
- **Proportions**: 70% train, 15% val, 15% test
- **Reproducibility**: `random_state=42`
- **Files**: `data/splits/{train,val,test}.csv`

### 3.3 Image Preprocessing

```python
Preprocessing Pipeline:
├─ Resize to 224×224 (standard for CNNs)
├─ Convert PIL → Tensor
├─ Normalize: ImageNet mean/std
│  (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225])
└─ Transfer learning justification: radiographs share features with natural images

Augmentation (Training):
├─ HorizontalFlip(p=0.5)
├─ Rotation(±15°, p=0.4)
├─ ShiftScaleRotate(p=0.3)
├─ GaussNoise or GaussianBlur(p=0.2)
├─ RandomBrightnessContrast(p=0.3)
└─ [No augmentation for Val/Test]
```

---

## 4. Model Architecture

### 4.1 Student Model Design

```python
StudentModel:
├─ Backbone (CNN):
│  ├─ Option 1: EfficientNet-B0 (3.7M params)
│  ├─ Option 2: MobileNetV3-Small (2.5M params)
│  └─ Option 3: ResNet18 (11.2M params)
│
├─ Attention Module (Optional):
│  ├─ None (baseline)
│  ├─ MHSA (Multi-Head Self-Attention)
│  └─ Performer (Linear attention)
│
├─ Global Average Pooling (1280 features)
│
└─ Classification Head:
   ├─ FC(1280 → 256)
   ├─ ReLU + Dropout(0.3)
   ├─ FC(256 → 14)
   └─ Sigmoid (multi-label)
```

### 4.2 Knowledge Distillation

**Teacher-Student Framework:**

```
Teacher (CheXNet):
├─ Pre-trained DenseNet121
├─ AUC 0.84 baseline
└─ Frozen (no gradient updates)

Student (Lightweight):
├─ EfficientNet-B0 / MobileNetV3
├─ Trainable with KD loss
└─ Goal: Match teacher's soft predictions

KD Loss:
├─ Soft loss: KL(T_soft || S_soft) with temperature τ
├─ Hard loss: BCE(S_logits || y_true)
└─ Combined: L_KD = α × L_soft + (1-α) × L_hard
```

**Hyperparameter Search:**

- Temperature τ ∈ {2, 4, 6, 8}
- Alpha α ∈ {0.5, 0.7, 0.9}
- Total combinations: 8 (test on 20% subset)

---

## 5. Training Configuration

### 5.1 Hyperparameters (Baseline)

| Parameter         | Value                 | Rationale                               |
| ----------------- | --------------------- | --------------------------------------- |
| **Batch Size**    | 64                    | RTX 4070: 16GB, 64 fits comfortably     |
| **Learning Rate** | 1e-3                  | Standard for Adam, adjusted by schedule |
| **Optimizer**     | AdamW                 | L2 regularization built-in              |
| **Weight Decay**  | 1e-4                  | Prevent overfitting                     |
| **Epochs**        | 50                    | Early stopping after 10 patience        |
| **LR Scheduler**  | CosineAnnealingLR     | Smooth convergence                      |
| **Loss**          | BCE + Focal + Weights | Class imbalance handling                |

### 5.2 Training Strategy

**Phase 1: Baseline Student (20% data)**

- Train 4 student models (2 CNN × 2 attention)
- Identify best architecture combo
- Time: ~2 hours

**Phase 2: Knowledge Distillation (20% data)**

- Test 8 KD configurations (2 CNN × 2 temp × 2 alpha)
- Find optimal distillation hyperparameters
- Time: ~6 hours

**Phase 3: Final Model (Full data)**

- Train best configuration on 100% data
- Apply quantization/pruning if needed
- Time: ~4-6 hours

---

## 6. Evaluation & Metrics

### 6.1 Primary Metrics

**Per-Class AUC-ROC**

- Compute for each disease individually
- Target: > 0.80
- Rationale: class-imbalance robust, interpretable

**Macro-Averaged F1**

- Equal weight to each disease
- Sensitive to rare class performance

### 6.2 Secondary Metrics

**Per-Class Precision/Recall**

- Identify which diseases are over/under-predicted
- Adjust thresholds per class if needed

**Confusion Matrix**

- Multi-label: check for disease co-occurrence errors

**Calibration**

- Reliability diagrams
- Isotonic regression if miscalibrated

### 6.3 Avoided Metrics

❌ **Accuracy**: Misleading with 60% "No Finding"  
❌ **Micro-averaged F1**: Dominated by majority class  
❌ **Single global threshold**: Different optimal thresholds per disease

---

## 7. Interpretability: Grad-CAM

### 7.1 Visualization Strategy

```python
For each prediction:
├─ Generate Grad-CAM heatmap (disease-specific)
├─ Overlay on original X-ray
├─ Display top 3 predicted diseases + heatmaps
└─ Validate anatomical correctness
```

**Use Cases:**

- Explain high-confidence predictions
- Debug misclassifications
- Clinical validation

---

## 8. Implementation Timeline

| Day     | Task                       | Duration | Deliverable                    |
| ------- | -------------------------- | -------- | ------------------------------ |
| **1**   | EDA + Design               | 2h       | EDA_REPORT.md                  |
| **2**   | Preprocessing + Dataloader | 2h       | Working train/val/test loaders |
| **3-4** | Baseline students          | 8h       | Best architecture selected     |
| **5-6** | Knowledge distillation     | 12h      | Best KD hyperparameters        |
| **7**   | Backend/Frontend           | 8h       | Working web interface          |

---

## 9. Version Control & Documentation

### 9.1 Experiment Tracking

**CSV Format: `experiments/results.csv`**

```
experiment_id, date, model, backbone, attention, kd_temp, kd_alpha,
train_loss, val_auc, test_auc, inference_time_ms, model_size_mb,
notes
```

### 9.2 Git Workflow

- **Main branch**: Production code only
- **dev branches**: Per-component (data, models, training)
- **Commits**: Frequent, descriptive messages

### 9.3 Documentation

```
docs/
├─ DESIGN.md (this file)
├─ EDA_REPORT.md (dataset analysis)
├─ PREPROCESSING.md (data handling decisions)
├─ EXPERIMENTS.md (all training runs)
├─ RESULTS.md (final performance)
└─ IMPLEMENTATION.md (technical decisions)
```

---

## 10. Risk Mitigation

| Risk                     | Likelihood | Impact | Mitigation                                      |
| ------------------------ | ---------- | ------ | ----------------------------------------------- |
| **Out of GPU memory**    | Medium     | High   | Use smaller batch size, gradient checkpointing  |
| **Class imbalance bias** | High       | High   | Class weights + focal loss + balanced sampling  |
| **Overfitting**          | Medium     | Medium | Stratified splits, dropout, L2 regularization   |
| **Poor distillation**    | Medium     | Medium | Test multiple τ/α values, monitor both losses   |
| **Incomplete training**  | Low        | High   | Save checkpoints every epoch, resume capability |

---

## 11. Expected Outcomes

### 11.1 Performance Targets

- **Teacher (CheXNet)**: AUC 0.84 (baseline, frozen)
- **Student Baseline**: AUC 0.75+ (70% of teacher)
- **Student + KD**: AUC 0.78+ (93% of teacher)
- **Inference Speed**: <500ms CPU, <100ms GPU

### 11.2 Model Size

- EfficientNet-B0: 3.7M params (14MB)
- MobileNetV3-Small: 2.5M params (10MB)
- DenseNet121 (teacher): 6.9M params (28MB)

**Target**: Final student <50% of DenseNet121 ✓

---

## 12. Future Work (Post Deadline)

1. Extended hyperparameter tuning with Optuna
2. Model ensemble (multiple student architectures)
3. Quantization (INT8) for mobile deployment
4. Federated learning for privacy-preserving training
5. Active learning for efficient data collection

---

## Appendix: Configuration Files

**`config/config.py`**

- All paths, hyperparameters, model configs
- Single source of truth for reproducibility

**`requirements.txt`**

- PyTorch 2.9.1
- Albumentations (augmentation)
- Scikit-learn (metrics)
- Pandas, NumPy, Matplotlib

---

**Document Status**: ✅ **APPROVED** for Phase 1 Implementation  
**Next Review**: After EDA completion  
**Author**: AI Assistant  
**Last Updated**: January 21, 2026
