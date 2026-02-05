# Fixed Training Guide

## Problem Summary

The original trained model predicted **all 14 diseases as positive (~0.99 confidence)** for every image because:

1. **Final layer bias was too high** (mean=0.24, causing sigmoid outputs >0.5 before seeing any image)
2. **Positive class weights were too aggressive** (90x imbalance between common/rare diseases)
3. **No validation of prediction distributions during training**

## Solution

The new training script [`train_kd_fixed.py`](train_kd_fixed.py) fixes these issues:

### Key Fixes

1. **Proper Bias Initialization**
   - Final layer bias initialized to log-odds: `log(n_positive / n_negative)`
   - Prevents model from starting biased toward all-positive or all-negative
   - Example: For a disease with 1000 positives and 77000 negatives, bias = -4.34

2. **Reduced Positive Weight Smoothing**
   - Changed from `alpha=0.5` to `alpha=0.25` by default
   - Reduces weight difference between common and rare classes
   - Prevents extreme bias toward rare diseases

3. **Focal Loss Option**
   - Alternative to weighted BCE for handling imbalance
   - Automatically focuses on hard examples without manual weight tuning

4. **Prediction Validation**
   - Checks mean predictions before and after training
   - Warns if model predicts all positives or all negatives

## Usage

### Basic Training (Recommended)

```bash
python scripts/train_kd_fixed.py \
    --student_model convnext_tiny_mhsa \
    --loss_type bce \
    --pos_weight_alpha 0.25 \
    --epochs 40 \
    --batch_size 32
```

### With Focal Loss (Better for Severe Imbalance)

```bash
python scripts/train_kd_fixed.py \
    --student_model convnext_tiny_mhsa \
    --loss_type focal \
    --focal_gamma 2.0 \
    --epochs 40 \
    --batch_size 32
```

### All Options

```
--student_model        Model architecture (default: convnext_tiny_mhsa)
--loss_type            Loss function: bce or focal (default: bce)
--pos_weight_alpha     Weight smoothing 0.1-1.0 (default: 0.25, lower=less aggressive)
--focal_gamma          Focal loss gamma (default: 2.0, higher=focus more on hard examples)
--temperature          KD temperature (default: 4.0)
--alpha                KD weight 0-1 (default: 0.7, higher=more teacher influence)
--epochs               Training epochs (default: 40)
--batch_size           Batch size (default: 32)
--early_stopping_patience  Stop after N epochs without improvement (default: 8)
```

## What to Expect

### Initial Predictions (Before Training)

After bias initialization, you should see **balanced** mean predictions:

```
Mean predictions per disease:
  Atelectasis               0.1034
  Cardiomegaly              0.0244
  Effusion                  0.1186
  Infiltration              0.1761
  ...
✓ Initial predictions look balanced
```

### Training Output

```
Epoch 1/40
  Train Loss: 0.2543  Val Loss: 0.2187  Val AUC: 0.7234
  
Epoch 5/40
  Train Loss: 0.1876  Val Loss: 0.1923  Val AUC: 0.7891
  ...
```

### Final Validation

After training, predictions should still be balanced:

```
Mean predictions per disease:
  Atelectasis               0.2145
  Cardiomegaly              0.0876
  Effusion                  0.1923
  ...
✓ Final predictions look balanced
```

**Warning Signs:**
- If you see `⚠️ Model predicting all positives!` → Training failed, try lower `pos_weight_alpha`
- If you see `⚠️ Model predicting all negatives!` → Try higher `pos_weight_alpha` or focal loss

## Output Files

- **Best model:** `ml/models/checkpoints/kd_fixed/{model_name}/best_model.pth`
- **Final model:** `ml/models/checkpoints/final/X-Lite_fixed_{model_name}.pth`
- **Results:** `experiments/kd_fixed_results.csv`

## Comparison with Old Training

| Aspect | Old Training | Fixed Training |
|--------|--------------|----------------|
| Final layer bias | Random (~0.24 mean) | Log-odds (~-2.5 to -0.5) |
| Pos weight alpha | 0.5 (aggressive) | 0.25 (mild) |
| Validation checks | None | Every 5 epochs |
| Mean prediction | ~0.99 (all positive) | ~0.05-0.20 (balanced) |
| Usable model | ❌ No | ✅ Yes |

## Recommended Next Steps

1. **Train with default settings first** to establish baseline
2. **If rare diseases are still underdetected**, try `--pos_weight_alpha 0.35`
3. **If common diseases dominate**, try `--loss_type focal`
4. **Monitor the validation checks** - they'll warn you early if training is going wrong

## Testing the New Model

After training completes, test with:

```bash
# Test single image
python -c "
from pathlib import Path
import torch
from PIL import Image
from ml.models.student_model import create_student_model
from ml.data.preprocessing import get_medical_transforms
from config import DISEASE_LABELS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_student_model('convnext_tiny_mhsa', num_classes=14, pretrained=False)

checkpoint = torch.load('ml/models/checkpoints/final/X-Lite_fixed_convnext_tiny_mhsa.pth', 
                        map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['student_state_dict'])
model.to(device).eval()

transform = get_medical_transforms(use_clahe=True, use_denoising=False)
image = Image.open('data/raw/images/00004946_000.png').convert('RGB')
tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

print('Predictions:')
for i, disease in enumerate(DISEASE_LABELS):
    if probs[i] >= 0.5:
        print(f'  {disease}: {probs[i]:.4f}')
if (probs < 0.5).all():
    print('  No Finding')
"
```

If you see **only 1-2 diseases** instead of all 14, training succeeded! 🎉
