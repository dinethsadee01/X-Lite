"""
Improved Full Dataset Training Script v2 (14-Class, Balanced)
=============================================================
Final version with all stability and performance fixes:

1. Per-class alpha Focal Loss (rare diseases get higher weight, capped at 0.95)
2. Dampened WeightedRandomSampler (sqrt-frequency, not raw inverse)
3. Medical-grade augmentation (flips/rotations/distortions)
4. 14-class approach (No_Finding handled implicitly)
5. CosineAnnealingWarmRestarts scheduler + LR warmup
6. Gradient clipping at 1.0 + loss clamping for AMP stability

Usage:
  python scripts/train_improved.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json

from ml.models.student_model import create_student_model, MODEL_CONFIGS
from ml.data.loader import ChestXrayDataset
from ml.data.augmentation import get_augmentation_pipeline
from ml.data.preprocessing import get_medical_transforms
from ml.training.trainer import create_trainer
from ml.training.losses import FocalLoss, compute_focal_alpha_per_class, calculate_pos_weights
from torch.utils.data import WeightedRandomSampler
from scripts.training_utils import evaluate_final_metrics


# ============================================================
# CONFIGURATION — Edit these values to change training settings
# ============================================================
MODEL_NAME = 'efficientnet_b0_performer'
NUM_EPOCHS = 70
BATCH_SIZE = 128
LEARNING_RATE = 3e-5               # Slightly lower for stability
EARLY_STOPPING_PATIENCE = 5
GRADIENT_CLIP_VAL = 1.0            # Tight clipping to prevent NaN with AMP
USE_WEIGHTED_SAMPLER = True
AUGMENTATION_STRENGTH = 'medical'
CHECKPOINT_SUFFIX = '_full_dataset_14class_v2'
SAVE_EACH_EPOCH_CHECKPOINT = True
WARMUP_EPOCHS = 2                  # LR warmup for first N epochs
# 14-class approach: No_Finding is NOT an output class
NUM_CLASSES = 14
# ============================================================

# 14 disease labels (without No_Finding)
DISEASE_LABELS_14 = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]


def count_labels_14class(df: pd.DataFrame) -> np.ndarray:
    """Count positive samples per disease class (14 classes, no No_Finding)."""
    label_counts = np.zeros(14)
    for label_str in df['labels']:
        if pd.isna(label_str) or label_str == 'No Finding':
            continue
        labels = label_str.split('|')
        for label in labels:
            label = label.strip()
            if label in DISEASE_LABELS_14:
                idx = DISEASE_LABELS_14.index(label)
                label_counts[idx] += 1
    return label_counts


def get_dampened_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Compute sample weights using sqrt-dampened inverse frequency.
    
    Unlike raw inverse-frequency (1/freq), sqrt-dampening (1/√freq) gives
    moderate upsampling to rare classes without neglecting common ones.
    This prevents the precision collapse seen with aggressive oversampling.
    """
    label_counts = count_labels_14class(df)
    
    # sqrt-dampened inverse frequency
    class_weights = 1.0 / np.sqrt(label_counts + 1.0)
    class_weights = class_weights / class_weights.mean()  # normalize to mean=1
    no_finding_weight = class_weights.min()  # healthy samples get lowest weight
    
    sample_weights = []
    for label_str in df['labels']:
        if pd.isna(label_str) or label_str == 'No Finding':
            sample_weights.append(float(no_finding_weight))
            continue
        
        labels = [lbl.strip() for lbl in label_str.split('|') if lbl.strip() in DISEASE_LABELS_14]
        if not labels:
            sample_weights.append(float(no_finding_weight))
            continue
        
        # Sample weight = max of its class weights (dominated by rarest disease)
        weights = [class_weights[DISEASE_LABELS_14.index(lbl)] for lbl in labels]
        sample_weights.append(float(np.max(weights)))
    
    return np.array(sample_weights, dtype=np.float64)


def main():
    """Train single model on full dataset with all performance fixes applied."""
    print("\n" + "=" * 70)
    print("IMPROVED FULL DATASET TRAINING")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Classes: {NUM_CLASSES} (14 diseases, no No_Finding)")
    print(f"Epochs: {NUM_EPOCHS} (early stopping patience={EARLY_STOPPING_PATIENCE})")
    print(f"Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"Augmentation: {AUGMENTATION_STRENGTH}")
    print(f"Weighted Sampler: {USE_WEIGHTED_SAMPLER}")
    print(f"Loss: Per-class alpha Focal Loss (γ=2.0)")
    print("=" * 70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Paths
    clahe_cache_dir = project_root / "data" / "clahe_cache"
    train_csv = project_root / "data" / "splits" / "train_df.csv"
    val_csv = project_root / "data" / "splits" / "val_df.csv"

    # Validate cache exists
    if not clahe_cache_dir.exists():
        print("\n❌ CLAHE cache not found.")
        print(f"  Expected at: {clahe_cache_dir}")
        print("  Run: python scripts/precompute_clahe.py")
        return

    # Data transforms
    print("\nPreparing data loaders...")
    train_aug = get_augmentation_pipeline(augmentation_strength=AUGMENTATION_STRENGTH)
    val_transform = get_medical_transforms(use_clahe=False, use_denoising=False)

    # Load CSVs
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    if 'Image Index' in train_df.columns:
        train_df = train_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
    if 'Image Index' in val_df.columns:
        val_df = val_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})

    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples: {len(val_df):,}")

    # Create datasets with explicit 14-class configuration
    # (passed as constructor args so DataLoader workers see the right values)
    train_dataset = ChestXrayDataset(
        str(clahe_cache_dir), train_df, transform=train_aug, is_training=True,
        num_classes=NUM_CLASSES, disease_labels=DISEASE_LABELS_14
    )
    val_dataset = ChestXrayDataset(
        str(clahe_cache_dir), val_df, transform=val_transform, is_training=False,
        num_classes=NUM_CLASSES, disease_labels=DISEASE_LABELS_14
    )

    # ---- Dampened Weighted Random Sampler ----
    if USE_WEIGHTED_SAMPLER:
        print("\n✓ Computing sqrt-dampened sample weights...")
        sample_weights = get_dampened_sample_weights(train_df)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_shuffle = False  # Sampler handles shuffling
        print(f"  Weight range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
        print(f"  Weight ratio (max/min): {sample_weights.max() / sample_weights.min():.1f}x")
        print(f"  Mean weight: {sample_weights.mean():.4f}")
    else:
        sampler = None
        train_shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=train_shuffle,
        sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ---- Train model ----
    print("\n" + "=" * 70)
    print(f"TRAINING: {MODEL_NAME} (14-class improved)")
    print("=" * 70)

    # Create model (14 classes)
    model = create_student_model(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True)
    num_params = model.get_num_params()
    model_size_mb = model.get_model_size_mb()

    print(f"Parameters: {num_params:,}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Backbone: {MODEL_CONFIGS[MODEL_NAME]['backbone']}")
    print(f"Attention: {MODEL_CONFIGS[MODEL_NAME]['attention']}")

    # ---- Per-class alpha Focal Loss ----
    label_counts = count_labels_14class(train_df)
    label_counts_tensor = torch.tensor(label_counts)

    # Compute per-class alpha: rare diseases get higher alpha
    alpha_per_class = compute_focal_alpha_per_class(
        label_counts_tensor,
        total_samples=len(train_df),
        min_alpha=0.5,
        max_alpha=0.95  # Capped at 0.95 to prevent float16 overflow with AMP
    )

    print(f"\nPer-class alpha weights for Focal Loss:")
    for i, label in enumerate(DISEASE_LABELS_14):
        freq_pct = label_counts[i] / len(train_df) * 100
        print(f"  {label:<25} count={int(label_counts[i]):>6,}  "
              f"freq={freq_pct:>5.2f}%  α={alpha_per_class[i]:.4f}")

    criterion = FocalLoss(alpha=alpha_per_class, gamma=2.0, reduction='mean')
    print(f"\nLoss: Per-class alpha Focal Loss (γ=2.0)")

    # Checkpoint directory
    checkpoint_dir = project_root / "ml" / "models" / "new checkpoints fix" / f"{MODEL_NAME}{CHECKPOINT_SUFFIX}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Gradient Clipping: {GRADIENT_CLIP_VAL}")

    # Create trainer

    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        learning_rate=LEARNING_RATE,
        weight_decay=1e-5,
        checkpoint_dir=checkpoint_dir,
        device=device,
        use_amp=True,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        num_classes=NUM_CLASSES
    )

    # Per-epoch checkpoint saving
    if SAVE_EACH_EPOCH_CHECKPOINT:
        epoch_ckpt_dir = checkpoint_dir / "epoch_checkpoints"
        epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
        original_save_checkpoint = trainer.save_checkpoint

        def save_checkpoint_with_epoch_copy(epoch, is_best=False, metrics=None):
            original_save_checkpoint(epoch, is_best, metrics)
            val_auc = 0.0
            if isinstance(metrics, dict):
                val_auc = float(metrics.get('auc_macro', 0.0))
            ckpt_path = epoch_ckpt_dir / f"model_epoch_{epoch:03d}_{val_auc:.4f}.pth"
            torch.save(trainer.model.state_dict(), ckpt_path)
            print(f"  ✓ Saved epoch checkpoint: {ckpt_path.name}")

        trainer.save_checkpoint = save_checkpoint_with_epoch_copy
        print(f"Per-epoch checkpoints enabled: {epoch_ckpt_dir}")

    # Scheduler: CosineAnnealingWarmRestarts for smoother LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        trainer.optimizer,
        T_0=10,        # First restart after 10 epochs
        T_mult=2,      # Double period after each restart
        eta_min=1e-7   # Minimum LR
    )
    print(f"Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")

    # LR Warmup: linearly ramp up LR over first N epochs
    if WARMUP_EPOCHS > 0:
        base_lr = LEARNING_RATE
        warmup_factor = 0.1  # start at 10% of base LR
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            trainer.optimizer,
            start_factor=warmup_factor,
            end_factor=1.0,
            total_iters=WARMUP_EPOCHS
        )
        # Chain warmup then cosine
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            trainer.optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[WARMUP_EPOCHS]
        )
        print(f"LR Warmup: {WARMUP_EPOCHS} epochs ({base_lr * warmup_factor:.1e} → {base_lr:.1e})")

    # Train
    start_time = time.time()
    history = trainer.train(
        num_epochs=NUM_EPOCHS,
        scheduler=scheduler,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        verbose=True
    )
    train_time = time.time() - start_time

    # Evaluate final metrics
    print("\nComputing final metrics (AUC, F1, PR-AUC, Precision, Recall)...")
    final_train_metrics = evaluate_final_metrics(model, train_loader, device, disease_labels=DISEASE_LABELS_14)
    final_val_metrics = evaluate_final_metrics(model, val_loader, device, disease_labels=DISEASE_LABELS_14)

    # Collect results
    results = {
        'model_name': MODEL_NAME,
        'dataset': 'full',
        'approach': '14_class_improved',
        'fixes_applied': [
            'per_class_alpha_focal_loss',
            'weighted_random_sampler',
            'medical_augmentation',
            '14_class_no_nofinding'
        ],
        'backbone': MODEL_CONFIGS[MODEL_NAME]['backbone'],
        'attention': MODEL_CONFIGS[MODEL_NAME]['attention'],
        'num_classes': NUM_CLASSES,
        'num_train_samples': len(train_df),
        'num_val_samples': len(val_df),
        'num_parameters': num_params,
        'model_size_mb': model_size_mb,
        'best_val_auc': trainer.best_val_auc,
        'best_epoch': trainer.best_epoch,
        'final_train_auc': final_train_metrics['AUC_macro'],
        'final_val_auc': final_val_metrics['AUC_macro'],
        'final_train_f1': final_train_metrics['F1_macro'],
        'final_val_f1': final_val_metrics['F1_macro'],
        'final_train_pr_auc': final_train_metrics['PR_AUC_macro'],
        'final_val_pr_auc': final_val_metrics['PR_AUC_macro'],
        'final_train_precision': final_train_metrics['Precision_macro'],
        'final_val_precision': final_val_metrics['Precision_macro'],
        'final_train_recall': final_train_metrics['Recall_macro'],
        'final_val_recall': final_val_metrics['Recall_macro'],
        'alpha_per_class': {DISEASE_LABELS_14[i]: float(alpha_per_class[i]) for i in range(14)},
        'training_time_minutes': train_time / 60,
        'timestamp': datetime.now().isoformat(),
        'checkpoint_dir': str(checkpoint_dir)
    }

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"✅ TRAINING COMPLETE: {MODEL_NAME} (14-class improved)")
    print(f"{'=' * 70}")
    print(f"   Best Val AUC: {results['best_val_auc']:.4f} (epoch {results['best_epoch']})")
    print(f"   Final Val AUC: {results['final_val_auc']:.4f}")
    print(f"   Final Val F1: {results['final_val_f1']:.4f}")
    print(f"   Final Val PR-AUC: {results['final_val_pr_auc']:.4f}")
    print(f"   Final Val Precision: {results['final_val_precision']:.4f}")
    print(f"   Final Val Recall: {results['final_val_recall']:.4f}")
    print(f"   Training time: {results['training_time_minutes']:.1f} minutes")
    print(f"   Checkpoints: {checkpoint_dir}")

    # Save results
    results_file = checkpoint_dir / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved: {results_file}")

    # Save history
    history_file = checkpoint_dir / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   History saved: {history_file}")


    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
