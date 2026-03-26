"""
Full Dataset Training Script (15-Class System - Fixed)
=======================================================
Trains EfficientNet-B0 Performer on the FULL training set with 15 classes.

IMPORTANT: Now includes "No Finding" as explicit 15th class to fix overprediction issue.
FIXED: Reduced learning rate, increased gradient clipping, switched to Focal Loss for stability.

Based on train_baseline.py but adapted for single model, full dataset training.

Configuration:
- Model: efficientnet_b0_performer  
- Classes: 15 (14 diseases + No_Finding)
- Dataset: Full training set (precomputed CLAHE cached images)
- Epochs: 50 (early stopping patience=8)
- Batch size: 32
- Optimizer: AdamW (lr=5e-5, weight_decay=1e-5)  [REDUCED from 1e-4]
- Loss: Focal Loss (α=0.25, γ=2.0)  [SWITCHED from WeightedBCE]
- Gradient Clipping: 5.0  [INCREASED from 1.0]
- Augmentation: Medium strength on CLAHE-preprocessed images

Note: CLAHE images must be precomputed first:
  python scripts/precompute_clahe.py

Output:
- Checkpoints: ml/models/new checkpoints/efficientnet_b0_performer_full_dataset_15class/
- Training history & results in checkpoint directory

Usage:
  python scripts/train_full_dataset_patientwise.py
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
from ml.training.losses import WeightedBCEWithLogitsLoss, FocalLoss, calculate_pos_weights
from config.disease_labels import DISEASE_LABELS
from scripts.training_utils import evaluate_final_metrics


# ============================================================
# CONFIGURATION - Edit these values to change training settings
# ============================================================
MODEL_NAME = 'efficientnet_b0_performer'
NUM_EPOCHS = 50
BATCH_SIZE = 64  # Increased from 32 for faster training on full dataset (monitor GPU memory)
LEARNING_RATE = 5e-5  # Reduced from 1e-4 for 15-class stability
EARLY_STOPPING_PATIENCE = 8
GRADIENT_CLIP_VAL = 5.0  # Increased from 1.0 for more aggressive gradient clipping
USE_FOCAL_LOSS = True  # Use Focal Loss instead of BCE for better stability
CHECKPOINT_SUFFIX = '_full_dataset_15class_patientwise_lol'  # 15-class system (14 diseases + No_Finding)
SAVE_EACH_EPOCH_CHECKPOINT = True  # Save model_epoch_{epoch}_{auc}.pth in addition to best/last
# ============================================================


def main():
    """Train single model on full dataset (adapted from train_baseline.py)"""
    print("\n" + "=" * 70)
    print("FULL DATASET TRAINING")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {NUM_EPOCHS} (early stopping patience={EARLY_STOPPING_PATIENCE})")
    print(f"Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"Loss: Weighted BCE with class weights")
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

    # Data transforms (same as train_baseline.py)
    print("\nPreparing data loaders...")
    train_aug = get_augmentation_pipeline(augmentation_strength='medium')
    val_transform = get_medical_transforms(use_clahe=False, use_denoising=False)

    # Load CSVs (train and val ONLY - test data kept completely unseen)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    if 'Image Index' in train_df.columns:
        train_df = train_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
    if 'Image Index' in val_df.columns:
        val_df = val_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})

    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples: {len(val_df):,}")

    # Create datasets from CLAHE cache
    train_dataset = ChestXrayDataset(
        str(clahe_cache_dir), train_df, transform=train_aug, is_training=True
    )
    val_dataset = ChestXrayDataset(
        str(clahe_cache_dir), val_df, transform=val_transform, is_training=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ---- Train model (same pattern as train_baseline.py train_single_model) ----
    print("\n" + "=" * 70)
    print(f"TRAINING: {MODEL_NAME}")
    print("=" * 70)

    # Create model (15 classes: 14 diseases + No_Finding)
    from config import NUM_CLASSES
    model = create_student_model(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True)
    num_params = model.get_num_params()
    model_size_mb = model.get_model_size_mb()

    print(f"Parameters: {num_params:,}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Backbone: {MODEL_CONFIGS[MODEL_NAME]['backbone']}")
    print(f"Attention: {MODEL_CONFIGS[MODEL_NAME]['attention']}")

    # Calculate class weights for loss (now includes No_Finding as 15th class)
    label_counts = np.zeros(len(DISEASE_LABELS))  # 15 classes
    for label_str in train_df['labels']:
        if pd.isna(label_str) or label_str == 'No Finding':
            # Count No_Finding (index 14)
            label_counts[14] += 1
            continue
        # Count disease classes (indices 0-13)
        labels = label_str.split('|')
        for label in labels:
            label = label.strip()
            if label in DISEASE_LABELS:
                idx = DISEASE_LABELS.index(label)
                label_counts[idx] += 1

    label_counts_tensor = torch.tensor(label_counts)
    pos_weights = calculate_pos_weights(label_counts_tensor, total_samples=len(train_df))
    pos_weights = pos_weights.to(device)

    # Print class distribution (verify No_Finding is counted)
    print(f"\nClass distribution (showing bottom 5 and top 5):")
    counts_with_labels = [(DISEASE_LABELS[i], int(label_counts[i])) for i in range(len(DISEASE_LABELS))]
    counts_with_labels.sort(key=lambda x: x[1])
    print("  Least common:")
    for label, count in counts_with_labels[:5]:
        print(f"    {label:<20} {count:,}")
    print("  Most common:")
    for label, count in counts_with_labels[-5:]:
        print(f"    {label:<20} {count:,}")

    # Use Focal Loss for better stability with 15 classes
    if USE_FOCAL_LOSS:
        from ml.training.losses import FocalLoss
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        print("Loss: Focal Loss (α=0.25, γ=2.0)")
    else:
        criterion = WeightedBCEWithLogitsLoss(pos_weights=pos_weights)
        print("Loss: Weighted BCE with class weights")

    # Checkpoint directory (separate from baseline checkpoints)
    checkpoint_dir = project_root / "ml" / "models" / "new checkpoints" / f"{MODEL_NAME}{CHECKPOINT_SUFFIX}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Gradient Clipping: {GRADIENT_CLIP_VAL}")

    # Create trainer (same as train_baseline.py)
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
        gradient_clip_val=GRADIENT_CLIP_VAL,  # Pass increased clipping value
        num_classes=NUM_CLASSES  # Pass correct number of classes (15)
    )

    # Optional: also save a uniquely named checkpoint file for every epoch.
    # Keeps trainer's default best/last behavior intact while adding audit-friendly snapshots.
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

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    # Train
    start_time = time.time()
    history = trainer.train(
        num_epochs=NUM_EPOCHS,
        scheduler=scheduler,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        verbose=True
    )
    train_time = time.time() - start_time

    # Evaluate final metrics (same as train_baseline.py)
    print("\nComputing final metrics (AUC, F1, PR-AUC, Precision, Recall)...")
    final_train_metrics = evaluate_final_metrics(model, train_loader, device)
    final_val_metrics = evaluate_final_metrics(model, val_loader, device)

    # Collect results
    results = {
        'model_name': MODEL_NAME,
        'dataset': 'full',
        'backbone': MODEL_CONFIGS[MODEL_NAME]['backbone'],
        'attention': MODEL_CONFIGS[MODEL_NAME]['attention'],
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
        'training_time_minutes': train_time / 60,
        'timestamp': datetime.now().isoformat(),
        'checkpoint_dir': str(checkpoint_dir)
    }

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"✅ TRAINING COMPLETE: {MODEL_NAME}")
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
