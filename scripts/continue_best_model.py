"""
Continue training a single model for a few extra epochs from last checkpoint.

Usage:
  python scripts/continue_best_model.py --model_name convnext_tiny_mhsa --extra_epochs 5
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
import torch

from ml.models.student_model import create_student_model, MODEL_CONFIGS
from ml.data.loader import get_balanced_data_loaders
from ml.data.preprocessing import get_medical_transforms
from ml.data.augmentation import get_augmentation_pipeline
from ml.training.trainer import create_trainer
from ml.training.losses import WeightedBCEWithLogitsLoss, calculate_pos_weights
from config.disease_labels import DISEASE_LABELS


def compute_pos_weights(train_csv: Path, device: torch.device) -> torch.Tensor:
    train_df = pd.read_csv(train_csv)

    if 'Image Index' in train_df.columns:
        train_df = train_df.rename(columns={
            'Image Index': 'image_id',
            'Finding Labels': 'labels'
        })

    label_counts = np.zeros(len(DISEASE_LABELS))
    for label_str in train_df['labels']:
        if pd.isna(label_str) or label_str == 'No Finding':
            continue
        labels = label_str.split('|')
        for label in labels:
            label = label.strip()
            if label in DISEASE_LABELS:
                idx = DISEASE_LABELS.index(label)
                label_counts[idx] += 1

    label_counts_tensor = torch.tensor(label_counts)
    pos_weights = calculate_pos_weights(label_counts_tensor, total_samples=len(train_df))
    return pos_weights.to(device)


def main():
    parser = argparse.ArgumentParser(description="Continue training a single model from last checkpoint")
    parser.add_argument("--model_name", type=str, default="convnext_tiny_mhsa")
    parser.add_argument("--extra_epochs", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    model_name = args.model_name
    extra_epochs = args.extra_epochs

    print("\n" + "=" * 70)
    print("CONTINUE TRAINING FROM LAST CHECKPOINT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Extra epochs: {extra_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print("=" * 70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Paths
    clahe_cache_dir = project_root / "data" / "clahe_cache"
    train_csv = project_root / "data" / "splits" / "train.csv"
    val_csv = project_root / "data" / "splits" / "val.csv"
    test_csv = project_root / "data" / "splits" / "test.csv"

    if not clahe_cache_dir.exists():
        print("\n✗ ERROR: CLAHE cache not found.")
        print(f"  Expected at: {clahe_cache_dir}")
        return

    # Data transforms
    train_aug = get_augmentation_pipeline(augmentation_strength='medium')
    val_transform = get_medical_transforms(use_clahe=False, use_denoising=False)

    loaders = get_balanced_data_loaders(
        data_dir=str(clahe_cache_dir),
        train_split_csv=str(train_csv),
        val_split_csv=str(val_csv),
        test_split_csv=str(test_csv),
        train_transform=train_aug,
        val_transform=val_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=False
    )

    train_loader = loaders['train']
    val_loader = loaders['val']

    # Model + loss
    model = create_student_model(model_name, num_classes=14, pretrained=True)
    pos_weights = compute_pos_weights(train_csv, device)
    criterion = WeightedBCEWithLogitsLoss(pos_weights=pos_weights)

    # Checkpoint dir
    checkpoint_dir = project_root / "ml" / "models" / "checkpoints" / model_name
    last_ckpt = checkpoint_dir / "last_checkpoint.pth"

    if not last_ckpt.exists():
        print(f"\n✗ ERROR: No last_checkpoint.pth found at {last_ckpt}")
        return

    # Trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        learning_rate=1e-4,
        weight_decay=1e-5,
        checkpoint_dir=checkpoint_dir,
        device=device,
        use_amp=True
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    # Load checkpoint
    start_epoch = trainer.load_checkpoint(last_ckpt)

    # Restore best_epoch + epochs_without_improvement from history
    val_auc = trainer.history.get('val_auc', [])
    if val_auc:
        best_epoch = int(np.argmax(val_auc)) + 1
        trainer.best_epoch = best_epoch
        trainer.best_val_auc = float(max(val_auc))
        trainer.epochs_without_improvement = len(val_auc) - best_epoch
    else:
        trainer.best_epoch = start_epoch
        trainer.epochs_without_improvement = 0

    print(f"\nLoaded checkpoint at epoch {start_epoch}")
    print(f"Best AUC so far: {trainer.best_val_auc:.4f} (epoch {trainer.best_epoch})")
    print(f"Epochs without improvement: {trainer.epochs_without_improvement}")

    # Continue training for extra epochs
    target_epoch = start_epoch + extra_epochs

    for epoch in range(start_epoch + 1, target_epoch + 1):
        epoch_start = time.time()

        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate_epoch(epoch)

        # Update history
        trainer.history['train_loss'].append(train_metrics['loss'])
        trainer.history['val_loss'].append(val_metrics['loss'])
        trainer.history['train_auc'].append(train_metrics['auc_macro'])
        trainer.history['val_auc'].append(val_metrics['auc_macro'])
        trainer.history['train_f1'].append(train_metrics['f1_macro'])
        trainer.history['val_f1'].append(val_metrics['f1_macro'])
        trainer.history['learning_rates'].append(
            trainer.optimizer.param_groups[0]['lr']
        )

        # Scheduler step
        scheduler.step(val_metrics['auc_macro'])

        # Check best
        is_best = val_metrics['auc_macro'] > trainer.best_val_auc
        if is_best:
            trainer.best_val_auc = val_metrics['auc_macro']
            trainer.best_epoch = epoch
            trainer.epochs_without_improvement = 0
        else:
            trainer.epochs_without_improvement += 1

        # Save checkpoint
        trainer.save_checkpoint(epoch, is_best, val_metrics)

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | AUC: {train_metrics['auc_macro']:.4f} | F1: {train_metrics['f1_macro']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} | AUC: {val_metrics['auc_macro']:.4f} | F1: {val_metrics['f1_macro']:.4f}")
        print(f"  LR: {trainer.optimizer.param_groups[0]['lr']:.6f}")

        if is_best:
            print(f"  🏆 New best AUC: {trainer.best_val_auc:.4f}")

        if trainer.epochs_without_improvement >= args.early_stopping_patience:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs")
            print(f"  Best AUC: {trainer.best_val_auc:.4f} at epoch {trainer.best_epoch}")
            break

    # Save updated history
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(trainer.history, f, indent=2)

    print("\n" + "=" * 70)
    print("CONTINUE TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation AUC: {trainer.best_val_auc:.4f} at epoch {trainer.best_epoch}")
    print(f"History saved: {history_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
