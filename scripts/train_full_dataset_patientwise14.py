"""
Full dataset training script for 14-class setup (without No_Finding).
This does not modify the existing 15-class pipeline.
"""

import json
import sys
import time
from typing import Any
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.disease_labels14 import DISEASE_LABELS14, NUM_CLASSES14
from ml.data.augmentation import get_augmentation_pipeline
from ml.data.loader14 import ChestXrayDataset14
from ml.data.preprocessing import get_medical_transforms
from ml.models.student_model import MODEL_CONFIGS, create_student_model
from ml.training.losses import FocalLoss, WeightedBCEWithLogitsLoss, calculate_pos_weights
from ml.training.trainer14 import create_trainer14
from scripts.training_utils14 import evaluate_final_metrics14

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = 'efficientnet_b0_performer'
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
EARLY_STOPPING_PATIENCE = 8
GRADIENT_CLIP_VAL = 5.0
USE_FOCAL_LOSS = True
CHECKPOINT_SUFFIX = '_full_dataset_14class_patientwise_lol'
SAVE_EACH_EPOCH_CHECKPOINT = True
# ============================================================


def main():
    print('\n' + '=' * 70)
    print('FULL DATASET TRAINING (14-CLASS)')
    print('=' * 70)
    print(f'Model: {MODEL_NAME}')
    print(f'Epochs: {NUM_EPOCHS} (early stopping patience={EARLY_STOPPING_PATIENCE})')
    print(f'Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}')
    print('Classes: 14 (No_Finding removed)')
    print('=' * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        cuda_version = getattr(getattr(torch, 'version', None), 'cuda', 'unknown')
        print(f'CUDA Version: {cuda_version}')

    clahe_cache_dir = project_root / 'data' / 'clahe_cache'
    train_csv = project_root / 'data' / 'splits' / 'train_df.csv'
    val_csv = project_root / 'data' / 'splits' / 'val_df.csv'

    if not clahe_cache_dir.exists():
        print('\nCLAHE cache not found.')
        print(f'Expected at: {clahe_cache_dir}')
        print('Run: python scripts/precompute_clahe.py')
        return

    print('\nPreparing data loaders...')
    train_aug = get_augmentation_pipeline(augmentation_strength='medium')
    val_transform = get_medical_transforms(use_clahe=False, use_denoising=False)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    if 'Image Index' in train_df.columns:
        train_df = train_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
    if 'Image Index' in val_df.columns:
        val_df = val_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})

    print(f'Train samples: {len(train_df):,}')
    print(f'Val samples: {len(val_df):,}')

    train_dataset = ChestXrayDataset14(str(clahe_cache_dir), train_df, transform=train_aug, is_training=True)
    val_dataset = ChestXrayDataset14(str(clahe_cache_dir), val_df, transform=val_transform, is_training=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')

    print('\n' + '=' * 70)
    print(f'TRAINING: {MODEL_NAME} (14-CLASS)')
    print('=' * 70)

    model = create_student_model(MODEL_NAME, num_classes=NUM_CLASSES14, pretrained=True)
    num_params = model.get_num_params()
    model_size_mb = model.get_model_size_mb()

    print(f'Parameters: {num_params:,}')
    print(f'Model Size: {model_size_mb:.2f} MB')
    print(f"Backbone: {MODEL_CONFIGS[MODEL_NAME]['backbone']}")
    print(f"Attention: {MODEL_CONFIGS[MODEL_NAME]['attention']}")

    label_counts = np.zeros(NUM_CLASSES14)
    for label_str in train_df['labels']:
        if pd.isna(label_str) or label_str == 'No Finding':
            continue

        labels = label_str.split('|')
        for label in labels:
            label = label.strip()
            if label in DISEASE_LABELS14:
                idx = DISEASE_LABELS14.index(label)
                label_counts[idx] += 1

    label_counts_tensor = torch.tensor(label_counts)
    pos_weights = calculate_pos_weights(label_counts_tensor, total_samples=len(train_df)).to(device)

    print('\nClass distribution (bottom 5 and top 5):')
    counts_with_labels = [(DISEASE_LABELS14[i], int(label_counts[i])) for i in range(NUM_CLASSES14)]
    counts_with_labels.sort(key=lambda x: x[1])
    print('  Least common:')
    for label, count in counts_with_labels[:5]:
        print(f'    {label:<20} {count:,}')
    print('  Most common:')
    for label, count in counts_with_labels[-5:]:
        print(f'    {label:<20} {count:,}')

    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        print('Loss: Focal Loss (alpha=0.25, gamma=2.0)')
    else:
        criterion = WeightedBCEWithLogitsLoss(pos_weights=pos_weights)
        print('Loss: Weighted BCE with class weights')

    checkpoint_dir = project_root / 'ml' / 'models' / 'new checkpoints 14' / f'{MODEL_NAME}{CHECKPOINT_SUFFIX}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f'Checkpoints: {checkpoint_dir}')
    print(f'Learning Rate: {LEARNING_RATE}')
    print(f'Gradient Clipping: {GRADIENT_CLIP_VAL}')

    trainer = create_trainer14(
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
        num_classes=NUM_CLASSES14,
    )

    if SAVE_EACH_EPOCH_CHECKPOINT:
        epoch_ckpt_dir = checkpoint_dir / 'epoch_checkpoints'
        epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
        original_save_checkpoint = trainer.save_checkpoint

        def save_checkpoint_with_epoch_copy(epoch, is_best=False, metrics=None):
            original_save_checkpoint(epoch, is_best, metrics)

            val_auc = 0.0
            if isinstance(metrics, dict):
                val_auc = float(metrics.get('auc_macro', 0.0))

            ckpt_path = epoch_ckpt_dir / f'model_epoch_{epoch:03d}_{val_auc:.4f}.pth'
            torch.save(trainer.model.state_dict(), ckpt_path)
            print(f'  Saved epoch checkpoint: {ckpt_path.name}')

        trainer.save_checkpoint = save_checkpoint_with_epoch_copy
        print(f'Per-epoch checkpoints enabled: {epoch_ckpt_dir}')

    scheduler: Any = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode='max',
        factor=0.5,
        patience=5,
    )

    start_time = time.time()
    history = trainer.train(
        num_epochs=NUM_EPOCHS,
        scheduler=scheduler,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        verbose=True,
    )
    train_time = time.time() - start_time

    print('\nComputing final metrics (AUC, F1, PR-AUC, Precision, Recall)...')
    final_train_metrics = evaluate_final_metrics14(model, train_loader, device)
    final_val_metrics = evaluate_final_metrics14(model, val_loader, device)

    results = {
        'model_name': MODEL_NAME,
        'dataset': 'full',
        'setup': '14_class_without_no_finding',
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
        'checkpoint_dir': str(checkpoint_dir),
    }

    print('\n' + '=' * 70)
    print(f'TRAINING COMPLETE: {MODEL_NAME} (14-CLASS)')
    print('=' * 70)
    print(f"Best Val AUC: {results['best_val_auc']:.4f} (epoch {results['best_epoch']})")
    print(f"Final Val AUC: {results['final_val_auc']:.4f}")
    print(f"Final Val F1: {results['final_val_f1']:.4f}")
    print(f"Final Val PR-AUC: {results['final_val_pr_auc']:.4f}")
    print(f"Final Val Precision: {results['final_val_precision']:.4f}")
    print(f"Final Val Recall: {results['final_val_recall']:.4f}")
    print(f"Training time: {results['training_time_minutes']:.1f} minutes")
    print(f'Checkpoints: {checkpoint_dir}')

    results_file = checkpoint_dir / 'training_results14.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved: {results_file}')

    history_file = checkpoint_dir / 'training_history14.json'
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f'History saved: {history_file}')

    print('=' * 70)


if __name__ == '__main__':
    main()
