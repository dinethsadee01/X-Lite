"""
Knowledge Distillation Training Script
======================================
Trains a student model using a CheXNet teacher with KD loss.
"""

import sys
from pathlib import Path
import time
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.models.teacher_model import create_teacher_model
from ml.models.student_model import create_student_model, MODEL_CONFIGS
from ml.data.loader import get_balanced_data_loaders
from ml.data.preprocessing import get_medical_transforms
from ml.data.augmentation import get_augmentation_pipeline
from ml.training.kd_trainer import KDTrainer
from ml.training.kd_losses import DistillationLoss
from ml.training.losses import calculate_pos_weights
from config.disease_labels import DISEASE_LABELS
from scripts.training_utils import evaluate_final_metrics


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
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    parser.add_argument("--student_model", type=str, default="convnext_tiny_mhsa")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--teacher_weights_path", type=str, default="")
    args = parser.parse_args()

    student_model_name = args.student_model
    if student_model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown student model: {student_model_name}")

    print("\n" + "=" * 70)
    print("KNOWLEDGE DISTILLATION TRAINING")
    print("=" * 70)
    print(f"Teacher: CheXNet (DenseNet121)")
    print(f"Student: {student_model_name}")
    print(f"Temperature: {args.temperature}")
    print(f"Alpha: {args.alpha}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print("=" * 70)

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

    # Teacher (CheXNet)
    teacher_weights_path = Path(args.teacher_weights_path) if args.teacher_weights_path else None
    teacher = create_teacher_model(
        device=device,
        chexnet_weights_path=teacher_weights_path,
        chexnet_weights_url='https://github.com/arnoweng/CheXNet/raw/master/model.pth.tar'
    )

    # Student
    student = create_student_model(student_model_name, num_classes=14, pretrained=True)

    # Loss
    pos_weights = compute_pos_weights(train_csv, device)
    kd_loss = DistillationLoss(
        temperature=args.temperature,
        alpha=args.alpha,
        num_classes=14,
        pos_weights=pos_weights
    )

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-5)

    # Checkpoint directory
    checkpoint_dir = project_root / "ml" / "models" / "checkpoints" / "kd" / student_model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Trainer
    trainer = KDTrainer(
        student_model=student,
        teacher_model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=kd_loss,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        use_amp=True
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    # Train
    start_time = time.time()
    trainer.train(
        num_epochs=args.epochs,
        scheduler=scheduler,
        early_stopping_patience=args.early_stopping_patience,
        verbose=True
    )
    train_time = time.time() - start_time

    # Final metrics
    print("\nComputing final metrics (AUC, F1, PR-AUC, Precision, Recall)...")
    final_train_metrics = evaluate_final_metrics(student, train_loader, device)
    final_val_metrics = evaluate_final_metrics(student, val_loader, device)

    # Save results
    results_dir = project_root / "experiments"
    results_path = results_dir / "kd_results.csv"
    results_dir.mkdir(exist_ok=True)

    results = {
        'student_model': student_model_name,
        'backbone': MODEL_CONFIGS[student_model_name]['backbone'],
        'attention': MODEL_CONFIGS[student_model_name]['attention'],
        'temperature': args.temperature,
        'alpha': args.alpha,
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
        'timestamp': datetime.now().isoformat()
    }

    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        results_df = pd.concat([existing_df, pd.DataFrame([results])], ignore_index=True)
    else:
        results_df = pd.DataFrame([results])

    results_df.to_csv(results_path, index=False)
    print(f"\n✅ KD results saved: {results_path}")


if __name__ == "__main__":
    main()
