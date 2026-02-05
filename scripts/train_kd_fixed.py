"""
Fixed Knowledge Distillation Training Script
=============================================
Fixes the bias initialization and loss weighting issues that caused
the model to predict all diseases as positive.

Key fixes:
1. Proper final layer bias initialization based on class frequencies
2. Reduced positive weight smoothing (alpha=0.25 instead of 0.5)
3. Focal loss option for better imbalance handling
4. Validation checks during training
"""

import sys
from pathlib import Path
import time
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.models.teacher_model import create_teacher_model
from ml.models.student_model import create_student_model, MODEL_CONFIGS
from ml.data.loader import get_balanced_data_loaders
from ml.data.preprocessing import get_medical_transforms
from ml.data.augmentation import get_augmentation_pipeline
from ml.training.kd_trainer import KDTrainer
from ml.training.kd_losses import DistillationLoss, FocalDistillationLoss
from ml.training.losses import calculate_pos_weights
from config.disease_labels import DISEASE_LABELS
from scripts.training_utils import evaluate_final_metrics


def compute_class_frequencies(train_csv: Path) -> tuple:
    """
    Compute class frequencies for bias initialization and loss weighting
    
    Returns:
        (positive_counts, negative_counts, pos_weights)
    """
    train_df = pd.read_csv(train_csv)
    
    # Count positive and negative examples per disease
    positive_counts = np.zeros(len(DISEASE_LABELS))
    total_samples = len(train_df)
    
    for label_str in train_df['Finding Labels']:
        if pd.isna(label_str) or label_str == 'No Finding':
            continue
        labels = label_str.split('|')
        for label in labels:
            label = label.strip()
            if label in DISEASE_LABELS:
                idx = DISEASE_LABELS.index(label)
                positive_counts[idx] += 1
    
    negative_counts = total_samples - positive_counts
    
    return positive_counts, negative_counts, total_samples


def initialize_final_layer_bias(model: nn.Module, positive_counts: np.ndarray, 
                                  negative_counts: np.ndarray) -> None:
    """
    Initialize final classification layer bias to prior probabilities
    
    For binary classification with imbalanced data:
    bias_i = log(P(y_i=1) / P(y_i=0)) = log(n_pos / n_neg)
    
    This prevents the model from starting with a bias toward predicting
    all positives or all negatives.
    """
    # Find the final linear layer
    final_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Store the last linear layer
            final_layer = module
    
    if final_layer is None:
        print("⚠️  Warning: Could not find final linear layer for bias initialization")
        return
    
    # Calculate log odds for each class
    # Add smoothing to avoid log(0)
    log_odds = np.log((positive_counts + 1) / (negative_counts + 1))
    
    # Set bias
    with torch.no_grad():
        if final_layer.bias is not None:
            final_layer.bias.copy_(torch.tensor(log_odds, dtype=torch.float32))
            print(f"\n✓ Initialized final layer bias to log-odds:")
            print(f"  Range: [{log_odds.min():.4f}, {log_odds.max():.4f}]")
            print(f"  Mean: {log_odds.mean():.4f}")
            for i, disease in enumerate(DISEASE_LABELS):
                print(f"    {disease:<25} pos={int(positive_counts[i]):>6}  "
                      f"neg={int(negative_counts[i]):>6}  bias={log_odds[i]:>7.4f}")


def compute_pos_weights_fixed(positive_counts: np.ndarray, negative_counts: np.ndarray,
                                device: torch.device, alpha: float = 0.25) -> torch.Tensor:
    """
    Compute positive class weights with reduced smoothing
    
    Using alpha=0.25 instead of 0.5 reduces the weight difference between
    common and rare classes, preventing extreme bias.
    """
    total_samples = positive_counts.sum() + negative_counts.sum()
    label_counts_tensor = torch.tensor(positive_counts, dtype=torch.float32)
    
    pos_weights = calculate_pos_weights(
        label_counts_tensor, 
        total_samples=int(total_samples),
        smoothing=1.0,
        alpha=alpha  # Reduced from 0.5 to 0.25
    )
    
    print(f"\n✓ Computed positive class weights (alpha={alpha}):")
    print(f"  Range: [{pos_weights.min():.4f}, {pos_weights.max():.4f}]")
    print(f"  Mean: {pos_weights.mean():.4f}")
    
    return pos_weights.to(device)


def validate_predictions(model: nn.Module, loader, device: torch.device, 
                          num_samples: int = 100) -> dict:
    """
    Validate that model doesn't predict all positives or all negatives
    
    Returns:
        dict with mean predictions per disease and warning flags
    """
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_samples // loader.batch_size:
                break
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    mean_preds = all_preds.mean(dim=0).numpy()
    
    warnings = []
    if (mean_preds > 0.9).all():
        warnings.append("⚠️  Model predicting all positives!")
    if (mean_preds < 0.1).all():
        warnings.append("⚠️  Model predicting all negatives!")
    
    return {
        'mean_predictions': mean_preds,
        'warnings': warnings
    }


def main():
    parser = argparse.ArgumentParser(description="Fixed Knowledge Distillation Training")
    parser.add_argument("--student_model", type=str, default="convnext_tiny_mhsa",
                        help="Student model architecture")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="KD loss weight (0=pure CE, 1=pure KD)")
    parser.add_argument("--loss_type", type=str, default="bce", choices=["bce", "focal"],
                        help="Loss function type: bce or focal")
    parser.add_argument("--pos_weight_alpha", type=float, default=0.25,
                        help="Positive weight smoothing (0.25=mild, 0.5=moderate, 1.0=strong)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Data loader workers")
    parser.add_argument("--early_stopping_patience", type=int, default=8,
                        help="Early stopping patience")
    parser.add_argument("--teacher_weights_path", type=str, default="",
                        help="Path to teacher model weights")
    parser.add_argument("--validate_every", type=int, default=5,
                        help="Validate predictions every N epochs")
    args = parser.parse_args()

    if args.student_model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown student model: {args.student_model}")

    print("\n" + "=" * 80)
    print("FIXED KNOWLEDGE DISTILLATION TRAINING")
    print("=" * 80)
    print(f"Student model: {args.student_model}")
    print(f"Loss type: {args.loss_type.upper()}")
    print(f"Temperature: {args.temperature}")
    print(f"Alpha (KD weight): {args.alpha}")
    print(f"Positive weight alpha: {args.pos_weight_alpha}")
    if args.loss_type == "focal":
        print(f"Focal gamma: {args.focal_gamma}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Paths
    clahe_cache_dir = project_root / "data" / "clahe_cache"
    train_csv = project_root / "data" / "splits" / "train.csv"
    val_csv = project_root / "data" / "splits" / "val.csv"
    test_csv = project_root / "data" / "splits" / "test.csv"

    if not clahe_cache_dir.exists():
        print("✗ ERROR: CLAHE cache not found.")
        print(f"  Expected at: {clahe_cache_dir}")
        return

    # Compute class frequencies
    print("Computing class frequencies...")
    positive_counts, negative_counts, total_samples = compute_class_frequencies(train_csv)
    
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
    print("\nLoading teacher model (CheXNet)...")
    teacher_weights_path = Path(args.teacher_weights_path) if args.teacher_weights_path else None
    teacher = create_teacher_model(
        device=device,
        chexnet_weights_path=teacher_weights_path,
        chexnet_weights_url='https://github.com/arnoweng/CheXNet/raw/master/model.pth.tar'
    )

    # Student
    print(f"Creating student model ({args.student_model})...")
    print("  ⚠️  Starting from RANDOM weights (not ImageNet pre-trained)")
    print("     Student will learn chest X-ray patterns from CheXNet teacher + ground truth")
    student = create_student_model(args.student_model, num_classes=14, pretrained=False)
    
    # Initialize final layer bias to log-odds
    print("\nInitializing final layer bias...")
    initialize_final_layer_bias(student, positive_counts, negative_counts)

    # Compute positive weights
    pos_weights = compute_pos_weights_fixed(
        positive_counts, 
        negative_counts, 
        device, 
        alpha=args.pos_weight_alpha
    )

    # Loss function
    print(f"\nCreating {args.loss_type.upper()} loss...")
    if args.loss_type == "focal":
        kd_loss = FocalDistillationLoss(
            temperature=args.temperature,
            alpha=args.alpha,
            gamma=args.focal_gamma,
            num_classes=14,
            pos_weights=pos_weights
        )
    else:
        kd_loss = DistillationLoss(
            temperature=args.temperature,
            alpha=args.alpha,
            num_classes=14,
            pos_weights=pos_weights
        )

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-5)

    # Checkpoint directory
    checkpoint_dir = project_root / "ml" / "models" / "checkpoints" / "kd_fixed" / args.student_model
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

    # Validate initial predictions
    print("\n" + "=" * 80)
    print("INITIAL PREDICTION CHECK")
    print("=" * 80)
    initial_check = validate_predictions(student, val_loader, device, num_samples=100)
    print("Mean predictions per disease:")
    for i, disease in enumerate(DISEASE_LABELS):
        print(f"  {disease:<25} {initial_check['mean_predictions'][i]:.4f}")
    if initial_check['warnings']:
        for warning in initial_check['warnings']:
            print(warning)
    else:
        print("✓ Initial predictions look balanced")

    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    start_time = time.time()
    
    try:
        trainer.train(
            num_epochs=args.epochs,
            scheduler=scheduler,
            early_stopping_patience=args.early_stopping_patience,
            verbose=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    train_time = time.time() - start_time

    # Final validation check
    print("\n" + "=" * 80)
    print("FINAL PREDICTION CHECK")
    print("=" * 80)
    final_check = validate_predictions(student, val_loader, device, num_samples=100)
    print("Mean predictions per disease:")
    for i, disease in enumerate(DISEASE_LABELS):
        print(f"  {disease:<25} {final_check['mean_predictions'][i]:.4f}")
    if final_check['warnings']:
        for warning in final_check['warnings']:
            print(warning)
    else:
        print("✓ Final predictions look balanced")

    # Final metrics
    print("\nComputing final metrics...")
    final_train_metrics = evaluate_final_metrics(student, train_loader, device)
    final_val_metrics = evaluate_final_metrics(student, val_loader, device)

    # Save results
    results_dir = project_root / "experiments"
    results_path = results_dir / "kd_fixed_results.csv"
    results_dir.mkdir(exist_ok=True)

    results = {
        'student_model': args.student_model,
        'loss_type': args.loss_type,
        'pos_weight_alpha': args.pos_weight_alpha,
        'temperature': args.temperature,
        'alpha': args.alpha,
        'focal_gamma': args.focal_gamma if args.loss_type == "focal" else None,
        'best_val_auc': trainer.best_val_auc,
        'best_epoch': trainer.best_epoch,
        'final_train_auc': final_train_metrics['AUC_macro'],
        'final_val_auc': final_val_metrics['AUC_macro'],
        'final_train_f1': final_train_metrics['F1_macro'],
        'final_val_f1': final_val_metrics['F1_macro'],
        'final_train_pr_auc': final_train_metrics['PR_AUC_macro'],
        'final_val_pr_auc': final_val_metrics['PR_AUC_macro'],
        'training_time_minutes': train_time / 60,
        'timestamp': datetime.now().isoformat()
    }

    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        results_df = pd.concat([existing_df, pd.DataFrame([results])], ignore_index=True)
    else:
        results_df = pd.DataFrame([results])

    results_df.to_csv(results_path, index=False)
    
    # Copy best model to final directory
    final_dir = project_root / "ml" / "models" / "checkpoints" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    best_checkpoint = checkpoint_dir / "best_model.pth"
    if best_checkpoint.exists():
        import shutil
        final_path = final_dir / f"X-Lite_fixed_{args.student_model}.pth"
        shutil.copy(best_checkpoint, final_path)
        print(f"\n✅ Best model copied to: {final_path}")

    print(f"\n✅ Training complete!")
    print(f"   Best validation AUC: {trainer.best_val_auc:.4f}")
    print(f"   Training time: {train_time/60:.1f} minutes")
    print(f"   Results saved: {results_path}")


if __name__ == "__main__":
    main()
