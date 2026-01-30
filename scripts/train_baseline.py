"""
Baseline Student Training Script
=================================
Trains all 6 hybrid CNN-Transformer student models on subset of data.

Purpose: Establish baselines before knowledge distillation experiments.

Models Trained:
1. efficientnet_b0_mhsa
2. efficientnet_b0_performer
3. convnext_tiny_mhsa
4. convnext_tiny_performer
5. mobilenet_v3_large_mhsa
6. mobilenet_v3_large_performer

Training Configuration:
- Dataset: 20% of training data (stratified sampling)
- Epochs: 30 (with early stopping patience=10)
- Batch size: 32
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Loss: WeightedBCEWithLogitsLoss (class-weighted)
- Augmentation: Medium strength (HFlip, Rotation, Brightness/Contrast)
- Mixed precision: Enabled (AMP)

Output:
- Checkpoints: ml/models/checkpoints/{model_name}/
- Results: experiments/baseline_results.csv
- Metrics: Per-model AUC-ROC, F1, parameters, size, training time
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import Subset
import pandas as pd
import time
from datetime import datetime
import numpy as np
import json
from sklearn.metrics import precision_recall_curve, auc as pr_auc

from ml.models.student_model import create_student_model, MODEL_CONFIGS
from ml.data.loader import get_balanced_data_loaders
from ml.data.preprocessing import get_medical_transforms
from ml.data.augmentation import get_augmentation_pipeline
from ml.training.trainer import create_trainer
from ml.training.losses import WeightedBCEWithLogitsLoss, calculate_pos_weights
from ml.training.metrics import compute_metrics
from config.disease_labels import DISEASE_LABELS
from scripts.training_utils import (
    evaluate_final_metrics,
    load_training_progress,
    save_training_progress
)


# ============================================================================
# COMMENTED OUT: 20% Subset Creation (Now using full dataset)
# ============================================================================
# def create_subset_loaders(
#     full_train_loader,
#     full_val_loader,
#     subset_fraction: float = 0.2,
#     seed: int = 42
# ):
#     """
#     Create subset of training data for faster baseline experiments
#     
#     Args:
#         full_train_loader: Full training data loader
#         full_val_loader: Full validation data loader
#         subset_fraction: Fraction of training data to use
#         seed: Random seed for reproducibility
#     
#     Returns:
#         tuple: (subset_train_loader, val_loader)
#     """
#     # Get dataset from loader
#     train_dataset = full_train_loader.dataset
#     
#     # Create stratified subset
#     np.random.seed(seed)
#     subset_size = int(len(train_dataset) * subset_fraction)
#     indices = np.random.choice(len(train_dataset), subset_size, replace=False)
#     
#     subset_dataset = Subset(train_dataset, indices)
#     
#     # Create new loader with subset
#     subset_loader = torch.utils.data.DataLoader(
#         subset_dataset,
#         batch_size=full_train_loader.batch_size,
#         shuffle=True,
#         num_workers=full_train_loader.num_workers,
#         pin_memory=full_train_loader.pin_memory,
#         drop_last=True
#     )
#     
#     return subset_loader, full_val_loader
# ============================================================================


def train_single_model(
    model_name: str,
    train_loader,
    val_loader,
    device: torch.device,
    num_epochs: int = 30,
    learning_rate: float = 1e-4,
    use_clahe: bool = True
) -> dict:
    """
    Train a single student model
    
    Args:
        model_name: Name of model architecture
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Training device
        num_epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        use_clahe: Use CLAHE preprocessing
    
    Returns:
        dict: Training results (metrics, time, model info)
    """
    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)
    
    # Create model
    model = create_student_model(model_name, num_classes=14, pretrained=True)
    num_params = model.get_num_params()
    model_size_mb = model.get_model_size_mb()
    
    print(f"Parameters: {num_params:,}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Backbone: {MODEL_CONFIGS[model_name]['backbone']}")
    print(f"Attention: {MODEL_CONFIGS[model_name]['attention']}")
    
    # Calculate class weights for loss
    # Load training labels to compute weights
    train_df = pd.read_csv(project_root / "data" / "splits" / "train.csv")
    
    # Rename columns if needed
    if 'Image Index' in train_df.columns:
        train_df = train_df.rename(columns={
            'Image Index': 'image_id',
            'Finding Labels': 'labels'
        })
    
    # Calculate label counts
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
    pos_weights = pos_weights.to(device)
    
    # Loss function with class weights
    criterion = WeightedBCEWithLogitsLoss(pos_weights=pos_weights)
    
    # Checkpoint directory
    checkpoint_dir = project_root / "ml" / "models" / "checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        checkpoint_dir=checkpoint_dir,
        device=device,
        use_amp=True
    )
    
    # Learning rate scheduler
    # torch 2.4+ ReduceLROnPlateau signature drops the 'verbose' kwarg
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    
    # Train
    start_time = time.time()
    history = trainer.train(
        num_epochs=num_epochs,
        scheduler=scheduler,
        early_stopping_patience=5,
        verbose=True
    )
    train_time = time.time() - start_time
    
    # Evaluate final comprehensive metrics
    print("\nComputing final metrics (AUC, F1, PR-AUC, Precision, Recall)...")
    final_train_metrics = evaluate_final_metrics(model, train_loader, device)
    final_val_metrics = evaluate_final_metrics(model, val_loader, device)
    
    # Collect results
    results = {
        'model_name': model_name,
        'backbone': MODEL_CONFIGS[model_name]['backbone'],
        'attention': MODEL_CONFIGS[model_name]['attention'],
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
        'use_clahe': use_clahe,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n✅ Final Metrics:")
    print(f"   Val AUC: {results['final_val_auc']:.4f}")
    print(f"   Val F1: {results['final_val_f1']:.4f}")
    print(f"   Val PR-AUC: {results['final_val_pr_auc']:.4f}")
    print(f"   Val Precision: {results['final_val_precision']:.4f}")
    print(f"   Val Recall: {results['final_val_recall']:.4f}")
    
    return results


def main():
    """Run baseline training for all 6 student models"""
    print("\n" + "=" * 70)
    print("BASELINE STUDENT MODEL TRAINING (FULL DATASET)")
    print("=" * 70)
    print(f"Training 6 hybrid CNN-Transformer models")
    print(f"Dataset: 100% of training data (stratified, full 78,484 samples)")
    print(f"Epochs: 50 (early stopping patience=10)")
    print(f"Loss: Weighted BCE with class weights")
    print(f"Metrics: AUC, F1, PR-AUC, Precision, Recall")
    print(f"Resume: Automatically skips completed models (training_progress.json)")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Paths - use CLAHE cache for faster data loading
    clahe_cache_dir = project_root / "data" / "clahe_cache"
    train_csv = project_root / "data" / "splits" / "train.csv"
    val_csv = project_root / "data" / "splits" / "val.csv"
    test_csv = project_root / "data" / "splits" / "test.csv"

    # Validate cache exists
    if not clahe_cache_dir.exists():
        print("\n✗ ERROR: CLAHE cache not found.")
        print(f"  Expected at: {clahe_cache_dir}")
        print("  Run: python scripts/precompute_clahe.py")
        return
    
    # Data transforms
    print("\nPreparing data loaders...")
    
    # Training: Albumentations with medium augmentation (CLAHE already baked in)
    train_aug = get_augmentation_pipeline(augmentation_strength='medium')
    
    # Validation: basic transforms only (CLAHE already baked in)
    val_transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    
    # Create full data loaders from cached CLAHE images
    # num_workers can be >0 because cached images are plain files
    loaders = get_balanced_data_loaders(
        data_dir=str(clahe_cache_dir),
        train_split_csv=str(train_csv),
        val_split_csv=str(val_csv),
        test_split_csv=str(test_csv),
        train_transform=train_aug,
        val_transform=val_transform,
        batch_size=32,
        num_workers=4,
        use_weighted_sampler=False
    )
    
    # ========================================================================
    # USING FULL DATASET (Commented out 20% subset code)
    # ========================================================================
    # print("\nCreating 20% training subset...")
    # train_loader, val_loader = create_subset_loaders(
    #     loaders['train'],
    #     loaders['val'],
    #     subset_fraction=0.2,
    #     seed=42
    # )
    
    # Use full loaders directly
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    print(f"\n✅ Using FULL training dataset:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Train samples: ~{len(train_loader) * 32:,}")
    print(f"  Val samples: ~{len(val_loader) * 32:,}")
    
    # Load training progress (for resume capability)
    completed_models = load_training_progress()
    
    if completed_models:
        print(f"\n✅ Found {len(completed_models)} completed model(s): {', '.join(completed_models)}")
        print(f"   These will be skipped. Delete training_progress.json to retrain all.")
    
    # Results storage
    all_results = []
    
    # Load existing results if resuming
    results_dir = project_root / "experiments"
    results_path = results_dir / "baseline_results.csv"
    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        # Keep only completed models
        existing_df = existing_df[existing_df['model_name'].isin(completed_models)]
        all_results = existing_df.to_dict('records')
        print(f"\n✅ Loaded {len(all_results)} existing result(s) from baseline_results.csv")
    
    # Train each model
    models_to_train = [m for m in MODEL_CONFIGS.keys() if m not in completed_models]
    
    print(f"\n{'='*70}")
    print(f"TRAINING QUEUE: {len(models_to_train)} model(s) remaining")
    print(f"{'='*70}")
    for i, model_name in enumerate(models_to_train, 1):
        print(f"  {i}. {model_name}")
    print(f"{'='*70}\n")
    
    for model_name in models_to_train:
        try:
            print(f"\n\n{'#'*70}")
            print(f"# TRAINING MODEL {models_to_train.index(model_name) + 1}/{len(models_to_train)}: {model_name}")
            print(f"{'#'*70}\n")
            
            results = train_single_model(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_epochs=50,
                learning_rate=1e-4,
                use_clahe=True
            )
            all_results.append(results)
            
            # Mark as completed and save progress immediately
            completed_models.add(model_name)
            save_training_progress(completed_models)
            
            # Save intermediate results after each model
            temp_df = pd.DataFrame(all_results)
            temp_df.to_csv(results_path, index=False)
            print(f"\n✅ Results saved (intermediate): {results_path}")
            
        except Exception as e:
            print(f"\n✗ Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"\n⚠️  Training failed for {model_name}. Progress saved. You can resume later.")
            continue
    
    # Save final results
    if all_results:
        results_dir.mkdir(exist_ok=True)
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_path, index=False)
        
        print(f"\n\n{'='*70}")
        print(f"✅ FINAL RESULTS SAVED: {results_path}")
        print(f"{'='*70}\n")
        
        # Sort by best validation AUC
        results_df_sorted = results_df.sort_values('best_val_auc', ascending=False)
        
        print("\n" + "=" * 70)
        print("BASELINE TRAINING SUMMARY (FULL DATASET)")
        print("=" * 70)
        print(f"{'Model':<30} {'Val AUC':<10} {'Val F1':<10} {'PR-AUC':<10} {'Prec':<8} {'Rec':<8} {'Time':<8}")
        print("-" * 70)
        
        for _, row in results_df_sorted.iterrows():
            print(f"{row['model_name']:<30} "
                  f"{row['best_val_auc']:<10.4f} "
                  f"{row['final_val_f1']:<10.4f} "
                  f"{row['final_val_pr_auc']:<10.4f} "
                  f"{row['final_val_precision']:<8.4f} "
                  f"{row['final_val_recall']:<8.4f} "
                  f"{row['training_time_minutes']:<8.1f}")
        
        print("=" * 70)
        print(f"✅ Best model: {results_df_sorted.iloc[0]['model_name']} "
              f"(AUC: {results_df_sorted.iloc[0]['best_val_auc']:.4f})")
        print(f"✅ All {len(all_results)} models completed successfully!")
        print(f"✅ Results: {results_path}")
        print("=" * 70)
    else:
        print("\n⚠️  No models were trained (all already completed or errors occurred).")
        print("   Check training_progress.json or error logs above.")


if __name__ == "__main__":
    main()
