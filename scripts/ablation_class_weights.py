"""
Ablation Study: Focal Loss vs BCE
=================================
Compares standard BCE against Focal Loss (gamma=2.0, alpha=0.25) to handle imbalance
without manual class weights.

Experimental Setup:
- Same model: efficientnet_b0_mhsa (smallest, fastest)
- Same data: 20% training subset
- Same hyperparameters: lr, batch_size, optimizer, epochs
- Same augmentation: medium strength

Only Difference:
1. Baseline: Standard BCE, no weighted sampler
2. Focal: Focal Loss (alpha=0.25, gamma=2.0), no weighted sampler

Metrics:
- Overall AUC-ROC, F1
- Per-class AUC-ROC, F1

Purpose: Find a stable imbalance mitigation without extreme class weights.
"""

import sys
from pathlib import Path

# Add project root to path.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import Subset
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, f1_score

from ml.models.student_model import create_student_model
from ml.data.loader import get_balanced_data_loaders
from ml.data.augmentation import get_augmentation_pipeline
from ml.data.preprocessing import get_medical_transforms
from ml.training.losses import FocalLoss
from config.disease_labels import DISEASE_LABELS


def create_subset(loader, fraction=0.2, seed=42):
    """Create subset of data for faster experiments"""
    dataset = loader.dataset
    np.random.seed(seed)
    subset_size = int(len(dataset) * fraction)
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    
    subset_dataset = Subset(dataset, indices.tolist())
    subset_loader = torch.utils.data.DataLoader(
        subset_dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid pickle issues
        pin_memory=loader.pin_memory,
        drop_last=True
    )
    
    return subset_loader


def evaluate_per_class(model, loader, device):
    """Evaluate per-class AUC and F1 scores"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # Per-class metrics
    per_class_auc = []
    per_class_f1 = []
    
    for i in range(len(DISEASE_LABELS)):
        # AUC (skip if class has no positive samples)
        if targets[:, i].sum() > 0:
            try:
                auc = roc_auc_score(targets[:, i], preds[:, i])
            except:
                auc = 0.0
        else:
            auc = 0.0
        per_class_auc.append(auc)
        
        # F1 (use threshold 0.5)
        pred_binary = (preds[:, i] > 0.5).astype(int)
        f1 = f1_score(targets[:, i], pred_binary, zero_division=0)
        per_class_f1.append(f1)
    
    # Overall metrics (macro average)
    macro_auc = np.mean(per_class_auc)
    macro_f1 = np.mean(per_class_f1)
    
    return {
        'per_class_auc': per_class_auc,
        'per_class_f1': per_class_f1,
        'macro_auc': macro_auc,
        'macro_f1': macro_f1
    }


def train_one_config(
    config_name: str,
    train_loader,
    val_loader,
    loss_type: str,
    device: torch.device,
    num_epochs: int = 10
):
    """Train model with specific configuration"""
    
    print(f"\n{'='*70}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*70}")
    print(f"Loss Type: {loss_type}")
    print(f"Weighted Sampler: False")
    print(f"Epochs: {num_epochs}")
    
    # Create model
    model = create_student_model('efficientnet_b0_mhsa', num_classes=14, pretrained=True)
    model = model.to(device)
    
    # Loss function
    if loss_type == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print(f"✓ Using Focal Loss (alpha=0.25, gamma=2.0)")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"✓ Using standard BCE loss")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Training loop
    best_val_auc = 0.0
    history = {'train_loss': [], 'val_auc': [], 'val_f1': []}
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        
        # Validate
        metrics = evaluate_per_class(model, val_loader, device)
        val_auc = metrics['macro_auc']
        val_f1 = metrics['macro_f1']
        
        history['train_loss'].append(avg_train_loss)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Val F1: {val_f1:.4f}")
    
    # Final evaluation
    final_metrics = evaluate_per_class(model, val_loader, device)
    
    return {
        'config_name': config_name,
        'best_val_auc': best_val_auc,
        'final_val_auc': final_metrics['macro_auc'],
        'final_val_f1': final_metrics['macro_f1'],
        'per_class_auc': final_metrics['per_class_auc'],
        'per_class_f1': final_metrics['per_class_f1'],
        'history': history
    }


def main():
    print("\n" + "="*70)
    print("ABLATION STUDY: FOCAL LOSS vs BCE")
    print("="*70)
    print("Comparing standard BCE against Focal Loss (no class weights)")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Paths
    clahe_cache_dir = project_root / "data" / "clahe_cache"
    train_csv = project_root / "data" / "splits" / "train.csv"
    val_csv = project_root / "data" / "splits" / "val.csv"
    test_csv = project_root / "data" / "splits" / "test.csv"
    
    if not clahe_cache_dir.exists():
        print(f"\n✗ ERROR: CLAHE cache not found at {clahe_cache_dir}")
        print("Run: python scripts/precompute_clahe.py")
        return
    
    # Prepare data loaders
    print("\n" + "-"*70)
    print("PREPARING DATA")
    print("-"*70)
    
    train_aug = get_augmentation_pipeline(augmentation_strength='medium')
    val_transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    
    # Single loader setup (no weighted sampler for either config)
    print("\nLoading data (no weighted sampler)...")
    loaders = get_balanced_data_loaders(
        data_dir=str(clahe_cache_dir),
        train_split_csv=str(train_csv),
        val_split_csv=str(val_csv),
        test_split_csv=str(test_csv),
        train_transform=train_aug,
        val_transform=val_transform,
        batch_size=32,
        num_workers=0,
        use_weighted_sampler=False
    )
    
    # Create 20% subsets for speed
    print("\nCreating 20% training subsets for faster comparison...")
    train_subset = create_subset(loaders['train'], fraction=0.2)
    val_loader = loaders['val']
    
    print(f"  Train batches: {len(train_subset)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Run experiments
    results = []
    
    # Experiment 1: Baseline (no weights)
    result_baseline = train_one_config(
        config_name="BASELINE (BCE)",
        train_loader=train_subset,
        val_loader=val_loader,
        loss_type="bce",
        device=device,
        num_epochs=10
    )
    results.append(result_baseline)
    
    # Experiment 2: Weighted
    result_focal = train_one_config(
        config_name="FOCAL (alpha=0.25, gamma=2)",
        train_loader=train_subset,
        val_loader=val_loader,
        loss_type="focal",
        device=device,
        num_epochs=10
    )
    results.append(result_focal)
    
    # Compare results
    print("\n" + "="*70)
    print("ABLATION RESULTS")
    print("="*70)
    
    # Overall comparison
    print("\nOverall Performance:")
    print("-"*70)
    print(f"{'Configuration':<40} {'Best AUC':<12} {'Final AUC':<12} {'Final F1':<12}")
    print("-"*70)
    for r in results:
        print(f"{r['config_name']:<30} {r['best_val_auc']:<12.4f} "
              f"{r['final_val_auc']:<12.4f} {r['final_val_f1']:<12.4f}")
    
    # Per-class comparison
    print("\n" + "="*70)
    print("PER-CLASS AUC COMPARISON")
    print("="*70)
    print(f"{'Disease':<20} {'Baseline AUC':<15} {'Focal AUC':<15} {'Improvement':<12}")
    print("-"*70)
    
    baseline_aucs = results[0]['per_class_auc']
    focal_aucs = results[1]['per_class_auc']
    
    # Get class counts for reference
    train_df = pd.read_csv(train_csv)
    if 'Image Index' in train_df.columns:
        train_df = train_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
    
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
    
    # Sort by count (rarest first) to highlight improvement
    sorted_indices = np.argsort(label_counts)
    
    improvements = []
    for idx in sorted_indices:
        disease = DISEASE_LABELS[idx]
        baseline_auc = baseline_aucs[idx]
        focal_auc = focal_aucs[idx]
        improvement = focal_auc - baseline_auc
        improvements.append(improvement)
        
        improvement_str = f"+{improvement:.4f}" if improvement >= 0 else f"{improvement:.4f}"
        print(f"{disease:<20} {baseline_auc:<15.4f} {focal_auc:<15.4f} {improvement_str:<12} "
              f"(n={int(label_counts[idx])})")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    avg_improvement = np.mean(improvements)
    rare_class_improvement = np.mean(improvements[:5])  # 5 rarest
    
    print(f"\nAverage AUC improvement across all classes: {avg_improvement:+.4f}")
    print(f"Average AUC improvement for 5 rarest classes: {rare_class_improvement:+.4f}")
    
    overall_auc_diff = results[1]['final_val_auc'] - results[0]['final_val_auc']
    overall_f1_diff = results[1]['final_val_f1'] - results[0]['final_val_f1']
    
    print(f"\nOverall macro-AUC improvement: {overall_auc_diff:+.4f}")
    print(f"Overall macro-F1 improvement: {overall_f1_diff:+.4f}")
    
    # Save results
    results_dir = project_root / "experiments"
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed CSV
    comparison_data = []
    for idx, disease in enumerate(DISEASE_LABELS):
        comparison_data.append({
            'disease': disease,
            'sample_count': int(label_counts[idx]),
            'baseline_auc': baseline_aucs[idx],
            'focal_auc': focal_aucs[idx],
            'auc_improvement': focal_aucs[idx] - baseline_aucs[idx],
            'baseline_f1': results[0]['per_class_f1'][idx],
            'focal_f1': results[1]['per_class_f1'][idx],
            'f1_improvement': results[1]['per_class_f1'][idx] - results[0]['per_class_f1'][idx]
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('sample_count')  # Sort by rarity
    csv_path = results_dir / "ablation_class_weights.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Detailed results saved to: {csv_path}")
    
    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION FOR SUPERVISOR")
    print("="*70)
    print("\nKey Findings:")
    print(f"  1. Focal Loss rare-class AUC uplift (5 rarest avg): {rare_class_improvement:+.4f}")
    print(f"  2. Overall macro-AUC change: {overall_auc_diff:+.4f}")
    print(f"  3. Overall macro-F1 change:  {overall_f1_diff:+.4f}")
    print(f"  4. Focal Loss removes need for manual class weights")
    print("\nInterpretation: if macro-AUC improves or holds while rare classes improve, keep Focal;")
    print("otherwise stick to BCE or adjust focal hyperparameters (alpha/gamma).")
    print("="*70)


if __name__ == "__main__":
    main()
