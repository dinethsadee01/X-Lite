"""
Data Balance Verification
=========================
Verify that batches are actually balanced before training.

Shows:
1. Original class distribution in data
2. Expected class frequency from WeightedRandomSampler
3. Actual class frequency in real batches (ground truth)
4. Whether balancing is working or over-correcting
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from ml.data.loader import get_balanced_data_loaders
from ml.data.augmentation import get_augmentation_pipeline
from ml.data.preprocessing import get_medical_transforms
from config.disease_labels import DISEASE_LABELS


def analyze_batches(loader, num_batches=20, loader_name=""):
    """Analyze actual class distribution in batches"""
    class_counts = np.zeros(len(DISEASE_LABELS))
    batch_count = 0
    total_images = 0
    
    print(f"\n{'='*70}")
    print(f"ANALYZING BATCHES: {loader_name}")
    print(f"{'='*70}")
    
    for images, targets, _ in loader:
        if batch_count >= num_batches:
            break
        
        # targets shape: [batch_size, num_classes]
        # Sum positive labels per class across batch
        class_counts += targets.sum(dim=0).numpy()
        total_images += images.shape[0]
        batch_count += 1
    
    # Normalize to percentages
    total_labels = class_counts.sum()
    class_percentages = (class_counts / total_labels) * 100 if total_labels > 0 else class_counts
    
    print(f"Analyzed {batch_count} batches ({total_images} images, {int(total_labels)} labels)")
    
    return class_counts, class_percentages


def main():
    print("\n" + "="*70)
    print("DATA BALANCE VERIFICATION")
    print("="*70)
    print("Goal: Verify that WeightedRandomSampler is actually balancing batches")
    
    # Paths
    clahe_cache_dir = project_root / "data" / "clahe_cache"
    train_csv = project_root / "data" / "splits" / "train.csv"
    val_csv = project_root / "data" / "splits" / "val.csv"
    test_csv = project_root / "data" / "splits" / "test.csv"
    
    if not clahe_cache_dir.exists():
        print(f"\n✗ ERROR: CLAHE cache not found at {clahe_cache_dir}")
        print("Run: python scripts/precompute_clahe.py")
        return
    
    # Load training data to compute original distribution
    print("\n" + "-"*70)
    print("STEP 1: ORIGINAL DATA DISTRIBUTION")
    print("-"*70)
    
    train_df = pd.read_csv(train_csv)
    if 'Image Index' in train_df.columns:
        train_df = train_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
    
    original_counts = np.zeros(len(DISEASE_LABELS))
    for label_str in train_df['labels']:
        if pd.isna(label_str) or label_str == 'No Finding':
            continue
        labels = label_str.split('|')
        for label in labels:
            label = label.strip()
            if label in DISEASE_LABELS:
                idx = DISEASE_LABELS.index(label)
                original_counts[idx] += 1
    
    original_pct = (original_counts / original_counts.sum()) * 100
    
    print(f"\nOriginal distribution ({len(train_df)} images):")
    print(f"{'Disease':<20} {'Count':<8} {'%':<8}")
    print("-"*36)
    sorted_idx = np.argsort(original_counts)[::-1]
    for idx in sorted_idx:
        print(f"{DISEASE_LABELS[idx]:<20} {int(original_counts[idx]):<8} {original_pct[idx]:<8.2f}")
    
    print(f"\nImbalance ratio (most/least): {original_counts.max() / original_counts.min():.1f}:1")
    uniform_pct = 100.0 / len(DISEASE_LABELS)
    print(f"Uniform distribution would be: {uniform_pct:.2f}% per class")
    
    # Load with NO weighting
    print("\n" + "-"*70)
    print("STEP 2: BASELINE LOADER (NO WEIGHTING)")
    print("-"*70)
    
    train_aug = get_augmentation_pipeline(augmentation_strength='medium')
    val_transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    
    loaders_baseline = get_balanced_data_loaders(
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
    
    baseline_counts, baseline_pct = analyze_batches(loaders_baseline['train'], num_batches=20, loader_name="No Weighting")
    
    # Load WITH weighting
    print("\n" + "-"*70)
    print("STEP 3: WEIGHTED LOADER (WITH WEIGHTING)")
    print("-"*70)
    
    loaders_weighted = get_balanced_data_loaders(
        data_dir=str(clahe_cache_dir),
        train_split_csv=str(train_csv),
        val_split_csv=str(val_csv),
        test_split_csv=str(test_csv),
        train_transform=train_aug,
        val_transform=val_transform,
        batch_size=32,
        num_workers=0,
        use_weighted_sampler=True
    )
    
    weighted_counts, weighted_pct = analyze_batches(loaders_weighted['train'], num_batches=20, loader_name="With Weighting")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON: DOES WEIGHTING ACTUALLY BALANCE?")
    print("="*70)
    
    print(f"\n{'Disease':<20} {'Original %':<12} {'Baseline %':<12} {'Weighted %':<12} {'vs Uniform':<12}")
    print("-"*80)
    
    for idx in np.argsort(original_counts):
        disease = DISEASE_LABELS[idx]
        orig = original_pct[idx]
        base = baseline_pct[idx]
        wght = weighted_pct[idx]
        
        # Distance from perfect balance (7.14%)
        baseline_dist = abs(base - uniform_pct)
        weighted_dist = abs(wght - uniform_pct)
        
        print(f"{disease:<20} {orig:<12.2f} {base:<12.2f} {wght:<12.2f} {weighted_dist:<12.2f}")
    
    # Compute balance metrics
    print("\n" + "-"*70)
    print("BALANCE METRICS (Coefficient of Variation)")
    print("-"*70)
    print("CV measures how spread out the distribution is.")
    print("Lower CV = more balanced. CV=0 = perfect balance.")
    
    cv_original = np.std(original_pct) / np.mean(original_pct)
    cv_baseline = np.std(baseline_pct) / np.mean(baseline_pct)
    cv_weighted = np.std(weighted_pct) / np.mean(weighted_pct)
    
    print(f"\nOriginal data:              CV = {cv_original:.4f}")
    print(f"Baseline batches:           CV = {cv_baseline:.4f}  (Δ {(cv_baseline - cv_original):+.4f})")
    print(f"Weighted batches:           CV = {cv_weighted:.4f}  (Δ {(cv_weighted - cv_original):+.4f})")
    print(f"Perfect balance (uniform):  CV = 0.0000")
    
    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    if cv_weighted < cv_baseline and cv_weighted < cv_original:
        print("\n✓✓✓ WEIGHTING STRATEGY IS WORKING CORRECTLY")
        print(f"    Batches are MORE balanced than both original data and baseline")
        print(f"    CV improved from {cv_original:.4f} → {cv_weighted:.4f}")
        print(f"\n    ACTION: Ready to proceed with weighted training")
    elif cv_weighted < cv_baseline:
        print("\n✓ WEIGHTING HELPS SOMEWHAT")
        print(f"    Batches are more balanced than baseline")
        print(f"    But still less balanced than needed")
        print(f"\n    ACTION: May need stronger weighting formula")
    else:
        print("\n✗ WEIGHTING STRATEGY IS NOT WORKING")
        print(f"    Batches are NOT more balanced than original/baseline")
        print(f"    CV got worse: {cv_original:.4f} → {cv_weighted:.4f}")
        print(f"\n    ACTION: Weighting formula needs revision or alternative strategy needed")
    
    # Visualization
    print("\n" + "-"*70)
    print("CREATING VISUALIZATION")
    print("-"*70)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    sorted_diseases = [DISEASE_LABELS[i] for i in np.argsort(original_counts)]
    sorted_orig = np.sort(original_pct)
    sorted_base = np.sort(baseline_pct)
    sorted_wght = np.sort(weighted_pct)
    
    # Plot 1: Original
    bars1 = axes[0].barh(sorted_diseases, sorted_orig, color='coral', edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Percentage of Labels (%)', fontweight='bold')
    axes[0].set_title('Original Data Distribution\n(Severely Imbalanced)', fontweight='bold', fontsize=12)
    axes[0].axvline(uniform_pct, color='green', linestyle='--', linewidth=2, label=f'Uniform ({uniform_pct:.1f}%)')
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].text(0.95, 0.05, f'CV = {cv_original:.3f}', transform=axes[0].transAxes, 
                ha='right', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Baseline batches
    bars2 = axes[1].barh(sorted_diseases, sorted_base, color='skyblue', edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Percentage of Labels (%)', fontweight='bold')
    axes[1].set_title('Baseline Batches\n(No Weighting, Standard Sampling)', fontweight='bold', fontsize=12)
    axes[1].axvline(uniform_pct, color='green', linestyle='--', linewidth=2, label=f'Uniform ({uniform_pct:.1f}%)')
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].text(0.95, 0.05, f'CV = {cv_baseline:.3f}', transform=axes[1].transAxes,
                ha='right', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3: Weighted batches
    bars3 = axes[2].barh(sorted_diseases, sorted_wght, color='lightgreen', edgecolor='black', linewidth=0.5)
    axes[2].set_xlabel('Percentage of Labels (%)', fontweight='bold')
    axes[2].set_title('Weighted Batches\n(WeightedRandomSampler)', fontweight='bold', fontsize=12)
    axes[2].axvline(uniform_pct, color='green', linestyle='--', linewidth=2, label=f'Uniform ({uniform_pct:.1f}%)')
    axes[2].legend(fontsize=10)
    axes[2].grid(axis='x', alpha=0.3)
    axes[2].text(0.95, 0.05, f'CV = {cv_weighted:.3f}', transform=axes[2].transAxes,
                ha='right', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    fig.suptitle('Data Balance Verification: Does WeightedRandomSampler Work?', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    viz_path = results_dir / "data_balance_verification.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Visualization saved to: {viz_path}")
    
    # Save detailed results
    print("\n" + "-"*70)
    print("SAVING DETAILED RESULTS")
    print("-"*70)
    
    verification_data = []
    for idx, disease in enumerate(DISEASE_LABELS):
        verification_data.append({
            'disease': disease,
            'sample_count': int(original_counts[idx]),
            'original_pct': original_pct[idx],
            'baseline_batch_pct': baseline_pct[idx],
            'weighted_batch_pct': weighted_pct[idx],
            'distance_from_uniform_baseline': abs(baseline_pct[idx] - uniform_pct),
            'distance_from_uniform_weighted': abs(weighted_pct[idx] - uniform_pct),
            'closer_to_uniform': 'YES' if abs(weighted_pct[idx] - uniform_pct) < abs(baseline_pct[idx] - uniform_pct) else 'NO'
        })
    
    df_verify = pd.DataFrame(verification_data)
    df_verify = df_verify.sort_values('sample_count')
    
    csv_path = project_root / "experiments" / "data_balance_verification.csv"
    df_verify.to_csv(csv_path, index=False)
    print(f"✓ Detailed results saved to: {csv_path}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
