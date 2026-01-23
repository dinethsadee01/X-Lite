"""
Class Balance Mitigation Visualization
=======================================
Shows before/after comparison of class imbalance handling strategies.

Visualizations:
1. Original class distribution (imbalanced)
2. Class weights applied in loss function
3. Effective sampling distribution with WeightedRandomSampler
4. Expected class frequency after sampling

Purpose: Document how we address severe class imbalance in training.

Usage:
    python scripts/visualize_class_balance.py

Output:
    results/class_balance_mitigation.png - Comprehensive comparison
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config.disease_labels import DISEASE_LABELS
from ml.training.losses import calculate_pos_weights
import torch


def load_class_distribution(splits_dir: Path):
    """Load and compute class distribution from training data"""
    train_df = pd.read_csv(splits_dir / "train.csv")
    
    # Rename columns if needed
    if 'Image Index' in train_df.columns:
        train_df = train_df.rename(columns={
            'Image Index': 'image_id',
            'Finding Labels': 'labels'
        })
    
    # Count disease occurrences
    disease_counts = {disease: 0 for disease in DISEASE_LABELS}
    total_images = len(train_df)
    images_with_no_finding = 0
    
    for label_str in train_df['labels']:
        if pd.isna(label_str) or label_str == 'No Finding':
            images_with_no_finding += 1
            continue
        
        labels = label_str.split('|')
        for label in labels:
            label = label.strip()
            if label in DISEASE_LABELS:
                disease_counts[label] += 1
    
    return disease_counts, total_images, images_with_no_finding


def compute_class_weights(disease_counts, total_samples):
    """Compute class weights for loss function"""
    label_counts = np.array([disease_counts[d] for d in DISEASE_LABELS])
    label_counts_tensor = torch.tensor(label_counts, dtype=torch.float32)
    pos_weights = calculate_pos_weights(label_counts_tensor, total_samples)
    
    class_weights = {disease: weight.item() 
                    for disease, weight in zip(DISEASE_LABELS, pos_weights)}
    
    return class_weights


def compute_effective_sampling(disease_counts, total_samples, subset_fraction: float = 0.2):
    """
    Expected sampling distribution with WeightedRandomSampler.

    Formula (matches training code): w_i = 1.0 / (count_i + 1.0)
    """
    label_counts = np.array([disease_counts[d] for d in DISEASE_LABELS], dtype=float)

    # Inverse-frequency weights (exact match to training)
    weights = 1.0 / (label_counts + 1.0)
    weights = weights / weights.sum()

    # Estimated samples per disease in one epoch (approximate)
    samples_per_epoch = total_samples * subset_fraction  # we train on 20% subset
    effective_samples = weights * samples_per_epoch

    effective_counts = {disease: count for disease, count in zip(DISEASE_LABELS, effective_samples)}

    return effective_counts


def compute_effective_contribution(disease_counts, class_weights, total_samples, subset_fraction: float = 0.2):
    """
    Approximate effective gradient contribution per class from the combination of:
    - WeightedRandomSampler (controls how often a class appears)
    - Weighted BCE loss (scales the gradient magnitude per class)

    contribution_i ≈ (sampler_expected_i) * (loss_weight_i)
    """
    # Expected samples from sampler
    sampler_counts = compute_effective_sampling(disease_counts, total_samples, subset_fraction)

    # Loss weights array aligned to class order
    loss_weights = np.array([class_weights[d] for d in DISEASE_LABELS], dtype=float)
    sampler_arr = np.array([sampler_counts[d] for d in DISEASE_LABELS], dtype=float)

    contribution = sampler_arr * loss_weights
    contribution_dict = {d: c for d, c in zip(DISEASE_LABELS, contribution)}

    return contribution_dict


def create_visualization(splits_dir: Path, output_path: Path):
    """Create comprehensive class balance mitigation visualization"""
    
    # Load data
    disease_counts, total_samples, no_finding = load_class_distribution(splits_dir)
    class_weights = compute_class_weights(disease_counts, total_samples)
    effective_counts = compute_effective_sampling(disease_counts, total_samples, subset_fraction=0.2)
    contribution_counts = compute_effective_contribution(
        disease_counts,
        class_weights,
        total_samples,
        subset_fraction=0.2
    )
    
    # Sort by original count (descending)
    sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
    diseases = [d for d, _ in sorted_diseases]
    
    # Prepare data for plotting
    original_counts = np.array([disease_counts[d] for d in diseases])
    weights = np.array([class_weights[d] for d in diseases])
    effective = np.array([effective_counts[d] for d in diseases])
    contribution = np.array([contribution_counts[d] for d in diseases])
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.35)
    
    # Color palette
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(diseases)))
    
    # ========================================================================
    # Plot 1: Original Imbalanced Distribution (BEFORE)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    bars1 = ax1.barh(diseases, original_counts, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Number of Cases', fontsize=12, fontweight='bold')
    ax1.set_title('BEFORE: Original Class Distribution (Severely Imbalanced)\n', 
                  fontsize=14, fontweight='bold', color='darkred')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars1, original_counts)):
        percentage = (count / total_samples) * 100
        ax1.text(count, i, f' {count:,} ({percentage:.1f}%)', 
                va='center', fontsize=9, fontweight='bold')
    
    # Highlight imbalance ratio
    imbalance_ratio = original_counts.max() / original_counts.min()
    ax1.text(0.98, 0.02, f'Imbalance Ratio: {imbalance_ratio:.1f}:1',
            transform=ax1.transAxes, ha='right', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========================================================================
    # Plot 2: Class Weights (Loss Function Balancing)
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    bars2 = ax2.barh(diseases, weights, color=colors[::-1], edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Weight Factor', fontsize=11, fontweight='bold')
    ax2.set_title('Strategy 1: Weighted Loss Function\n(Higher weights for rare diseases)\n',
                  fontsize=12, fontweight='bold', color='darkgreen')
    ax2.invert_yaxis()
    ax2.set_xscale('log')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add weight labels
    for i, (bar, weight) in enumerate(zip(bars2, weights)):
        ax2.text(weight, i, f' {weight:.2f}×', va='center', fontsize=8)
    
    # ========================================================================
    # Plot 3: Effective Sampling Distribution (WeightedRandomSampler)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    bars3 = ax3.barh(diseases, effective, color='steelblue', edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Expected Samples per Epoch (20% subset)', fontsize=11, fontweight='bold')
    ax3.set_title('Strategy 2: Weighted Random Sampling\n'
                  '(Inverse-frequency weights: w = 1/(count+1))\n',
                  fontsize=12, fontweight='bold', color='darkgreen')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)
    
    # Add sample labels
    for i, (bar, count) in enumerate(zip(bars3, effective)):
        ax3.text(count, i, f' {count:.0f}', va='center', fontsize=8)
    
    # ========================================================================
    # Plot 4: Combined Effect (Sampler + Weighted Loss)
    # ========================================================================
    ax4 = fig.add_subplot(gs[2, :])
    
    # Normalize to percentages for fair comparison
    original_pct = (original_counts / original_counts.sum()) * 100
    contribution_pct = (contribution / contribution.sum()) * 100
    
    x = np.arange(len(diseases))
    width = 0.35
    
    bars_orig = ax4.bar(x - width/2, original_pct, width, 
                       label='Original (Imbalanced)', 
                       color='coral', edgecolor='black', linewidth=0.5)
    bars_eff = ax4.bar(x + width/2, contribution_pct, width,
                      label='Combined Effect (Sampler × Loss Weight)',
                      color='lightgreen', edgecolor='black', linewidth=0.5)
    
    ax4.set_ylabel('Percentage of Total Samples (%)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Disease Class', fontsize=11, fontweight='bold')
    ax4.set_title('\nCombined Effect: Weighted Sampling + Weighted Loss\n'
                 'Approximate per-class contribution to training signal\n',
                  fontsize=13, fontweight='bold', color='darkblue')
    ax4.set_xticks(x)
    ax4.set_xticklabels(diseases, rotation=45, ha='right', fontsize=9)
    ax4.legend(fontsize=11, loc='upper left')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add balance improvement metric using combined contribution
    cv_before = np.std(original_pct) / np.mean(original_pct)
    cv_after = np.std(contribution_pct) / np.mean(contribution_pct)
    rare_emphasis = contribution.max() / contribution.min()
    
    ax4.text(0.98, 0.95, 
            f'Rare-Class Emphasis (max/min): {rare_emphasis:.1f}:1\n'
            f'CV Before: {cv_before:.2f} → After: {cv_after:.2f}',
            transform=ax4.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # ========================================================================
    # Overall title and footer
    # ========================================================================
    fig.suptitle('Class Imbalance Mitigation Strategies in Training',
                 fontsize=16, fontweight='bold', y=0.98)
    
    fig.text(0.5, 0.01,
             f'Dataset: {total_samples:,} training images | '
             f'No Finding: {no_finding:,} ({no_finding/total_samples*100:.1f}%) | '
             f'14 disease classes (multi-label) | '
             f'Training subset: 20% with weighted sampling',
             ha='center', fontsize=9, style='italic', color='dimgray')
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.subplots_adjust(hspace=0.6, wspace=0.35)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Class balance visualization saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("CLASS IMBALANCE MITIGATION SUMMARY")
    print("=" * 70)
    print(f"\nOriginal Imbalance Ratio: {imbalance_ratio:.1f}:1")
    print(f"Most frequent: {diseases[0]} ({original_counts[0]:,} cases)")
    print(f"Least frequent: {diseases[-1]} ({original_counts[-1]:,} cases)")
    print(f"\nMitigation Strategies Applied:")
    print(f"  1. ✓ Weighted BCE Loss (inverse frequency class weights)")
    print(f"  2. ✓ WeightedRandomSampler (balanced batch composition)")
    print(f"  3. ✓ Stratified data splits (preserve distribution)")
    print(f"\nRare-Class Emphasis (Sampler × Loss, max/min): {rare_emphasis:.1f}:1")
    print(f"Coefficient of Variation: {cv_before:.2f} → {cv_after:.2f}")
    print("=" * 70)


def main():
    # Paths
    splits_dir = project_root / "data" / "splits"
    results_dir = project_root / "results"
    output_path = results_dir / "class_balance_mitigation.png"
    
    # Validate
    if not splits_dir.exists():
        print(f"✗ Error: Splits directory not found: {splits_dir}")
        return
    
    print("=" * 70)
    print("CLASS BALANCE MITIGATION VISUALIZATION")
    print("=" * 70)
    print(f"Analyzing training data from: {splits_dir}")
    print(f"Output: {output_path}")
    print()
    
    # Create visualization
    create_visualization(splits_dir, output_path)
    
    print(f"\nUse this visualization alongside 01_eda_overview.png to show:")
    print(f"  • Problem: Severe class imbalance (before)")
    print(f"  • Solution: Multi-strategy mitigation (after)")
    print(f"  • Impact: Quantified improvement in balance")


if __name__ == "__main__":
    main()
