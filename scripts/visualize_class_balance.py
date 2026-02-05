"""
Training Data Balance Analysis
==============================
Compares full training set vs smart subset to validate balancing strategy.

Visualizations:
1. Full train vs subset disease distribution
2. Recalculated class weights (full vs subset)
3. No Finding ratio comparison
4. Subset composition breakdown

Purpose: Validate that smart subset preserves diversity while improving balance.

Usage:
    python scripts/visualize_class_balance.py

Output:
    results/subset_balance_analysis.png - Full vs Subset comparison
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


def load_class_distribution(csv_path: Path):
    """Load and compute class distribution from a CSV file"""
    train_df = pd.read_csv(csv_path)
    
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


def compute_weight_differences(full_weights, subset_weights):
    """Calculate how much weights change from full to subset"""
    differences = {}
    for disease in DISEASE_LABELS:
        diff = subset_weights[disease] - full_weights[disease]
        pct_change = (diff / full_weights[disease]) * 100 if full_weights[disease] > 0 else 0
        differences[disease] = {'abs': diff, 'pct': pct_change}
    return differences


def create_visualization(splits_dir: Path, output_path: Path):
    """Create full vs subset comparison visualization"""
    
    # Load both datasets
    full_csv = splits_dir / "train.csv"
    subset_csv = splits_dir / "train_subset.csv"
    
    if not subset_csv.exists():
        print(f"✗ Error: Subset not found at {subset_csv}")
        print("  Run: python scripts/create_smart_subset.py")
        return
    
    full_counts, full_total, full_no_finding = load_class_distribution(full_csv)
    subset_counts, subset_total, subset_no_finding = load_class_distribution(subset_csv)
    
    full_weights = compute_class_weights(full_counts, full_total)
    subset_weights = compute_class_weights(subset_counts, subset_total)
    weight_diffs = compute_weight_differences(full_weights, subset_weights)
    
    # Sort by full dataset count (descending)
    sorted_diseases = sorted(full_counts.items(), key=lambda x: x[1], reverse=True)
    diseases = [d for d, _ in sorted_diseases]
    
    # Prepare data for plotting
    full_vals = np.array([full_counts[d] for d in diseases])
    subset_vals = np.array([subset_counts[d] for d in diseases])
    full_wts = np.array([full_weights[d] for d in diseases])
    subset_wts = np.array([subset_weights[d] for d in diseases])
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)
    
    # Colors
    color_full = '#e08e79'
    color_subset = '#59a14f'
    
    # ========================================================================
    # Plot 1: Disease counts comparison (Full vs Subset)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    y_pos = np.arange(len(diseases))
    width = 0.35
    
    ax1.barh(y_pos - width/2, full_vals, width, label='Full Train', 
             color=color_full, edgecolor='black', linewidth=0.5)
    ax1.barh(y_pos + width/2, subset_vals, width, label='Smart Subset', 
             color=color_subset, edgecolor='black', linewidth=0.5)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(diseases)
    ax1.invert_yaxis()
    ax1.set_xlabel('Disease Count', fontsize=12, fontweight='bold')
    ax1.set_title('Disease Distribution: Full Train vs Smart Subset', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add retention percentages
    for i, (disease, full_val, subset_val) in enumerate(zip(diseases, full_vals, subset_vals)):
        retention_pct = (subset_val / full_val * 100) if full_val > 0 else 0
        ax1.text(max(full_val, subset_val), i, f' {retention_pct:.0f}%', 
                va='center', fontsize=8, style='italic')
    
    # ========================================================================
    # Plot 2: Class weights comparison (Full vs Subset)
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.barh(y_pos - width/2, full_wts, width, label='Full Train Weights',
             color=color_full, edgecolor='black', linewidth=0.5)
    ax2.barh(y_pos + width/2, subset_wts, width, label='Subset Weights',
             color=color_subset, edgecolor='black', linewidth=0.5)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(diseases)
    ax2.invert_yaxis()
    ax2.set_xlabel('Loss Weight', fontsize=11, fontweight='bold')
    ax2.set_title('Recalculated Class Weights\n(Subset-specific for training)', 
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_xscale('log')
    ax2.grid(axis='x', alpha=0.3)
    
    # ========================================================================
    # Plot 3: Weight change percentage
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    weight_pct_changes = np.array([weight_diffs[d]['pct'] for d in diseases])
    colors_change = ['green' if x > 0 else 'red' for x in weight_pct_changes]
    
    ax3.barh(diseases, weight_pct_changes, color=colors_change, 
             edgecolor='black', linewidth=0.5, alpha=0.7)
    ax3.set_xlabel('Weight Change (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Impact of Smart Subset on Weights\n(% change from full train)',
                  fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax3.grid(axis='x', alpha=0.3)
    
    # ========================================================================
    # Plot 4: Imbalance ratio comparison
    # ========================================================================
    ax4 = fig.add_subplot(gs[2, :])
    
    full_imbalance = full_vals.max() / full_vals.min()
    subset_imbalance = subset_vals.max() / subset_vals.min()
    
    full_no_finding_ratio = full_no_finding / full_total
    subset_no_finding_ratio = subset_no_finding / subset_total
    
    metrics = {
        'Dataset Size': [full_total, subset_total],
        'Imbalance Ratio\n(max/min)': [full_imbalance, subset_imbalance],
        'No Finding\nRatio': [full_no_finding_ratio, subset_no_finding_ratio],
        'Sick Images': [full_total - full_no_finding, subset_total - subset_no_finding]
    }
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    full_metrics = [metrics[k][0] for k in metrics.keys()]
    subset_metrics = [metrics[k][1] for k in metrics.keys()]
    
    # Normalize for comparison
    normalized_full = [v / full_total for v in full_metrics]
    normalized_subset = [v / subset_total for v in subset_metrics]
    normalized_full[1] = full_imbalance / 100  # Scale imbalance ratio
    normalized_subset[1] = subset_imbalance / 100
    
    ax4.bar(x_pos - width/2, [full_metrics[0], full_imbalance, full_no_finding_ratio*100, full_metrics[3]], 
            width, label='Full Train', color=color_full, edgecolor='black', linewidth=0.6)
    ax4.bar(x_pos + width/2, [subset_metrics[0], subset_imbalance, subset_no_finding_ratio*100, subset_metrics[3]], 
            width, label='Smart Subset', color=color_subset, edgecolor='black', linewidth=0.6)
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['Dataset Size', 'Imbalance\nRatio (max/min)', 'No Finding\n(%)', 'Sick Images'])
    ax4.set_title('Key Metrics Comparison', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (f_val, s_val) in enumerate(zip([full_metrics[0], full_imbalance, full_no_finding_ratio*100, full_metrics[3]],
                                             [subset_metrics[0], subset_imbalance, subset_no_finding_ratio*100, subset_metrics[3]])):
        if i == 1:  # Imbalance ratio
            ax4.text(i - width/2, f_val, f'{f_val:.1f}:1', ha='center', va='bottom', fontsize=9)
            ax4.text(i + width/2, s_val, f'{s_val:.1f}:1', ha='center', va='bottom', fontsize=9)
        elif i == 2:  # No Finding %
            ax4.text(i - width/2, f_val, f'{f_val:.1f}%', ha='center', va='bottom', fontsize=9)
            ax4.text(i + width/2, s_val, f'{s_val:.1f}%', ha='center', va='bottom', fontsize=9)
        else:
            ax4.text(i - width/2, f_val, f'{int(f_val):,}', ha='center', va='bottom', fontsize=9)
            ax4.text(i + width/2, s_val, f'{int(s_val):,}', ha='center', va='bottom', fontsize=9)
    
    # ========================================================================
    # Overall title and save
    # ========================================================================
    fig.suptitle('Training Data Balance: Full vs Smart Subset Analysis',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Balance analysis saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING DATA BALANCE ANALYSIS")
    print("=" * 70)
    print(f"\nFull Training Set:")
    print(f"  Total: {full_total:,} images")
    print(f"  Sick: {full_total - full_no_finding:,}")
    print(f"  No Finding: {full_no_finding:,} ({full_no_finding_ratio*100:.1f}%)")
    print(f"  Imbalance ratio: {full_imbalance:.1f}:1")
    
    print(f"\nSmart Subset (for baseline training):")
    print(f"  Total: {subset_total:,} images")
    print(f"  Sick: {subset_total - subset_no_finding:,}")
    print(f"  No Finding: {subset_no_finding:,} ({subset_no_finding_ratio*100:.1f}%)")
    print(f"  Imbalance ratio: {subset_imbalance:.1f}:1")
    
    print(f"\nBalancing Improvements:")
    imbalance_reduction = ((full_imbalance - subset_imbalance) / full_imbalance) * 100
    print(f"  Imbalance ratio reduced by: {imbalance_reduction:.1f}%")
    print(f"  No Finding ratio: {full_no_finding_ratio*100:.1f}% → {subset_no_finding_ratio*100:.1f}%")
    
    print(f"\nWeight Recalculation (largest changes):")
    sorted_diffs = sorted(weight_diffs.items(), key=lambda x: abs(x[1]['pct']), reverse=True)
    for disease, diff in sorted_diffs[:5]:
        sign = "+" if diff['pct'] > 0 else ""
        print(f"  {disease:<25} {sign}{diff['pct']:>6.1f}%")
    
    print("=" * 70)


def main():
    # Paths
    splits_dir = project_root / "data" / "splits"
    results_dir = project_root / "results"
    output_path = results_dir / "subset_balance_analysis.png"
    
    # Validate
    if not splits_dir.exists():
        print(f"✗ Error: Splits directory not found: {splits_dir}")
        return
    
    print("=" * 70)
    print("TRAINING DATA BALANCE ANALYSIS")
    print("=" * 70)
    print(f"Analyzing full train vs smart subset")
    print(f"Output: {output_path}")
    print()
    
    # Create visualization
    create_visualization(splits_dir, output_path)
    
    print(f"\nThis visualization shows:")
    print(f"  • Disease distribution changes (full → subset)")
    print(f"  • Recalculated class weights for subset-specific training")
    print(f"  • Imbalance ratio improvements")
    print(f"  • No Finding balance (50/50 target)")


if __name__ == "__main__":
    main()
