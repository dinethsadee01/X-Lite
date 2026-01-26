"""
Verify Multi-Label Stratification
==================================
Check if the train/val/test splits preserve per-disease label frequencies.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.disease_labels import DISEASE_LABELS


def load_and_analyze_splits():
    """Load splits and compute per-disease frequencies"""
    
    splits_dir = project_root / "data" / "splits"
    
    # Load CSVs
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    
    print(f"Loaded splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    print(f"  Total: {len(train_df) + len(val_df) + len(test_df)} samples\n")
    
    # Normalize column names (handle both 'Image Index' and 'image_id', etc.)
    for df in [train_df, val_df, test_df]:
        if 'Image Index' in df.columns:
            df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'}, inplace=True)
    
    # Count disease frequencies in each split
    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    disease_counts = {}
    disease_ratios = {}
    
    for split_name, df in splits.items():
        counts = {disease: 0 for disease in DISEASE_LABELS}
        
        for label_str in df['labels']:
            if pd.isna(label_str) or label_str == 'No Finding':
                continue
            labels = [lbl.strip() for lbl in str(label_str).split('|')]
            for label in labels:
                if label in counts:
                    counts[label] += 1
        
        disease_counts[split_name] = counts
        
        # Compute ratios (% of split that has this disease)
        total_positives = sum(counts.values())
        ratios = {disease: (counts[disease] / len(df) * 100) 
                  for disease in DISEASE_LABELS}
        disease_ratios[split_name] = ratios
    
    # Compute overall frequencies
    overall_counts = {disease: 0 for disease in DISEASE_LABELS}
    for split_counts in disease_counts.values():
        for disease, count in split_counts.items():
            overall_counts[disease] += count
    
    total_all = sum(overall_counts.values())
    total_samples_all = len(train_df) + len(val_df) + len(test_df)
    overall_ratios = {disease: (overall_counts[disease] / total_samples_all * 100) 
                      for disease in DISEASE_LABELS}
    
    # Print detailed comparison
    print("="*100)
    print("PER-DISEASE LABEL FREQUENCIES (% of split)")
    print("="*100)
    print(f"{'Disease':<25} {'Train %':<12} {'Val %':<12} {'Test %':<12} {'Overall %':<12} {'Max Dev':<10}")
    print("-"*100)
    
    max_deviations = []
    
    for disease in DISEASE_LABELS:
        train_pct = disease_ratios['train'][disease]
        val_pct = disease_ratios['val'][disease]
        test_pct = disease_ratios['test'][disease]
        overall_pct = overall_ratios[disease]
        
        # Max deviation from overall
        deviations = [
            abs(train_pct - overall_pct),
            abs(val_pct - overall_pct),
            abs(test_pct - overall_pct)
        ]
        max_dev = max(deviations)
        max_deviations.append(max_dev)
        
        print(f"{disease:<25} {train_pct:>10.2f}% {val_pct:>10.2f}% {test_pct:>10.2f}% {overall_pct:>10.2f}% {max_dev:>8.2f}%")
    
    print("="*100)
    print(f"\nStratification Quality Metrics:")
    print(f"  Mean max deviation: {np.mean(max_deviations):.2f}%")
    print(f"  Max overall deviation: {np.max(max_deviations):.2f}%")
    print(f"  Std of deviations: {np.std(max_deviations):.2f}%")
    
    # Interpretation
    if np.max(max_deviations) < 5:
        print(f"\n✅ Excellent stratification! Max deviation < 5%")
    elif np.max(max_deviations) < 10:
        print(f"\n✅ Good stratification. Max deviation < 10%")
    else:
        print(f"\n⚠️  Moderate stratification. Max deviation >= 10%")
    
    return disease_ratios, overall_ratios, DISEASE_LABELS


def plot_stratification(disease_ratios, overall_ratios, disease_labels):
    """Create visualization of stratification"""
    
    # Prepare data for plotting
    plot_data = []
    for disease in disease_labels:
        plot_data.append({
            'Disease': disease,
            'Train': disease_ratios['train'][disease],
            'Val': disease_ratios['val'][disease],
            'Test': disease_ratios['test'][disease],
            'Overall': overall_ratios[disease]
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Grouped bar chart
    ax1 = axes[0]
    x = np.arange(len(disease_labels))
    width = 0.2
    
    ax1.bar(x - 1.5*width, plot_df['Train'], width, label='Train', alpha=0.8)
    ax1.bar(x - 0.5*width, plot_df['Val'], width, label='Val', alpha=0.8)
    ax1.bar(x + 0.5*width, plot_df['Test'], width, label='Test', alpha=0.8)
    ax1.bar(x + 1.5*width, plot_df['Overall'], width, label='Overall', alpha=0.8, linestyle='--')
    
    ax1.set_xlabel('Disease', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency (% of split)', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Disease Label Frequencies Across Splits', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(disease_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Heatmap of deviation from overall
    deviations = []
    for disease in disease_labels:
        train_dev = abs(disease_ratios['train'][disease] - overall_ratios[disease])
        val_dev = abs(disease_ratios['val'][disease] - overall_ratios[disease])
        test_dev = abs(disease_ratios['test'][disease] - overall_ratios[disease])
        deviations.append([train_dev, val_dev, test_dev])
    
    deviation_df = pd.DataFrame(
        deviations,
        index=disease_labels,
        columns=['Train Deviation', 'Val Deviation', 'Test Deviation']
    )
    
    ax2 = axes[1]
    sns.heatmap(deviation_df, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax2, cbar_kws={'label': 'Absolute Deviation (%)'})
    ax2.set_title('Deviation from Overall Distribution (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Disease', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Split', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    plot_path = results_dir / "stratification_verification.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {plot_path}")
    
    # Also save data to CSV
    csv_path = project_root / "experiments" / "stratification_verification.csv"
    plot_df.to_csv(csv_path, index=False)
    print(f"✅ Data saved to: {csv_path}")


if __name__ == "__main__":
    print("\n" + "="*100)
    print("MULTI-LABEL STRATIFICATION VERIFICATION")
    print("="*100 + "\n")
    
    disease_ratios, overall_ratios, disease_labels = load_and_analyze_splits()
    plot_stratification(disease_ratios, overall_ratios, disease_labels)
    
    print("\n" + "="*100)
    print("✅ Stratification verification complete!")
    print("="*100)
