"""
Generate Visualization Charts for KD Results
==============================================
Creates comparison charts between baseline, KD, and test set results.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.disease_labels import DISEASE_LABELS


def load_results():
    """Load all results for comparison"""
    
    # Baseline KD results (validation)
    kd_csv = project_root / "experiments" / "kd_results.csv"
    kd_df = pd.read_csv(kd_csv)
    kd_data = kd_df.iloc[0]  # First (and only) run
    
    # Test set results
    test_json = project_root / "experiments" / "test_evaluation_results.json"
    with open(test_json) as f:
        test_data = json.load(f)
    
    return kd_data, test_data


def create_comparison_chart(kd_data, test_data):
    """Create bar chart comparing Val AUC vs Test AUC"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['AUC (macro)', 'F1 (macro)', 'PR-AUC (macro)', 'Precision (macro)']
    val_values = [
        kd_data['final_val_auc'],
        kd_data['final_val_f1'],
        kd_data['final_val_pr_auc'],
        kd_data['final_val_precision']
    ]
    test_values = [
        test_data['auc_macro'],
        test_data['f1_macro'],
        test_data['pr_auc_macro'],
        test_data['precision_macro']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, val_values, width, label='Validation Set', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_values, width, label='Test Set (Unseen)', color='#A23B72', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Knowledge Distillation: Validation vs Test Performance\nConvNext Tiny MHSA Student', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(project_root / "results" / "kd_validation_vs_test.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: results/kd_validation_vs_test.png")
    plt.close()


def create_per_disease_chart(test_data):
    """Create per-disease AUC comparison chart"""
    
    diseases = list(test_data['per_disease_metrics'].keys())
    aucs = [test_data['per_disease_metrics'][d]['auc'] for d in diseases]
    
    # Sort by AUC descending
    sorted_pairs = sorted(zip(diseases, aucs), key=lambda x: x[1], reverse=True)
    diseases_sorted = [x[0] for x in sorted_pairs]
    aucs_sorted = [x[1] for x in sorted_pairs]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color bars by performance
    colors = []
    for auc in aucs_sorted:
        if auc >= 0.85:
            colors.append('#27AE60')  # Green - excellent
        elif auc >= 0.80:
            colors.append('#3498DB')  # Blue - good
        elif auc >= 0.75:
            colors.append('#F39C12')  # Orange - fair
        else:
            colors.append('#E74C3C')  # Red - poor
    
    bars = ax.barh(diseases_sorted, aucs_sorted, color=colors, alpha=0.8)
    
    ax.set_xlabel('AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Disease AUC on Test Set\nConvNext Tiny MHSA Student (Knowledge Distillation)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0.65, 0.95)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (disease, auc) in enumerate(zip(diseases_sorted, aucs_sorted)):
        ax.text(auc + 0.005, i, f'{auc:.4f}', va='center', fontsize=10, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27AE60', alpha=0.8, label='Excellent (≥0.85)'),
        Patch(facecolor='#3498DB', alpha=0.8, label='Good (0.80-0.85)'),
        Patch(facecolor='#F39C12', alpha=0.8, label='Fair (0.75-0.80)'),
        Patch(facecolor='#E74C3C', alpha=0.8, label='Poor (<0.75)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(project_root / "results" / "test_per_disease_auc.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: results/test_per_disease_auc.png")
    plt.close()


def create_baseline_vs_kd_comparison():
    """Create comparison between baseline (no KD) vs KD results"""
    
    # Note: We compare against the validation set since we're tracking KD results
    kd_csv = project_root / "experiments" / "kd_results.csv"
    kd_df = pd.read_csv(kd_csv)
    kd_data = kd_df.iloc[0]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = ['Baseline\n(ConvNext Tiny MHSA)\n50 epochs, no KD', 
              'KD Student\n(ConvNext Tiny MHSA)\n40 epochs, T=4.0, α=0.7']
    baseline_auc = 0.8351  # From EXP-007b best
    kd_auc = kd_data['best_val_auc']
    test_auc = 0.8390  # From test eval
    
    metrics = ['Best Val AUC', 'Final Val AUC']
    baseline_vals = [0.8351, 0.8314]
    kd_vals = [kd_auc, kd_data['final_val_auc']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (No KD)', color='#E67E22', alpha=0.8)
    bars2 = ax.bar(x + width/2, kd_vals, width, label='Knowledge Distillation', color='#9B59B6', alpha=0.8)
    
    ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('Baseline vs Knowledge Distillation\nConvNext Tiny MHSA Student Model', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11, loc='lower left')
    ax.set_ylim(0.81, 0.845)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add test AUC annotation
    ax.axhline(y=test_auc, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Test AUC: {test_auc:.4f}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(project_root / "results" / "baseline_vs_kd_comparison.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: results/baseline_vs_kd_comparison.png")
    plt.close()


def create_training_efficiency_chart():
    """Create chart showing training efficiency (epochs vs AUC)"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    configs = ['Baseline\n(50 epochs)', 'KD\n(40 epochs)', 'Best at Test\n(Converged)']
    epochs = [50, 40, 33]  # KD converged at epoch 15 but ran 40, test eval on best
    aucs = [0.8351, 0.8446, 0.8390]  # Baseline best, KD best val, Test best
    colors_list = ['#E67E22', '#9B59B6', '#27AE60']
    
    bars = ax.bar(configs, aucs, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Best AUC', fontsize=12, fontweight='bold')
    ax.set_title('Training Efficiency: AUC vs Configuration\nConvNext Tiny MHSA Model Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0.82, 0.850)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels with epoch info
    for i, (bar, auc, ep) in enumerate(zip(bars, aucs, epochs)):
        ax.text(bar.get_x() + bar.get_width()/2., auc + 0.002,
               f'AUC: {auc:.4f}\nEpochs: {ep}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(project_root / "results" / "training_efficiency.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: results/training_efficiency.png")
    plt.close()


def create_summary_table():
    """Create and save summary comparison table"""
    
    kd_csv = project_root / "experiments" / "kd_results.csv"
    kd_df = pd.read_csv(kd_csv)
    kd_data = kd_df.iloc[0]
    
    test_json = project_root / "experiments" / "test_evaluation_results.json"
    with open(test_json) as f:
        test_data = json.load(f)
    
    # Create summary
    summary_data = {
        'Configuration': ['Baseline', 'KD (Validation)', 'KD (Test)'],
        'Model': ['ConvNext Tiny MHSA', 'ConvNext Tiny MHSA', 'ConvNext Tiny MHSA'],
        'Epochs': [50, 40, 'N/A'],
        'Best AUC': [0.8351, kd_data['best_val_auc'], test_data['auc_macro']],
        'Final AUC': [0.8314, kd_data['final_val_auc'], 'N/A'],
        'Macro F1': [0.1791, kd_data['final_val_f1'], test_data['f1_macro']],
        'Macro PR-AUC': [0.2635, kd_data['final_val_pr_auc'], test_data['pr_auc_macro']],
        'Notes': ['50 epochs full dataset', 'T=4.0, α=0.7, 40 epochs', 'Unseen test set']
    }
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = project_root / "experiments" / "phase_comparison_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: experiments/phase_comparison_summary.csv")
    
    return summary_df


def main():
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION CHARTS")
    print("="*70 + "\n")
    
    # Ensure results directory exists
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Load results
    kd_data, test_data = load_results()
    
    # Generate charts
    print("Creating charts...")
    create_comparison_chart(kd_data, test_data)
    create_per_disease_chart(test_data)
    create_baseline_vs_kd_comparison()
    create_training_efficiency_chart()
    summary_df = create_summary_table()
    
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    print("\n✅ All visualizations generated successfully!")
    print("   - results/kd_validation_vs_test.png")
    print("   - results/test_per_disease_auc.png")
    print("   - results/baseline_vs_kd_comparison.png")
    print("   - results/training_efficiency.png")
    print("   - experiments/phase_comparison_summary.csv")


if __name__ == "__main__":
    main()
