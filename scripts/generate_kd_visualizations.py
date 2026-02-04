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
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
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
    """Create chart showing training efficiency (epochs to convergence vs AUC)"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data - epochs to reach best AUC (convergence), not total training epochs
    configs = ['Baseline\n(Converged at Epoch 28)', 'KD\n(Converged at Epoch 15)', 'Test Set\n(Best Checkpoint)']
    epochs = [28, 15, 28]  # Actual epochs where best AUC was achieved
    aucs = [0.8351, 0.8446, 0.8390]  # Baseline best, KD best val, Test best
    colors_list = ['#E67E22', '#9B59B6', '#27AE60']
    
    bars = ax.bar(configs, aucs, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Best AUC', fontsize=12, fontweight='bold')
    ax.set_title('Training Efficiency: AUC vs Convergence Epochs\nConvNext Tiny MHSA Model Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0.82, 0.850)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels with epoch info
    for i, (bar, auc, ep) in enumerate(zip(bars, aucs, epochs)):
        ax.text(bar.get_x() + bar.get_width()/2., auc + 0.002,
               f'AUC: {auc:.4f}\nConverged: Epoch {ep}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(project_root / "results" / "training_efficiency.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: results/training_efficiency.png")
    plt.close()


def create_calibration_plot():
    """Create calibration plot showing predicted probabilities vs true positive rate"""
    
    # Load test evaluation results with predictions
    test_results_path = project_root / "experiments" / "test_evaluation_results.json"
    with open(test_results_path) as f:
        test_data = json.load(f)
    
    # We need to regenerate predictions - load model and test data
    import torch
    from torch.utils.data import DataLoader
    from ml.models.student_model import create_student_model
    from ml.data.preprocessing import get_medical_transforms
    from scripts.evaluate_test_set import RawTestDataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint_path = project_root / "ml" / "models" / "checkpoints" / "kd" / "convnext_tiny_mhsa" / "best_checkpoint.pth"
    student = create_student_model('convnext_tiny_mhsa', num_classes=14, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    student.load_state_dict(checkpoint['student_state_dict'])
    student.to(device)
    student.eval()
    
    # Load test data
    test_csv = project_root / "data" / "splits" / "test.csv"
    raw_images_dir = project_root / "data" / "raw" / "images"
    val_transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    test_dataset = RawTestDataset(test_csv, raw_images_dir, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Get predictions
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = student(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Create calibration plot for top 6 diseases (by AUC)
    disease_aucs = [(i, test_data['per_disease_metrics'][DISEASE_LABELS[i]]['auc']) 
                    for i in range(14)]
    disease_aucs_sorted = sorted(disease_aucs, key=lambda x: x[1], reverse=True)
    top_diseases = [d[0] for d in disease_aucs_sorted[:6]]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    bins = np.linspace(0, 1, 11)  # 10 bins: [0-0.1), [0.1-0.2), ..., [0.9-1.0]
    
    for idx, disease_idx in enumerate(top_diseases):
        ax = axes[idx]
        disease_name = DISEASE_LABELS[disease_idx]
        
        disease_preds = preds[:, disease_idx]
        disease_targets = targets[:, disease_idx]
        
        # Calculate calibration
        bin_means = []
        bin_true_rates = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            bin_mask = (disease_preds >= bins[i]) & (disease_preds < bins[i+1])
            if i == len(bins) - 2:  # Last bin includes 1.0
                bin_mask = (disease_preds >= bins[i]) & (disease_preds <= bins[i+1])
            
            if bin_mask.sum() > 0:
                bin_preds = disease_preds[bin_mask]
                bin_targets = disease_targets[bin_mask]
                bin_means.append(bin_preds.mean())
                bin_true_rates.append(bin_targets.mean())
                bin_counts.append(len(bin_preds))
        
        # Plot calibration curve
        ax.plot(bin_means, bin_true_rates, 'o-', linewidth=2, markersize=8, 
                color='#2E86AB', label='Model Calibration')
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], '--', color='#E74C3C', linewidth=2, 
                alpha=0.7, label='Perfect Calibration')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=10, fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontsize=10, fontweight='bold')
        ax.set_title(f'{disease_name}\nAUC: {test_data["per_disease_metrics"][disease_name]["auc"]:.3f}',
                     fontsize=11, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=8, loc='upper left')
        
        # Add sample count annotation
        total_positives = int(disease_targets.sum())
        total_samples = len(disease_targets)
        ax.text(0.98, 0.02, f'Positives: {total_positives}/{total_samples}',
                transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Calibration Curves: Top 6 Diseases by AUC\nConvNext Tiny MHSA on Test Set',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(project_root / "results" / "calibration_curves.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: results/calibration_curves.png")
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
    create_calibration_plot()
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
    print("   - results/calibration_curves.png")
    print("   - experiments/phase_comparison_summary.csv")


if __name__ == "__main__":
    main()
