"""
Analyze Validation Curves for Early Stopping Behavior
======================================================
Reads training_history.json from all model checkpoints and visualizes:
- Validation AUC curves
- Early stopping points
- Convergence patterns
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
checkpoint_dir = project_root / "ml" / "models" / "checkpoints"
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)

# Model names
models = [
    "efficientnet_b0_mhsa",
    "efficientnet_b0_performer",
    "convnext_tiny_mhsa",
    "convnext_tiny_performer",
    "mobilenet_v3_large_mhsa",
    "mobilenet_v3_large_performer"
]

# Load all histories
histories = {}
for model_name in models:
    history_path = checkpoint_dir / model_name / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            histories[model_name] = json.load(f)
        print(f"✓ Loaded {model_name}: {len(histories[model_name]['val_auc'])} epochs")
    else:
        print(f"✗ Missing {model_name}")

# Create visualization
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Validation AUC Curves - Early Stopping Analysis', fontsize=16, fontweight='bold')

for idx, (model_name, ax) in enumerate(zip(models, axes.flatten())):
    if model_name not in histories:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title(model_name)
        continue
    
    history = histories[model_name]
    val_auc = history['val_auc']
    train_auc = history.get('train_auc', [])
    
    epochs = range(1, len(val_auc) + 1)
    
    # Plot validation AUC
    ax.plot(epochs, val_auc, 'b-', linewidth=2, label='Val AUC', marker='o', markersize=4)
    
    # Plot training AUC if available
    if train_auc and len(train_auc) == len(val_auc):
        ax.plot(epochs, train_auc, 'g--', linewidth=1.5, label='Train AUC', alpha=0.7)
    
    # Mark best epoch
    best_epoch = np.argmax(val_auc) + 1
    best_auc = max(val_auc)
    ax.axvline(best_epoch, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best Epoch: {best_epoch}')
    ax.scatter([best_epoch], [best_auc], color='red', s=100, zorder=5, marker='*')
    
    # Mark early stopping point (last epoch trained)
    stopped_epoch = len(val_auc)
    ax.axvline(stopped_epoch, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Stopped: Epoch {stopped_epoch}')
    
    # Styling
    ax.set_title(f'{model_name}\nBest: {best_auc:.4f} @ Epoch {best_epoch}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_ylim([0.5, 1.0])

plt.tight_layout()
output_path = results_dir / "validation_curves_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved visualization: {output_path}")

# Print detailed analysis
print("\n" + "="*70)
print("EARLY STOPPING ANALYSIS")
print("="*70)

print(f"\n{'Model':<30} {'Epochs':<8} {'Best@':<8} {'Stopped@':<10} {'Gap':<8} {'Best AUC':<10} {'Final AUC':<10}")
print("-"*100)

for model_name in models:
    if model_name not in histories:
        continue
    
    history = histories[model_name]
    val_auc = history['val_auc']
    
    best_epoch = np.argmax(val_auc) + 1
    best_auc = max(val_auc)
    stopped_epoch = len(val_auc)
    final_auc = val_auc[-1]
    gap = stopped_epoch - best_epoch
    
    print(f"{model_name:<30} {stopped_epoch:<8} {best_epoch:<8} {stopped_epoch:<10} {gap:<8} {best_auc:<10.4f} {final_auc:<10.4f}")

# Analysis insights
print("\n" + "="*70)
print("INSIGHTS")
print("="*70)

for model_name in models:
    if model_name not in histories:
        continue
    
    history = histories[model_name]
    val_auc = history['val_auc']
    
    best_epoch = np.argmax(val_auc) + 1
    stopped_epoch = len(val_auc)
    gap = stopped_epoch - best_epoch
    
    # Check if early stopping triggered appropriately
    if gap >= 10:
        status = "✅ PROPERLY STOPPED (patience=10 reached)"
    elif gap < 5:
        status = f"⚠️ STOPPED TOO EARLY (only {gap} epochs after best)"
    else:
        status = f"⏳ STOPPED WITHIN PATIENCE ({gap} epochs after best)"
    
    # Check convergence
    last_10_auc = val_auc[-10:] if len(val_auc) >= 10 else val_auc
    auc_std = np.std(last_10_auc)
    if auc_std < 0.001:
        convergence = "✅ CONVERGED (stable AUC)"
    elif auc_std < 0.005:
        convergence = "⏳ NEAR CONVERGENCE (low variance)"
    else:
        convergence = f"⚠️ NOT CONVERGED (high variance: {auc_std:.4f})"
    
    print(f"\n{model_name}:")
    print(f"  {status}")
    print(f"  {convergence}")
    
    # Check if validation was still improving
    if stopped_epoch > 10:
        last_5 = val_auc[-5:]
        prev_5 = val_auc[-10:-5]
        if np.mean(last_5) > np.mean(prev_5):
            print(f"  ⚠️ MIGHT BENEFIT FROM MORE EPOCHS (validation still improving in last 5 epochs)")

print("\n" + "="*70)
print("View detailed curves at: results/validation_curves_analysis.png")
print("="*70)
