"""
Phase 1 Baseline Visuals (No KD)
================================
Generates charts for the 15-class baseline with optimized thresholds.

Outputs (results/):
- phase1_auc_per_disease.png
- phase1_f1_default_vs_optimal.png
- phase1_thresholds.png
- phase1_macro_metrics.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

results_dir = project_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

metrics_path = project_root / "experiments" / "test_results_15class_optimized.json"
thresholds_path = project_root / "scripts" / "optimal_thresholds.json"

CHECKPOINT_PATH = (
    project_root
    / "ml/models/checkpoints/efficientnet_b0_performer_full_dataset_15class/best_checkpoint.pth"
)


def load_data():
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)
    return metrics, thresholds


def plot_auc_per_disease(metrics):
    diseases = list(metrics["optimal"].keys())
    aucs = [metrics["optimal"][d]["auc"] for d in diseases]

    plt.figure(figsize=(12, 6))
    plt.bar(diseases, aucs, color="#1f77b4")
    plt.title("Test AUC per Disease (15-class baseline)")
    plt.ylabel("AUC")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / "phase1_auc_per_disease.png", dpi=200)
    plt.close()


def plot_f1_default_vs_optimal(metrics):
    diseases = list(metrics["optimal"].keys())
    f1_default = [metrics["default_0.5"][d]["f1"] for d in diseases]
    f1_optimal = [metrics["optimal"][d]["f1"] for d in diseases]

    x = np.arange(len(diseases))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, f1_default, width, label="Default 0.5", color="#d62728")
    plt.bar(x + width / 2, f1_optimal, width, label="Optimal", color="#2ca02c")
    plt.title("Test F1: Default 0.5 vs Optimal Thresholds")
    plt.ylabel("F1")
    plt.ylim(0.0, 1.0)
    plt.xticks(x, diseases, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "phase1_f1_default_vs_optimal.png", dpi=200)
    plt.close()


def plot_thresholds(thresholds):
    diseases = list(thresholds.keys())
    values = [thresholds[d] for d in diseases]

    plt.figure(figsize=(12, 6))
    plt.bar(diseases, values, color="#9467bd")
    plt.title("Optimal Thresholds per Disease")
    plt.ylabel("Threshold")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / "phase1_thresholds.png", dpi=200)
    plt.close()


def plot_macro_metrics(metrics):
    macro = metrics["macro_averages"]

    labels = ["F1", "Precision", "Recall"]
    default_vals = [macro["f1_default"], macro["precision_default"], macro["recall_default"]]
    optimal_vals = [macro["f1_optimal"], macro["precision_optimal"], macro["recall_optimal"]]

    x = np.arange(len(labels))
    width = 0.4

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, default_vals, width, label="Default 0.5", color="#d62728")
    plt.bar(x + width / 2, optimal_vals, width, label="Optimal", color="#2ca02c")
    plt.title("Macro Metrics: Default 0.5 vs Optimal")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "phase1_macro_metrics.png", dpi=200)
    plt.close()


def get_test_predictions():
    """Load model and run inference on the full test set to get raw predictions."""
    import torch
    import pandas as pd
    from sklearn.metrics import roc_curve, roc_auc_score
    from ml.models.student_model import create_student_model
    from ml.data.loader import ChestXrayDataset
    from ml.data.preprocessing import get_medical_transforms
    from config.disease_labels import DISEASE_LABELS, NUM_CLASSES
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = create_student_model("efficientnet_b0_performer", num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if "student_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["student_state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("✓ Model loaded")

    # Load test set
    clahe_cache_dir = project_root / "data" / "clahe_cache"
    test_csv = project_root / "data" / "splits" / "test.csv"
    test_df = pd.read_csv(test_csv)
    if "Image Index" in test_df.columns:
        test_df = test_df.rename(columns={"Image Index": "image_id", "Finding Labels": "labels"})

    val_transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    test_dataset = ChestXrayDataset(str(clahe_cache_dir), test_df, transform=val_transform, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            images = batch_data[0].to(device)
            labels = batch_data[1]
            probs = torch.sigmoid(model(images)).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    print(f"✓ Inference complete on {len(targets)} samples")
    return preds, targets, DISEASE_LABELS


def plot_roc_curves():
    """Plot per-disease AUC-ROC curves for the Phase 1 best checkpoint."""
    from sklearn.metrics import roc_curve, roc_auc_score

    print("\nGenerating ROC curves (requires model inference)...")
    preds, targets, disease_labels = get_test_predictions()

    n_diseases = len(disease_labels)  # 15
    ncols = 4
    nrows = (n_diseases + ncols) // ncols  # +ncols to reserve last panel for overlay

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    auc_scores = {}
    curve_data = {}

    for i, disease in enumerate(disease_labels):
        ax = axes[i]
        y_true = targets[:, i]
        y_score = preds[:, i]

        if len(np.unique(y_true)) < 2:
            ax.text(0.5, 0.5, "No positive samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(disease)
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        auc_scores[disease] = auc
        curve_data[disease] = (fpr, tpr)

        ax.plot(fpr, tpr, lw=2, color="#1f77b4", label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(disease.replace("_", " "), fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")

    # Last panel: overlay all curves
    ax_overlay = axes[n_diseases]
    cmap = plt.colormaps.get_cmap("tab20").resampled(n_diseases)
    for j, disease in enumerate(disease_labels):
        if disease in curve_data:
            fpr, tpr = curve_data[disease]
            ax_overlay.plot(fpr, tpr, lw=1.2, color=cmap(j),
                            label=f"{disease.replace('_', ' ')[:18]} ({auc_scores[disease]:.2f})")
    ax_overlay.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    macro_auc = np.mean(list(auc_scores.values()))
    ax_overlay.set_title(f"All Diseases Overlay\nMacro AUC = {macro_auc:.3f}", fontsize=10, fontweight="bold")
    ax_overlay.set_xlabel("False Positive Rate", fontsize=9)
    ax_overlay.set_ylabel("True Positive Rate", fontsize=9)
    ax_overlay.legend(fontsize=6, loc="lower right", ncol=2)
    ax_overlay.grid(alpha=0.3, linestyle="--")

    # Hide any unused axes
    for k in range(n_diseases + 1, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        "Phase 1 AUC-ROC Curves — EfficientNet-B0 Performer (Best Checkpoint)\nTest Set (15 Classes)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(results_dir / "phase1_roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("✓ Saved: results/phase1_roc_curves.png")


def main():
    metrics, thresholds = load_data()
    plot_auc_per_disease(metrics)
    plot_f1_default_vs_optimal(metrics)
    plot_thresholds(thresholds)
    plot_macro_metrics(metrics)
    plot_roc_curves()
    print("Phase 1 visuals saved to results/ directory")


if __name__ == "__main__":
    main()
