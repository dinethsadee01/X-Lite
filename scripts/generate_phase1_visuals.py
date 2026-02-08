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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


project_root = Path(__file__).parent.parent
results_dir = project_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

metrics_path = project_root / "experiments" / "test_results_15class_optimized.json"
thresholds_path = project_root / "scripts" / "optimal_thresholds.json"


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


def main():
    metrics, thresholds = load_data()
    plot_auc_per_disease(metrics)
    plot_f1_default_vs_optimal(metrics)
    plot_thresholds(thresholds)
    plot_macro_metrics(metrics)
    print("Phase 1 visuals saved to results/ directory")


if __name__ == "__main__":
    main()
