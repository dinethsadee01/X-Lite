"""
Generate confusion matrices for the fixed evaluation setup:
- Checkpoint: patientwise_lol best checkpoint
- Thresholds: scripts/optimal_thresholds.json
- Split: data/splits/test_df.csv
- Images: data/clahe_cache
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import multilabel_confusion_matrix

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.disease_labels import DISEASE_LABELS
from ml.data.loader import ChestXrayDataset
from ml.data.preprocessing import get_medical_transforms
from ml.models.student_model import create_student_model

# Fixed configuration requested by user
CHECKPOINT_PATH = project_root / "ml/models/new checkpoints/efficientnet_b0_performer_full_dataset_15class_patientwise_lol/best_checkpoint.pth"
THRESHOLDS_PATH = project_root / "scripts/optimal_thresholds.json"
SPLIT_CSV_PATH = project_root / "data/splits/test_df.csv"
CLAHE_CACHE_DIR = project_root / "data/clahe_cache"
MODEL_NAME = "efficientnet_b0_performer"
BATCH_SIZE = 64
NUM_WORKERS = 4
OUTPUT_DIR = project_root / "experiments/confusion_matrices"


def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Image Index" in df.columns:
        df = df.rename(columns={"Image Index": "image_id"})
    if "Finding Labels" in df.columns:
        df = df.rename(columns={"Finding Labels": "labels"})

    required = {"image_id", "labels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df


def load_classwise_thresholds(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Threshold file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    missing = [label for label in DISEASE_LABELS if label not in payload]
    if missing:
        raise ValueError("Threshold file missing labels: " + ", ".join(missing))

    thresholds = np.array([payload[label] for label in DISEASE_LABELS], dtype=np.float32)
    if np.any(thresholds < 0.0) or np.any(thresholds > 1.0):
        raise ValueError("All thresholds must be in [0, 1]")

    return thresholds


def load_model(device: torch.device):
    model = create_student_model(MODEL_NAME, num_classes=len(DISEASE_LABELS), pretrained=False)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if "student_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["student_state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def run_inference(model, loader, device):
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0).astype(np.int32)
    return y_true, y_prob


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def summarize_confusions(mcm: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, label in enumerate(DISEASE_LABELS):
        tn, fp, fn, tp = mcm[i].ravel()
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        f1 = safe_div(2 * precision * recall, precision + recall)
        accuracy = safe_div(tp + tn, tp + tn + fp + fn)

        rows.append(
            {
                "class": label,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "accuracy": accuracy,
            }
        )

    return pd.DataFrame(rows)


def _annot_cell(count: float, total: float) -> str:
    pct = 100.0 * count / total if total > 0 else 0.0
    return f"{int(count)}\n({pct:.1f}%)"


def save_single_cm_plot(cm: np.ndarray, class_name: str, out_path: Path):
    total = cm.sum()
    annot = np.array(
        [
            [_annot_cell(cm[0, 0], total), _annot_cell(cm[0, 1], total)],
            [_annot_cell(cm[1, 0], total), _annot_cell(cm[1, 1], total)],
        ]
    )

    plt.figure(figsize=(4.6, 4.0))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        linewidths=0.5,
        linecolor="white",
    )
    plt.title(f"Confusion Matrix - {class_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_combined_grid_plot(mcm: np.ndarray, out_path: Path):
    cols = 5
    rows = math.ceil(len(DISEASE_LABELS) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.8))
    axes = np.array(axes).reshape(rows, cols)

    for i, class_name in enumerate(DISEASE_LABELS):
        r = i // cols
        c = i % cols
        ax = axes[r, c]

        cm = mcm[i]
        total = cm.sum()
        annot = np.array(
            [
                [_annot_cell(cm[0, 0], total), _annot_cell(cm[0, 1], total)],
                [_annot_cell(cm[1, 0], total), _annot_cell(cm[1, 1], total)],
            ]
        )

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=["P0", "P1"],
            yticklabels=["T0", "T1"],
            linewidths=0.3,
            linecolor="white",
            ax=ax,
        )
        ax.set_title(class_name, fontsize=9)
        ax.tick_params(axis="both", labelsize=7)

    for i in range(len(DISEASE_LABELS), rows * cols):
        r = i // cols
        c = i % cols
        axes[r, c].axis("off")

    fig.suptitle("Per-Class Confusion Matrices", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(f"Thresholds file not found: {THRESHOLDS_PATH}")
    if not SPLIT_CSV_PATH.exists():
        raise FileNotFoundError(f"Split CSV not found: {SPLIT_CSV_PATH}")
    if not CLAHE_CACHE_DIR.exists():
        raise FileNotFoundError(f"CLAHE cache directory not found: {CLAHE_CACHE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = prepare_dataframe(SPLIT_CSV_PATH)

    transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    dataset = ChestXrayDataset(str(CLAHE_CACHE_DIR), df, transform=transform, is_training=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_model(device)
    y_true, y_prob = run_inference(model, loader, device)

    thresholds = load_classwise_thresholds(THRESHOLDS_PATH)
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int32)

    mcm = multilabel_confusion_matrix(y_true, y_pred)
    summary_df = summarize_confusions(mcm)

    per_class_dir = OUTPUT_DIR / "per_class"
    per_class_dir.mkdir(exist_ok=True)

    for i, class_name in enumerate(DISEASE_LABELS):
        save_single_cm_plot(mcm[i], class_name, per_class_dir / f"cm_{i:02d}_{class_name}.png")

    combined_path = OUTPUT_DIR / "confusion_matrices_grid.png"
    save_combined_grid_plot(mcm, combined_path)

    summary_csv = OUTPUT_DIR / "confusion_matrix_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_json = OUTPUT_DIR / "confusion_matrix_summary.json"
    summary_payload = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(CHECKPOINT_PATH),
        "thresholds_file": str(THRESHOLDS_PATH),
        "split_csv": str(SPLIT_CSV_PATH),
        "clahe_cache": str(CLAHE_CACHE_DIR),
        "num_samples": int(y_true.shape[0]),
        "macro_precision_from_cm": float(summary_df["precision"].mean()),
        "macro_recall_from_cm": float(summary_df["recall"].mean()),
        "macro_f1_from_cm": float(summary_df["f1"].mean()),
        "macro_accuracy_from_cm": float(summary_df["accuracy"].mean()),
        "outputs": {
            "combined_plot": str(combined_path),
            "per_class_dir": str(per_class_dir),
            "summary_csv": str(summary_csv),
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\nConfusion matrices generated successfully.")
    print(f"Samples: {y_true.shape[0]}")
    print(f"Combined grid: {combined_path}")
    print(f"Per-class plots: {per_class_dir}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}\n")


if __name__ == "__main__":
    main()
