"""
Evaluate subset accuracy and hamming loss for fixed 14-class setup.

Fixed setup:
- Checkpoint: efficientnet_b0_performer_full_dataset_14class_patientwise_lol/best_checkpoint.pth
- Data split: data/splits/test_df.csv
- Images: data/clahe_cache
- Thresholds: optimal_thresholds14.json (per-class)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, hamming_loss

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.disease_labels14 import DISEASE_LABELS14, NUM_CLASSES14
from ml.data.loader14 import ChestXrayDataset14
from ml.data.preprocessing import get_medical_transforms
from ml.models.student_model import create_student_model

# Fixed configuration
CHECKPOINT_PATH = project_root / "ml/models/new checkpoints 14/efficientnet_b0_performer_full_dataset_14class_patientwise_lol/best_checkpoint.pth"
SPLIT_CSV_PATH = project_root / "data/splits/test_df.csv"
CLAHE_CACHE_DIR = project_root / "data/clahe_cache"
MODEL_NAME = "efficientnet_b0_performer"
BATCH_SIZE = 64
NUM_WORKERS = 4
THRESHOLDS_PATH_14 = project_root / "scripts/optimal_thresholds14.json"
OUT_DIR = project_root / "experiments"


def load_thresholds_for_14_classes() -> dict:
    """Load per-class optimal thresholds with fallback logic."""
    # Try 14-class specific thresholds first
    if THRESHOLDS_PATH_14.exists():
        with open(THRESHOLDS_PATH_14, "r") as f:
            return json.load(f)
    
    # Default: 0.5 for all classes
    print("Warning: Threshold files not found, using 0.5 for all classes")
    return {label: 0.5 for label in DISEASE_LABELS14}


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


def load_model(device: torch.device):
    model = create_student_model(MODEL_NAME, num_classes=NUM_CLASSES14, pretrained=False)

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


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not SPLIT_CSV_PATH.exists():
        raise FileNotFoundError(f"Split CSV not found: {SPLIT_CSV_PATH}")
    if not CLAHE_CACHE_DIR.exists():
        raise FileNotFoundError(f"CLAHE cache directory not found: {CLAHE_CACHE_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    thresholds_dict = load_thresholds_for_14_classes()
    thresholds = np.array([thresholds_dict[label] for label in DISEASE_LABELS14], dtype=np.float32)

    df = prepare_dataframe(SPLIT_CSV_PATH)
    transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    dataset = ChestXrayDataset14(str(CLAHE_CACHE_DIR), df, transform=transform, is_training=False)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_model(device)
    y_true, y_prob = run_inference(model, loader, device)
    # Apply per-class optimal thresholds
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int32)

    subset_acc = float(accuracy_score(y_true, y_pred))
    hamming = float(hamming_loss(y_true, y_pred))

    per_class_acc = []
    for i, label in enumerate(DISEASE_LABELS14):
        cls_acc = float((y_true[:, i] == y_pred[:, i]).mean())
        per_class_acc.append({"class": label, "accuracy": cls_acc, "threshold": float(thresholds[i])})

    result = {
        "timestamp": datetime.now().isoformat(),
        "setup": "14_class_without_no_finding",
        "checkpoint": str(CHECKPOINT_PATH),
        "split_csv": str(SPLIT_CSV_PATH),
        "num_samples": int(y_true.shape[0]),
        "thresholds": thresholds_dict,
        "subset_accuracy": subset_acc,
        "hamming_loss": hamming,
        "per_class_accuracy": per_class_acc,
    }

    OUT_DIR.mkdir(exist_ok=True)
    out_file = OUT_DIR / "subset_hamming14_best_checkpoint_test.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nSUBSET ACCURACY & HAMMING LOSS (14-CLASS)")
    print(f"Samples: {y_true.shape[0]}")
    print(f"Thresholds: {thresholds_dict}")
    print(f"Subset Accuracy: {subset_acc:.6f}")
    print(f"Hamming Loss: {hamming:.6f}")
    print(f"Saved: {out_file}\n")


if __name__ == "__main__":
    main()
