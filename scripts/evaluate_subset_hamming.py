"""
Evaluate Subset Accuracy and Hamming Loss for a Multi-label Checkpoint.

Usage examples:
  python scripts/evaluate_subset_hamming.py
  python scripts/evaluate_subset_hamming.py --checkpoint ml/models/new checkpoints/kd_student_best.pth --split test
  python scripts/evaluate_subset_hamming.py --threshold 0.4 --split-dir data/splits_old
"""

import argparse
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

from config import NUM_CLASSES
from config.disease_labels import DISEASE_LABELS
from ml.data.loader import ChestXrayDataset
from ml.data.preprocessing import get_medical_transforms
from ml.models.student_model import create_student_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute subset accuracy and hamming loss for a trained checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ml\\models\\new checkpoints\\efficientnet_b0_performer_full_dataset_15class_patientwise_lol\\best_checkpoint.pth",
        help="Path to model checkpoint relative to project root",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="efficientnet_b0_performer",
        help="Student model architecture name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default="splits",
        help="Optional split folder under data/ (example: splits_old). If empty, auto-detects.",
    )
    parser.add_argument(
        "--clahe-cache-dir",
        type=str,
        default="data/clahe_cache",
        help="Path to CLAHE cache directory relative to project root",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for binarizing predictions",
    )
    parser.add_argument(
        "--thresholds-file",
        type=str,
        default="",
        help=(
            "Optional JSON file for class-wise thresholds. "
            "Supports either a list of 15 floats or a dict with key 'thresholds'."
        ),
    )
    return parser.parse_args()


def load_thresholds(args) -> np.ndarray:
    if args.thresholds_file:
        thresholds_path = project_root / args.thresholds_file
        if not thresholds_path.exists():
            raise FileNotFoundError(f"Threshold file not found: {thresholds_path}")

        with open(thresholds_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            thresholds = payload
        elif isinstance(payload, dict) and "thresholds" in payload:
            thresholds = payload["thresholds"]
        elif isinstance(payload, dict):
            missing = [label for label in DISEASE_LABELS if label not in payload]
            if missing:
                raise ValueError(
                    "Threshold dict is missing labels: " + ", ".join(missing)
                )
            thresholds = [payload[label] for label in DISEASE_LABELS]
        else:
            raise ValueError(
                "Invalid thresholds JSON format. Use a list, {'thresholds': [...]}, "
                "or {'Atelectasis': ..., ..., 'No_Finding': ...}."
            )

        if len(thresholds) != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} thresholds, got {len(thresholds)}"
            )

        thresholds = np.asarray(thresholds, dtype=np.float32)
        if np.any(thresholds < 0.0) or np.any(thresholds > 1.0):
            raise ValueError("All thresholds must be in [0, 1]")
        return thresholds

    return np.full((NUM_CLASSES,), args.threshold, dtype=np.float32)


def resolve_split_csv(split: str, split_dir: str) -> Path:
    if split_dir:
        candidate = project_root / "data" / split_dir / f"{split}_df.csv"
        if not candidate.exists():
            raise FileNotFoundError(f"Split file not found: {candidate}")
        return candidate

    candidates = [
        project_root / "data" / "splits_old" / f"{split}_df.csv",
        project_root / "data" / "splits" / f"{split}_df.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find split file in data/splits_old or data/splits. "
        "Provide --split-dir explicitly."
    )


def load_model_checkpoint(checkpoint_path: Path, model_name: str, device: torch.device):
    model = create_student_model(model_name, num_classes=NUM_CLASSES, pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = project_root / args.checkpoint
    split_csv = resolve_split_csv(args.split, args.split_dir)
    clahe_cache_dir = project_root / args.clahe_cache_dir

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not clahe_cache_dir.exists():
        raise FileNotFoundError(f"CLAHE cache directory not found: {clahe_cache_dir}")

    df = prepare_dataframe(split_csv)

    transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    dataset = ChestXrayDataset(
        str(clahe_cache_dir), df, transform=transform, is_training=False
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_model_checkpoint(checkpoint_path, args.model_name, device)
    y_true, y_prob = run_inference(model, loader, device)
    thresholds = load_thresholds(args)
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int32)

    metrics_15 = compute_metrics(y_true, y_pred)
    metrics_14 = compute_metrics(y_true[:, :14], y_pred[:, :14])

    no_finding_true = y_true[:, 14]
    no_finding_pred = y_pred[:, 14]
    no_finding_accuracy = float((no_finding_true == no_finding_pred).mean())

    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "model_name": args.model_name,
        "split_csv": str(split_csv),
        "split": args.split,
        "num_samples": int(y_true.shape[0]),
        "threshold_mode": "classwise" if args.thresholds_file else "global",
        "threshold": None if args.thresholds_file else args.threshold,
        "thresholds": thresholds.tolist(),
        "metrics_15_classes": metrics_15,
        "metrics_14_pathologies_only": metrics_14,
        "no_finding_binary_accuracy": no_finding_accuracy,
    }

    print("\n" + "=" * 80)
    print("SUBSET ACCURACY & HAMMING LOSS")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split CSV:  {split_csv}")
    print(f"Samples:    {y_true.shape[0]}")
    if args.thresholds_file:
        print(f"Thresholds: class-wise ({project_root / args.thresholds_file})")
    else:
        print(f"Threshold:  {args.threshold}")
    print("-" * 80)
    print("15 classes (14 pathologies + No_Finding)")
    print(f"  Subset Accuracy: {metrics_15['subset_accuracy']:.6f}")
    print(f"  Hamming Loss:    {metrics_15['hamming_loss']:.6f}")
    print("14 pathologies only (No_Finding excluded)")
    print(f"  Subset Accuracy: {metrics_14['subset_accuracy']:.6f}")
    print(f"  Hamming Loss:    {metrics_14['hamming_loss']:.6f}")
    print(f"No_Finding binary accuracy: {no_finding_accuracy:.6f}")

    out_dir = project_root / "experiments"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"subset_hamming_{Path(args.checkpoint).stem}_{args.split}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results: {out_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
