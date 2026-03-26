import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys

# Add project root to import model factory
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.models.student_model import MODEL_CONFIGS, create_student_model


def normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean([1, 2], keepdim=True)) / (x.std([1, 2], keepdim=True) + 1e-7)


def resolve_label_columns(df: pd.DataFrame, requested_labels: List[str]) -> Tuple[List[str], List[str]]:
    normalized_to_actual = {c.strip().lower().replace("_", " "): c for c in df.columns}
    resolved = []
    missing = []
    for label in requested_labels:
        key = label.strip().lower().replace("_", " ")
        actual = normalized_to_actual.get(key)
        if actual is None:
            missing.append(label)
        else:
            resolved.append(actual)
    return resolved, missing


class XRayEvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, image_col: str, label_cols: List[str], transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_col = image_col
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(self.image_dir) / row[self.image_col]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[self.label_cols].values.astype(np.float32), dtype=torch.float32)
        return image, labels


def infer_num_classes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    for key in ("head.classifier.3.weight", "module.head.classifier.3.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return None


def find_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
    # Handles plain state_dict checkpoints and wrapped trainer checkpoints.
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def build_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    architecture: Optional[str] = None,
) -> Tuple[torch.nn.Module, str, int]:
    raw_ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = find_state_dict(raw_ckpt)

    num_classes = infer_num_classes_from_state_dict(state_dict)
    if num_classes is None:
        raise RuntimeError(f"Could not infer num_classes from checkpoint: {checkpoint_path}")

    if architecture:
        model = create_student_model(architecture, num_classes=num_classes, pretrained=False).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model, architecture, num_classes

    # Auto-detect architecture by trying all known configs.
    for arch in MODEL_CONFIGS.keys():
        try:
            model = create_student_model(arch, num_classes=num_classes, pretrained=False).to(device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            return model, arch, num_classes
        except Exception:
            continue

    raise RuntimeError(f"Could not auto-detect architecture for: {checkpoint_path}")


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(y.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = (y_prob >= threshold).astype(np.int32)

    metrics = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
    }

    try:
        metrics["auc_macro"] = float(roc_auc_score(y_true, y_prob, average="macro"))
    except ValueError:
        metrics["auc_macro"] = 0.0

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense121 checkpoints on test split.")
    parser.add_argument("--checkpoints-root", type=str, default="scripts/dense121", help="Root folder to search checkpoints")
    parser.add_argument("--checkpoint-glob", type=str, default="**/*best*.pth", help="Glob for checkpoint files")
    parser.add_argument("--architecture", type=str, default=None, help="Force architecture instead of auto-detect")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-dir", type=str, default="data/clahe_cache")
    parser.add_argument("--test-csv", type=str, default="scripts/dense121/split_csv/test_df.csv")
    parser.add_argument("--image-col", type=str, default="Image")
    parser.add_argument("--output-dir", type=str, default="scripts/dense121/results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    requested_labels = [
        "Cardiomegaly",
        "Emphysema",
        "Effusion",
        "Hernia",
        "Infiltration",
        "Mass",
        "Nodule",
        "Atelectasis",
        "Pneumothorax",
        "Pleural_Thickening",
        "Pneumonia",
        "Fibrosis",
        "Edema",
        "Consolidation",
        "No_Finding",
    ]

    test_csv_path = project_root / args.test_csv
    image_dir = project_root / args.image_dir
    checkpoints_root = project_root / args.checkpoints_root
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_csv_path)
    labels, missing = resolve_label_columns(test_df, requested_labels)
    if missing:
        print(f"Warning: Missing labels skipped: {missing}")
    if not labels:
        raise RuntimeError("No valid label columns were found in test CSV.")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(normalize_tensor),
        ]
    )

    test_dataset = XRayEvalDataset(test_df, str(image_dir), args.image_col, labels, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint_paths = sorted(checkpoints_root.glob(args.checkpoint_glob))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints found with pattern '{args.checkpoint_glob}' under {checkpoints_root}"
        )

    rows = []
    for ckpt_path in checkpoint_paths:
        print(f"\nEvaluating: {ckpt_path}")
        try:
            model, arch, num_classes = build_model_from_checkpoint(
                ckpt_path, device=device, architecture=args.architecture
            )
            metrics = evaluate_model(model, test_loader, device=device, threshold=args.threshold)

            row = {
                "checkpoint": str(ckpt_path.relative_to(project_root)),
                "architecture": arch,
                "num_classes": num_classes,
                "threshold": args.threshold,
            }
            row.update(metrics)
            rows.append(row)
            print(
                f"  f1_macro={metrics['f1_macro']:.4f} | "
                f"precision_macro={metrics['precision_macro']:.4f} | "
                f"recall_macro={metrics['recall_macro']:.4f} | "
                f"auc_macro={metrics['auc_macro']:.4f}"
            )
        except Exception as exc:
            print(f"  Failed: {exc}")

    if not rows:
        raise RuntimeError("No checkpoints could be evaluated successfully.")

    results_df = pd.DataFrame(rows).sort_values(by="f1_macro", ascending=False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"checkpoint_eval_{ts}.csv"
    json_path = output_dir / f"checkpoint_eval_{ts}.json"

    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=2)

    print("\nTop checkpoints by macro F1:")
    print(results_df[["checkpoint", "architecture", "f1_macro", "precision_macro", "recall_macro", "auc_macro"]].head(10).to_string(index=False))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
