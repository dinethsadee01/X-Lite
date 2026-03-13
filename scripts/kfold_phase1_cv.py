"""
True K-Fold Cross-Validation for Phase 1 Model (15-class)
==========================================================
Performs true K-Fold CV by re-training the selected Phase 1 model on each fold,
then evaluating on that fold's validation split.

Default model: efficientnet_b0_performer
Default classes: 15 (14 diseases + No_Finding)

Outputs:
- experiments/kfold_phase1_fold_results.csv
- experiments/kfold_phase1_summary.json

Usage:
  python scripts/kfold_phase1_cv.py
  python scripts/kfold_phase1_cv.py --n_splits 5 --epochs 20 --batch_size 32
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import DISEASE_LABELS, NUM_CLASSES
from ml.data.augmentation import get_augmentation_pipeline
from ml.data.loader import ChestXrayDataset, get_sample_weights
from ml.data.preprocessing import get_medical_transforms
from ml.models.student_model import create_student_model
from ml.training.losses import WeightedBCEWithLogitsLoss, calculate_pos_weights
from ml.training.trainer import create_trainer
from scripts.training_utils import evaluate_final_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="True K-Fold CV for Phase 1 best model")
    parser.add_argument("--student_model", type=str, default="efficientnet_b0_performer")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(project_root / "data" / "clahe_cache"),
        help="Path to image directory (CLAHE cache recommended for Phase 1 parity)",
    )
    parser.add_argument(
        "--cv_pool",
        type=str,
        choices=["train", "train_val"],
        default="train_val",
        help="Use train only or train+val as the K-Fold pool",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_split_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Image Index" in df.columns:
        df = df.rename(columns={"Image Index": "image_id", "Finding Labels": "labels"})
    required = {"image_id", "labels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[["image_id", "labels"]].reset_index(drop=True)


def load_cv_pool(cv_pool: str) -> pd.DataFrame:
    train_csv = project_root / "data" / "splits" / "train.csv"
    val_csv = project_root / "data" / "splits" / "val.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Train split not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation split not found: {val_csv}")

    train_df = normalize_split_df(pd.read_csv(train_csv))
    val_df = normalize_split_df(pd.read_csv(val_csv))

    if cv_pool == "train":
        pool_df = train_df.copy()
    else:
        pool_df = pd.concat([train_df, val_df], ignore_index=True)

    # Drop any duplicate image ids defensively.
    pool_df = pool_df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    return pool_df


def build_stratification_targets(df: pd.DataFrame) -> np.ndarray:
    # Simple multi-label stratification proxy: primary label from label string.
    # This preserves rough class balance without requiring external iterative stratification libs.
    no_finding_idx = DISEASE_LABELS.index("No_Finding") if "No_Finding" in DISEASE_LABELS else 0
    targets = np.full(len(df), no_finding_idx, dtype=np.int64)

    for i, label_str in enumerate(df["labels"].fillna("No Finding")):
        if label_str == "No Finding":
            targets[i] = no_finding_idx
            continue

        labels = [x.strip() for x in str(label_str).split("|") if x.strip()]
        primary = labels[0] if labels else "No Finding"

        if primary == "No Finding":
            targets[i] = no_finding_idx
        elif primary in DISEASE_LABELS:
            targets[i] = DISEASE_LABELS.index(primary)
        else:
            targets[i] = no_finding_idx

    return targets


def calculate_fold_pos_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    label_counts = np.zeros(NUM_CLASSES, dtype=np.float64)

    no_finding_idx = DISEASE_LABELS.index("No_Finding") if "No_Finding" in DISEASE_LABELS else None

    for label_str in train_df["labels"].fillna("No Finding"):
        if label_str == "No Finding":
            if no_finding_idx is not None:
                label_counts[no_finding_idx] += 1
            continue

        labels = [x.strip() for x in str(label_str).split("|") if x.strip()]
        if not labels and no_finding_idx is not None:
            label_counts[no_finding_idx] += 1

        for lbl in labels:
            if lbl in DISEASE_LABELS:
                label_counts[DISEASE_LABELS.index(lbl)] += 1

    pos_weights = calculate_pos_weights(torch.tensor(label_counts), total_samples=len(train_df))
    return pos_weights.to(device)


def build_fold_loaders(
    data_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    use_weighted_sampler: bool,
):
    train_transform = get_augmentation_pipeline(augmentation_strength="medium")
    val_transform = get_medical_transforms(use_clahe=False, use_denoising=False)

    train_ds = ChestXrayDataset(str(data_dir), train_df, transform=train_transform, is_training=True)
    val_ds = ChestXrayDataset(str(data_dir), val_df, transform=val_transform, is_training=False)

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sample_weights = get_sample_weights(train_df)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_and_evaluate_fold(
    fold_idx: int,
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args,
    device: torch.device,
):
    print("\n" + "=" * 80)
    print(f"FOLD {fold_idx + 1}/{args.n_splits}")
    print("=" * 80)
    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples:   {len(val_df):,}")

    train_loader, val_loader = build_fold_loaders(
        data_dir=Path(args.data_dir),
        train_df=train_df,
        val_df=val_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=args.use_weighted_sampler,
    )

    model = create_student_model(model_name, num_classes=NUM_CLASSES, pretrained=True)

    pos_weights = calculate_fold_pos_weights(train_df, device)
    criterion = WeightedBCEWithLogitsLoss(pos_weights=pos_weights)

    fold_ckpt_dir = (
        project_root
        / "ml"
        / "models"
        / "checkpoints"
        / "kfold_phase1"
        / model_name
        / f"fold_{fold_idx + 1}"
    )
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_dir=fold_ckpt_dir,
        device=device,
        use_amp=True,
        num_classes=NUM_CLASSES,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    start = time.time()
    trainer.train(
        num_epochs=args.epochs,
        scheduler=scheduler,
        early_stopping_patience=args.early_stopping_patience,
        verbose=True,
    )
    elapsed_min = (time.time() - start) / 60.0

    best_ckpt_path = fold_ckpt_dir / "best_checkpoint.pth"
    if not best_ckpt_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found for fold {fold_idx + 1}: {best_ckpt_path}")

    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    if "student_state_dict" in ckpt:
        trainer.model.load_state_dict(ckpt["student_state_dict"])
    elif "model_state_dict" in ckpt:
        trainer.model.load_state_dict(ckpt["model_state_dict"])
    else:
        trainer.model.load_state_dict(ckpt)

    val_metrics = evaluate_final_metrics(trainer.model, val_loader, device)

    fold_result = {
        "fold": fold_idx + 1,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "best_epoch": int(trainer.best_epoch),
        "best_val_auc_during_training": float(trainer.best_val_auc),
        "val_auc_macro": float(val_metrics.get("AUC_macro", 0.0)),
        "val_f1_macro": float(val_metrics.get("F1_macro", 0.0)),
        "val_precision_macro": float(val_metrics.get("Precision_macro", 0.0)),
        "val_recall_macro": float(val_metrics.get("Recall_macro", 0.0)),
        "val_pr_auc_macro": float(val_metrics.get("PR_AUC_macro", 0.0)),
        "training_time_minutes": float(elapsed_min),
        "checkpoint_dir": str(fold_ckpt_dir),
    }

    print(
        f"Fold {fold_idx + 1} done | "
        f"AUC={fold_result['val_auc_macro']:.4f}, "
        f"F1={fold_result['val_f1_macro']:.4f}, "
        f"PR-AUC={fold_result['val_pr_auc_macro']:.4f}"
    )

    return fold_result


def summarize_results(results_df: pd.DataFrame) -> dict:
    summary_metrics = {}
    for metric in [
        "val_auc_macro",
        "val_f1_macro",
        "val_precision_macro",
        "val_recall_macro",
        "val_pr_auc_macro",
        "training_time_minutes",
    ]:
        summary_metrics[metric] = {
            "mean": float(results_df[metric].mean()),
            "std": float(results_df[metric].std(ddof=0)),
            "min": float(results_df[metric].min()),
            "max": float(results_df[metric].max()),
        }
    return summary_metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    print("\n" + "=" * 80)
    print("TRUE K-FOLD CROSS-VALIDATION (PHASE 1 MODEL)")
    print("=" * 80)
    print(f"Model: {args.student_model}")
    print(f"K: {args.n_splits}")
    print(f"Epochs/fold: {args.epochs}")
    print(f"CV pool: {args.cv_pool}")
    print(f"Data dir: {args.data_dir}")
    print("=" * 80)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pool_df = load_cv_pool(args.cv_pool)
    print(f"\nCV pool size: {len(pool_df):,}")

    strat_targets = build_stratification_targets(pool_df)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    fold_results = []
    overall_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(pool_df, strat_targets)):
        train_df = pool_df.iloc[train_idx].reset_index(drop=True)
        val_df = pool_df.iloc[val_idx].reset_index(drop=True)

        fold_result = train_and_evaluate_fold(
            fold_idx=fold_idx,
            model_name=args.student_model,
            train_df=train_df,
            val_df=val_df,
            args=args,
            device=device,
        )
        fold_results.append(fold_result)

    total_minutes = (time.time() - overall_start) / 60.0
    results_df = pd.DataFrame(fold_results)
    summary = summarize_results(results_df)

    out_csv = project_root / "experiments" / "kfold_phase1_fold_results.csv"
    out_json = project_root / "experiments" / "kfold_phase1_summary.json"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)

    payload = {
        "config": {
            "student_model": args.student_model,
            "n_splits": args.n_splits,
            "epochs": args.epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "use_weighted_sampler": args.use_weighted_sampler,
            "seed": args.seed,
            "cv_pool": args.cv_pool,
            "data_dir": str(data_dir),
            "num_classes": NUM_CLASSES,
            "class_labels": DISEASE_LABELS,
        },
        "summary": summary,
        "total_runtime_minutes": float(total_minutes),
        "timestamp": datetime.now().isoformat(),
        "fold_results": fold_results,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 80)
    print("K-FOLD CV COMPLETE")
    print("=" * 80)
    print(results_df[["fold", "val_auc_macro", "val_f1_macro", "val_pr_auc_macro", "training_time_minutes"]].to_string(index=False))
    print("\nSummary (mean +- std):")
    print(f"AUC_macro:       {summary['val_auc_macro']['mean']:.4f} +- {summary['val_auc_macro']['std']:.4f}")
    print(f"F1_macro:        {summary['val_f1_macro']['mean']:.4f} +- {summary['val_f1_macro']['std']:.4f}")
    print(f"Precision_macro: {summary['val_precision_macro']['mean']:.4f} +- {summary['val_precision_macro']['std']:.4f}")
    print(f"Recall_macro:    {summary['val_recall_macro']['mean']:.4f} +- {summary['val_recall_macro']['std']:.4f}")
    print(f"PR-AUC_macro:    {summary['val_pr_auc_macro']['mean']:.4f} +- {summary['val_pr_auc_macro']['std']:.4f}")
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
