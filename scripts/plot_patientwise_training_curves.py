"""
Plot Training Curves for train_full_dataset_patientwise.py runs
===============================================================
Generates Train/Val Loss and Train/Val AUC curves from saved training history.
Works even for interrupted runs by recovering history from checkpoint files.

Usage:
  python scripts/plot_patientwise_training_curves.py
  python scripts/plot_patientwise_training_curves.py --checkpoint_dir "ml/models/new checkpoints/efficientnet_b0_performer_full_dataset_15class_patientwise_lol"
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="Plot train/val loss and AUC curves from training history")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Run checkpoint directory. If omitted, auto-detect latest patientwise run.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Output directory for plots. Defaults to <checkpoint_dir>/plots",
    )
    return parser.parse_args()


def find_latest_patientwise_dir() -> Path:
    root = project_root / "ml" / "models" / "new checkpoints"
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")

    candidates = [
        p for p in root.iterdir()
        if p.is_dir() and "efficientnet_b0_performer_full_dataset_15class_patientwise_old" in p.name
    ]

    if not candidates:
        raise FileNotFoundError(
            "No patientwise checkpoint directories found under ml/models/new checkpoints"
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_history(checkpoint_dir: Path) -> dict:
    """Load history from training_history.json or fallback to checkpoint['history']."""
    history_file = checkpoint_dir / "training_history.json"
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
        print(f"Loaded history from: {history_file}")
        return history

    # Fallback for interrupted runs
    for ckpt_name in ["last_checkpoint.pth", "best_checkpoint.pth"]:
        ckpt_path = checkpoint_dir / ckpt_name
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "history" in ckpt:
                print(f"Loaded history from: {ckpt_path} -> history")
                return ckpt["history"]

    raise FileNotFoundError(
        f"Could not find training history in {checkpoint_dir} (expected training_history.json or checkpoint with history)."
    )


def validate_history(history: dict):
    required = ["train_loss", "val_loss", "train_auc", "val_auc"]
    missing = [k for k in required if k not in history]
    if missing:
        raise ValueError(f"History missing required keys: {missing}")

    n = min(len(history["train_loss"]), len(history["val_loss"]), len(history["train_auc"]), len(history["val_auc"]))
    if n == 0:
        raise ValueError("History is empty. No epochs available to plot.")

    # Trim to equal length for safe plotting
    for k in required:
        history[k] = history[k][:n]

    return n


def plot_curves(history: dict, save_dir: Path):
    n_epochs = len(history["train_loss"])
    epochs = np.arange(1, n_epochs + 1)

    train_loss = np.array(history["train_loss"], dtype=float)
    val_loss = np.array(history["val_loss"], dtype=float)
    train_auc = np.array(history["train_auc"], dtype=float)
    val_auc = np.array(history["val_auc"], dtype=float)

    best_epoch = int(np.argmax(val_auc)) + 1
    best_val_auc = float(np.max(val_auc))

    # Combined figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Loss subplot
    ax = axes[0]
    ax.plot(epochs, train_loss, color="#1f77b4", linewidth=2, label="Train Loss")
    ax.plot(epochs, val_loss, color="#d62728", linewidth=2, label="Val Loss")
    ax.set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    # AUC subplot
    ax = axes[1]
    ax.plot(epochs, train_auc, color="#2ca02c", linewidth=2, label="Train AUC")
    ax.plot(epochs, val_auc, color="#9467bd", linewidth=2, label="Val AUC")
    ax.axvline(best_epoch, color="orange", linestyle="--", linewidth=1.5, label=f"Best Epoch: {best_epoch}")
    ax.scatter([best_epoch], [best_val_auc], color="red", zorder=5)
    ax.set_title("Training vs Validation AUC", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    fig.suptitle("Patientwise Training Curves", fontsize=15, fontweight="bold")
    fig.tight_layout()

    combined_path = save_dir / "patientwise_training_curves_combined.png"
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Separate AUC-only figure (useful for thesis)
    fig_auc, ax_auc = plt.subplots(figsize=(10, 6))
    ax_auc.plot(epochs, train_auc, color="#2ca02c", linewidth=2, label="Train AUC")
    ax_auc.plot(epochs, val_auc, color="#9467bd", linewidth=2, label="Val AUC")
    ax_auc.axvline(best_epoch, color="orange", linestyle="--", linewidth=1.5, label=f"Best Epoch: {best_epoch}")
    ax_auc.scatter([best_epoch], [best_val_auc], color="red", zorder=5)
    ax_auc.set_title("AUC Curves (Train vs Validation)", fontsize=14, fontweight="bold")
    ax_auc.set_xlabel("Epoch")
    ax_auc.set_ylabel("AUC")
    ax_auc.grid(True, linestyle="--", alpha=0.3)
    ax_auc.legend()
    fig_auc.tight_layout()

    auc_path = save_dir / "patientwise_auc_curves.png"
    fig_auc.savefig(auc_path, dpi=300, bbox_inches="tight")
    plt.close(fig_auc)

    # Separate loss-only figure
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    ax_loss.plot(epochs, train_loss, color="#1f77b4", linewidth=2, label="Train Loss")
    ax_loss.plot(epochs, val_loss, color="#d62728", linewidth=2, label="Val Loss")
    ax_loss.set_title("Loss Curves (Train vs Validation)", fontsize=14, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle="--", alpha=0.3)
    ax_loss.legend()
    fig_loss.tight_layout()

    loss_path = save_dir / "patientwise_loss_curves.png"
    fig_loss.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close(fig_loss)

    return {
        "combined": combined_path,
        "auc": auc_path,
        "loss": loss_path,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "num_epochs": n_epochs,
    }


def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else find_latest_patientwise_dir()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    save_dir = Path(args.save_dir) if args.save_dir else (checkpoint_dir / "plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("PLOT PATIENTWISE TRAINING CURVES")
    print("=" * 80)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Save dir:       {save_dir}")

    history = load_history(checkpoint_dir)
    n_epochs = validate_history(history)
    print(f"Recovered epochs: {n_epochs}")

    out = plot_curves(history, save_dir)

    print("\n✓ Plots generated successfully")
    print(f"  - Combined: {out['combined']}")
    print(f"  - AUC:      {out['auc']}")
    print(f"  - Loss:     {out['loss']}")
    print(f"  - Best Epoch: {out['best_epoch']} (Val AUC: {out['best_val_auc']:.4f})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
