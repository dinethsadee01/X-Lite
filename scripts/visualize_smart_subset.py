"""
Smart Subset Visualization
==========================
Shows what the smart training subset contains and how it differs from full train.

Usage:
    python scripts/visualize_smart_subset.py

Output:
    results/smart_subset_summary.png
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.disease_labels import DISEASE_LABELS


RARE_LABELS = {"Hernia", "Pneumonia", "Fibrosis", "Emphysema"}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Image Index" in df.columns:
        df = df.rename(columns={
            "Image Index": "image_id",
            "Finding Labels": "labels",
        })
    return df


def parse_labels(label_str: str) -> list:
    if pd.isna(label_str) or str(label_str).strip() == "No Finding":
        return []
    return [lbl.strip() for lbl in str(label_str).split("|") if lbl.strip()]


def compute_label_counts(df: pd.DataFrame) -> dict:
    counts = {disease: 0 for disease in DISEASE_LABELS}
    for label_str in df["labels"]:
        for label in parse_labels(label_str):
            if label in counts:
                counts[label] += 1
    return counts


def compute_subset_composition(df: pd.DataFrame) -> dict:
    label_lists = df["labels"].apply(parse_labels)
    is_no_finding = label_lists.apply(lambda x: len(x) == 0)
    has_rare = label_lists.apply(lambda x: any(lbl in RARE_LABELS for lbl in x))
    sick = ~is_no_finding
    other_sick = sick & (~has_rare)

    return {
        "Rare Sick": int(has_rare.sum()),
        "Other Sick": int(other_sick.sum()),
        "No Finding": int(is_no_finding.sum()),
    }


def main():
    splits_dir = project_root / "data" / "splits"
    train_csv = splits_dir / "train.csv"
    subset_csv = splits_dir / "train_subset.csv"

    if not train_csv.exists():
        print(f"✗ Train CSV not found: {train_csv}")
        return
    if not subset_csv.exists():
        print(f"✗ Subset CSV not found: {subset_csv}")
        print("  Run: python scripts/create_smart_subset.py")
        return

    train_df = normalize_columns(pd.read_csv(train_csv))
    subset_df = normalize_columns(pd.read_csv(subset_csv))

    if "labels" not in train_df.columns or "labels" not in subset_df.columns:
        print("✗ Could not find labels column in one of the CSVs")
        return

    # Compute counts
    full_counts = compute_label_counts(train_df)
    subset_counts = compute_label_counts(subset_df)
    subset_comp = compute_subset_composition(subset_df)

    # Prepare for plots
    diseases = list(DISEASE_LABELS)
    full_vals = np.array([full_counts[d] for d in diseases])
    subset_vals = np.array([subset_counts[d] for d in diseases])

    # Figure layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # Plot 1: Subset composition
    ax1 = fig.add_subplot(gs[0, 0])
    comp_labels = list(subset_comp.keys())
    comp_values = list(subset_comp.values())
    colors = ["#d95f02", "#1b9e77", "#7570b3"]
    ax1.bar(comp_labels, comp_values, color=colors, edgecolor="black", linewidth=0.6)
    ax1.set_title("Smart Subset Composition", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Images")
    for i, v in enumerate(comp_values):
        ax1.text(i, v, f" {v:,}", va="bottom", fontsize=9)

    # Plot 2: Full vs Subset per-disease counts
    ax2 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(diseases))
    ax2.barh(y, full_vals, color="#d9d9d9", edgecolor="black", linewidth=0.4, label="Full Train")
    ax2.barh(y, subset_vals, color="#4c78a8", edgecolor="black", linewidth=0.4, label="Smart Subset")
    ax2.set_yticks(y)
    ax2.set_yticklabels(diseases)
    ax2.invert_yaxis()
    ax2.set_xlabel("Disease Count")
    ax2.set_title("Per-Disease Counts: Full vs Subset", fontsize=12, fontweight="bold")
    ax2.legend()

    # Plot 3: Subset prevalence (percentage)
    ax3 = fig.add_subplot(gs[1, :])
    subset_pct = (subset_vals / max(len(subset_df), 1)) * 100
    ax3.bar(diseases, subset_pct, color="#59a14f", edgecolor="black", linewidth=0.4)
    ax3.set_ylabel("% of Subset")
    ax3.set_title("Subset Disease Prevalence", fontsize=12, fontweight="bold")
    ax3.set_xticks(range(len(diseases)))
    ax3.set_xticklabels(diseases, rotation=45, ha="right")
    ax3.grid(axis="y", alpha=0.25)

    # Save
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "smart_subset_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("=" * 70)
    print("SMART SUBSET VISUALIZATION")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Subset size: {len(subset_df):,}")
    print(f"Full train size: {len(train_df):,}")


if __name__ == "__main__":
    main()
