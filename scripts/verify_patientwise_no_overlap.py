"""
Verify patientwise split integrity by checking patient ID overlap across splits.

Outputs:
- Console summary of unique patients and pairwise overlaps
- A figure showing overlap matrix and split patient counts
- CSV files with overlap counts and (optionally) overlapping patient IDs

Usage:
  python scripts/verify_patientwise_no_overlap.py
  python scripts/verify_patientwise_no_overlap.py --splits-dir data/splits_old
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def detect_patient_column(df: pd.DataFrame) -> str:
    """Find the most likely patient-id column name."""
    candidates = [
        "Patient ID",
        "patient_id",
        "patientid",
        "patientId",
        "pid",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        "Could not find a patient ID column. Expected one of: "
        + ", ".join(candidates)
    )


def load_split_patient_sets(splits_dir: Path) -> tuple[dict[str, set], str]:
    """Load train/val/test CSVs and return patient-id sets."""
    split_files = {
        "Train": splits_dir / "train_df.csv",
        "Val": splits_dir / "val_df.csv",
        "Test": splits_dir / "test_df.csv",
    }

    missing = [name for name, path in split_files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split files in {splits_dir}: {', '.join(missing)}"
        )

    dataframes = {name: pd.read_csv(path) for name, path in split_files.items()}
    patient_col = detect_patient_column(next(iter(dataframes.values())))

    patient_sets = {}
    for split_name, df in dataframes.items():
        if patient_col not in df.columns:
            raise KeyError(
                f"Column '{patient_col}' not found in {split_name} split. "
                "All splits must use the same patient-id column."
            )

        ids = df[patient_col].dropna().astype(str).str.strip()
        patient_sets[split_name] = set(ids.tolist())

    return patient_sets, patient_col


def compute_overlap_matrix(patient_sets: dict[str, set]) -> pd.DataFrame:
    """Compute pairwise patient overlap counts for all splits."""
    split_names = list(patient_sets.keys())
    matrix = np.zeros((len(split_names), len(split_names)), dtype=int)

    for i, s1 in enumerate(split_names):
        for j, s2 in enumerate(split_names):
            matrix[i, j] = len(patient_sets[s1].intersection(patient_sets[s2]))

    return pd.DataFrame(matrix, index=split_names, columns=split_names)


def save_overlap_figure(overlap_df: pd.DataFrame, patient_sets: dict[str, set], out_path: Path) -> None:
    """Create and save visualization for overlap verification."""
    split_names = list(patient_sets.keys())
    counts = [len(patient_sets[name]) for name in split_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap: diagonal = unique patients per split, off-diagonal should be 0
    ax1 = axes[0]
    sns.heatmap(
        overlap_df,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Patient Count"},
        ax=ax1,
    )
    ax1.set_title("Patient Overlap Matrix", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Split", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Split", fontsize=12, fontweight="bold")

    # Bar chart: unique patients per split
    ax2 = axes[1]
    bars = ax2.bar(split_names, counts, color=["#4C78A8", "#F58518", "#54A24B"], alpha=0.9)
    ax2.set_title("Unique Patients per Split", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Split", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Unique Patient Count", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    off_diag = overlap_df.values.copy()
    np.fill_diagonal(off_diag, 0)
    max_off_diag = int(off_diag.max())
    verdict = "PASS" if max_off_diag == 0 else "FAIL"
    fig.suptitle(
        f"Patientwise Overlap Check: {verdict} (max off-diagonal overlap = {max_off_diag})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify no patient overlap across train/val/test splits and generate figure."
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="data/splits_old",
        help="Directory containing train_df.csv, val_df.csv, test_df.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    splits_dir = project_root / args.splits_dir
    out_dir = project_root / args.out_dir

    patient_sets, patient_col = load_split_patient_sets(splits_dir)
    overlap_df = compute_overlap_matrix(patient_sets)

    # Summaries
    print("\n" + "=" * 80)
    print("PATIENTWISE SPLIT OVERLAP VERIFICATION")
    print("=" * 80)
    print(f"Splits directory : {splits_dir}")
    print(f"Patient ID column: {patient_col}\n")

    for split_name, ids in patient_sets.items():
        print(f"{split_name:<6} unique patients: {len(ids):,}")

    train_val = len(patient_sets["Train"] & patient_sets["Val"])
    train_test = len(patient_sets["Train"] & patient_sets["Test"])
    val_test = len(patient_sets["Val"] & patient_sets["Test"])

    print("\nPairwise overlaps:")
    print(f"  Train ∩ Val : {train_val}")
    print(f"  Train ∩ Test: {train_test}")
    print(f"  Val ∩ Test  : {val_test}")

    max_overlap = max(train_val, train_test, val_test)
    if max_overlap == 0:
        print("\nPASS: No overlapping patients across splits.")
    else:
        print("\nFAIL: Overlapping patients detected.")

    # Save outputs
    figure_path = out_dir / "patientwise_overlap_verification.png"
    overlap_csv = out_dir / "patientwise_overlap_matrix.csv"

    save_overlap_figure(overlap_df, patient_sets, figure_path)
    overlap_df.to_csv(overlap_csv)

    print(f"\nSaved figure: {figure_path}")
    print(f"Saved matrix: {overlap_csv}")

    # Save detailed overlap IDs only when overlaps exist
    overlap_rows = []
    pairs = [("Train", "Val"), ("Train", "Test"), ("Val", "Test")]
    for a, b in pairs:
        common = sorted(patient_sets[a] & patient_sets[b])
        for pid in common:
            overlap_rows.append({"split_a": a, "split_b": b, "patient_id": pid})

    if overlap_rows:
        detail_path = out_dir / "patientwise_overlapping_patient_ids.csv"
        pd.DataFrame(overlap_rows).to_csv(detail_path, index=False)
        print(f"Saved overlap IDs: {detail_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
