"""
Create a fixed-size training subset for baseline experiments.

Strategy:
- Keep ALL rare cases: Hernia, Pneumonia, Fibrosis, Emphysema
- Sample exactly 3,000 images from remaining sick cases
- Sample "No Finding" to match total sick count (1:1 balance)
- Target size: ~20,000 images
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.disease_labels import DISEASE_LABELS


RARE_LABELS = {"Hernia", "Pneumonia", "Fibrosis", "Emphysema"}
COMMON_LABELS = {"Infiltration", "Effusion"}
OTHER_SICK_TARGET = 6000  # Adjusted to reach ~20k total (10k sick + 10k No Finding)


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


def main():
    train_csv = project_root / "data" / "splits" / "train.csv"
    output_csv = project_root / "data" / "splits" / "train_subset.csv"

    if not train_csv.exists():
        print(f"✗ Train CSV not found: {train_csv}")
        return

    df = pd.read_csv(train_csv)
    df = normalize_columns(df)

    if "labels" not in df.columns:
        print("✗ Could not find labels column in train.csv")
        return

    # Build flags
    label_lists = df["labels"].apply(parse_labels)
    df = df.copy()
    df["_labels_list"] = label_lists
    df["_is_no_finding"] = df["_labels_list"].apply(lambda x: len(x) == 0)
    df["_has_rare"] = df["_labels_list"].apply(lambda x: any(lbl in RARE_LABELS for lbl in x))
    df["_has_common"] = df["_labels_list"].apply(lambda x: any(lbl in COMMON_LABELS for lbl in x))

    # 1) Keep all rare cases
    rare_df = df[df["_has_rare"]].copy()

    # 2) Sample exactly N from remaining sick cases (excluding rare)
    remaining_sick = df[(~df["_is_no_finding"]) & (~df["_has_rare"])].copy()
    if len(remaining_sick) <= OTHER_SICK_TARGET:
        other_sick_sample = remaining_sick
        print("⚠️  Remaining sick pool smaller than target; using all remaining sick images.")
    else:
        other_sick_sample = remaining_sick.sample(n=OTHER_SICK_TARGET, random_state=42)

    sick_subset = pd.concat([rare_df, other_sick_sample], ignore_index=True)
    sick_subset = sick_subset.drop_duplicates(subset=["image_id"], keep="first")

    # 4) Sample "No Finding" to match sick count (1:1)
    no_finding_df = df[df["_is_no_finding"]].copy()
    target_no_finding = len(sick_subset)
    if target_no_finding > len(no_finding_df):
        print("⚠️  Not enough 'No Finding' images to match sick count; using all available.")
        target_no_finding = len(no_finding_df)

    no_finding_sample = no_finding_df.sample(n=target_no_finding, random_state=42)

    subset_df = pd.concat([sick_subset, no_finding_sample], ignore_index=True)
    subset_df = subset_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Clean temp columns
    subset_df = subset_df.drop(columns=["_labels_list", "_is_no_finding", "_has_rare", "_has_common"])

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_csv(output_csv, index=False)

    # Summary
    total = len(subset_df)
    sick_count = len(sick_subset)
    no_finding_count = len(no_finding_sample)

    print("=" * 70)
    print("SMART TRAINING SUBSET CREATED")
    print("=" * 70)
    print(f"Output: {output_csv}")
    print(f"Total subset size: {total:,}")
    print(f"Sick images: {sick_count:,}")
    print(f"No Finding images: {no_finding_count:,}")
    print(f"No Finding ratio: {no_finding_count / total:.2f}")

    # Label counts for recalculated weights
    label_counts = compute_label_counts(subset_df)
    print("\nPer-disease counts in subset:")
    for disease, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {disease:<25} {count:>6,}")

    if total != (sick_count * 2):
        print("\n⚠️  No Finding count did not match sick count (1:1).")
    if total < 19000 or total > 21000:
        print("\n⚠️  Subset size outside the ~20k target range.")
        print("    If needed, adjust OTHER_SICK_TARGET or rare selection criteria.")


if __name__ == "__main__":
    main()
