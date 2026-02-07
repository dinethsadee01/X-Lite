"""
Check No Finding Distribution
==============================
Analyze how many samples have "No Finding" vs diseases.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

# Load splits
train_df = pd.read_csv(project_root / "data" / "splits" / "train.csv")
val_df = pd.read_csv(project_root / "data" / "splits" / "val.csv")
test_df = pd.read_csv(project_root / "data" / "splits" / "test.csv")

# Check column names
label_col = 'Finding Labels' if 'Finding Labels' in train_df.columns else 'labels'

print("\n" + "=" * 70)
print("NO FINDING DISTRIBUTION ANALYSIS")
print("=" * 70)

for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    no_finding = df[df[label_col] == 'No Finding']
    with_diseases = df[df[label_col] != 'No Finding']
    
    total = len(df)
    no_finding_pct = len(no_finding) / total * 100
    
    print(f"\n{split_name} Split:")
    print(f"  Total samples:        {total:,}")
    print(f"  No Finding:           {len(no_finding):,} ({no_finding_pct:.1f}%)")
    print(f"  With Diseases:        {len(with_diseases):,} ({100-no_finding_pct:.1f}%)")
    print(f"  Imbalance Ratio:      {len(no_finding) / len(with_diseases):.2f}:1")

print("\n" + "=" * 70)
print("RECOMMENDATIONS:")
print("=" * 70)
print("""
Option 1: Keep All Data + Weighted Loss (RECOMMENDED)
  ✓ Keeps full dataset (78k train samples)
  ✓ Use class weights to handle imbalance
  ✓ Model sees more examples overall
  ✗ Training takes longer

Option 2: Undersample "No Finding"
  ✓ Balanced classes (better learning)
  ✓ Faster training
  ✗ Loses valuable "No Finding" examples
  ✗ Smaller effective dataset
  
  Suggested ratio: Keep 1.5:1 or 2:1 (No Finding : Diseases)
  - 1.5:1 → Keep ~50% of No Finding samples
  - 2:1   → Keep ~67% of No Finding samples

Option 3: Hybrid Approach
  ✓ Moderate undersampling (2:1 ratio) + weighted loss
  ✓ Best of both worlds
""")
print("=" * 70 + "\n")
