import pandas as pd
import sys
import torch
sys.path.insert(0, '.')

from config import DISEASE_LABELS, NUM_CLASSES

def parse_labels(label_str: str) -> torch.Tensor:
    """Same logic as in ChestXrayDataset._parse_labels"""
    label_vector = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    
    # Handle "No Finding" as explicit 15th class
    if pd.isna(label_str) or label_str == 'No Finding':
        label_vector[14] = 1.0  # Set No_Finding class (index 14) to 1
        return label_vector
    
    # Split labels and encode diseases (indices 0-13)
    labels = label_str.split('|')
    for label in labels:
        label = label.strip()
        if label in DISEASE_LABELS:
            idx = DISEASE_LABELS.index(label)
            label_vector[idx] = 1.0
    
    return label_vector


# Load a few samples
df = pd.read_csv('data/splits/train.csv')

# Create dataset to test label parsing
print("Testing label encoding:\n")
print(f"DISEASE_LABELS ({len(DISEASE_LABELS)} total):")
for i, label in enumerate(DISEASE_LABELS):
    print(f"  [{i}] {label}")

print(f"\n{'='*60}")
print("Sample encodings:\n")

for i in range(10):
    label_str = df.iloc[i]['Finding Labels']
    label_vec = parse_labels(label_str)
    
    # Show which classes are 1
    active_classes = []
    for j, val in enumerate(label_vec):
        if val == 1.0:
            active_classes.append(DISEASE_LABELS[j])
    
    print(f"Label: '{label_str}'")
    print(f"  Encoded as: {active_classes}")
    print(f"  Vector sum: {label_vec.sum().item()}")
    print()

print(f"{'='*60}")
print("Sanity check: No_Finding samples")
no_finding_count = (df['Finding Labels'] == 'No Finding').sum()
print(f"Samples with 'No Finding': {no_finding_count}")
