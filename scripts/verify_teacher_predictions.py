"""
Teacher Model Prediction Verification
======================================
Spot-check TorchXRayVision teacher model predictions on random test samples.
Uses optimal per-disease thresholds from Phase 1 baseline.

Usage:
    python scripts/verify_teacher_predictions.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
import json
from ml.models.teacher_model import create_teacher_model
from ml.data.xrv_preprocessing import get_xrv_teacher_preprocessor

# Explicit local imports to avoid conflicts with torchxrayvision's config module
try:
    from config.disease_labels import DISEASE_LABELS
except ImportError:
    # Fallback: define directly
    DISEASE_LABELS = [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
        'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
        'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
        'No_Finding'
    ]


# Configuration
OPTIMAL_THRESHOLDS_PATH = 'scripts/optimal_thresholds.json'
NUM_SAMPLES = 10
DEFAULT_THRESHOLD = 0.5
USE_OPTIMAL_THRESHOLDS = False  # Set to False to use fixed 0.5 threshold


def parse_labels_to_multihot(label_str: str) -> torch.Tensor:
    """Convert a ChestX-ray14 label string to 15-class multi-hot vector."""
    target = torch.zeros(len(DISEASE_LABELS), dtype=torch.float32)

    if pd.isna(label_str) or label_str == 'No Finding':
        target[14] = 1.0
        return target

    for disease in str(label_str).split('|'):
        disease = disease.strip()
        if disease in DISEASE_LABELS:
            target[DISEASE_LABELS.index(disease)] = 1.0

    return target


def main():
    print("\n" + "=" * 80)
    print("TEACHER MODEL VERIFICATION - TorchXRayVision DenseNet-121 (NIH Pretrained)")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    threshold_mode = "Optimal Per-Disease" if USE_OPTIMAL_THRESHOLDS else "Fixed 0.5"
    print(f"Threshold Mode: {threshold_mode}\n")
    
    # Load optimal thresholds if requested
    optimal_thresholds = None
    if USE_OPTIMAL_THRESHOLDS:
        thresholds_path = project_root / OPTIMAL_THRESHOLDS_PATH
        with open(thresholds_path, 'r') as f:
            optimal_thresholds = json.load(f)
        print("Loaded optimal thresholds:")
        for disease, threshold in sorted(optimal_thresholds.items(), key=lambda x: x[1]):
            print(f"  {disease:<25} {threshold:.2f}")
        print()
    
    # Paths
    clahe_cache_dir = project_root / "data" / "clahe_cache"
    test_csv = project_root / "data" / "splits" / "test.csv"
    
    # Load test data
    test_df = pd.read_csv(test_csv)
    if 'Image Index' in test_df.columns:
        test_df = test_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
    
    # Sample random images
    random_indices = np.random.choice(len(test_df), size=NUM_SAMPLES, replace=False)
    sample_df = test_df.iloc[random_indices].reset_index(drop=True)
    
    print(f"Randomly selected {NUM_SAMPLES} test images:\n")
    
    # Load teacher model
    print("Loading TorchXRayVision DenseNet-121 teacher model...")
    model = create_teacher_model(device=device)
    model.eval()
    
    # Print model info
    total_params = model.get_num_params()
    model_size_mb = model.get_model_size_mb()
    print(f"✓ Teacher model loaded")
    print(f"  Parameters: {total_params:,}")
    print(f"  Size: {model_size_mb:.1f} MB\n")
    
    # Use official TorchXRayVision preprocessing for teacher inputs
    teacher_preprocessor = get_xrv_teacher_preprocessor(image_size=224)
    
    # Process each sample
    print("=" * 80)
    with torch.no_grad():
        for idx in range(len(sample_df)):
            row = sample_df.iloc[idx]
            img_id = row['image_id']
            target = parse_labels_to_multihot(row['labels'])
            
            # Get prediction
            image_path = clahe_cache_dir / img_id
            image_batch = teacher_preprocessor.preprocess_single_image(str(image_path)).to(device)
            output = model(image_batch)
            probs = torch.sigmoid(output).cpu().numpy()[0]
            
            # Ground truth
            actual_diseases = []
            for i, label in enumerate(DISEASE_LABELS):
                if target[i] == 1:
                    actual_diseases.append(label)
            
            # Predicted diseases (using appropriate threshold)
            # Note: Teacher outputs only 14 classes (no No_Finding)
            predicted_diseases = []
            for i in range(14):  # Only first 14 classes from teacher
                label = DISEASE_LABELS[i]
                # Get threshold for this disease
                if USE_OPTIMAL_THRESHOLDS and optimal_thresholds:
                    threshold = optimal_thresholds.get(label, DEFAULT_THRESHOLD)
                else:
                    threshold = DEFAULT_THRESHOLD
                
                if probs[i] >= threshold:
                    predicted_diseases.append(f"{label} ({probs[i]:.3f})")
            
            # Display
            print(f"\n[Sample {idx + 1}] Image: {img_id}")
            print(f"-" * 80)
            
            # Format actual labels
            if not actual_diseases:
                actual_str = "No Finding"
            else:
                actual_str = ', '.join(actual_diseases)
            
            # Format predicted labels  
            if not predicted_diseases:
                predicted_str = "No Finding"
            else:
                predicted_str = ', '.join(predicted_diseases)
            
            print(f"ACTUAL:    {actual_str}")
            print(f"PREDICTED: {predicted_str}")
            
            # Show all probabilities (sorted by confidence, first 14 classes only)
            # Note: Teacher outputs 14 classes; No_Finding is handled separately by hard labels
            prob_list = [(DISEASE_LABELS[i], probs[i]) for i in range(14)]
            prob_list.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nDisease Probabilities (top 5):")
            for disease, prob in prob_list[:5]:
                disease_idx = DISEASE_LABELS.index(disease)
                marker = "✓" if target[disease_idx] == 1 else " "
                
                # Get threshold for this disease
                if USE_OPTIMAL_THRESHOLDS and optimal_thresholds:
                    threshold = optimal_thresholds.get(disease, DEFAULT_THRESHOLD)
                else:
                    threshold = DEFAULT_THRESHOLD
                
                pred_marker = "!" if prob >= threshold else ""
                print(f"  [{marker}] {disease:<25} {prob:.4f} {pred_marker}")
            
            # Note: No_Finding is NOT part of teacher's 14-class output
            # It's handled separately as a hard label target during KD training
            print(f"\n[Teacher outputs only 14 classes; No_Finding handled separately in KD]")
            
            print("=" * 80)
    
    print(f"\n✓ Teacher model verification complete!")
    print(f"Legend: [✓] = Actually present in ground truth, [!] = Predicted positive at threshold\n")


if __name__ == '__main__':
    main()
