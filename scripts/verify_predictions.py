"""
Prediction Verification Script
===============================
Spot-check model predictions on random test samples to verify correctness.

Usage:
    python scripts/verify_predictions.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
from ml.models.student_model import create_student_model
from ml.data.loader import ChestXrayDataset
from ml.data.preprocessing import get_medical_transforms
from config.disease_labels import DISEASE_LABELS


# Configuration
MODEL_NAME = 'efficientnet_b0_performer'
CHECKPOINT_PATH = 'ml/models/checkpoints/efficientnet_b0_performer_full_dataset_15class/best_checkpoint.pth'
NUM_SAMPLES = 10
THRESHOLD = 0.5


def main():
    print("\n" + "=" * 80)
    print("PREDICTION VERIFICATION - Random Test Samples")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Paths
    clahe_cache_dir = project_root / "data" / "clahe_cache"
    test_csv = project_root / "data" / "splits" / "test.csv"
    checkpoint_path = project_root / CHECKPOINT_PATH
    
    # Load test data
    test_df = pd.read_csv(test_csv)
    if 'Image Index' in test_df.columns:
        test_df = test_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
    
    # Sample random images
    random_indices = np.random.choice(len(test_df), size=NUM_SAMPLES, replace=False)
    sample_df = test_df.iloc[random_indices].reset_index(drop=True)
    
    print(f"Randomly selected {NUM_SAMPLES} test images:\n")
    
    # Load model (15 classes: 14 diseases + No_Finding)
    print("Loading model...")
    from config import NUM_CLASSES
    model = create_student_model(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'student_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['student_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("✓ Model loaded\n")
    
    # Create dataset with single-image batches
    val_transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    test_dataset = ChestXrayDataset(
        str(clahe_cache_dir), sample_df, transform=val_transform, is_training=False
    )
    
    # Process each sample
    print("=" * 80)
    with torch.no_grad():
        for idx in range(len(sample_df)):
            image, target, img_id = test_dataset[idx]
            
            # Get prediction
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            probs = torch.sigmoid(output).cpu().numpy()[0]
            
            # Ground truth
            actual_diseases = []
            for i, label in enumerate(DISEASE_LABELS):
                if target[i] == 1:
                    actual_diseases.append(label)
            
            # Predicted (threshold at 0.5)
            predicted_diseases = []
            for i, label in enumerate(DISEASE_LABELS):
                if probs[i] >= THRESHOLD:
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
            
            # Show all probabilities (sorted by confidence, excluding No_Finding)
            prob_list = [(DISEASE_LABELS[i], probs[i]) for i in range(14)]  # First 14 classes only
            prob_list.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nDisease Probabilities (top 5):")
            for disease, prob in prob_list[:5]:
                marker = "✓" if target[DISEASE_LABELS.index(disease)] == 1 else " "
                print(f"  [{marker}] {disease:<25} {prob:.4f}")
            
            # Show No_Finding probability separately
            no_finding_prob = probs[14]
            no_finding_actual = target[14] == 1
            marker = "✓" if no_finding_actual else " "
            print(f"\nNo Finding:")
            print(f"  [{marker}] No_Finding               {no_finding_prob:.4f}")
            
            print("=" * 80)
    
    print(f"\n✓ Verification complete!")
    print(f"Legend: [✓] = Actually present in ground truth\n")


if __name__ == '__main__':
    main()
