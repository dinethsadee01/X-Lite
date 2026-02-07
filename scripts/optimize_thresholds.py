"""
Threshold Optimization Script
==============================
Finds optimal prediction thresholds per disease to maximize F1 score.

Instead of using fixed 0.5 threshold for all classes, each disease gets its own
optimal threshold based on validation set performance.

Usage:
    python scripts/✅ TRAINING COMPLETE: efficientnet_b0_performer
======================================================================
   Best Val AUC: 0.8162 (epoch 12)
   Final Val AUC: 0.7998
   Final Val F1: 0.0990
   Final Val PR-AUC: 0.2658
   Final Val Precision: 0.4482
   Final Val Recall: 0.0680
   Training time: 89.5 minutes

Output:
    - Saves optimal thresholds to: scripts/optimal_thresholds.json
    - Prints per-disease F1 scores at optimal thresholds
    - Compares with default 0.5 threshold baseline
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
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from ml.models.student_model import create_student_model
from ml.data.loader import ChestXrayDataset
from ml.data.preprocessing import get_medical_transforms
from config.disease_labels import DISEASE_LABELS


# Configuration
MODEL_NAME = 'efficientnet_b0_performer'
CHECKPOINT_PATH = 'ml/models/checkpoints/efficientnet_b0_performer_full_dataset_15class/best_checkpoint.pth'
NUM_CLASSES = 15
THRESHOLD_RANGE = np.arange(0.1, 1.0, 0.05)  # Test thresholds from 0.1 to 0.95 in 0.05 steps
OUTPUT_PATH = project_root / 'scripts' / 'optimal_thresholds.json'


def main():
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Threshold range: {THRESHOLD_RANGE[0]:.2f} to {THRESHOLD_RANGE[-1]:.2f}")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Paths
    clahe_cache_dir = project_root / "data" / "clahe_cache"
    val_csv = project_root / "data" / "splits" / "val.csv"
    checkpoint_path = project_root / CHECKPOINT_PATH
    
    # Load validation data
    print("Loading validation data...")
    val_df = pd.read_csv(val_csv)
    if 'Image Index' in val_df.columns:
        val_df = val_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
    
    val_transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    val_dataset = ChestXrayDataset(
        str(clahe_cache_dir), val_df, transform=val_transform, is_training=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Load model
    print("Loading model...")
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
    
    # Get predictions on entire validation set
    print("Computing predictions on validation set...")
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_img, batch_target, _ in tqdm(val_loader, desc="Validation"):
            batch_img = batch_img.to(device)
            output = model(batch_img)
            probs = torch.sigmoid(output).cpu().numpy()
            
            all_probs.append(probs)
            all_targets.append(batch_target.numpy())
    
    all_probs = np.vstack(all_probs)  # [num_val_samples, 14]
    all_targets = np.vstack(all_targets)  # [num_val_samples, 14]
    
    print(f"✓ Predictions computed: {all_probs.shape}")
    print(f"✓ Targets shape: {all_targets.shape}\n")
    
    # Find optimal threshold for each disease
    print("=" * 80)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("=" * 80)
    
    optimal_thresholds = {}
    results_summary = []
    
    for class_idx in range(NUM_CLASSES):
        disease_name = DISEASE_LABELS[class_idx]
        
        # Get predictions and targets for this class
        class_probs = all_probs[:, class_idx]
        class_targets = all_targets[:, class_idx]
        
        # Skip if no positive samples
        if class_targets.sum() == 0:
            optimal_thresholds[disease_name] = 0.5
            print(f"\n{disease_name:<25} [SKIPPED - no positive samples]")
            continue
        
        # Find threshold that maximizes F1
        best_f1 = -1
        best_threshold = 0.5
        best_precision = 0
        best_recall = 0
        
        for threshold in THRESHOLD_RANGE:
            class_preds = (class_probs >= threshold).astype(int)
            
            # Compute F1
            f1 = f1_score(class_targets, class_preds, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision_score(class_targets, class_preds, zero_division=0)
                best_recall = recall_score(class_targets, class_preds, zero_division=0)
        
        optimal_thresholds[disease_name] = best_threshold
        results_summary.append({
            'disease': disease_name,
            'optimal_threshold': best_threshold,
            'f1': best_f1,
            'precision': best_precision,
            'recall': best_recall,
            'num_positives': int(class_targets.sum())
        })
        
        # Compare with default 0.5 threshold
        class_preds_default = (class_probs >= 0.5).astype(int)
        f1_default = f1_score(class_targets, class_preds_default, zero_division=0)
        precision_default = precision_score(class_targets, class_preds_default, zero_division=0)
        recall_default = recall_score(class_targets, class_preds_default, zero_division=0)
        
        # Print results
        print(f"\n{disease_name:<25} (Positives: {int(class_targets.sum()):,})")
        print(f"  Optimal Threshold:      {best_threshold:.2f}")
        print(f"    F1:        {best_f1:.4f}  (default 0.5: {f1_default:.4f})")
        print(f"    Precision: {best_precision:.4f}  (default 0.5: {precision_default:.4f})")
        print(f"    Recall:    {best_recall:.4f}  (default 0.5: {recall_default:.4f})")
        
        if best_f1 > f1_default:
            improvement = ((best_f1 - f1_default) / f1_default * 100) if f1_default > 0 else 0
            print(f"  ✓ F1 IMPROVED by {improvement:.1f}%")
        elif best_f1 < f1_default:
            degradation = ((f1_default - best_f1) / f1_default * 100) if f1_default > 0 else 0
            print(f"  ✗ F1 degraded by {degradation:.1f}%")
        else:
            print(f"  = F1 unchanged")
    
    # Save optimal thresholds
    print("\n" + "=" * 80)
    print(f"Saving optimal thresholds to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(optimal_thresholds, f, indent=2)
    print("✓ Saved\n")
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    thresholds = [t for t in optimal_thresholds.values()]
    print(f"Mean optimal threshold:   {np.mean(thresholds):.3f}")
    print(f"Median optimal threshold: {np.median(thresholds):.3f}")
    print(f"Min optimal threshold:    {np.min(thresholds):.3f}")
    print(f"Max optimal threshold:    {np.max(thresholds):.3f}")
    print(f"Std dev:                  {np.std(thresholds):.3f}")
    
    print("\n✓ Threshold optimization complete!")
    print(f"Use optimal thresholds in test_model.py by setting: USE_OPTIMAL_THRESHOLDS = True")


if __name__ == '__main__':
    main()
