"""
Test Set Evaluation with Optimal Thresholds
=============================================
Evaluates the 15-class model on held-out test set using optimized per-disease thresholds.

Compares:
  1. Default 0.5 threshold (baseline)
  2. Optimal per-disease thresholds (from validation set optimization)

Usage:
    python scripts/test_model_optimized.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

from ml.models.student_model import create_student_model
from ml.data.loader import ChestXrayDataset
from ml.data.preprocessing import get_medical_transforms
from config.disease_labels import DISEASE_LABELS


# Configuration
MODEL_NAME = 'efficientnet_b0_performer'
CHECKPOINT_PATH = 'ml/models/checkpoints/efficientnet_b0_performer_full_dataset_15class/best_checkpoint.pth'
OPTIMAL_THRESHOLDS_PATH = 'scripts/optimal_thresholds.json'
NUM_CLASSES = 15
BATCH_SIZE = 32


def main():
    print("\n" + "=" * 90)
    print("TEST SET EVALUATION WITH OPTIMAL THRESHOLDS")
    print("=" * 90)
    print(f"Model: {MODEL_NAME}")
    print(f"Classes: {NUM_CLASSES} (14 diseases + No_Finding)")
    print("=" * 90)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load paths
    clahe_cache_dir = project_root / "data" / "clahe_cache"
    test_csv = project_root / "data" / "splits" / "test.csv"
    checkpoint_path = project_root / CHECKPOINT_PATH
    thresholds_path = project_root / OPTIMAL_THRESHOLDS_PATH
    
    # Load optimal thresholds
    print("Loading optimal thresholds...")
    with open(thresholds_path, 'r') as f:
        optimal_thresholds = json.load(f)
    print("✓ Loaded optimal thresholds:")
    for disease, threshold in sorted(optimal_thresholds.items(), key=lambda x: x[1]):
        print(f"  {disease:<25} {threshold:.2f}")
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(test_csv)
    if 'Image Index' in test_df.columns:
        test_df = test_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
    
    test_transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    test_dataset = ChestXrayDataset(
        str(clahe_cache_dir), test_df, transform=test_transform, is_training=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    print(f"✓ Test set: {len(test_df):,} images, {len(test_loader)} batches")
    
    # Load model
    print("\nLoading model...")
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
    
    # Get predictions
    print("=" * 90)
    print("Computing predictions on test set...")
    print("=" * 90)
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_img, batch_target, _ in tqdm(test_loader, desc="Test"):
            batch_img = batch_img.to(device)
            output = model(batch_img)
            probs = torch.sigmoid(output).cpu().numpy()
            
            all_probs.append(probs)
            all_targets.append(batch_target.numpy())
    
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    
    print(f"\nPredictions shape: {all_probs.shape}")
    print(f"Targets shape: {all_targets.shape}\n")
    
    # Compute metrics with both threshold strategies
    print("=" * 90)
    print("METRICS COMPARISON: Default 0.5 vs Optimal Thresholds")
    print("=" * 90)
    
    results = {
        'default_0.5': {},
        'optimal': {}
    }
    
    # Process each disease
    for class_idx in range(NUM_CLASSES):
        disease_name = DISEASE_LABELS[class_idx]
        class_probs = all_probs[:, class_idx]
        class_targets = all_targets[:, class_idx]
        
        # Default threshold (0.5)
        preds_default = (class_probs >= 0.5).astype(int)
        
        # Optimal threshold
        optimal_threshold = optimal_thresholds.get(disease_name, 0.5)
        preds_optimal = (class_probs >= optimal_threshold).astype(int)
        
        # Compute metrics
        auc = roc_auc_score(class_targets, class_probs) if len(np.unique(class_targets)) > 1 else 0
        pr_auc = average_precision_score(class_targets, class_probs) if class_targets.sum() > 0 else 0
        
        # Default 0.5
        f1_default = f1_score(class_targets, preds_default, zero_division=0)
        prec_default = precision_score(class_targets, preds_default, zero_division=0)
        rec_default = recall_score(class_targets, preds_default, zero_division=0)
        
        # Optimal
        f1_optimal = f1_score(class_targets, preds_optimal, zero_division=0)
        prec_optimal = precision_score(class_targets, preds_optimal, zero_division=0)
        rec_optimal = recall_score(class_targets, preds_optimal, zero_division=0)
        
        results['default_0.5'][disease_name] = {
            'auc': auc,
            'pr_auc': pr_auc,
            'f1': f1_default,
            'precision': prec_default,
            'recall': rec_default
        }
        
        results['optimal'][disease_name] = {
            'auc': auc,
            'pr_auc': pr_auc,
            'f1': f1_optimal,
            'precision': prec_optimal,
            'recall': rec_optimal,
            'threshold': optimal_threshold
        }
        
        # Print comparison
        f1_improvement = ((f1_optimal - f1_default) / max(f1_default, 0.001) * 100) if f1_default > 0 else 0
        prec_change = prec_optimal - prec_default
        rec_change = rec_optimal - rec_default
        
        print(f"\n{disease_name:<25} (Threshold: {optimal_threshold:.2f})")
        print(f"  AUC:              {auc:.4f}")
        print(f"  PR-AUC:           {pr_auc:.4f}")
        print(f"  F1:     {f1_optimal:7.4f}  (default: {f1_default:.4f}, change: {f1_improvement:+.1f}%)")
        print(f"  Precision: {prec_optimal:7.4f}  (default: {prec_default:.4f}, change: {prec_change:+.4f})")
        print(f"  Recall:    {rec_optimal:7.4f}  (default: {rec_default:.4f}, change: {rec_change:+.4f})")
    
    # Compute macro averages
    print("\n" + "=" * 90)
    print("MACRO-AVERAGE METRICS")
    print("=" * 90)
    
    # Default 0.5
    preds_default_all = (all_probs >= 0.5).astype(int)
    f1_macro_default = f1_score(all_targets, preds_default_all, average='macro', zero_division=0)
    prec_macro_default = precision_score(all_targets, preds_default_all, average='macro', zero_division=0)
    rec_macro_default = recall_score(all_targets, preds_default_all, average='macro', zero_division=0)
    
    # Optimal
    preds_optimal_all = np.zeros_like(all_probs)
    for class_idx in range(NUM_CLASSES):
        disease_name = DISEASE_LABELS[class_idx]
        optimal_threshold = optimal_thresholds.get(disease_name, 0.5)
        preds_optimal_all[:, class_idx] = (all_probs[:, class_idx] >= optimal_threshold).astype(int)
    
    f1_macro_optimal = f1_score(all_targets, preds_optimal_all, average='macro', zero_division=0)
    prec_macro_optimal = precision_score(all_targets, preds_optimal_all, average='macro', zero_division=0)
    rec_macro_optimal = recall_score(all_targets, preds_optimal_all, average='macro', zero_division=0)
    
    # AUC macro
    auc_macro = np.mean([roc_auc_score(all_targets[:, i], all_probs[:, i]) if len(np.unique(all_targets[:, i])) > 1 else 0 for i in range(NUM_CLASSES)])
    
    print(f"\n{'Metric':<20} {'Default 0.5':>15} {'Optimal':>15} {'Improvement':>15}")
    print("-" * 65)
    print(f"{'AUC (macro)':<20} {auc_macro:>15.4f} {auc_macro:>15.4f} {'N/A':>15}")
    print(f"{'F1 (macro)':<20} {f1_macro_default:>15.4f} {f1_macro_optimal:>15.4f} {(f1_macro_optimal-f1_macro_default)*100:>14.1f}%")
    print(f"{'Precision (macro)':<20} {prec_macro_default:>15.4f} {prec_macro_optimal:>15.4f} {(prec_macro_optimal-prec_macro_default)*100:>14.1f}%")
    print(f"{'Recall (macro)':<20} {rec_macro_default:>15.4f} {rec_macro_optimal:>15.4f} {(rec_macro_optimal-rec_macro_default)*100:>14.1f}%")
    
    # Save results
    results['macro_averages'] = {
        'auc': auc_macro,
        'f1_default': f1_macro_default,
        'f1_optimal': f1_macro_optimal,
        'precision_default': prec_macro_default,
        'precision_optimal': prec_macro_optimal,
        'recall_default': rec_macro_default,
        'recall_optimal': rec_macro_optimal
    }
    
    results_path = project_root / "experiments" / "test_results_15class_optimized.json"
    print(f"\n\nSaving results to: {results_path}")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Saved")
    
    print("\n" + "=" * 90)
    print("TEST EVALUATION COMPLETE")
    print("=" * 90)


if __name__ == '__main__':
    main()
