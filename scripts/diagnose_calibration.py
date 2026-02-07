"""
Diagnostic Script: Probability Calibration Analysis
=====================================================
Analyzes the probability distributions to understand calibration issues.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from ml.models.student_model import create_student_model
from ml.data.loader import ChestXrayDataset
from ml.data.preprocessing import get_medical_transforms
from config.disease_labels import DISEASE_LABELS


def main():
    print("\n" + "=" * 80)
    print("PROBABILITY CALIBRATION DIAGNOSTIC")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    clahe_cache_dir = project_root / "data" / "clahe_cache"
    val_csv = project_root / "data" / "splits" / "val.csv"
    checkpoint_path = project_root / "ml" / "models" / "checkpoints" / "efficientnet_b0_performer_full_dataset" / "best_checkpoint.pth"
    NUM_CLASSES = 14  # The baseline checkpoint has 14 classes
    
    val_df = pd.read_csv(val_csv)
    if 'Image Index' in val_df.columns:
        val_df = val_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
    
    val_transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    val_dataset = ChestXrayDataset(str(clahe_cache_dir), val_df, transform=val_transform, is_training=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load model
    print(f"Loading model with {NUM_CLASSES} classes...")
    model = create_student_model('efficientnet_b0_performer', num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'student_state_dict' in checkpoint:
        print("Found 'student_state_dict' in checkpoint")
        model.load_state_dict(checkpoint['student_state_dict'], strict=False)
    elif 'model_state_dict' in checkpoint:
        print("Found 'model_state_dict' in checkpoint")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print("Loading checkpoint as direct state dict")
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    print("Model loaded\n")
    
    # Get predictions
    print("Computing predictions...")
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_img, batch_target, _ in tqdm(val_loader, desc="Val"):
            batch_img = batch_img.to(device)
            output = model(batch_img)
            probs = torch.sigmoid(output).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(batch_target.numpy())
    
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    
    print("\n" + "=" * 80)
    print("PROBABILITY DISTRIBUTIONS BY CLASS")
    print("=" * 80)
    
    for class_idx in range(NUM_CLASSES):
        disease_name = DISEASE_LABELS[class_idx]
        probs_positive = all_probs[all_targets[:, class_idx] == 1, class_idx]  # Probs when label=1
        probs_negative = all_probs[all_targets[:, class_idx] == 0, class_idx]  # Probs when label=0
        
        num_pos = len(probs_positive)
        num_neg = len(probs_negative)
        
        if num_pos == 0:
            print(f"\n{disease_name:<25} [NO POSITIVE SAMPLES]")
            continue
        
        print(f"\n{disease_name:<25} (Pos: {num_pos:,}, Neg: {num_neg:,})")
        print(f"  Positive label probabilities:")
        print(f"    Mean:   {probs_positive.mean():.4f}")
        print(f"    Median: {np.median(probs_positive):.4f}")
        print(f"    Min:    {probs_positive.min():.4f}")
        print(f"    Max:    {probs_positive.max():.4f}")
        print(f"    Std:    {probs_positive.std():.4f}")
        
        print(f"  Negative label probabilities:")
        print(f"    Mean:   {probs_negative.mean():.4f}")
        print(f"    Median: {np.median(probs_negative):.4f}")
        print(f"    Min:    {probs_negative.min():.4f}")
        print(f"    Max:    {probs_negative.max():.4f}")
        print(f"    Std:    {probs_negative.std():.4f}")
        
        # Show separation
        overlap = np.sum((probs_positive < 0.5) & (probs_positive > 0)) + np.sum((probs_negative > 0.5) & (probs_negative < 1))
        print(f"  Separation: {(1 - overlap / (num_pos + num_neg)) * 100:.1f}% (higher is better)")
        
        # Show F1 at different thresholds
        print(f"  F1 scores at different thresholds:")
        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            preds = (all_probs[:, class_idx] >= t).astype(int)
            f1 = f1_score(all_targets[:, class_idx], preds, zero_division=0)
            prec = precision_score(all_targets[:, class_idx], preds, zero_division=0)
            rec = recall_score(all_targets[:, class_idx], preds, zero_division=0)
            print(f"    T={t:.1f}: F1={f1:.4f} (P={prec:.4f}, R={rec:.4f})")


if __name__ == '__main__':
    main()
