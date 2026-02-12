"""
Test Set Evaluation Script
==========================
Evaluates the best KD model on the held-out test set.
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc as sklearn_auc, precision_score, recall_score
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.models.student_model import create_student_model, MODEL_CONFIGS
from ml.data.loader import get_balanced_data_loaders
from ml.data.preprocessing import get_medical_transforms
from config.disease_labels import DISEASE_LABELS
from scripts.training_utils import evaluate_final_metrics


class RawTestDataset(Dataset):
    """Dataset for raw test images (no preprocessing/CLAHE)"""
    
    def __init__(self, csv_path: Path, images_dir: Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Normalize column names
        if 'Image Index' in self.df.columns:
            self.df = self.df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
        
        self.image_ids = self.df['image_id'].values
        
        # Build label mappings
        self.disease_labels = DISEASE_LABELS
        self.label_to_idx = {label: idx for idx, label in enumerate(self.disease_labels)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load image
        image_id = self.image_ids[idx]
        image_path = self.images_dir / image_id
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Load labels
        label_str = self.df.iloc[idx]['labels']
        labels = np.zeros(len(self.disease_labels), dtype=np.float32)
        
        if pd.notna(label_str) and label_str != 'No Finding':
            disease_list = label_str.split('|')
            for disease in disease_list:
                disease = disease.strip()
                if disease in self.label_to_idx:
                    labels[self.label_to_idx[disease]] = 1.0
        
        return torch.from_numpy(image) if isinstance(image, np.ndarray) else image, torch.from_numpy(labels)


def load_checkpoint(checkpoint_path: Path, model: torch.nn.Module, device: torch.device):
    """Load model checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'student_state_dict' in checkpoint:
        state_dict = checkpoint['student_state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✓ Checkpoint loaded successfully")
    return model


def evaluate_on_test_set(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    model_name: str
):
    """Evaluate model on test set and return comprehensive metrics"""
    print(f"\n{'='*70}")
    print(f"TEST SET EVALUATION: {model_name}")
    print(f"{'='*70}")
    print("(Using raw test images - no CLAHE preprocessing)")
    
    all_preds = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Convert to probabilities (sigmoid for multi-label)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed batch {batch_idx + 1}")
    
    # Concatenate all batches
    preds = np.concatenate(all_preds, axis=0)  # (N, 14)
    targets = np.concatenate(all_targets, axis=0)  # (N, 14)
    
    print(f"\nTest set size: {preds.shape[0]} images")
    
    # Calculate per-disease metrics
    test_results = {
        'model': model_name,
        'test_set_size': preds.shape[0],
        'per_disease_metrics': {}
    }
    
    auc_scores = []
    f1_scores = []
    pr_auc_scores = []
    precision_scores = []
    recall_scores = []
    
    print(f"\n{'Disease':<30} {'AUC':>8} {'F1':>8} {'PR-AUC':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 75)
    
    for disease_idx, disease_name in enumerate(DISEASE_LABELS):
        disease_preds = preds[:, disease_idx]
        disease_targets = targets[:, disease_idx]
        
        # Calculate metrics
        auc = roc_auc_score(disease_targets, disease_preds)
        auc_scores.append(auc)
        
        # F1, precision, recall (using 0.5 threshold)
        binary_preds = (disease_preds >= 0.5).astype(int)
        f1 = f1_score(disease_targets, binary_preds, zero_division=0)
        f1_scores.append(f1)
        
        precision = precision_score(disease_targets, binary_preds, zero_division=0)
        precision_scores.append(precision)
        
        recall = recall_score(disease_targets, binary_preds, zero_division=0)
        recall_scores.append(recall)
        
        # PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(disease_targets, disease_preds)
        pr_auc = sklearn_auc(recall_curve, precision_curve)
        pr_auc_scores.append(pr_auc)
        
        print(f"{disease_name:<30} {auc:>8.4f} {f1:>8.4f} {pr_auc:>8.4f} {precision:>10.4f} {recall:>8.4f}")
        
        test_results['per_disease_metrics'][disease_name] = {
            'auc': float(auc),
            'f1': float(f1),
            'pr_auc': float(pr_auc),
            'precision': float(precision),
            'recall': float(recall)
        }
    
    # Macro-averaged metrics
    auc_macro = np.mean(auc_scores)
    f1_macro = np.mean(f1_scores)
    pr_auc_macro = np.mean(pr_auc_scores)
    precision_macro = np.mean(precision_scores)
    recall_macro = np.mean(recall_scores)
    
    # Micro-averaged (for reference)
    auc_micro = roc_auc_score(targets.ravel(), preds.ravel())
    f1_micro = f1_score(targets.ravel(), (preds >= 0.5).ravel(), zero_division=0)
    
    print("-" * 75)
    print(f"{'MACRO AVERAGE':<30} {auc_macro:>8.4f} {f1_macro:>8.4f} {pr_auc_macro:>8.4f} {precision_macro:>10.4f} {recall_macro:>8.4f}")
    print(f"{'MICRO AVERAGE (reference)':<30} {auc_micro:>8.4f} {f1_micro:>8.4f}")
    
    test_results['auc_macro'] = float(auc_macro)
    test_results['f1_macro'] = float(f1_macro)
    test_results['pr_auc_macro'] = float(pr_auc_macro)
    test_results['precision_macro'] = float(precision_macro)
    test_results['recall_macro'] = float(recall_macro)
    test_results['auc_micro'] = float(auc_micro)
    test_results['f1_micro'] = float(f1_micro)
    
    return test_results


def main():
    parser = argparse.ArgumentParser(description="Test Set Evaluation")
    parser.add_argument("--student_model", type=str, default="convnext_tiny_mhsa")
    parser.add_argument("--checkpoint_path", type=str, default="")
    args = parser.parse_args()
    
    student_model_name = args.student_model
    if student_model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown student model: {student_model_name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Paths
    raw_images_dir = project_root / "data" / "raw" / "images"
    test_csv = project_root / "data" / "splits" / "test.csv"
    
    if not raw_images_dir.exists():
        print(f"✗ ERROR: Raw images directory not found at {raw_images_dir}")
        return
    
    if not test_csv.exists():
        print(f"✗ ERROR: Test split not found at {test_csv}")
        return
    
    # Load test CSV
    print("Loading test set from raw images...")
    test_df = pd.read_csv(test_csv)
    print(f"✓ Test set size: {len(test_df)} images")
    
    # Data transforms (minimal for raw images)
    val_transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    
    # Create test dataset
    test_dataset = RawTestDataset(test_csv, raw_images_dir, transform=val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"✓ Test loader ready: {len(test_dataset)} images")
    
    # Load model
    print(f"\nCreating student model: {student_model_name}")
    student = create_student_model(student_model_name, num_classes=14, pretrained=False)
    student.to(device)
    
    # Load checkpoint
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
    else:
        checkpoint_path = project_root / "ml" / "models" / "checkpoints" / "kd" / student_model_name / "best_checkpoint.pth"
    
    if not checkpoint_path.exists():
        print(f"✗ ERROR: Checkpoint not found at {checkpoint_path}")
        print(f"  Expected: {checkpoint_path}")
        return
    
    student = load_checkpoint(checkpoint_path, student, device)
    
    # Evaluate on test set
    test_results = evaluate_on_test_set(
        model=student,
        test_loader=test_loader,
        device=device,
        model_name=student_model_name
    )
    
    # Save results
    results_dir = project_root / "experiments"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "test_evaluation_results.json"
    
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✅ Test evaluation results saved to: {results_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {student_model_name}")
    print(f"Test Set AUC (macro): {test_results['auc_macro']:.4f}")
    print(f"Test Set F1 (macro): {test_results['f1_macro']:.4f}")
    print(f"Test Set PR-AUC (macro): {test_results['pr_auc_macro']:.4f}")
    print(f"Test Set Precision (macro): {test_results['precision_macro']:.4f}")
    print(f"Test Set Recall (macro): {test_results['recall_macro']:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
