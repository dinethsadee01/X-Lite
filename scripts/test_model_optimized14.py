"""
Test set evaluation with threshold comparison for 14-class setup.

Compares:
  1) Default 0.5 threshold
  2) Class-wise optimal thresholds (if available)

Threshold loading priority:
  - scripts/optimal_thresholds14.json
  - scripts/optimal_thresholds.json (filtered to 14 classes)
  - fallback 0.5 for missing classes

Usage:
  python scripts/test_model_optimized14.py
"""

import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.models.student_model import create_student_model
from ml.data.loader14 import ChestXrayDataset14
from ml.data.preprocessing import get_medical_transforms
from config.disease_labels14 import DISEASE_LABELS14, NUM_CLASSES14


# Configuration
MODEL_NAME = 'efficientnet_b0_performer'
CHECKPOINT_PATH = 'ml/models/new checkpoints fix/efficientnet_b0_performer_full_dataset_14class_v2/best_checkpoint.pth'
THRESHOLDS_14_PATH = 'scripts/optimal_thresholds14_fixed_v2.json'
THRESHOLDS_15_PATH = 'scripts/optimal_thresholds.json'
BATCH_SIZE = 32


def load_thresholds_for_14_classes() -> dict:
    """Load best available thresholds and return mapping for the 14 classes."""
    thresholds14_path = project_root / THRESHOLDS_14_PATH
    thresholds15_path = project_root / THRESHOLDS_15_PATH

    loaded_from = 'default_0.5'
    threshold_payload = {}

    if thresholds14_path.exists():
        with open(thresholds14_path, 'r', encoding='utf-8') as f:
            threshold_payload = json.load(f)
        loaded_from = str(thresholds14_path)
    elif thresholds15_path.exists():
        with open(thresholds15_path, 'r', encoding='utf-8') as f:
            threshold_payload = json.load(f)
        loaded_from = str(thresholds15_path)

    thresholds = {}
    for label in DISEASE_LABELS14:
        value = threshold_payload.get(label, 0.5)
        thresholds[label] = float(value)

    return {
        'thresholds': thresholds,
        'source': loaded_from,
    }


def load_model(checkpoint_path: Path, device: torch.device):
    model = create_student_model(MODEL_NAME, num_classes=NUM_CLASSES14, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'student_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['student_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def main():
    print('\n' + '=' * 90)
    print('TEST SET EVALUATION WITH OPTIMAL THRESHOLDS (14-CLASS)')
    print('=' * 90)
    print(f'Model: {MODEL_NAME}')
    print(f'Classes: {NUM_CLASSES14} (No_Finding removed)')
    print('=' * 90)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    clahe_cache_dir = project_root / 'data' / 'clahe_cache'
    test_csv = project_root / 'data' / 'splits' / 'test_df.csv'
    checkpoint_path = project_root / CHECKPOINT_PATH

    if not clahe_cache_dir.exists():
        raise FileNotFoundError(f'CLAHE cache directory not found: {clahe_cache_dir}')
    if not test_csv.exists():
        raise FileNotFoundError(f'Test CSV not found: {test_csv}')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    print('Loading thresholds...')
    threshold_info = load_thresholds_for_14_classes()
    optimal_thresholds = threshold_info['thresholds']
    print(f"✓ Threshold source: {threshold_info['source']}")
    for disease, threshold in sorted(optimal_thresholds.items(), key=lambda x: x[1]):
        print(f'  {disease:<25} {threshold:.3f}')

    print('\nLoading test data...')
    test_df = pd.read_csv(test_csv)
    if 'Image Index' in test_df.columns:
        test_df = test_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})

    test_transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    test_dataset = ChestXrayDataset14(
        str(clahe_cache_dir), test_df, transform=test_transform, is_training=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    print(f'✓ Test set: {len(test_df):,} images, {len(test_loader)} batches')

    print('\nLoading model...')
    model = load_model(checkpoint_path, device)
    print('✓ Model loaded\n')

    print('=' * 90)
    print('Computing predictions on test set...')
    print('=' * 90)

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch_img, batch_target, _ in tqdm(test_loader, desc='Test14'):
            batch_img = batch_img.to(device)
            output = model(batch_img)
            probs = torch.sigmoid(output).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(batch_target.numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    print(f'\nPredictions shape: {all_probs.shape}')
    print(f'Targets shape: {all_targets.shape}\n')

    print('=' * 90)
    print('METRICS COMPARISON: Default 0.5 vs Optimal Thresholds (14-CLASS)')
    print('=' * 90)

    results = {
        'default_0.5': {},
        'optimal': {},
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'setup': '14_class_without_no_finding',
            'threshold_source': threshold_info['source'],
            'checkpoint': str(checkpoint_path),
        },
    }

    for class_idx in range(NUM_CLASSES14):
        disease_name = DISEASE_LABELS14[class_idx]
        class_probs = all_probs[:, class_idx]
        class_targets = all_targets[:, class_idx]

        preds_default = (class_probs >= 0.5).astype(int)

        optimal_threshold = float(optimal_thresholds.get(disease_name, 0.5))
        preds_optimal = (class_probs >= optimal_threshold).astype(int)

        auc = roc_auc_score(class_targets, class_probs) if len(np.unique(class_targets)) > 1 else 0.0
        pr_auc = average_precision_score(class_targets, class_probs) if class_targets.sum() > 0 else 0.0

        f1_default = f1_score(class_targets, preds_default, zero_division=0)
        prec_default = precision_score(class_targets, preds_default, zero_division=0)
        rec_default = recall_score(class_targets, preds_default, zero_division=0)

        f1_optimal = f1_score(class_targets, preds_optimal, zero_division=0)
        prec_optimal = precision_score(class_targets, preds_optimal, zero_division=0)
        rec_optimal = recall_score(class_targets, preds_optimal, zero_division=0)

        results['default_0.5'][disease_name] = {
            'auc': auc,
            'pr_auc': pr_auc,
            'f1': f1_default,
            'precision': prec_default,
            'recall': rec_default,
        }

        results['optimal'][disease_name] = {
            'auc': auc,
            'pr_auc': pr_auc,
            'f1': f1_optimal,
            'precision': prec_optimal,
            'recall': rec_optimal,
            'threshold': optimal_threshold,
        }

        f1_improvement = ((f1_optimal - f1_default) / max(f1_default, 0.001) * 100.0) if f1_default > 0 else 0.0
        prec_change = prec_optimal - prec_default
        rec_change = rec_optimal - rec_default

        print(f'\n{disease_name:<25} (Threshold: {optimal_threshold:.2f})')
        print(f'  AUC:              {auc:.4f}')
        print(f'  PR-AUC:           {pr_auc:.4f}')
        print(f'  F1:     {f1_optimal:7.4f}  (default: {f1_default:.4f}, change: {f1_improvement:+.1f}%)')
        print(f'  Precision: {prec_optimal:7.4f}  (default: {prec_default:.4f}, change: {prec_change:+.4f})')
        print(f'  Recall:    {rec_optimal:7.4f}  (default: {rec_default:.4f}, change: {rec_change:+.4f})')

    print('\n' + '=' * 90)
    print('MACRO-AVERAGE METRICS (14-CLASS)')
    print('=' * 90)

    preds_default_all = (all_probs >= 0.5).astype(int)
    f1_macro_default = f1_score(all_targets, preds_default_all, average='macro', zero_division=0)
    prec_macro_default = precision_score(all_targets, preds_default_all, average='macro', zero_division=0)
    rec_macro_default = recall_score(all_targets, preds_default_all, average='macro', zero_division=0)

    preds_optimal_all = np.zeros_like(all_probs)
    for class_idx in range(NUM_CLASSES14):
        disease_name = DISEASE_LABELS14[class_idx]
        threshold = float(optimal_thresholds.get(disease_name, 0.5))
        preds_optimal_all[:, class_idx] = (all_probs[:, class_idx] >= threshold).astype(int)

    f1_macro_optimal = f1_score(all_targets, preds_optimal_all, average='macro', zero_division=0)
    prec_macro_optimal = precision_score(all_targets, preds_optimal_all, average='macro', zero_division=0)
    rec_macro_optimal = recall_score(all_targets, preds_optimal_all, average='macro', zero_division=0)

    auc_scores = np.array(
        [
            roc_auc_score(all_targets[:, i], all_probs[:, i])
            if len(np.unique(all_targets[:, i])) > 1
            else 0.0
            for i in range(NUM_CLASSES14)
        ],
        dtype=np.float64,
    )
    auc_macro = float(np.mean(auc_scores))

    print(f"\n{'Metric':<20} {'Default 0.5':>15} {'Optimal':>15} {'Improvement':>15}")
    print('-' * 65)
    print(f"{'AUC (macro)':<20} {auc_macro:>15.4f} {auc_macro:>15.4f} {'N/A':>15}")
    print(f"{'F1 (macro)':<20} {f1_macro_default:>15.4f} {f1_macro_optimal:>15.4f} {(f1_macro_optimal-f1_macro_default)*100:>14.1f}%")
    print(f"{'Precision (macro)':<20} {prec_macro_default:>15.4f} {prec_macro_optimal:>15.4f} {(prec_macro_optimal-prec_macro_default)*100:>14.1f}%")
    print(f"{'Recall (macro)':<20} {rec_macro_default:>15.4f} {rec_macro_optimal:>15.4f} {(rec_macro_optimal-rec_macro_default)*100:>14.1f}%")

    results['macro_averages'] = {
        'auc': float(auc_macro),
        'f1_default': float(f1_macro_default),
        'f1_optimal': float(f1_macro_optimal),
        'precision_default': float(prec_macro_default),
        'precision_optimal': float(prec_macro_optimal),
        'recall_default': float(rec_macro_default),
        'recall_optimal': float(rec_macro_optimal),
    }

    results_path = project_root / 'experiments' / 'test_results_14class_v2.json'
    print(f'\n\nSaving results to: {results_path}')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('✓ Saved')

    print('\n' + '=' * 90)
    print('TEST EVALUATION COMPLETE (14-CLASS)')
    print('=' * 90)


if __name__ == '__main__':
    main()
