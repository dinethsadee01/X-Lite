"""
Test Set Evaluation Script
===========================
Evaluates a trained model on the held-out test set.

This script is reusable for all training stages:
  1. Before KD (baseline model)
  2. After KD (student with distillation)
  3. After full tuning (final production model)

Configuration:
- Model checkpoint path (configurable)
- Uses precomputed CLAHE cache (same as training)
- Computes comprehensive metrics (AUC, F1, PR-AUC, Precision, Recall)
- Saves results to JSON for documentation

Note: Test data has been completely unseen during training.
Note: CLAHE must be precomputed first: python scripts/precompute_clahe.py

Usage Examples:
  # Baseline model (before KD)
  python scripts/test_model.py

  # After editing CHECKPOINT_PATH to point to KD model
  python scripts/test_model.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import json
from datetime import datetime

from ml.models.student_model import create_student_model, MODEL_CONFIGS
from ml.data.loader import ChestXrayDataset
from ml.data.preprocessing import get_medical_transforms
from scripts.training_utils import evaluate_final_metrics

# ============================================================
# CONFIGURATION - Edit these values to test different models
# ============================================================

# Model architecture name (must match checkpoint)
MODEL_NAME = 'efficientnet_b0_performer'

# Path to checkpoint file (best_checkpoint.pth or last_checkpoint.pth)
CHECKPOINT_PATH = 'ml/models/new checkpoints/efficientnet_b0_performer_full_dataset_15class_patientwise_lol/best_checkpoint.pth'

# Stage identifier (for results file naming)
STAGE = 'baseline'  # Options: 'baseline', 'after_kd', 'final_tuned'

# Batch size for inference
BATCH_SIZE = 32

# ============================================================

def load_model_checkpoint(checkpoint_path: Path, model_name: str, device: torch.device):
    """
    Load model checkpoint (works for both baseline and KD models)
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_name: Model architecture name
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    print(f"\nLoading checkpoint...")
    print(f"  Path: {checkpoint_path}")
    print(f"  Model: {model_name}")
    
    # Create model (15 classes: 14 diseases + No_Finding)
    from config import NUM_CLASSES
    model = create_student_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Try different state dict keys (baseline vs KD checkpoints)
    if 'student_state_dict' in checkpoint:
        # KD checkpoint format
        model.load_state_dict(checkpoint['student_state_dict'])
        print(f"  ✓ Loaded KD checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    elif 'model_state_dict' in checkpoint:
        # Baseline checkpoint format
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded baseline checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        print(f"  ✓ Loaded checkpoint (direct state dict)")
    
    model.to(device)
    model.eval()
    
    return model

def main():
    """Evaluate model on test set"""
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Stage: {STAGE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Paths
    clahe_cache_dir = project_root / "data" / "clahe_cache"
    test_csv = project_root / "data" / "splits" / "test_df.csv"
    checkpoint_path = project_root / CHECKPOINT_PATH

    # Validate paths
    if not clahe_cache_dir.exists():
        print(f"\n❌ CLAHE cache not found at: {clahe_cache_dir}")
        print("   Run: python scripts/precompute_clahe.py")
        return

    if not test_csv.exists():
        print(f"\n❌ Test split not found at: {test_csv}")
        return

    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found at: {checkpoint_path}")
        return

    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(test_csv)

    if 'Image Index' in test_df.columns:
        test_df = test_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})

    print(f"Test samples: {len(test_df):,}")
    print("⚠️  This data has been completely unseen during training!")

    # Create test dataset (use precomputed CLAHE cache, no augmentation)
    val_transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    
    test_dataset = ChestXrayDataset(
        str(clahe_cache_dir), test_df, transform=val_transform, is_training=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True  # Fast with precomputed CLAHE
    )

    print(f"Test batches: {len(test_loader)}")

    # Load model
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    
    model = load_model_checkpoint(checkpoint_path, MODEL_NAME, device)
    
    num_params = model.get_num_params()
    model_size_mb = model.get_model_size_mb()
    
    print(f"  Parameters: {num_params:,}")
    print(f"  Model Size: {model_size_mb:.2f} MB")
    print(f"  Backbone: {MODEL_CONFIGS[MODEL_NAME]['backbone']}")
    print(f"  Attention: {MODEL_CONFIGS[MODEL_NAME]['attention']}")

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)
    print("Computing comprehensive metrics (AUC, F1, PR-AUC, Precision, Recall)...")
    print("This may take a few minutes...\n")
    
    test_metrics = evaluate_final_metrics(model, test_loader, device)

    # Extract per-disease metrics from individual keys
    from config.disease_labels import DISEASE_LABELS
    per_disease_auc = {label: test_metrics.get(f'AUC_{label}', 0.0) for label in DISEASE_LABELS}
    per_disease_f1 = {label: test_metrics.get(f'F1_{label}', 0.0) for label in DISEASE_LABELS}

    # Collect results
    results = {
        'stage': STAGE,
        'model_name': MODEL_NAME,
        'checkpoint_path': str(checkpoint_path),
        'backbone': MODEL_CONFIGS[MODEL_NAME]['backbone'],
        'attention': MODEL_CONFIGS[MODEL_NAME]['attention'],
        'num_parameters': num_params,
        'model_size_mb': model_size_mb,
        'num_test_samples': len(test_df),
        'test_auc_macro': test_metrics['AUC_macro'],
        'test_f1_macro': test_metrics['F1_macro'],
        'test_pr_auc_macro': test_metrics['PR_AUC_macro'],
        'test_precision_macro': test_metrics['Precision_macro'],
        'test_recall_macro': test_metrics['Recall_macro'],
        'per_disease_auc': per_disease_auc,
        'per_disease_f1': per_disease_f1,
        'timestamp': datetime.now().isoformat()
    }

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"✅ TEST EVALUATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"   Stage: {STAGE}")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Test Set Size: {len(test_df):,} images")
    print(f"")
    print(f"   Test AUC (macro): {results['test_auc_macro']:.4f}")
    print(f"   Test F1 (macro): {results['test_f1_macro']:.4f}")
    print(f"   Test PR-AUC (macro): {results['test_pr_auc_macro']:.4f}")
    print(f"   Test Precision (macro): {results['test_precision_macro']:.4f}")
    print(f"   Test Recall (macro): {results['test_recall_macro']:.4f}")

    # Per-disease breakdown
    print(f"\n   Per-Disease AUC:")
    for disease, auc in results['per_disease_auc'].items():
        print(f"     {disease:<30} {auc:.4f}")

    # Save results
    results_dir = project_root / "experiments"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"test_results_{STAGE}_lol.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n   Results saved: {results_file}")
    print(f"{'=' * 70}\n")

    # Performance context
    print("📊 Performance Context:")
    print(f"   This establishes the '{STAGE}' benchmark on unseen test data.")
    if STAGE == 'baseline':
        print(f"   Next: Train with Knowledge Distillation (Phase 2)")
        print(f"   Goal: Match or exceed {results['test_auc_macro']:.4f} AUC after KD")
    elif STAGE == 'after_kd':
        print(f"   Compare with baseline test results to see KD improvement")
    elif STAGE == 'final_tuned':
        print(f"   This is the final production model performance")
    
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
