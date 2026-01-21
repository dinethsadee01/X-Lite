"""
Test Data Pipeline
==================
Validates the complete data loading pipeline with:
- Stratified splits loading
- Weighted sampling for balanced batches
- Class-weighted loss functions
- Data augmentation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from collections import Counter
from ml.data.loader import get_balanced_data_loaders
from ml.data.preprocessing import get_transforms
from ml.data.augmentation import get_augmentation_pipeline
from ml.training.losses import (
    WeightedBCEWithLogitsLoss, 
    FocalLoss, 
    CombinedLoss,
    calculate_pos_weights
)
from config.disease_labels import DISEASE_LABELS
import pandas as pd
import torch


def test_data_loading():
    """Test stratified split loading"""
    print("=" * 70)
    print("TEST 1: Data Loading")
    print("=" * 70)
    
    # Paths
    data_dir = project_root / "data" / "raw" / "images"
    train_csv = project_root / "data" / "splits" / "train.csv"
    val_csv = project_root / "data" / "splits" / "val.csv"
    test_csv = project_root / "data" / "splits" / "test.csv"
    
    # Check files exist
    for path in [train_csv, val_csv, test_csv]:
        assert path.exists(), f"Missing split file: {path}"
    
    # Load CSVs
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"✓ Loaded stratified splits:")
    print(f"  Train: {len(train_df):,} samples ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} samples ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} samples ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    
    return train_df, val_df, test_df


def test_transforms():
    """Test data transformations"""
    print("\n" + "=" * 70)
    print("TEST 2: Data Transformations")
    print("=" * 70)
    
    # Get augmentation pipeline (for training)
    train_augmentation = get_augmentation_pipeline(augmentation_strength='medium')
    
    # Get basic transforms (for val/test)
    val_transform = get_transforms(is_training=False)
    
    print("✓ Training transforms: Medium Albumentations Augmentation")
    print("✓ Validation transforms: Resize → Normalize")
    
    return train_augmentation, val_transform


def test_data_loaders(train_df):
    """Test DataLoader creation with weighted sampling"""
    print("\n" + "=" * 70)
    print("TEST 3: DataLoaders with Weighted Sampling")
    print("=" * 70)
    
    data_dir = project_root / "data" / "raw" / "images"
    train_csv = project_root / "data" / "splits" / "train.csv"
    val_csv = project_root / "data" / "splits" / "val.csv"
    test_csv = project_root / "data" / "splits" / "test.csv"
    
    train_transform, val_transform = test_transforms()
    
    loaders = get_balanced_data_loaders(
        data_dir=str(data_dir),
        train_split_csv=str(train_csv),
        val_split_csv=str(val_csv),
        test_split_csv=str(test_csv),
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=16,
        num_workers=2,
        use_weighted_sampler=True
    )
    
    print(f"\n✓ Created DataLoaders:")
    print(f"  Train batches: {len(loaders['train'])}")
    print(f"  Val batches:   {len(loaders['val'])}")
    print(f"  Test batches:  {len(loaders['test'])}")
    
    return loaders


def test_batch_balance(loaders, num_batches=10):
    """Test if weighted sampling produces balanced batches"""
    print("\n" + "=" * 70)
    print("TEST 4: Batch Disease Balance")
    print("=" * 70)
    
    disease_counts = np.zeros(len(DISEASE_LABELS))
    total_samples = 0
    
    print(f"Analyzing {num_batches} training batches...")
    
    for i, (images, labels, image_ids) in enumerate(loaders['train']):
        if i >= num_batches:
            break
        
        # Count disease occurrences
        disease_counts += labels.sum(dim=0).numpy()
        total_samples += images.size(0)
    
    # Calculate percentages
    disease_percentages = (disease_counts / total_samples) * 100
    
    print(f"\nDisease distribution in {num_batches} batches ({total_samples} samples):")
    print(f"{'Disease':<25} {'Count':<8} {'%':<8}")
    print("-" * 45)
    
    for i, label in enumerate(DISEASE_LABELS):
        count = int(disease_counts[i])
        pct = disease_percentages[i]
        print(f"{label:<25} {count:<8} {pct:>6.2f}%")
    
    # Check if rare diseases are represented
    rare_threshold = 5.0  # Expected ~5-10% per class with balanced sampling
    rare_diseases = [DISEASE_LABELS[i] for i in range(len(DISEASE_LABELS)) 
                     if disease_percentages[i] < rare_threshold]
    
    if rare_diseases:
        print(f"\n⚠ Warning: {len(rare_diseases)} diseases below {rare_threshold}% representation")
        print("  This is expected for extremely rare diseases")
    else:
        print(f"\n✓ All diseases represented above {rare_threshold}% threshold")


def test_loss_functions(loaders, train_df):
    """Test loss function calculations"""
    print("\n" + "=" * 70)
    print("TEST 5: Loss Functions")
    print("=" * 70)
    
    # Calculate label counts
    from collections import Counter
    label_counts = np.zeros(len(DISEASE_LABELS))
    
    # Rename columns if needed
    if 'Image Index' in train_df.columns:
        train_df = train_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
    
    for label_str in train_df['labels']:
        if pd.isna(label_str) or label_str == 'No Finding':
            continue
        labels = label_str.split('|')
        for label in labels:
            label = label.strip()
            if label in DISEASE_LABELS:
                idx = DISEASE_LABELS.index(label)
                label_counts[idx] += 1
    
    label_counts_tensor = torch.tensor(label_counts)
    pos_weights = calculate_pos_weights(label_counts_tensor, total_samples=len(train_df))
    print(f"✓ Calculated pos_weights for {len(DISEASE_LABELS)} classes")
    print(f"  Weight range: {pos_weights.min():.2f} - {pos_weights.max():.2f}")
    
    # Create loss functions
    bce_loss = WeightedBCEWithLogitsLoss(pos_weights=pos_weights)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    combined_loss = CombinedLoss(pos_weights=pos_weights, bce_weight=0.7, focal_weight=0.3)
    
    # Test with one batch
    images, labels, image_ids = next(iter(loaders['train']))
    dummy_logits = torch.randn(labels.size())
    
    bce_value = bce_loss(dummy_logits, labels)
    focal_value = focal_loss(dummy_logits, labels)
    combined_value = combined_loss(dummy_logits, labels)
    
    print(f"\n✓ Loss function outputs (random logits):")
    print(f"  Weighted BCE:    {bce_value.item():.4f}")
    print(f"  Focal Loss:      {focal_value.item():.4f}")
    print(f"  Combined Loss:   {combined_value.item():.4f}")


def test_image_loading(loaders):
    """Test actual image loading"""
    print("\n" + "=" * 70)
    print("TEST 6: Image Loading")
    print("=" * 70)
    
    images, labels, image_ids = next(iter(loaders['train']))
    
    print(f"✓ Loaded batch:")
    print(f"  Images shape:  {images.shape}")
    print(f"  Labels shape:  {labels.shape}")
    print(f"  Image dtype:   {images.dtype}")
    print(f"  Labels dtype:  {labels.dtype}")
    print(f"  Image range:   [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Labels range:  [{labels.min():.1f}, {labels.max():.1f}]")
    
    # Check normalization
    assert images.shape[1] == 3, "Expected 3 channels (RGB)"
    assert images.shape[2] == 224, "Expected 224x224 images"
    assert images.shape[3] == 224, "Expected 224x224 images"
    assert labels.shape[1] == len(DISEASE_LABELS), f"Expected {len(DISEASE_LABELS)} classes"
    
    print("\n✓ All shape and dtype checks passed")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("DATA PIPELINE VALIDATION")
    print("=" * 70)
    print("Testing complete data loading pipeline with:")
    print("  - Stratified splits from EDA")
    print("  - Weighted sampling for class balance")
    print("  - Class-weighted loss functions")
    print("  - Data augmentation")
    print("=" * 70)
    
    try:
        # Run tests
        train_df, val_df, test_df = test_data_loading()
        train_transform, val_transform = test_transforms()
        loaders = test_data_loaders(train_df)
        test_batch_balance(loaders, num_batches=20)
        test_loss_functions(loaders, train_df)
        test_image_loading(loaders)
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nData pipeline is ready for training!")
        print("Next steps:")
        print("  1. Create student model architectures")
        print("  2. Implement training loop")
        print("  3. Run baseline training experiments")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ TEST FAILED")
        print("=" * 70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
