"""
CLAHE Pre-computation Script
=============================
Pre-computes CLAHE-enhanced images and caches them to disk.

Purpose:
- Avoid real-time CLAHE computation during training (expensive on CPU)
- Enable multi-worker data loading (num_workers > 0)
- Speedup: ~3x faster data loading during training

Usage:
    python scripts/precompute_clahe.py

Output:
    data/clahe_cache/ - Pre-computed CLAHE images matching directory structure
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from ml.data.preprocessing import CLAHEPreprocessor


def precompute_clahe_cache(
    data_dir: Path,
    splits_dir: Path,
    output_dir: Path,
    clahe_processor: CLAHEPreprocessor
):
    """
    Pre-compute CLAHE for all training and validation images
    
    Args:
        data_dir: Directory containing original images
        splits_dir: Directory containing train.csv, val.csv
        output_dir: Directory to save pre-computed CLAHE images
        clahe_processor: CLAHE processor instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load splits
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    
    print(f"\n{'='*70}")
    print(f"CLAHE PRE-COMPUTATION")
    print(f"{'='*70}")
    print(f"Images to process: {len(all_df):,}")
    print(f"Output directory: {output_dir}")
    print(f"\nProcessing images...\n")
    
    failed_count = 0
    success_count = 0
    
    # Get Image Index column name
    img_col = 'Image Index' if 'Image Index' in train_df.columns else 'image_id'
    
    for idx, row in tqdm(all_df.iterrows(), total=len(all_df), desc="CLAHE"):
        try:
            img_path = data_dir / row[img_col]
            
            if not img_path.exists():
                failed_count += 1
                continue
            
            # Read grayscale image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                failed_count += 1
                continue
            
            # Apply CLAHE
            img_clahe = clahe_processor.apply(img)
            
            # Save to cache with same directory structure
            cache_path = output_dir / row[img_col]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(cache_path), img_clahe)
            success_count += 1
            
        except Exception as e:
            failed_count += 1
            if idx < 5:  # Print first few errors
                print(f"  Error processing {row[img_col]}: {str(e)}")
    
    print(f"\n{'='*70}")
    print(f"CLAHE PRE-COMPUTATION COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successfully processed: {success_count:,} images")
    print(f"✗ Failed: {failed_count:,} images")
    print(f"\nCache location: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Update ml/data/loader.py to use CLAHE cache directory")
    print(f"2. Run training with num_workers > 0 for faster data loading")
    print(f"{'='*70}\n")


def main():
    # Paths
    data_dir = project_root / "data" / "raw" / "images"
    splits_dir = project_root / "data" / "splits"
    cache_dir = project_root / "data" / "clahe_cache"
    
    # Validate inputs
    if not data_dir.exists():
        print(f"✗ Error: Data directory not found: {data_dir}")
        return
    
    if not splits_dir.exists():
        print(f"✗ Error: Splits directory not found: {splits_dir}")
        return
    
    # Create CLAHE processor
    clahe_processor = CLAHEPreprocessor(clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Pre-compute CLAHE
    precompute_clahe_cache(
        data_dir=data_dir,
        splits_dir=splits_dir,
        output_dir=cache_dir,
        clahe_processor=clahe_processor
    )


if __name__ == "__main__":
    main()
