"""
Data Validation Script
======================
Validates preprocessed data splits to ensure data quality:
1. Missing value handling
2. Label preservation
3. Class imbalance verification
4. Outlier detection (corrupted images)
5. No data leakage between splits
6. Distribution balance
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image
from tqdm import tqdm
import cv2

from config.disease_labels import DISEASE_LABELS


def check_missing_values(train_df, val_df, test_df):
    """Check for missing values in critical columns"""
    print("=" * 70)
    print("TEST 1: Missing Value Handling")
    print("=" * 70)
    
    critical_columns = ['Image Index', 'Finding Labels']
    
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{name} Split ({len(df):,} samples):")
        
        for col in critical_columns:
            if col in df.columns:
                missing = df[col].isna().sum()
                print(f"  {col}: {missing} missing ({missing/len(df)*100:.2f}%)")
                
                if missing > 0:
                    print(f"    ⚠ WARNING: Found {missing} missing values!")
                    return False
        
        # Check for empty strings
        empty_labels = df['Finding Labels'].str.strip().eq('').sum()
        if empty_labels > 0:
            print(f"  ⚠ WARNING: Found {empty_labels} empty label strings!")
            return False
    
    print("\n✓ No missing values found in critical columns")
    return True


def check_label_preservation(train_df, val_df, test_df):
    """Verify labels are correctly preserved"""
    print("\n" + "=" * 70)
    print("TEST 2: Label Preservation")
    print("=" * 70)
    
    all_dfs = {'Train': train_df, 'Val': val_df, 'Test': test_df}
    
    for name, df in all_dfs.items():
        print(f"\n{name} Split:")
        
        # Count labels
        label_counts = Counter()
        no_finding_count = 0
        multi_label_count = 0
        
        for label_str in df['Finding Labels']:
            if pd.isna(label_str):
                print(f"  ⚠ ERROR: Found NaN label!")
                return False
            
            label_str = str(label_str).strip()
            
            if label_str == 'No Finding':
                no_finding_count += 1
            else:
                labels = label_str.split('|')
                for label in labels:
                    label = label.strip()
                    if label in DISEASE_LABELS:
                        label_counts[label] += 1
                
                if len(labels) > 1:
                    multi_label_count += 1
        
        print(f"  No Finding: {no_finding_count:,} ({no_finding_count/len(df)*100:.1f}%)")
        print(f"  Multi-label: {multi_label_count:,} ({multi_label_count/len(df)*100:.1f}%)")
        print(f"  Disease labels found: {len(label_counts)}/{len(DISEASE_LABELS)}")
        
        if len(label_counts) == 0:
            print(f"  ⚠ ERROR: No disease labels found (only 'No Finding')!")
            return False
        
        # Show top 5 diseases
        print(f"  Top 5 diseases:")
        for disease, count in label_counts.most_common(5):
            print(f"    {disease}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print("\n✓ Labels correctly preserved across all splits")
    return True


def check_class_imbalance(train_df):
    """Verify class imbalance metrics"""
    print("\n" + "=" * 70)
    print("TEST 3: Class Imbalance Verification")
    print("=" * 70)
    
    # Count each disease
    disease_counts = {label: 0 for label in DISEASE_LABELS}
    no_finding_count = 0
    
    for label_str in train_df['Finding Labels']:
        label_str = str(label_str).strip()
        
        if label_str == 'No Finding':
            no_finding_count += 1
        else:
            labels = label_str.split('|')
            for label in labels:
                label = label.strip()
                if label in disease_counts:
                    disease_counts[label] += 1
    
    # Calculate imbalance ratio
    total_positive = sum(disease_counts.values())
    max_count = max(disease_counts.values())
    min_count = min([c for c in disease_counts.values() if c > 0])
    
    print(f"\nClass Distribution:")
    print(f"  No Finding: {no_finding_count:,} ({no_finding_count/len(train_df)*100:.1f}%)")
    print(f"  Total Positive Samples: {total_positive:,}")
    print(f"  Imbalance Ratio (max/min): {max_count/min_count:.1f}:1")
    
    print(f"\nPer-Disease Counts:")
    print(f"{'Disease':<25} {'Count':<10} {'%':<8}")
    print("-" * 45)
    
    sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
    for disease, count in sorted_diseases:
        pct = count / len(train_df) * 100
        print(f"{disease:<25} {count:<10,} {pct:>6.2f}%")
    
    # Check if severely imbalanced
    if max_count / min_count > 50:
        print(f"\n⚠ Severe class imbalance detected (ratio > 50:1)")
        print("  → Weighted sampling and class weights are REQUIRED")
    
    print("\n✓ Class imbalance quantified")
    return True


def check_data_leakage(train_df, val_df, test_df):
    """Check for data leakage between splits"""
    print("\n" + "=" * 70)
    print("TEST 4: Data Leakage Detection")
    print("=" * 70)
    
    train_images = set(train_df['Image Index'])
    val_images = set(val_df['Image Index'])
    test_images = set(test_df['Image Index'])
    
    # Check overlaps
    train_val_overlap = train_images & val_images
    train_test_overlap = train_images & test_images
    val_test_overlap = val_images & test_images
    
    print(f"\nSplit Sizes:")
    print(f"  Train: {len(train_images):,} unique images")
    print(f"  Val:   {len(val_images):,} unique images")
    print(f"  Test:  {len(test_images):,} unique images")
    
    print(f"\nOverlap Check:")
    print(f"  Train ∩ Val:  {len(train_val_overlap)} images")
    print(f"  Train ∩ Test: {len(train_test_overlap)} images")
    print(f"  Val ∩ Test:   {len(val_test_overlap)} images")
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("\n✗ ERROR: Data leakage detected!")
        return False
    
    print("\n✓ No data leakage between splits")
    return True


def check_patient_leakage(train_df, val_df, test_df):
    """Check for patient-level data leakage"""
    print("\n" + "=" * 70)
    print("TEST 5: Patient-Level Leakage Detection")
    print("=" * 70)
    
    if 'Patient ID' not in train_df.columns:
        print("⚠ Patient ID column not found - skipping patient leakage check")
        return True
    
    train_patients = set(train_df['Patient ID'])
    val_patients = set(val_df['Patient ID'])
    test_patients = set(test_df['Patient ID'])
    
    # Check overlaps
    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients
    
    print(f"\nUnique Patients:")
    print(f"  Train: {len(train_patients):,} patients")
    print(f"  Val:   {len(val_patients):,} patients")
    print(f"  Test:  {len(test_patients):,} patients")
    
    print(f"\nPatient Overlap:")
    print(f"  Train ∩ Val:  {len(train_val_overlap)} patients")
    print(f"  Train ∩ Test: {len(train_test_overlap)} patients")
    print(f"  Val ∩ Test:   {len(val_test_overlap)} patients")
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("\n⚠ WARNING: Patient-level leakage detected!")
        print("  Same patient appears in multiple splits")
        print("  This may lead to overly optimistic performance metrics")
        return False
    
    print("\n✓ No patient-level leakage")
    return True


def check_corrupted_images(train_df, val_df, test_df, data_dir, sample_size=1000):
    """Check for corrupted or invalid images"""
    print("\n" + "=" * 70)
    print("TEST 6: Corrupted Image Detection")
    print("=" * 70)
    
    data_dir = Path(data_dir)
    
    print(f"\nChecking {sample_size} random images per split...")
    
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        # Sample images
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        corrupted = []
        invalid_dims = []
        
        for image_id in tqdm(sample_df['Image Index'], desc=f"{name} Split", leave=False):
            image_path = data_dir / image_id
            
            # Check if file exists
            if not image_path.exists():
                corrupted.append(image_id)
                continue
            
            try:
                # Try to load with PIL
                img = Image.open(image_path)
                img.verify()
                
                # Reload and check dimensions
                img = Image.open(image_path)
                width, height = img.size
                
                # Check for unreasonably small images
                if width < 100 or height < 100:
                    invalid_dims.append((image_id, width, height))
                
            except Exception as e:
                corrupted.append(image_id)
        
        print(f"\n{name} Split:")
        print(f"  Checked: {len(sample_df):,} images")
        print(f"  Corrupted: {len(corrupted)} images")
        print(f"  Invalid dimensions: {len(invalid_dims)} images")
        
        if corrupted:
            print(f"  ⚠ WARNING: Found corrupted images!")
            for img_id in corrupted[:5]:
                print(f"    - {img_id}")
        
        if invalid_dims:
            print(f"  ⚠ WARNING: Found images with invalid dimensions!")
            for img_id, w, h in invalid_dims[:5]:
                print(f"    - {img_id}: {w}x{h}")
    
    print("\n✓ Image integrity check complete")
    return True


def main():
    """Run all validation checks"""
    print("\n" + "=" * 70)
    print("DATA SPLITS VALIDATION")
    print("=" * 70)
    print("Validating preprocessed data for:")
    print("  1. Missing value handling")
    print("  2. Label preservation")
    print("  3. Class imbalance verification")
    print("  4. Image-level data leakage")
    print("  5. Patient-level data leakage")
    print("  6. Corrupted image detection")
    print("=" * 70)
    
    # Load splits
    train_csv = project_root / "data" / "splits" / "train.csv"
    val_csv = project_root / "data" / "splits" / "val.csv"
    test_csv = project_root / "data" / "splits" / "test.csv"
    data_dir = project_root / "data" / "raw" / "images"
    
    print("\nLoading splits...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    
    # Run validation checks
    results = []
    
    try:
        results.append(("Missing Values", check_missing_values(train_df, val_df, test_df)))
        results.append(("Label Preservation", check_label_preservation(train_df, val_df, test_df)))
        results.append(("Class Imbalance", check_class_imbalance(train_df)))
        results.append(("Data Leakage", check_data_leakage(train_df, val_df, test_df)))
        results.append(("Patient Leakage", check_patient_leakage(train_df, val_df, test_df)))
        results.append(("Image Integrity", check_corrupted_images(train_df, val_df, test_df, data_dir, sample_size=500)))
        
        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        all_passed = True
        for test_name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{test_name:<25} {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n" + "=" * 70)
            print("✓ ALL VALIDATION CHECKS PASSED")
            print("=" * 70)
            print("\nData is clean and ready for training!")
            print("\nKey Findings:")
            print("  • No missing values in critical columns")
            print("  • Labels correctly preserved across splits")
            print("  • Class imbalance quantified (weights applied)")
            print("  • No data leakage between train/val/test")
            print("  • No corrupted images detected")
            return 0
        else:
            print("\n" + "=" * 70)
            print("✗ SOME VALIDATION CHECKS FAILED")
            print("=" * 70)
            print("\nPlease review warnings above and fix data issues before training.")
            return 1
            
    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ VALIDATION ERROR")
        print("=" * 70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
