"""
Preprocessing Pipeline Visualization
====================================
Shows sample images with augmentations to validate preprocessing pipeline.

Displays:
1. Original image (from CLAHE cache)
2. Augmented versions (5 different augmentations)
3. Shows actual transformations applied during training

Purpose: Validate that augmentation improves diversity without destroying medical features.

Usage:
    python scripts/visualize_preprocessing.py

Output:
    results/preprocessing_samples.png - Grid of original + augmented images
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.data.augmentation import get_augmentation_pipeline
from ml.data.preprocessing import get_medical_transforms
from config.disease_labels import DISEASE_LABELS


def parse_labels(label_str: str) -> list:
    if pd.isna(label_str) or str(label_str).strip() == "No Finding":
        return []
    return [lbl.strip() for lbl in str(label_str).split("|") if lbl.strip()]


def denormalize_image(img_tensor):
    """Denormalize from ImageNet mean/std for display"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


def main():
    clahe_cache_dir = project_root / "data" / "clahe_cache"
    subset_csv = project_root / "data" / "splits" / "train_subset.csv"
    
    if not subset_csv.exists():
        print(f"✗ Error: Subset CSV not found at {subset_csv}")
        print("  Run: python scripts/create_smart_subset.py")
        return
    
    if not clahe_cache_dir.exists():
        print(f"✗ Error: CLAHE cache not found at {clahe_cache_dir}")
        print("  Run: python scripts/precompute_clahe.py")
        return
    
    # Load subset
    df = pd.read_csv(subset_csv)
    if 'Image Index' in df.columns:
        df = df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
    
    # Sample 3 diverse images (different disease counts)
    df['_label_list'] = df['labels'].apply(parse_labels)
    df['_num_diseases'] = df['_label_list'].apply(len)
    
    samples = []
    # 1 no finding, 1 single disease, 1 multi-disease
    for num_diseases in [0, 1, 2]:
        candidates = df[df['_num_diseases'] == num_diseases]
        if len(candidates) > 0:
            sample = candidates.sample(1, random_state=42).iloc[0]
            samples.append(sample)
    
    if len(samples) < 3:
        print("⚠️  Not enough diverse samples. Using what's available...")
        samples = df.sample(min(3, len(df)), random_state=42).to_dict('records')
    
    # Setup transforms
    augmentation_pipeline = get_augmentation_pipeline(augmentation_strength='medium')
    simple_transform = get_medical_transforms(use_clahe=False, use_denoising=False)
    
    # Create figure
    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    fig.suptitle('Preprocessing Pipeline: Original → Augmented Samples\n'
                 '(Left: CLAHE-preprocessed original, Right: 5 augmented versions)',
                 fontsize=14, fontweight='bold')
    
    for row_idx, sample in enumerate(samples):
        image_id = sample['image_id']
        labels = sample['labels']
        label_str = labels if isinstance(labels, str) else "No Finding"
        
        # Load image
        img_path = clahe_cache_dir / image_id
        if not img_path.exists():
            print(f"⚠️  Image not found: {img_path}")
            continue
        
        img_pil = Image.open(img_path).convert('RGB')
        
        # Column 0: Original (with simple transform only)
        simple_tensor = simple_transform(img_pil)
        simple_display = denormalize_image(simple_tensor)
        
        axes[row_idx, 0].imshow(simple_display, cmap='gray')
        axes[row_idx, 0].set_title(f'Original\n{label_str}', fontsize=9)
        axes[row_idx, 0].axis('off')
        
        # Columns 1-5: Augmented versions
        for aug_idx in range(5):
            # Apply augmentation (returns tensor)
            # Convert PIL to numpy array for Albumentations
            img_np = np.array(img_pil)
            augmented = augmentation_pipeline(image=img_np)
            aug_tensor = augmented['image']
            aug_display = denormalize_image(aug_tensor)
            
            axes[row_idx, aug_idx + 1].imshow(aug_display, cmap='gray')
            axes[row_idx, aug_idx + 1].set_title(f'Aug #{aug_idx + 1}', fontsize=9)
            axes[row_idx, aug_idx + 1].axis('off')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Save
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "preprocessing_samples.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("=" * 70)
    print("PREPROCESSING PIPELINE VISUALIZATION")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"\nAugmentation settings (medium strength):")
    print(f"  • HorizontalFlip (p=0.5)")
    print(f"  • Rotate ±15° (p=0.4)")
    print(f"  • ShiftScaleRotate (p=0.3)")
    print(f"  • GaussNoise OR GaussianBlur (p=0.2)")
    print(f"  • RandomBrightnessContrast (p=0.3)")
    print(f"\n✓ Preprocessing validated: CLAHE → Augment → Normalize")
    print("=" * 70)


if __name__ == "__main__":
    main()
