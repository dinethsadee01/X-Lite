"""
CLAHE Visualization Script
===========================
Creates before/after comparison images showing the effect of CLAHE preprocessing.

Purpose:
- Document CLAHE enhancement visually
- Show local contrast improvement in chest X-rays
- Demonstrate preprocessing quality for technical documentation

Usage:
    python scripts/visualize_clahe.py

Output:
    results/clahe_comparison.png - Grid of before/after examples
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ml.data.preprocessing import CLAHEPreprocessor
import random


def create_clahe_comparison(
    data_dir: Path,
    splits_dir: Path,
    output_path: Path,
    num_examples: int = 3
):
    """
    Create before/after CLAHE comparison visualization
    
    Args:
        data_dir: Directory containing original images
        splits_dir: Directory containing train.csv
        output_path: Path to save comparison image
        num_examples: Number of example images to show
    """
    # Load training data
    train_df = pd.read_csv(splits_dir / "train.csv")
    
    # Get Image Index column name
    img_col = 'Image Index' if 'Image Index' in train_df.columns else 'image_id'
    
    # Sample random images (prefer ones with findings for better visual impact)
    if 'Finding Labels' in train_df.columns or 'labels' in train_df.columns:
        label_col = 'Finding Labels' if 'Finding Labels' in train_df.columns else 'labels'
        # Get images with pathologies (not "No Finding")
        with_findings = train_df[train_df[label_col] != 'No Finding']
        if len(with_findings) >= num_examples:
            samples = with_findings.sample(n=num_examples, random_state=42)
        else:
            samples = train_df.sample(n=num_examples, random_state=42)
    else:
        samples = train_df.sample(n=num_examples, random_state=42)
    
    # Create CLAHE processor
    clahe_processor = CLAHEPreprocessor(clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Create figure
    fig, axes = plt.subplots(num_examples, 2, figsize=(12, 4 * num_examples))
    
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    print(f"Creating CLAHE comparison with {num_examples} examples...")
    
    for idx, (ax_row, (_, sample)) in enumerate(zip(axes, samples.iterrows())):
        img_path = data_dir / sample[img_col]
        
        # Load original image
        img_original = Image.open(img_path).convert('RGB')
        
        # Apply CLAHE
        img_clahe = clahe_processor(img_original)
        
        # Display original
        ax_row[0].imshow(img_original, cmap='gray')
        ax_row[0].set_title(f'Original: {sample[img_col]}', fontsize=10, fontweight='bold')
        ax_row[0].axis('off')
        
        # Display CLAHE-enhanced
        ax_row[1].imshow(img_clahe, cmap='gray')
        ax_row[1].set_title(f'CLAHE Enhanced (clip={clahe_processor.clip_limit})', 
                           fontsize=10, fontweight='bold')
        ax_row[1].axis('off')
        
        # Add label info if available
        if 'Finding Labels' in train_df.columns or 'labels' in train_df.columns:
            label_col = 'Finding Labels' if 'Finding Labels' in train_df.columns else 'labels'
            labels = sample[label_col]
            # Truncate long labels
            if len(str(labels)) > 60:
                labels = str(labels)[:60] + '...'
            fig.text(0.5, 1 - (idx + 0.85) / num_examples, f'Labels: {labels}',
                    ha='center', fontsize=9, style='italic', color='gray')
        
        print(f"  ✓ Processed {sample[img_col]}")
    
    # Overall title
    fig.suptitle('CLAHE Preprocessing Effect on Chest X-Rays\n'
                 'Contrast Limited Adaptive Histogram Equalization enhances local contrast',
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Add footer with technical details
    fig.text(0.5, 0.01, 
             'CLAHE Parameters: Clip Limit=2.0, Tile Grid=8×8 | '
             'Effect: Enhances subtle pathology features while limiting noise amplification',
             ha='center', fontsize=8, style='italic', color='dimgray')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ CLAHE comparison saved to: {output_path}")


def main():
    # Paths
    data_dir = project_root / "data" / "raw" / "images"
    splits_dir = project_root / "data" / "splits"
    results_dir = project_root / "results"
    output_path = results_dir / "clahe_comparison.png"
    
    # Validate inputs
    if not data_dir.exists():
        print(f"✗ Error: Data directory not found: {data_dir}")
        return
    
    if not splits_dir.exists():
        print(f"✗ Error: Splits directory not found: {splits_dir}")
        return
    
    print("=" * 70)
    print("CLAHE VISUALIZATION")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output: {output_path}")
    print()
    
    # Create visualization
    create_clahe_comparison(
        data_dir=data_dir,
        splits_dir=splits_dir,
        output_path=output_path,
        num_examples=3
    )
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nUse this image in documentation to show:")
    print(f"  • CLAHE enhances local contrast in medical images")
    print(f"  • Subtle pathology features become more visible")
    print(f"  • Noise is controlled (clip limit prevents over-enhancement)")
    print(f"\nNext step: Include {output_path.name} in technical documentation")
    print("=" * 70)


if __name__ == "__main__":
    main()
