"""
Dataset Loader for ChestX-ray14
Handles multi-label chest X-ray images with 14 disease classes
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, Optional, Dict, List

from config import DISEASE_LABELS, NUM_CLASSES


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for ChestX-ray14
    
    Args:
        data_dir (str): Path to image directory
        labels_df (pd.DataFrame): DataFrame with columns ['image_id', 'labels']
        transform (callable, optional): Image transformations
        is_training (bool): Whether this is training data (affects augmentation)
    """
    
    def __init__(
        self,
        data_dir: str,
        labels_df: pd.DataFrame,
        transform=None,
        is_training: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform
        self.is_training = is_training
        
        # Verify data exists
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        print(f"Loaded {len(self.labels_df)} images")
    
    def __len__(self) -> int:
        return len(self.labels_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            image (torch.Tensor): Preprocessed image tensor
            label (torch.Tensor): Multi-label binary vector (14 classes)
            image_id (str): Image filename for reference
        """
        row = self.labels_df.iloc[idx]
        image_id = row['image_id']
        
        # Load image
        image_path = self.data_dir / image_id
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")
        
        # Apply transformations
        if self.transform:
            # Check if Albumentations or torchvision transforms
            if hasattr(self.transform, '__module__') and 'albumentations' in self.transform.__module__:
                # Albumentations: needs numpy array with named argument
                import numpy as np
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image = transformed['image']
            else:
                # Torchvision transforms: works with PIL
                image = self.transform(image)
        
        # Parse labels (multi-hot encoding)
        label_vector = self._parse_labels(row['labels'])
        
        return image, label_vector, image_id
    
    def _parse_labels(self, label_str: str) -> torch.Tensor:
        """
        Convert label string to multi-hot encoded vector
        
        Args:
            label_str (str): Pipe-separated labels (e.g., "Atelectasis|Effusion")
        
        Returns:
            torch.Tensor: Binary vector of shape (NUM_CLASSES,)
        """
        label_vector = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        
        if pd.isna(label_str) or label_str == 'No Finding':
            return label_vector  # All zeros for normal case
        
        # Split labels and encode
        labels = label_str.split('|')
        for label in labels:
            label = label.strip()
            if label in DISEASE_LABELS:
                idx = DISEASE_LABELS.index(label)
                label_vector[idx] = 1.0
        
        return label_vector


def load_metadata(csv_path: str) -> pd.DataFrame:
    """
    Load and process ChestX-ray14 metadata CSV
    
    Args:
        csv_path (str): Path to metadata CSV file
    
    Returns:
        pd.DataFrame: Processed metadata with columns ['image_id', 'labels']
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    required_cols = ['Image Index', 'Finding Labels']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    # Rename columns for consistency
    df = df.rename(columns={
        'Image Index': 'image_id',
        'Finding Labels': 'labels'
    })
    
    return df[['image_id', 'labels']]


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets
    
    Args:
        df (pd.DataFrame): Full dataset
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio
        test_ratio (float): Test set ratio
        seed (int): Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Shuffle dataset
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    n = len(df_shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df_shuffled[:train_end]
    val_df = df_shuffled[train_end:val_end]
    test_df = df_shuffled[val_end:]
    
    print(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df


def get_data_loaders(
    data_dir: str,
    metadata_csv: str,
    train_transform,
    val_transform,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets
    
    Args:
        data_dir (str): Path to image directory
        metadata_csv (str): Path to metadata CSV
        train_transform: Transformations for training data
        val_transform: Transformations for validation/test data
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        pin_memory (bool): Pin memory for faster GPU transfer
        train_ratio, val_ratio, test_ratio (float): Split ratios
        seed (int): Random seed
    
    Returns:
        dict: Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Load and split metadata
    df = load_metadata(metadata_csv)
    train_df, val_df, test_df = split_dataset(
        df, train_ratio, val_ratio, test_ratio, seed
    )
    
    # Create datasets
    train_dataset = ChestXrayDataset(
        data_dir, train_df, transform=train_transform, is_training=True
    )
    val_dataset = ChestXrayDataset(
        data_dir, val_df, transform=val_transform, is_training=False
    )
    test_dataset = ChestXrayDataset(
        data_dir, test_df, transform=val_transform, is_training=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def get_class_weights(labels_df: pd.DataFrame) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance
    Uses inverse frequency weighting
    
    Args:
        labels_df (pd.DataFrame): DataFrame with labels
    
    Returns:
        torch.Tensor: Class weights of shape (NUM_CLASSES,)
    """
    label_counts = np.zeros(NUM_CLASSES)
    
    for label_str in labels_df['labels']:
        if pd.isna(label_str) or label_str == 'No Finding':
            continue
        
        labels = label_str.split('|')
        for label in labels:
            label = label.strip()
            if label in DISEASE_LABELS:
                idx = DISEASE_LABELS.index(label)
                label_counts[idx] += 1
    
    # Inverse frequency weighting
    total = len(labels_df)
    weights = total / (label_counts + 1e-6)  # Avoid division by zero
    weights = weights / weights.sum() * NUM_CLASSES  # Normalize
    
    return torch.tensor(weights, dtype=torch.float32)


def get_sample_weights(labels_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate per-sample weights for WeightedRandomSampler
    Samples with rare diseases get higher weights
    
    Args:
        labels_df (pd.DataFrame): DataFrame with labels
    
    Returns:
        np.ndarray: Sample weights of shape (num_samples,)
    """
    # Calculate class frequencies
    label_counts = np.zeros(NUM_CLASSES)
    for label_str in labels_df['labels']:
        if pd.isna(label_str) or label_str == 'No Finding':
            continue
        labels = label_str.split('|')
        for label in labels:
            label = label.strip()
            if label in DISEASE_LABELS:
                idx = DISEASE_LABELS.index(label)
                label_counts[idx] += 1

    # Smoothed inverse frequency per class (alpha=0.5 for moderate balancing)
    alpha = 0.5
    class_weights = 1.0 / np.power(label_counts + 1.0, alpha)
    # Normalize so mean weight ~1.0
    class_weights = class_weights / class_weights.mean()
    no_finding_weight = class_weights.min()

    # Per-sample weight = mean of its class weights (multi-label safe)
    sample_weights = []
    for label_str in labels_df['labels']:
        if pd.isna(label_str) or label_str == 'No Finding':
            sample_weights.append(float(no_finding_weight))
            continue

        labels = [lbl.strip() for lbl in label_str.split('|') if lbl.strip() in DISEASE_LABELS]
        if not labels:
            sample_weights.append(float(no_finding_weight))
            continue

        weights = [class_weights[DISEASE_LABELS.index(lbl)] for lbl in labels]
        sample_weights.append(float(np.mean(weights)))

    return np.array(sample_weights, dtype=np.float64)


def get_balanced_data_loaders(
    data_dir: str,
    train_split_csv: str,
    val_split_csv: str,
    test_split_csv: str,
    train_transform,
    val_transform,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampler: bool = True
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders with stratified splits and optional weighted sampling
    
    Args:
        data_dir (str): Path to image directory
        train_split_csv (str): Path to train split CSV (from EDA)
        val_split_csv (str): Path to validation split CSV
        test_split_csv (str): Path to test split CSV
        train_transform: Transformations for training data
        val_transform: Transformations for validation/test data
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        pin_memory (bool): Pin memory for faster GPU transfer
        use_weighted_sampler (bool): Use WeightedRandomSampler for balanced batches
    
    Returns:
        dict: Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Load stratified splits
    train_df = pd.read_csv(train_split_csv)
    val_df = pd.read_csv(val_split_csv)
    test_df = pd.read_csv(test_split_csv)
    
    print(f"Loaded stratified splits:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    
    # Rename columns if needed (EDA notebook uses 'Image Index', 'Finding Labels')
    if 'Image Index' in train_df.columns:
        train_df = train_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
        val_df = val_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
        test_df = test_df.rename(columns={'Image Index': 'image_id', 'Finding Labels': 'labels'})
    
    # Create datasets
    train_dataset = ChestXrayDataset(
        data_dir, train_df, transform=train_transform, is_training=True
    )
    val_dataset = ChestXrayDataset(
        data_dir, val_df, transform=val_transform, is_training=False
    )
    test_dataset = ChestXrayDataset(
        data_dir, test_df, transform=val_transform, is_training=False
    )
    
    # Create weighted sampler for training if requested
    if use_weighted_sampler:
        print("\n✓ Using WeightedRandomSampler for balanced batches")
        sample_weights = get_sample_weights(train_df)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # Allow oversampling minority classes
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        shuffle = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
