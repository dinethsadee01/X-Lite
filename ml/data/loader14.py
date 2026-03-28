"""
Dataset loader for 14-class setup (without No_Finding).
Images labeled as "No Finding" become all-zero vectors.
"""

import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

from config.disease_labels14 import DISEASE_LABELS14, NUM_CLASSES14


class ChestXrayDataset14(Dataset):
    """
    PyTorch Dataset for ChestX-ray14 14-class training.

    Args:
        data_dir: Path to image directory
        labels_df: DataFrame with columns ['image_id', 'labels']
        transform: Image transforms
        is_training: Flag for compatibility with existing calls
    """

    def __init__(self, data_dir: str, labels_df: pd.DataFrame, transform=None, is_training: bool = True):
        self.data_dir = Path(data_dir)
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform
        self.is_training = is_training

        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        print(f"Loaded {len(self.labels_df)} images")

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int):
        row = self.labels_df.iloc[idx]
        image_id = row['image_id']

        image_path = self.data_dir / image_id
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        if self.transform:
            if hasattr(self.transform, '__module__') and 'albumentations' in self.transform.__module__:
                import numpy as np

                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image = transformed['image']
            else:
                image = self.transform(image)

        label_vector = self._parse_labels14(row['labels'])
        return image, label_vector, image_id

    def _parse_labels14(self, label_str: str) -> torch.Tensor:
        label_vector = torch.zeros(NUM_CLASSES14, dtype=torch.float32)

        # No Finding is represented as all zeros in 14-class setup.
        if pd.isna(label_str) or label_str == 'No Finding':
            return label_vector

        labels = label_str.split('|')
        for label in labels:
            label = label.strip()
            if label in DISEASE_LABELS14:
                idx = DISEASE_LABELS14.index(label)
                label_vector[idx] = 1.0

        return label_vector
