import random
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image, ImageFilter
from PIL.Image import Resampling, Transform, Transpose
from torch.utils.data import Dataset

AUGMENTATION_OPS = [
    "rotate_90_right",
    "rotate_90_left",
    "rotate_45_horizontal",
    "vertical_flip",
    "rotate_45_vertical",
    "translate_28_13",
    "horizontal_flip",
]


def _normalize_label_key(label: str) -> str:
    return label.strip().lower().replace("_", " ")


def _resolve_label_columns(df: pd.DataFrame, requested_labels: List[str]) -> Tuple[List[str], List[str]]:
    normalized_to_actual = {_normalize_label_key(c): c for c in df.columns}
    resolved = []
    missing = []
    for label in requested_labels:
        key = _normalize_label_key(label)
        actual = normalized_to_actual.get(key)
        if actual is None:
            missing.append(label)
        else:
            resolved.append(actual)
    return resolved, missing


def apply_augmentation_op(image: Image.Image, op_name: str) -> Image.Image:
    if op_name == "rotate_90_right":
        return image.rotate(-90, resample=Resampling.BILINEAR)
    if op_name == "rotate_90_left":
        return image.rotate(90, resample=Resampling.BILINEAR)
    if op_name == "rotate_45_horizontal":
        return image.rotate(45, resample=Resampling.BILINEAR).transpose(Transpose.FLIP_LEFT_RIGHT)
    if op_name == "vertical_flip":
        return image.transpose(Transpose.FLIP_TOP_BOTTOM)
    if op_name == "rotate_45_vertical":
        return image.rotate(45, resample=Resampling.BILINEAR).transpose(Transpose.FLIP_TOP_BOTTOM)
    if op_name == "translate_28_13":
        return image.transform(
            image.size,
            Transform.AFFINE,
            (1, 0, 28, 0, 1, 13),
            resample=Resampling.BILINEAR,
            fillcolor=0,
        )
    if op_name == "horizontal_flip":
        return image.transpose(Transpose.FLIP_LEFT_RIGHT)
    return image


def prepare_balanced_training_dataframe(
    train_df: pd.DataFrame,
    image_col: str,
    undersample_targets: Dict[str, int],
    oversample_targets: Dict[str, int],
    random_state: int = 42,
    oversample_improvement_ratio: float = 0.35,
    min_improvement_samples: int = 100,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rng = random.Random(random_state)

    balanced_df = train_df.copy().reset_index(drop=True)
    if "__aug_op" not in balanced_df.columns:
        balanced_df["__aug_op"] = "none"

    # Resolve requested labels against dataframe columns
    undersample_labels, undersample_missing = _resolve_label_columns(
        balanced_df, list(undersample_targets.keys())
    )
    undersample_mapping = {
        actual: undersample_targets[req]
        for req, actual in zip(undersample_targets.keys(), undersample_labels)
    }

    oversample_labels, oversample_missing = _resolve_label_columns(
        balanced_df, list(oversample_targets.keys())
    )
    oversample_mapping = {
        actual: oversample_targets[req]
        for req, actual in zip(oversample_targets.keys(), oversample_labels)
    }

    generated_rows = []
    dropped_for_cap = 0

    # Build a combined hard-cap map.
    hard_caps = {}
    for col, cap in undersample_mapping.items():
        hard_caps[col] = cap
    for col, cap in oversample_mapping.items():
        if col in hard_caps:
            hard_caps[col] = min(hard_caps[col], cap)
        else:
            hard_caps[col] = cap

    # 1) Cap first: reduce any class above its cap before augmentation starts.
    for label_col, cap in hard_caps.items():
        positive_idx = balanced_df.index[balanced_df[label_col] == 1].tolist()
        if len(positive_idx) <= cap:
            continue
        keep_positive = set(rng.sample(positive_idx, cap))
        drop_positive = [idx for idx in positive_idx if idx not in keep_positive]
        dropped_for_cap += len(drop_positive)
        balanced_df = balanced_df.drop(index=drop_positive).reset_index(drop=True)

    # 2) Augment only minority classes. Improve counts without forcing exact 5000.
    working_df = balanced_df.copy()
    for label_col, target_count in oversample_mapping.items():
        positive_rows = working_df[working_df[label_col] == 1]
        current_count = len(positive_rows)
        if current_count == 0 or current_count >= target_count:
            continue

        gap = target_count - current_count
        improvement_count = int(gap * oversample_improvement_ratio)
        improvement_count = max(min_improvement_samples, improvement_count)
        needed = min(gap, improvement_count)
        if needed <= 0:
            continue

        sampled_rows = positive_rows.sample(n=needed, replace=True, random_state=random_state)

        for _, row in sampled_rows.iterrows():
            new_row = row.copy()
            new_row["__aug_op"] = rng.choice(AUGMENTATION_OPS)
            generated_rows.append(new_row)

        generated_batch = sampled_rows.copy()
        generated_batch["__aug_op"] = [rng.choice(AUGMENTATION_OPS) for _ in range(len(generated_batch))]
        working_df = pd.concat([working_df, generated_batch], ignore_index=True)

    balanced_df = working_df

    # 3) Safety cap pass: after augmentation, keep every tracked class <= cap.
    for label_col, cap in hard_caps.items():
        current_count = int((balanced_df[label_col] == 1).sum())
        if current_count <= cap:
            continue

        excess = current_count - cap
        aug_positive_idx = balanced_df.index[
            (balanced_df[label_col] == 1) & (balanced_df["__aug_op"] != "none")
        ].tolist()
        drop_idx = []

        if aug_positive_idx:
            take_from_aug = min(excess, len(aug_positive_idx))
            drop_idx.extend(rng.sample(aug_positive_idx, take_from_aug))

        remaining_excess = excess - len(drop_idx)
        if remaining_excess > 0:
            remaining_positive_idx = [
                idx for idx in balanced_df.index[balanced_df[label_col] == 1].tolist() if idx not in drop_idx
            ]
            drop_idx.extend(rng.sample(remaining_positive_idx, remaining_excess))

        dropped_for_cap += len(drop_idx)
        balanced_df = balanced_df.drop(index=drop_idx).reset_index(drop=True)

    class_counts_after = {}
    for col in hard_caps.keys():
        class_counts_after[col] = int((balanced_df[col] == 1).sum())

    generated_samples_after_cap = int((balanced_df["__aug_op"] != "none").sum())

    stats = {
        "original_size": int(len(train_df)),
        "balanced_size": int(len(balanced_df)),
        "generated_samples": generated_samples_after_cap,
        "dropped_for_cap": int(dropped_for_cap),
        "oversample_improvement_ratio": float(oversample_improvement_ratio),
        "min_improvement_samples": int(min_improvement_samples),
        "undersample_missing_labels": undersample_missing,
        "oversample_missing_labels": oversample_missing,
        "class_counts_after": class_counts_after,
    }
    return balanced_df, stats


class AugmentedXRayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        image_col: str,
        label_cols: List[str],
        transform=None,
        aug_col: str = "__aug_op",
        apply_gaussian_blur: bool = False,
        gaussian_radius: float = 1.0,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_col = image_col
        self.label_cols = label_cols
        self.transform = transform
        self.aug_col = aug_col
        self.apply_gaussian_blur = apply_gaussian_blur
        self.gaussian_radius = gaussian_radius

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.image_dir}/{row[self.image_col]}"
        image = Image.open(img_path).convert("RGB")

        if self.aug_col in row.index:
            aug_op = row[self.aug_col]
            if isinstance(aug_op, str) and aug_op != "none":
                image = apply_augmentation_op(image, aug_op)

        if self.apply_gaussian_blur:
            image = image.filter(ImageFilter.GaussianBlur(radius=self.gaussian_radius))

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[self.label_cols].values.astype("float32"), dtype=torch.float32)
        return image, labels
