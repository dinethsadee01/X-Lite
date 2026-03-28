"""
Trainer wrapper for 14-class setup.
Uses existing trainer logic but swaps metric label names to 14-class labels.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.disease_labels14 import DISEASE_LABELS14
from ml.training.metrics import MetricsTracker
from ml.training.trainer import ModelTrainer


class ModelTrainer14(ModelTrainer):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: Path,
        num_classes: int = 14,
        use_amp: bool = True,
        gradient_clip_val: float = 1.0,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            checkpoint_dir=checkpoint_dir,
            num_classes=num_classes,
            use_amp=use_amp,
            gradient_clip_val=gradient_clip_val,
        )

        self.train_metrics = MetricsTracker(num_classes, DISEASE_LABELS14)
        self.val_metrics = MetricsTracker(num_classes, DISEASE_LABELS14)


def create_trainer14(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    checkpoint_dir: Path = Path('checkpoints'),
    device: Optional[torch.device] = None,
    use_amp: bool = True,
    gradient_clip_val: float = 1.0,
    num_classes: int = 14,
) -> ModelTrainer14:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    trainer = ModelTrainer14(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        num_classes=num_classes,
        use_amp=use_amp,
        gradient_clip_val=gradient_clip_val,
    )

    return trainer
