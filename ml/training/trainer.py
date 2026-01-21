"""
Training Module for Student Models
===================================
Handles training loop, validation, checkpointing, and metric tracking.

Design Principles:
1. Modular: Supports both baseline and knowledge distillation training
2. Robust: Early stopping, gradient clipping, learning rate scheduling
3. Reproducible: Seeds, deterministic ops, checkpoint versioning
4. Observable: Progress bars, metric logging, tensorboard integration
5. Efficient: Mixed precision training (AMP), gradient accumulation

Key Features:
- Early stopping based on validation AUC-ROC
- Model checkpointing (best + last)
- Per-epoch metric tracking (AUC, F1, precision, recall)
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping for stability
- Support for weighted loss functions
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Callable
from pathlib import Path
import time
import json
from tqdm import tqdm
import numpy as np

from ml.training.metrics import MetricsTracker
from config.disease_labels import DISEASE_LABELS


class ModelTrainer:
    """
    Trainer for student models with comprehensive training loop
    
    Features:
    - Training and validation loops
    - Early stopping on validation AUC
    - Model checkpointing (best model based on metric)
    - Learning rate scheduling
    - Gradient clipping
    - Mixed precision training (optional)
    - Metric tracking and logging
    
    Args:
        model (nn.Module): Student model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        checkpoint_dir (Path): Directory to save checkpoints
        num_classes (int): Number of disease classes
        use_amp (bool): Use automatic mixed precision
        gradient_clip_val (float): Gradient clipping threshold
    """
    
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
        gradient_clip_val: float = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.use_amp = use_amp
        self.gradient_clip_val = gradient_clip_val
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Metric trackers
        self.train_metrics = MetricsTracker(num_classes, DISEASE_LABELS)
        self.val_metrics = MetricsTracker(num_classes, DISEASE_LABELS)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch (int): Current epoch number
        
        Returns:
            dict: Training metrics for this epoch
        """
        self.model.train()
        self.train_metrics.reset()
        
        running_loss = 0.0
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False
        )
        
        for batch_idx, (images, labels, _) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )
                
                self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            
            # Get predictions (sigmoid for multi-label)
            with torch.no_grad():
                preds = torch.sigmoid(logits)
                self.train_metrics.update(preds, labels)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': running_loss / (batch_idx + 1)
            })
        
        # Compute epoch metrics
        avg_loss = running_loss / len(self.train_loader)
        metrics = self.train_metrics.compute()
        
        return {
            'loss': avg_loss,
            'auc_macro': metrics.get('AUC_macro', 0.0),
            'f1_macro': metrics.get('F1_macro', 0.0),
            'precision_macro': metrics.get('Precision_macro', 0.0),
            'recall_macro': metrics.get('Recall_macro', 0.0)
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            epoch (int): Current epoch number
        
        Returns:
            dict: Validation metrics for this epoch
        """
        self.model.eval()
        self.val_metrics.reset()
        
        running_loss = 0.0
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch} [Val]",
            leave=False
        )
        
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(progress_bar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                
                running_loss += loss.item()
                
                # Get predictions
                preds = torch.sigmoid(logits)
                self.val_metrics.update(preds, labels)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': running_loss / (batch_idx + 1)
                })
        
        # Compute epoch metrics
        avg_loss = running_loss / len(self.val_loader)
        metrics = self.val_metrics.compute()
        
        return {
            'loss': avg_loss,
            'auc_macro': metrics.get('AUC_macro', 0.0),
            'f1_macro': metrics.get('F1_macro', 0.0),
            'precision_macro': metrics.get('Precision_macro', 0.0),
            'recall_macro': metrics.get('Recall_macro', 0.0)
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        metrics: Optional[Dict] = None
    ):
        """
        Save model checkpoint
        
        Args:
            epoch (int): Current epoch
            is_best (bool): Whether this is the best model so far
            metrics (dict): Metrics to save with checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auc': self.best_val_auc,
            'history': self.history,
            'metrics': metrics
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last_checkpoint.pth'
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (AUC: {self.best_val_auc:.4f})")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint to resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_auc = checkpoint['best_val_auc']
        self.history = checkpoint['history']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint['epoch']
    
    def train(
        self,
        num_epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Complete training loop
        
        Args:
            num_epochs (int): Number of epochs to train
            scheduler: Learning rate scheduler
            early_stopping_patience (int): Epochs to wait before early stopping
            verbose (bool): Print training progress
        
        Returns:
            dict: Training history
        """
        print("=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"Mixed precision: {self.use_amp}")
        print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_auc'].append(train_metrics['auc_macro'])
            self.history['val_auc'].append(val_metrics['auc_macro'])
            self.history['train_f1'].append(train_metrics['f1_macro'])
            self.history['val_f1'].append(val_metrics['f1_macro'])
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['auc_macro'])
                else:
                    scheduler.step()
            
            # Check if best model
            is_best = val_metrics['auc_macro'] > self.best_val_auc
            if is_best:
                self.best_val_auc = val_metrics['auc_macro']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best, val_metrics)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            if verbose:
                print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_metrics['loss']:.4f} | "
                      f"AUC: {train_metrics['auc_macro']:.4f} | "
                      f"F1: {train_metrics['f1_macro']:.4f}")
                print(f"  Val Loss:   {val_metrics['loss']:.4f} | "
                      f"AUC: {val_metrics['auc_macro']:.4f} | "
                      f"F1: {val_metrics['f1_macro']:.4f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                if is_best:
                    print(f"  🏆 New best AUC: {self.best_val_auc:.4f}")
            
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                print(f"  Best AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")
                break
        
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Best validation AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")
        print("=" * 70)
        
        # Save final history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    checkpoint_dir: Path = Path('checkpoints'),
    device: Optional[torch.device] = None,
    use_amp: bool = True
) -> ModelTrainer:
    """
    Factory function to create trainer with default optimizer
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        checkpoint_dir: Directory for checkpoints
        device: Training device
        use_amp: Use mixed precision training
    
    Returns:
        ModelTrainer: Configured trainer instance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # AdamW optimizer (better than Adam for transformers)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        num_classes=14,
        use_amp=use_amp,
        gradient_clip_val=1.0
    )
    
    return trainer
