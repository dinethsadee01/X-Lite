"""
Knowledge Distillation Trainer
==============================
Trainer for knowledge distillation from teacher to student models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from pathlib import Path
import time
import json
from tqdm import tqdm
import numpy as np

from ml.training.metrics import MetricsTracker
from config.disease_labels import DISEASE_LABELS


class KDTrainer:
    """
    Trainer for knowledge distillation
    
    Trains a student model using knowledge from a teacher model.
    Teacher weights are frozen during training.
    
    Args:
        student_model (nn.Module): Student model to train
        teacher_model (nn.Module): Teacher model (frozen)
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Distillation loss function
        optimizer: Optimizer
        device (torch.device): Training device
        checkpoint_dir (Path): Directory to save checkpoints
        use_amp (bool): Use automatic mixed precision
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion,
        optimizer,
        device: torch.device,
        checkpoint_dir: Path,
        use_amp: bool = True
    ):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Scaler for AMP
        if use_amp:
            try:
                self.scaler = torch.amp.GradScaler('cuda')
            except Exception:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Metrics
        self.train_metrics = MetricsTracker(14, DISEASE_LABELS)
        self.val_metrics = MetricsTracker(14, DISEASE_LABELS)
        
        # History
        self.history = {
            'train_loss': [],
            'train_kd_loss': [],
            'train_ce_loss': [],
            'val_loss': [],
            'val_kd_loss': [],
            'val_ce_loss': [],
            'train_auc': [],
            'val_auc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch"""
        self.student.train()
        self.train_metrics.reset()
        
        running_total_loss = 0.0
        running_kd_loss = 0.0
        running_ce_loss = 0.0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False
        )
        
        for images, labels, _ in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    student_logits = self.student(images)
                    with torch.no_grad():
                        teacher_logits = self.teacher(images)
                    
                    loss_dict = self.criterion(
                        student_logits,
                        teacher_logits,
                        labels
                    )
                    total_loss = loss_dict['total_loss']
            else:
                student_logits = self.student(images)
                with torch.no_grad():
                    teacher_logits = self.teacher(images)
                
                loss_dict = self.criterion(
                    student_logits,
                    teacher_logits,
                    labels
                )
                total_loss = loss_dict['total_loss']
            
            # Backward
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
            
            # Track
            running_total_loss += total_loss.item()
            running_kd_loss += loss_dict['kd_loss'].item()
            running_ce_loss += loss_dict['ce_loss'].item()
            
            with torch.no_grad():
                preds = torch.sigmoid(student_logits)
                self.train_metrics.update(preds, labels)
            
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'kd': loss_dict['kd_loss'].item()
            })
        
        metrics = self.train_metrics.compute()
        
        return {
            'loss': running_total_loss / len(self.train_loader),
            'kd_loss': running_kd_loss / len(self.train_loader),
            'ce_loss': running_ce_loss / len(self.train_loader),
            'auc_macro': metrics.get('AUC_macro', 0.0),
            'f1_macro': metrics.get('F1_macro', 0.0)
        }
    
    def validate_epoch(self, epoch: int) -> dict:
        """Validate for one epoch"""
        self.student.eval()
        self.val_metrics.reset()
        
        running_total_loss = 0.0
        running_kd_loss = 0.0
        running_ce_loss = 0.0
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch} [Val]",
            leave=False
        )
        
        with torch.no_grad():
            for images, labels, _ in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                student_logits = self.student(images)
                teacher_logits = self.teacher(images)
                
                loss_dict = self.criterion(
                    student_logits,
                    teacher_logits,
                    labels
                )
                
                running_total_loss += loss_dict['total_loss'].item()
                running_kd_loss += loss_dict['kd_loss'].item()
                running_ce_loss += loss_dict['ce_loss'].item()
                
                preds = torch.sigmoid(student_logits)
                self.val_metrics.update(preds, labels)
                
                progress_bar.set_postfix({
                    'loss': loss_dict['total_loss'].item()
                })
        
        metrics = self.val_metrics.compute()
        
        return {
            'loss': running_total_loss / len(self.val_loader),
            'kd_loss': running_kd_loss / len(self.val_loader),
            'ce_loss': running_ce_loss / len(self.val_loader),
            'auc_macro': metrics.get('AUC_macro', 0.0),
            'f1_macro': metrics.get('F1_macro', 0.0)
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auc': self.best_val_auc,
            'history': self.history
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save last
        torch.save(checkpoint, self.checkpoint_dir / 'last_checkpoint.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_checkpoint.pth')
            print(f"  ✓ Saved best model (AUC: {self.best_val_auc:.4f})")
    
    def train(
        self,
        num_epochs: int,
        scheduler=None,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> dict:
        """Complete training loop"""
        print("=" * 70)
        print("KNOWLEDGE DISTILLATION TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Student parameters: {sum(p.numel() for p in self.student.parameters()):,}")
        print(f"Teacher parameters: {sum(p.numel() for p in self.teacher.parameters()):,}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train & validate
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_kd_loss'].append(train_metrics['kd_loss'])
            self.history['train_ce_loss'].append(train_metrics['ce_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_kd_loss'].append(val_metrics['kd_loss'])
            self.history['val_ce_loss'].append(val_metrics['ce_loss'])
            self.history['train_auc'].append(train_metrics['auc_macro'])
            self.history['val_auc'].append(val_metrics['auc_macro'])
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['auc_macro'])
                else:
                    scheduler.step()
            
            # Check best
            is_best = val_metrics['auc_macro'] > self.best_val_auc
            if is_best:
                self.best_val_auc = val_metrics['auc_macro']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save
            self.save_checkpoint(epoch, is_best)
            
            # Print
            epoch_time = time.time() - epoch_start
            if verbose:
                print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
                print(f"  Train: Loss={train_metrics['loss']:.4f} "
                      f"(KD={train_metrics['kd_loss']:.4f}, CE={train_metrics['ce_loss']:.4f}) "
                      f"AUC={train_metrics['auc_macro']:.4f}")
                print(f"  Val:   Loss={val_metrics['loss']:.4f} "
                      f"(KD={val_metrics['kd_loss']:.4f}, CE={val_metrics['ce_loss']:.4f}) "
                      f"AUC={val_metrics['auc_macro']:.4f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                if is_best:
                    print(f"  🏆 New best AUC: {self.best_val_auc:.4f}")
            
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠ Early stopping at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Best Val AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")
        print("=" * 70)
        
        # Save history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
