"""
Knowledge Distillation Training with TorchXRayVision Teacher
=============================================================
Uses official TorchXRayVision preprocessing for teacher model.
Student uses standard medical preprocessing with CLAHE.

Alpha = 0.3: Conservative approach (30% KD + 70% ground truth)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.models.teacher_model import create_teacher_model
from ml.models.student_model import create_student_model
from ml.data.loader import ChestXrayDataset
from ml.data.preprocessing import get_medical_transforms
from ml.data.xrv_preprocessing import get_xrv_teacher_preprocessor
from config.kd_config import KDConfig
from config.disease_labels import DISEASE_LABELS
from config.disease_mapping import reorder_labels_batch_from_xrv


class KDLoss(nn.Module):
    """Knowledge Distillation Loss with Focal Loss for medical imaging"""
    
    def __init__(self, temperature=6.0, alpha=0.3, focal_gamma=2.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for KD loss
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.focal_gamma = focal_gamma
    
    def focal_loss(self, logits, targets):
        """Focal loss for class imbalance"""
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma) * bce_loss
        return focal_loss.mean()
    
    def forward(self, student_logits, teacher_logits, targets):
        """
        Args:
            student_logits: [batch, 15] - all student outputs
            teacher_logits: [batch, 14] - reordered teacher outputs (in student order)
            targets: [batch, 15] - ground truth labels
        """
        # Soft loss (KD) - only first 14 classes
        T = self.temperature
        soft_student = torch.log_softmax(student_logits[:, :14] / T, dim=1)
        soft_teacher = torch.softmax(teacher_logits / T, dim=1)
        soft_loss = self.kl_div(soft_student, soft_teacher) * (T * T)
        
        # Hard loss (ground truth) - all 15 classes
        hard_loss = self.focal_loss(student_logits, targets)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, soft_loss, hard_loss


def train_kd():
    """Main KD training function"""
    
    print("=" * 80)
    print("KNOWLEDGE DISTILLATION TRAINING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Teacher: TorchXRayVision DenseNet121 (densenet121-res224-all)")
    print(f"  Student: {KDConfig.PRIMARY_STUDENT}")
    print(f"  Alpha: {KDConfig.ALPHA} (KD weight)")
    print(f"  Temperature: {KDConfig.TEMPERATURE}")
    print(f"  Epochs: {KDConfig.EPOCHS}")
    print(f"  Batch Size: {KDConfig.BATCH_SIZE}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}\n")
    
    # Load data
    train_csv = KDConfig.DATA_DIR / 'splits' / 'train.csv'
    val_csv = KDConfig.DATA_DIR / 'splits' / 'val.csv'
    clahe_cache = KDConfig.CLAHE_CACHE
    
    # Student preprocessing (standard with CLAHE)
    student_transform = get_medical_transforms(
        use_clahe=KDConfig.USE_CLAHE,
        use_denoising=False
    )
    
    # Teacher preprocessing (official XRV)
    teacher_preprocessor = get_xrv_teacher_preprocessor(image_size=224)
    
    print("Loading datasets...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # Create data loaders (using student preprocessing)
    train_dataset = ChestXrayDataset(
        str(clahe_cache), train_df, transform=student_transform, is_training=True
    )
    val_dataset = ChestXrayDataset(
        str(clahe_cache), val_df, transform=student_transform, is_training=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=KDConfig.BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=KDConfig.BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}\n")
    
    # Create models
    print("Creating teacher model...")
    teacher = create_teacher_model(device=device)
    teacher.eval()  # Always in eval mode
    
    print("\nCreating student model...")
    student = create_student_model(
        KDConfig.PRIMARY_STUDENT,
        num_classes=15,  # 14 + No_Finding
        pretrained=True  # Start with ImageNet weights
    )
    student = student.to(device)
    
    # Loss and optimizer
    criterion = KDLoss(
        temperature=KDConfig.TEMPERATURE,
        alpha=KDConfig.ALPHA,
        focal_gamma=KDConfig.FOCAL_GAMMA
    )
    
    optimizer = optim.AdamW(
        student.parameters(),
        lr=KDConfig.LEARNING_RATE,
        weight_decay=KDConfig.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80 + "\n")
    
    for epoch in range(KDConfig.EPOCHS):
        # Train
        student.train()
        train_loss = 0.0
        train_soft_loss = 0.0
        train_hard_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{KDConfig.EPOCHS}")
        for batch_idx, (student_images, targets, img_ids) in enumerate(pbar):
            student_images = student_images.to(device)
            targets = targets.to(device)
            
            # Get teacher predictions (with official XRV preprocessing)
            with torch.no_grad():
                # Preprocess images for teacher
                teacher_images = []
                for img_id in img_ids:
                    img_path = clahe_cache / img_id
                    teacher_img = teacher_preprocessor.preprocess_single_image(str(img_path))
                    teacher_images.append(teacher_img)
                teacher_images = torch.cat(teacher_images, dim=0).to(device)
                
                # Get teacher logits (14 classes in XRV order)
                teacher_logits_xrv = teacher(teacher_images)
                
                # Reorder to student's order
                teacher_logits = reorder_labels_batch_from_xrv(teacher_logits_xrv)
            
            # Student forward pass
            student_logits = student(student_images)
            
            # Compute loss
            loss, soft_loss, hard_loss = criterion(student_logits, teacher_logits, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), KDConfig.GRADIENT_CLIP)
            optimizer.step()
            
            # Track losses
            train_loss += loss.item()
            train_soft_loss += soft_loss.item()
            train_hard_loss += hard_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'soft': f'{soft_loss.item():.4f}',
                'hard': f'{hard_loss.item():.4f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        avg_soft_loss = train_soft_loss / len(train_loader)
        avg_hard_loss = train_hard_loss / len(train_loader)
        
        # Validation
        student.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for student_images, targets, img_ids in val_loader:
                student_images = student_images.to(device)
                targets = targets.to(device)
                
                # Teacher predictions
                teacher_images = []
                for img_id in img_ids:
                    img_path = clahe_cache / img_id
                    teacher_img = teacher_preprocessor.preprocess_single_image(str(img_path))
                    teacher_images.append(teacher_img)
                teacher_images = torch.cat(teacher_images, dim=0).to(device)
                
                teacher_logits_xrv = teacher(teacher_images)
                teacher_logits = reorder_labels_batch_from_xrv(teacher_logits_xrv)
                
                # Student predictions
                student_logits = student(student_images)
                
                # Loss
                loss, _, _ = criterion(student_logits, teacher_logits, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{KDConfig.EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f} (Soft: {avg_soft_loss:.4f}, Hard: {avg_hard_loss:.4f})")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            checkpoint_path = KDConfig.CHECKPOINT_DIR / f'kd_student_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': {
                    'alpha': KDConfig.ALPHA,
                    'temperature': KDConfig.TEMPERATURE,
                    'student_model': KDConfig.PRIMARY_STUDENT
                }
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{KDConfig.EARLY_STOPPING_PATIENCE}")
        
        # Early stopping
        if patience_counter >= KDConfig.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    train_kd()
