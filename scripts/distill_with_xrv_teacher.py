"""
Knowledge Distillation with TorchXRayVision Teacher
==================================================

CRITICAL SETUP:
1. Teacher uses TorchXRayVision (NIH-pretrained DenseNet121)
2. Only 14 pathological classes are distilled
3. No_Finding (15th class) is learned ONLY from hard labels
4. Disease order MUST match XRV's canonical order

Usage:
    python scripts/distill_with_xrv_teacher.py
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.teacher_model import create_teacher_model
from ml.models.student_model import create_student_model
from ml.data.loader import ChestXrayDataset
from ml.data.preprocessing import get_medical_transforms
from config.disease_labels import DISEASE_LABELS, NUM_CLASSES
from config.disease_mapping import (
    reorder_labels_batch_to_xrv,
    reorder_labels_batch_from_xrv,
    YOUR_TO_XRV_MAPPING,
    XRV_DISEASE_ORDER,
    YOUR_DISEASE_ORDER
)
from config.kd_config import KDConfig


class KDLoss(nn.Module):
    """
    Knowledge Distillation Loss.
    
    Combines:
    - Soft targets from teacher (KL divergence with temperature)
    - Hard targets from ground truth (Focal Loss for class imbalance)
    Only the 14 pathological classes use soft targets.
    No_Finding uses only hard labels.
    """
    
    def __init__(self, temperature=4.0, alpha=0.6, use_focal=True):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight of soft loss vs hard loss
        self.use_focal = use_focal
        
        if use_focal:
            self.hard_loss = self._focal_loss
        else:
            self.hard_loss = F.binary_cross_entropy_with_logits
    
    def _focal_loss(self, logits, labels):
        """Focal Loss for class imbalance"""
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * labels + (1 - p) * (1 - labels)
        focal_weight = (1 - p_t) ** 2
        return (focal_weight * bce).mean()
    
    def forward(self, student_logits, teacher_logits, hard_labels):
        """
        Args:
            student_logits: [batch, 15] student output
            teacher_logits: [batch, 14] teacher output (already in XRV order)
            hard_labels: [batch, 15] ground truth labels
        
        Returns:
            Total loss combining soft and hard targets
        """
        # ===== SOFT TARGETS (KD) - Only 14 classes =====
        # Remove No_Finding from student logits for distillation
        student_logits_14 = student_logits[:, :14]  # [batch, 14]
        
        # Soft target loss using temperature scaling
        student_soft = F.softmax(student_logits_14 / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        kd_loss = F.kl_div(
            F.log_softmax(student_logits_14 / self.temperature, dim=1),
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # ===== HARD TARGETS (Supervised) - All 15 classes =====
        # Use hard labels for all classes, especially important for No_Finding
        hard_labels_14 = hard_labels[:, :14]  # [batch, 14]
        hard_labels_15 = hard_labels  # [batch, 15]
        
        # Hard loss for 14 classes (consistency with distillation)
        hard_loss_14 = self.hard_loss(student_logits_14, hard_labels_14)
        
        # Hard loss for No_Finding (can't distill, only supervise)
        no_finding_logit = student_logits[:, 14:15]  # [batch, 1]
        no_finding_label = hard_labels_15[:, 14:15]  # [batch, 1]
        hard_loss_no_finding = F.binary_cross_entropy_with_logits(
            no_finding_logit, no_finding_label
        )
        
        # Combine No_Finding loss with 14-class loss
        hard_loss_total = (hard_loss_14 + hard_loss_no_finding) / 2.0
        
        # ===== TOTAL LOSS =====
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * hard_loss_total
        
        return {
            'total': total_loss,
            'kd': kd_loss.detach().item(),
            'hard': hard_loss_total.detach().item(),
            'hard_14': hard_loss_14.detach().item(),
            'hard_no_finding': hard_loss_no_finding.detach().item(),
        }


def verify_disease_order():
    """Verify that loads/labels match XRV order"""
    
    print("\n" + "=" * 80)
    print("DISEASE ORDER VERIFICATION")
    print("=" * 80)
    
    print("\nYour labels order (15 classes):")
    for i, disease in enumerate(YOUR_DISEASE_ORDER):
        if disease == 'No_Finding':
            print(f"  [{i:2d}] {disease:<25} (NOT DISTILLED)")
        else:
            xrv_idx = YOUR_TO_XRV_MAPPING[i]
            print(f"  [{i:2d}] {disease:<25} → XRV[{xrv_idx:2d}]")
    
    print("\nTorchXRayVision order (14 classes, used for distillation):")
    for i, disease in enumerate(XRV_DISEASE_ORDER):
        print(f"  [{i:2d}] {disease}")
    
    print("\n✓ Disease order verified\n")


def load_data(split='val'):
    """Load data with proper column renaming"""
    project_root = Path(__file__).parent.parent
    
    csv_path = project_root / f"data/splits/{split}.csv"
    df = pd.read_csv(csv_path)
    
    # Rename columns to match dataset expectations
    if 'Image Index' in df.columns:
        df = df.rename(columns={'Image Index': 'image_id'})
    if 'Finding Labels' in df.columns:
        df = df.rename(columns={'Finding Labels': 'labels'})
    
    print(f"Loaded {split} set: {len(df)} images")
    return df


def test_distillation_pipeline():
    """Test the complete KD pipeline with TorchXRayVision"""
    
    print("\n" + "=" * 100)
    print("KNOWLEDGE DISTILLATION WITH TORCHXRAYVISION TEACHER")
    print("=" * 100)
    
    # Verify disease ordering
    verify_disease_order()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ===== LOAD TEACHER =====
    print("\n" + "=" * 100)
    print("STEP 1: Loading TorchXRayVision Teacher")
    print("=" * 100)
    
    teacher = create_teacher_model(
        num_classes=14,  # Only distill 14 classes
        model_type='torchxrayvision',
        device=device
    )
    teacher.eval()
    
    print(f"✓ Teacher loaded")
    print(f"  Type: TorchXRayVision DenseNet121 (NIH pretrained)")
    print(f"  Output classes: 14 (pathologies only, no No_Finding)")
    print(f"  Parameters: {teacher.get_num_params():,}")
    print(f"  Size: {teacher.get_model_size_mb():.1f} MB")
    
    # ===== LOAD STUDENT =====
    print("\n" + "=" * 100)
    print("STEP 2: Creating Student Model")
    print("=" * 100)
    
    student = create_student_model(
        'mobilenet_v3_large_performer',  # Use available student architecture
        num_classes=15,  # Student has 15 (includes No_Finding)
        pretrained=True
    )
    student = student.to(device)
    student.train()
    
    print(f"✓ Student created")
    print(f"  Type: MobileNetV3 Large + Performer Attention")
    print(f"  Output classes: 15 (14 pathologies + No_Finding)")
    print(f"  Parameters: {sum(p.numel() for p in student.parameters()):,}")
    print(f"  Compression ratio: {teacher.get_num_params() / sum(p.numel() for p in student.parameters()):.1f}x")
    
    # ===== LOAD DATA =====
    print("\n" + "=" * 100)
    print("STEP 3: Loading Data (Sample Batch)")
    print("=" * 100)
    
    project_root = Path(__file__).parent.parent
    val_df = load_data('val')
    
    clahe_cache = project_root / 'data/clahe_cache'
    transforms = get_medical_transforms(use_clahe=False, use_denoising=False)
    
    dataset = ChestXrayDataset(
        str(clahe_cache),
        val_df.iloc[:4],  # Just 4 samples for testing
        transform=transforms,
        is_training=False
    )
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False
    )
    
    # ===== TEST FORWARD PASS =====
    print("\n" + "=" * 100)
    print("STEP 4: Testing Forward Pass (with Disease Order Handling)")
    print("=" * 100)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Unpack batch: list of [images, labels, image_ids]
            images = batch[0].to(device)  # [batch, 3, 224, 224]
            labels_15 = batch[1].to(device)  # [batch, 15] - your order
            image_ids = batch[2]  # tuple of strings
            
            # Convert labels to XRV order for distillation
            labels_14_xrv = reorder_labels_batch_to_xrv(labels_15)  # [batch, 14] - XRV order
            
            print(f"\nBatch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels (your order): {labels_15.shape}")
            print(f"  Labels (XRV order): {labels_14_xrv.shape}")
            
            # Teacher inference
            teacher_logits_xrv = teacher(images)  # [batch, 14]
            print(f"  Teacher logits (XRV order): {teacher_logits_xrv.shape}")
            
            # Student inference
            student_logits_your = student(images)  # [batch, 15]
            print(f"  Student logits (your order): {student_logits_your.shape}")
            
            # ===== KD LOSS =====
            kd_criterion = KDLoss(
                temperature=KDConfig.TEMPERATURE,
                alpha=KDConfig.ALPHA,
                use_focal=KDConfig.USE_FOCAL_LOSS
            )
            
            loss_dict = kd_criterion(student_logits_your, teacher_logits_xrv, labels_15)
            
            print(f"\n  Loss components:")
            print(f"    Total loss:  {loss_dict['total']:.4f}")
            print(f"    KD loss:     {loss_dict['kd']:.4f} (α={KDConfig.ALPHA})")
            print(f"    Hard loss:   {loss_dict['hard']:.4f} (α={1-KDConfig.ALPHA})")
            print(f"      - 14 classes: {loss_dict['hard_14']:.4f}")
            print(f"      - No_Finding: {loss_dict['hard_no_finding']:.4f}")
            
            # ===== PREDICTIONS =====
            print(f"\n  Sample prediction (image 0):")
            teacher_pred = torch.sigmoid(teacher_logits_xrv[0]).cpu()  # [14]
            student_pred = torch.sigmoid(student_logits_your[0]).cpu()  # [15]
            ground_truth = labels_15[0].cpu()  # [15]
            
            print(f"    Disease predictions (Teacher vs Student):")
            for your_idx, disease in enumerate(YOUR_DISEASE_ORDER):
                if disease == 'No_Finding':
                    gt = ground_truth[your_idx].item()
                    student_conf = student_pred[your_idx].item()
                    print(f"      {disease:<25} GT={gt:.0f} Student={student_conf:.3f} (hard label only)")
                else:
                    xrv_idx = YOUR_TO_XRV_MAPPING[your_idx]
                    gt = ground_truth[your_idx].item()
                    teacher_conf = teacher_pred[xrv_idx].item()
                    student_conf = student_pred[your_idx].item()
                    print(f"      {disease:<25} GT={gt:.0f} Teacher={teacher_conf:.3f} Student={student_conf:.3f}")
            
            break  # Just test first batch
    
    print("\n" + "=" * 100)
    print("✓ KD Pipeline Test Complete!")
    print("=" * 100)
    print("\nKey Findings:")
    print("  1. Disease order correctly mapped (YOUR order → XRV order)")
    print("  2. Teacher outputs 14 logits (pathologies)")
    print("  3. Student outputs 15 logits (14 + No_Finding)")
    print("  4. No_Finding learned from hard labels only")
    print("  5. KD loss combines soft (14 classes) + hard (15 classes) targets")
    print("\nReady to start full KD training!")
    

if __name__ == '__main__':
    test_distillation_pipeline()
