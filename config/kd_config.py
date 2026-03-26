"""
Knowledge Distillation Configuration
====================================

CRITICAL SETTINGS for TorchXRayVision teacher:

1. Normalization: TorchXRayVision expects images in range suitable for
   medical imaging (roughly [-1024, 1024] as Hounsfield units).
   However, our CLAHE preprocessing outputs [0, 255] uint8.
   The teacher will internally normalize appropriately.

2. Disease Classes: Only distill the 14 pathological classes.
   The 15th class (No_Finding) is taught via hard labels only.

3. Disease Order: Must match TorchXRayVision's NIH order, not alphabetical.
   See config/disease_mapping.py for the bidirectional mapping.
"""

import os
from pathlib import Path


class KDConfig:
    """Knowledge Distillation specific configuration"""
    
    # ============= Project Paths =============
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / 'data'
    CHECKPOINT_DIR = ROOT_DIR / 'ml' / 'models' / 'new checkpoints'
    
    # ============= Teacher Model Settings =============
    TEACHER_MODEL_TYPE = 'torchxrayvision'  # MUST USE XRV
    TEACHER_NUM_CLASSES = 14  # Only distill pathological classes
    TEACHER_CHECKPOINT = None  # Will use pretrained XRV weights
    
    # TorchXRayVision expects these normalizations
    # (images are converted from [0, 255] uint8 to float, then normalized)
    TEACHER_IMAGE_MEAN = [0.485, 0.456, 0.406]  # XRV's internal normalization
    TEACHER_IMAGE_STD = [0.229, 0.224, 0.225]
    
    # ============= Student Model Settings =============
    STUDENT_NUM_CLASSES = 15  # 14 pathologies + explicit No_Finding
    
    # PRIMARY STUDENT: Your best baseline model for KD comparison
    PRIMARY_STUDENT = 'efficientnet_b0_performer'  # Best from Phase 1
    
    # ALTERNATIVE STUDENTS: For ablation studies
    STUDENT_BACKBONES = [
        'efficientnet_b0_performer',  # PRIMARY - Your best baseline (~5.3M params)
        'mobilenetv3_large_performer', # Alternative 1 (~5.4M params, faster)
        'shufflenet_v2_x1_0_performer', # Alternative 2 (~2.3M params, smallest)
    ]
    
    # ============= Knowledge Distillation Hyperparameters =============
    # Temperature: Controls softness of soft targets
    # - Higher T (T=8): Softer targets, more regularization
    # - Lower T (T=2): Harder targets, more focused learning
    TEMPERATURE = 4.0  # Recommended range: 4-8
    
    # Alpha: Weight balance between hard and soft losses
    # - alpha=0.9: 90% KD loss (soft) + 10% CE loss (hard)
    # - alpha=0.5: 50% KD loss + 50% CE loss (balanced)
    # - alpha=0.3: 30% KD loss + 70% CE loss (more reliable)
    # For medical imaging with explicit No_Finding class, recommend 0.5-0.7
    # Using 0.3 for conservative KD approach with weak teacher
    ALPHA = 0.3  # Conservative: 30% KD loss + 70% ground truth
    
    # ============= Training Settings =============
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0
    EARLY_STOPPING_PATIENCE = 5
    
    # ============= Loss Function =============
    # Hard loss: CE or BCE depending on multi-label setup
    # Always use Focal Loss for medical imaging (class imbalance)
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # ============= Data Settings =============
    # Image preprocessing
    IMAGE_SIZE = 224
    USE_CLAHE = True
    CLAHE_CACHE = DATA_DIR / 'clahe_cache'
    
    # Augmentation
    USE_AUGMENTATION = True
    AUGMENTATION_STRENGTH = 'medium'
    
    # ============= Distillation Strategy =============
    DISTILL_CLASSES = 14  # Only the 14 pathological classes
    HARD_LABELS_CLASSES = [14]  # No_Finding learned only from hard labels
    
    # Class-wise temperature scaling (optional)
    # Can use higher T for rare diseases, lower T for common ones
    USE_CLASS_WISE_TEMPERATURE = False
    
    # ============= Validation & Checkpointing =============
    SAVE_BEST_ONLY = True
    SAVE_FREQUENCY = 5  # Save every N epochs
    VALIDATE_FREQUENCY = 1
    
    # ============= Disease Order =============
    # DO NOT CHANGE - this matches TorchXRayVision's canonical order
    XRV_DISEASE_ORDER = [
        'Atelectasis',           # 0
        'Consolidation',         # 1
        'Infiltration',          # 2
        'Pneumothorax',          # 3
        'Edema',                 # 4
        'Emphysema',             # 5
        'Fibrosis',              # 6
        'Effusion',              # 7
        'Pneumonia',             # 8
        'Pleural_Thickening',    # 9
        'Cardiomegaly',          # 10
        'Nodule',                # 11
        'Mass',                  # 12
        'Hernia',                # 13
    ]


class KD_TemperatureSweep:
    """Configuration for temperature sweep experiments"""
    TEMPERATURES = [2.0, 4.0, 6.0, 8.0]
    ALPHA_VALUES = [0.3, 0.5, 0.7, 0.9]
