"""
Central Configuration File for X-Lite Project
Manages paths, hyperparameters, and model settings
"""

import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # ============= Project Paths =============
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    CHECKPOINT_DIR = ROOT_DIR / 'ml' / 'models' / 'checkpoints'
    LOGS_DIR = ROOT_DIR / 'logs'
    
    # Dataset metadata
    METADATA_CSV = DATA_DIR / 'metadata.csv'
    
    # ============= Dataset Settings =============
    DATASET_NAME = 'ChestX-ray14'
    NUM_CLASSES = 14
    IMAGE_SIZE = 224  # Default input size (can be overridden per model)
    
    # Data splits
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # ============= Image Preprocessing =============
    IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
    IMAGE_STD = [0.229, 0.224, 0.225]   # ImageNet std
    
    # ============= Training Hyperparameters =============
    # These are baseline values - will be tuned during experiments
    
    # Teacher Model
    TEACHER_EPOCHS = 30
    TEACHER_BATCH_SIZE = 32
    TEACHER_LR = 1e-4
    TEACHER_WEIGHT_DECAY = 1e-4
    
    # Student Model
    STUDENT_EPOCHS = 40
    STUDENT_BATCH_SIZE = 64
    STUDENT_LR = 1e-3
    STUDENT_WEIGHT_DECAY = 1e-4
    
    # Knowledge Distillation
    KD_TEMPERATURE = 4.0  # Temperature for soft targets (will experiment with 2, 4, 6, 8)
    KD_ALPHA = 0.7        # Weight for distillation loss (1-alpha for hard loss)
    
    # Optimizer
    OPTIMIZER = 'Adam'  # Options: Adam, AdamW, SGD
    SCHEDULER = 'ReduceLROnPlateau'  # Options: ReduceLROnPlateau, CosineAnnealing
    
    # ============= Model Architectures =============
    # Teacher Model
    TEACHER_BACKBONE = 'densenet121'  # CheXNet-inspired
    
    # Student Model Variants (to be experimented)
    STUDENT_BACKBONES = [
        'efficientnet_b0',
        'convnext_tiny',
        'mobilenetv3_large_100',
        'resnet50'
    ]
    
    # Transformer Attention Modules (to be experimented)
    ATTENTION_TYPES = [
        'mhsa',           # Multi-Head Self-Attention
        'performer',      # Performer (linear attention)
        'linear',         # Linear attention
        'none'            # Baseline without attention
    ]
    
    # ============= Model Settings =============
    DROPOUT_RATE = 0.3
    NUM_HEADS = 8          # For multi-head attention
    EMBED_DIM = 512        # Embedding dimension
    
    # ============= Training Settings =============
    EARLY_STOPPING_PATIENCE = 7
    GRADIENT_CLIP_NORM = 1.0
    MIXED_PRECISION = True  # Use AMP for faster training
    
    # ============= Evaluation Metrics =============
    METRICS = [
        'auc_roc',
        'precision',
        'recall',
        'f1_score',
        'average_precision'
    ]
    
    PRIMARY_METRIC = 'auc_roc'  # Main metric for model selection
    
    # ============= Inference Settings =============
    CONFIDENCE_THRESHOLD = 0.5  # Default threshold for binary classification
    GRAD_CAM_LAYER = 'layer4'   # Layer for Grad-CAM visualization
    
    # ============= Backend Settings =============
    API_HOST = '0.0.0.0'
    API_PORT = 8000
    UPLOAD_FOLDER = ROOT_DIR / 'backend' / 'uploads'
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.dcm'}
    
    # ============= Frontend Settings =============
    FRONTEND_PORT = 3000
    
    # ============= Logging =============
    LOG_LEVEL = 'INFO'
    TENSORBOARD_LOG_DIR = LOGS_DIR / 'tensorboard'
    WANDB_PROJECT = 'x-lite-chest-xray'
    WANDB_ENTITY = 'dinethsadee01'  # Replace with your W&B username
    
    # ============= Device Settings =============
    DEVICE = 'cuda'  # Will auto-detect in code: cuda, mps, or cpu
    NUM_WORKERS = 4  # DataLoader workers
    PIN_MEMORY = True
    
    # ============= Reproducibility =============
    SEED = 42
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.CHECKPOINT_DIR,
            cls.LOGS_DIR,
            cls.TENSORBOARD_LOG_DIR,
            cls.UPLOAD_FOLDER
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_checkpoint_path(cls, model_name):
        """Get checkpoint path for a specific model"""
        return cls.CHECKPOINT_DIR / f'{model_name}.pth'
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }


class ExperimentConfig(Config):
    """Configuration for hyperparameter experiments"""
    
    # Experiment tracking
    EXPERIMENT_NAME = 'baseline'
    EXPERIMENT_ID = None  # Will be auto-generated
    
    # Grid search parameters
    TEMPERATURES = [2.0, 4.0, 6.0, 8.0]
    ALPHAS = [0.5, 0.7, 0.9]
    LEARNING_RATES = [1e-4, 5e-4, 1e-3]
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
