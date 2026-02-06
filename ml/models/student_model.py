"""
Hybrid CNN-Transformer Student Models for Knowledge Distillation
=================================================================
Lightweight hybrid architectures combining CNNs with Transformer attention:

CNN Backbones (Feature Extraction):
- EfficientNet-B0 (5.3M params)
- MobileNetV3-Large (5.4M params)
- ShuffleNetV2 x1.0 (2.3M params)

Transformer Attention Modules (Spatial Relationships):
- MHSA (Multi-Head Self-Attention): Standard transformer attention
- Performer (FAVOR+): Efficient linear attention approximation

Architecture: CNN Backbone → Transformer Attention → Classification Head

Design Rationale:
1. CNN: Local feature extraction, translation invariance
2. Transformer: Long-range dependencies, global context
3. Hybrid: Best of both - local patterns + spatial relationships
4. Medical imaging: Pathologies can be small, scattered, or correlated

Total Variants: 3 backbones × 2 attention types = 6 models
"""

import torch
import torch.nn as nn
import timm
import torchvision.models as tv_models
from typing import Optional, Tuple, Callable
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) - Standard Transformer Attention
    
    Complexity: O(N²) where N = sequence length (H×W patches)
    
    Why MHSA?
    - Proven transformer architecture (ViT, BERT, etc.)
    - Captures long-range spatial dependencies
    - Multiple heads learn different feature relationships
    - Medical imaging: Correlations between distant regions (e.g., bilateral findings)
    
    Args:
        embed_dim (int): Feature dimension from CNN backbone
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate for regularization
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features [batch_size, num_patches, embed_dim]
        
        Returns:
            torch.Tensor: Attention-weighted features [batch_size, num_patches, embed_dim]
        """
        B, N, C = x.shape  # batch, num_patches, channels
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention: softmax(QK^T/√d)V
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x_attn = self.proj(x_attn)
        x_attn = self.dropout(x_attn)
        
        # Residual connection + normalization
        x = self.norm(x + x_attn)
        
        return x


class PerformerAttention(nn.Module):
    """
    Performer Attention - Efficient Linear Attention using FAVOR+
    
    Paper: "Rethinking Attention with Performers" (Choromanski et al., 2021)
    Complexity: O(N) instead of O(N²) - much faster for large images
    
    Why Performer?
    - Linear complexity: Scales better to larger feature maps
    - FAVOR+ kernel: Approximates standard attention accurately
    - Memory efficient: Critical for medical imaging (high resolution)
    - Deployment ready: Faster inference on edge devices
    
    How it works:
    - Instead of softmax(QK^T)V, uses random feature maps φ(Q)φ(K)^TV
    - Kernel approximation: softmax similarity ≈ φ(q)^Tφ(k)
    - Associativity: Compute φ(K)^TV first → O(N) instead of O(N²)
    
    Args:
        embed_dim (int): Feature dimension
        num_heads (int): Number of attention heads
        num_features (int): Number of random features for kernel approximation
        dropout (float): Dropout rate
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 8,
        num_features: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features
        
        # QKV projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Random features for kernel approximation (frozen)
        self.register_buffer(
            'random_features',
            torch.randn(num_features, self.head_dim) / math.sqrt(self.head_dim)
        )
    
    def _kernel_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random feature map: φ(x) = exp(ωx - ||x||²/2) / √m
        
        Args:
            x (torch.Tensor): Input [B, H, N, D]
        
        Returns:
            torch.Tensor: Feature map [B, H, N, M]
        """
        # Normalize
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2.0
        
        # Project onto random features
        x_proj = x @ self.random_features.T  # [B, H, N, M]
        
        # Apply kernel transformation
        phi_x = torch.exp(x_proj - x_norm_sq) / math.sqrt(self.num_features)
        
        return phi_x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features [batch_size, num_patches, embed_dim]
        
        Returns:
            torch.Tensor: Attention-weighted features [batch_size, num_patches, embed_dim]
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply kernel feature maps
        q_prime = self._kernel_feature_map(q)  # [B, H, N, M]
        k_prime = self._kernel_feature_map(k)  # [B, H, N, M]
        
        # Linear attention: φ(Q)(φ(K)^TV) instead of softmax(QK^T)V
        # This is O(N) instead of O(N²)
        kv = k_prime.transpose(-2, -1) @ v  # [B, H, M, D]
        qkv_out = q_prime @ kv  # [B, H, N, D]
        
        # Normalization
        q_sum = q_prime.sum(dim=-1, keepdim=True) + 1e-6
        k_sum = k_prime.transpose(-2, -1).sum(dim=-1, keepdim=True)  # [B, H, M, 1]
        normalizer = q_prime @ k_sum  # [B, H, N, 1]
        qkv_out = qkv_out / (normalizer + 1e-6)
        
        # Reshape and project
        x_attn = qkv_out.transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)
        x_attn = self.dropout(x_attn)
        
        # Residual connection + normalization
        x = self.norm(x + x_attn)
        
        return x


class StudentClassificationHead(nn.Module):
    """
    Multi-label classification head for chest X-ray pathology detection
    
    Architecture:
        Input Features → GAP → FC(intermediate) → ReLU → Dropout → FC(num_classes) → Sigmoid
    
    Why this design?
    - GAP: Reduces parameters, provides spatial invariance
    - Intermediate FC: Non-linear feature transformation (better than direct projection)
    - Dropout: Critical regularization for small medical datasets
    - Sigmoid: Multi-label output (not softmax which is mutually exclusive)
    
    Args:
        in_features (int): Number of input features from backbone
        num_classes (int): Number of disease classes (14 for ChestX-ray14)
        dropout (float): Dropout probability
        hidden_dim (int): Intermediate feature dimension
    """
    
    def __init__(
        self, 
        in_features: int, 
        num_classes: int = 14,
        dropout: float = 0.3,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        
        # Hidden dimension defaults to in_features // 2
        if hidden_dim is None:
            hidden_dim = in_features // 2
        
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
            # Note: No sigmoid here - will be applied in loss function (BCEWithLogitsLoss)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization for ReLU activations"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Feature maps [batch_size, channels, height, width]
        
        Returns:
            torch.Tensor: Logits [batch_size, num_classes]
        """
        # Global average pooling
        x = self.gap(x)  # [B, C, H, W] -> [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C, 1, 1] -> [B, C]
        
        # Classification
        logits = self.classifier(x)  # [B, C] -> [B, num_classes]
        
        return logits


class HybridStudentModel(nn.Module):
    """
    Hybrid CNN-Transformer Student Model for Knowledge Distillation
    
    Architecture:
        Input Image → CNN Backbone → Transformer Attention → Classification Head
    
    CNN Backbones (Feature Extraction):
    - efficientnet_b0: 5.3M params, compound scaling
    - mobilenet_v3_large: 5.4M params, efficient mobile architecture
    - shufflenet_v2_x1_0: 2.3M params, lightweight architecture
    
    Transformer Attention (Spatial Relationships):
    - mhsa: Multi-Head Self-Attention (standard, O(N²))
    - performer: Performer with FAVOR+ (efficient, O(N))
    
    Design Rationale:
    - CNN: Local feature extraction (edges, textures, shapes)
    - Transformer: Long-range dependencies (spatial correlations)
    - Medical imaging: Pathologies can be scattered, subtle, or correlated
    
    Args:
        backbone (str): CNN backbone name
        attention_type (str): Transformer attention type ('mhsa' or 'performer')
        num_classes (int): Number of disease classes
        pretrained (bool): Use ImageNet pre-trained weights
        attention_heads (int): Number of attention heads
        dropout (float): Dropout rate
        num_features (int): Random features for Performer (if attention_type='performer')
    """
    
    def __init__(
        self,
        backbone: str = 'efficientnet_b0',
        attention_type: str = 'mhsa',
        num_classes: int = 14,
        pretrained: bool = True,
        attention_heads: int = 8,
        dropout: float = 0.3,
        num_features: int = 256
    ):
        super().__init__()
        
        self.backbone_name = backbone
        self.attention_type = attention_type
        self.num_classes = num_classes
        
        # Load CNN backbone
        self.backbone, self.feature_dim = self._load_backbone(backbone, pretrained)
        
        # Transformer attention module
        if attention_type == 'mhsa':
            self.attention = MultiHeadSelfAttention(
                embed_dim=self.feature_dim,
                num_heads=attention_heads,
                dropout=dropout
            )
        elif attention_type == 'performer':
            self.attention = PerformerAttention(
                embed_dim=self.feature_dim,
                num_heads=attention_heads,
                num_features=num_features,
                dropout=dropout
            )
        else:
            raise ValueError(f"Invalid attention_type: {attention_type}. Choose 'mhsa' or 'performer'")
        
        # Classification head
        self.head = StudentClassificationHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            dropout=dropout
        )
    
    def _load_backbone(self, backbone: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """
        Load pre-trained CNN backbone and return feature dimension
        
        Returns:
            Tuple[nn.Module, int]: (backbone_model, feature_dimension)
        """
        # Torchvision models (not available in timm)
        torchvision_models = {'shufflenet_v2_x1_0'}
        
        # timm models
        timm_models = {'efficientnet_b0', 'mobilenetv3_large_100'}
        
        supported_backbones = timm_models | torchvision_models

        if backbone not in supported_backbones:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Choose from: {', '.join(sorted(supported_backbones))}"
            )

        # Handle torchvision models
        if backbone in torchvision_models:
            if backbone == 'shufflenet_v2_x1_0':
                # Load pretrained ShuffleNetV2 from torchvision
                weights = tv_models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
                full_model = tv_models.shufflenet_v2_x1_0(weights=weights)
                
                # Extract feature extractor (remove classifier)
                # ShuffleNetV2 structure: conv1 -> maxpool -> stage2-4 -> conv5
                feature_extractor = nn.Sequential(
                    full_model.conv1,
                    full_model.maxpool,
                    full_model.stage2,
                    full_model.stage3,
                    full_model.stage4,
                    full_model.conv5
                )
                
                # ShuffleNetV2 x1.0 outputs 1024 channels after conv5
                feature_dim = 1024
                
                # Wrap to return list format (consistent with timm features_only)
                class TorchvisionWrapper(nn.Module):
                    def __init__(self, feature_extractor):
                        super().__init__()
                        self.features = feature_extractor
                    
                    def forward(self, x):
                        return [self.features(x)]  # Return as list
                
                model = TorchvisionWrapper(feature_extractor)
                return model, feature_dim
        
        # Handle timm models
        else:
            # Use features_only=True to get raw spatial feature maps (B, C, H, W)
            # This prevents MobileNetV3 and others from applying final head layers
            model = timm.create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                out_indices=[-1]  # Only return the last feature stage
            )

            # Get feature dimension from the last stage
            feature_info = model.feature_info[-1]
            feature_dim = feature_info['num_chs']

            return model, feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input images [batch_size, 3, 224, 224]
        
        Returns:
            torch.Tensor: Logits [batch_size, num_classes]
        """
        # Extract features from CNN backbone (features_only=True returns a list)
        features = self.backbone(x)[-1]  # Get the last (and only) feature map [B, C, H, W]
        
        # Now reliably 4D
        B, C, H, W = features.shape
        
        # Apply transformer attention (always enabled in hybrid models)
        if H * W > 1:
            # Reshape to sequence: [B, C, H, W] -> [B, H*W, C]
            features_seq = features.view(B, C, H * W).permute(0, 2, 1)
            
            # Apply transformer attention (MHSA or Performer)
            features_seq = self.attention(features_seq)  # [B, H*W, C]
            
            # Reshape back: [B, H*W, C] -> [B, C, H, W]
            features = features_seq.permute(0, 2, 1).view(B, C, H, W)
        
        # Classification head
        logits = self.head(features)
        
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb


def create_student_model(
    architecture: str,
    num_classes: int = 14,
    pretrained: bool = True
) -> HybridStudentModel:
    """
    Factory function to create hybrid CNN-Transformer student models
    
    Naming Convention: {backbone}_{attention}
    
    Available Architectures:
    - 'efficientnet_b0_mhsa': EfficientNet-B0 + MHSA
    - 'efficientnet_b0_performer': EfficientNet-B0 + Performer
    - 'mobilenet_v3_large_mhsa': MobileNetV3-Large + MHSA
    - 'mobilenet_v3_large_performer': MobileNetV3-Large + Performer
    - 'shufflenet_v2_x1_0_mhsa': ShuffleNetV2 x1.0 + MHSA
    - 'shufflenet_v2_x1_0_performer': ShuffleNetV2 x1.0 + Performer
    
    Total: 3 backbones × 2 attention = 6 hybrid models
    
    Args:
        architecture (str): Model architecture name
        num_classes (int): Number of output classes (14 for ChestX-ray14)
        pretrained (bool): Use ImageNet pre-trained weights
    
    Returns:
        HybridStudentModel: Initialized hybrid model
    
    Example:
        >>> model = create_student_model('efficientnet_b0_mhsa')
        >>> print(f"Parameters: {model.get_num_params():,}")
        >>> print(f"Size: {model.get_model_size_mb():.2f} MB")
    """
    # Look up configuration from MODEL_CONFIGS
    if architecture not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Choose from: {', '.join(MODEL_CONFIGS.keys())}"
        )
    
    config = MODEL_CONFIGS[architecture]
    backbone = config['backbone']  # Use timm-compatible name from config
    attention_type = config['attention']
    
    model = HybridStudentModel(
        backbone=backbone,
        attention_type=attention_type,
        num_classes=num_classes,
        pretrained=pretrained,
        attention_heads=8,
        dropout=0.3,
        num_features=256  # For Performer
    )
    
    return model


# Model configurations for experiments
MODEL_CONFIGS = {
    # EfficientNet-B0 variants
    'efficientnet_b0_mhsa': {
        'backbone': 'efficientnet_b0',
        'attention': 'mhsa',
        'description': 'EfficientNet-B0 + Multi-Head Self-Attention'
    },
    'efficientnet_b0_performer': {
        'backbone': 'efficientnet_b0',
        'attention': 'performer',
        'description': 'EfficientNet-B0 + Performer (efficient attention)'
    },

    # MobileNetV3-Large variants
    'mobilenet_v3_large_mhsa': {
        'backbone': 'mobilenetv3_large_100',
        'attention': 'mhsa',
        'description': 'MobileNetV3-Large + Multi-Head Self-Attention'
    },
    'mobilenet_v3_large_performer': {
        'backbone': 'mobilenetv3_large_100',
        'attention': 'performer',
        'description': 'MobileNetV3-Large + Performer (efficient attention)'
    },

    # ShuffleNetV2 x1.0 variants (from torchvision)
    'shufflenet_v2_x1_0_mhsa': {
        'backbone': 'shufflenet_v2_x1_0',
        'attention': 'mhsa',
        'description': 'ShuffleNetV2 x1.0 + Multi-Head Self-Attention'
    },
    'shufflenet_v2_x1_0_performer': {
        'backbone': 'shufflenet_v2_x1_0',
        'attention': 'performer',
        'description': 'ShuffleNetV2 x1.0 + Performer (efficient attention)'
    }
}


if __name__ == "__main__":
    # Test hybrid CNN-Transformer model creation
    print("=" * 70)
    print("HYBRID CNN-TRANSFORMER STUDENT MODELS")
    print("=" * 70)
    print("\nTesting 6 model variants:")
    print("  3 CNN backbones × 2 Transformer attention = 6 models\n")
    
    for arch_name, config in MODEL_CONFIGS.items():
        print(f"\n{arch_name}:")
        print(f"  {config['description']}")
        print(f"  Backbone: {config['backbone']}")
        print(f"  Attention: {config['attention'].upper()}")
        
        try:
            model = create_student_model(arch_name, num_classes=14)
            
            num_params = model.get_num_params()
            model_size = model.get_model_size_mb()
            
            print(f"  Parameters: {num_params:,}")
            print(f"  Model Size: {model_size:.2f} MB")
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"  Input:  {list(dummy_input.shape)}")
            print(f"  Output: {list(output.shape)}")
            print(f"  Logits Range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"  ✓ Forward pass successful")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    print("\n" + "=" * 70)
    print("Model Testing Complete")
    print("=" * 70)
