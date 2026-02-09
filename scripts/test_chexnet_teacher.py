"""Test CheXNet teacher model loading"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.teacher_model import create_teacher_model

print("=" * 80)
print("TESTING CHEXNET TEACHER MODEL")
print("=" * 80)

# Create teacher
print("\nCreating CheXNet teacher...")
teacher = create_teacher_model(
    num_classes=14,
    pretrained=True,
    model_type='chexnet'
)

print(f"\n✓ Teacher loaded successfully")
print(f"  Architecture: CheXNet (DenseNet121)")
print(f"  Classes: 14 (perfect match with ChestX-ray14)")
print(f"  Parameters: {teacher.get_num_params():,}")
print(f"  Size: {teacher.get_model_size_mb():.1f} MB")

# Test forward pass
print("\nTesting forward pass...")
teacher.eval()
with torch.no_grad():
    x = torch.randn(4, 3, 224, 224)
    logits = teacher(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output range: [{logits.min():.3f}, {logits.max():.3f}]")

# Test predictions
print("\nSample prediction (untrained image):")
probs = torch.sigmoid(logits[0])
print(f"  Probabilities: min={probs.min():.3f}, max={probs.max():.3f}, mean={probs.mean():.3f}")

print("\n✓ CheXNet teacher model is ready for Knowledge Distillation!")
print("=" * 80)
