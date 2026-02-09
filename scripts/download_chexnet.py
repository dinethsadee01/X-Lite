"""Download and inspect CheXNet weights"""
import torch
import urllib.request
from pathlib import Path

# Setup paths
weights_dir = Path('data/weights/chexnet')
weights_dir.mkdir(parents=True, exist_ok=True)
weights_path = weights_dir / 'model.pth.tar'

# Download if not exists
if not weights_path.exists():
    print("Downloading CheXNet weights...")
    url = 'https://github.com/arnoweng/CheXNet/raw/master/model.pth.tar'
    urllib.request.urlretrieve(url, weights_path)
    print(f"✓ Downloaded to {weights_path}")
else:
    print(f"✓ CheXNet weights already exist at {weights_path}")

# Inspect structure
print("\nInspecting checkpoint structure...")
try:
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
except TypeError:
    checkpoint = torch.load(weights_path, map_location='cpu')

print(f"Checkpoint type: {type(checkpoint)}")
if isinstance(checkpoint, dict):
    print(f"Keys: {checkpoint.keys()}")
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    print(f"\nModel parameters (first 20):")
    for i, (name, tensor) in enumerate(list(state_dict.items())[:20]):
        print(f"  {name:<60} {list(tensor.shape)}")
        if i >= 19:
            print(f"  ... ({len(state_dict)} total parameters)")
            break
    
    # Check classifier
    print("\nClassifier layers:")
    for name, tensor in state_dict.items():
        if 'classifier' in name:
            print(f"  {name:<60} {list(tensor.shape)}")

print("\n✓ CheXNet checkpoint ready for use!")
