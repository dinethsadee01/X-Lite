"""
Quick Model Loading Test
========================
Tests all 6 hybrid CNN-Transformer models to verify they load correctly before training.
Does a forward pass with a dummy batch to catch dimension mismatches early.

Usage:
    python scripts/test_model_loading.py

Output:
    - Reports which models load successfully
    - Shows output dimensions
    - Flags any errors before training starts
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from ml.models.student_model import create_student_model, MODEL_CONFIGS


def test_single_model(model_name: str, device: torch.device, verbose: bool = True) -> bool:
    """
    Test loading and forward pass for a single model
    
    Args:
        model_name (str): Model name from MODEL_CONFIGS
        device (torch.device): Device to load on
        verbose (bool): Print details
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing: {model_name}")
            print(f"{'='*60}")
        
        # Load model
        if verbose:
            print(f"  Loading model...", end=" ")
        model = create_student_model(model_name, num_classes=14, pretrained=True)
        model = model.to(device)
        model.eval()
        if verbose:
            print("✓")
        
        # Get model info
        num_params = model.get_num_params()
        model_size = model.get_model_size_mb()
        if verbose:
            print(f"  Parameters: {num_params:,}")
            print(f"  Model Size: {model_size:.2f} MB")
        
        # Forward pass with dummy input
        if verbose:
            print(f"  Forward pass (1 image)...", end=" ")
        
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        if verbose:
            print("✓")
            print(f"  Input Shape:  {list(dummy_input.shape)}")
            print(f"  Output Shape: {list(output.shape)}")
            print(f"  Output Range: [{output.min():.4f}, {output.max():.4f}]")
        
        # Verify output shape
        assert output.shape == (1, 14), f"Expected (1, 14), got {output.shape}"
        
        if verbose:
            print(f"  ✅ PASS: {model_name}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"\n  ❌ FAIL: {model_name}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
        return False


def main():
    """Test all 6 models"""
    
    print("\n" + "=" * 60)
    print("HYBRID CNN-TRANSFORMER MODEL LOADING TEST")
    print("=" * 60)
    print(f"\nTesting {len(MODEL_CONFIGS)} models for:")
    print("  ✓ Successfully load from timm")
    print("  ✓ Correct feature dimension extraction")
    print("  ✓ Forward pass produces correct output shape")
    print("  ✓ No dimension mismatches")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test each model
    results = {}
    for model_name in MODEL_CONFIGS.keys():
        results[model_name] = test_single_model(model_name, device, verbose=True)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for model_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {model_name}")
    
    print(f"\n  Result: {passed}/{total} models passed")
    
    if passed == total:
        print("\n✅ ALL MODELS LOADING SUCCESSFULLY")
        print("   Safe to proceed with baseline training: python scripts/train_baseline.py")
        return 0
    else:
        print(f"\n❌ {total - passed} MODEL(S) FAILED LOADING")
        print("   Fix issues before running training")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
