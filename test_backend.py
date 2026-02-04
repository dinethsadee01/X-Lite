"""
Quick test to verify backend prediction service is working
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.services.prediction_service import PredictionService
from config import Config

def test_prediction_service():
    """Test if prediction service loads and initializes correctly"""
    print("=" * 60)
    print("Testing X-Lite Backend Prediction Service")
    print("=" * 60)
    
    try:
        # Initialize service
        print("\n1. Initializing PredictionService...")
        service = PredictionService()
        
        # Check if model is loaded
        if service.model is None:
            print("   ✗ FAILED: Model is not loaded")
            return False
        else:
            print(f"   ✓ Model loaded successfully")
            print(f"   ✓ Model name: {service.model_name}")
            print(f"   ✓ Model architecture: {service.model_arch}")
            print(f"   ✓ Device: {service.device}")
        
        # Check model parameters
        print("\n2. Checking model parameters...")
        num_params = sum(p.numel() for p in service.model.parameters())
        print(f"   ✓ Total parameters: {num_params:,}")
        
        # Check if model is in eval mode
        print("\n3. Checking model mode...")
        is_training = service.model.training
        if is_training:
            print("   ✗ WARNING: Model is in training mode")
        else:
            print("   ✓ Model is in eval mode")
        
        # Check upload folder
        print("\n4. Checking upload folder...")
        if Config.UPLOAD_FOLDER.exists():
            print(f"   ✓ Upload folder exists: {Config.UPLOAD_FOLDER}")
        else:
            print(f"   ✗ Upload folder missing: {Config.UPLOAD_FOLDER}")
            return False
        
        print("\n" + "=" * 60)
        print("✓ All checks passed! Backend is ready.")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction_service()
    sys.exit(0 if success else 1)
