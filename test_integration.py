"""
Frontend & Backend Integration Test
Quick verification that everything is connected and working
"""

import requests
import json
import time
import sys
from pathlib import Path

# Colors for console output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def check_backend_health():
    """Check if backend is running and healthy"""
    print_header("Checking Backend Health")
    
    try:
        response = requests.get('http://localhost:8000/api/health', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Backend is running!")
            print(f"  Status: {data['status']}")
            print(f"  Version: {data['version']}")
            print(f"  Device: {data['system']['device']}")
            print(f"  CUDA Available: {data['system']['cuda_available']}")
            print(f"  Classes: {data['config']['num_classes']}")
            return True
        else:
            print_error(f"Backend returned status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to backend at http://localhost:8000")
        print_warning("Make sure backend is running: python -m uvicorn backend.app:app --reload")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False

def check_upload_endpoint():
    """Test the upload endpoint"""
    print_header("Testing Upload Endpoint")
    
    try:
        # Create a simple test image (1x1 pixel PNG)
        import io
        from PIL import Image
        
        # Create test image
        img = Image.new('RGB', (224, 224), color='gray')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Try upload
        files = {'file': ('test_image.png', img_bytes, 'image/png')}
        response = requests.post(
            'http://localhost:8000/api/upload',
            files=files,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print_success("Image upload working!")
                print(f"  Filename: {data['data']['filename']}")
                print(f"  File Size: {data['data']['file_size_mb']} MB")
                return data['data']['filename']
            else:
                print_error(f"Upload returned error: {data}")
                return None
        else:
            print_error(f"Upload returned status: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
            
    except ImportError:
        print_warning("PIL not installed, skipping image creation test")
        print_warning("Install with: pip install Pillow")
        return None
    except Exception as e:
        print_error(f"Upload test failed: {e}")
        return None

def check_predict_endpoint(filename):
    """Test the predict endpoint"""
    if not filename:
        print_header("Skipping Prediction Test (no valid image)")
        return False
        
    print_header("Testing Prediction Endpoint")
    
    try:
        payload = {
            "filename": filename,
            "return_heatmap": True,
            "confidence_threshold": 0.5
        }
        
        response = requests.post(
            'http://localhost:8000/api/predict',
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print_success("Prediction working!")
                print(f"  Model: {data.get('model_name', 'N/A')}")
                print(f"  Processing Time: {data.get('processing_time_ms', 0):.2f}ms")
                print(f"  Findings: {data.get('num_positive', 0)} condition(s)")
                print(f"  Predictions returned: {len(data.get('predictions', []))}")
                
                # Show top predictions
                predictions = sorted(
                    data.get('predictions', []),
                    key=lambda x: x['probability'],
                    reverse=True
                )[:3]
                
                if predictions:
                    print("\n  Top Predictions:")
                    for pred in predictions:
                        prob = pred['probability'] * 100
                        print(f"    • {pred['disease']}: {prob:.1f}% ({pred['risk_level']})")
                
                return True
            else:
                print_error(f"Prediction returned error: {data}")
                return False
        else:
            print_error(f"Prediction returned status: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_frontend():
    """Check if frontend is accessible"""
    print_header("Checking Frontend")
    
    try:
        response = requests.get('http://localhost:3000', timeout=5)
        if response.status_code == 200:
            print_success("Frontend is running at http://localhost:3000")
            return True
        else:
            print_warning(f"Frontend returned status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_warning("Cannot connect to frontend at http://localhost:3000")
        print_warning("Make sure frontend is running: cd frontend && npm start")
        return False
    except Exception as e:
        print_warning(f"Could not check frontend: {e}")
        return False

def main():
    print(f"\n{Colors.BLUE}")
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         X-Lite Backend & Frontend Integration Test         ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    print(Colors.END)
    
    results = {}
    
    # Test backend health
    results['backend_health'] = check_backend_health()
    
    if not results['backend_health']:
        print_header("Cannot Proceed")
        print_error("Backend is not running. Please start it first:")
        print("  python -m uvicorn backend.app:app --reload")
        return 1
    
    # Test upload
    filename = check_upload_endpoint()
    results['upload'] = filename is not None
    
    # Test predict
    if filename:
        results['predict'] = check_predict_endpoint(filename)
    else:
        print_header("Skipping Prediction Test")
        print_warning("Upload test didn't return a valid filename")
        results['predict'] = False
    
    # Check frontend (optional)
    results['frontend'] = check_frontend()
    
    # Summary
    print_header("Test Summary")
    
    tests = [
        ("Backend Health", results.get('backend_health', False)),
        ("Image Upload", results.get('upload', False)),
        ("Predictions", results.get('predict', False)),
        ("Frontend", results.get('frontend', False)),
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        if result:
            print_success(f"{test_name}: OK")
        else:
            print_warning(f"{test_name}: Not available")
    
    print(f"\n{Colors.BLUE}Passed: {passed}/{total}{Colors.END}\n")
    
    if passed >= 3:
        print_success("System is ready for use!")
        print(f"\nAccess the application at: {Colors.BLUE}http://localhost:3000{Colors.END}\n")
        return 0
    else:
        print_error("Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.END}\n")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
