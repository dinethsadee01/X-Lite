"""
Quick start script to verify backend is working
Run this before starting the full application
"""

import requests
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config

def check_backend():
    """Check if backend is running and responding"""
    try:
        print("Checking backend health...")
        response = requests.get('http://localhost:8000/api/health', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Backend is running!")
            print(f"  Status: {data['status']}")
            print(f"  Version: {data['version']}")
            print(f"  Device: {data['system']['device']}")
            print(f"  CUDA Available: {data['system']['cuda_available']}")
            return True
        else:
            print(f"✗ Backend returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Backend is not running!")
        print("  Please start the backend with: python -m uvicorn backend.app:app --reload")
        return False
    except Exception as e:
        print(f"✗ Error checking backend: {e}")
        return False

def check_frontend_build():
    """Check if frontend dependencies are installed"""
    frontend_path = Path(__file__).parent / 'frontend'
    node_modules = frontend_path / 'node_modules'
    
    if node_modules.exists():
        print("✓ Frontend dependencies are installed")
        return True
    else:
        print("✗ Frontend dependencies not installed")
        print("  Please run: cd frontend && npm install")
        return False

def main():
    print("=" * 50)
    print("X-Lite Quick Start Verification")
    print("=" * 50)
    print()
    
    # Create necessary directories
    print("Setting up directories...")
    Config.create_directories()
    print("✓ Directories ready")
    print()
    
    # Check backend
    backend_ok = check_backend()
    print()
    
    # Check frontend
    frontend_ok = check_frontend_build()
    print()
    
    if backend_ok and frontend_ok:
        print("=" * 50)
        print("✓ Everything is ready!")
        print("Open http://localhost:3000 in your browser")
        print("=" * 50)
        return 0
    else:
        print("=" * 50)
        print("✗ Please fix the issues above before starting")
        print("=" * 50)
        return 1

if __name__ == "__main__":
    sys.exit(main())
