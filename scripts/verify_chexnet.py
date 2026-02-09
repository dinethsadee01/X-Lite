"""Verify CheXNet weights availability"""
import urllib.request

url = 'https://github.com/arnoweng/CheXNet/raw/master/model.pth.tar'

try:
    req = urllib.request.Request(url, method='HEAD')
    resp = urllib.request.urlopen(req, timeout=10)
    
    print("✓ CheXNet weights are accessible!")
    print(f"  URL: {url}")
    print(f"  Status: {resp.status}")
    
    size_bytes = int(resp.headers.get('Content-Length', 0))
    size_mb = size_bytes / 1024 / 1024
    print(f"  Size: {size_mb:.1f} MB")
    
    print("\nYou can use CheXNet as your teacher model!")
    
except Exception as e:
    print(f"✗ Cannot access CheXNet weights")
    print(f"  Error: {e}")
    print("\nMay need to use TorchXRayVision instead.")
