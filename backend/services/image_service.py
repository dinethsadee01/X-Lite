"""
Image Service
Handles image upload, validation, and storage
"""

from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import Config


class ImageService:
    """Service for handling chest X-ray image operations"""
    
    def __init__(self):
        self.upload_folder = Config.UPLOAD_FOLDER
        self.allowed_extensions = Config.ALLOWED_EXTENSIONS
        self.max_upload_size = Config.MAX_UPLOAD_SIZE
    
    def validate_image(self, image_path: Path) -> tuple[bool, str]:
        """
        Validate uploaded image
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file size
            file_size = image_path.stat().st_size
            if file_size > self.max_upload_size:
                return False, f"File too large. Max size: {self.max_upload_size / (1024*1024)}MB"
            
            # Try to open and verify image
            with Image.open(image_path) as img:
                img.verify()
            
            # Re-open for format check (verify closes the file)
            with Image.open(image_path) as img:
                # Check dimensions
                width, height = img.size
                if width < 100 or height < 100:
                    return False, "Image too small. Minimum size: 100x100"
                
                if width > 5000 or height > 5000:
                    return False, "Image too large. Maximum size: 5000x5000"
            
            return True, ""
        
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    def get_image_info(self, image_path: Path) -> dict:
        """
        Get image metadata
        
        Args:
            image_path: Path to image
        
        Returns:
            dict: Image metadata
        """
        with Image.open(image_path) as img:
            return {
                "size": img.size,
                "mode": img.mode,
                "format": img.format,
                "width": img.width,
                "height": img.height
            }
