"""
Prediction Service
Orchestrates model inference and result formatting
"""

from pathlib import Path
import sys
import time
import torch
from typing import Dict, List, Optional
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import Config, DISEASE_LABELS, DISEASE_DESCRIPTIONS, get_risk_level, get_risk_color
from ml.data.preprocessing import get_medical_transforms
from ml.models.student_model import create_student_model, MODEL_CONFIGS

DEFAULT_MODEL_ARCH = "convnext_tiny_mhsa"
DEFAULT_MODEL_NAME = "X-Lite model 001"
DEFAULT_MODEL_PATH = Config.CHECKPOINT_DIR / "final" / "X-Lite_model_001.pth"


class PredictionService:
    """Service for running model inference"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize prediction service
        
        Args:
            model_path: Path to model checkpoint (optional)
        """
        self.model = None
        self.model_arch = DEFAULT_MODEL_ARCH
        self.model_name = DEFAULT_MODEL_NAME
        self.device = self._get_device()
        self.transform = get_medical_transforms(use_clahe=True, use_denoising=False)

        if model_path:
            self.load_model(model_path)
        else:
            if DEFAULT_MODEL_PATH.exists():
                self.load_model(
                    str(DEFAULT_MODEL_PATH),
                    model_arch=DEFAULT_MODEL_ARCH,
                    model_display_name=DEFAULT_MODEL_NAME
                )
            else:
                default_ckpt = self._find_default_checkpoint()
                if default_ckpt is not None:
                    self.load_model(str(default_ckpt))
    
    def _get_device(self) -> torch.device:
        """Get computation device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _find_default_checkpoint(self) -> Optional[Path]:
        """Find the best available checkpoint for inference."""
        checkpoint_dir = Config.CHECKPOINT_DIR
        if not checkpoint_dir.exists():
            return None

        best_candidates = sorted(checkpoint_dir.rglob("best_checkpoint.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if best_candidates:
            return best_candidates[0]

        last_candidates = sorted(checkpoint_dir.rglob("last_checkpoint.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if last_candidates:
            return last_candidates[0]

        return None

    def _infer_model_arch(self, model_path: Path) -> Optional[str]:
        """Infer model architecture from checkpoint path."""
        if model_path.parent.name in MODEL_CONFIGS:
            return model_path.parent.name
        return None

    def load_model(
        self,
        model_path: str,
        model_arch: Optional[str] = None,
        model_display_name: Optional[str] = None
    ):
        """
        Load trained model from checkpoint
        
        Args:
            model_path: Path to model checkpoint
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        inferred_arch = self._infer_model_arch(model_path)
        self.model_arch = model_arch or inferred_arch or self.model_arch
        self.model_name = model_display_name or inferred_arch or self.model_name

        model = create_student_model(self.model_arch, num_classes=Config.NUM_CLASSES, pretrained=False)

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)

        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)

        self.model = model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        image_path: str,
        return_heatmap: bool = True,
        threshold: float = 0.5
    ) -> Dict:
        """
        Run prediction on chest X-ray image
        
        Args:
            image_path: Path to image
            return_heatmap: Whether to generate Grad-CAM heatmap
            threshold: Confidence threshold
        
        Returns:
            dict: Prediction results
        """
        start_time = time.time()
        
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please check checkpoint availability.")

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        probabilities = probs
        
        # Format predictions
        predictions = []
        positive_findings = []
        
        for i, (disease, prob) in enumerate(zip(DISEASE_LABELS, probabilities)):
            risk = get_risk_level(prob)
            color = get_risk_color(prob)
            
            pred_result = {
                "disease": disease,
                "probability": round(prob, 4),
                "risk_level": risk,
                "color": color,
                "description": DISEASE_DESCRIPTIONS.get(disease, "")
            }
            
            predictions.append(pred_result)
            
            if prob >= threshold:
                positive_findings.append(disease)
        
        # Processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Heatmap path (placeholder)
        heatmap_path = None
        if return_heatmap:
            # TODO: Generate actual Grad-CAM heatmap
            heatmap_path = "/static/heatmaps/dummy_heatmap.png"
        
        return {
            "success": True,
            "predictions": predictions,
            "positive_findings": positive_findings,
            "num_positive": len(positive_findings),
            "heatmap_path": heatmap_path,
            "processing_time_ms": round(processing_time, 2),
            "model_name": self.model_name
        }
