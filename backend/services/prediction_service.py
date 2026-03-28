"""
Prediction Service (14-Class with Grad-CAM)
Orchestrates model inference, preprocessing, and heatmap generation
"""

from pathlib import Path
import sys
import time
import uuid
import torch
import json
from typing import Dict, List, Optional
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import Config, DISEASE_DESCRIPTIONS, get_risk_level, get_risk_color
from config.disease_labels14 import DISEASE_LABELS14, NUM_CLASSES14
from ml.data.preprocessing import get_medical_transforms
from ml.models.student_model import create_student_model, MODEL_CONFIGS
from backend.services.gradcam_service import generate_gradcam, save_heatmap_overlay

DEFAULT_MODEL_ARCH = "efficientnet_b0_performer"
DEFAULT_MODEL_NAME = "X-Lite v2 (14-class)"
DEFAULT_MODEL_PATH = (
    Config.ROOT_DIR / "ml" / "models" / "new checkpoints fix"
    / "efficientnet_b0_performer_full_dataset_14class_v2" / "best_checkpoint.pth"
)
OPTIMAL_THRESHOLDS_PATH = Config.ROOT_DIR / "scripts" / "optimal_thresholds14_fixed_v2.json"

# Heatmap output directory (served as static files)
HEATMAP_DIR = Config.UPLOAD_FOLDER / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)


class PredictionService:
    """Service for running 14-class model inference with Grad-CAM"""
    
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
        # CLAHE enabled — raw uploaded images need contrast enhancement
        # since the model was trained on CLAHE-preprocessed images
        self.transform = get_medical_transforms(use_clahe=True, use_denoising=False)
        self.optimal_thresholds = self._load_optimal_thresholds()
        self.disease_labels = DISEASE_LABELS14
        self.num_classes = NUM_CLASSES14

        if model_path:
            self.load_model(model_path)
        else:
            if not DEFAULT_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Required model checkpoint not found: {DEFAULT_MODEL_PATH}"
                )
            self.load_model(
                str(DEFAULT_MODEL_PATH),
                model_arch=DEFAULT_MODEL_ARCH,
                model_display_name=DEFAULT_MODEL_NAME
            )
    
    def _get_device(self) -> torch.device:
        """Get computation device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _load_optimal_thresholds(self) -> Dict[str, float]:
        """Load optimal per-disease thresholds from JSON file"""
        try:
            if OPTIMAL_THRESHOLDS_PATH.exists():
                with open(OPTIMAL_THRESHOLDS_PATH, 'r') as f:
                    thresholds = json.load(f)
                # Only load thresholds for the 14 disease classes
                loaded = {k: v for k, v in thresholds.items() if k in DISEASE_LABELS14}
                print(f"✓ Loaded {len(loaded)} optimal thresholds from {OPTIMAL_THRESHOLDS_PATH.name}")
                return loaded
            else:
                print(f"⚠ Threshold file not found: {OPTIMAL_THRESHOLDS_PATH}")
                return {}
        except Exception as e:
            print(f"Warning: Could not load optimal thresholds: {e}")
            return {}

    def load_model(
        self,
        model_path: str,
        model_arch: Optional[str] = None,
        model_display_name: Optional[str] = None
    ):
        """Load trained 14-class model from checkpoint"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        self.model_arch = model_arch or self.model_arch
        self.model_name = model_display_name or self.model_name

        # 14-class model
        model = create_student_model(self.model_arch, num_classes=self.num_classes, pretrained=False)

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
        print(f"✓ Model loaded: {self.model_name} ({self.num_classes} classes)")
    
    def predict(
        self,
        image_path: str,
        return_heatmap: bool = True,
        threshold: float = 0.5
    ) -> Dict:
        """
        Run prediction on chest X-ray image with full preprocessing pipeline.
        
        Pipeline: Raw image → CLAHE → Resize → Normalize → Model → Grad-CAM
        
        Args:
            image_path: Path to raw uploaded image
            return_heatmap: Whether to generate Grad-CAM heatmap
            threshold: Default confidence threshold (overridden by optimal thresholds)
        
        Returns:
            dict: Prediction results with heatmap path
        """
        start_time = time.time()
        
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please check checkpoint availability.")

        # Load raw image
        original_image = Image.open(image_path).convert('RGB')
        
        # Preprocess: CLAHE → Resize → Normalize → Tensor
        tensor = self.transform(original_image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        # Format predictions (14 disease classes only — no No_Finding)
        predictions = []
        positive_findings = []
        
        for i, (disease, prob) in enumerate(zip(self.disease_labels, probs)):
            # Use optimal threshold if available, otherwise default
            disease_threshold = self.optimal_thresholds.get(disease, threshold)
            
            risk = get_risk_level(prob).title()
            color = get_risk_color(prob)
            
            pred_result = {
                "disease": disease,
                "probability": round(prob, 4),
                "risk_level": risk,
                "color": color,
                "description": DISEASE_DESCRIPTIONS.get(disease, ""),
                "threshold": round(disease_threshold, 3)
            }
            
            predictions.append(pred_result)
            
            if prob >= disease_threshold:
                positive_findings.append(disease)
        
        # Generate Grad-CAM heatmap
        heatmap_path = None
        heatmap_disease = None
        if return_heatmap and self.model is not None:
            try:
                top_class_idx = max(range(len(probs)), key=lambda i: probs[i])
                
                heatmap_filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
                save_path = HEATMAP_DIR / heatmap_filename
                
                cam = generate_gradcam(self.model, tensor, top_class_idx, self.device)
                save_heatmap_overlay(original_image, cam, save_path, alpha=0.4)
                
                heatmap_path = f"/static/heatmaps/{heatmap_filename}"
                heatmap_disease = self.disease_labels[top_class_idx]
            except Exception as e:
                print(f"Warning: Grad-CAM generation failed: {e}")
                heatmap_path = None
        
        # Processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Overall assessment
        if len(positive_findings) == 0:
            overall_assessment = "No significant findings detected"
        elif len(positive_findings) <= 2:
            overall_assessment = f"Possible findings: {', '.join(positive_findings)}"
        else:
            overall_assessment = f"Multiple findings detected ({len(positive_findings)} conditions)"
        
        return {
            "success": True,
            "predictions": predictions,
            "positive_findings": positive_findings,
            "num_positive": len(positive_findings),
            "overall_assessment": overall_assessment,
            "heatmap_path": heatmap_path,
            "heatmap_target_disease": heatmap_disease,
            "processing_time_ms": round(processing_time, 2),
            "model_name": self.model_name,
            "num_classes": self.num_classes
        }
