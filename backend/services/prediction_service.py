"""
Prediction Service
Orchestrates model inference and result formatting
"""

from pathlib import Path
import sys
import time
import torch
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import Config, DISEASE_LABELS, DISEASE_DESCRIPTIONS, get_risk_level, get_risk_color


class PredictionService:
    """Service for running model inference"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize prediction service
        
        Args:
            model_path: Path to model checkpoint (optional)
        """
        self.model = None
        self.model_name = "student_model"  # Default
        self.device = self._get_device()
        
        if model_path:
            self.load_model(model_path)
    
    def _get_device(self) -> torch.device:
        """Get computation device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def load_model(self, model_path: str):
        """
        Load trained model from checkpoint
        
        Args:
            model_path: Path to model checkpoint
        """
        # TODO: Implement actual model loading
        # For now, this is a placeholder
        print(f"Loading model from: {model_path}")
        self.model_name = Path(model_path).stem
        # self.model = load_student_model(model_path)
        # self.model.to(self.device)
        # self.model.eval()
    
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
        
        # TODO: Implement actual inference
        # For now, return dummy predictions for demonstration
        
        # Dummy predictions (replace with actual model inference)
        import random
        random.seed(42)
        
        probabilities = [random.random() * 0.8 for _ in range(len(DISEASE_LABELS))]
        
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
