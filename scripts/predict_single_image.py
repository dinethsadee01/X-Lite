"""
Single Image Prediction (Standalone)
====================================
Run prediction for one image using the Phase 1 best checkpoint, without backend services.

Usage:
    python scripts/predict_single_image.py --image_path "C:/path/to/image.png"
    python scripts/predict_single_image.py --image_path "C:/path/to/image.png" --use_optimal_thresholds
    python scripts/predict_single_image.py --image_path "C:/path/to/image.png" --top_k 8
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import NUM_CLASSES
from config.disease_labels import DISEASE_LABELS
from ml.data.preprocessing import get_medical_transforms
from ml.models.student_model import create_student_model


MODEL_NAME = "efficientnet_b0_performer"
CHECKPOINT_PATH = project_root / "ml" / "models" / "checkpoints" / "efficientnet_b0_performer_full_dataset_15class" / "best_checkpoint.pth"
OPTIMAL_THRESHOLDS_PATH = project_root / "scripts" / "optimal_thresholds.json"
DEFAULT_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Predict diseases for a single chest X-ray image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Default threshold if optimal thresholds are not used")
    parser.add_argument("--use_optimal_thresholds", action="store_true", help="Use per-disease thresholds from scripts/optimal_thresholds.json")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top probabilities to display")
    return parser.parse_args()


def load_optimal_thresholds(enabled: bool):
    if not enabled:
        return {}

    if not OPTIMAL_THRESHOLDS_PATH.exists():
        print(f"Warning: optimal thresholds file not found: {OPTIMAL_THRESHOLDS_PATH}")
        print("Falling back to fixed threshold.")
        return {}

    with open(OPTIMAL_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    return {k: v for k, v in thresholds.items() if k in DISEASE_LABELS}


def load_model(device: torch.device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = create_student_model(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "student_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["student_state_dict"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print("SINGLE IMAGE PREDICTION (STANDALONE)")
    print("=" * 80)
    print(f"Image: {image_path}")
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    optimal_thresholds = load_optimal_thresholds(args.use_optimal_thresholds)
    threshold_mode = "Optimal Per-Disease" if optimal_thresholds else f"Fixed {args.threshold:.2f}"
    print(f"Threshold Mode: {threshold_mode}")

    model = load_model(device)
    print("✓ Model loaded")

    # For a raw image path, apply same preprocessing style as inference path.
    transform = get_medical_transforms(use_clahe=True, use_denoising=False)

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Build prediction details
    per_label = []
    predicted_positive = []

    for i, label in enumerate(DISEASE_LABELS):
        threshold = optimal_thresholds.get(label, args.threshold)
        prob = float(probs[i])
        is_positive = prob >= threshold

        per_label.append(
            {
                "disease": label,
                "probability": prob,
                "threshold": float(threshold),
                "is_positive": bool(is_positive),
            }
        )

        if is_positive:
            predicted_positive.append((label, prob, threshold))

    per_label_sorted = sorted(per_label, key=lambda x: x["probability"], reverse=True)

    print("\nPredicted Positive Findings:")
    if not predicted_positive:
        print("  No Finding")
    else:
        for label, prob, threshold in sorted(predicted_positive, key=lambda x: x[1], reverse=True):
            print(f"  - {label:<25} prob={prob:.4f}  threshold={threshold:.2f}")

    print(f"\nTop {args.top_k} Probabilities:")
    for item in per_label_sorted[: args.top_k]:
        marker = "!" if item["is_positive"] else " "
        print(
            f"  [{marker}] {item['disease']:<25} "
            f"prob={item['probability']:.4f}  th={item['threshold']:.2f}"
        )

    print("\nAll Class Probabilities:")
    for item in per_label:
        marker = "!" if item["is_positive"] else " "
        print(
            f"  [{marker}] {item['disease']:<25} "
            f"prob={item['probability']:.4f}  th={item['threshold']:.2f}"
        )

    print("\nLegend: [!] = Predicted positive at threshold")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
