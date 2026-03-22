"""
Count parameters from a PyTorch checkpoint.

Supports:
- Full nn.Module checkpoints (best accuracy for trainable/non-trainable split)
- Dict checkpoints with keys like model_state_dict / student_state_dict / state_dict
- Raw state_dict checkpoints

Examples:
  python scripts/count_checkpoint_params.py --checkpoint ml/models/checkpoints/kd_student_best.pth

    # Exact mode with your student model factory
    python scripts/count_checkpoint_params.py \
        --checkpoint ml/models/checkpoints/kd_student_best.pth \
        --architecture efficientnet_b0_performer \
        --num-classes 15

    # Fully custom factory kwargs (advanced)
    python scripts/count_checkpoint_params.py \
        --checkpoint ml/models/checkpoints/efficientnet_best.pth \
        --model-module ml.models.student_model \
        --model-factory create_student_model \
        --model-kwargs '{"architecture":"efficientnet_b0_performer","num_classes":15,"pretrained":false}'
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _count_from_model(model: nn.Module) -> tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def _find_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model_state_dict", "student_state_dict", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key], key
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj, "raw_state_dict"
    return None, None


def _count_from_state_dict(state_dict: dict) -> int:
    # Heuristic exclusion for common BN buffers if model class is unavailable.
    buffer_suffixes = (
        "running_mean",
        "running_var",
        "num_batches_tracked",
    )
    total = 0
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if name.endswith(buffer_suffixes):
            continue
        total += tensor.numel()
    return total


def _build_model(module_name: str, factory_name: str, model_kwargs: dict) -> nn.Module:
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    model = factory(**model_kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Factory '{factory_name}' from '{module_name}' did not return nn.Module."
        )
    return model


def _extract_architecture_from_checkpoint(checkpoint: object):
    if not isinstance(checkpoint, dict):
        return None
    for key in ("architecture", "model_name", "model_arch", "arch"):
        value = checkpoint.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def main():
    parser = argparse.ArgumentParser(description="Count parameters from a PyTorch checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument(
        "--model-module",
        default="ml.models.student_model",
        help="Python module containing model factory (optional)",
    )
    parser.add_argument(
        "--model-factory",
        default="create_student_model",
        help="Factory function name to build model (optional)",
    )
    parser.add_argument(
        "--architecture",
        default=None,
        help="Model architecture name used by create_student_model (recommended)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=14,
        help="Number of output classes for model reconstruction",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Build reconstruction model with pretrained=True (default False)",
    )
    parser.add_argument(
        "--model-kwargs",
        default="{}",
        help="JSON kwargs for model factory (overrides --architecture flags)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    print("=" * 70)
    print("CHECKPOINT PARAMETER REPORT")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")

    # Case 1: Serialized full model
    if isinstance(checkpoint, nn.Module):
        total, trainable, non_trainable = _count_from_model(checkpoint)
        print("Format: full nn.Module")
        print(f"Total params:        {total:,}")
        print(f"Trainable params:    {trainable:,}")
        print(f"Non-trainable params:{non_trainable:,}")
        print(f"Approx size (fp32):  {total * 4 / (1024 ** 2):.2f} MB")
        return

    # Case 2/3: state_dict-style checkpoint
    state_dict, state_key = _find_state_dict(checkpoint)
    if state_dict is None:
        raise ValueError(
            "Unsupported checkpoint format. Provide --model-module/--model-factory to "
            "reconstruct model, or use a checkpoint with a recognized state_dict key."
        )

    print(f"Format: dict ({state_key})")

    # If user provides or defaults to model constructor, we can report exact split.
    if args.model_module and args.model_factory:
        if args.model_kwargs.strip() != "{}":
            model_kwargs = json.loads(args.model_kwargs)
        else:
            inferred_arch = args.architecture or _extract_architecture_from_checkpoint(checkpoint)
            if inferred_arch is None:
                # No architecture metadata available; fall back to estimate mode.
                total = _count_from_state_dict(state_dict)
                print("Mode: estimate (state_dict only)")
                print(f"Estimated params:    {total:,}")
                print(f"Approx size (fp32):  {total * 4 / (1024 ** 2):.2f} MB")
                print(
                    "Note: pass --architecture (e.g. efficientnet_b0_performer) "
                    "for exact trainable/non-trainable counts."
                )
                return

            model_kwargs = {
                "architecture": inferred_arch,
                "num_classes": args.num_classes,
                "pretrained": args.pretrained,
            }

        model = _build_model(args.model_module, args.model_factory, model_kwargs)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        total, trainable, non_trainable = _count_from_model(model)

        print("Mode: exact (model reconstructed)")
        print(f"Architecture:        {model_kwargs.get('architecture', 'n/a')}")
        print(f"Total params:        {total:,}")
        print(f"Trainable params:    {trainable:,}")
        print(f"Non-trainable params:{non_trainable:,}")
        print(f"Approx size (fp32):  {total * 4 / (1024 ** 2):.2f} MB")
        if missing:
            print(f"Missing keys:        {len(missing)}")
        if unexpected:
            print(f"Unexpected keys:     {len(unexpected)}")
        return

    # Fallback without model class: best-effort total only.
    total = _count_from_state_dict(state_dict)
    print("Mode: estimate (state_dict only)")
    print(f"Estimated params:    {total:,}")
    print(f"Approx size (fp32):  {total * 4 / (1024 ** 2):.2f} MB")
    print("Note: trainable/non-trainable split needs model reconstruction.")


if __name__ == "__main__":
    main()