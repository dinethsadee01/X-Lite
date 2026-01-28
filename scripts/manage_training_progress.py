"""
Training Progress Manager
==========================
Helper script to manage baseline training progress.

Usage:
  python scripts/manage_training_progress.py --status     # Show current progress
  python scripts/manage_training_progress.py --reset      # Reset all progress
  python scripts/manage_training_progress.py --remove MODEL_NAME  # Remove specific model
"""

import sys
import json
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.training_utils import (
    load_training_progress,
    save_training_progress,
    reset_training_progress
)
from ml.models.student_model import MODEL_CONFIGS


def show_status():
    """Show current training progress"""
    completed = load_training_progress()
    all_models = list(MODEL_CONFIGS.keys())
    remaining = [m for m in all_models if m not in completed]
    
    print("\n" + "="*70)
    print("BASELINE TRAINING PROGRESS")
    print("="*70)
    
    if completed:
        print(f"\n✅ Completed ({len(completed)}/{len(all_models)}):")
        for model in sorted(completed):
            print(f"   • {model}")
    else:
        print(f"\n⚠️  No models completed yet")
    
    if remaining:
        print(f"\n⏳ Remaining ({len(remaining)}/{len(all_models)}):")
        for model in sorted(remaining):
            print(f"   • {model}")
    else:
        print(f"\n✅ All models completed!")
    
    print("="*70)
    
    # Show results file status
    results_path = project_root / "experiments" / "baseline_results.csv"
    if results_path.exists():
        import pandas as pd
        df = pd.read_csv(results_path)
        print(f"\n📊 Results file: {results_path}")
        print(f"   Contains {len(df)} model(s)")
    else:
        print(f"\n⚠️  No results file found yet")
    
    print()


def remove_model(model_name):
    """Remove a specific model from completed list"""
    completed = load_training_progress()
    
    if model_name not in MODEL_CONFIGS:
        print(f"\n✗ Error: '{model_name}' is not a valid model name")
        print(f"\nValid models: {', '.join(MODEL_CONFIGS.keys())}")
        return
    
    if model_name in completed:
        completed.remove(model_name)
        save_training_progress(completed)
        print(f"\n✅ Removed '{model_name}' from completed list")
        print(f"   It will be retrained on next run")
    else:
        print(f"\n⚠️  '{model_name}' was not in completed list (nothing to remove)")


def main():
    parser = argparse.ArgumentParser(
        description="Manage baseline training progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/manage_training_progress.py --status
  python scripts/manage_training_progress.py --reset
  python scripts/manage_training_progress.py --remove efficientnet_b0_mhsa
        """
    )
    
    parser.add_argument('--status', action='store_true',
                       help='Show current training progress')
    parser.add_argument('--reset', action='store_true',
                       help='Reset all progress (retrain all models)')
    parser.add_argument('--remove', metavar='MODEL_NAME',
                       help='Remove specific model from completed list')
    
    args = parser.parse_args()
    
    if args.status or (not args.reset and not args.remove):
        show_status()
    elif args.reset:
        reset_training_progress()
        print("\n✅ Training progress reset. All models will be retrained on next run.\n")
    elif args.remove:
        remove_model(args.remove)


if __name__ == "__main__":
    main()
