"""
X-Lite Deployment Feasibility Benchmark
========================================
Measures inference time, memory footprint, model size, and CPU compatibility
for the finalized EfficientNet-B0 + Performer model.

Outputs a structured JSON report to: experiments/deployment_feasibility.json
"""

import sys
import os
import time
import json
import platform
import psutil
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from PIL import Image

from config import Config
from config.disease_labels14 import DISEASE_LABELS14, NUM_CLASSES14
from ml.data.preprocessing import get_medical_transforms
from ml.models.student_model import create_student_model
from backend.services.gradcam_service import generate_gradcam, save_heatmap_overlay


# ── Configuration ──────────────────────────────────────────────────────
MODEL_ARCH = "efficientnet_b0_performer"
MODEL_PATH = (
    ROOT / "ml" / "models" / "new checkpoints fix"
    / "efficientnet_b0_performer_full_dataset_14class_v2" / "best_checkpoint.pth"
)
THRESHOLDS_PATH = ROOT / "scripts" / "optimal_thresholds14_fixed_v2.json"

# Use an existing uploaded image as test input
TEST_IMAGE = ROOT / "backend" / "uploads" / "20260328_135340_30c37aefb5e1.png"

NUM_WARMUP = 5       # Warmup runs (excluded from timing)
NUM_BENCHMARK = 30   # Timed runs for averaging


def get_system_info() -> dict:
    """Collect system hardware/software information."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_name": platform.machine(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 2)
    return info


def get_model_size(model_path: Path) -> dict:
    """Get model file size and parameter count."""
    # File size
    file_size_bytes = model_path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    # Parameter count
    model = create_student_model(MODEL_ARCH, num_classes=NUM_CLASSES14, pretrained=False)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "total_parameters_millions": round(total_params / 1e6, 2),
    }


def load_model(device: torch.device):
    """Load model and return it with load time."""
    start = time.perf_counter()
    model = create_student_model(MODEL_ARCH, num_classes=NUM_CLASSES14, pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
    model.eval()
    load_time = (time.perf_counter() - start) * 1000
    return model, load_time


def benchmark_device(device: torch.device, device_label: str) -> dict:
    """Run full benchmark on a specific device (CPU or GPU)."""
    print(f"\n{'='*60}")
    print(f"  Benchmarking on: {device_label} ({device})")
    print(f"{'='*60}")

    # ── Model Loading ──
    model, load_time_ms = load_model(device)
    print(f"  Model load time: {load_time_ms:.1f} ms")

    # ── Memory before inference ──
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        gpu_mem_before = torch.cuda.memory_allocated() / (1024 * 1024)

    # ── Preprocessing ──
    transform = get_medical_transforms(use_clahe=True, use_denoising=False)
    original_image = Image.open(TEST_IMAGE).convert('RGB')

    preprocess_times = []
    for _ in range(NUM_BENCHMARK):
        t0 = time.perf_counter()
        tensor = transform(original_image).unsqueeze(0).to(device)
        preprocess_times.append((time.perf_counter() - t0) * 1000)

    # ── Inference Only (no Grad-CAM) ──
    # Warmup
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = model(tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    inference_times = []
    for _ in range(NUM_BENCHMARK):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        inference_times.append((time.perf_counter() - t0) * 1000)

    # ── Grad-CAM Generation ──
    gradcam_times = []
    top_class = max(range(len(probs)), key=lambda i: probs[i])
    for _ in range(NUM_BENCHMARK):
        t0 = time.perf_counter()
        cam = generate_gradcam(model, tensor, top_class, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        gradcam_times.append((time.perf_counter() - t0) * 1000)

    # ── Full Pipeline (preprocess + inference + grad-cam) ──
    full_times = []
    for _ in range(NUM_BENCHMARK):
        t0 = time.perf_counter()
        t = transform(original_image).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t)
            _ = torch.sigmoid(out).squeeze(0).cpu().tolist()
        _ = generate_gradcam(model, t, top_class, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        full_times.append((time.perf_counter() - t0) * 1000)

    # ── Memory after inference ──
    mem_after = process.memory_info().rss / (1024 * 1024)

    result = {
        "device": device_label,
        "device_type": str(device),
        "model_load_time_ms": round(load_time_ms, 2),
        "preprocessing": {
            "mean_ms": round(np.mean(preprocess_times), 2),
            "std_ms": round(np.std(preprocess_times), 2),
            "min_ms": round(np.min(preprocess_times), 2),
            "max_ms": round(np.max(preprocess_times), 2),
        },
        "inference_only": {
            "mean_ms": round(np.mean(inference_times), 2),
            "std_ms": round(np.std(inference_times), 2),
            "min_ms": round(np.min(inference_times), 2),
            "max_ms": round(np.max(inference_times), 2),
        },
        "gradcam": {
            "mean_ms": round(np.mean(gradcam_times), 2),
            "std_ms": round(np.std(gradcam_times), 2),
            "min_ms": round(np.min(gradcam_times), 2),
            "max_ms": round(np.max(gradcam_times), 2),
        },
        "full_pipeline": {
            "mean_ms": round(np.mean(full_times), 2),
            "std_ms": round(np.std(full_times), 2),
            "min_ms": round(np.min(full_times), 2),
            "max_ms": round(np.max(full_times), 2),
        },
        "memory": {
            "ram_before_mb": round(mem_before, 2),
            "ram_after_mb": round(mem_after, 2),
            "ram_delta_mb": round(mem_after - mem_before, 2),
        },
        "meets_nfr03_2s_target": bool(round(np.mean(full_times), 2) < 2000),
        "num_warmup_runs": NUM_WARMUP,
        "num_benchmark_runs": NUM_BENCHMARK,
    }

    if device.type == 'cuda':
        gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        result["memory"]["gpu_allocated_before_mb"] = round(gpu_mem_before, 2)
        result["memory"]["gpu_peak_mb"] = round(gpu_peak, 2)

    # Print summary
    print(f"  Preprocessing:    {result['preprocessing']['mean_ms']:.1f} ± {result['preprocessing']['std_ms']:.1f} ms")
    print(f"  Inference only:   {result['inference_only']['mean_ms']:.1f} ± {result['inference_only']['std_ms']:.1f} ms")
    print(f"  Grad-CAM:         {result['gradcam']['mean_ms']:.1f} ± {result['gradcam']['std_ms']:.1f} ms")
    print(f"  Full pipeline:    {result['full_pipeline']['mean_ms']:.1f} ± {result['full_pipeline']['std_ms']:.1f} ms")
    print(f"  RAM usage delta:  {result['memory']['ram_delta_mb']:.1f} MB")
    print(f"  NFR03 (<2s):      {'✅ PASS' if result['meets_nfr03_2s_target'] else '❌ FAIL'}")

    return result


def main():
    print("=" * 60)
    print("  X-Lite Deployment Feasibility Benchmark")
    print("=" * 60)

    # System info
    sys_info = get_system_info()
    print(f"\n  System: {sys_info['platform']}")
    print(f"  CPU: {sys_info['processor']} ({sys_info['cpu_cores_physical']}P/{sys_info['cpu_cores_logical']}L cores)")
    print(f"  RAM: {sys_info['ram_total_gb']} GB")
    if sys_info['cuda_available']:
        print(f"  GPU: {sys_info['gpu_name']} ({sys_info['gpu_vram_gb']} GB VRAM)")
    print(f"  PyTorch: {sys_info['pytorch_version']}")
    print(f"  Test image: {TEST_IMAGE.name}")

    # Model size
    model_info = get_model_size(MODEL_PATH)
    print(f"\n  Model: {MODEL_ARCH}")
    print(f"  Parameters: {model_info['total_parameters_millions']}M")
    print(f"  Checkpoint size: {model_info['file_size_mb']} MB")

    # ── Benchmark on CPU (always) ──
    cpu_result = benchmark_device(torch.device('cpu'), "CPU")

    # ── Benchmark on GPU (if available) ──
    gpu_result = None
    if torch.cuda.is_available():
        gpu_result = benchmark_device(torch.device('cuda'), f"GPU ({torch.cuda.get_device_name(0)})")

    # ── Compile final report ──
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_image": str(TEST_IMAGE.name),
        "system_info": sys_info,
        "model_info": model_info,
        "benchmarks": {
            "cpu": cpu_result,
        },
    }
    if gpu_result:
        report["benchmarks"]["gpu"] = gpu_result

    # ── Summary comparison table ──
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'CPU':>12}", end="")
    if gpu_result:
        print(f" {'GPU':>12}", end="")
    print()
    print(f"  {'-'*54}")
    print(f"  {'Model load (ms)':<30} {cpu_result['model_load_time_ms']:>12.1f}", end="")
    if gpu_result:
        print(f" {gpu_result['model_load_time_ms']:>12.1f}", end="")
    print()
    print(f"  {'Preprocessing (ms)':<30} {cpu_result['preprocessing']['mean_ms']:>12.1f}", end="")
    if gpu_result:
        print(f" {gpu_result['preprocessing']['mean_ms']:>12.1f}", end="")
    print()
    print(f"  {'Inference only (ms)':<30} {cpu_result['inference_only']['mean_ms']:>12.1f}", end="")
    if gpu_result:
        print(f" {gpu_result['inference_only']['mean_ms']:>12.1f}", end="")
    print()
    print(f"  {'Grad-CAM (ms)':<30} {cpu_result['gradcam']['mean_ms']:>12.1f}", end="")
    if gpu_result:
        print(f" {gpu_result['gradcam']['mean_ms']:>12.1f}", end="")
    print()
    print(f"  {'Full pipeline (ms)':<30} {cpu_result['full_pipeline']['mean_ms']:>12.1f}", end="")
    if gpu_result:
        print(f" {gpu_result['full_pipeline']['mean_ms']:>12.1f}", end="")
    print()
    print(f"  {'RAM delta (MB)':<30} {cpu_result['memory']['ram_delta_mb']:>12.1f}", end="")
    if gpu_result:
        print(f" {gpu_result['memory']['ram_delta_mb']:>12.1f}", end="")
    print()
    print(f"  {'NFR03 <2s target':<30} {'✅ PASS' if cpu_result['meets_nfr03_2s_target'] else '❌ FAIL':>12}", end="")
    if gpu_result:
        print(f" {'✅ PASS' if gpu_result['meets_nfr03_2s_target'] else '❌ FAIL':>12}", end="")
    print()

    # ── Save report ──
    output_path = ROOT / "experiments" / "deployment_feasibility.json"
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {output_path}")


if __name__ == "__main__":
    main()
