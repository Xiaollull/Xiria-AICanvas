import json
import hashlib
import math
from math import ceil
from importlib.util import find_spec
from pathlib import Path

import torch


DEFAULT_PERFORMANCE_SETTINGS = {
    "memory_mode": "auto",
    "attention_backend": "auto",
    "compute_dtype": "fp16",
    "vae_mode": "auto",
    "cuda_math": "balanced",
    "keep_model_cached": True,
    "allow_shared_memory": True,
    "calculate_model_hash": False,
    "staged_vae_decode": False,
    # Inductor compilation of the native Anima transformer blocks. Off by default:
    # it needs Triton, costs a one-off compile per latent shape, and changes the
    # image a fixed Seed produces.
    "compile_transformer": False,
    # Zero means automatic: use real free memory with the platform reserve applied by admission.
    "vram_limit_gb": 0.0,
}

SETTING_CHOICES = {
    "memory_mode": {"auto", "high_vram", "sdxl_balanced", "normal_vram", "low_vram", "ultra_low_vram"},
    # `sage` is Anima-only and is applied by the native attention processor rather
    # than by Diffusers' dispatch, which requires a SageAttention release that has
    # no Windows wheel. Other engines fall back to native SDPA.
    "attention_backend": {"auto", "sdpa", "xformers", "sage", "sliced"},
    "compute_dtype": {"fp16", "bf16"},
    "vae_mode": {"auto", "full", "sliced", "tiled"},
    "cuda_math": {"balanced", "strict"},
}


def normalize_performance_settings(value):
    source = value if isinstance(value, dict) else {}
    normalized = DEFAULT_PERFORMANCE_SETTINGS.copy()
    for key, choices in SETTING_CHOICES.items():
        candidate = source.get(key)
        if isinstance(candidate, str) and candidate in choices:
            normalized[key] = candidate
    for key in ("keep_model_cached", "allow_shared_memory", "calculate_model_hash", "staged_vae_decode", "compile_transformer"):
        if isinstance(source.get(key), bool):
            normalized[key] = source[key]
    candidate_limit = source.get("vram_limit_gb")
    if isinstance(candidate_limit, (int, float)) and not isinstance(candidate_limit, bool):
        try:
            candidate_limit = float(candidate_limit)
        except (TypeError, ValueError):
            candidate_limit = 0.0
        if math.isfinite(candidate_limit) and 0.0 <= candidate_limit <= 1024.0:
            normalized["vram_limit_gb"] = round(candidate_limit, 1)
    if normalized["memory_mode"] == "ultra_low_vram":
        normalized.update({
            "attention_backend": "sliced",
            "compute_dtype": "fp16",
            "vae_mode": "tiled",
            "cuda_math": "strict",
            "keep_model_cached": False,
            "allow_shared_memory": False,
            "calculate_model_hash": False,
            "staged_vae_decode": True,
            # Ultra-low is a survival mode: it spends memory nowhere, and a
            # compiled graph holds workspace the sequential offload path needs.
            "compile_transformer": False,
        })
    return normalized


def read_performance_settings(path: Path):
    try:
        return normalize_performance_settings(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_PERFORMANCE_SETTINGS.copy()


def write_performance_settings(path: Path, settings):
    normalized = normalize_performance_settings(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(f"{json.dumps(normalized, ensure_ascii=False, indent=2)}\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return normalized


def file_sha256(path: Path, chunk_size=8 * 1024**2):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_mode_for_family(requested_mode, family):
    if requested_mode != "sdxl_balanced":
        return requested_mode
    return "normal_vram" if family == "sdxl" else "auto"


def split_batch_value(value, start, end, batch_size):
    if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == batch_size:
        return value[start:end]
    if isinstance(value, dict):
        return {key: split_batch_value(item, start, end, batch_size) for key, item in value.items()}
    if isinstance(value, list):
        return [split_batch_value(item, start, end, batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(split_batch_value(item, start, end, batch_size) for item in value)
    return value


def enable_sequential_batch_forward(module):
    if module.forward is getattr(module, "_xiriai_sequential_batch_forward", None):
        return
    original_forward = module.forward
    module._xiriai_full_batch_forward = original_forward

    def sequential_forward(sample, *args, **kwargs):
        batch_size = sample.shape[0]
        if batch_size <= 1:
            return original_forward(sample, *args, **kwargs)
        outputs = []
        for index in range(batch_size):
            chunk_args = tuple(split_batch_value(value, index, index + 1, batch_size) for value in args)
            chunk_kwargs = {
                key: split_batch_value(value, index, index + 1, batch_size) for key, value in kwargs.items()
            }
            outputs.append(original_forward(sample[index:index + 1], *chunk_args, **chunk_kwargs))
        if isinstance(outputs[0], tuple):
            return tuple(
                torch.cat([output[component] for output in outputs], dim=0)
                if torch.is_tensor(value) and value.shape[0] == 1 else value
                for component, value in enumerate(outputs[0])
            )
        raise TypeError("Sequential batch forward requires tuple results")

    module._xiriai_sequential_batch_forward = sequential_forward
    module.forward = sequential_forward


def vae_decode_tile_count(latent_height, latent_width, tile_size, overlap_factor):
    step = max(1, int(tile_size * (1 - overlap_factor)))
    return ceil(latent_height / step) * ceil(latent_width / step)


def nvfp4_capabilities(torch_module, package_available=None):
    cuda_available = bool(torch_module.cuda.is_available())
    capability = torch_module.cuda.get_device_capability() if cuda_available else (0, 0)
    hardware_supported = cuda_available and capability >= (10, 0)
    torch_supported = hasattr(torch_module, "float4_e2m1fn_x2")
    if package_available is None:
        package_available = lambda name: find_spec(name) is not None
    torchao_available = package_available("torchao")
    modelopt_available = package_available("modelopt")
    runtime_ready = hardware_supported and torch_supported and (torchao_available or modelopt_available)
    return {
        "hardware_supported": hardware_supported,
        "torch_supported": torch_supported,
        "torchao_available": torchao_available,
        "modelopt_available": modelopt_available,
        "runtime_ready": runtime_ready,
        "enabled": False,
        "note": "NVFP4 is detected but not enabled for SD/SDXL single-file pipelines until LoRA and offload compatibility is validated.",
    }
