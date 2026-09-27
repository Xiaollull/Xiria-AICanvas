import asyncio
import base64
import binascii
import copy
import gc
import hashlib
import hmac
import io
import itertools
import json
import math
import os
import platform
import shutil
import signal
import sys
import threading
import time
import traceback
import uuid
import warnings
import re
import secrets
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as package_version
from importlib.util import find_spec
from math import ceil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Callable, Literal, Union
try:
    from .benchmark_lease import HIRES_ARTIFACT_PURPOSE, validate_lease
    from .hires_artifacts import NULL_CAPTURE, StageArtifactCapture, canonical_parameter_digest, capture_gate, prompt_facts
    from .progress_console import ProgressConsole
except ImportError:
    from benchmark_lease import HIRES_ARTIFACT_PURPOSE, validate_lease
    from hires_artifacts import NULL_CAPTURE, StageArtifactCapture, canonical_parameter_digest, capture_gate, prompt_facts
    from progress_console import ProgressConsole

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANIMA_TOKENIZER_DIRECTORY = PROJECT_ROOT / "backend" / "resources" / "anima-tokenizers"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def configured_path(name: str, default: str):
    configured = os.environ.get(name)
    value = default if configured is None else configured.strip()
    if not value:
        raise RuntimeError(f"{name} must not be empty")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def configured_int(name: str, default: int, minimum: int, maximum: int):
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


def configured_float(name: str, default: float, minimum: float):
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number") from error
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    return value


def anima_group_cfg_batch_requested():
    """Internal benchmark-only gate; it is deliberately not a user setting."""
    return os.environ.get("XIRAI_ANIMA_GROUP_CFG_BATCH", "").strip() == "1"


def anima_compile_requested():
    """Inductor compilation of the Cosmos blocks: a Performance setting, or the
    environment switch the benchmark harness uses to override it."""
    return bool(performance_settings.get("compile_transformer")) or os.environ.get("XIRAI_ANIMA_COMPILE", "").strip() == "1"


def anima_compile_mode():
    mode = os.environ.get("XIRAI_ANIMA_COMPILE_MODE", "default").strip() or "default"
    if mode not in {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}:
        raise RuntimeError(
            "XIRAI_ANIMA_COMPILE_MODE must be default, reduce-overhead, max-autotune or max-autotune-no-cudagraphs"
        )
    return mode


def anima_sage_attention_requested():
    """SageAttention for Anima's self-attention: the `sage` attention backend, or
    the environment switch the benchmark harness uses to override it.

    Never automatic. Sage quantises Q/K to INT8, so it is a deliberate accuracy
    trade rather than a free win, and it has to be chosen.
    """
    return (
        performance_settings.get("attention_backend") == "sage"
        or os.environ.get("XIRAI_ANIMA_SAGE_ATTENTION", "").strip() == "1"
    )


def anima_resident_cfg_batch_requested():
    """Require an isolated benchmark child; never enable this in the main service."""
    return bool(
        os.environ.get("XIRAI_BENCHMARK_CHILD", "").strip() == "1"
        and os.environ.get("XIRAI_ANIMA_RESIDENT_CFG_BATCH", "").strip() == "1"
        and INFERENCE_PORT != 8718
        and OUTPUT_DIRECTORY.resolve() != (PROJECT_ROOT / "outputs").resolve()
        and STATE_DIRECTORY.resolve() != (PROJECT_ROOT / "state-cache").resolve()
    )


def anima_resident_cfg_batch_force_requested():
    return os.environ.get("XIRAI_ANIMA_RESIDENT_CFG_BATCH_FORCE", "").strip() == "1"


def anima_resident_cfg_batch_force_effective():
    """A four-factor isolated benchmark override, never a product admission."""
    if not bool(
        anima_resident_cfg_batch_requested()
        and anima_resident_cfg_batch_force_requested()
        and os.environ.get("XIRAI_BENCHMARK_PURPOSE", "").strip() == "resident_cfg_batch_probe"
    ):
        return False
    return benchmark_lease_validation()["valid"]


CACHE_DIRECTORY = configured_path("XIRAI_CACHE_DIR", ".cache")
PROJECT_CACHE = CACHE_DIRECTORY / "huggingface"
os.environ.setdefault("HF_HOME", str(PROJECT_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
warnings.filterwarnings("ignore", message="Failed to find CUDA.*", module="triton.windows_utils")

import torch
import uvicorn
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
)
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo
from safetensors import safe_open

try:
    from .gallery import (
        UNSET as GALLERY_UNSET,
        GalleryConflictError,
        GalleryNotFoundError,
        GalleryStorageError,
        GalleryStore,
        GalleryValidationError,
    )
except ImportError:
    from gallery import (
        UNSET as GALLERY_UNSET,
        GalleryConflictError,
        GalleryNotFoundError,
        GalleryStorageError,
        GalleryStore,
        GalleryValidationError,
    )

try:
    from .adetailer import (
        detection_mask,
        discover_detector_models,
        expand_prompt,
        resolve_detector_model,
        run_detector,
        select_detections,
    )
    from .anima_pipeline import ANIMA_PIXEL_ALIGNMENT, load_anima_runtime
    from .anima_sampling import ANIMA_SAMPLERS, ANIMA_SCHEDULERS, anima_sampling_diagnostics
    from .flux_pipeline import FLUX_MAX_EDGE, FLUX_PIXEL_ALIGNMENT, flux_component_bytes, load_flux_runtime
    from .flux_sampling import (
        FLUX_SAMPLERS,
        FLUX_SCHEDULERS,
        flux_resolution_shift,
        flux_sampling_diagnostics,
    )
    from .flux2_pipeline import FLUX2_MAX_EDGE, FLUX2_PIXEL_ALIGNMENT, flux2_component_bytes, load_flux2_runtime
    from .flux2_sampling import (
        FLUX2_SAMPLERS,
        FLUX2_SCHEDULERS,
        flux2_resolution_shift,
        flux2_sampling_diagnostics,
    )
    from .krea2_pipeline import (
        KREA2_MAX_EDGE,
        KREA2_PIXEL_ALIGNMENT,
        krea2_component_bytes,
        load_krea2_runtime,
    )
    from .krea2_sampling import (
        KREA2_SAMPLERS,
        KREA2_SCHEDULERS,
        krea2_sampling_diagnostics,
        KREA2_SHIFT,
    )
    from .hardware_probe import CpuSampler, NvidiaSmiSampler, probe as probe_hardware, read_gpu_memory, read_system_memory
    from .memory_policy import (
        GIB,
        error_looks_like_oom,
        estimate_inference_bytes,
        estimate_largest_component_bytes,
        estimate_vae_decode_bytes,
        normalize_memory_mode,
        reserved_vram_bytes,
        effective_vram_limit_bytes,
        vram_limit_bounds,
        select_memory_policy,
    )
    from .model_paths import resolve_model_directory
    from .shared_model_paths import (
        is_shared_ref,
        load_shared_roots,
        resolve_shared_model,
        shared_model_reference,
    )
    from .background_removal import (
        background_removal_status,
        clear_background_removal_session,
        extract_foreground,
        parse_prompt_directives,
        require_model as require_background_removal_model,
        transparent_conditioning_prompt,
    )
    from .prompt_encoding import parse_prompt_weights, prompt_diagnostics
    from .forge_prompt import PreparedConditioning, forge_prompt_diagnostics
    from .output_paths import (
        DEFAULT_OUTPUT_TEMPLATE,
        OUTPUT_TEMPLATE_PLACEHOLDERS,
        OutputPathError,
        normalize_output_template,
        output_root_display,
        output_target,
        preview_output_name,
        publish_exclusive,
        temporary_output_path,
    )
    from .dynamic_prompt import (
        PromptTimeline,
        compile_conditioning_syntax,
        prepare_dynamic_prompt_conditioning,
        prompt_plan_audit,
        sampling_timeline,
        static_conditioning_lease,
    )
    from .inline_lora import InlineLoraError, discover_candidates, has_extra_network_tag, resolve_inline_loras
    from .comfy_workflow import ResampleStage, WorkflowFacts, build_workflow
    from .guidance import (
        CFG_ZERO_STAR_ENGINES,
        PAG_APPLIED_LAYERS,
        PAG_SCALE,
        cfg_zero_star_zero_steps,
        guidance_diagnostics,
        guidance_prediction_copies,
        pag_layer_pattern,
    )
    from . import rtx_vsr
    from .comfy_quant import quantization_capabilities
    from .performance_settings import (
        file_sha256,
        disable_sequential_batch_forward,
        enable_sequential_batch_forward,
        memory_mode_for_family,
        nvfp4_capabilities,
        read_performance_settings,
        vae_decode_tile_count,
        write_performance_settings,
    )
    from .pipeline_configs import PIPELINE_CONFIG_REPOSITORIES, find_pipeline_config, require_pipeline_config
    from .console_noise import mark_long_prompts_handled, quiet_lora_loading, quiet_pipeline_loading
    from .lora_patch import (
        MAX_GROUP_WORKSPACE_BYTES,
        HostWeightPrefault,
        LoraPatchError,
        apply_lora_plan,
        build_lora_groups,
        build_target_index,
        classify_lora_keys,
        fold_peft_adapters,
        lora_header_requires_direct_patch,
        preflight_lora_patch,
        read_lora_header,
        readable_lora_error,
        validate_lora_plan_materialized,
    )
    from .residency import (
        RESIDENCY_TIER_MEMORY_MODES,
        RESIDENCY_TIER_OFFLOAD_MODES,
        ResidencyController,
        remove_group_offload,
        tier_for_memory_mode,
    )
    from .stage_scheduler import (
        AUTO_LABEL,
        BELOW_RECOMMENDED_WARNING,
        NATIVE_BLOCK,
        NATIVE_RESIDENT,
        RECOMMENDED_MINIMUM_VRAM_BYTES,
        TEXT_ENCODE_TRANSIENT_BYTES,
        StageBudget,
        StageMemoryError,
        allocator_safety_margin_bytes,
        auto_label,
        below_recommended_vram,
        cap_invariant,
        choose_native_placement,
        decision_record,
        path_for_tier,
        path_label,
        png_schedule_note,
    )
    from .conditioning_reuse import StaticConditioningCache, lora_fusion_signature, tower_signature
    from .resize import DEFAULT_RESIZE_METHOD, RESIZE_METHODS, align_size, resize_image, scaled_size
    from .upscaler import resolve_model as resolve_upscaler_model, status as upscaler_status, upscale_image
    from .usdu_tiles import TileCompositor, plan_tiles, prepare_tile
except ImportError:
    from adetailer import (
        detection_mask,
        discover_detector_models,
        expand_prompt,
        resolve_detector_model,
        run_detector,
        select_detections,
    )
    from anima_pipeline import ANIMA_PIXEL_ALIGNMENT, load_anima_runtime
    from anima_sampling import ANIMA_SAMPLERS, ANIMA_SCHEDULERS, anima_sampling_diagnostics
    from flux_pipeline import FLUX_MAX_EDGE, FLUX_PIXEL_ALIGNMENT, flux_component_bytes, load_flux_runtime
    from flux_sampling import (
        FLUX_SAMPLERS,
        FLUX_SCHEDULERS,
        flux_resolution_shift,
        flux_sampling_diagnostics,
    )
    from flux2_pipeline import FLUX2_MAX_EDGE, FLUX2_PIXEL_ALIGNMENT, flux2_component_bytes, load_flux2_runtime
    from flux2_sampling import (
        FLUX2_SAMPLERS,
        FLUX2_SCHEDULERS,
        flux2_resolution_shift,
        flux2_sampling_diagnostics,
    )
    from krea2_pipeline import (
        KREA2_MAX_EDGE,
        KREA2_PIXEL_ALIGNMENT,
        krea2_component_bytes,
        load_krea2_runtime,
    )
    from krea2_sampling import (
        KREA2_SAMPLERS,
        KREA2_SCHEDULERS,
        krea2_sampling_diagnostics,
        KREA2_SHIFT,
    )
    from hardware_probe import CpuSampler, NvidiaSmiSampler, probe as probe_hardware, read_gpu_memory, read_system_memory
    from memory_policy import (
        GIB,
        error_looks_like_oom,
        estimate_inference_bytes,
        estimate_largest_component_bytes,
        estimate_vae_decode_bytes,
        normalize_memory_mode,
        reserved_vram_bytes,
        effective_vram_limit_bytes,
        vram_limit_bounds,
        select_memory_policy,
    )
    from model_paths import resolve_model_directory
    from shared_model_paths import (
        is_shared_ref,
        load_shared_roots,
        resolve_shared_model,
        shared_model_reference,
    )
    from background_removal import (
        background_removal_status,
        clear_background_removal_session,
        extract_foreground,
        parse_prompt_directives,
        require_model as require_background_removal_model,
        transparent_conditioning_prompt,
    )
    from prompt_encoding import parse_prompt_weights, prompt_diagnostics
    from forge_prompt import PreparedConditioning, forge_prompt_diagnostics
    from output_paths import (
        DEFAULT_OUTPUT_TEMPLATE,
        OUTPUT_TEMPLATE_PLACEHOLDERS,
        OutputPathError,
        normalize_output_template,
        output_root_display,
        output_target,
        preview_output_name,
        publish_exclusive,
        temporary_output_path,
    )
    from dynamic_prompt import (
        PromptTimeline,
        compile_conditioning_syntax,
        prepare_dynamic_prompt_conditioning,
        prompt_plan_audit,
        sampling_timeline,
        static_conditioning_lease,
    )
    from inline_lora import InlineLoraError, discover_candidates, has_extra_network_tag, resolve_inline_loras
    from comfy_workflow import ResampleStage, WorkflowFacts, build_workflow
    from guidance import (
        CFG_ZERO_STAR_ENGINES,
        PAG_APPLIED_LAYERS,
        PAG_SCALE,
        cfg_zero_star_zero_steps,
        guidance_diagnostics,
        guidance_prediction_copies,
        pag_layer_pattern,
    )
    import rtx_vsr
    from comfy_quant import quantization_capabilities
    from performance_settings import (
        file_sha256,
        disable_sequential_batch_forward,
        enable_sequential_batch_forward,
        memory_mode_for_family,
        nvfp4_capabilities,
        read_performance_settings,
        vae_decode_tile_count,
        write_performance_settings,
    )
    from pipeline_configs import PIPELINE_CONFIG_REPOSITORIES, find_pipeline_config, require_pipeline_config
    from console_noise import mark_long_prompts_handled, quiet_lora_loading, quiet_pipeline_loading
    from lora_patch import (
        MAX_GROUP_WORKSPACE_BYTES,
        HostWeightPrefault,
        LoraPatchError,
        apply_lora_plan,
        build_lora_groups,
        build_target_index,
        classify_lora_keys,
        fold_peft_adapters,
        lora_header_requires_direct_patch,
        preflight_lora_patch,
        read_lora_header,
        readable_lora_error,
        validate_lora_plan_materialized,
    )
    from residency import (
        RESIDENCY_TIER_MEMORY_MODES,
        RESIDENCY_TIER_OFFLOAD_MODES,
        ResidencyController,
        remove_group_offload,
        tier_for_memory_mode,
    )
    from stage_scheduler import (
        AUTO_LABEL,
        BELOW_RECOMMENDED_WARNING,
        NATIVE_BLOCK,
        NATIVE_RESIDENT,
        RECOMMENDED_MINIMUM_VRAM_BYTES,
        TEXT_ENCODE_TRANSIENT_BYTES,
        StageBudget,
        StageMemoryError,
        allocator_safety_margin_bytes,
        auto_label,
        below_recommended_vram,
        cap_invariant,
        choose_native_placement,
        decision_record,
        path_for_tier,
        path_label,
        png_schedule_note,
    )
    from conditioning_reuse import StaticConditioningCache, lora_fusion_signature, tower_signature
    from resize import DEFAULT_RESIZE_METHOD, RESIZE_METHODS, align_size, resize_image, scaled_size
    from upscaler import resolve_model as resolve_upscaler_model, status as upscaler_status, upscale_image
    from usdu_tiles import TileCompositor, plan_tiles, prepare_tile


OUTPUT_DIRECTORY = configured_path("XIRAI_OUTPUT_DIR", "outputs")
# Where the withdrawn latent process preview wrote its frames. Nothing writes here any more; the
# name survives so an installation upgrading from a build that did keeps the leftovers swept and
# out of the output browser rather than suddenly showing them as a folder full of stray frames.
LEGACY_PREVIEW_DIRECTORY = OUTPUT_DIRECTORY / ".previews"
# Stage previews outlive the run that made them — the point of them is to be looked at after a
# stage finishes, while the next stage is still running.
STAGE_PREVIEW_DIRECTORY = OUTPUT_DIRECTORY / ".stages"
# Folders under the output root that hold machinery rather than pictures. `.stages` was missing
# from the old guard, which was written for `.previews` alone, so the browser listed the stage
# folder as an empty folder of its own.
INTERNAL_OUTPUT_FOLDER_NAMES = (".previews", ".stages")


def is_internal_output_directory(path: Path) -> bool:
    """Whether `path` is one of those folders, or sits inside one.

    The roots are derived on each call rather than frozen beside the constants above: tests
    redirect `OUTPUT_DIRECTORY`, and so does an installation that moves its output folder, and a
    tuple captured at import time would go on guarding the folder the server started with.
    """
    resolved = path.resolve()
    for name in INTERNAL_OUTPUT_FOLDER_NAMES:
        root = (OUTPUT_DIRECTORY / name).resolve()
        if resolved == root or root in resolved.parents:
            return True
    return False
LOG_DIRECTORY = PROJECT_ROOT / "logs"
STATE_DIRECTORY = configured_path("XIRAI_STATE_DIR", "state-cache")
PERFORMANCE_SETTINGS_FILE = STATE_DIRECTORY / "performance.json"
ADETAILER_MODEL_DIRECTORY = resolve_model_directory(PROJECT_ROOT, "yolo")
UPSCALER_MODEL_DIRECTORY = resolve_model_directory(PROJECT_ROOT, "upscalers")
# The interpreter running this server, which inside a virtual environment is that environment's
# own `python`. It must not be resolved through its symlink: on Linux the installer builds the
# environment with uv against a managed CPython, so `.venv/bin/python` points at an interpreter
# that has none of the environment's packages, and the detector worker launched with it dies on
# `import PIL` before it reaches a model. Windows copies the executable into the environment
# rather than linking it, which is why resolving only ever broke the Linux installs.
ADETAILER_PYTHON = Path(sys.executable)
ADETAILER_WORKER = PROJECT_ROOT / "backend" / "adetailer_detector.py"
YOLO_CATALOG_PATH = PROJECT_ROOT / "models" / "yolo-models.json"
# A stage preview is a finished picture the user is invited to enlarge and inspect, so it keeps
# a lot of detail; it is still bounded, because a 3072x2304 Hires result would otherwise be copied
# in full for every stage of every job in the queue history.
STAGE_PREVIEW_MAX_EDGE = configured_int("STAGE_PREVIEW_MAX_EDGE", 1536, 384, 4096)


def installed_vram_bytes():
    """Physical VRAM on the first device, or 0 when there is no CUDA device to ask."""
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return 0
    return 0


# The card decides one of the shipped defaults -- see `default_performance_settings` -- so the
# settings are read with it in hand rather than being corrected afterwards.
performance_settings = read_performance_settings(PERFORMANCE_SETTINGS_FILE, installed_vram_bytes())
try:
    performance_settings["memory_mode"] = normalize_memory_mode(
        os.environ.get("XIRAI_MEMORY_MODE", performance_settings["memory_mode"])
    )
except ValueError as error:
    raise RuntimeError(str(error)) from error
RESERVED_VRAM_GB = os.environ.get("XIRAI_RESERVE_VRAM_GB")
if RESERVED_VRAM_GB is not None:
    try:
        RESERVED_VRAM_GB = float(RESERVED_VRAM_GB)
    except ValueError as error:
        raise RuntimeError("XIRAI_RESERVE_VRAM_GB must be a number") from error
    if RESERVED_VRAM_GB < 0:
        raise RuntimeError("XIRAI_RESERVE_VRAM_GB must be at least 0")


def _configure_initial_cuda_memory_limit():
    # Defined before the service starts so the first model allocation observes the wall.
    if torch.cuda.is_available():
        try:
            physical_total = int(torch.cuda.get_device_properties(0).total_memory)
            configured_limit = performance_settings.get("vram_limit_gb", 0.0)
            limit = effective_vram_limit_bytes(
                physical_total,
                configured_limit,
                allow_shared_memory=performance_settings["allow_shared_memory"],
            )
            fraction = min(1.0, max(0.01, limit / physical_total)) if physical_total else 1.0
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
        except Exception:
            # Policy admission remains active on CUDA builds without allocator fractions.
            pass


_configure_initial_cuda_memory_limit()
INFERENCE_HOST = os.environ.get("INFERENCE_HOST", "127.0.0.1")
INFERENCE_PORT = configured_int("INFERENCE_PORT", 8718, 1, 65535)
def benchmark_lease_validation(expected_purpose=None):
    """Revalidate the current lease for each force strategy evaluation."""
    purpose = {} if expected_purpose is None else {"expected_purpose": expected_purpose}
    return validate_lease(
        os.environ.get("XIRAI_BENCHMARK_LEASE_PATH") or os.devnull,
        os.environ.get("XIRAI_BENCHMARK_LEASE_NONCE"), STATE_DIRECTORY, OUTPUT_DIRECTORY,
        os.environ.get("INFERENCE_WORKSPACE_ID"), PROJECT_ROOT, **purpose,
    )


def hires_artifact_capture_gate():
    """Benchmark-only stage capture gate; diagnostics-safe fields only, never the nonce."""
    return capture_gate(
        environ=os.environ, inference_port=INFERENCE_PORT, project_root=PROJECT_ROOT,
        state_root=STATE_DIRECTORY, output_root=OUTPUT_DIRECTORY,
        workspace_id=os.environ.get("INFERENCE_WORKSPACE_ID"),
        lease_validator=lambda: benchmark_lease_validation(HIRES_ARTIFACT_PURPOSE),
    )


def open_hires_artifact_capture(job_id, base_seed):
    """Return the no-op capture unless the isolated benchmark gate and its own lease both pass."""
    gate = hires_artifact_capture_gate()
    if not gate["enabled"]:
        return NULL_CAPTURE, gate
    root = STATE_DIRECTORY / "benchmark" / f"hires-artifacts-{job_id}" / str(base_seed)
    return StageArtifactCapture(root), gate
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
STAGE_PREVIEW_DIRECTORY.mkdir(parents=True, exist_ok=True)
LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
ADETAILER_MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
UPSCALER_MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
HISTORY_STARTED_AT = time.time()

# The engines that run their own sampler over ComfyUI-layout component files rather than a
# Diffusers pipeline built from one checkpoint. They share the staged load, the group-offload
# escape hatch and the refinement contract, which is why so much of the run loop asks "is this a
# native family" rather than naming one.
NATIVE_ENGINES = ("Anima", "Flux", "Flux2", "Krea2")
NATIVE_FAMILIES = ("anima", "flux", "flux2", "krea2")
NATIVE_FAMILY_BY_ENGINE = {"Anima": "anima", "Flux": "flux", "Flux2": "flux2", "Krea2": "krea2"}
# Both Flux generations are guidance distilled: one forward per step steered by a scalar in the
# timestep embedding, with no unconditional branch to hold a negative prompt, PAG or CFG-Zero*.
# Krea 2 is deliberately absent: it carries no guidance conditioning and runs ordinary CFG.
DISTILLED_GUIDANCE_ENGINES = ("Flux", "Flux2")
# Every native runtime refuses a refinement canvas past this edge: their positional encodings and
# their VRAM envelope were both measured below it.
NATIVE_MAX_REFINEMENT_EDGE = 4096


def is_native_family(family):
    return family in NATIVE_FAMILIES


# USDU refines one tile at a time and hands every tile the same conditioning and the same sigma
# suffix, so an engine qualifies once it can prepare and validate both. FLUX.1 and FLUX.2 are
# guidance distilled and have no unconditional branch to prepare, and neither exposes the prepare
# surface, so they still run Hires full frame.
USDU_TILED_FAMILIES = frozenset({"anima", "krea2"})

# The canvas granularity each engine accepts. A tile is sampled at the plan's processing size, so
# the plan has to be built to the engine's own alignment: rounding the padded core to the nearest
# multiple of 8 lands off Anima's 32 for most paddings, and off Krea 2's 16 for many of them.
USDU_PIXEL_ALIGNMENT = {"anima": ANIMA_PIXEL_ALIGNMENT, "krea2": KREA2_PIXEL_ALIGNMENT}

# What each engine's latent grid accepts as a redraw canvas. A resample unit's target is exact --
# `round(edge x scale)` and nothing else -- but the canvas handed to a diffusion redraw is not free:
# every runtime here rejects a size its VAE and patch embedding cannot divide. So the target is the
# number the chain reports and this is the number the redraw runs at, and a unit records both
# whenever they differ rather than quietly presenting one as the other.
ENGINE_PIXEL_ALIGNMENT = {
    "sd": 8,
    "anima": ANIMA_PIXEL_ALIGNMENT,
    "flux": FLUX_PIXEL_ALIGNMENT,
    "flux2": FLUX2_PIXEL_ALIGNMENT,
    "krea2": KREA2_PIXEL_ALIGNMENT,
}
# The edge each engine refuses a refinement past, by family rather than by engine name so the walk
# that validates a chain and the stage that executes it consult one table.
ENGINE_MAX_REFINEMENT_EDGE = {
    "sd": rtx_vsr.MAX_OUTPUT_EDGE,
    "anima": NATIVE_MAX_REFINEMENT_EDGE,
    "flux": FLUX_MAX_EDGE,
    "flux2": FLUX2_MAX_EDGE,
    "krea2": KREA2_MAX_EDGE,
}


def supports_usdu_tiled(family):
    return family in USDU_TILED_FAMILIES


def usdu_pixel_alignment(family):
    return USDU_PIXEL_ALIGNMENT[family]


def engine_pixel_alignment(family):
    return ENGINE_PIXEL_ALIGNMENT.get(family, ENGINE_PIXEL_ALIGNMENT["sd"])


def engine_max_refinement_edge(family):
    return ENGINE_MAX_REFINEMENT_EDGE.get(family, ENGINE_MAX_REFINEMENT_EDGE["sd"])


def redraw_canvas_size(size, family):
    """The canvas an engine will actually sample, for a unit target of `size`."""
    return align_size(size, engine_pixel_alignment(family))


def request_family(engine: str):
    """The runtime family an engine name maps to. SD and iL share the Diffusers path."""
    return NATIVE_FAMILY_BY_ENGINE.get(engine, "sd")


# One worker, so submitting several jobs queues them rather than running them together: a second
# generation sharing the card with the first would slow both and could exhaust VRAM outright.
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
# Position in this session's queue, counting from one. It numbers what the user submitted rather
# than what survived: a cancelled job keeps its number so the list reads as a history. The
# registry lives in memory only, so restarting the package starts the count again at one.
job_sequence = itertools.count(1)
# A queue is a history as well as a backlog, and a long session would otherwise hold every job's
# metadata for as long as the process runs. Finished jobs are trimmed oldest-first past this many;
# anything still queued or running is never trimmed.
JOB_HISTORY_LIMIT = 200
pipeline_lock = threading.Lock()
job_controls = {}
loaded_pipeline = None
pipeline_cpu_parked = False
loaded_checkpoint = None
loaded_checkpoint_hash = None
loaded_family = None
loaded_engine = None
loaded_model_assets = {}
loaded_model_revisions = {}
active_memory_strategy = None
# Owns where the diffusion model lives for the loaded Diffusers pipeline, and the only thing allowed
# to install an offload mechanism on it. None for the native engines, which schedule their own.
active_residency = None
# The job the controller's moves are announced on. The controller outlives jobs; this does not.
residency_job_id = None
# AUTO · 动态决策 (QPlan P8): every stage decision this job has made, in one vocabulary for every
# engine, and the job's shared static prompt encodings. Both belong to one job and are reset by it.
active_stage_schedule = []
active_conditioning_cache = None
active_attention_backend = "none"
active_compute_dtype = "none"
active_vae_mode = "none"
shutdown_requested = False

# The monitor's two samplers are process-wide on purpose. The GPU one owns a background thread so
# `nvidia-smi` never lands on the request path, and the CPU one owns the interval that
# `psutil.cpu_percent` measures over — an interval that is global to the process, so exactly one
# caller may hold it.
gpu_sensor_sampler = NvidiaSmiSampler()
cpu_sampler = CpuSampler()

def model_memory_snapshot():
    """Device and system memory as they stand, for the line printed once a model is resident.

    Read at the moment it is asked for rather than sampled in the background: this is called once
    per model load, and a figure taken seconds earlier would describe the memory before the weights
    arrived — which is precisely the comparison the line exists to let the reader make.
    """
    snapshot = {}
    try:
        snapshot.update(read_gpu_memory(torch) or {})
    except Exception:
        pass
    try:
        psutil = psutil_module()
        if psutil is not None:
            snapshot.update(read_system_memory(psutil))
    except Exception:
        pass
    return snapshot


# Sampling progress for the terminal and the in-app drawer, which are the same stdout stream. The
# budget caps how many lines one run may print, however many steps it has; XIRAI_PROGRESS_CONSOLE=0
# turns the lines off without touching the web UI, which reads the job record directly.
progress_console = ProgressConsole(
    enabled=os.environ.get("XIRAI_PROGRESS_CONSOLE", "1").strip() != "0",
    budget=configured_int("XIRAI_PROGRESS_LINES", 12, 2, 200),
    memory=model_memory_snapshot,
)


@asynccontextmanager
async def lifespan(_app):
    global shutdown_requested
    rtx_vsr.start_probe()
    yield
    shutdown_requested = True
    gpu_sensor_sampler.stop()
    with jobs_lock:
        controls = list(job_controls.values())
    for control in controls:
        control.cancel()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with jobs_lock:
            if not job_controls:
                break
        await asyncio.sleep(0.1)
    executor.shutdown(wait=False, cancel_futures=True)
    with jobs_lock:
        idle = not job_controls
    if idle:
        clear_pipeline()
    clear_background_removal_session()
    # The queue history resets on restart, so the stage images it referred to are orphans.
    for stage_path in STAGE_PREVIEW_DIRECTORY.glob("*"):
        if stage_path.is_file():
            remove_preview(stage_path)
    # Frames left behind by the withdrawn latent preview. Sweeping them on the way out means an
    # installation upgrading from that build reclaims the space without anyone going looking.
    if LEGACY_PREVIEW_DIRECTORY.is_dir():
        for preview_path in LEGACY_PREVIEW_DIRECTORY.glob("*"):
            if preview_path.is_file():
                remove_preview(preview_path)
        with suppress(OSError):
            LEGACY_PREVIEW_DIRECTORY.rmdir()


app = FastAPI(title="XiriaCanvas AI Inference", docs_url=None, redoc_url=None, lifespan=lifespan)
INFERENCE_PROTOCOL = 39
WORKSPACE_ID = os.environ.get("INFERENCE_WORKSPACE_ID")


def installed_version(package: str):
    try:
        return package_version(package)
    except PackageNotFoundError:
        return None


def builtin_yolo_models():
    try:
        payload = json.loads(YOLO_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    valid = []
    for model in models:
        if not isinstance(model, dict):
            continue
        name = model.get("name")
        if not isinstance(name, str) or Path(name).name != name or Path(name).suffix.lower() != ".pt":
            continue
        valid.append({
            "name": name,
            "label": str(model.get("label") or name),
            "description": str(model.get("description") or ""),
        })
    return valid


def adetailer_runtime_available():
    return installed_version("ultralytics") is not None


class GenerationCancelled(Exception):
    pass


class JobControl:
    def __init__(self):
        self.condition = threading.Condition()
        self.paused = False
        self.cancelled = False
        self.pause_started = None
        self.paused_seconds = 0.0

    def request_pause(self):
        with self.condition:
            if self.cancelled:
                return False
            self.paused = True
            return True

    def resume(self):
        with self.condition:
            if self.cancelled:
                return False
            self.paused = False
            self.condition.notify_all()
            return True

    def cancel(self):
        with self.condition:
            self.cancelled = True
            self.paused = False
            self.condition.notify_all()

    def active_elapsed(self, started_at):
        paused = self.paused_seconds
        if self.pause_started is not None:
            paused += time.monotonic() - self.pause_started
        return max(0.0, time.time() - started_at - paused)

    def total_paused(self):
        paused = self.paused_seconds
        if self.pause_started is not None:
            paused += time.monotonic() - self.pause_started
        return paused

    def raise_if_cancelled(self):
        with self.condition:
            if self.cancelled:
                raise GenerationCancelled()

    def checkpoint(self, job_id: str, running_phase: str):
        with self.condition:
            if self.cancelled:
                raise GenerationCancelled()
            if not self.paused:
                return

            if self.pause_started is None:
                self.pause_started = time.monotonic()
            update_job(job_id, status="paused", phase="Paused")
            while self.paused and not self.cancelled:
                self.condition.wait(timeout=0.5)

            if self.pause_started is not None:
                self.paused_seconds += time.monotonic() - self.pause_started
                self.pause_started = None
            if self.cancelled:
                raise GenerationCancelled()
            update_job(job_id, status="running", phase=running_phase, paused_seconds=round(self.paused_seconds, 1))


class LoraInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    weight: float = Field(ge=-5, le=5)


# How many detail passes one render may queue. Each is a full detect-and-inpaint
# cycle over the previous pass's output, so the ceiling is about the user not
# accidentally buying an hour of GPU time, not about the format.
ADETAILER_UNIT_LIMIT = 6


class ADetailerUnitInput(BaseModel):
    """One detail pass: which detector, and the settings that repair what it finds.

    The settings that fix a face are the wrong settings for a hand, which is why
    each unit carries its own denoise, steps, cfg and prompts rather than
    inheriting one shared set.
    """

    model_config = ConfigDict(extra="forbid")

    detector: str = ""
    confidence: float = Field(default=0.3, ge=0, le=1)
    max_detections: int = Field(default=3, ge=1, le=8)
    mask_min_ratio: float = Field(default=0, ge=0, le=1)
    mask_max_ratio: float = Field(default=1, ge=0, le=1)
    dilate_erode: int = Field(default=4, ge=-128, le=128)
    mask_blur: int = Field(default=4, ge=0, le=64)
    padding: int = Field(default=32, ge=0, le=256)
    denoise: float = Field(default=0.4, gt=0, le=1)
    use_steps: bool = False
    steps: int = Field(default=28, ge=1, le=100)
    use_cfg: bool = False
    cfg: float = Field(default=7, ge=0, le=30)
    # Uncapped for the same reason the main prompt is: a unit's prompt reaches the same encoders.
    prompt: str = ""
    negative_prompt: str = ""

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.mask_min_ratio > self.mask_max_ratio:
            raise ValueError("ADetailer minimum mask ratio cannot exceed maximum mask ratio")
        return self


class ADetailerInput(BaseModel):
    """The stage: an ordered list of passes, each seeing what the last produced."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["adetailer"] = "adetailer"
    enabled: bool = False
    units: list[ADetailerUnitInput] = Field(default_factory=list, max_length=ADETAILER_UNIT_LIMIT)

    @model_validator(mode="after")
    def validate_configuration(self):
        # A detector is required by the *stage*, not by the unit, and only when the stage will
        # actually run. Asking each unit for one made a switched-off stage able to refuse the whole
        # generation: the editor always holds a first unit, that unit starts with no detector
        # chosen, and a request carrying `enabled: false` alongside it never reached this check —
        # the unit's own validator had already rejected it.
        if not self.enabled:
            return self
        if not self.units:
            raise ValueError("ADetailer requires at least one unit when enabled")
        for index, unit in enumerate(self.units, start=1):
            if not unit.detector:
                raise ValueError(f"ADetailer unit {index} requires a detector")
        return self

    @property
    def active_units(self) -> list[ADetailerUnitInput]:
        return list(self.units) if self.enabled else []


RESAMPLE_UNIT_LIMIT = 8

LATENT_UPSCALE_MODES = {
    "latent": {"label": "Latent", "mode": "bilinear", "antialias": False},
    "latent-antialias": {"label": "Latent (antialiased)", "mode": "bilinear", "antialias": True},
    "latent-bicubic": {"label": "Latent (bicubic)", "mode": "bicubic", "antialias": False},
    "latent-bicubic-antialias": {
        "label": "Latent (bicubic antialiased)", "mode": "bicubic", "antialias": True,
    },
    "latent-nearest": {"label": "Latent (nearest)", "mode": "nearest", "antialias": False},
    "latent-nearest-exact": {
        "label": "Latent (nearest-exact)", "mode": "nearest-exact", "antialias": False,
    },
}
LatentUpscaleAlgorithm = Literal[
    "latent",
    "latent-antialias",
    "latent-bicubic",
    "latent-bicubic-antialias",
    "latent-nearest",
    "latent-nearest-exact",
]


class ResampleUnitInput(BaseModel):
    """One pass of the resample chain: upscale, resize, redraw — each part optional.

    This is what the single Hires.fix block became. The fields it carried all survive with their
    meanings intact; what changed is that none of the three parts is mandatory any more, so one
    model describes the whole behaviour matrix:

    - an upscaler with a scale above 1 and a denoise above 0 is the second pass as it always was;
    - `upscaler = None` drops the model and resizes algorithmically, which is an image resize;
    - `denoise = 0` drops the redraw, which is a pure enlargement;
    - `scale = 1` keeps the size and redraws, which is a detail pass.

    `steps`, `cfg`, `sampler` and `scheduler` are `None` by default and inherit the request's own,
    so a unit added without being configured runs the settings the picture was sampled with rather
    than a second set of defaults nobody chose.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["resample"] = "resample"
    enabled: bool = False
    name: str = Field(default="", max_length=80)
    # `None` is "no model, resize only" and is the default: a unit is useful without one, and an
    # empty string would be indistinguishable from a model that has not been chosen yet.
    upscaler: str | None = Field(default=None, max_length=500)
    latent_algorithm: LatentUpscaleAlgorithm | None = None
    scale: float = Field(default=1.0, ge=1, le=4)
    resize_algorithm: Literal["nearest", "nearest-exact", "bilinear", "area", "bicubic", "lanczos"] = "lanczos"
    # Zero is a real value here, not a disabled unit: it is what asks for the enlargement without
    # the redraw. The old field refused it because a Hires pass that did not redraw was a no-op.
    denoise: float = Field(default=0.35, ge=0, le=1)
    steps: int | None = Field(default=None, ge=1, le=100)
    cfg: float | None = Field(default=None, ge=0, le=30)
    sampler: str | None = None
    scheduler: str | None = None
    seed_mode: Literal["inherit", "fixed", "random"] = "inherit"
    seed: str | None = None
    tile_size: int = Field(default=192, ge=32, le=2048)
    tile_overlap: int = Field(default=16, ge=0, le=512)
    execution_mode: Literal["full_frame", "usdu_tiled"] = "full_frame"
    tile_width: Literal["auto"] = "auto"
    tile_height: Literal["auto"] = "auto"
    padding: int = Field(default=32, ge=0, le=256)
    mask_blur: int = Field(default=8, ge=0, le=64)
    seam_mode: Literal["none"] = "none"
    uniform_tiles: bool = True
    tiled_decode: bool = True
    # The engine rounds the redraw canvas to its own grid, so a unit can finish a few pixels off the
    # size it asked for. Off by default, which is ComfyUI's behaviour — its VAE encode truncates to
    # the same kind of grid and nothing resizes the result back.
    exact_output: bool = False

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed(cls, value):
        if value is None:
            return None
        if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
            raise ValueError("Resample seed must be an unsigned 64-bit decimal string")
        parsed = int(value, 10)
        if parsed > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("Resample seed must not exceed 18446744073709551615")
        return str(parsed)

    @field_validator("scale")
    @classmethod
    def normalize_scale(cls, value: float):
        return round(value, 1)

    @model_validator(mode="after")
    def validate_configuration(self):
        if not math.isfinite(self.scale):
            raise ValueError("Resample scale must be finite")
        if self.enabled and self.latent_algorithm is not None and self.upscaler is not None:
            raise ValueError("Latent resampling cannot be combined with an upscaler model")
        if self.enabled and self.latent_algorithm is not None and self.execution_mode != "full_frame":
            raise ValueError("Latent resampling supports full_frame execution only; USDU tiled mode is unavailable")
        if self.enabled and self.tile_overlap > self.tile_size // 2:
            raise ValueError("Resample tile overlap cannot exceed half the tile size")
        if self.execution_mode == "usdu_tiled" and (not self.uniform_tiles or not self.tiled_decode):
            raise ValueError("USDU tiled resampling requires uniform_tiles and tiled_decode")
        if self.seed_mode == "fixed" and self.seed is None:
            raise ValueError("Resample fixed seed mode requires seed")
        if self.seed_mode != "fixed" and self.seed is not None:
            raise ValueError("Resample seed is only valid in fixed seed mode")
        # A unit that neither enlarges nor redraws would consume a chain slot, a progress band and a
        # preview to hand back the picture it was given.
        if self.enabled and self.scale == 1.0 and self.denoise == 0:
            raise ValueError("A resample unit at scale 1.0 with denoise 0 would do nothing")
        return self

    @property
    def redraws(self) -> bool:
        return self.enabled and self.denoise > 0

    def resolved_steps(self, request_steps: int) -> int:
        return request_steps if self.steps is None else self.steps

    def resolved_cfg(self, request_cfg: float) -> float:
        return request_cfg if self.cfg is None else self.cfg


DIFFUSERS_RESAMPLE_SAMPLERS = frozenset({
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2",
    "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast",
    "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm",
    "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp", "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde",
    "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2",
})
DIFFUSERS_RESAMPLE_SCHEDULERS = frozenset({
    "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal",
    "linear_quadratic", "kl_optimal",
})


def resample_sampling_catalog(family: str):
    return (
        (ANIMA_SAMPLERS, ANIMA_SCHEDULERS) if family == "anima"
        else (FLUX_SAMPLERS, FLUX_SCHEDULERS) if family == "flux"
        else (FLUX2_SAMPLERS, FLUX2_SCHEDULERS) if family == "flux2"
        else (KREA2_SAMPLERS, KREA2_SCHEDULERS) if family == "krea2"
        else (DIFFUSERS_RESAMPLE_SAMPLERS, DIFFUSERS_RESAMPLE_SCHEDULERS)
    )


def validate_resample_sampling_override(sampler: str | None, scheduler: str | None, family: str, unit_index: int | None = None):
    """Validate one unit's explicit overrides without constructing a derived pipeline.

    `unit_index` is 1-based and names the card in the chain: with several units, "unsupported
    sampler" alone leaves the user to find which one they set it on.
    """
    samplers, schedulers = resample_sampling_catalog(family)
    where = f" in resample unit {unit_index}" if unit_index else ""
    if sampler is not None and sampler not in samplers:
        raise ValueError(f"Unsupported resample sampler{where}: {sampler}")
    if scheduler is not None and scheduler not in schedulers:
        raise ValueError(f"Unsupported resample scheduler{where}: {scheduler}")


def serializable_runtime_metrics(value):
    """Deep snapshot runtime metrics before the next tile mutates them."""
    return json.loads(json.dumps(copy.deepcopy(value), default=str))


class SourceImageInput(BaseModel):
    """The picture an image-to-image run starts from.

    ``enabled`` and ``image_data`` are validated against each other rather than treated as
    independent fields, so a request can never carry a megabyte of pixels that nothing reads, and
    an enabled run can never reach the sampler with nothing to sample from.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    image_data: str = Field(default="", max_length=200_000_000)
    resize_mode: Literal["cover", "contain", "stretch"] = "cover"
    name: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def validate_source(self):
        if self.enabled and not self.image_data:
            raise ValueError("Image-to-image requires a source image")
        if not self.enabled and self.image_data:
            raise ValueError("source_image.image_data requires source_image.enabled")
        return self


class RtxInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["rtx"] = "rtx"
    enabled: bool = False
    scale: float = Field(default=2.0, ge=1, le=4)
    quality: Literal["low", "medium", "high", "ultra"] = "ultra"

    @field_validator("scale")
    @classmethod
    def normalize_scale(cls, value: float):
        if not math.isfinite(value):
            raise ValueError("RTX VSR scale must be finite")
        return round(value, 2)


# One entry of the chain. The type is the discriminator and the payload is inline rather than
# nested under a `settings` key: the three stage shapes have no field in common to confuse, and a
# flat entry is what both the workspace record and the PNG parameter block store.
StageInstance = Annotated[
    Union[ResampleUnitInput, ADetailerInput, RtxInput],
    Field(discriminator="type"),
]
POSTPROCESS_STAGE_TYPES = ("resample", "adetailer", "rtx")
# Eight resample units, plus at most one ADetailer stage and one RTX stage.
POSTPROCESS_CHAIN_LIMIT = RESAMPLE_UNIT_LIMIT + 2
# What a chain without that stage reads as. Both stages are single-instance, so a request that omits
# one is asking for it switched off, and the code that reads their settings should not have to ask
# whether the entry exists before asking whether it is enabled.
ABSENT_ADETAILER_STAGE = ADetailerInput()
ABSENT_RTX_STAGE = RtxInput()
# The sampler's canvas: a latent grid, so 64-aligned. 2048 is where the memory policy was measured
# and what every record before protocol 39 was held to; a canvas past it is allowed up to the hard
# 8192 safety ceiling (and each engine's own refinement edge), but only after
# `validate_expanded_canvas_resources` has checked it against this device before the job is queued.
# Post-processing has no latent grid of its own — every stage works on pixels it is handed — so a
# post-processing run is bounded by the RTX safety envelope instead and keeps the picture at
# whatever size the user's file actually is.
LEGACY_MAX_SAMPLING_EDGE = 2048
MAX_SAMPLING_EDGE = 8192
SAMPLING_EDGE_STEP = 64
# CLIP skip exists only where a CLIP tower encodes the prompt. SD reads its final layer by default;
# SDXL/iL reads the penultimate layer of both towers, which is also Forge's floor for them, so a
# requested 1 on iL would execute 2 — and is refused rather than displayed as 1 and run as 2.
CLIP_SKIP_DEFAULTS = {"SD": 1, "iL": 2}
CLIP_SKIP_MAXIMUM = 12
MAX_POSTPROCESS_SOURCE_EDGE = rtx_vsr.MAX_OUTPUT_EDGE
MAX_POSTPROCESS_SOURCE_PIXELS = rtx_vsr.MAX_OUTPUT_PIXELS
ANIMA_TOKENIZER_ARTIFACTS = {
    "qwen": (
        "anima-qwen3-tokenizer.json",
        7_031_645,
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    "qwen_config": (
        "anima-qwen3-tokenizer-config.json",
        9_678,
        "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
    ),
    "t5": (
        "anima-t5-tokenizer.json",
        1_389_353,
        "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066",
    ),
}
# Git's text conversion can produce either of these byte-exact JSON resources.
# Keep each pair pinned: accepting a matching size alone would make corruption
# indistinguishable from an expected LF/CRLF checkout conversion.
ANIMA_TOKENIZER_ARTIFACT_VARIANTS = {
    "qwen": (
        ANIMA_TOKENIZER_ARTIFACTS["qwen"],
        ("anima-qwen3-tokenizer.json", 7_334_926, "47ec9be242d3ef39b9c97ac0a3f06c1752f061b234e295bc0a2842067a3fe4f9"),
    ),
    "qwen_config": (
        ANIMA_TOKENIZER_ARTIFACTS["qwen_config"],
        ("anima-qwen3-tokenizer-config.json", 9_916, "7992a7924330571ac9b97d58e39d4a4993ccdb865335034cec29cf2c482fd460"),
    ),
    "t5": (ANIMA_TOKENIZER_ARTIFACTS["t5"],),
}


class PagInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale: float = Field(default=PAG_SCALE, ge=0, le=5)
    applied_layers: Literal["mid", "all"] = PAG_APPLIED_LAYERS

    @field_validator("scale")
    @classmethod
    def normalize_scale(cls, value: float):
        if not math.isfinite(value):
            raise ValueError("PAG scale must be finite")
        return round(value, 2)


MAX_GALLERY_SETTINGS_BYTES = 128 * 1024
MAX_GALLERY_SETTINGS_DEPTH = 12
MAX_GALLERY_SETTINGS_ITEMS = 128


def normalize_gallery_settings_payload(value, depth: int = 0):
    """Keep gallery-only metadata bounded without imposing a limit on inference prompts."""
    if depth > MAX_GALLERY_SETTINGS_DEPTH:
        raise ValueError("gallery settings are nested too deeply")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("gallery settings must contain finite numbers")
        return value
    if isinstance(value, list):
        if len(value) > MAX_GALLERY_SETTINGS_ITEMS:
            raise ValueError("gallery settings contain too many list items")
        return [normalize_gallery_settings_payload(item, depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > MAX_GALLERY_SETTINGS_ITEMS:
            raise ValueError("gallery settings contain too many fields")
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise ValueError("gallery settings contain an invalid field name")
            normalized[key] = normalize_gallery_settings_payload(item, depth + 1)
        return normalized
    raise ValueError("gallery settings contain an unsupported value")


def validated_gallery_settings_payload(value):
    if value is None:
        return None
    normalized = normalize_gallery_settings_payload(value)
    if not isinstance(normalized, dict):
        raise ValueError("gallery settings must be an object")
    serialized = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(serialized.encode("utf-8")) > MAX_GALLERY_SETTINGS_BYTES:
        raise ValueError("gallery settings exceed the 128 KiB snapshot limit")
    return json.loads(serialized)


def gallery_access_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def has_gallery_access(job: dict, token: str | None) -> bool:
    digest = job.get("gallery_access_token_digest")
    return isinstance(token, str) and isinstance(digest, str) and hmac.compare_digest(digest, gallery_access_digest(token))


class GenerateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: Literal["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"]
    checkpoint: str | None = Field(default=None, min_length=1, max_length=500)
    diffusion_model: str | None = Field(default=None, min_length=1, max_length=500)
    text_encoder: str | None = Field(default=None, min_length=1, max_length=500)
    # FLUX.1 encodes a prompt with two encoders — CLIP-L for the pooled vector and T5-XXL for the
    # sequence — so it is the one engine that mounts a second text encoder. Which file is which is
    # resolved from the files themselves at load, not from the order they arrive in.
    text_encoder_2: str | None = Field(default=None, min_length=1, max_length=500)
    vae: str | None = Field(default=None, min_length=1, max_length=500)
    # A prompt has a floor and no ceiling. Every engine encodes at whatever length the prompt
    # tokenises to, so a character count here would be a policy invented by the request model
    # rather than a limit any model has — and one that refuses the whole generation, which is the
    # worst possible way to tell someone their prompt is long.
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    # SD/iL only, and strict: the layer the CLIP towers are read from, counted from the last. Absent
    # means the family default (SD 1, iL 2), which is the encoding every earlier record ran with.
    clip_skip: StrictInt | None = None
    # A relative path template beneath the output root, frozen with the job. Absent (or the default
    # template) keeps the historical date-folder naming and its `-02` collision suffixes.
    output_template: str | None = Field(default=None, max_length=400)
    # The ceiling here is the post-processing one. A sampling run is held to 64-pixel alignment, the
    # 8192 safety ceiling and its engine's own edge by `validate_dimensions`, which is where the two
    # contracts are told apart.
    width: int = Field(ge=64, le=MAX_POSTPROCESS_SOURCE_EDGE)
    height: int = Field(ge=64, le=MAX_POSTPROCESS_SOURCE_EDGE)
    steps: int = Field(ge=1, le=100)
    cfg: float = Field(ge=0, le=30)
    denoise: float = Field(ge=0, le=1)
    seed: int = Field(ge=0, le=0xFFFFFFFFFFFFFFFF)
    images_per_batch: int = Field(default=1, ge=1, le=10)
    batch_count: int = Field(default=1, ge=1, le=20)
    sampler: str
    scheduler: str
    guidance: Literal["none", "pag", "cfg_zero_star"] = "none"
    pag: PagInput = Field(default_factory=PagInput)
    background_removal_model: str | None = Field(default=None, max_length=300)
    source_image: SourceImageInput = Field(default_factory=SourceImageInput)
    # Forge-compatible opt-in semantics for the SD/iL base image-to-image pass. Off preserves the
    # historical Diffusers `int(steps * strength)` update count; on expands only the internal
    # schedule so the configured slider count is what actually executes.
    img2img_fix_steps: bool = False
    # Post-processing mode: run the enabled stages on the source picture and nothing else. The base
    # sampling pass is skipped outright rather than run at a denoise low enough to be invisible, so
    # the pixels the user brought are the pixels the first stage receives. Everything the stages
    # read — model, LoRAs, prompt, sampler, seed — is still the mounted configuration, which is why
    # this is a flag on the ordinary request rather than an endpoint of its own.
    postprocess_only: bool = False
    # The post-processing chain: stage instances in execution order. It replaces the three fixed
    # blocks and the separate order list, because with a variable number of resample units the two
    # could no longer describe the same thing — an order of stage *names* cannot say which of four
    # resample units runs third.
    postprocess_chain: list[StageInstance] = Field(default_factory=list, max_length=POSTPROCESS_CHAIN_LIMIT)
    loras: list[LoraInput] = Field(default_factory=list, max_length=16)
    # This client-only snapshot never affects inference. It is retained with the job so a completed
    # task reopened after a browser refresh can still be saved to the gallery with exact settings.
    gallery_settings: dict[str, object] | None = None

    # ADetailer and RTX stay single-instance, so naming one is unambiguous and the code that reads
    # their settings is unchanged by the chain. A resample unit has no such accessor on purpose:
    # there are up to eight of them and the question "which one" always has to be answered.
    @property
    def adetailer(self) -> ADetailerInput:
        return next((stage for stage in self.postprocess_chain if stage.type == "adetailer"), ABSENT_ADETAILER_STAGE)

    @property
    def rtx(self) -> RtxInput:
        return next((stage for stage in self.postprocess_chain if stage.type == "rtx"), ABSENT_RTX_STAGE)

    @property
    def resample_units(self) -> list[ResampleUnitInput]:
        return [stage for stage in self.postprocess_chain if stage.type == "resample"]

    @property
    def active_resample_units(self) -> list[ResampleUnitInput]:
        return [unit for unit in self.resample_units if unit.enabled]

    def resample_unit_number(self, unit: ResampleUnitInput) -> int:
        """Which resample card this is, 1-based — the number the user sees, not the chain index."""
        for number, candidate in enumerate(self.resample_units, start=1):
            if candidate is unit:
                return number
        return 0

    @field_validator("gallery_settings")
    @classmethod
    def validate_gallery_settings(cls, value):
        return validated_gallery_settings_payload(value)

    @field_validator("output_template")
    @classmethod
    def validate_output_template(cls, value):
        try:
            return normalize_output_template(value)
        except OutputPathError as error:
            raise ValueError(str(error)) from error

    @model_validator(mode="after")
    def validate_clip_skip(self):
        if self.clip_skip is None:
            return self
        minimum = CLIP_SKIP_DEFAULTS.get(self.engine)
        if minimum is None:
            # The native engines encode with language models or T5, not a CLIP stack a layer can be
            # skipped in. Accepting the field and ignoring it would be a control that does nothing.
            raise ValueError(f"{self.engine} does not support CLIP skip; it applies to SD and iL only")
        if not minimum <= self.clip_skip <= CLIP_SKIP_MAXIMUM:
            detail = "; SDXL/iL CLIP-L/G towers have a minimum effective skip of 2" if self.engine == "iL" else ""
            raise ValueError(f"{self.engine} CLIP skip must be between {minimum} and {CLIP_SKIP_MAXIMUM}{detail}")
        return self

    @model_validator(mode="after")
    def validate_chain_shape(self):
        counts = {name: 0 for name in POSTPROCESS_STAGE_TYPES}
        for stage in self.postprocess_chain:
            counts[stage.type] += 1
        if counts["resample"] > RESAMPLE_UNIT_LIMIT:
            raise ValueError(f"The resample chain accepts at most {RESAMPLE_UNIT_LIMIT} units")
        if counts["adetailer"] > 1:
            raise ValueError("The chain accepts at most one ADetailer stage")
        if counts["rtx"] > 1:
            raise ValueError("The chain accepts at most one RTX VSR stage")
        return self

    @model_validator(mode="after")
    def validate_dimensions(self):
        # A sampling canvas is a latent grid and keeps the alignment and the ceiling the memory
        # policy was measured against. A post-processing run has no latent grid of its own, so the
        # picture keeps its real size and is bounded by the same envelope the stages themselves are.
        if not self.postprocess_only:
            if self.width % SAMPLING_EDGE_STEP != 0 or self.height % SAMPLING_EDGE_STEP != 0:
                raise ValueError("Image dimensions must be divisible by 64")
            ceiling = min(MAX_SAMPLING_EDGE, engine_max_refinement_edge(request_family(self.engine)))
            if self.width > ceiling or self.height > ceiling:
                raise ValueError(f"{self.engine} image dimensions cannot exceed {ceiling}")
        elif self.width * self.height > MAX_POSTPROCESS_SOURCE_PIXELS:
            raise ValueError("Post-processing source exceeds the safe 8192-edge / 32-megapixel limit")
        return self

    @model_validator(mode="after")
    def validate_postprocess_only(self):
        if not self.postprocess_only:
            return self
        # Both halves of the mode have to hold: without a picture there is nothing to enhance, and
        # without a stage the run would load a model, skip sampling and save the input unchanged.
        if not self.source_image.enabled:
            raise ValueError("Post-processing mode requires a source image")
        if not postprocessing_stages(self):
            raise ValueError("Post-processing mode requires at least one enabled resample, ADetailer, or RTX VSR stage")
        return self

    @model_validator(mode="after")
    def validate_prompt_directives(self):
        cleaned_prompt, directives = parse_prompt_directives(self.prompt)
        if directives["transparent_background"] and not cleaned_prompt:
            raise ValueError("透明背景特殊标签之外还需要填写主体提示词")
        # `denoise` is the image-to-image strength. Text-to-image has always ignored it, so it is
        # only now that zero becomes a contradiction rather than a no-op: it would ask the sampler
        # to change nothing at all.
        if self.source_image.enabled and self.denoise <= 0:
            raise ValueError("Image-to-image denoise must be greater than 0")
        split_fields = (self.diffusion_model, self.text_encoder, self.vae)
        if self.engine in {"SD", "iL"}:
            if not self.checkpoint:
                raise ValueError("SD / iL requires checkpoint")
            if any(value is not None for value in split_fields) or self.text_encoder_2 is not None:
                raise ValueError("SD / iL forbids diffusion_model, text_encoder, text_encoder_2, and vae")
            if self.guidance == "cfg_zero_star":
                raise ValueError("CFG-Zero* 仅适用于 Flow Matching 模型；当前 SD / iL 不兼容")
        elif self.engine == "Anima":
            if self.checkpoint is not None:
                raise ValueError("Anima forbids checkpoint; use diffusion_model, text_encoder, and vae")
            if any(value is None for value in split_fields):
                raise ValueError("Anima requires diffusion_model, text_encoder, and vae")
            if self.text_encoder_2 is not None:
                raise ValueError("Anima mounts one text encoder; text_encoder_2 belongs to Flux")
            if self.sampler not in ANIMA_SAMPLERS:
                raise ValueError(f"Unsupported Anima sampler: {self.sampler}")
            if self.scheduler not in ANIMA_SCHEDULERS:
                raise ValueError(f"Unsupported Anima scheduler: {self.scheduler}")
        elif self.engine == "Flux":
            if self.checkpoint is not None:
                raise ValueError("Flux forbids checkpoint; use diffusion_model, text_encoder, text_encoder_2, and vae")
            if any(value is None for value in split_fields) or self.text_encoder_2 is None:
                raise ValueError("Flux requires diffusion_model, text_encoder, text_encoder_2, and vae")
            if self.text_encoder == self.text_encoder_2:
                raise ValueError("Flux requires two different text encoders: CLIP-L and T5-XXL")
            if self.sampler not in FLUX_SAMPLERS:
                raise ValueError(f"Unsupported Flux sampler: {self.sampler}")
            if self.scheduler not in FLUX_SCHEDULERS:
                raise ValueError(f"Unsupported Flux scheduler: {self.scheduler}")
            # FLUX.1 is guidance distilled: one forward per step, steered by a scalar baked into the
            # timestep embedding. There is no unconditional branch, so a negative prompt would be
            # silently discarded and PAG / CFG-Zero* have nothing to perturb.
            if self.guidance != "none":
                raise ValueError("Flux is guidance distilled and does not support PAG or CFG-Zero*; select None")
            if self.negative_prompt:
                raise ValueError("Flux has no unconditional branch; the negative prompt must be empty")
            # A detail pass encodes its own negative prompt through the same runtime, so a per-unit
            # one is the same contradiction as the main one — and failing here is failing at submit
            # rather than several stages into a job that has already sampled.
            if any(unit.negative_prompt for unit in self.adetailer.active_units):
                raise ValueError("Flux has no unconditional branch; ADetailer negative prompts must be empty")
        elif self.engine == "Flux2":
            if self.checkpoint is not None:
                raise ValueError("Flux2 forbids checkpoint; use diffusion_model, text_encoder, and vae")
            if any(value is None for value in split_fields):
                raise ValueError("Flux2 requires diffusion_model, text_encoder, and vae")
            # FLUX.2 conditions on one language model, not on a CLIP tower beside a T5.
            if self.text_encoder_2 is not None:
                raise ValueError("Flux2 mounts one text encoder; text_encoder_2 belongs to Flux")
            if self.sampler not in FLUX2_SAMPLERS:
                raise ValueError(f"Unsupported Flux2 sampler: {self.sampler}")
            if self.scheduler not in FLUX2_SCHEDULERS:
                raise ValueError(f"Unsupported Flux2 scheduler: {self.scheduler}")
            # FLUX.2 is guidance distilled for the same reason FLUX.1 is, and refuses the same
            # three settings rather than encoding them and throwing the result away.
            if self.guidance != "none":
                raise ValueError("Flux2 is guidance distilled and does not support PAG or CFG-Zero*; select None")
            if self.negative_prompt:
                raise ValueError("Flux2 has no unconditional branch; the negative prompt must be empty")
            if any(unit.negative_prompt for unit in self.adetailer.active_units):
                raise ValueError("Flux2 has no unconditional branch; ADetailer negative prompts must be empty")
        else:
            if self.checkpoint is not None:
                raise ValueError("Krea2 forbids checkpoint; use diffusion_model, text_encoder, and vae")
            if any(value is None for value in split_fields):
                raise ValueError("Krea2 requires diffusion_model, text_encoder, and vae")
            # Krea 2 conditions on one Qwen3-VL, read at twelve depths rather than at its output.
            if self.text_encoder_2 is not None:
                raise ValueError("Krea2 mounts one text encoder; text_encoder_2 belongs to Flux")
            if self.sampler not in KREA2_SAMPLERS:
                raise ValueError(f"Unsupported Krea2 sampler: {self.sampler}")
            if self.scheduler not in KREA2_SCHEDULERS:
                raise ValueError(f"Unsupported Krea2 scheduler: {self.scheduler}")
            # Unlike the Flux engines, Krea 2 has a real unconditional branch, so the negative
            # prompt and CFG-Zero* both work. PAG does not: it needs an identity-self-attention
            # override of the transformer's own blocks, which this runtime does not install.
            if self.guidance == "pag":
                raise ValueError("Krea2 does not support PAG; select None or CFG-Zero*")
        family = request_family(self.engine)
        distilled = self.engine in DISTILLED_GUIDANCE_ENGINES
        for number, unit in enumerate(self.resample_units, start=1):
            if not unit.enabled:
                continue
            if unit.latent_algorithm is not None and self.engine not in {"SD", "iL"}:
                raise ValueError(
                    f"Resample unit {number}: 潜空间放大目前仅支持 SD/iL；{self.engine} 暂不支持"
                )
            validate_resample_sampling_override(unit.sampler, unit.scheduler, family, number)
            if unit.execution_mode == "usdu_tiled" and not supports_usdu_tiled(family):
                raise ValueError(f"USDU tiled resampling in resample unit {number} is supported only by Anima and Krea2")
            # A model-less unit is a plain resize and never reaches a sampler, so the settings that
            # only a redraw can honour are refused rather than silently ignored.
            if not unit.redraws and (unit.sampler is not None or unit.scheduler is not None):
                raise ValueError(f"Resample unit {number} sets a sampler but does not redraw (denoise is 0)")
            if not unit.redraws and unit.execution_mode == "usdu_tiled":
                raise ValueError(f"Resample unit {number} asks for USDU tiling but does not redraw (denoise is 0)")
            if distilled and unit.redraws and self.negative_prompt:
                raise ValueError(f"{self.engine} has no unconditional branch; resample unit {number} cannot carry a negative prompt")
        # Walk the chain the way the run will, accumulating sizes. `plan` is the same computation the
        # stage itself performs, so a chain accepted here is one the engines can actually execute.
        for entry in resample_chain_plan(self):
            if entry["type"] == "resample":
                number = entry["unit_number"]
                if max(entry["redraw_size"]) > engine_max_refinement_edge(family) and entry["redraws"]:
                    raise ValueError(
                        f"Resample unit {number} reaches {entry['redraw_size'][0]}x{entry['redraw_size'][1]}, "
                        f"past the {self.engine} refinement limit of {engine_max_refinement_edge(family)}"
                    )
            size = entry["output_size"]
            if max(size) > MAX_POSTPROCESS_SOURCE_EDGE or size[0] * size[1] > MAX_POSTPROCESS_SOURCE_PIXELS:
                where = f"Resample unit {entry['unit_number']}" if entry["type"] == "resample" else f"The {entry['type']} stage"
                raise ValueError(f"{where} exceeds the safe 8192-edge / 32-megapixel limit")
        return self


def effective_clip_skip(request: GenerateInput) -> int | None:
    """The CLIP layer this job's SD/iL passes read, frozen for base, image-to-image, Hires and ADetailer."""
    default = CLIP_SKIP_DEFAULTS.get(request.engine)
    if default is None:
        return None
    return default if request.clip_skip is None else request.clip_skip


class PerformanceInput(BaseModel):
    # The field list is the settings file's, not a subset of it: `model_dump()` is written straight
    # to disk, so a setting missing here is a setting the UI can never turn on — it is dropped on
    # the way in and normalised back to its default on the way out.
    memory_mode: Literal["auto", "high_vram", "sdxl_balanced", "normal_vram", "low_vram", "ultra_low_vram"] = "auto"
    attention_backend: Literal["auto", "sdpa", "xformers", "sage", "sliced"] = "auto"
    compute_dtype: Literal["fp16", "bf16"] = "fp16"
    vae_mode: Literal["auto", "full", "sliced", "tiled"] = "auto"
    cuda_math: Literal["balanced", "strict"] = "balanced"
    keep_model_cached: bool = True
    allow_shared_memory: bool = True
    calculate_model_hash: bool = False
    staged_vae_decode: bool = False
    compile_transformer: bool = False
    # The decoder is the one stage that wants the whole card: it works at full resolution while
    # the transformer, which has nothing left to do once the last step is taken, is still holding
    # its weights. Evicting it first is what lets a large canvas decode in one piece instead of in
    # tiles, so it is the default rather than an opt-in.
    unload_before_decode: bool = True
    vram_limit_gb: float = Field(default=0.0, ge=0.0, le=1024.0)


COLLAGE_MAX_DECODED_BYTES = 128 * 1024 * 1024
COLLAGE_MAX_DATA_URL_CHARS = ((COLLAGE_MAX_DECODED_BYTES + 2) // 3) * 4 + 64
COLLAGE_MAX_PIXELS = 64 * 1024 * 1024
COLLAGE_MAX_GIF_FRAMES = 240
COLLAGE_MAX_TOTAL_FRAME_PIXELS = 120_000_000
COLLAGE_MAX_MANUAL_LAYOUT_BYTES = 1024 * 1024
COLLAGE_MAX_LAYOUT_LAYERS = 100
COLLAGE_MAX_LAYOUT_STROKES = 1000
COLLAGE_MAX_STROKE_POINTS = 10_000
COLLAGE_MAX_LAYOUT_POINTS = 50_000
COLLAGE_MAX_ANIMATED_SOURCE_BYTES = 32 * 1024 * 1024
COLLAGE_MAX_ANIMATED_INPUT_BYTES = 128 * 1024 * 1024
COLLAGE_MAX_ANIMATED_SOURCE_PIXELS = 64 * 1024 * 1024
COLLAGE_MAX_ANIMATED_INPUT_PIXELS = 128 * 1024 * 1024
COLLAGE_MAX_ANIMATED_SOURCE_FRAMES = 240
COLLAGE_MAX_ANIMATED_INPUT_FRAMES = 1000
COLLAGE_MAX_ANIMATED_FRAME_PIXELS = 256 * 1024 * 1024
COLLAGE_MAX_ANIMATED_TARGET_PIXELS = 128 * 1024 * 1024
COLLAGE_MAX_ENCODED_OUTPUT_BYTES = 128 * 1024 * 1024
COLLAGE_MAX_HISTORY_COPY_BYTES = 128 * 1024 * 1024
COLLAGE_MAX_LAYOUT_SOURCE_BYTES = 256 * 1024 * 1024
COLLAGE_MAX_LAYOUT_SOURCE_PIXELS = 128 * 1024 * 1024
COLLAGE_MAX_LAYOUT_SOURCE_FRAMES = 1000
COLLAGE_MAX_LAYOUT_SOURCE_FRAME_PIXELS = 256 * 1024 * 1024
COLLAGE_MAX_REQUEST_BYTES = max(COLLAGE_MAX_DATA_URL_CHARS, ((COLLAGE_MAX_ANIMATED_INPUT_BYTES + 2) // 3) * 4) + COLLAGE_MAX_MANUAL_LAYOUT_BYTES + 2 * 1024 * 1024
COLLAGE_GIF_DURATION_DEFAULT_MS = 100
COLLAGE_GIF_DURATION_MIN_MS = 20
COLLAGE_GIF_DURATION_MAX_MS = 60_000

_COLLAGE_HISTORY_SOURCE = re.compile(r"^/api/inference/history/assets/[A-Za-z0-9_-]+$")
_COLLAGE_IMAGE_LAYER_KEYS = {
    "kind", "type", "assetId", "url", "originalUrl", "name", "naturalWidth", "naturalHeight",
    "x", "y", "scale", "rotation", "mimeType", "paintStrokes",
}
_COLLAGE_STROKE_KEYS = {"id", "tool", "color", "size", "opacity", "points"}


class CollageResourceLimitError(ValueError):
    pass


def normalize_collage_gif_duration(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return COLLAGE_GIF_DURATION_DEFAULT_MS
    try:
        duration = float(value)
    except (TypeError, ValueError, OverflowError):
        return COLLAGE_GIF_DURATION_DEFAULT_MS
    if not math.isfinite(duration) or duration <= 0:
        return COLLAGE_GIF_DURATION_DEFAULT_MS
    return int(round(min(COLLAGE_GIF_DURATION_MAX_MS, max(COLLAGE_GIF_DURATION_MIN_MS, duration))))


class BoundedOutputWriter:
    def __init__(self, stream, maximum_bytes: int):
        self.stream = stream
        self.maximum_bytes = maximum_bytes
        self.length = 0

    def write(self, data):
        projected = max(self.length, self.stream.tell() + len(data))
        if projected > self.maximum_bytes:
            raise CollageResourceLimitError("Encoded collage output exceeds the 128 MiB budget")
        written = self.stream.write(data)
        self.length = max(self.length, self.stream.tell())
        return written

    def seek(self, *args):
        return self.stream.seek(*args)

    def tell(self):
        return self.stream.tell()

    def flush(self):
        return self.stream.flush()

    def truncate(self, *args):
        result = self.stream.truncate(*args)
        self.length = min(self.length, self.stream.tell())
        return result

    def __getattr__(self, name):
        return getattr(self.stream, name)


def bounded_output_buffer(maximum_bytes: int | None = None):
    stream = io.BytesIO()
    return stream, BoundedOutputWriter(stream, maximum_bytes or COLLAGE_MAX_ENCODED_OUTPUT_BYTES)


class CollageRequestBodyLimitMiddleware:
    def __init__(self, app, maximum_bytes: int):
        self.app = app
        self.maximum_bytes = maximum_bytes

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("path") not in {"/api/inference/collages", "/api/inference/collages/animated"}:
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = 0
        if declared > self.maximum_bytes:
            response = Response(content=json.dumps({"detail": "Collage request body exceeds the upload budget"}), status_code=413, media_type="application/json")
            await response(scope, receive, send)
            return
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.maximum_bytes:
                raise CollageResourceLimitError("Collage request body exceeds the upload budget")
            return message

        try:
            await self.app(scope, limited_receive, send)
        except CollageResourceLimitError:
            response = Response(content=json.dumps({"detail": "Collage request body exceeds the upload budget"}), status_code=413, media_type="application/json")
            await response(scope, receive, send)


app.add_middleware(CollageRequestBodyLimitMiddleware, maximum_bytes=COLLAGE_MAX_REQUEST_BYTES)


def _collage_finite_number(value, label: str):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"manual_layout {label} must be a finite number")
    return float(value)


def history_image_resource_metrics(path: Path, maximum_bytes: int, label: str, decode_frames: bool = False):
    try:
        file_bytes = path.stat().st_size
    except OSError as error:
        raise ValueError(f"Unable to inspect {label}") from error
    if file_bytes > maximum_bytes:
        raise CollageResourceLimitError(f"{label} exceeds the file byte budget")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = image.size
                frame_count = max(1, int(getattr(image, "n_frames", 1)))
                image_format = image.format
                if decode_frames:
                    for frame_index in range(frame_count):
                        image.seek(frame_index)
                        image.load()
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise CollageResourceLimitError(f"{label} triggered the decompression-bomb guard") from error
    except (MemoryError, OverflowError) as error:
        raise CollageResourceLimitError(f"{label} could not be decoded inside the allocation budget") from error
    except (OSError, SyntaxError, ValueError) as error:
        raise ValueError(f"Unable to read {label}") from error
    pixels = width * height
    frame_pixels = pixels * frame_count
    if pixels > COLLAGE_MAX_PIXELS or frame_count > COLLAGE_MAX_GIF_FRAMES or frame_pixels > COLLAGE_MAX_TOTAL_FRAME_PIXELS:
        raise CollageResourceLimitError(f"{label} exceeds the pixel or GIF frame budget")
    return {"bytes": file_bytes, "pixels": pixels, "frames": frame_count, "frame_pixels": frame_pixels, "format": image_format}


def validate_manual_layout_schema(layout: dict | None):
    if layout is None:
        return None
    if not isinstance(layout, dict) or set(layout) != {"version", "layers"}:
        raise ValueError("manual_layout must contain only version and layers")
    version = layout.get("version")
    layers = layout.get("layers")
    if isinstance(version, bool) or version not in (1, 2):
        raise ValueError("manual_layout version must be 1 or 2")
    if not isinstance(layers, list) or not 1 <= len(layers) <= COLLAGE_MAX_LAYOUT_LAYERS:
        raise ValueError(f"manual_layout must contain 1 to {COLLAGE_MAX_LAYOUT_LAYERS} layers")
    stroke_count = 0
    point_count = 0
    source_bytes = 0
    source_pixels = 0
    source_frames = 0
    source_frame_pixels = 0
    for layer_index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"manual_layout layer {layer_index} must be an object")
        # Text layers were withdrawn from the editor, so no client may still submit one; a layout
        # saved while they existed is read back with its text layers dropped, never re-saved.
        if layer.get("kind") == "text" or layer.get("type") == "text":
            raise ValueError("manual_layout cannot contain text layers")
        if not set(layer).issubset(_COLLAGE_IMAGE_LAYER_KEYS):
            raise ValueError(f"manual_layout layer {layer_index} contains unsupported fields")
        for key in ("x", "y", "scale", "rotation", "naturalWidth", "naturalHeight"):
            if key in layer:
                number = _collage_finite_number(layer[key], f"layer {layer_index}.{key}")
                if key in {"x", "y"} and abs(number) > 10_000_000:
                    raise ValueError(f"manual_layout layer {layer_index}.{key} is outside the coordinate budget")
                if key == "scale" and not 0.1 <= number <= 8:
                    raise ValueError(f"manual_layout layer {layer_index}.scale is outside 0.1 to 8")
                if key == "rotation" and not -180 <= number <= 180:
                    raise ValueError(f"manual_layout layer {layer_index}.rotation is outside -180 to 180")
                if key in {"naturalWidth", "naturalHeight"} and not 1 <= number <= 24576:
                    raise ValueError(f"manual_layout layer {layer_index}.{key} exceeds the dimension budget")
        if "naturalWidth" in layer and "naturalHeight" in layer and float(layer["naturalWidth"]) * float(layer["naturalHeight"]) > COLLAGE_MAX_PIXELS:
            raise ValueError(f"manual_layout layer {layer_index} exceeds the pixel budget")
        name = layer.get("name", "")
        if not isinstance(name, str) or len(name) > 200:
            raise ValueError(f"manual_layout layer {layer_index} name is invalid")
        asset_id = layer.get("assetId")
        if not isinstance(asset_id, str) or not asset_id or len(asset_id) > 512:
            raise ValueError(f"manual_layout image layer {layer_index} requires assetId")
        sources = [layer[key] for key in ("url", "originalUrl") if key in layer and layer[key]]
        if not sources or any(not isinstance(source, str) or not _COLLAGE_HISTORY_SOURCE.fullmatch(source) for source in sources):
            raise ValueError("manual_layout image sources must use protected history asset routes")
        source_tokens = [source.rsplit("/", 1)[-1] for source in sources]
        if any(token != asset_id for token in source_tokens):
            raise ValueError("manual_layout assetId must match every history source token")
        try:
            source_path = history_asset_path(asset_id)
        except HTTPException as error:
            raise ValueError("manual_layout image source token is invalid") from error
        if not source_path.is_file():
            raise ValueError("manual_layout image source is missing")
        metrics = history_image_resource_metrics(source_path, COLLAGE_MAX_HISTORY_COPY_BYTES, "manual_layout history source")
        source_bytes += metrics["bytes"]
        source_pixels += metrics["pixels"]
        source_frames += metrics["frames"]
        source_frame_pixels += metrics["frame_pixels"]
        if source_bytes > COLLAGE_MAX_LAYOUT_SOURCE_BYTES or source_pixels > COLLAGE_MAX_LAYOUT_SOURCE_PIXELS or source_frames > COLLAGE_MAX_LAYOUT_SOURCE_FRAMES or source_frame_pixels > COLLAGE_MAX_LAYOUT_SOURCE_FRAME_PIXELS:
            raise CollageResourceLimitError("manual_layout history sources exceed the aggregate byte, pixel, or frame budget")
        strokes = layer.get("paintStrokes", [])
        if not isinstance(strokes, list):
            raise ValueError("manual_layout paintStrokes must be a list")
        if len(strokes) > COLLAGE_MAX_LAYOUT_STROKES - stroke_count:
            raise ValueError(f"manual_layout exceeds {COLLAGE_MAX_LAYOUT_STROKES} strokes")
        for stroke in strokes:
            if not isinstance(stroke, dict) or not set(stroke).issubset(_COLLAGE_STROKE_KEYS):
                raise ValueError("manual_layout contains an invalid stroke")
            points = stroke.get("points")
            if not isinstance(points, list) or len(points) > COLLAGE_MAX_STROKE_POINTS or point_count + len(points) > COLLAGE_MAX_LAYOUT_POINTS:
                raise ValueError("manual_layout exceeds the stroke point budget")
            if stroke.get("tool", "brush") not in {"brush", "eraser"}:
                raise ValueError("manual_layout stroke tool is invalid")
            for key in ("size", "opacity"):
                if key in stroke:
                    number = _collage_finite_number(stroke[key], f"stroke.{key}")
                    if key == "size" and not 1 <= number <= 300:
                        raise ValueError("manual_layout stroke size is outside 1 to 300")
                    if key == "opacity" and not 0.01 <= number <= 1:
                        raise ValueError("manual_layout stroke opacity is outside 0.01 to 1")
            for key, maximum in (("id", 96), ("color", 16)):
                if key in stroke and (not isinstance(stroke[key], str) or len(stroke[key]) > maximum):
                    raise ValueError(f"manual_layout stroke field {key} is invalid")
            for point in points:
                if not isinstance(point, dict) or set(point) != {"x", "y"}:
                    raise ValueError("manual_layout contains an invalid stroke point")
                _collage_finite_number(point["x"], "stroke point x")
                _collage_finite_number(point["y"], "stroke point y")
            stroke_count += 1
            point_count += len(points)
    try:
        encoded = json.dumps(layout, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("manual_layout is not JSON serializable") from error
    if len(encoded) > COLLAGE_MAX_MANUAL_LAYOUT_BYTES:
        raise ValueError("manual_layout exceeds the 1 MiB metadata budget")
    return layout


class CollageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_data: str = Field(min_length=32, max_length=COLLAGE_MAX_DATA_URL_CHARS)
    name: str = Field(default="XirAI-collage.png", max_length=160)
    manual_layout: dict | None = None
    idempotency_key: str = Field(default="", max_length=96, pattern=r"^[A-Za-z0-9._:-]*$")

    @field_validator("manual_layout", mode="before")
    @classmethod
    def validate_layout(cls, layout):
        return validate_manual_layout_schema(layout)


class AnimatedCollageClip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=-24576, le=24576)
    y: float = Field(ge=-24576, le=24576)
    width: float = Field(gt=0, le=24576)
    height: float = Field(gt=0, le=24576)


class AnimatedCollageLayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=32, max_length=((COLLAGE_MAX_ANIMATED_SOURCE_BYTES + 2) // 3) * 4 + 64)
    x: float = Field(default=0, ge=-24576, le=24576)
    y: float = Field(default=0, ge=-24576, le=24576)
    width: float = Field(gt=0, le=24576)
    height: float = Field(gt=0, le=24576)
    clip: AnimatedCollageClip | None = None
    hidden_sides: list[Literal["top", "right", "bottom", "left"]] = Field(default_factory=list, max_length=4)


class AnimatedCollageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layers: list[AnimatedCollageLayer] = Field(min_length=1, max_length=100)
    width: int = Field(ge=1, le=4096)
    height: int = Field(ge=1, le=4096)
    edge_line: dict | None = None

    @field_validator("layers", mode="before")
    @classmethod
    def validate_raw_encoded_budget(cls, layers):
        if not isinstance(layers, list) or not 1 <= len(layers) <= 100:
            return layers
        encoded_total = 0
        for layer in layers:
            if not isinstance(layer, dict) or not isinstance(layer.get("url"), str):
                continue
            encoded_total += len(layer["url"].split(",", 1)[-1])
            if encoded_total > ((COLLAGE_MAX_ANIMATED_INPUT_BYTES + 2) // 3) * 4:
                raise ValueError("Animated collage encoded input exceeds its total budget")
        return layers

    @model_validator(mode="after")
    def validate_encoded_budget(self):
        target_pixels = 0
        for layer in self.layers:
            if not re.match(r"^data:image/(?:png|gif|jpeg|webp);base64,", layer.url, re.IGNORECASE):
                raise ValueError("Animated collage sources must be supported image data URLs")
            if math.ceil(layer.width) * math.ceil(layer.height) > COLLAGE_MAX_PIXELS:
                raise ValueError("Animated collage layer target exceeds the pixel budget")
            target_pixels += math.ceil(layer.width) * math.ceil(layer.height)
            if target_pixels > COLLAGE_MAX_ANIMATED_TARGET_PIXELS:
                raise ValueError("Animated collage layers exceed the aggregate target pixel budget")
        if self.width * self.height > COLLAGE_MAX_PIXELS:
            raise ValueError("Animated collage output exceeds the pixel budget")
        return self


class HistoryDeleteInput(BaseModel):
    asset_ids: list[str] = Field(min_length=1, max_length=200)
    delete_source: bool = False


class GalleryCollectionCreateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    description: str | None = Field(default=None, max_length=1000)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str):
        value = value.strip()
        if not 1 <= len(value) <= 64:
            raise ValueError("Collection id must contain 1 to 64 characters")
        return value


class GalleryCollectionUpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str | None = None
    description: str | None = Field(default=None, max_length=1000)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str | None):
        if value is None:
            raise ValueError("Collection id cannot be null")
        value = value.strip()
        if not 1 <= len(value) <= 64:
            raise ValueError("Collection id must contain 1 to 64 characters")
        return value

    @model_validator(mode="after")
    def validate_update(self):
        if not self.model_fields_set:
            raise ValueError("At least one collection field is required")
        return self


class GalleryImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    gallery_image_id: str | None = Field(default=None, min_length=1, max_length=64)
    asset_id: str | None = Field(default=None, min_length=1, max_length=4096)
    data_url: str | None = Field(default=None, min_length=1, max_length=70_000_000)
    name: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def validate_source(self):
        sources = (self.gallery_image_id, self.asset_id, self.data_url)
        if sum(value is not None for value in sources) != 1:
            raise ValueError("Exactly one of gallery_image_id, asset_id, or data_url is required")
        return self


class GalleryCardCreateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    collection_id: str
    title: str | None = Field(default=None, max_length=160)
    settings: dict = Field(default_factory=dict)
    images: list[GalleryImageInput] = Field(default_factory=list)

    @field_validator("collection_id")
    @classmethod
    def validate_collection_id(cls, value: str):
        value = value.strip()
        if not 1 <= len(value) <= 64:
            raise ValueError("Collection id must contain 1 to 64 characters")
        return value

    @model_validator(mode="after")
    def validate_upload_size(self):
        if sum(len(image.data_url or "") for image in self.images) > 140_000_000:
            raise ValueError("Gallery image uploads must not exceed 100 MiB total")
        return self


class GalleryCardBatchItemInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str | None = Field(default=None, max_length=160)
    settings: dict = Field(default_factory=dict)
    images: list[GalleryImageInput] = Field(default_factory=list)


class GalleryCardsCreateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    collection_id: str
    cards: list[GalleryCardBatchItemInput] = Field(min_length=1)

    @field_validator("collection_id")
    @classmethod
    def validate_collection_id(cls, value: str):
        value = value.strip()
        if not 1 <= len(value) <= 64:
            raise ValueError("Collection id must contain 1 to 64 characters")
        return value

    @model_validator(mode="after")
    def validate_upload_size(self):
        if sum(len(image.data_url or "") for card in self.cards for image in card.images) > 140_000_000:
            raise ValueError("Gallery image uploads must not exceed 100 MiB total")
        return self


class GalleryCardUpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    collection_id: str | None = None
    title: str | None = Field(default=None, max_length=160)
    settings: dict | None = None
    images: list[GalleryImageInput] | None = None

    @field_validator("collection_id")
    @classmethod
    def validate_collection_id(cls, value: str | None):
        if value is None:
            raise ValueError("collection_id cannot be null")
        value = value.strip()
        if not 1 <= len(value) <= 64:
            raise ValueError("Collection id must contain 1 to 64 characters")
        return value

    @model_validator(mode="after")
    def validate_update(self):
        if not self.model_fields_set:
            raise ValueError("At least one card field is required")
        if "settings" in self.model_fields_set and self.settings is None:
            raise ValueError("settings must be a JSON object")
        if "images" in self.model_fields_set and self.images is None:
            raise ValueError("images must be a list")
        if self.images and sum(len(image.data_url or "") for image in self.images) > 140_000_000:
            raise ValueError("Gallery image uploads must not exceed 100 MiB total")
        return self


class GalleryCardOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    card_ids: list[str] = Field(min_length=1)

    @field_validator("card_ids")
    @classmethod
    def validate_card_ids(cls, value: list[str]):
        if any(not card_id or len(card_id) > 64 for card_id in value):
            raise ValueError("Every card id must contain 1 to 64 characters")
        if len(set(value)) != len(value):
            raise ValueError("card_ids must not contain duplicates")
        return value


class GalleryPromptCreateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(min_length=1, max_length=160)
    # The prompts carry no ceiling for the same reason a generation's do not; the title and notes
    # keep theirs because they are labels rather than prompts.
    positive_prompt: str = ""
    negative_prompt: str = ""
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str):
        value = value.strip()
        if not value:
            raise ValueError("Prompt title is required")
        return value

    @model_validator(mode="after")
    def validate_prompts(self):
        if not self.positive_prompt.strip() and not self.negative_prompt.strip():
            raise ValueError("At least one positive or negative prompt is required")
        return self


class GalleryPromptUpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str | None = Field(default=None, min_length=1, max_length=160)
    positive_prompt: str | None = None
    negative_prompt: str | None = None
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None):
        if value is None:
            raise ValueError("Prompt title cannot be null")
        value = value.strip()
        if not value:
            raise ValueError("Prompt title is required")
        return value

    @model_validator(mode="after")
    def validate_update(self):
        if not self.model_fields_set:
            raise ValueError("At least one prompt field is required")
        for field in ("positive_prompt", "negative_prompt"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        return self


JOB_ACTIVE_STATUSES = frozenset({"queued", "running", "pausing", "paused", "cancelling"})
JOB_FINISHED_STATUSES = frozenset({"complete", "error", "cancelled"})
# Fields that name a place on this machine rather than something a client can fetch.
JOB_PRIVATE_FIELDS = frozenset({
    "output_path", "output_paths", "stage_preview_paths", "gallery_settings", "gallery_access_token_digest",
})


def public_job_view(job: dict) -> dict:
    return {key: value for key, value in job.items() if key not in JOB_PRIVATE_FIELDS}


def trim_job_history():
    """Forget the oldest finished jobs once the queue history outgrows its limit.

    Called with `jobs_lock` held. Only finished jobs are candidates: a queued job trimmed away
    would still run, with nothing left to report its progress to.
    """
    finished = [job for job in jobs.values() if job["status"] in JOB_FINISHED_STATUSES]
    excess = len(finished) - JOB_HISTORY_LIMIT
    if excess <= 0:
        return
    for job in sorted(finished, key=lambda item: item.get("sequence", 0))[:excess]:
        jobs.pop(job["id"], None)
        job_controls.pop(job["id"], None)
        # The row is gone, so the pictures it pointed at have nothing left to reach them.
        cleanup_job_stage_previews(job["id"])


def job_output_summary(output: dict | None) -> dict | None:
    if not output:
        return None
    return {
        "index": output.get("index"),
        "output_name": output.get("output_name"),
        "output_relative": output.get("output_relative"),
        "image_url": output.get("image_url"),
        "seed": output.get("seed"),
        "width": output.get("width"),
        "height": output.get("height"),
        "asset_id": output.get("asset_id"),
        "transparent_background": output.get("transparent_background"),
    }


def job_stage_preview_summary(preview: dict | None) -> dict | None:
    if not preview:
        return None
    return {
        key: preview.get(key)
        for key in ("index", "stage", "url", "label", "width", "height", "batch_index", "image_index")
        if preview.get(key) is not None
    }


def job_queue_entry(job: dict, running_seen: bool) -> dict:
    """One bounded row of the queue list: what it is, where it got to, and one thumbnail.

    `queued` is reported as its own state rather than folded into `running`, because the two mean
    different things to someone deciding whether to submit another: one is being worked on, the
    rest are waiting behind it.
    """
    outputs = job.get("outputs") or []
    stage_previews = job.get("stage_previews") or []
    return {
        "id": job["id"],
        "sequence": job.get("sequence"),
        "status": job["status"],
        "phase": job.get("phase") or "",
        "stage": job.get("stage") or "",
        "stage_step": job.get("stage_step") or 0,
        "stage_total": job.get("stage_total") or 0,
        "progress": job.get("progress") or 0,
        "step": job.get("step") or 0,
        "total_steps": job.get("total_steps") or 0,
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "elapsed_seconds": job.get("elapsed_seconds"),
        "error": job.get("error"),
        "engine": job.get("requested_engine"),
        "checkpoint": job.get("requested_checkpoint"),
        "model_assets": job.get("requested_model_assets") or {},
        "width": job.get("width"),
        "height": job.get("height"),
        "seed": job.get("seed"),
        "total_images": job.get("total_images"),
        "completed_images": job.get("completed_images"),
        "images_per_batch": job.get("images_per_batch"),
        "batch_count": job.get("batch_count"),
        "batch_index": job.get("batch_index"),
        # The list is polled repeatedly. Keep it bounded to one thumbnail; reopening a row fetches
        # the full result detail once instead of resending all prior previews and images every tick.
        "stage_preview": job_stage_preview_summary(stage_previews[-1] if stage_previews else None),
        "output": job_output_summary(outputs[0] if outputs else None),
        "output_count": len(outputs),
        # A queued job's place in the line, counting only what is still ahead of it.
        "waiting": job["status"] == "queued",
        "active": job["status"] in JOB_ACTIVE_STATUSES,
        "running_ahead": running_seen,
    }


def update_job(job_id: str, **updates):
    # Every stage reports its step count through here — base sampling, Hires.fix, ADetailer, the
    # tiled VAE decode — so this is the one place the consoles have to be fed from. The line is
    # composed under the lock, where the record is consistent, and printed outside it, so a slow
    # or blocked console cannot stall a sampler callback that is holding up the whole job.
    line = None
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return
        job.update(updates)
        line = progress_console.observe(job_id, job, updates)
    if line is not None:
        progress_console.write(line)


def history_asset_token(path: Path):
    relative = path.resolve().relative_to(OUTPUT_DIRECTORY.resolve()).as_posix().encode("utf-8")
    return base64.urlsafe_b64encode(relative).decode("ascii").rstrip("=")


def history_folder_token(path: Path):
    resolved = path.resolve()
    relative = resolved.relative_to(OUTPUT_DIRECTORY.resolve()).as_posix() if resolved != OUTPUT_DIRECTORY.resolve() else "."
    return base64.urlsafe_b64encode(relative.encode("utf-8")).decode("ascii").rstrip("=")


def history_asset_path(asset_id: str):
    try:
        encoded = asset_id + "=" * (-len(asset_id) % 4)
        relative = base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
        path = (OUTPUT_DIRECTORY / relative).resolve()
    except (ValueError, UnicodeError, binascii.Error) as error:
        raise HTTPException(status_code=400, detail="Invalid image asset") from error
    if OUTPUT_DIRECTORY.resolve() not in path.parents or path.suffix.lower() not in {".png", ".gif"}:
        raise HTTPException(status_code=400, detail="Image asset is outside the output directory")
    return path


def history_folder_path(folder_id: str):
    try:
        encoded = folder_id + "=" * (-len(folder_id) % 4)
        relative = base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
        path = OUTPUT_DIRECTORY.resolve() if relative == "." else (OUTPUT_DIRECTORY / relative).resolve()
    except (ValueError, UnicodeError, binascii.Error) as error:
        raise HTTPException(status_code=400, detail="Invalid output folder") from error
    output_root = OUTPUT_DIRECTORY.resolve()
    if (path != output_root and output_root not in path.parents) or not path.is_dir() or is_internal_output_directory(path):
        raise HTTPException(status_code=404, detail="Output folder is unavailable")
    return path


def history_parameters(path: Path):
    try:
        with Image.open(path) as image:
            raw = image.info.get("parameters", image.info.get("comment", "{}"))
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        parameters = json.loads(raw) if isinstance(raw, str) else {}
        return parameters if isinstance(parameters, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def history_file_record(path: Path):
    parameters = history_parameters(path)
    modified_at = path.stat().st_mtime
    def positive_integer(value, fallback=1):
        try:
            return max(1, int(value or fallback))
        except (TypeError, ValueError):
            return fallback
    return {
        "id": history_asset_token(path),
        "name": path.name,
        "url": f"/api/inference/history/assets/{history_asset_token(path)}",
        "created_at": modified_at,
        "job_id": str(parameters.get("job_id", "")),
        "legacy_group": re.sub(r"-\d{2}$", "", path.stem),
        "images_per_batch": positive_integer(parameters.get("images_per_batch")),
        "batch_count": positive_integer(parameters.get("batch_count")),
        "batch_index": positive_integer(parameters.get("batch_index")),
        "image_index": positive_integer(parameters.get("image_index")),
        "width": parameters.get("output_width", parameters.get("width")),
        "height": parameters.get("output_height", parameters.get("height")),
        "seed": str(parameters.get("seed", "")),
        "base_seed": str(parameters.get("base_seed", parameters.get("seed", ""))),
        "hires_seed_mode": parameters.get("hires_seed_mode", "inherit"),
        "hires_seed": str(parameters["hires_seed"]) if parameters.get("hires_seed") is not None else None,
        "transparent_background": isinstance(parameters.get("transparent_background"), dict),
        "background_removal": parameters.get("transparent_background") if isinstance(parameters.get("transparent_background"), dict) else None,
        "postprocess_chain": parameters.get("postprocess_chain") if isinstance(parameters.get("postprocess_chain"), list) else None,
        "rtx": parameters.get("rtx") if isinstance(parameters.get("rtx"), dict) else None,
        "manual_layout": parameters.get("manual_layout") if isinstance(parameters.get("manual_layout"), dict) else None,
        "mime_type": "image/gif" if path.suffix.lower() == ".gif" else "image/png",
    }


def history_directory_listing(folder: Path | None = None):
    output_root = OUTPUT_DIRECTORY.resolve()
    current = (folder or output_root).resolve()
    try:
        relative = current.relative_to(output_root)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Output folder is unavailable") from error
    if is_internal_output_directory(current):
        raise HTTPException(status_code=404, detail="Output folder is unavailable")

    directories = []
    image_count = 0
    try:
        entries = list(current.iterdir())
    except OSError as error:
        raise HTTPException(status_code=404, detail="Output folder is unavailable") from error
    for entry in entries:
        try:
            resolved = entry.resolve()
            if entry.is_file() and entry.suffix.lower() in {".png", ".gif"}:
                image_count += 1
                continue
            if not entry.is_dir() or is_internal_output_directory(resolved) or output_root not in resolved.parents:
                continue
            child_images = sum(1 for path in entry.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".gif"})
            child_folders = sum(1 for path in entry.iterdir() if path.is_dir() and not is_internal_output_directory(path))
            directories.append({
                "id": history_folder_token(resolved),
                "name": entry.name,
                "label": resolved.relative_to(output_root).as_posix(),
                "count": child_images,
                "folder_count": child_folders,
                "modified_at": entry.stat().st_mtime,
            })
        except OSError:
            continue
    directories.sort(key=lambda item: item["name"].casefold())
    parent = current.parent if current != output_root else None
    return {
        "id": history_folder_token(current),
        "name": current.name if current != output_root else "输出目录",
        "label": "/" if current == output_root else relative.as_posix(),
        "parent_id": history_folder_token(parent) if parent else "",
        "image_count": image_count,
        "folders": directories,
    }


def list_history_folders():
    return history_directory_listing()["folders"]


def list_history_cards(folder: Path | None = None, *, session_only=True):
    records = []
    output_root = OUTPUT_DIRECTORY.resolve()
    candidates = [path for path in (folder.iterdir() if folder else OUTPUT_DIRECTORY.rglob("*")) if path.is_file() and path.suffix.lower() in {".png", ".gif"}]
    for path in candidates:
        try:
            resolved = path.resolve()
            if output_root not in resolved.parents or is_internal_output_directory(resolved.parent):
                continue
            file_stat = resolved.stat()
            # An empty file is a name another writer has reserved but not yet published.
            if file_stat.st_size == 0:
                continue
            if session_only and file_stat.st_mtime < HISTORY_STARTED_AT:
                continue
            records.append(history_file_record(resolved))
        except (OSError, ValueError):
            continue
    records.sort(key=lambda item: item["created_at"], reverse=True)
    cards = []
    grouped = {}
    for record in records:
        is_batch = record["images_per_batch"] > 1
        group_reference = record["job_id"] or record["legacy_group"]
        group_key = f"{group_reference}:{record['batch_index']}" if is_batch else record["id"]
        if group_key not in grouped:
            card = {
                "id": group_key,
                "kind": "batch" if is_batch else "image",
                "count": 0,
                "batch_index": record["batch_index"],
                "batch_count": record["batch_count"],
                "files": [],
                "preview": record,
                "created_at": record["created_at"],
            }
            grouped[group_key] = card
            cards.append(card)
        card = grouped[group_key]
        card["files"].append(record)
        card["count"] = len(card["files"])
        if record["image_index"] == 1:
            card["preview"] = record
    for card in cards:
        card["files"].sort(key=lambda item: item["image_index"])
        card["preview"] = card["preview"] or card["files"][0]
    return cards


def clean_output_name(name: str, fallback: str):
    candidate = Path(name).name.replace("\x00", "").strip()
    if not candidate.lower().endswith((".png", ".gif")):
        candidate += ".png"
    candidate = "".join(character for character in candidate if character.isalnum() or character in " ._-()[]")
    return candidate[:150] or fallback


def draw_collage_edge(image: Image.Image, layer: dict, options: dict | None):
    if not options or not options.get("enabled"):
        return
    candidate_color = options.get("color", "#c8acfb")
    color = candidate_color.lower() if isinstance(candidate_color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", candidate_color) else "#c8acfb"
    try:
        width = max(1, min(50, int(options.get("width", 2))))
    except (TypeError, ValueError, OverflowError):
        width = 2
    style = options.get("style", "solid") if options.get("style") in {"solid", "dashed", "dotted", "double", "glow"} else "solid"
    left, top = int(layer["x"]), int(layer["y"])
    right = left + max(1, int(layer["width"])) - 1
    bottom = top + max(1, int(layer["height"])) - 1
    hidden = {side for side in layer.get("hidden_sides", []) if side in {"top", "right", "bottom", "left"}}
    draw = ImageDraw.Draw(image)
    if style == "dashed":
        for x in range(left, right + 1, width * 6):
            if "top" not in hidden:
                draw.line((x, top, min(right, x + width * 3), top), fill=color, width=width)
            if "bottom" not in hidden:
                draw.line((x, bottom, min(right, x + width * 3), bottom), fill=color, width=width)
        for y in range(top, bottom + 1, width * 6):
            if "left" not in hidden:
                draw.line((left, y, left, min(bottom, y + width * 3)), fill=color, width=width)
            if "right" not in hidden:
                draw.line((right, y, right, min(bottom, y + width * 3)), fill=color, width=width)
    elif style == "dotted":
        for x in range(left, right + 1, width * 3):
            if "top" not in hidden:
                draw.ellipse((x, top, x + width, top + width), fill=color)
            if "bottom" not in hidden:
                draw.ellipse((x, bottom - width, x + width, bottom), fill=color)
        for y in range(top, bottom + 1, width * 3):
            if "left" not in hidden:
                draw.ellipse((left, y, left + width, y + width), fill=color)
            if "right" not in hidden:
                draw.ellipse((right - width, y, right, y + width), fill=color)
    else:
        shortest_edge = max(1, min(right - left + 1, bottom - top + 1))
        outer_width = min(width * (2 if style == "double" else 1), max(1, shortest_edge // 2))
        def draw_sides(rectangle, line_width):
            side_left, side_top, side_right, side_bottom = rectangle
            line_width = max(1, min(line_width, side_right - side_left + 1, side_bottom - side_top + 1))
            if "top" not in hidden:
                draw.rectangle((side_left, side_top, side_right, min(side_bottom, side_top + line_width - 1)), fill=color)
            if "right" not in hidden:
                draw.rectangle((max(side_left, side_right - line_width + 1), side_top, side_right, side_bottom), fill=color)
            if "bottom" not in hidden:
                draw.rectangle((side_left, max(side_top, side_bottom - line_width + 1), side_right, side_bottom), fill=color)
            if "left" not in hidden:
                draw.rectangle((side_left, side_top, min(side_right, side_left + line_width - 1), side_bottom), fill=color)
        draw_sides((left, top, right, bottom), outer_width)
        if style == "double":
            inset = width * 3
            if left + inset <= right - inset and top + inset <= bottom - inset:
                draw_sides((left + inset, top + inset, right - inset, bottom - inset), width)


def decode_collage_data_url(source: str, maximum_bytes: int, label: str):
    if not isinstance(source, str) or not re.match(r"^data:image/(?:png|gif|jpeg|webp);base64,", source, re.IGNORECASE):
        raise ValueError(f"{label} must be a supported image data URL")
    encoded = source.partition(",")[2]
    maximum_encoded = ((maximum_bytes + 2) // 3) * 4
    if len(encoded) > maximum_encoded:
        raise CollageResourceLimitError(f"{label} exceeds the decoded byte budget")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError(f"{label} contains invalid base64 data") from error
    if len(raw) > maximum_bytes:
        raise CollageResourceLimitError(f"{label} exceeds the decoded byte budget")
    return raw


def open_budgeted_collage_image(raw: bytes, maximum_pixels: int, label: str):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(raw))
            width, height = image.size
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise CollageResourceLimitError(f"{label} triggered the decompression-bomb guard") from error
    except (OSError, SyntaxError, ValueError) as error:
        raise ValueError(f"Unable to read {label}") from error
    if width < 1 or height < 1:
        image.close()
        raise ValueError(f"{label} has invalid dimensions")
    if width * height > maximum_pixels:
        image.close()
        raise CollageResourceLimitError(f"{label} exceeds the pixel budget")
    return image


def composite_animated_layer(output: Image.Image, current: Image.Image, layer: dict):
    width = max(1, int(round(layer.get("width", current.width))))
    height = max(1, int(round(layer.get("height", current.height))))
    if width * height > COLLAGE_MAX_PIXELS:
        raise CollageResourceLimitError("Animated collage resized layer exceeds the pixel budget")
    x, y = int(round(layer.get("x", 0))), int(round(layer.get("y", 0)))
    clip = layer.get("clip") if isinstance(layer.get("clip"), dict) else None
    raw_clip_left = int(round(clip.get("x", 0))) if clip else 0
    raw_clip_top = int(round(clip.get("y", 0))) if clip else 0
    raw_clip_right = raw_clip_left + max(1, int(round(clip.get("width", output.width)))) if clip else output.width
    raw_clip_bottom = raw_clip_top + max(1, int(round(clip.get("height", output.height)))) if clip else output.height
    destination_left = max(0, raw_clip_left, x)
    destination_top = max(0, raw_clip_top, y)
    destination_right = min(output.width, raw_clip_right, x + width)
    destination_bottom = min(output.height, raw_clip_bottom, y + height)
    if destination_right <= destination_left or destination_bottom <= destination_top:
        return None
    visible_width = destination_right - destination_left
    visible_height = destination_bottom - destination_top
    source_left = max(0, min(current.width - 1, math.floor((destination_left - x) * current.width / width)))
    source_top = max(0, min(current.height - 1, math.floor((destination_top - y) * current.height / height)))
    source_right = max(source_left + 1, min(current.width, math.ceil((destination_right - x) * current.width / width)))
    source_bottom = max(source_top + 1, min(current.height, math.ceil((destination_bottom - y) * current.height / height)))
    source_region = current.crop((source_left, source_top, source_right, source_bottom))
    rgba_region = None
    resized_region = None
    try:
        rgba_region = source_region.convert("RGBA")
        if rgba_region.size == (visible_width, visible_height):
            resized_region = rgba_region
        else:
            resized_region = rgba_region.resize((visible_width, visible_height), Image.Resampling.LANCZOS)
        output.alpha_composite(resized_region, (destination_left, destination_top))
    finally:
        if resized_region is not None and resized_region is not rgba_region:
            resized_region.close()
        if rgba_region is not None:
            rgba_region.close()
        source_region.close()
    edge_left = max(0, raw_clip_left if clip else x)
    edge_top = max(0, raw_clip_top if clip else y)
    edge_right = min(output.width, raw_clip_right if clip else x + width)
    edge_bottom = min(output.height, raw_clip_bottom if clip else y + height)
    return {
        "x": edge_left,
        "y": edge_top,
        "width": max(1, edge_right - edge_left),
        "height": max(1, edge_bottom - edge_top),
        "hidden_sides": layer.get("hidden_sides", []),
    }


def animated_collage_frames(input_data: AnimatedCollageInput):
    sources = []
    output_frame_count = 1
    decoded_total = 0
    source_pixels = 0
    source_frames = 0
    source_frame_pixels = 0
    try:
        for raw_layer in input_data.layers:
            layer = raw_layer.model_dump() if isinstance(raw_layer, BaseModel) else raw_layer
            raw = decode_collage_data_url(layer.get("url"), COLLAGE_MAX_ANIMATED_SOURCE_BYTES, "Animated collage source")
            decoded_total += len(raw)
            if decoded_total > COLLAGE_MAX_ANIMATED_INPUT_BYTES:
                raise CollageResourceLimitError("Animated collage exceeds the total decoded input byte budget")
            image = open_budgeted_collage_image(raw, COLLAGE_MAX_ANIMATED_SOURCE_PIXELS, "animated collage source")
            if image.format not in {"PNG", "GIF", "JPEG", "WEBP"}:
                image.close()
                raise ValueError("Unsupported animated collage source format")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    source_frame_count = max(1, int(getattr(image, "n_frames", 1)))
            except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
                image.close()
                raise CollageResourceLimitError("Animated collage source triggered the decompression-bomb guard") from error
            if source_frame_count > COLLAGE_MAX_ANIMATED_SOURCE_FRAMES:
                image.close()
                raise CollageResourceLimitError("Animated collage source exceeds the frame limit")
            source_pixels += image.width * image.height
            source_frames += source_frame_count
            source_frame_pixels += image.width * image.height * source_frame_count
            if source_pixels > COLLAGE_MAX_ANIMATED_INPUT_PIXELS or source_frames > COLLAGE_MAX_ANIMATED_INPUT_FRAMES or source_frame_pixels > COLLAGE_MAX_ANIMATED_FRAME_PIXELS:
                image.close()
                raise CollageResourceLimitError("Animated collage sources exceed their aggregate pixel or frame budget")
            sources.append((image, layer, source_frame_count))
            output_frame_count = max(output_frame_count, source_frame_count)
        if output_frame_count > COLLAGE_MAX_GIF_FRAMES or input_data.width * input_data.height * output_frame_count > COLLAGE_MAX_TOTAL_FRAME_PIXELS:
            raise CollageResourceLimitError("GIF collage exceeds the output frame budget")
        frames, durations = [], []
        for frame_index in range(output_frame_count):
            output = Image.new("RGBA", (input_data.width, input_data.height), (0, 0, 0, 0))
            duration = 20
            for image, layer, source_frame_count in sources:
                image.seek(frame_index % source_frame_count)
                edge = composite_animated_layer(output, image, layer)
                if edge:
                    draw_collage_edge(output, edge, input_data.edge_line)
                duration = max(duration, normalize_collage_gif_duration(image.info.get("duration")))
            frames.append(output)
            durations.append(duration)
        return frames, durations
    except (MemoryError, OverflowError) as error:
        raise CollageResourceLimitError("Animated collage could not allocate its bounded image buffers") from error
    finally:
        for image, _layer, _frame_count in sources:
            image.close()


def create_named_output_path(name: str):
    now = datetime.now()
    directory = OUTPUT_DIRECTORY / now.strftime("%Y-%m-%d")
    directory.mkdir(parents=True, exist_ok=True)
    requested = clean_output_name(name, f"XirAI-{now:%Y-%m-%d-%H-%M}.png")
    output_path = directory / requested
    stem = output_path.stem
    suffix = output_path.suffix
    index = 2
    while output_path.exists():
        output_path = directory / f"{stem}-{index:02d}{suffix}"
        index += 1
    return output_path


def model_roots(engine: str):
    if engine == "SD":
        key = "sd"
    elif engine == "iL":
        key = "illustrious"
    else:
        raise ValueError(f"Unsupported checkpoint engine: {engine}")
    return (
        resolve_model_directory(PROJECT_ROOT, "checkpoints", key),
        resolve_model_directory(PROJECT_ROOT, "loras", key),
    )


def anima_model_roots():
    return {
        "diffusion_model": resolve_model_directory(PROJECT_ROOT, "diffusion_models"),
        "text_encoder": resolve_model_directory(PROJECT_ROOT, "text_encoders"),
        "vae": resolve_model_directory(PROJECT_ROOT, "vae"),
        "lora": resolve_model_directory(PROJECT_ROOT, "loras", "anima"),
    }


def flux_model_roots():
    """The same ComfyUI component directories, with Flux's own LoRA root.

    Diffusion models, text encoders and VAEs are shared because that is how ComfyUI lays them out
    and how the files arrive. LoRAs are not: a Flux LoRA cannot patch an Anima transformer, so each
    engine keeps its own folder and the mount list can never offer the wrong one.
    """
    return {
        "diffusion_model": resolve_model_directory(PROJECT_ROOT, "diffusion_models"),
        "text_encoder": resolve_model_directory(PROJECT_ROOT, "text_encoders"),
        "vae": resolve_model_directory(PROJECT_ROOT, "vae"),
        "lora": resolve_model_directory(PROJECT_ROOT, "loras", "flux"),
    }


def flux2_model_roots():
    """FLUX.2's components, with its own LoRA root for the same reason FLUX.1 has one.

    A FLUX.2 LoRA targets a 4-axis, globally modulated transformer and cannot patch a FLUX.1 one,
    so the two generations never share a mount list even though they share a model directory.
    """
    return {
        "diffusion_model": resolve_model_directory(PROJECT_ROOT, "diffusion_models"),
        "text_encoder": resolve_model_directory(PROJECT_ROOT, "text_encoders"),
        "vae": resolve_model_directory(PROJECT_ROOT, "vae"),
        "lora": resolve_model_directory(PROJECT_ROOT, "loras", "flux2"),
    }


def krea2_model_roots():
    """Krea 2's components, with its own LoRA root for the same reason each Flux generation has one.

    A Krea 2 LoRA patches a single-stream DiT whose modules are named nothing like either Flux
    transformer's, so it can only ever apply here.
    """
    return {
        "diffusion_model": resolve_model_directory(PROJECT_ROOT, "diffusion_models"),
        "text_encoder": resolve_model_directory(PROJECT_ROOT, "text_encoders"),
        "vae": resolve_model_directory(PROJECT_ROOT, "vae"),
        "lora": resolve_model_directory(PROJECT_ROOT, "loras", "krea2"),
    }


def native_model_roots(engine: str):
    if engine == "Flux":
        return flux_model_roots()
    if engine == "Flux2":
        return flux2_model_roots()
    if engine == "Krea2":
        return krea2_model_roots()
    return anima_model_roots()


CHECKPOINT_EXTENSIONS = frozenset({".safetensors", ".ckpt"})
LORA_EXTENSIONS = frozenset({".safetensors", ".ckpt", ".pt", ".pth"})
ANIMA_LORA_EXTENSIONS = frozenset({".safetensors"})
# FLUX.1, FLUX.2 and Krea 2 also read a GGUF diffusion model, which is the only form some of their
# quantisations are published in. Since P9 their text encoders and autoencoders may be GGUF too:
# `gguf_loader.gguf_component_names` renames a llama.cpp encoder into ComfyUI's naming or refuses
# it, and the engine's classifier then checks the tensors as it does for safetensors. Anima's
# components stay safetensors.
NATIVE_DIFFUSION_EXTENSIONS = frozenset({".safetensors", ".gguf"})
NATIVE_COMPONENT_EXTENSIONS = NATIVE_DIFFUSION_EXTENSIONS


def validate_child_path(requested_path: str, root: Path, label: str):
    relative_path = Path(requested_path)
    if relative_path.is_absolute():
        raise ValueError(f"{label} path must be relative to its configured model directory")
    path = (root / relative_path).resolve(strict=True)
    if not path.is_file() or root not in path.parents:
        raise ValueError(f"{label} is outside its configured model directory")
    return path


def validate_anima_component(requested_path: str, root: Path, label: str):
    path = validate_child_path(requested_path, root, label)
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"{label} must be a .safetensors file")
    return path


def validate_native_diffusion_model(requested_path: str, root: Path, label: str):
    path = validate_child_path(requested_path, root, label)
    if path.suffix.lower() not in NATIVE_DIFFUSION_EXTENSIONS:
        raise ValueError(f"{label} must be a .safetensors or .gguf file")
    return path


def validate_native_component(requested_path: str, root: Path, label: str):
    """A FLUX.1 / FLUX.2 / Krea 2 text encoder or autoencoder: safetensors or GGUF."""
    path = validate_child_path(requested_path, root, label)
    if path.suffix.lower() not in NATIVE_COMPONENT_EXTENSIONS:
        raise ValueError(f"{label} must be a .safetensors or .gguf file")
    return path


def validate_native_lora(requested_path: str, root: Path, label: str):
    return resolve_model_path(requested_path, root, ANIMA_LORA_EXTENSIONS, label)


def validate_anima_lora(requested_path: str, root: Path):
    return validate_native_lora(requested_path, root, "Anima LoRA")


def resolve_model_path(requested_path: str, root: Path, extensions, label: str):
    """Resolve a model reference that may live in a registered shared folder.

    Everything on the wire is still relative — a shared file just carries its
    root in a `shared:<id>/` namespace instead of being relative to the
    project's own directory. Both branches end at an existing file proven to sit
    inside a root the user explicitly trusted.
    """
    if is_shared_ref(requested_path):
        path = resolve_shared_model(requested_path, PROJECT_ROOT, extensions, label)
        if extensions and path.suffix.lower() not in extensions:
            raise ValueError(f"{label} must be one of {', '.join(sorted(extensions))}")
        return path
    path = validate_child_path(requested_path, root, label)
    if extensions and path.suffix.lower() not in extensions:
        raise ValueError(f"{label} must be one of {', '.join(sorted(extensions))}")
    return path


def preflight_inline_lora_request(request: GenerateInput):
    """Resolve and strip SD/iL inline tags before a job can be created."""

    if request.engine not in {"SD", "iL"}:
        return request, []
    for index, unit in enumerate(request.adetailer.active_units, start=1):
        # A detail pass shares the job's mounted adapters; a tag of its own would
        # otherwise reach CLIP as literal text.
        if has_extra_network_tag(unit.prompt) or has_extra_network_tag(unit.negative_prompt):
            raise InlineLoraError(f"ADetailer unit {index}: inline <lora:…> tags are only read from the main prompt")
    if not has_extra_network_tag(request.prompt) and not has_extra_network_tag(request.negative_prompt):
        return request, []
    _checkpoint_root, lora_root = model_roots(request.engine)
    manual_paths = [
        resolve_model_path(item.path, lora_root, LORA_EXTENSIONS, "LoRA")
        for item in request.loras
    ]
    audit = []
    for order, (item, path) in enumerate(zip(request.loras, manual_paths)):
        reference = (
            shared_model_reference(item.path)
            if is_shared_ref(item.path)
            else path.relative_to(lora_root.resolve()).as_posix()
        )
        if not reference:
            raise InlineLoraError("explicit LoRA shared reference is invalid")
        audit.append({
            "id": reference,
            "weight": float(item.weight),
            "source": "manual",
            "match": "manual",
            "order": order,
        })
    candidates = discover_candidates(
        lora_root,
        load_shared_roots(PROJECT_ROOT),
        request.engine,
        LORA_EXTENSIONS,
    )
    result = resolve_inline_loras(
        request.prompt,
        request.negative_prompt,
        manual_paths,
        candidates,
    )
    merged = list(request.loras) + [
        LoraInput(path=match.reference, weight=match.weight)
        for match in result.matches
    ]
    runtime_request = request.model_copy(update={
        "prompt": result.prompt,
        "negative_prompt": result.negative_prompt,
        "loras": merged,
    })
    audit.extend(match.audit() for match in result.matches)
    return runtime_request, audit


SECOND_ORDER_SAMPLERS = frozenset({"heun", "dpm_2", "dpm_2_ancestral"})


def sampler_calls_per_step(sampler: str) -> int:
    """Denoiser calls per step of the scheduler `configure_scheduler` builds for `sampler`."""
    return 2 if str(sampler).lower() in SECOND_ORDER_SAMPLERS else 1


def request_prompt_timelines(request: GenerateInput):
    """The prompt-schedule timeline of every SD/iL sampling pass a request will run.

    Forge counts a schedule against the requested steps of a pass. A redraw after a base
    pass continues the base timeline, as Hires fix does; ADetailer passes start their own.
    """
    base = None
    if not request.postprocess_only:
        base = PromptTimeline(request.steps * sampler_calls_per_step(request.sampler))
    resample = []
    for settings in request.active_resample_units:
        if not settings.redraws:
            continue
        horizon = settings.resolved_steps(request.steps) * sampler_calls_per_step(settings.sampler or request.sampler)
        resample.append(PromptTimeline(horizon, base.steps, 1.0) if base else PromptTimeline(horizon))
    adetailer = [
        PromptTimeline((unit.steps if unit.use_steps else request.steps) * sampler_calls_per_step(request.sampler))
        for unit in request.adetailer.active_units
    ]
    return base, resample, adetailer


def preflight_dynamic_prompt_request(request: GenerateInput):
    """Compile every SD/iL prompt shape, for every pass that encodes it, before a job exists."""

    if request.engine not in {"SD", "iL"}:
        return None
    base_timeline, resample_timelines, adetailer_timelines = request_prompt_timelines(request)
    main_passes = [timeline for timeline in (base_timeline, *resample_timelines) if timeline is not None]
    base = compile_conditioning_syntax(request.prompt, request.negative_prompt, main_passes)
    detail_plans = []
    for index, (unit, timeline) in enumerate(zip(request.adetailer.active_units, adetailer_timelines), start=1):
        detail_prompt = expand_prompt(unit.prompt, request.prompt)
        detail_negative = expand_prompt(unit.negative_prompt, request.negative_prompt)
        detail_plans.append({
            "unit": index,
            **prompt_plan_audit(compile_conditioning_syntax(detail_prompt, detail_negative, (timeline,)), timeline),
        })
    return {
        "compiler": "forge_dynamic_v2",
        "base": prompt_plan_audit(base, base_timeline),
        "resample": [prompt_plan_audit(base, timeline) for timeline in resample_timelines],
        "adetailer": detail_plans,
    }


def shared_reference_for_path(absolute_path):
    """Map a loaded absolute path back to the `shared:` reference it came from.

    The loaded-model bookkeeping stores absolute paths, but every value that
    leaves this process has to stay a relative reference or the UI cannot match
    it against what the user selected — and the user's disk layout would leak.
    """
    if not absolute_path:
        return None
    try:
        roots = load_shared_roots(PROJECT_ROOT)
    except ValueError:
        return None
    resolved = Path(absolute_path)
    for root in roots:
        try:
            relative = resolved.relative_to(Path(root["path"]).resolve(strict=True))
        except (OSError, ValueError):
            continue
        return f"shared:{root['id']}/{relative.as_posix()}"
    return None


def public_model_reference(value):
    if not value:
        return None
    # A shared reference is already the safe, relative form of itself.
    shared = shared_model_reference(value)
    if shared:
        return shared
    # References can originate on another OS (for example from a remote client
    # or a persisted job).  pathlib.Path only recognises the host's syntax, so
    # inspect both grammars before deciding whether a path may leave the server.
    windows_path = PureWindowsPath(value)
    posix_path = PurePosixPath(value)
    if (
        windows_path.is_absolute()
        or posix_path.is_absolute()
        or ".." in windows_path.parts
        or ".." in posix_path.parts
    ):
        return windows_path.name if windows_path.is_absolute() or ".." in windows_path.parts else posix_path.name
    return posix_path.as_posix()


@lru_cache(maxsize=16)
def anima_tokenizer_sha256(path_value: str, size: int, modified_ns: int, changed_ns: int):
    del size, modified_ns, changed_ns
    return file_sha256(Path(path_value))


def anima_tokenizer_status(directory=None, force_hash=False):
    directory = Path(directory or ANIMA_TOKENIZER_DIRECTORY)
    statuses = {}
    labels = {
        "qwen": "Anima Qwen3 tokenizer",
        "qwen_config": "Anima Qwen3 tokenizer config",
        "t5": "Anima T5 tokenizer",
    }
    for name, (filename, _expected_size, _expected_sha256) in ANIMA_TOKENIZER_ARTIFACTS.items():
        try:
            path = validate_child_path(filename, directory, labels[name])
            file_stat = path.stat()
            variants = ANIMA_TOKENIZER_ARTIFACT_VARIANTS[name]
            size_matches = [variant for variant in variants if variant[1] == file_stat.st_size]
            if not size_matches:
                expected_sizes = ", ".join(str(variant[1]) for variant in variants)
                raise ValueError(f"expected one of {expected_sizes} bytes, found {file_stat.st_size}")
            actual_sha256 = file_sha256(path) if force_hash else anima_tokenizer_sha256(
                str(path), file_stat.st_size, file_stat.st_mtime_ns, file_stat.st_ctime_ns
            )
            if actual_sha256 not in {variant[2] for variant in size_matches}:
                raise ValueError(
                    "SHA-256 does not match a pinned Anima runtime artifact for "
                    f"the {file_stat.st_size}-byte variant"
                )
            statuses[name] = {"path": path, "installed": True, "reason": None}
        except FileNotFoundError:
            statuses[name] = {
                "path": directory / filename,
                "installed": False,
                "reason": "bundled program resource is missing",
            }
        except OSError:
            statuses[name] = {
                "path": directory / filename,
                "installed": False,
                "reason": "bundled program resource could not be read",
            }
        except ValueError as error:
            statuses[name] = {
                "path": directory / filename,
                "installed": False,
                "reason": str(error),
            }
    return statuses


def anima_tokenizer_sources(directory=None):
    statuses = anima_tokenizer_status(directory, force_hash=True)
    unavailable = [name for name, status in statuses.items() if not status["installed"]]
    if unavailable:
        details = "; ".join(f"{name}: {statuses[name]['reason']}" for name in unavailable)
        raise ValueError(f"Bundled Anima tokenizer resources are missing or corrupt ({details})")
    return {name: status["path"] for name, status in statuses.items()}


def anima_runtime_tokenizer_paths(sources):
    runtime_root = PROJECT_CACHE / "anima-tokenizers"
    aliases = {}
    destinations = {
        "qwen": ("qwen", "tokenizer.json"),
        "qwen_config": ("qwen", "tokenizer_config.json"),
        "t5": ("t5", "tokenizer.json"),
    }
    for name, source in sources.items():
        directory_name, filename = destinations[name]
        directory = runtime_root / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        alias = directory / filename
        source_stat = source.stat()
        source_sha256 = file_sha256(source)
        alias_ready = False
        if alias.is_file():
            alias_stat = alias.stat()
            alias_ready = (
                alias_stat.st_size == source_stat.st_size
                and file_sha256(alias) == source_sha256
            )
        if not alias_ready:
            temporary = alias.with_suffix(".tmp")
            try:
                shutil.copy2(source, temporary)
                os.replace(temporary, alias)
            finally:
                temporary.unlink(missing_ok=True)
        aliases[name] = alias
    return aliases


def project_storage_reference(path: Path):
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def anima_health_fields():
    try:
        roots = anima_model_roots()
        required_assets = {
            "diffusion_models": project_storage_reference(roots["diffusion_model"]),
            "text_encoders": project_storage_reference(roots["text_encoder"]),
            "vae": project_storage_reference(roots["vae"]),
            "loras": project_storage_reference(roots["lora"]),
            "qwen_tokenizer": project_storage_reference(ANIMA_TOKENIZER_DIRECTORY / "anima-qwen3-tokenizer.json"),
            "qwen_tokenizer_config": project_storage_reference(
                ANIMA_TOKENIZER_DIRECTORY / "anima-qwen3-tokenizer-config.json"
            ),
            "t5_tokenizer": project_storage_reference(ANIMA_TOKENIZER_DIRECTORY / "anima-t5-tokenizer.json"),
        }
        tokenizer_statuses = anima_tokenizer_status()
        tokenizers_ready = all(status["installed"] for status in tokenizer_statuses.values())
        tokenizer_reason = None if tokenizers_ready else "Bundled Anima tokenizer resources are missing or failed integrity validation"
        public_names = {
            "qwen": "qwen_tokenizer",
            "qwen_config": "qwen_tokenizer_config",
            "t5": "t5_tokenizer",
        }
        tokenizer_runtime = {
            public_names[name]: {
                "path": project_storage_reference(status["path"]),
                "installed": status["installed"],
                "reason": status["reason"],
            }
            for name, status in tokenizer_statuses.items()
        }
    except ValueError as error:
        required_assets = {}
        tokenizers_ready = False
        tokenizer_reason = str(error)
        tokenizer_runtime = {}

    try:
        import diffusers
        from accelerate import init_empty_weights as _init_empty_weights
        from diffusers import AutoencoderKLQwenImage as _AutoencoderKLQwenImage, CosmosTransformer3DModel as _CosmosTransformer3DModel
        from diffusers.loaders.single_file_utils import (
            convert_cosmos_transformer_checkpoint_to_diffusers as _convert_cosmos,
            convert_wan_vae_to_diffusers as _convert_wan_vae,
        )
        from safetensors.torch import load_file as _load_safetensors_file
        from transformers import (
            PreTrainedTokenizerFast as _PreTrainedTokenizerFast,
            Qwen3Config as _Qwen3Config,
            Qwen3Model as _Qwen3Model,
            T5TokenizerFast as _T5TokenizerFast,
        )
        del (
            _init_empty_weights,
            _AutoencoderKLQwenImage,
            _CosmosTransformer3DModel,
            _convert_cosmos,
            _convert_wan_vae,
            _load_safetensors_file,
            _PreTrainedTokenizerFast,
            _Qwen3Config,
            _Qwen3Model,
            _T5TokenizerFast,
        )
        runtime_ready = str(diffusers.__version__).startswith("0.38.")
        runtime_reason = None if runtime_ready else "Native Anima requires diffusers 0.38.x"
    except (ImportError, AttributeError) as error:
        runtime_ready = False
        runtime_reason = f"Native Anima runtime dependencies are unavailable: {error}"

    cuda_ready = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    available = runtime_ready and tokenizers_ready and cuda_ready
    reason = None
    if not runtime_ready:
        reason = runtime_reason
    elif not tokenizers_ready:
        reason = tokenizer_reason
    elif not cuda_ready:
        reason = "Native Anima requires a CUDA GPU with BF16 support"
    return {
        "available": available,
        "reason": reason,
        "runtime_ready": runtime_ready,
        "tokenizers_ready": tokenizers_ready,
        "runtime": tokenizer_runtime,
        "bf16_ready": cuda_ready,
        "required_assets": required_assets,
        "samplers": list(ANIMA_SAMPLERS),
        "schedulers": list(ANIMA_SCHEDULERS),
        "features": {
            "pag": True,
            "cfg_zero_star": True,
            "hires": True,
            "adetailer": True,
            "rtx": True,
            "staged_vae_decode": False,
            "lora": True,
            "transparent_background": True,
        },
    }


# FLUX.1's CLIP-L tokenizer is the Stable Diffusion one, byte for byte: both are OpenAI's
# CLIP ViT-L/14 BPE. Reading it out of the runtime config the environment configurator already
# installs avoids vendoring a second copy of the same 1.5 MB pair.
FLUX_CLIP_TOKENIZER_FILES = ("vocab.json", "merges.txt")


def flux_clip_tokenizer_directory():
    for family in ("sdxl", "sd"):
        directory = find_pipeline_config(family, os.environ["HF_HUB_CACHE"])
        if directory is None:
            continue
        tokenizer = Path(directory) / "tokenizer"
        if all((tokenizer / name).is_file() for name in FLUX_CLIP_TOKENIZER_FILES):
            return tokenizer
    return None


def flux_health_fields():
    clip_directory = None
    try:
        roots = flux_model_roots()
        required_assets = {
            "diffusion_models": project_storage_reference(roots["diffusion_model"]),
            "text_encoders": project_storage_reference(roots["text_encoder"]),
            "vae": project_storage_reference(roots["vae"]),
            "loras": project_storage_reference(roots["lora"]),
            "t5_tokenizer": project_storage_reference(ANIMA_TOKENIZER_DIRECTORY / "anima-t5-tokenizer.json"),
        }
        # The T5 v1.1 tokenizer is one file shared by both native engines; it is pinned once, under
        # the name it was first bundled with, rather than vendored twice.
        t5_status = anima_tokenizer_status()["t5"]
        clip_directory = flux_clip_tokenizer_directory()
        clip_ready = clip_directory is not None
        tokenizers_ready = t5_status["installed"] and clip_ready
        tokenizer_reason = (
            None if tokenizers_ready
            else "Bundled T5 tokenizer resource is missing or failed integrity validation"
            if not t5_status["installed"]
            else "缺少 CLIP-L 分词器本地运行配置，请联网重新运行环境配置器后再生成。"
        )
        tokenizer_runtime = {
            "t5_tokenizer": {
                "path": project_storage_reference(t5_status["path"]),
                "installed": t5_status["installed"],
                "reason": t5_status["reason"],
            },
            "clip_tokenizer": {
                "path": project_storage_reference(clip_directory) if clip_ready else None,
                "installed": clip_ready,
                "reason": None if clip_ready else "Stable Diffusion runtime config is not installed",
            },
        }
    except ValueError as error:
        required_assets = {}
        tokenizers_ready = False
        tokenizer_reason = str(error)
        tokenizer_runtime = {}

    try:
        import diffusers
        from accelerate import init_empty_weights as _init_empty_weights
        from diffusers import AutoencoderKL as _AutoencoderKL, FluxTransformer2DModel as _FluxTransformer2DModel
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint as _convert_ldm_vae
        from safetensors.torch import load_file as _load_safetensors_file
        from transformers import (
            CLIPTextModel as _CLIPTextModel,
            CLIPTokenizer as _CLIPTokenizer,
            T5EncoderModel as _T5EncoderModel,
            T5TokenizerFast as _T5TokenizerFast,
        )
        del (
            _init_empty_weights, _AutoencoderKL, _FluxTransformer2DModel, _convert_ldm_vae,
            _load_safetensors_file, _CLIPTextModel, _CLIPTokenizer, _T5EncoderModel, _T5TokenizerFast,
        )
        runtime_ready = str(diffusers.__version__).startswith("0.38.")
        runtime_reason = None if runtime_ready else "Native Flux requires diffusers 0.38.x"
    except (ImportError, AttributeError) as error:
        runtime_ready = False
        runtime_reason = f"Native Flux runtime dependencies are unavailable: {error}"

    cuda_ready = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    available = runtime_ready and tokenizers_ready and cuda_ready
    reason = None
    if not runtime_ready:
        reason = runtime_reason
    elif not tokenizers_ready:
        reason = tokenizer_reason
    elif not cuda_ready:
        reason = "Native Flux requires a CUDA GPU with BF16 support"
    return {
        "available": available,
        "reason": reason,
        "runtime_ready": runtime_ready,
        "tokenizers_ready": tokenizers_ready,
        "runtime": tokenizer_runtime,
        "bf16_ready": cuda_ready,
        "required_assets": required_assets,
        "samplers": list(FLUX_SAMPLERS),
        "schedulers": list(FLUX_SCHEDULERS),
        "features": {
            # Guidance distillation removes the unconditional branch, and with it everything built
            # on one: PAG has nothing to perturb against, CFG-Zero* has nothing to rescale, and a
            # negative prompt would be encoded and then discarded.
            "pag": False,
            "cfg_zero_star": False,
            "negative_prompt": False,
            "distilled_guidance": True,
            "hires": True,
            "adetailer": True,
            "rtx": True,
            "staged_vae_decode": False,
            "lora": True,
            "transparent_background": True,
        },
    }


def flux2_health_fields():
    try:
        roots = flux2_model_roots()
        required_assets = {
            "diffusion_models": project_storage_reference(roots["diffusion_model"]),
            "text_encoders": project_storage_reference(roots["text_encoder"]),
            "vae": project_storage_reference(roots["vae"]),
            "loras": project_storage_reference(roots["lora"]),
            "qwen_tokenizer": project_storage_reference(ANIMA_TOKENIZER_DIRECTORY / "anima-qwen3-tokenizer.json"),
        }
        qwen_status = anima_tokenizer_status()["qwen"]
        tokenizers_ready = qwen_status["installed"]
        tokenizer_reason = (
            None if tokenizers_ready
            else "Bundled Qwen tokenizer resource is missing or failed integrity validation"
        )
        tokenizer_runtime = {
            "qwen_tokenizer": {
                "path": project_storage_reference(qwen_status["path"]),
                "installed": qwen_status["installed"],
                "reason": qwen_status["reason"],
            },
            # FLUX.2 [dev]'s Mistral tokenizer is not a resource this project installs: ComfyUI
            # publishes it inside the text encoder checkpoint, so it is reported as satisfied by
            # the model file rather than by the environment.
            "tekken_tokenizer": {
                "path": None,
                "installed": True,
                "reason": None,
                "source": "embedded_in_text_encoder",
            },
        }
    except ValueError as error:
        required_assets = {}
        tokenizers_ready = False
        tokenizer_reason = str(error)
        tokenizer_runtime = {}

    try:
        import diffusers
        import regex as _regex
        from accelerate import init_empty_weights as _init_empty_weights
        from diffusers import (
            AutoencoderKLFlux2 as _AutoencoderKLFlux2,
            Flux2Transformer2DModel as _Flux2Transformer2DModel,
        )
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint as _convert_ldm_vae
        from safetensors.torch import load_file as _load_safetensors_file
        from transformers import (
            MistralModel as _MistralModel,
            PreTrainedTokenizerFast as _PreTrainedTokenizerFast,
            Qwen3Model as _Qwen3Model,
        )
        del (
            _regex, _init_empty_weights, _AutoencoderKLFlux2, _Flux2Transformer2DModel,
            _convert_ldm_vae, _load_safetensors_file, _MistralModel, _PreTrainedTokenizerFast,
            _Qwen3Model,
        )
        runtime_ready = str(diffusers.__version__).startswith("0.38.")
        runtime_reason = None if runtime_ready else "Native Flux2 requires diffusers 0.38.x"
    except (ImportError, AttributeError) as error:
        runtime_ready = False
        runtime_reason = f"Native Flux2 runtime dependencies are unavailable: {error}"

    cuda_ready = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    available = runtime_ready and tokenizers_ready and cuda_ready
    reason = None
    if not runtime_ready:
        reason = runtime_reason
    elif not tokenizers_ready:
        reason = tokenizer_reason
    elif not cuda_ready:
        reason = "Native Flux2 requires a CUDA GPU with BF16 support"
    return {
        "available": available,
        "reason": reason,
        "runtime_ready": runtime_ready,
        "tokenizers_ready": tokenizers_ready,
        "runtime": tokenizer_runtime,
        "bf16_ready": cuda_ready,
        "required_assets": required_assets,
        "samplers": list(FLUX2_SAMPLERS),
        "schedulers": list(FLUX2_SCHEDULERS),
        "features": {
            # Guidance distilled like FLUX.1, and refusing the same three settings for the same
            # reason: there is no unconditional branch for any of them to act on.
            "pag": False,
            "cfg_zero_star": False,
            "negative_prompt": False,
            "distilled_guidance": True,
            # A FLUX.2 prompt is read by a language model whose tokenizer ComfyUI runs with
            # weighting disabled, so `(word:1.2)` is literal text rather than emphasis.
            "prompt_weights": False,
            "hires": True,
            "adetailer": True,
            "rtx": True,
            "staged_vae_decode": False,
            "lora": True,
            "transparent_background": True,
        },
    }


def krea2_health_fields():
    try:
        roots = krea2_model_roots()
        required_assets = {
            "diffusion_models": project_storage_reference(roots["diffusion_model"]),
            "text_encoders": project_storage_reference(roots["text_encoder"]),
            "vae": project_storage_reference(roots["vae"]),
            "loras": project_storage_reference(roots["lora"]),
            "qwen_tokenizer": project_storage_reference(ANIMA_TOKENIZER_DIRECTORY / "anima-qwen3-tokenizer.json"),
        }
        qwen_status = anima_tokenizer_status()["qwen"]
        tokenizers_ready = qwen_status["installed"]
        tokenizer_reason = (
            None if tokenizers_ready
            else "Bundled Qwen tokenizer resource is missing or failed integrity validation"
        )
        tokenizer_runtime = {
            # Qwen3-VL reads text through the Qwen2.5 table, which is the same bundled resource
            # Anima and FLUX.2 [klein] already use; Krea 2 adds no new tokenizer to install.
            "qwen_tokenizer": {
                "path": project_storage_reference(qwen_status["path"]),
                "installed": qwen_status["installed"],
                "reason": qwen_status["reason"],
            },
        }
    except ValueError as error:
        required_assets = {}
        tokenizers_ready = False
        tokenizer_reason = str(error)
        tokenizer_runtime = {}

    try:
        import diffusers
        from accelerate import init_empty_weights as _init_empty_weights
        from diffusers import AutoencoderKLWan as _AutoencoderKLWan
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers as _convert_wan_vae
        from safetensors.torch import load_file as _load_safetensors_file
        from transformers import (
            PreTrainedTokenizerFast as _PreTrainedTokenizerFast,
            Qwen3VLTextModel as _Qwen3VLTextModel,
        )
        del (
            _init_empty_weights, _AutoencoderKLWan, _convert_wan_vae, _load_safetensors_file,
            _PreTrainedTokenizerFast, _Qwen3VLTextModel,
        )
        runtime_ready = str(diffusers.__version__).startswith("0.38.")
        runtime_reason = None if runtime_ready else "Native Krea2 requires diffusers 0.38.x"
    except (ImportError, AttributeError) as error:
        runtime_ready = False
        runtime_reason = f"Native Krea2 runtime dependencies are unavailable: {error}"

    cuda_ready = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    available = runtime_ready and tokenizers_ready and cuda_ready
    reason = None
    if not runtime_ready:
        reason = runtime_reason
    elif not tokenizers_ready:
        reason = tokenizer_reason
    elif not cuda_ready:
        reason = "Native Krea2 requires a CUDA GPU with BF16 support"
    return {
        "available": available,
        "reason": reason,
        "runtime_ready": runtime_ready,
        "tokenizers_ready": tokenizers_ready,
        "runtime": tokenizer_runtime,
        "bf16_ready": cuda_ready,
        "required_assets": required_assets,
        "samplers": list(KREA2_SAMPLERS),
        "schedulers": list(KREA2_SCHEDULERS),
        "features": {
            # Krea 2 is the one native engine that is not guidance distilled, so the negative
            # prompt and CFG-Zero* are real. PAG stays off: it needs an identity-self-attention
            # override of the transformer's own blocks, which this runtime does not install.
            "pag": False,
            "cfg_zero_star": True,
            "negative_prompt": True,
            "distilled_guidance": False,
            # `Qwen3VLTokenizer` passes `disable_weights=True`, so `(word:1.2)` is literal text.
            "prompt_weights": False,
            "hires": True,
            "adetailer": True,
            "rtx": True,
            "staged_vae_decode": False,
            "lora": True,
            "transparent_background": True,
        },
    }


def detect_model_family(checkpoint: Path, engine: str):
    if engine == "iL":
        return "sdxl"
    if checkpoint.suffix.lower() == ".safetensors":
        with safe_open(str(checkpoint), framework="pt", device="cpu") as model:
            keys = model.keys()
            if any(key.startswith("conditioner.embedders.1") for key in keys):
                return "sdxl"
    return "sd"


def clear_pipeline():
    global loaded_pipeline, pipeline_cpu_parked, loaded_checkpoint, loaded_checkpoint_hash, loaded_family, loaded_engine
    global loaded_model_assets, loaded_model_revisions
    global active_memory_strategy, active_attention_backend, active_compute_dtype, active_vae_mode
    global active_residency
    released = loaded_pipeline is not None
    if loaded_pipeline is not None:
        if is_native_family(loaded_family):
            try:
                loaded_pipeline.close()
            except Exception:
                pass
        try:
            loaded_pipeline.unload_lora_weights()
        except Exception:
            pass
        try:
            remove_group_offload(getattr(loaded_pipeline, "unet", None))
        except Exception:
            pass
        try:
            loaded_pipeline.remove_all_hooks()
        except Exception:
            pass
        del loaded_pipeline
    loaded_pipeline = None
    active_residency = None
    pipeline_cpu_parked = False
    loaded_checkpoint = None
    loaded_checkpoint_hash = None
    loaded_family = None
    loaded_engine = None
    loaded_model_assets = {}
    loaded_model_revisions = {}
    active_memory_strategy = None
    active_attention_backend = "none"
    active_compute_dtype = "none"
    active_vae_mode = "none"
    # Only when there was something to free. A full collection plus `empty_cache` costs real time on
    # a process holding a torch allocator pool, and `clear_pipeline` is called on paths that reach it
    # with nothing loaded — where it would buy exactly nothing.
    if released:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@lru_cache(maxsize=16)
def estimate_checkpoint_weight_bytes(checkpoint_path: str, file_size: int, modified_ns: int):
    del modified_ns
    checkpoint = Path(checkpoint_path)
    if checkpoint.suffix.lower() != ".safetensors":
        return file_size
    dtype_bytes = {
        "BOOL": 1,
        "I8": 1,
        "U8": 1,
        "I16": 2,
        "U16": 2,
        "I32": 4,
        "U32": 4,
        "I64": 8,
        "U64": 8,
    }
    try:
        total = 0
        with safe_open(str(checkpoint), framework="pt", device="cpu") as model:
            for key in model.keys():
                tensor_slice = model.get_slice(key)
                elements = 1
                for dimension in tensor_slice.get_shape():
                    elements *= dimension
                dtype = str(tensor_slice.get_dtype())
                total += elements * dtype_bytes.get(dtype, 2)
        return total or file_size
    except Exception:
        return file_size


@lru_cache(maxsize=16)
def checkpoint_sha256(checkpoint_path: str, file_size: int, modified_ns: int):
    del file_size, modified_ns
    return file_sha256(Path(checkpoint_path))


def effective_cuda_free_memory():
    free_bytes, physical_total_bytes = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    active = int(stats.get("active_bytes.all.current", torch.cuda.memory_allocated()))
    reclaimable = max(0, stats.get("reserved_bytes.all.current", 0) - stats.get("active_bytes.all.current", 0))
    limit = effective_vram_limit_bytes(
        physical_total_bytes,
        performance_settings.get("vram_limit_gb", 0.0),
        allow_shared_memory=performance_settings["allow_shared_memory"],
    )
    cap_available = max(0, limit - active)
    return min(limit, free_bytes + reclaimable, cap_available), physical_total_bytes, limit


def configure_cuda_memory_limit():
    """Install the allocator guardrail; admission still enforces the same wall."""
    if not torch.cuda.is_available():
        return None
    try:
        physical_total = int(torch.cuda.get_device_properties(0).total_memory)
        limit = effective_vram_limit_bytes(
            physical_total,
            performance_settings.get("vram_limit_gb", 0.0),
            allow_shared_memory=performance_settings["allow_shared_memory"],
        )
        fraction = min(1.0, max(0.01, limit / physical_total)) if physical_total else 1.0
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        return {"physical_bytes": physical_total, "limit_bytes": limit, "fraction": fraction}
    except Exception:
        # Older CUDA builds may not expose the allocator guardrail. The policy remains enforced.
        return None


MEMORY_MODE_LABELS = {
    "high_vram": "HIGH_VRAM 高速常驻",
    "normal_vram": "NORMAL_VRAM 智能调度",
    "low_vram": "LOW_VRAM 低显存",
    "ultra_low_vram": "ULTRA_LOW 极限省存",
}
MEMORY_OFFLOAD_MODES = {
    "high_vram": "none",
    "normal_vram": "model",
    "low_vram": "sequential",
    "ultra_low_vram": "sequential_cfg_split",
}
# What each admission mode installs, in the path vocabulary AUTO reports (QPlan P8). The mode names
# stay the manual tiers' own; a job under AUTO says "AUTO · 动态决策" and names the path instead.
MEMORY_MODE_PATHS = {
    "high_vram": "resident",
    "normal_vram": "component",
    "low_vram": "submodule",
    "ultra_low_vram": "submodule",
}
# The next-heavier rung, for a Diffusers pass that runs out of memory after admission let it start.
SD_OFFLOAD_ESCALATION = {"high_vram": "normal_vram", "normal_vram": "low_vram", "low_vram": "ultra_low_vram"}


def auto_dynamic_requested():
    """Whether the user asked for AUTO · 动态决策.

    Only the literal `auto` setting. `sdxl_balanced` resolves to the automatic admission for a model
    that is not SDXL, but it is still a manual choice and keeps the pre-P8 rule — manual tiers keep
    today's behaviour (QPlan P8 §2).
    """
    return performance_settings.get("memory_mode") == "auto"


def strategy_label(strategy, mode=None, path=None):
    """The label a strategy reports: its manual tier's name, or AUTO plus the path it is on."""
    mode = mode or strategy["mode"]
    if strategy.get("dynamic"):
        return auto_label(path or strategy.get("memory_path") or MEMORY_MODE_PATHS.get(mode))
    return MEMORY_MODE_LABELS[mode]


# The Diffusers families whose UNet admission is resident-first and whose CFG branches may be run
# one after another. The native engines reach `select_memory_policy` through their own loaders and
# keep the workspace-upfront rule, which is what their staged runtimes are written against.
SD_COMPONENT_FAMILIES = frozenset({"sd", "sdxl"})
# ComfyUI batches a CFG step only when the batched activation peak clears free memory by half again
# (`comfy/samplers.py::_calc_cond_batch`), and otherwise runs the branches one at a time. The margin
# is the same here; what differs is that this is settled once for the job rather than at every step,
# so a fixed seed keeps producing the same picture.
CFG_BATCH_FREE_MEMORY_MARGIN = 1.5


def serial_cfg_eligible(family: str, guidance: str) -> bool:
    """Whether this job's CFG branches may be split into one forward each.

    Splitting is only ever a re-ordering of the same arithmetic, but it is the *batch* that PAG
    reads: `PAGCFGIdentitySelfAttnProcessor2_0` chunks `hidden_states` into three along dim 0 inside
    the UNet, so a batch of one reaches it as a batch it cannot divide. Plain CFG carries no such
    structure — Diffusers concatenates `[negative, positive]` and chunks the *result* outside the
    UNet — so that is the one guidance mode this applies to.
    """
    return family in SD_COMPONENT_FAMILIES and guidance == "none"


def choose_memory_strategy(
    checkpoint: Path,
    family: str,
    width: int,
    height: int,
    guidance_scale: float,
    lora_bytes: int,
    images_per_batch=1,
    guidance_copies=None,
    include_loaded_model=False,
    base_weight_bytes=None,
    base_largest_component_bytes=None,
    resident_first=False,
    cfg_serial_eligible=False,
):
    free_bytes, physical_total_bytes, limit_bytes = effective_cuda_free_memory()
    fixed_vram_wall = performance_settings.get("vram_limit_gb", 0.0) > 0
    budget_total_bytes = limit_bytes if fixed_vram_wall else physical_total_bytes
    checkpoint_stat = checkpoint.stat()
    if base_weight_bytes is None:
        base_weight_bytes = estimate_checkpoint_weight_bytes(str(checkpoint), checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns)
    if base_largest_component_bytes is None:
        base_largest_component_bytes = estimate_largest_component_bytes(base_weight_bytes, family)
    runtime_weight_bytes = base_weight_bytes + lora_bytes
    largest_component_bytes = min(runtime_weight_bytes, base_largest_component_bytes + lora_bytes)
    if include_loaded_model:
        loaded_bytes = min(torch.cuda.memory_allocated(), int(runtime_weight_bytes * 1.12))
        free_bytes = min(budget_total_bytes, free_bytes + loaded_bytes)
    reserved_bytes = int(RESERVED_VRAM_GB * GIB) if RESERVED_VRAM_GB is not None else reserved_vram_bytes(
        physical_total_bytes,
        allow_shared_memory=performance_settings["allow_shared_memory"],
    )
    # A fixed wall already includes the platform/driver reserve. Automatic mode keeps
    # the same reserve in admission while allowing the allocator to use the full device.
    if fixed_vram_wall:
        reserved_bytes = 0
    # This is a physical peak: callers with sequential logical guidance may
    # explicitly budget one physical forward without changing SD/iL behavior.
    logical_guidance_copies = max(1, int(guidance_copies if guidance_copies is not None else 2 if guidance_scale > 1 else 1))
    physical_forward_copies = logical_guidance_copies
    batched_inference_bytes = estimate_inference_bytes(
        family, width, height, guidance_scale, images_per_batch, logical_guidance_copies
    )
    # ComfyUI's rule, measured against the memory a resident component leaves behind rather than
    # against the bare card: `_calc_cond_batch` asks its question after the model is loaded, so the
    # comparable headroom here is what survives the largest component and the platform reserve.
    cfg_batch_headroom_bytes = max(0, free_bytes - largest_component_bytes - reserved_bytes)
    cfg_serial_applies = bool(
        cfg_serial_eligible
        and logical_guidance_copies > 1
        and batched_inference_bytes * CFG_BATCH_FREE_MEMORY_MARGIN >= cfg_batch_headroom_bytes
    )
    if cfg_serial_applies:
        physical_forward_copies = 1
    cfg_mode = "serial" if cfg_serial_applies else "batched"
    inference_bytes = (
        estimate_inference_bytes(family, width, height, guidance_scale, images_per_batch, physical_forward_copies)
        if cfg_serial_applies
        else batched_inference_bytes
    )
    requested_memory_mode = memory_mode_for_family(performance_settings["memory_mode"], family)
    policy = select_memory_policy(
        requested_memory_mode,
         total_bytes=budget_total_bytes,
        free_bytes=free_bytes,
        runtime_weight_bytes=runtime_weight_bytes,
        inference_bytes=inference_bytes,
        largest_component_bytes=largest_component_bytes,
        reserved_bytes=reserved_bytes,
        resident_first=resident_first,
    )
    mode = policy["mode"]
    if policy["requested_mode"] != "auto" and mode != policy["requested_mode"]:
        # Quote what the mode that was actually asked for needs. Quoting HIGH's requirement for a
        # NORMAL request is what made an 8 GB card report "needs about 11.2 GB" for a job whose
        # component path wanted 7.8.
        shortfall_bytes = (
            policy["high_required_bytes"] if policy["requested_mode"] == "high_vram"
            else policy["normal_required_bytes"]
        )
        reason = (
            f"配置请求 {policy['requested_mode'].upper()}，但当前模型与画布需要约 "
            f"{shortfall_bytes / GIB:.1f} GB，已安全降级"
        )
    elif mode == "high_vram":
        reason = f"自动预算通过：{free_bytes / GIB:.1f} GB 可用，底模完整常驻 GPU"
    elif mode == "normal_vram":
        reason = f"自动预算：{free_bytes / GIB:.1f} GB 可用，按组件动态调度并缓存底模"
    elif mode == "ultra_low_vram":
        reason = "极限省存：逐层卸载、串行 CFG、VAE 分块，并在任务后释放全部模型"
    else:
        reason = f"自动预算：推理至少需要约 {policy['normal_required_bytes'] / GIB:.1f} GB，启用逐层低显存保护"
    if cfg_serial_applies and mode != "ultra_low_vram":
        # Say it where the mode is said. Serial CFG is the other half of why this job fits, and a
        # health report that named only the rung would leave the halved activation peak unexplained.
        reason += "；CFG 正负分支串行推理，激活峰值减半"
    dynamic = auto_dynamic_requested()
    return {
        **policy,
        # AUTO · 动态决策: this admission is only the mount-stage decision; every later stage is
        # re-decided (`plan_residency_for_stage`, `plan_native_stage`). `legacy_mode` is what the
        # pre-P8 scheduler would have published, which is all the autoencoder's tiling may follow.
        "dynamic": dynamic,
        "legacy_mode": mode,
        "memory_path": MEMORY_MODE_PATHS[mode],
        "label": auto_label(MEMORY_MODE_PATHS[mode]) if dynamic else MEMORY_MODE_LABELS[mode],
        "reason": reason,
        "total_gb": round(budget_total_bytes / GIB, 1),
        "physical_total_gb": round(physical_total_bytes / GIB, 1),
        "vram_limit_gb": round(limit_bytes / GIB, 1),
        "free_gb": round(free_bytes / GIB, 1),
        "weight_gb": round(runtime_weight_bytes / GIB, 1),
        "inference_gb": round(inference_bytes / GIB, 1),
        "reserved_gb": round(reserved_bytes / GIB, 1),
        "base_weight_bytes": base_weight_bytes,
        "base_largest_component_bytes": base_largest_component_bytes,
        "free_bytes": free_bytes,
        "inference_bytes": inference_bytes,
        "logical_guidance_copies": logical_guidance_copies,
        "physical_forward_copies": physical_forward_copies,
        "cfg_mode": cfg_mode,
        "cfg_batch": cfg_mode == "batched" and logical_guidance_copies > 1,
        "cfg_serial_eligible": bool(cfg_serial_eligible),
        "reserved_bytes": reserved_bytes,
        "physical_total_bytes": physical_total_bytes,
        "vram_limit_bytes": limit_bytes,
        "offload_mode": MEMORY_OFFLOAD_MODES[mode],
        "model_resident": mode == "high_vram",
        "admission": {
            "normal_admission_rule": policy["normal_admission_rule"],
            "allocator_free_bytes": int(free_bytes),
            "budget_total_bytes": int(budget_total_bytes),
            "largest_component_bytes": int(largest_component_bytes),
            "normal_required_bytes": int(policy["normal_required_bytes"]),
            "normal_residency_bytes": int(policy["normal_residency_bytes"]),
            "high_required_bytes": int(policy["high_required_bytes"]),
            "cfg_mode": cfg_mode,
            "cfg_batch_headroom_bytes": int(cfg_batch_headroom_bytes),
            "cfg_batch_margin": CFG_BATCH_FREE_MEMORY_MARGIN,
            "cfg_batch_inference_bytes": int(batched_inference_bytes),
            "serial_cfg_inference_bytes": int(inference_bytes) if cfg_serial_applies else None,
            "logical_guidance_copies": logical_guidance_copies,
            "physical_forward_copies": physical_forward_copies,
            "requested_offload_mode": MEMORY_OFFLOAD_MODES[mode],
            "actual_offload_mode": MEMORY_OFFLOAD_MODES[mode],
            "fallback": None,
        },
    }


def record_group_offload_fallback(pipeline, strategy):
    """Make the reported mode the one the run actually used.

    Every native runtime may drop from a resident transformer to block streaming when a resident
    sampling pass runs out of memory — the prompt is part of the attention sequence and nothing
    caps its length, so the admission estimate can be beaten by a long enough one. When that
    happens the run still produces its picture, and saying HIGH_VRAM afterwards would describe a
    pass that did not occur. Returns whether anything changed, so the caller can publish it.
    """
    if strategy is None or not getattr(pipeline, "transformer_group_offload_enabled", False):
        return False
    if strategy.get("transformer_group_offload", False):
        return False
    strategy.update(
        mode="low_vram",
        label=strategy_label(strategy, "low_vram", NATIVE_BLOCK),
        memory_path=NATIVE_BLOCK,
        # The runtime fell back by itself mid-stage; AUTO does not promote it again this job, or an
        # estimate that was just proven wrong would buy the same failed attempt at every stage.
        native_promotion_locked=True,
        offload_mode="staged_transformer_group_offload",
        model_resident=False,
        transformer_group_offload=True,
        transformer_blocks_per_group=1,
        reason=strategy["reason"] + "；完整常驻显存不足，已自动回退单块动态装入",
    )
    admission = strategy.get("admission")
    if isinstance(admission, dict):
        admission.update(
            actual_offload_mode="staged_transformer_group_offload",
            fallback={
                "stage": "sampling",
                "reason": "cuda_oom",
                "from": "staged_transformer_resident",
                "to": "staged_transformer_group_offload",
                "attempts": 1,
                "generator_states_restored": True,
            },
        )
    return True


def reported_memory_mode(strategy):
    """The mode a job or the health report names: `auto` under AUTO, never a manual tier's name."""
    if strategy is None:
        return None
    return "auto" if strategy.get("dynamic") else strategy["mode"]


def memory_path_fields(strategy):
    """AUTO's secondary fields: which path the current stage is actually on, and the schedule so far."""
    if strategy is None or not strategy.get("dynamic"):
        return {}
    path = strategy.get("memory_path")
    return {
        "memory_path": path,
        "memory_path_label": path_label(path) if path else None,
        "memory_tier": strategy.get("residency_tier"),
        "memory_schedule": list(active_stage_schedule),
    }


def memory_job_fields(strategy, model_cached=True):
    parked = bool(model_cached and pipeline_cpu_parked)
    return {
        "memory_mode": reported_memory_mode(strategy),
        "memory_label": strategy["label"],
        "memory_reason": strategy["reason"],
        "offload_mode": "cpu_parked" if parked else strategy["offload_mode"],
        "model_resident": False if parked else strategy["model_resident"],
        "model_cached": model_cached,
        "vram_limit_gb": strategy.get("vram_limit_gb"),
        "vram_limit_bytes": strategy.get("vram_limit_bytes"),
        # The estimated weight footprint, which is what the console reports beside the memory the
        # model actually took once it was resident.
        "model_weight_bytes": strategy.get("base_weight_bytes"),
        "cfg_batch": strategy.get("cfg_batch", False),
        # Whether the guidance branches share one forward or take one each. The native engines have
        # always run them one at a time and say so through `physical_forward_copies`; naming it here
        # is what lets an SD/iL job be checked against the mode it was admitted for.
        "cfg_mode": strategy.get("cfg_mode", "batched"),
        "memory_admission": strategy.get("admission"),
        "acceleration": strategy.get("acceleration"),
        **memory_path_fields(strategy),
    }


def configure_cuda_math():
    balanced = performance_settings["cuda_math"] == "balanced"
    torch.set_float32_matmul_precision("high" if balanced else "highest")
    torch.backends.cuda.matmul.allow_tf32 = balanced
    torch.backends.cudnn.allow_tf32 = balanced
    # Balanced keeps TF32 and gives up cuDNN benchmarking, as Anima already does below. The
    # benchmark picks each convolution's algorithm by timing it in *this* process, so a fixed seed
    # drew a different picture after every restart — within a process the reruns agreed, which is
    # all anyone had checked. Measured on the reference card, two fresh processes each, fused
    # adapters, 8 steps at 768x1280 plus a whole-image decode:
    #
    #     benchmark on    latents differ between processes    sample 8.6 s   decode 11-13 s
    #     benchmark off   latents identical                   sample 6.1 s   decode 1.3 s
    #
    # So the benchmark was not buying speed either: it was the cold-process tax on every first job,
    # paid again for every new canvas shape.
    torch.backends.cudnn.benchmark = False


def configure_anima_cuda_math():
    configure_cuda_math()
    # cuDNN benchmarking may select a different VAE convolution algorithm after restart.
    torch.backends.cudnn.benchmark = False


def configure_anima_attention_backend(runtime):
    global active_attention_backend
    configure = getattr(runtime, "configure_attention_backend", None)
    if not callable(configure):
        active_attention_backend = "PyTorch SDPA"
        return active_attention_backend
    requested = performance_settings["attention_backend"]
    # `auto` follows ComfyUI, which calls `F.scaled_dot_product_attention` with no
    # `sdpa_kernel` restriction and lets PyTorch choose. Diffusers' `_native_efficient`
    # pins every attention to SDPBackend.EFFICIENT_ATTENTION, which on Ada with
    # head_dim 128 in bf16 keeps the cutlass memory-efficient kernel even where the
    # Flash kernel is both eligible and faster. Naming no backend is what lets the
    # dispatcher pick Flash for Anima's 5888-token self-attention.
    # `sage` names the kernel the processor applies, not a Diffusers backend, so
    # the dispatch stays native — that is what cross-attention and any declined
    # self-attention actually run through.
    candidate = {
        "auto": "native",
        "sdpa": "native",
        "xformers": "xformers",
        "sage": "native",
        "sliced": "native",
    }[requested]
    try:
        selected = configure(candidate)
    except (ValueError, RuntimeError, ImportError):
        selected = configure("native")
    active_attention_backend = anima_attention_backend_label(selected)
    return active_attention_backend


def configure_anima_acceleration(runtime, strategy=None):
    """Apply the two opt-in accelerators and report what actually took effect.

    Both fail soft: a missing package or a refused compilation leaves the
    runtime on its ordinary path rather than failing the job, and the returned
    record says which, so job diagnostics never claim an accelerator that is not
    running.
    """
    global active_attention_backend
    record = {
        "compile_requested": anima_compile_requested(),
        "compile_active": False,
        "compile_mode": None,
        "compile_unavailable_reason": None,
        "sage_requested": anima_sage_attention_requested(),
        "sage_active": False,
        "sage_unavailable_reason": None,
    }

    configure_sage = getattr(runtime, "configure_sage_attention", None)
    if record["sage_requested"] and callable(configure_sage):
        try:
            record["sage_active"] = bool(configure_sage(True))
            if not record["sage_active"]:
                record["sage_unavailable_reason"] = "sageattention_not_installed"
        except Exception as error:
            record["sage_unavailable_reason"] = f"{type(error).__name__}: {error}"[:300]
    elif callable(configure_sage):
        configure_sage(False)
    if record["sage_active"]:
        # Report the kernel that runs the 5888x5888 self-attention, not the
        # dispatch its cross-attention falls back to.
        active_attention_backend = "SageAttention INT8"

    configure_compile = getattr(runtime, "configure_transformer_compilation", None)
    if record["compile_requested"] and callable(configure_compile):
        group_offloaded = bool(getattr(runtime, "transformer_group_offload_enabled", False))
        if strategy is not None and strategy.get("transformer_group_offload"):
            group_offloaded = True
        if group_offloaded:
            record["compile_unavailable_reason"] = "group_offload_active"
        elif find_spec("triton") is None:
            record["compile_unavailable_reason"] = "triton_not_installed"
        else:
            try:
                mode = anima_compile_mode()
                record["compile_active"] = bool(configure_compile(True, mode))
                record["compile_mode"] = mode if record["compile_active"] else None
            except Exception as error:
                # Keep the message: the first real failure here was an upstream
                # UnicodeDecodeError whose type alone said nothing about the cause.
                record["compile_unavailable_reason"] = f"{type(error).__name__}: {error}"[:300]
    elif callable(configure_compile):
        try:
            configure_compile(False)
        except Exception:
            pass
    return record


def anima_attention_backend_label(selected):
    return {
        "_native_efficient": "PyTorch SDPA efficient",
        "_native_flash": "PyTorch SDPA flash",
        "_native_cudnn": "cuDNN SDPA",
        "xformers": "xformers",
        "native": "PyTorch SDPA",
    }.get(selected, selected)


def current_attention_backend():
    if loaded_engine == "Anima" and loaded_pipeline is not None:
        # Sage is applied by the attention processor, so the runtime's Diffusers
        # backend still reads `native` while Sage runs the 5888x5888 self-attention.
        # Reporting that dispatch would name the fallback, not the kernel.
        if getattr(loaded_pipeline, "sage_attention_enabled", False):
            return "SageAttention INT8"
        selected = getattr(loaded_pipeline, "attention_backend", None)
        if selected:
            return anima_attention_backend_label(selected)
    return active_attention_backend


def configure_attention_backend(pipeline):
    global active_attention_backend
    requested = performance_settings["attention_backend"]
    sdpa_available = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    try:
        pipeline.disable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipeline.disable_attention_slicing()

    if requested == "xformers":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            active_attention_backend = "xformers"
            return
        except Exception:
            requested = "sdpa" if sdpa_available else "sliced"
    # `sage` is an Anima-only kernel; for SD/iL it means native SDPA rather than
    # falling through to the slicing path, which would be a large speed loss.
    if requested in {"auto", "sdpa", "sage"} and sdpa_available:
        active_attention_backend = "PyTorch SDPA"
        return
    if requested == "auto":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            active_attention_backend = "xformers"
            return
        except Exception:
            pass
    pipeline.enable_attention_slicing("max")
    active_attention_backend = "sliced attention"


def configure_vae_mode(pipeline, strategy):
    global active_vae_mode
    requested = performance_settings["vae_mode"]
    mode = requested
    if requested == "auto":
        # `legacy_mode`, not `mode`: tiling decides pixels, and under AUTO the rung the model is on
        # no longer is the rung the pre-P8 scheduler would have published. Following the legacy rung
        # keeps every decode exactly where it was (QPlan P8 §2 pixel identity); under a manual tier
        # the two are the same thing.
        mode = {
            "high_vram": "full",
            "normal_vram": "sliced",
            "low_vram": "tiled",
            "ultra_low_vram": "tiled",
        }[strategy.get("legacy_mode") or strategy["mode"]]
    # On the autoencoder itself: the pipeline-level wrappers are deprecated in Diffusers 0.38 and
    # printed a FutureWarning on every reconfigure, which the console showed as an error.
    vae = pipeline.vae
    vae.disable_slicing()
    vae.disable_tiling()
    if mode in {"sliced", "tiled"}:
        vae.enable_slicing()
    if mode == "tiled":
        vae.enable_tiling()
    active_vae_mode = mode


def resolve_compute_dtype():
    if performance_settings["compute_dtype"] == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def apply_memory_strategy(pipeline, strategy):
    mode = strategy["mode"]
    # Take the previous guidance wrapper off before the new offload mechanism goes on. Accelerate
    # captures whatever `forward` it finds as `_old_forward`, so a wrapper left in place would be
    # captured under the new hook, where nothing can reach it again — and the job that asked for
    # batched CFG would go on splitting every batch while reporting that it does not.
    unet = getattr(pipeline, "unet", None)
    if unet is not None:
        disable_sequential_batch_forward(unet)
    # Once a controller owns this pipeline it owns the mechanism too, and the rung it is on is not
    # always recoverable from `mode` — T3 and T4 both report `low_vram`. Asking it to reinstall is
    # what stops a restore from silently dropping a streaming job onto the per-submodule path.
    if active_residency is not None and active_residency.pipeline is pipeline and active_residency.tier:
        active_residency.reinstall()
        apply_cfg_execution_mode(pipeline, strategy)
        configure_vae_mode(pipeline, strategy)
        return
    if mode == "high_vram":
        pipeline.to("cuda")
    elif mode == "normal_vram":
        pipeline.enable_model_cpu_offload(device="cuda")
    else:
        if active_attention_backend == "sliced attention":
            pipeline.enable_attention_slicing("max")
        pipeline.enable_sequential_cpu_offload(device="cuda")
    apply_cfg_execution_mode(pipeline, strategy)
    configure_vae_mode(pipeline, strategy)


def apply_cfg_execution_mode(pipeline, strategy):
    """Put the UNet into the guidance execution order this job was admitted for.

    The wrapper goes on *after* the offload hooks, so each split forward still passes through
    whatever hook the chosen rung installed. Ultra-low keeps running serial unconditionally — it is
    a survival mode and the halved activation peak is the whole point — but it no longer does so for
    PAG, whose attention processor divides the batch itself and cannot be handed one sample at a
    time. Removing the wrapper is as necessary as installing it: a resident pass carries no
    accelerate hooks, so nothing else would ever take it off again.
    """
    unet = getattr(pipeline, "unet", None)
    if unet is None:
        return
    serial = strategy.get("cfg_mode") == "serial" or (
        strategy["mode"] == "ultra_low_vram" and strategy.get("cfg_serial_eligible", True)
    )
    if serial:
        enable_sequential_batch_forward(unet)
    else:
        disable_sequential_batch_forward(unet)


def residency_job_fields():
    """What the controller has done so far, for the job record and for health."""
    if active_residency is None:
        return {"residency": None, "residency_trace": []}
    return {"residency": active_residency.snapshot(), "residency_trace": list(active_residency.trace)}


def publish_residency(job_id=None):
    """Make the reported mode the rung the run is actually on.

    `memory_mode` and `offload_mode` are what every existing report reads, so a tier change has to
    land in them or the job would describe an offload path it stopped using two stages ago.
    """
    global active_memory_strategy
    if active_residency is None or active_memory_strategy is None:
        return
    tier = active_residency.tier
    if tier is None:
        return
    mode = RESIDENCY_TIER_MEMORY_MODES.get(tier, active_memory_strategy["mode"])
    legacy_tier = (active_residency.legacy_tier or tier) if active_residency.dynamic else tier
    path = path_for_tier(tier)
    active_memory_strategy = {
        **active_memory_strategy,
        "mode": mode,
        # Under a manual tier the legacy rung is the real one; under AUTO it is the rung the pre-P8
        # rule would have published, and the autoencoder's tiling is reconfigured from it.
        "legacy_mode": RESIDENCY_TIER_MEMORY_MODES.get(legacy_tier, mode),
        "memory_path": path,
        "label": strategy_label(active_memory_strategy, mode, path),
        "offload_mode": RESIDENCY_TIER_OFFLOAD_MODES.get(tier, active_memory_strategy["offload_mode"]),
        "model_resident": tier == "T0",
        "residency_tier": tier,
    }
    if job_id is not None:
        update_job(job_id, **memory_job_fields(active_memory_strategy), **residency_job_fields())


STAGE_SCHEDULE_LIMIT = 64


def stage_scheduling_active(strategy=None):
    """Whether this job is scheduled by AUTO · 动态决策 (decided at admission, fixed for the job)."""
    strategy = active_memory_strategy if strategy is None else strategy
    return bool(strategy and strategy.get("dynamic"))


def platform_reserve_bytes(physical_total_bytes):
    """The platform/driver reserve admission keeps; a fixed VRAM wall already includes it."""
    if performance_settings.get("vram_limit_gb", 0.0) > 0:
        return 0
    if RESERVED_VRAM_GB is not None:
        return int(RESERVED_VRAM_GB * GIB)
    return reserved_vram_bytes(
        physical_total_bytes, allow_shared_memory=performance_settings["allow_shared_memory"],
    )


def read_stage_budget(resident_bytes=0):
    """One reading of everything a stage decision needs (QPlan P8 §3.2), as a `StageBudget`.

    Driver free memory, PyTorch's allocated / reserved / active bytes, the budget scope (device or
    VRAM wall), the platform reserve and the pipeline's own resident weight. Memory other processes
    hold is not estimated: it is simply absent from the driver's free figure.
    """
    free_bytes, physical_total_bytes = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    allocated = int(torch.cuda.memory_allocated())
    reserved = int(stats.get("reserved_bytes.all.current", torch.cuda.memory_reserved()))
    active = int(stats.get("active_bytes.all.current", allocated))
    wall = performance_settings.get("vram_limit_gb", 0.0) > 0
    limit = effective_vram_limit_bytes(
        physical_total_bytes,
        performance_settings.get("vram_limit_gb", 0.0),
        allow_shared_memory=performance_settings["allow_shared_memory"],
    )
    return StageBudget(
        driver_free_bytes=int(free_bytes),
        physical_total_bytes=int(physical_total_bytes),
        limit_bytes=int(limit),
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        active_bytes=active,
        resident_bytes=int(resident_bytes),
        platform_reserve_bytes=platform_reserve_bytes(physical_total_bytes),
        allocator_ceiling_bytes=int(limit) if wall else None,
    )


def reset_stage_schedule():
    global active_stage_schedule
    active_stage_schedule = []


def record_stage_decision(record, job_id=None):
    """Append one stage decision to the job's schedule and put it in front of whoever is watching."""
    active_stage_schedule.append(record)
    if len(active_stage_schedule) > STAGE_SCHEDULE_LIMIT:
        del active_stage_schedule[:-STAGE_SCHEDULE_LIMIT]
    if job_id is not None:
        update_job(job_id, memory_schedule=list(active_stage_schedule), memory_stage=record)
    return record


def record_mount_decision(job_id, family, engine, width, height, reused):
    """Stage 1 of the matrix: where the model was mounted, sized by weights, adapters and the canvas.

    The admission is the mount-stage decision — it has to place the weights somewhere before any
    stage can be asked about — and a reused checkpoint/LoRA combination says so rather than
    pretending it was admitted again.
    """
    strategy = active_memory_strategy
    if not stage_scheduling_active(strategy):
        return None
    if is_native_family(family):
        tier = None
        path = NATIVE_BLOCK if strategy.get("transformer_group_offload") else NATIVE_RESIDENT
    else:
        tier = active_residency.tier if active_residency is not None else tier_for_memory_mode(strategy["mode"])
        path = path_for_tier(tier)
    admission = strategy.get("admission") or {}
    required = (
        admission.get("sampling_stage_required_bytes")
        or (strategy.get("high_required_bytes") if strategy["mode"] == "high_vram" else strategy.get("normal_required_bytes"))
    )
    return record_stage_decision(decision_record(
        stage="model_mount", engine=engine, family=family, path=path, tier=tier, canvas=(width, height),
        reason="mount_reused" if reused else "mount_admitted", required_bytes=required,
        detail={
            "admission_mode": strategy["mode"],
            "weight_bytes": strategy.get("base_weight_bytes"),
            "adapter_bytes": strategy.get("adapter_source_bytes"),
            "free_bytes": strategy.get("free_bytes"),
            "reused": bool(reused),
        },
    ), job_id)


def record_decode_decision(stage, family, canvas, evicted, job_id=None, reason=None, peak_bytes=None, tiled=None):
    """A decode boundary: whether the diffusion model left the card before the autoencoder ran."""
    if not stage_scheduling_active():
        return None
    return record_stage_decision(decision_record(
        stage=stage, engine=loaded_engine, family=family, path="evicted" if evicted else "shared",
        canvas=canvas, reason=reason or ("evict_before_decode" if evicted else "decode_beside_model"),
        required_bytes=peak_bytes, detail={"tiled": tiled} if tiled is not None else None,
    ), job_id)


NATIVE_PLACEMENT_METHODS = (
    "enable_transformer_group_offload",
    "_remove_transformer_group_offload",
    "configure_transformer_residency",
)


def native_resident_bytes(runtime):
    """How much of a native runtime's weight is on the card right now, shared storage counted once."""
    seen = set()
    total = 0
    for module in (getattr(runtime, "components", {}) or {}).values():
        if not isinstance(module, torch.nn.Module):
            continue
        for tensor in (*module.parameters(), *module.buffers()):
            if tensor.device.type != "cuda":
                continue
            pointer = tensor.data_ptr() or id(tensor)
            if pointer in seen:
                continue
            seen.add(pointer)
            total += tensor.numel() * tensor.element_size()
    return total


def native_placement_lock(runtime, strategy, current):
    """Why a native runtime's transformer must stay where it is for this stage, or None.

    Placement is only ever weight location. Where moving the transformer would change arithmetic or
    lose work, it is not moved: a compiled transformer cannot take group-offload hooks and come out
    the same, and Anima fuses its LoRA on the first GPU onload — hooks installed before that would
    leave the adapter unapplied. A runtime that fell back to streaming by itself this job is not
    promoted again: the estimate that admitted residency has just been shown to be wrong.
    """
    if not all(callable(getattr(runtime, name, None)) for name in NATIVE_PLACEMENT_METHODS):
        return "runtime_has_no_placement_hooks"
    if getattr(runtime, "_poisoned", False) or getattr(runtime, "_closed", False):
        return "runtime_unavailable"
    if bool((strategy.get("acceleration") or {}).get("compile_active")):
        return "compiled_transformer"
    if getattr(runtime, "lora_plans", None) and not getattr(runtime, "_diffusion_lora_applied", True):
        return "adapter_fusion_pending"
    if current == NATIVE_BLOCK and strategy.get("native_promotion_locked"):
        return "runtime_fallback_this_job"
    return None


def plan_native_stage(stage, family, canvas, guidance_scale, images_per_batch, runtime, job_id=None):
    """Re-decide a native runtime's transformer placement for the stage about to run (QPlan P8).

    Admission placed the transformer against the chain's most demanding stage; under AUTO each
    sampling stage — the base pass, every resample unit, every ADetailer region — is placed by what
    it needs, through the same decision engine and the same cap rule as the component ladder, and
    recorded in the same schedule. Resident when it fits (and, from streaming, with 15% to spare);
    one block at a time otherwise; a readable stage error when not even that fits.
    """
    global active_memory_strategy
    strategy = active_memory_strategy
    if (
        not stage_scheduling_active(strategy)
        or not is_native_family(family)
        or runtime is None
        or not torch.cuda.is_available()
    ):
        return None
    weights = getattr(runtime, "weight_sizes", {}) or {}
    transformer_bytes = int(weights.get("transformer", 0) or 0)
    block_bytes = int(weights.get("transformer_max_block", 0) or 0) + int(weights.get("transformer_unmatched", 0) or 0)
    if transformer_bytes <= 0 or block_bytes <= 0:
        return None
    copies = int(strategy.get("actual_physical_forward_copies") or strategy.get("physical_forward_copies") or 1)
    batch = int(strategy.get("sampling_batch_size") or images_per_batch or 1)
    inference_bytes = estimate_inference_bytes(family, canvas[0], canvas[1], guidance_scale, batch, copies)
    reserve = int(strategy.get("reserved_bytes", 0) or 0)
    # The same rule admission uses for these runtimes: the transformer (or one block group) with 10%
    # for the allocator, the stage's activations and the platform reserve.
    resident_required = int(transformer_bytes * 1.10) + inference_bytes + reserve
    block_required = int(block_bytes * 1.10) + inference_bytes + reserve
    budget = read_stage_budget(native_resident_bytes(runtime))
    margin = allocator_safety_margin_bytes(budget.limit_bytes)
    current = NATIVE_BLOCK if getattr(runtime, "transformer_group_offload_enabled", False) else NATIVE_RESIDENT
    lock = native_placement_lock(runtime, strategy, current)
    choice = choose_native_placement(
        current=current,
        resident_required=resident_required,
        block_required=block_required,
        budget=budget,
        margin_bytes=margin,
        resident_workspace=inference_bytes,
        block_workspace=inference_bytes,
        transformer_bytes=transformer_bytes,
        block_bytes=block_bytes,
        locked=lock is not None,
    )
    target = choice.tier
    reason = choice.reason
    if target == NATIVE_BLOCK and reason == "stage_plan_floor":
        if block_required > budget.planning_free_bytes:
            raise StageMemoryError(
                stage, canvas, block_required, budget.planning_free_bytes, detail=f"{family} 单块流式",
            )
        reason = "stage_floor_without_margin"
    if target != current:
        blocks = int(strategy.get("transformer_blocks_per_group") or 1)
        if target == NATIVE_BLOCK:
            runtime.configure_transformer_residency(False)
            runtime.enable_transformer_group_offload(max(1, blocks))
        else:
            runtime._remove_transformer_group_offload()
            runtime.configure_transformer_residency(bool(strategy.get("keep_transformer_resident", False)))
        block = target == NATIVE_BLOCK
        strategy.update(
            transformer_group_offload=block,
            transformer_blocks_per_group=max(1, blocks) if block else 0,
            offload_mode="staged_transformer_group_offload" if block else "staged_transformer_resident",
            mode="low_vram" if block else (strategy.get("native_resident_mode") or "normal_vram"),
        )
    if strategy.get("mode") != "low_vram":
        strategy["native_resident_mode"] = strategy["mode"]
    strategy.update(memory_path=target, label=strategy_label(strategy, path=target))
    invariant = cap_invariant(
        budget,
        inference_bytes + (block_bytes if target == NATIVE_BLOCK else transformer_bytes),
        margin,
    )
    record_stage_decision(decision_record(
        stage=stage, engine=loaded_engine, family=family, path=target, canvas=canvas, reason=reason,
        budget=budget, required_bytes=resident_required if target == NATIVE_RESIDENT else block_required,
        workspace_bytes=inference_bytes, invariant=invariant, changed=target != current,
        detail={"lock": lock, **choice.fits},
    ), job_id)
    if job_id is not None:
        update_job(job_id, **memory_job_fields(strategy))
    return target


def static_prompt(prompt, negative_prompt):
    """One conditioning state for the whole pass: no schedule, no alternation, no `AND`."""
    try:
        syntax = compile_conditioning_syntax(prompt, negative_prompt)
    except Exception:
        return False
    return not syntax.dynamic and not syntax.composable


def stage_prompt_conditioning(pipeline, family, prompt, negative_prompt, *, timeline, clip_skip,
                              before_encode=None):
    """SD/iL prompt conditioning for one stage, encoded once per job when the prompt is static.

    The base pass, every resample redraw and every ADetailer unit ask for the conditioning of the
    same strings through towers that do not change inside a job, so a static prompt is encoded by
    the first stage and handed to the rest (QPlan P8 §3.5). The key carries the text, the negative,
    CLIP skip, the family, the towers and the fused adapter set; a dynamic prompt, or an adapter
    state that cannot be trusted, is encoded exactly as before and never shared. `before_encode`
    runs only when a real encode is about to happen — a stage that reuses the shared encoding moves
    no tower and needs no plan for it.
    """
    def encode():
        if before_encode is not None:
            before_encode()
        return prepare_dynamic_prompt_conditioning(
            pipeline, family, prompt, negative_prompt, timeline=timeline, clip_skip=clip_skip,
        )

    cache = active_conditioning_cache
    if cache is None:
        return encode()
    owner = loaded_pipeline if loaded_pipeline is not None else pipeline
    adapters = lora_fusion_signature(owner)
    if adapters is None or not static_prompt(prompt, negative_prompt):
        return cache.bypass(encode)
    key = cache.key(
        family=family, prompt=prompt, negative_prompt=negative_prompt, clip_skip=clip_skip,
        emphasis="forge", model_signature=tower_signature(pipeline, loaded_checkpoint), lora_signature=adapters,
    )
    return cache.get_or_encode(key, encode)


# What one tiled pass of an ESRGAN-class upscaler allocates beside its weights: the RRDB trunk at the
# tile's own size and the upsampled feature maps at up to four times it. Generous on purpose — the
# alternative to parking first is the old answer, an OOM followed by the same park and a retry.
UPSCALER_TILE_WORKSPACE_BYTES = 768 * 1024**2


def plan_upscale_stage(stage, pipeline, family, source_size, destination, settings, job_id=None):
    """Stage 5a: whether the diffusion model has to leave the card before a GPU upscaler runs.

    The upscaler used to be tried beside the model and the model parked only after an OOM. Under
    AUTO the question is asked first: if what is free beside the model cannot hold the upscaler's
    weights and one tile's working set with the band's margin, the model is parked before it runs,
    and restored after it exactly as the OOM path did. The upscaler runs on the GPU either way.
    """
    if not stage_scheduling_active() or not torch.cuda.is_available():
        return None
    try:
        model_bytes = int(resolve_upscaler_model(UPSCALER_MODEL_DIRECTORY, settings.upscaler).stat().st_size)
    except Exception:
        model_bytes = 64 * 1024**2
    native = is_native_family(family)
    resident = native_resident_bytes(pipeline) if native else (
        active_residency.resident_bytes() if active_residency is not None else 0
    )
    budget = read_stage_budget(resident)
    margin = allocator_safety_margin_bytes(budget.limit_bytes)
    workspace = 2 * model_bytes + UPSCALER_TILE_WORKSPACE_BYTES
    fits = budget.ambient_free_bytes >= workspace + margin
    park = not fits and not native and not pipeline_cpu_parked
    if park:
        update_job(job_id, phase=f"{stage_display_label(stage)} · Parking diffusion model", stage="sampler_offload")
        park_pipeline_for_external_stage(pipeline, family)
    return record_stage_decision(decision_record(
        stage=stage, engine=loaded_engine, family=family, path="parked" if park else "shared",
        canvas=destination, reason="park_before_upscale" if park else "upscale_beside_model",
        budget=budget, required_bytes=workspace + margin, workspace_bytes=workspace,
        detail={"source_size": list(source_size), "model_bytes": model_bytes},
    ), job_id)


def record_native_decode(stage, family, canvas, job_id=None):
    """A native runtime decodes inside its own call; whether it evicts first is the configured setting.

    The runtimes choose tiled or whole decoding from the free memory they see at that moment, so
    AUTO does not move the transformer around their decode — that would change what they see, and
    with it the pixels. The schedule records the decision the setting made.
    """
    evicted = bool(performance_settings.get("unload_before_decode", True))
    return record_decode_decision(
        stage, family, canvas, evicted=evicted, job_id=job_id, reason="configured_unload_before_decode",
    )


def record_final_decode(family, canvas, job_id=None):
    """Stage 7: the decode that produced the picture about to be saved, and how it ran."""
    if not stage_scheduling_active():
        return None
    last = None
    for record in reversed(active_stage_schedule):
        stage = str(record.get("stage") or "")
        if stage.startswith(("base_decode", "hires_decode")):
            last = record
            break
    evicted = last.get("path") == "evicted" if last is not None else bool(
        performance_settings.get("unload_before_decode", True)
    )
    return record_decode_decision(
        "final_decode", family, canvas, evicted=evicted, job_id=job_id, reason="final_decode_and_save",
        peak_bytes=(last or {}).get("required_bytes"),
    )


def stage_display_label(stage):
    return str(stage).replace("_", " ")


def component_decode_bytes(pipeline, width, height, images_per_batch=1):
    """The autoencoder peak of a Diffusers call that decodes before it returns.

    Whole-image when the autoencoder is untiled; bounded by one tile plus the output buffer when it
    is tiled (measured 4.09 GiB for a tiled 1152x1920 SDXL decode, `estimate_vae_decode_bytes`).
    """
    vae = getattr(pipeline, "vae", None)
    sliced = bool(getattr(vae, "use_slicing", False))
    whole = estimate_vae_decode_bytes(width, height, images_per_batch=images_per_batch, sliced=sliced)
    if bool(getattr(vae, "use_tiling", False)):
        copies = 1 if sliced else max(1, int(images_per_batch))
        tiled = estimate_vae_decode_bytes(1024, 1024) * copies + 12 * int(width) * int(height) * copies
        return min(whole, tiled)
    return whole


def plan_residency_for_stage(stage, family, canvas, guidance_scale, images_per_batch, job_id=None, *,
                             kind="sampling", decode_in_call=False, legacy_point=True, inference_bytes=None):
    """Re-ask where the diffusion model should live, for the stage about to run.

    This is the difference Phase 1 exists to make. Admission sizes the whole chain against its most
    demanding stage, so a 768x1280 base pass used to be placed by what a 1152x1920 redraw needed;
    here each sampling stage is placed by what *it* needs against the memory free at that moment.
    Only the Diffusers families have a controller — the native engines go through
    `plan_native_stage`.

    Under AUTO (QPlan P8) the same call also carries what kind of stage it is — a sampling pass, a
    prompt encode — and whether the call decodes before it returns, and the decision is recorded in
    the job's stage schedule. A stage that cannot run even on the floor raises `StageMemoryError`
    rather than starting; any other failure to install a rung keeps the previous one, as before.
    """
    if torch.cuda.is_available() and pipeline_cpu_parked and not is_native_family(family):
        restore_parked_pipeline()
        if pipeline_cpu_parked:
            raise RuntimeError("扩散模型无法从 CPU 停放状态恢复，重绘不能在 GPU 上运行；请重启推理服务")
    if active_residency is None or family not in SD_COMPONENT_FAMILIES or loaded_pipeline is None:
        return None
    copies = int((active_memory_strategy or {}).get("physical_forward_copies", 1))
    if inference_bytes is None:
        inference_bytes = estimate_inference_bytes(
            family, canvas[0], canvas[1], guidance_scale, images_per_batch, copies
        )
    dynamic = active_residency.dynamic
    demand = None
    if dynamic:
        demand = {"kind": kind, "inference_bytes": int(inference_bytes)}
        if decode_in_call and canvas:
            demand["decode_bytes"] = component_decode_bytes(
                loaded_pipeline, canvas[0], canvas[1], images_per_batch
            )
    previous = active_residency.tier
    try:
        tier = active_residency.plan_for_stage(
            stage, inference_bytes, canvas=canvas, demand=demand, legacy_point=legacy_point,
        )
    except StageMemoryError:
        publish_residency(job_id)
        raise
    except Exception:
        # A rung that will not install is not a reason to lose the job: the pipeline keeps whatever
        # it had, and the run continues on it.
        traceback.print_exc()
        try:
            active_residency.set_tier(previous, reason="plan_failed_restore", stage=stage)
        except Exception:
            traceback.print_exc()
        return None
    if dynamic and active_residency.last_decision is not None:
        entry = active_residency.last_decision
        record_stage_decision(decision_record(
            stage=stage, engine=loaded_engine, family=family, path=path_for_tier(tier), tier=tier,
            canvas=canvas, reason=entry.get("reason"), changed=entry.get("changed", False),
            required_bytes=entry.get("required_bytes"), workspace_bytes=entry.get("workspace_bytes"),
            legacy_tier=active_residency.legacy_tier,
            # The same top-level cap fields a native decision carries, so one reader serves both.
            invariant={
                "cap_bytes": entry.get("cap_bytes"),
                "required_cap_bytes": entry.get("required_cap_bytes"),
                "holds": entry.get("cap_holds"),
            },
            detail={
                "kind": kind,
                "free_bytes": entry.get("free_bytes"),
                "ambient_free_bytes": entry.get("ambient_free_bytes"),
                "cap_demotions": entry.get("cap_demotions") or None,
                "decode_in_call": bool(decode_in_call),
            },
        ), job_id)
    if tier != previous:
        # Deliberately *not* reconfiguring the VAE here. A tiled decode and a whole-image decode do
        # not produce the same pixels, so letting the rung redecide it would make the picture depend
        # on where the weights were living; `whole_image_decode_fits` decides it per decode, with
        # the free memory of that moment, which is both safer and more informed.
        apply_cfg_execution_mode(loaded_pipeline, active_memory_strategy or {})
    publish_residency(job_id)
    return tier


def reconfigure_memory_strategy(pipeline, strategy):
    # Diffusers' group-offload hooks live under `_diffusers_hook` and `remove_all_hooks` removes
    # none of them, while `enable_model_cpu_offload` refuses to install over them. Taking them off
    # first is what keeps the two mechanisms from ever overlapping (plan P1-3).
    remove_group_offload(getattr(pipeline, "unet", None))
    pipeline.remove_all_hooks()
    pipeline.to("cpu", silence_dtype_warnings=True)
    pipeline.disable_attention_slicing()
    configure_cuda_math()
    configure_attention_backend(pipeline)
    apply_memory_strategy(pipeline, strategy)


def park_pipeline_for_vae(pipeline):
    global pipeline_cpu_parked
    try:
        remove_group_offload(getattr(pipeline, "unet", None))
        pipeline.remove_all_hooks()
        for name, component in pipeline.components.items():
            if name != "vae" and isinstance(component, torch.nn.Module):
                component.to("cpu")
        pipeline.vae.to("cpu")
    except Exception:
        pipeline_cpu_parked = False
        if pipeline is loaded_pipeline:
            clear_pipeline()
        raise
    gc.collect()
    torch.cuda.empty_cache()
    pipeline_cpu_parked = True
    if active_residency is not None and active_residency.pipeline is pipeline:
        active_residency.note_parked()


def park_pipeline_for_external_stage(pipeline, family: str):
    if not is_native_family(family):
        park_pipeline_for_vae(pipeline)
        return
    try:
        pipeline.to_cpu()
    except Exception:
        if pipeline is loaded_pipeline:
            clear_pipeline()
        raise
    gc.collect()
    torch.cuda.empty_cache()


def prepare_vae_latents(pipeline, latents, family):
    vae = pipeline.vae
    needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
    if needs_upcasting:
        vae.to(dtype=torch.float32)
        latents = latents.to(dtype=next(iter(vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != vae.dtype:
        latents = latents.to(dtype=vae.dtype)

    if family == "sdxl" and getattr(vae.config, "latents_mean", None) is not None and getattr(vae.config, "latents_std", None) is not None:
        mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        latents = latents * std / vae.config.scaling_factor + mean
    else:
        latents = latents / vae.config.scaling_factor
    return latents, needs_upcasting


def pipeline_execution_device(pipeline):
    """Resolve Diffusers' onload device without moving a component or installing another hook."""
    device = getattr(pipeline, "_execution_device", None)
    if device is not None:
        return torch.device(device)
    for component_name in ("unet", "vae"):
        component = getattr(pipeline, component_name, None)
        if component is None:
            continue
        tensor = next(component.parameters(), None)
        if tensor is None:
            tensor = next(component.buffers(), None)
        if tensor is not None:
            return tensor.device
    return torch.device("cpu")


def module_dtype(module, fallback=torch.float32):
    dtype = getattr(module, "dtype", None)
    if dtype is not None:
        return dtype
    tensor = next(module.parameters(), None)
    if tensor is None:
        tensor = next(module.buffers(), None)
    return tensor.dtype if tensor is not None else fallback


def pipeline_sampling_dtype(pipeline):
    unet = getattr(pipeline, "unet", None)
    dtype = getattr(unet, "dtype", None)
    if dtype is not None:
        return dtype
    if unet is not None:
        tensor = next(unet.parameters(), None)
        if tensor is not None:
            return tensor.dtype
    return module_dtype(pipeline.vae)


def pipeline_vae_scale_factor(pipeline) -> int:
    """Read the pipeline's real VAE stride; latent geometry must never assume eight."""
    raw = getattr(pipeline, "vae_scale_factor", None)
    if isinstance(raw, bool):
        raw = None
    try:
        factor = int(raw)
    except (TypeError, ValueError) as error:
        raise RuntimeError("The image pipeline does not expose a valid VAE scale factor") from error
    if factor < 1 or factor != raw:
        raise RuntimeError(f"Invalid VAE scale factor for latent resampling: {raw!r}")
    return factor


def encode_image_to_latents(image_pipeline, image: Image.Image, generator: torch.Generator):
    """Encode one picture exactly as Diffusers 0.38 img2img prepares its initial latents.

    The caller reuses ``generator`` for the subsequent img2img call. Sampling the VAE distribution
    here therefore consumes the same random stream that Diffusers would have consumed before drawing
    the sampler noise; creating a second generator would change the fixed-seed trajectory.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Latent resampling requires a PIL image to encode")
    if not isinstance(generator, torch.Generator):
        raise TypeError("Latent resampling requires a torch.Generator")
    if torch.cuda.is_available() and image_pipeline is loaded_pipeline and pipeline_cpu_parked:
        raise RuntimeError("扩散模型仍停放在 CPU，无法执行潜空间编码；请重启推理服务")

    vae = image_pipeline.vae
    original_dtype = module_dtype(vae)
    sampling_dtype = pipeline_sampling_dtype(image_pipeline)
    device = pipeline_execution_device(image_pipeline)
    pixels = image_pipeline.image_processor.preprocess(image)
    if not torch.isfinite(pixels).all():
        raise ValueError("Latent resampling input contains non-finite pixels")
    pixels = pixels.to(device=device, dtype=sampling_dtype)
    needs_upcasting = bool(
        original_dtype == torch.float16 and getattr(vae.config, "force_upcast", False)
    )
    try:
        if needs_upcasting:
            pixels = pixels.float()
            vae.to(dtype=torch.float32)
        with torch.inference_mode():
            encoder_output = vae.encode(pixels)
            if hasattr(encoder_output, "latent_dist"):
                latents = encoder_output.latent_dist.sample(generator)
            elif hasattr(encoder_output, "latents"):
                latents = encoder_output.latents
            else:
                raise AttributeError("VAE encoder output does not expose latents")
        latents = latents.to(dtype=sampling_dtype)
        mean_values = getattr(vae.config, "latents_mean", None)
        std_values = getattr(vae.config, "latents_std", None)
        if mean_values is not None and std_values is not None:
            mean = torch.as_tensor(mean_values, device=latents.device, dtype=latents.dtype).view(1, 4, 1, 1)
            std = torch.as_tensor(std_values, device=latents.device, dtype=latents.dtype).view(1, 4, 1, 1)
            latents = (latents - mean) * vae.config.scaling_factor / std
        else:
            latents = latents * vae.config.scaling_factor
        if latents.ndim != 4 or latents.shape[0] != 1 or latents.shape[1] != 4:
            raise RuntimeError(f"VAE encoding returned an invalid latent shape: {tuple(latents.shape)}")
        if not torch.isfinite(latents).all():
            raise RuntimeError("VAE encoding produced non-finite latents")
        return latents
    finally:
        if needs_upcasting:
            vae.to(dtype=original_dtype)


def interpolate_image_latents(image_pipeline, latents, target_size, latent_algorithm: str):
    """Apply a Forge latent mode on the largest VAE grid contained by the pixel target."""
    mode = LATENT_UPSCALE_MODES.get(latent_algorithm)
    if mode is None:
        raise ValueError(f"Unsupported latent resampling algorithm: {latent_algorithm}")
    target_width, target_height = (int(target_size[0]), int(target_size[1]))
    factor = pipeline_vae_scale_factor(image_pipeline)
    latent_width = max(1, target_width // factor)
    latent_height = max(1, target_height // factor)
    if not isinstance(latents, torch.Tensor) or latents.ndim != 4 or latents.shape[1] != 4:
        raise ValueError("Latent resampling requires a four-channel BCHW tensor")
    target_device = pipeline_execution_device(image_pipeline)
    target_dtype = pipeline_sampling_dtype(image_pipeline)
    latents = latents.to(device=target_device, dtype=target_dtype)
    if not torch.isfinite(latents).all():
        raise RuntimeError("Latent resampling received non-finite latents")
    # CPU is a probe/fallback device for this CUDA-only service. Torch does not implement every
    # antialiased half-precision CPU kernel, so compute there in float32 and project back to the
    # sampler dtype; CUDA keeps the exact model dtype used by Forge and by the following sampler.
    interpolation_input = (
        latents.float()
        if target_device.type == "cpu" and target_dtype in {torch.float16, torch.bfloat16}
        else latents
    )
    resized = torch.nn.functional.interpolate(
        interpolation_input,
        size=(latent_height, latent_width),
        mode=mode["mode"],
        antialias=mode["antialias"],
    ).to(dtype=target_dtype)
    if not torch.isfinite(resized).all():
        raise RuntimeError("Latent interpolation produced non-finite latents")
    return resized, factor


def decode_resident_latents(
    pipeline, latents, family, job_id, control, *, start_progress, final_progress, phase_prefix,
):
    """Decode scaled SD/SDXL latents without changing the configured residency mechanism."""
    if torch.cuda.is_available() and pipeline is loaded_pipeline and pipeline_cpu_parked:
        restore_parked_pipeline()
        if pipeline_cpu_parked:
            raise RuntimeError("扩散模型无法从 CPU 停放状态恢复，潜空间结果无法解码；请重启推理服务")
    control.checkpoint(job_id, f"{phase_prefix} latent decode")
    update_job(
        job_id,
        phase=f"{phase_prefix} · VAE decode",
        stage="resample_latent_decode",
        stage_step=0,
        stage_total=1,
        progress=start_progress,
    )
    vae = pipeline.vae
    original_dtype = module_dtype(vae)
    decoded = None
    prepared = None
    try:
        prepared, _needs_upcasting = prepare_vae_latents(pipeline, latents, family)
        prepared = prepared.to(pipeline_execution_device(pipeline))
        with torch.inference_mode():
            decoded = vae.decode(prepared, return_dict=False)[0]
        if not torch.isfinite(decoded).all():
            raise RuntimeError("VAE decode produced non-finite pixels")
        watermark = getattr(pipeline, "watermark", None)
        if family == "sdxl" and watermark is not None:
            decoded = watermark.apply_watermark(decoded)
        images = pipeline.image_processor.postprocess(decoded, output_type="pil")
        control.checkpoint(job_id, f"{phase_prefix} latent decode")
        update_job(job_id, stage_step=1, stage_total=1, progress=final_progress)
        return images
    finally:
        if getattr(vae, "dtype", original_dtype) != original_dtype:
            vae.to(dtype=original_dtype)
        del decoded, prepared


@dataclass
class LatentImageArtifact:
    """A scaled four-channel latent with a logical size and memoized PIL materialization."""

    latents: torch.Tensor
    pixel_size: tuple[int, int]
    family: str
    vae_scale_factor: int
    decoder: Callable[[torch.Tensor], Image.Image] = field(repr=False)
    _image: Image.Image | None = field(default=None, init=False, repr=False)

    @property
    def size(self):
        return self.pixel_size

    def materialize(self) -> Image.Image:
        if self._image is None:
            image = self.decoder(self.latents)
            if not isinstance(image, Image.Image):
                raise RuntimeError("Latent materialization did not return a PIL image")
            if image.size != self.pixel_size:
                raise RuntimeError(
                    f"Latent materialization returned {image.size[0]}x{image.size[1]}, "
                    f"expected {self.pixel_size[0]}x{self.pixel_size[1]}"
                )
            self._image = image
        return self._image


def materialize_image(image):
    if isinstance(image, LatentImageArtifact):
        return image.materialize()
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected a PIL image or latent artifact, got {type(image).__name__}")
    return image


def make_latent_image_artifact(
    latents,
    decoded_pixel_size,
    pipeline,
    family,
    job_id,
    control,
    *,
    staged_decode,
    start_progress,
    final_progress,
    phase_prefix,
    output_size=None,
    output_resize_algorithm=None,
):
    factor = pipeline_vae_scale_factor(pipeline)
    decoded_pixel_size = (int(decoded_pixel_size[0]), int(decoded_pixel_size[1]))
    output_size = (
        decoded_pixel_size
        if output_size is None
        else (int(output_size[0]), int(output_size[1]))
    )
    if not isinstance(latents, torch.Tensor) or latents.ndim != 4 or latents.shape[:2] != (1, 4):
        shape = tuple(latents.shape) if isinstance(latents, torch.Tensor) else type(latents).__name__
        raise RuntimeError(f"Latent resampling produced an invalid output shape: {shape}")
    expected = (latents.shape[-1] * factor, latents.shape[-2] * factor)
    if expected != decoded_pixel_size:
        raise RuntimeError(
            f"Latent output grid {latents.shape[-1]}x{latents.shape[-2]} with VAE factor {factor} "
            f"does not match {decoded_pixel_size[0]}x{decoded_pixel_size[1]}"
        )
    if output_size != decoded_pixel_size and output_resize_algorithm is None:
        raise ValueError("A latent artifact with a non-grid output size requires a pixel resize algorithm")
    if not torch.isfinite(latents).all():
        raise RuntimeError("Latent resampling produced non-finite output")
    stored_latents = latents.to("cpu") if staged_decode else latents

    def decode(current):
        if staged_decode:
            images = decode_staged_latents(
                pipeline,
                current,
                family,
                job_id,
                control,
                active_vae_mode == "tiled",
                start_progress=start_progress,
                final_progress=final_progress,
                phase_prefix=phase_prefix,
            )
        else:
            images = decode_resident_latents(
                pipeline,
                current,
                family,
                job_id,
                control,
                start_progress=start_progress,
                final_progress=final_progress,
                phase_prefix=phase_prefix,
            )
        if len(images) != 1:
            raise RuntimeError(f"Latent materialization returned {len(images)} images instead of 1")
        result = images[0].convert("RGB")
        if result.size != decoded_pixel_size:
            raise RuntimeError(
                f"VAE decoded {result.size[0]}x{result.size[1]}, expected "
                f"{decoded_pixel_size[0]}x{decoded_pixel_size[1]}"
            )
        if output_size != decoded_pixel_size:
            control.checkpoint(job_id, f"{phase_prefix} latent output resize")
            update_job(
                job_id,
                phase=f"{phase_prefix} · Restoring exact output {output_size[0]} x {output_size[1]}",
                stage="resample_resize",
                stage_step=0,
                stage_total=1,
                progress=final_progress,
            )
            result = resize_image(
                result, output_size[0], output_size[1], output_resize_algorithm
            )
            update_job(job_id, stage_step=1, stage_total=1, progress=final_progress)
        return result

    return LatentImageArtifact(stored_latents, output_size, family, factor, decode)


def whole_image_decode_fits(latents, vae):
    """Whether this decode can run on the card in one piece, now that the sampler has been parked.

    ComfyUI asks the same question at the same moment (`comfy/sd.py::VAE.decode`: size the decode,
    evict the diffusion model, then decide) but can afford to ask it by trying: a decode that does
    not fit raises, and it retries tiled. That does not transfer to Windows, where the driver serves
    an oversized allocation out of host memory instead of failing — the picture comes back correct
    and a hundred times slower. So the size is compared before the decode rather than after it.

    Asked here rather than at load, because this is the first moment the free memory is the memory
    the decoder will actually have, and because a job's stages do not share a canvas: the reference
    run's base pass fits whole and its hires redraw does not.
    """
    if not torch.cuda.is_available():
        # Without a device there is no free memory to measure and nothing to spill into; whatever
        # the caller configured stands.
        return True
    free_bytes, physical_total_bytes, _limit_bytes = effective_cuda_free_memory()
    reserved_bytes = int(RESERVED_VRAM_GB * GIB) if RESERVED_VRAM_GB is not None else reserved_vram_bytes(
        physical_total_bytes,
        allow_shared_memory=performance_settings["allow_shared_memory"],
    )
    # How Diffusers derives the same number from the autoencoder it was handed, fallback included.
    scale = 2 ** (len(getattr(vae.config, "block_out_channels", (0,) * 4)) - 1)
    required_bytes = estimate_vae_decode_bytes(
        latents.shape[-1] * scale,
        latents.shape[-2] * scale,
        images_per_batch=latents.shape[0],
        sliced=bool(getattr(vae, "use_slicing", False)),
    )
    # The estimate is `max_memory_allocated`, which does not count what the allocator holds reserved
    # around it, so the comparison carries the margin rather than the measurement.
    return required_bytes * 1.10 <= max(0, free_bytes - reserved_bytes)


def decode_staged_latents(
    pipeline,
    latents,
    family,
    job_id,
    control,
    use_tiling,
    start_progress=94,
    final_progress=98,
    phase_prefix=None,
    decision_stage=None,
):
    global active_vae_mode
    latents = latents.to("cpu")
    control.checkpoint(job_id, "Unloading sampler")
    update_job(job_id, phase="Moving sampler to system memory", stage="sampler_offload", stage_step=0, stage_total=0, progress=start_progress)
    park_pipeline_for_vae(pipeline)
    control.checkpoint(job_id, "Preparing VAE")
    vae = pipeline.vae
    if not use_tiling and not whole_image_decode_fits(latents, vae):
        use_tiling = True
    if decision_stage:
        # The staged decode is the user's setting and always evicts first; what the schedule records
        # is that it did, the decode's own peak, and which of tiled/whole it ran — the one decision
        # here that decides pixels, and one AUTO never makes differently.
        scale = 2 ** (len(getattr(vae.config, "block_out_channels", (0,) * 4)) - 1)
        record_decode_decision(
            decision_stage, family, (int(latents.shape[-1]) * scale, int(latents.shape[-2]) * scale),
            evicted=True, job_id=job_id, reason="staged_decode_setting",
            peak_bytes=estimate_vae_decode_bytes(
                int(latents.shape[-1]) * scale, int(latents.shape[-2]) * scale,
                images_per_batch=int(latents.shape[0]), sliced=bool(getattr(vae, "use_slicing", False)),
            ),
            tiled=bool(use_tiling),
        )
    # The configured mode is a preference; what the report has to name is the decode that ran.
    active_vae_mode = "tiled" if use_tiling else active_vae_mode
    original_dtype = vae.dtype
    needs_upcasting = False
    decoded = None
    decoder_forward = None
    tiling_active = use_tiling and (
        latents.shape[-2] > vae.tile_latent_min_size or latents.shape[-1] > vae.tile_latent_min_size
    )
    tile_total = vae_decode_tile_count(
        latents.shape[-2], latents.shape[-1], vae.tile_latent_min_size, vae.tile_overlap_factor
    ) * latents.shape[0] if tiling_active else 1
    tile_progress = [0]

    def cancellable_decode(*args, **kwargs):
        control.checkpoint(job_id, "VAE decoding")
        result = decoder_forward(*args, **kwargs)
        tile_progress[0] += 1
        completed = min(tile_progress[0], tile_total)
        progress = start_progress + round((final_progress - start_progress) * completed / tile_total)
        update_job(job_id, stage_step=completed, stage_total=tile_total, progress=progress)
        control.checkpoint(job_id, "VAE decoding")
        return result

    try:
        if use_tiling:
            vae.enable_tiling()
        else:
            vae.disable_tiling()
        latents, needs_upcasting = prepare_vae_latents(pipeline, latents, family)
        vae.to("cuda")
        latents = latents.to("cuda")
        decoder_forward = vae.decoder.forward
        vae.decoder.forward = cancellable_decode
        control.checkpoint(job_id, "VAE decoding")
        update_job(
            job_id,
            phase=(
                f"{phase_prefix} · VAE tiled decode" if phase_prefix and tiling_active
                else f"{phase_prefix} · VAE decode" if phase_prefix
                else "VAE tiled decode" if tiling_active
                else "VAE decode"
            ),
            stage="vae_decode",
            stage_step=0,
            stage_total=tile_total,
            progress=start_progress,
            **pipeline_status_fields(),
        )
        with torch.inference_mode():
            decoded = vae.decode(latents, return_dict=False)[0]
        if not torch.isfinite(decoded).all():
            raise RuntimeError("VAE decode produced non-finite pixels")
        control.checkpoint(job_id, "VAE decoding")
        update_job(job_id, stage_step=tile_total, stage_total=tile_total, progress=final_progress)
        watermark = getattr(pipeline, "watermark", None)
        if family == "sdxl" and watermark is not None:
            decoded = watermark.apply_watermark(decoded)
        return pipeline.image_processor.postprocess(decoded, output_type="pil")
    finally:
        if decoder_forward is not None:
            vae.decoder.forward = decoder_forward
        del decoded, latents
        vae.to("cpu", dtype=original_dtype)
        gc.collect()
        torch.cuda.empty_cache()


def restore_cached_pipeline_state(job_id=None, request=None):
    if loaded_pipeline is None or active_memory_strategy is None:
        return True
    lora_state = getattr(loaded_pipeline, "_xiriai_lora_fusion_state", None)
    if isinstance(lora_state, dict) and lora_state.get("state") == "dirty":
        # A failed irreversible fold/patch may have changed only a prefix of the
        # requested groups.  No hook or parking operation can restore that base.
        return False
    try:
        if is_native_family(loaded_family):
            loaded_pipeline.to_cpu()
            return True
        # The published mode the pre-P8 scheduler would have left: a reconfigure here also resets the
        # autoencoder's tiling for the next job, so it has to happen exactly when it used to.
        published = active_memory_strategy.get("legacy_mode") or active_memory_strategy["mode"]
        if pipeline_cpu_parked:
            return True
        elif performance_settings["staged_vae_decode"]:
            park_pipeline_for_vae(loaded_pipeline)
        elif published == "normal_vram":
            loaded_pipeline.maybe_free_model_hooks()
        elif published in {"low_vram", "ultra_low_vram"}:
            reconfigure_memory_strategy(loaded_pipeline, active_memory_strategy)
        return True
    except Exception as error:
        traceback.print_exc()
        if job_id is not None and request is not None:
            write_generation_failure_log(job_id, request, error, "memory-recovery-failure")
        return False


def restore_parked_pipeline():
    global pipeline_cpu_parked
    if loaded_pipeline is None or active_memory_strategy is None or not pipeline_cpu_parked:
        return
    with pipeline_lock:
        if is_native_family(loaded_family):
            loaded_pipeline.to_cpu()
            pipeline_cpu_parked = False
            return
        reconfigure_memory_strategy(loaded_pipeline, active_memory_strategy)
        pipeline_cpu_parked = False


def release_pipeline_after_job():
    if performance_settings["keep_model_cached"] or loaded_pipeline is None:
        return
    with pipeline_lock:
        clear_pipeline()


def release_prompt_encoders(pipeline):
    from accelerate.hooks import remove_hook_from_module

    for name in ("text_encoder", "text_encoder_2"):
        component = getattr(pipeline, name, None)
        if component is not None:
            remove_hook_from_module(component, recurse=True)
            setattr(pipeline, name, None)
    pipeline.tokenizer = None
    pipeline.tokenizer_2 = None
    gc.collect()
    torch.cuda.empty_cache()


def pipeline_weight_sizes(pipeline):
    total = 0
    largest = 0
    globally_seen = set()
    for component in pipeline.components.values():
        if not isinstance(component, torch.nn.Module):
            continue
        size = 0
        for tensor in (*component.parameters(), *component.buffers()):
            pointer = tensor.data_ptr()
            identity = pointer if pointer else id(tensor)
            if identity in globally_seen:
                continue
            globally_seen.add(identity)
            tensor_bytes = tensor.numel() * tensor.element_size()
            size += tensor_bytes
            total += tensor_bytes
        largest = max(largest, size)
    return total, largest


def is_oom_error(error):
    if isinstance(error, (torch.OutOfMemoryError, StageMemoryError)):
        return True
    accelerator_error = getattr(torch, "AcceleratorError", None)
    return bool(accelerator_error and isinstance(error, accelerator_error) and error_looks_like_oom(error)) or error_looks_like_oom(error)


def pipeline_status_fields():
    if active_memory_strategy is None:
        return {
            "memory_mode": None,
            "memory_label": None,
        "memory_reason": None,
        "offload_mode": None,
        "attention_backend": "none",
            "compute_dtype": "none",
            "vae_mode": "none",
            "model_resident": False,
            "model_cached": False,
            "loaded_checkpoint": None,
            "loaded_checkpoint_path": None,
            "loaded_engine": None,
            "loaded_model_assets": {},
        }
    return {
        **memory_job_fields(active_memory_strategy),
        **residency_job_fields(),
        "attention_backend": current_attention_backend(),
        "compute_dtype": active_compute_dtype,
        "vae_mode": active_vae_mode,
        "loaded_checkpoint": Path(loaded_checkpoint).name,
        "loaded_checkpoint_path": loaded_checkpoint_reference(),
        "loaded_engine": loaded_engine,
        "loaded_model_assets": loaded_model_asset_references(),
    }


def loaded_checkpoint_reference():
    if not loaded_checkpoint or not loaded_engine:
        return None
    checkpoint_root = (
        native_model_roots(loaded_engine)["diffusion_model"]
        if loaded_engine in NATIVE_ENGINES
        else model_roots(loaded_engine)[0]
    )
    try:
        return Path(loaded_checkpoint).relative_to(checkpoint_root).as_posix()
    except ValueError:
        # A checkpoint loaded from a shared folder is outside every project
        # root by definition. Reporting only its filename would stop the UI
        # matching it against the selection the user actually made.
        return shared_reference_for_path(loaded_checkpoint) or Path(loaded_checkpoint).name


def loaded_model_asset_references():
    if loaded_engine not in NATIVE_ENGINES or not loaded_model_assets:
        return {}
    roots = native_model_roots(loaded_engine)
    references = {}
    for name in ("diffusion_model", "text_encoder", "text_encoder_2", "vae"):
        value = loaded_model_assets.get(name)
        if not value:
            continue
        # Flux's second text encoder lives in the same directory as the first; the field name is
        # a slot in the request, not a separate model root.
        root = roots["text_encoder"] if name == "text_encoder_2" else roots[name]
        try:
            references[name] = Path(value).relative_to(root).as_posix()
        except ValueError:
            references[name] = Path(value).name
    return references


def announce_residency_transition(entry):
    """Put a move on the job record and the console as it happens (plan P1-7)."""
    if residency_job_id is not None:
        update_job(residency_job_id, residency_event=entry)


def build_residency_controller(pipeline, strategy):
    """Hand the loaded pipeline to the controller, told what is already installed on it.

    Admission (Phase 0) still chooses where a job starts; the controller is what re-asks the question
    at each stage boundary, against the free memory of that moment rather than the whole chain's
    worst case.
    """
    fixed_wall = performance_settings.get("vram_limit_gb", 0.0) > 0
    return ResidencyController(
        pipeline,
        reserved_bytes=int(strategy.get("reserved_bytes", 0)),
        free_memory=lambda: effective_cuda_free_memory()[0],
        tier=tier_for_memory_mode(strategy["mode"]),
        # A configured VRAM wall is the ceiling the allocator cap may never exceed; without one the
        # ceiling is the device, and the cap is what keeps the allocator inside what is free.
        allocator_ceiling_bytes=int(strategy.get("vram_limit_bytes", 0)) if fixed_wall else None,
        on_transition=announce_residency_transition,
        dynamic=stage_scheduling_active(strategy),
        budget_reader=read_stage_budget,
        # The band is the budget scope's: a 24 GB card behind an 8 GB wall is an 8 GB card here.
        safety_margin_bytes=allocator_safety_margin_bytes(int(strategy.get("vram_limit_bytes", 0) or 0)),
    )


def pipeline_lora_cache_compatible(pipeline, signature):
    if pipeline is None:
        return False
    state = getattr(pipeline, "_xiriai_lora_fusion_state", None)
    if isinstance(state, dict):
        return state.get("state") == "clean" and state.get("signature") == tuple(signature or ())
    # Compatibility with a pipeline created earlier in this process before the
    # structured state existed.  An absent legacy marker is a clean base only
    # for an empty request; it can never authorize reuse for a requested set.
    fused = getattr(pipeline, "_xiriai_fused_loras", None)
    return (not signature and fused is None) or fused == tuple(signature or ())


def load_pipeline(checkpoint: Path, family: str, engine: str, width: int, height: int, guidance_scale: float, lora_bytes: int, job_id: str, images_per_batch=1, guidance="none", loras=()):
    global loaded_pipeline, loaded_checkpoint, loaded_checkpoint_hash, loaded_family, loaded_engine
    global pipeline_cpu_parked, active_memory_strategy, active_attention_backend, active_compute_dtype
    global active_residency, residency_job_id
    residency_job_id = job_id
    with pipeline_lock:
        guidance_copies = guidance_prediction_copies(guidance, guidance_scale)
        # SD/iL is the family that mounts one large UNet beside two small towers, so it is the one
        # whose NORMAL admission should ask whether the component fits rather than whether the whole
        # workspace does, and the one whose CFG branches may be taken one at a time.
        resident_first = family in SD_COMPONENT_FAMILIES
        cfg_serial_eligible = serial_cfg_eligible(family, guidance)
        if active_residency is not None:
            # One job, one trace. The controller outlives a job because the pipeline does, but the
            # record published on a job has to describe that job rather than every job since load.
            active_residency.begin_job()
        # A fused adapter set cannot be taken off again by `unload_lora_weights`, so a job that
        # wants different adapters cannot reuse this pipeline — it has to be built from the
        # checkpoint again. Same set, and both the load and the fuse are skipped entirely.
        adapters_compatible = pipeline_lora_cache_compatible(loaded_pipeline, tuple(loras or ()))
        if (loaded_pipeline is not None and loaded_checkpoint == str(checkpoint)
                and loaded_family == family and loaded_engine == engine and adapters_compatible):
            strategy = choose_memory_strategy(
                checkpoint,
                family,
                width,
                height,
                guidance_scale,
                lora_bytes,
                images_per_batch,
                guidance_copies,
                include_loaded_model=True,
                base_weight_bytes=active_memory_strategy["base_weight_bytes"],
                base_largest_component_bytes=active_memory_strategy["base_largest_component_bytes"],
                resident_first=resident_first,
                cfg_serial_eligible=cfg_serial_eligible,
            )
            # The guidance execution order is part of what is installed on the UNet, so a job that
            # is admitted for a different one has to reinstall it even when the rung has not moved.
            # The comparison is with the *legacy* published mode: a reconfigure also resets the
            # autoencoder's tiling, and under AUTO the model's real rung is not what the pre-P8
            # scheduler would have left published — comparing with it would reconfigure (and so
            # re-tile) in different jobs than before (QPlan P8 pixel identity).
            if (
                pipeline_cpu_parked
                or strategy["mode"] != (active_memory_strategy.get("legacy_mode") or active_memory_strategy["mode"])
                or strategy.get("cfg_mode") != active_memory_strategy.get("cfg_mode")
            ):
                update_job(job_id, phase=f"Adjusting VRAM policy · {strategy['label']}", progress=8)
                previous_strategy = active_memory_strategy
                try:
                    reconfigure_memory_strategy(loaded_pipeline, strategy)
                    pipeline_cpu_parked = False
                except Exception:
                    try:
                        reconfigure_memory_strategy(loaded_pipeline, previous_strategy)
                    except Exception:
                        clear_pipeline()
                    raise
            # Idempotent, and the one thing a reused pipeline cannot inherit from the last job: a
            # PAG run behind a plain-CFG one has the same rung and the same `cfg_mode`, and would
            # otherwise reach its attention processor with a batch of one it cannot divide by three.
            apply_cfg_execution_mode(loaded_pipeline, strategy)
            active_memory_strategy = strategy
            if active_residency is None:
                active_residency = build_residency_controller(loaded_pipeline, strategy)
            else:
                active_residency.dynamic = stage_scheduling_active(strategy)
                # The rung a restore reinstalls is the controller's, not the admission's — the last
                # job may have left it streaming — so the strategy is told what is actually on the
                # pipeline. The base stage plan re-checks it against this job's canvas before any
                # step runs, which is why starting where the last job left off is safe and cheap.
                publish_residency()
            record_mount_decision(job_id, family, engine, width, height, reused=True)
            update_job(
                job_id,
                phase="Reusing loaded model",
                progress=10,
                attention_backend=active_attention_backend,
                compute_dtype=active_compute_dtype,
                vae_mode=active_vae_mode,
                loaded_checkpoint=checkpoint.name,
                loaded_checkpoint_path=loaded_checkpoint_reference(),
                loaded_engine=engine,
                model_reused=True,
                **memory_job_fields(active_memory_strategy),
            )
            return loaded_pipeline

        clear_pipeline()
        update_job(job_id, phase="Loading checkpoint", progress=4)
        strategy = choose_memory_strategy(
            checkpoint, family, width, height, guidance_scale, lora_bytes, images_per_batch, guidance_copies,
            resident_first=resident_first,
            cfg_serial_eligible=cfg_serial_eligible,
        )
        update_job(
            job_id,
            phase=f"Loading checkpoint · {strategy['label']}",
            progress=4,
            **memory_job_fields(strategy, model_cached=False),
        )
        pipeline_class = StableDiffusionXLPipeline if family == "sdxl" else StableDiffusionPipeline
        config_path = require_pipeline_config(family, os.environ["HF_HUB_CACHE"])
        pipeline_dtype, active_compute_dtype = resolve_compute_dtype()
        kwargs = {
            "torch_dtype": pipeline_dtype,
            "use_safetensors": checkpoint.suffix.lower() == ".safetensors",
            "config": str(config_path),
            "local_files_only": True,
        }
        if family == "sd":
            kwargs.update({"safety_checker": None, "requires_safety_checker": False})

        # The console's own "Model ready" line reports this load; Diffusers' bar only duplicated it.
        with quiet_pipeline_loading():
            pipeline = pipeline_class.from_single_file(str(checkpoint), **kwargs)
        pipeline.set_progress_bar_config(disable=True)
        # Long prompts are chunked into 77-token CLIP blocks (`prompt_encoding`), so the tokenizers'
        # "will result in indexing errors" warning is false for this pipeline.
        mark_long_prompts_handled(getattr(pipeline, "tokenizer", None), getattr(pipeline, "tokenizer_2", None))
        if performance_settings["calculate_model_hash"]:
            checkpoint_stat = checkpoint.stat()
            update_job(job_id, phase="Calculating model identity", progress=15)
            loaded_checkpoint_hash = checkpoint_sha256(str(checkpoint), checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns)
        else:
            loaded_checkpoint_hash = None
        actual_weight_bytes, actual_largest_component = pipeline_weight_sizes(pipeline)
        strategy = choose_memory_strategy(
            checkpoint,
            family,
            width,
            height,
            guidance_scale,
            lora_bytes,
            images_per_batch,
            guidance_copies,
            base_weight_bytes=actual_weight_bytes or strategy["base_weight_bytes"],
            base_largest_component_bytes=actual_largest_component or strategy["base_largest_component_bytes"],
            resident_first=resident_first,
            cfg_serial_eligible=cfg_serial_eligible,
        )
        configure_cuda_math()
        configure_attention_backend(pipeline)
        apply_memory_strategy(pipeline, strategy)
        active_residency = build_residency_controller(pipeline, strategy)
        loaded_pipeline = pipeline
        loaded_checkpoint = str(checkpoint)
        loaded_family = family
        loaded_engine = engine
        active_memory_strategy = strategy
        record_mount_decision(job_id, family, engine, width, height, reused=False)
        update_job(
            job_id,
            phase=f"Model ready · {strategy['label']}",
            progress=18,
            attention_backend=active_attention_backend,
            compute_dtype=active_compute_dtype,
            vae_mode=active_vae_mode,
            loaded_checkpoint=checkpoint.name,
            loaded_checkpoint_path=loaded_checkpoint_reference(),
            loaded_engine=engine,
            model_reused=False,
            **memory_job_fields(strategy),
        )
        return pipeline


def anima_weight_estimates(assets):
    estimates = []
    for path in assets.values():
        stat = path.stat()
        estimates.append(estimate_checkpoint_weight_bytes(str(path), stat.st_size, stat.st_mtime_ns))
    return sum(estimates), max(estimates)


# What a text-encoding pass allocates beyond the weights themselves, when the DiT is being held on
# the card across jobs. Anima conditions a few hundred tokens through a 0.6B Qwen3 tower, so this is
# headroom rather than a measurement of a large tensor — and it is deliberately not the component
# engines' `RESIDENT_HOLD_ENCODE_HEADROOM_BYTES`, which sizes the same pass through a 4B decoder.
ANIMA_RESIDENT_HOLD_ENCODE_HEADROOM_BYTES = 256 * 1024**2


def choose_anima_memory_strategy(
    diffusion_model,
    width,
    height,
    guidance_scale,
    images_per_batch,
    guidance,
    total_weight_bytes,
    largest_component_bytes,
    include_loaded=False,
    lora_bytes=0,
    transformer_group_bytes=512 * 1024**2,
    force_group_offload=False,
    pag_scale=0.3,
):
    logical_guidance_copies = (
        (2 if guidance_scale > 1.0 else 1)
        if guidance == "pag" and float(pag_scale) == 0.0
        else guidance_prediction_copies(guidance, guidance_scale)
    )
    # Anima executes CFG and PAG branches sequentially; batch CFG is separately
    # admitted below and must not inflate normal resident admission.
    physical_forward_copies = 1

    def choose(batch_size, forward_copies=physical_forward_copies):
        return choose_memory_strategy(
            diffusion_model,
            "anima",
            width,
            height,
            guidance_scale,
            0,
            batch_size,
            forward_copies,
            include_loaded_model=include_loaded,
            base_weight_bytes=total_weight_bytes,
            base_largest_component_bytes=largest_component_bytes,
        )

    serial_strategy = choose(images_per_batch)
    strategy = serial_strategy
    group_offload = force_group_offload or strategy["mode"] == "low_vram"
    microbatch = group_offload and images_per_batch > 1
    if microbatch:
        strategy = choose(1)
        group_offload = True
    group_required_bytes = (
        int(transformer_group_bytes * 1.10)
        + strategy.get("inference_bytes", 0)
        + strategy.get("reserved_bytes", 0)
    )
    if group_offload and strategy.get("free_bytes", group_required_bytes) < group_required_bytes:
        raise RuntimeError(
            "当前 Anima 画布连单个 Transformer 分组和推理张量也无法安全装入显存；"
            "请减小画布或批量，或释放其他 GPU 程序后重试"
        )
    if group_offload:
        strategy.update(mode="low_vram", label=strategy_label(strategy, "low_vram", NATIVE_BLOCK))
    group_cfg_requested = anima_group_cfg_batch_requested()
    resident_cfg_env_requested = os.environ.get("XIRAI_ANIMA_RESIDENT_CFG_BATCH", "").strip() == "1"
    resident_cfg_requested = anima_resident_cfg_batch_requested()
    resident_cfg_force_requested = anima_resident_cfg_batch_force_requested()
    resident_cfg_force_effective = anima_resident_cfg_batch_force_effective()
    resident_cfg_lease = benchmark_lease_validation() if resident_cfg_force_requested else {
        "valid": False, "reason": "not_requested", "expiry": None, "purpose": None,
    }
    resident_cfg_probe = bool(
        resident_cfg_requested and not group_offload and guidance != "pag"
        and guidance_scale > 1.0 and torch.cuda.is_available()
    )
    # Always calculate the strict two-forward product for diagnostics. It is
    # not an enablement path; normal production remains serial by default.
    cfg_batch_candidate = bool(guidance != "pag" and guidance_scale > 1.0)
    # Re-run generic policy with the actual physical two-forward peak. This is
    # both the batch candidate's inference estimate and its independent mode
    # admission; the serial strategy remains preserved for diagnostics.
    cfg_batch_strategy = choose(images_per_batch, 2) if cfg_batch_candidate else None
    cfg_batch_inference_bytes = int(
        cfg_batch_strategy.get("inference_bytes", 0)
        if cfg_batch_strategy is not None
        else estimate_inference_bytes("anima", width, height, guidance_scale, images_per_batch, guidance_copies=2)
    )
    # Full two-forward workspace in addition to the normal resident peak. This
    # protects the serial 8 GB trajectory; batch CFG is never the admission fix.
    cfg_batch_required_bytes = max(
        int(serial_strategy.get("normal_required_bytes", 0)) + cfg_batch_inference_bytes,
        int(largest_component_bytes * 1.10) + cfg_batch_inference_bytes + int(strategy.get("reserved_bytes", 0)),
        int(total_weight_bytes * 1.08) + cfg_batch_inference_bytes + int(strategy.get("reserved_bytes", 0)),
    )
    resident_cfg_delta_bytes = max(0, cfg_batch_inference_bytes - int(serial_strategy.get("inference_bytes", 0)))
    resident_cfg_margin_bytes = max(64 * 1024**2, int(resident_cfg_delta_bytes * 0.10))
    resident_cfg_required_bytes = int(serial_strategy.get("normal_required_bytes", 0)) + resident_cfg_delta_bytes + resident_cfg_margin_bytes
    # This is intentionally independent of the group gate.  The old strict
    # product admission is retained as a diagnostic: it rejects the normal
    # resident path because it budgets an entire second resident product.
    resident_cfg_strict_product_admitted = bool(
        cfg_batch_strategy is not None and serial_strategy.get("free_bytes", 0) >= cfg_batch_required_bytes
    )
    resident_cfg_experimental_probe = bool(
        resident_cfg_probe and cfg_batch_strategy is not None
        and cfg_batch_strategy.get("mode") in {"normal_vram", "high_vram"}
        and serial_strategy.get("free_bytes", 0) >= resident_cfg_required_bytes
    )
    group_cfg_required_bytes = (
        int(transformer_group_bytes * 1.10)
        + estimate_inference_bytes("anima", width, height, guidance_scale, strategy.get("sampling_batch_size", images_per_batch), guidance_copies=2)
        + int(strategy.get("reserved_bytes", 0))
        # Explicit fragmentation allowance for group swaps, independent of the
        # resident-model estimate.  Do not count unmatched transformer bytes twice:
        # transformer_group_bytes already contains max-block + unmatched bytes.
        + max(64 * 1024**2, int(transformer_group_bytes * 0.10))
    )
    group_cfg_admitted = bool(
        group_offload
        and group_cfg_requested
        and guidance != "pag"
        and guidance_scale > 1.0
        and strategy.get("free_bytes", 0) >= group_cfg_required_bytes
    )
    cfg_batch = group_cfg_admitted if group_offload else bool(
        resident_cfg_experimental_probe or (resident_cfg_force_effective and resident_cfg_probe)
    )
    if cfg_batch:
        strategy = cfg_batch_strategy
    actual_physical_forward_copies = 2 if cfg_batch else 1
    actual_inference_bytes = int(
        estimate_inference_bytes("anima", width, height, guidance_scale, strategy.get("sampling_batch_size", images_per_batch), guidance_copies=2)
        if (cfg_batch and group_offload) else strategy.get("inference_bytes", 0)
    )
    # Holding the DiT between jobs is admitted on its own terms rather than by requiring HIGH_VRAM,
    # which is what `choose_flux_memory_strategy` already does for the component engines. HIGH is a
    # strictly stronger condition — it budgets the sampling tensors alongside every weight — and on
    # a card where the DiT is half the device it is unreachable at any canvas, so Anima parked a
    # transformer that fits perfectly well and bought a PCIe round trip per job for nothing.
    #
    # What the hold has to survive is whatever shares the card with the held DiT: this job's decode
    # (DiT + VAE + the decode's own tiles) and the next job's text encoding (DiT + encoder + its
    # activations), never both at once. Only `total_weight_bytes` and the largest component are
    # known here, so the bound counts the encoder and the VAE together — over-counting one of them
    # is what pays for the activations neither figure includes. The component engines' 1 GiB
    # headroom is not reused: it sizes conditioning through a 4B decoder, where Anima encodes
    # through a 0.6B tower, and on an 8 GB card that difference decides the whole question.
    #
    # A decode that cannot share the card with a held DiT is not a failed job — the runtime parks
    # it, sets `_co_residency_failed`, retries, and never offers co-residency again for that
    # runtime — so being wrong here costs one retry, once.
    resident_hold_bytes = (
        int(total_weight_bytes * 1.02)
        + int(strategy.get("reserved_bytes", 0))
        + ANIMA_RESIDENT_HOLD_ENCODE_HEADROOM_BYTES
    )
    resident_hold_admitted = strategy.get("free_bytes", 0) >= resident_hold_bytes
    keep_transformer_resident = bool(
        performance_settings["keep_model_cached"]
        and not group_offload
        and (
            strategy.get("mode") == "high_vram"
            or (strategy.get("mode") == "normal_vram" and resident_hold_admitted)
        )
    )
    strategy.update(
        offload_mode=(
            "staged_transformer_group_offload_microbatch"
            if group_offload and microbatch
            else "staged_transformer_group_offload"
            if group_offload
            else "staged_transformer_resident"
        ),
        sampling_batch_size=1 if microbatch else images_per_batch,
        model_resident=False,
        transformer_group_offload=group_offload,
        transformer_blocks_per_group=1 if group_offload else 0,
        cfg_batch=cfg_batch,
        cfg_batch_required_bytes=cfg_batch_required_bytes,
        logical_guidance_copies=logical_guidance_copies,
        physical_forward_copies=actual_physical_forward_copies,
        actual_physical_forward_copies=actual_physical_forward_copies,
        logical_inference_bytes=estimate_inference_bytes(
            "anima", width, height, guidance_scale, images_per_batch, guidance_copies=logical_guidance_copies,
        ),
        physical_inference_bytes=actual_inference_bytes,
        admission={
            "ambient_free_bytes": None,
            "allocator_free_bytes": int(strategy.get("free_bytes", 0)),
            "sampling_stage_required_bytes": int(cfg_batch_required_bytes if cfg_batch else strategy.get("normal_required_bytes", 0)),
            "serial_resident_sampling_required_bytes": int(serial_strategy.get("normal_required_bytes", 0)),
            "cfg_batch_sampling_required_bytes": int(cfg_batch_required_bytes),
            "group_required_bytes": int(group_required_bytes),
            "group_sequential_required_bytes": int(group_required_bytes),
            "group_cfg_required_bytes": int(group_cfg_required_bytes),
            "group_cfg_fragmentation_margin_bytes": max(64 * 1024**2, int(transformer_group_bytes * 0.10)),
            "group_cfg_feature_requested": group_cfg_requested,
            "group_cfg_feature_admitted": group_cfg_admitted,
            "group_cfg_feature_actual": cfg_batch,
            "group_cfg_feature_fallback": None,
            "resident_cfg_batch_requested": resident_cfg_env_requested,
            "resident_cfg_batch_probe": resident_cfg_probe,
            "resident_cfg_batch_probe_eligible": resident_cfg_experimental_probe,
            "resident_cfg_batch_admitted": False,
            "resident_cfg_batch_admission_kind": (
                "forced_experimental_speculative_no_safety_guarantee"
                if resident_cfg_force_effective else "experimental_estimate_not_safety_guarantee"
            ),
            "resident_cfg_batch_force_requested": resident_cfg_force_requested,
            "resident_cfg_batch_force_effective": resident_cfg_force_effective,
            "resident_cfg_batch_lease": resident_cfg_lease,
            "resident_cfg_batch_actual": False,
            "resident_cfg_batch_fallback": None,
            "resident_cfg_batch_gate_reason": (
                "forced_experimental_speculative_probe"
                if resident_cfg_force_effective else
                "experimental_speculative_probe"
                if resident_cfg_experimental_probe else
                "strict_product_default_rejects_second_resident_peak"
                if not resident_cfg_env_requested else
                "probe_ineligible_or_insufficient_stage_isolated_headroom"
            ),
            "resident_cfg_batch_allocator_free_bytes": int(serial_strategy.get("free_bytes", 0)),
            "resident_cfg_batch_serial_required_bytes": int(serial_strategy.get("normal_required_bytes", 0)),
            "resident_cfg_batch_estimated_delta_bytes": resident_cfg_delta_bytes,
            "resident_cfg_batch_fragmentation_margin_bytes": resident_cfg_margin_bytes,
            "resident_cfg_batch_strict_product_admitted": resident_cfg_strict_product_admitted,
            "logical_inference_bytes": estimate_inference_bytes(
                "anima", width, height, guidance_scale, images_per_batch, guidance_copies=logical_guidance_copies,
            ),
            "serial_resident_inference_bytes": int(serial_strategy.get("inference_bytes", 0)),
            "cfg_batch_inference_bytes": int(cfg_batch_inference_bytes),
            "physical_inference_bytes": actual_inference_bytes,
            "logical_guidance_copies": logical_guidance_copies,
            "serial_resident_physical_forward_copies": 1,
            "cfg_batch_physical_forward_copies": 2,
            "physical_forward_copies": actual_physical_forward_copies,
            "actual_physical_forward_copies": actual_physical_forward_copies,
            "decision_reason": "resident_physical_peak_admitted" if not group_offload else "group_offload_required_by_resident_peak",
            "requested_offload_mode": "staged_transformer_group_offload" if group_offload else "staged_transformer_resident",
            "actual_offload_mode": "staged_transformer_group_offload" if group_offload else "staged_transformer_resident",
            "fallback": None,
        },
        keep_transformer_resident=keep_transformer_resident,
        adapter_source_bytes=max(0, int(lora_bytes)),
        reason=(
            "Anima 原生运行时按文本编码器、扩散模型、VAE 分阶段调度"
            + ("，Cosmos Transformer 按单块动态装入 GPU" if group_offload else "")
            + ("，采样期间 Cosmos Transformer 完整常驻 GPU" if not group_offload else "")
            + "；"
            f"显存预算模式为 {strategy['mode'].upper()}，聚合权重约 {strategy['weight_gb']:.1f} GB"
            + ("；批量按单图串行采样" if microbatch else "")
            + ("；任务间保留 Transformer GPU 驻留" if keep_transformer_resident else "")
            + ("；完整双前向预算通过，正负条件合并为单次批量前向" if cfg_batch else "；默认保持串行 CFG")
        ),
    )
    # The mount-stage path, in AUTO's vocabulary; `cfg_batch_strategy` may have replaced the dict
    # above, so it is set on the one actually returned.
    strategy["memory_path"] = NATIVE_BLOCK if group_offload else NATIVE_RESIDENT
    if strategy.get("dynamic"):
        strategy["label"] = strategy_label(strategy)
    return strategy


def anima_memory_strategy(
    diffusion_model,
    assets,
    width,
    height,
    guidance_scale,
    images_per_batch,
    guidance,
    include_loaded=False,
    lora_bytes=0,
    pag_scale=0.3,
):
    total_weight_bytes, largest_component_bytes = anima_weight_estimates(assets)
    return choose_anima_memory_strategy(
        diffusion_model,
        width,
        height,
        guidance_scale,
        images_per_batch,
        guidance,
        total_weight_bytes,
        largest_component_bytes,
        include_loaded,
        lora_bytes,
        pag_scale=pag_scale,
    )


@lru_cache(maxsize=16)
def flux_component_weight_bytes(component_path: str, file_size: int, modified_ns: int, family: str = "flux"):
    """Loaded size of one component, which is not its size on disk.

    The Flux engines expand every quantisation to BF16, so an fp8 file costs twice what it
    occupies; budgeting from the file size would under-admit one by half and let it OOM mid-load.
    Krea 2 no longer expands anything — :mod:`quantized_linear` applies the scales inside the
    linear — so its fp8 components cost what they occupy. Budgeting the expansion that stopped
    happening overstated its transformer by 11.6 GiB, which was enough on its own to send a model
    that fits comfortably down the block-streaming path.
    """
    del modified_ns
    try:
        if family == "krea2":
            return krea2_component_bytes([component_path])
        return flux_component_bytes([component_path])
    except Exception:
        return file_size


def flux_weight_estimates(assets, family="flux"):
    estimates = []
    for path in assets.values():
        stat = path.stat()
        estimates.append(flux_component_weight_bytes(str(path), stat.st_size, stat.st_mtime_ns, family))
    return sum(estimates), max(estimates)


# What a text-encoding pass allocates beyond the weights themselves. Conditioning is a few hundred
# tokens through a 4B decoder, so this is headroom rather than a measurement of a large tensor.
RESIDENT_HOLD_ENCODE_HEADROOM_BYTES = 1024**3


def choose_flux_memory_strategy(
    diffusion_model,
    width,
    height,
    guidance_scale,
    images_per_batch,
    total_weight_bytes,
    largest_component_bytes,
    include_loaded=False,
    lora_bytes=0,
    transformer_group_bytes=512 * 1024**2,
    force_group_offload=False,
    family="flux",
    guidance_copies=1,
):
    """Admission for a component-mounted native engine: staged components, one resident stage.

    Much simpler than Anima's because the branches are never batched. The only real decision is
    whether the transformer fits resident or has to stream a block at a time, and the floor for the
    streaming path is one block plus the sampling tensors.

    All three component engines are admitted through this one path: they differ in how wide the DiT
    is, in what the text encoder costs, and — for Krea 2 — in whether an unconditional branch runs
    at all. Each of those reaches the policy as a number rather than as a second copy of the
    decision: measured bytes for the first two, ``guidance_copies`` for the third.
    """
    engine_label = {"flux2": "Flux2", "krea2": "Krea2"}.get(family, "Flux")
    guidance_copies = max(1, int(guidance_copies))

    def choose(batch_size):
        return choose_memory_strategy(
            diffusion_model,
            family,
            width,
            height,
            guidance_scale,
            0,
            batch_size,
            # One *physical* forward, always. Krea 2's two guidance branches run one after the
            # other, so its peak is a single branch plus the prediction the first one left behind
            # — which the family's own per-megapixel base already carries. Budgeting two here
            # would be the same over-count Anima's sequential path deliberately avoids.
            1,
            include_loaded_model=include_loaded,
            base_weight_bytes=total_weight_bytes,
            base_largest_component_bytes=largest_component_bytes,
        )

    serial_strategy = choose(images_per_batch)
    strategy = serial_strategy
    group_offload = force_group_offload or strategy["mode"] in {"low_vram", "ultra_low_vram"}
    microbatch = group_offload and images_per_batch > 1
    if microbatch:
        strategy = choose(1)
        group_offload = True
    group_required_bytes = (
        int(transformer_group_bytes * 1.10)
        + strategy.get("inference_bytes", 0)
        + strategy.get("reserved_bytes", 0)
    )
    if group_offload and strategy.get("free_bytes", group_required_bytes) < group_required_bytes:
        raise RuntimeError(
            f"当前 {engine_label} 画布连单个 Transformer 分组和推理张量也无法安全装入显存；"
            "请减小画布或批量，或释放其他 GPU 程序后重试"
        )
    if group_offload:
        strategy.update(mode="low_vram", label=strategy_label(strategy, "low_vram", NATIVE_BLOCK))
    # Holding the transformer between jobs is admitted on its own terms rather than by requiring
    # HIGH_VRAM. What the hold has to survive is the *next* job's conditioning, which runs with the
    # transformer still on the card: the peak is every weight plus the encoder's own activations,
    # and not the sampling tensors, because the encoder is parked again before sampling starts.
    # HIGH_VRAM is a strictly stronger condition — it also budgets the sampling tensors alongside
    # every weight — so requiring it sent the transformer back to host memory between jobs in cases
    # that fit perfectly well, which is a PCIe round trip per job bought for nothing.
    resident_hold_bytes = (
        int(total_weight_bytes * 1.02)
        + int(strategy.get("reserved_bytes", 0))
        + RESIDENT_HOLD_ENCODE_HEADROOM_BYTES
    )
    resident_hold_admitted = strategy.get("free_bytes", 0) >= resident_hold_bytes
    keep_transformer_resident = bool(
        not group_offload
        and performance_settings["keep_model_cached"]
        and (strategy["mode"] == "high_vram" or (strategy["mode"] == "normal_vram" and resident_hold_admitted))
    )
    offload_mode = (
        "staged_transformer_group_offload" if group_offload else "staged_transformer_resident"
    )
    strategy.update(
        offload_mode=offload_mode,
        sampling_batch_size=1 if microbatch else images_per_batch,
        model_resident=False,
        transformer_group_offload=group_offload,
        transformer_blocks_per_group=1 if group_offload else 0,
        keep_transformer_resident=keep_transformer_resident,
        cfg_batch=False,
        logical_guidance_copies=guidance_copies,
        # The conditioned and unconditional forwards are executed one after the other, so the
        # activation peak is one branch however many the guidance mode asks for.
        physical_forward_copies=1,
        actual_physical_forward_copies=1,
        adapter_source_bytes=max(0, int(lora_bytes)),
        admission={
            "allocator_free_bytes": int(strategy.get("free_bytes", 0)),
            "sampling_stage_required_bytes": int(strategy.get("normal_required_bytes", 0)),
            "group_required_bytes": int(group_required_bytes),
            "logical_guidance_copies": guidance_copies,
            "physical_forward_copies": 1,
            "actual_physical_forward_copies": 1,
            "physical_inference_bytes": int(strategy.get("inference_bytes", 0)),
            "decision_reason": (
                "group_offload_required_by_resident_peak" if group_offload
                else "resident_physical_peak_admitted"
            ),
            "requested_offload_mode": offload_mode,
            "actual_offload_mode": offload_mode,
            "fallback": None,
        },
        reason=(
            f"{engine_label} 原生运行时按文本编码器、扩散模型、VAE 分阶段调度"
            + ("，DiT 按单块动态装入 GPU" if group_offload else "，采样期间 DiT 完整常驻 GPU")
            + "；"
            f"显存预算模式为 {strategy['mode'].upper()}，聚合权重约 {strategy['weight_gb']:.1f} GB"
            + ("；批量按单图串行采样" if microbatch else "")
            + ("；任务间保留 DiT GPU 驻留" if keep_transformer_resident else "")
            + ("；正负分支串行执行，单次前向峰值" if guidance_copies > 1 else "；蒸馏引导，单次前向无负向分支")
        ),
    )
    strategy["memory_path"] = NATIVE_BLOCK if group_offload else NATIVE_RESIDENT
    if strategy.get("dynamic"):
        strategy["label"] = strategy_label(strategy)
    return strategy


def flux_memory_strategy(
    diffusion_model,
    assets,
    width,
    height,
    guidance_scale,
    images_per_batch,
    include_loaded=False,
    lora_bytes=0,
    family="flux",
    guidance_copies=1,
):
    total_weight_bytes, largest_component_bytes = flux_weight_estimates(assets, family)
    return choose_flux_memory_strategy(
        diffusion_model,
        width,
        height,
        guidance_scale,
        images_per_batch,
        total_weight_bytes,
        largest_component_bytes,
        include_loaded,
        lora_bytes,
        family=family,
        guidance_copies=guidance_copies,
    )


def configure_decode_residency(runtime):
    """Tell a native runtime whether to evict the transformer before the autoencoder runs.

    The decoder is the one stage that wants the card to itself, and by the time it runs the
    transformer has nothing left to do. Runtimes that do not offer the hook simply keep their own
    behaviour, which is what makes this safe to call for every engine.
    """
    configure = getattr(runtime, "configure_decode_residency", None)
    if configure is None:
        return None
    return configure(performance_settings["unload_before_decode"])


def configure_flux_vae(runtime, strategy):
    requested = performance_settings["vae_mode"]
    forced = requested == "tiled" or (
        requested == "auto" and strategy["mode"] in {"low_vram", "ultra_low_vram"}
    )
    runtime.configure_vae_tiling(forced)
    return "tiled" if forced else "full"


def configure_anima_vae(runtime, strategy):
    requested = performance_settings["vae_mode"]
    tiled = (
        requested == "tiled"
        or bool(getattr(runtime, "_vae_tiling_required", False))
        or (requested == "auto" and strategy["mode"] in {"low_vram", "ultra_low_vram"})
    )
    if tiled:
        runtime.vae.enable_tiling()
    else:
        runtime.vae.disable_tiling()
    return "tiled" if tiled else "full"


def load_anima_pipeline(
    diffusion_model,
    text_encoder,
    vae,
    tokenizer_sources,
    width,
    height,
    guidance_scale,
    job_id,
    images_per_batch=1,
    guidance="none",
    pag_scale=0.3,
    loras=None,
    lora_bytes=None,
):
    global loaded_pipeline, loaded_checkpoint, loaded_checkpoint_hash, loaded_family, loaded_engine, loaded_model_assets
    global loaded_model_revisions
    global pipeline_cpu_parked, active_memory_strategy, active_attention_backend, active_compute_dtype, active_vae_mode
    assets = {
        "diffusion_model": diffusion_model,
        "text_encoder": text_encoder,
        "vae": vae,
    }
    lora_descriptors = []
    for descriptor in loras or []:
        if isinstance(descriptor, dict):
            path, multiplier = descriptor["path"], descriptor["multiplier"]
        else:
            path, multiplier = descriptor
        if float(multiplier) != 0.0:
            lora_descriptors.append((Path(path), float(multiplier)))
    identity = {
        **{name: str(path) for name, path in assets.items()},
        "loras": [(str(path), multiplier) for path, multiplier in lora_descriptors],
    }
    revisions = {}
    revision_assets = {**assets, **{f"tokenizer_{key}": value for key, value in tokenizer_sources.items()}}
    for name, path in revision_assets.items():
        file_stat = path.stat()
        content_identity = file_sha256(path) if name.startswith("tokenizer_") else None
        revisions[name] = (
            str(path), file_stat.st_size, file_stat.st_mtime_ns, file_stat.st_ctime_ns, content_identity
        )
    lora_revisions = []
    measured_lora_bytes = 0
    for path, multiplier in lora_descriptors:
        file_stat = path.stat()
        measured_lora_bytes += file_stat.st_size
        lora_revisions.append(
            (str(path), file_stat.st_size, file_stat.st_mtime_ns, file_stat.st_ctime_ns, multiplier)
        )
    lora_bytes = measured_lora_bytes if lora_bytes is None else max(int(lora_bytes), measured_lora_bytes)
    revisions["loras"] = tuple(lora_revisions)
    with pipeline_lock:
        if not torch.cuda.is_available():
            raise RuntimeError("Native Anima generation requires CUDA")
        if not hasattr(torch.cuda, "is_bf16_supported") or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native Anima requires a CUDA GPU with BF16 support")
        configure_anima_cuda_math()
        if (
            loaded_pipeline is not None
            and loaded_engine == "Anima"
            and loaded_family == "anima"
            and loaded_model_assets == identity
            and loaded_model_revisions == revisions
        ):
            strategy = anima_memory_strategy(
                diffusion_model,
                assets,
                width,
                height,
                guidance_scale,
                images_per_batch,
                guidance,
                include_loaded=True,
                lora_bytes=lora_bytes,
                pag_scale=pag_scale,
            )
            if getattr(loaded_pipeline, "transformer_group_offload_enabled", False):
                strategy = choose_anima_memory_strategy(
                    diffusion_model,
                    width,
                    height,
                    guidance_scale,
                    images_per_batch,
                    guidance,
                    int(loaded_pipeline.weight_sizes.get("total", 0)),
                    int(loaded_pipeline.weight_sizes.get("transformer", 0)),
                    include_loaded=True,
                    lora_bytes=lora_bytes,
                    transformer_group_bytes=(
                        int(loaded_pipeline.weight_sizes.get("transformer_max_block", 0))
                        + int(loaded_pipeline.weight_sizes.get("transformer_unmatched", 0))
                    ),
                    pag_scale=pag_scale,
                )
                if not strategy.get("transformer_group_offload", False):
                    # A cache can safely move back to resident sampling only after
                    # all Diffusers group hooks are removed and the transformer is
                    # known CPU-placed.  Never leave a half-hooked cache reusable.
                    try:
                        loaded_pipeline._remove_transformer_group_offload()
                    except BaseException:
                        loaded_pipeline._poisoned = True
                        try:
                            loaded_pipeline.close()
                        finally:
                            clear_pipeline()
                        raise RuntimeError("Discarded Anima cache after group-offload hook removal failed")
            if strategy.get("transformer_group_offload", False) and not getattr(loaded_pipeline, "transformer_group_offload_enabled", False):
                loaded_pipeline.configure_transformer_residency(False)
                loaded_pipeline.enable_transformer_group_offload(strategy["transformer_blocks_per_group"])
            elif not strategy.get("keep_transformer_resident", False):
                to_cpu = getattr(loaded_pipeline, "to_cpu", None)
                if callable(to_cpu):
                    to_cpu()
            active_memory_strategy = strategy
            configure_anima_attention_backend(loaded_pipeline)
            strategy["acceleration"] = configure_anima_acceleration(loaded_pipeline, strategy)
            loaded_pipeline.batch_cfg = strategy.get("cfg_batch", False)
            configure_residency = getattr(loaded_pipeline, "configure_transformer_residency", None)
            if callable(configure_residency):
                configure_residency(strategy.get("keep_transformer_resident", False))
            active_compute_dtype = "bf16"
            active_vae_mode = configure_anima_vae(loaded_pipeline, strategy)
            pipeline_cpu_parked = False
            record_mount_decision(job_id, "anima", "Anima", width, height, reused=True)
            update_job(
                job_id,
                phase="Reusing loaded Anima runtime",
                progress=10,
                attention_backend=active_attention_backend,
                compute_dtype=active_compute_dtype,
                vae_mode=active_vae_mode,
                loaded_checkpoint=diffusion_model.name,
                loaded_checkpoint_path=loaded_checkpoint_reference(),
                loaded_model_assets=loaded_model_asset_references(),
                loaded_engine="Anima",
                model_reused=True,
                **memory_job_fields(strategy),
            )
            return loaded_pipeline

        clear_pipeline()
        strategy = anima_memory_strategy(
            diffusion_model,
            assets,
            width,
            height,
            guidance_scale,
            images_per_batch,
            guidance,
            lora_bytes=lora_bytes,
            pag_scale=pag_scale,
        )
        update_job(
            job_id,
            phase=f"Loading Anima components · {strategy['label']}",
            progress=4,
            **memory_job_fields(strategy, model_cached=False),
        )
        runtime_tokenizers = anima_runtime_tokenizer_paths(tokenizer_sources)
        runtime = None
        try:
            runtime = load_anima_runtime(
                diffusion_model,
                text_encoder,
                vae,
                runtime_tokenizers["qwen"],
                runtime_tokenizers["t5"],
                dtype=torch.bfloat16,
                loras=lora_descriptors,
            )
            if performance_settings["calculate_model_hash"]:
                stat = diffusion_model.stat()
                update_job(job_id, phase="Calculating model identity", progress=15)
                loaded_checkpoint_hash = checkpoint_sha256(
                    str(diffusion_model), stat.st_size, stat.st_mtime_ns
                )
            else:
                loaded_checkpoint_hash = None
            actual_total = int(runtime.weight_sizes.get("total", 0))
            actual_largest = max(
                (int(size) for name, size in runtime.weight_sizes.items() if name != "total"),
                default=0,
            )
            if actual_total and actual_largest:
                strategy = choose_anima_memory_strategy(
                    diffusion_model,
                    width,
                    height,
                    guidance_scale,
                    images_per_batch,
                    guidance,
                    actual_total,
                    actual_largest,
                    lora_bytes=lora_bytes,
                    transformer_group_bytes=(
                        int(runtime.weight_sizes.get("transformer_max_block", 0))
                        + int(runtime.weight_sizes.get("transformer_unmatched", 0))
                    ),
                    force_group_offload=strategy.get("transformer_group_offload", False),
                    pag_scale=pag_scale,
                )
            if strategy.get("transformer_group_offload", False):
                runtime.enable_transformer_group_offload(strategy["transformer_blocks_per_group"])
            configure_anima_attention_backend(runtime)
            strategy["acceleration"] = configure_anima_acceleration(runtime, strategy)
            runtime.batch_cfg = strategy.get("cfg_batch", False)
            configure_residency = getattr(runtime, "configure_transformer_residency", None)
            if callable(configure_residency):
                configure_residency(strategy.get("keep_transformer_resident", False))
            loaded_pipeline = runtime
            loaded_checkpoint = str(diffusion_model)
            loaded_family = "anima"
            loaded_engine = "Anima"
            loaded_model_assets = identity
            loaded_model_revisions = revisions
            pipeline_cpu_parked = False
            active_memory_strategy = strategy
            active_compute_dtype = "bf16"
            active_vae_mode = configure_anima_vae(runtime, strategy)
            record_mount_decision(job_id, "anima", "Anima", width, height, reused=False)
        except Exception:
            if runtime is not None:
                runtime.close()
            clear_pipeline()
            raise
        update_job(
            job_id,
            phase=f"Anima ready · {strategy['label']}",
            progress=18,
            attention_backend=active_attention_backend,
            compute_dtype=active_compute_dtype,
            vae_mode=active_vae_mode,
            loaded_checkpoint=diffusion_model.name,
            loaded_checkpoint_path=loaded_checkpoint_reference(),
            loaded_model_assets=loaded_model_asset_references(),
            loaded_engine="Anima",
            model_reused=False,
            **memory_job_fields(strategy),
        )
        return runtime


def flux_tokenizer_sources():
    """The two tokenizer resources a Flux run needs, both validated before any weight is read."""
    status = anima_tokenizer_status(force_hash=True)["t5"]
    if not status["installed"]:
        raise ValueError(f"Bundled T5 tokenizer resource is missing or corrupt ({status['reason']})")
    directory = flux_clip_tokenizer_directory()
    if directory is None:
        raise ValueError("缺少 CLIP-L 分词器本地运行配置，请联网重新运行环境配置器后再生成。")
    return {"t5": status["path"], "clip": directory}


def prompt_carries_weight_syntax(prompt: str) -> bool:
    """Whether a prompt asks for emphasis an engine may not honour.

    Read through the same parser the weighting engines use, so what counts as a weight here is
    exactly what would have counted as one there. A prompt the parser refuses is not a weighted
    prompt for this purpose — it will fail its own way, later, with its own message.
    """
    try:
        return any(weight != 1.0 for _segment, weight in parse_prompt_weights(prompt))
    except (TypeError, ValueError):
        return False


def flux2_tokenizer_sources():
    """The one tokenizer resource a FLUX.2 run may need from disk.

    FLUX.2 [dev] needs none: ComfyUI packs Mistral's tekken table into the text encoder checkpoint
    itself, so it travels with the weights. [klein] reads Qwen's, which is the same pinned file
    Anima already uses — ComfyUI points its Qwen3 encoders at the Qwen2.5 tokenizer too.
    """
    status = anima_tokenizer_status(force_hash=True)["qwen"]
    if not status["installed"]:
        raise ValueError(f"Bundled Qwen tokenizer resource is missing or corrupt ({status['reason']})")
    return {"qwen": status["path"]}


def krea2_tokenizer_sources():
    """Krea 2's one tokenizer resource, which is the same pinned Qwen2.5 table.

    Qwen3-VL reads text through Qwen2.5's vocabulary, so a Krea 2 install needs nothing Anima and
    FLUX.2 [klein] have not already put on disk.
    """
    return flux2_tokenizer_sources()


def load_flux_pipeline(
    diffusion_model,
    text_encoder,
    text_encoder_2,
    vae,
    tokenizer_sources,
    width,
    height,
    guidance_scale,
    job_id,
    images_per_batch=1,
    loras=None,
    lora_bytes=None,
    engine="Flux",
    guidance_copies=1,
):
    """Stage one component-mounted generation's components into a runtime.

    FLUX.1, FLUX.2 and Krea 2 mount a different number of files, a different tokenizer and — for
    Krea 2 — a different number of branches per step, but everything around that (the identity
    check that lets a reload be skipped, the admission decision, the group-offload escape hatch and
    the job bookkeeping) is the same, so they share this path rather than each carrying a copy of
    it that can drift.
    """
    family = NATIVE_FAMILY_BY_ENGINE[engine]
    global loaded_pipeline, loaded_checkpoint, loaded_checkpoint_hash, loaded_family, loaded_engine, loaded_model_assets
    global loaded_model_revisions
    global pipeline_cpu_parked, active_memory_strategy, active_attention_backend, active_compute_dtype, active_vae_mode
    assets = {
        "diffusion_model": diffusion_model,
        "text_encoder": text_encoder,
        **({"text_encoder_2": text_encoder_2} if text_encoder_2 is not None else {}),
        "vae": vae,
    }
    lora_descriptors = []
    for descriptor in loras or []:
        path, multiplier = (descriptor["path"], descriptor["multiplier"]) if isinstance(descriptor, dict) else descriptor
        if float(multiplier) != 0.0:
            lora_descriptors.append((Path(path), float(multiplier)))
    identity = {
        **{name: str(path) for name, path in assets.items()},
        "loras": [(str(path), multiplier) for path, multiplier in lora_descriptors],
    }
    revisions = {}
    for name, path in assets.items():
        file_stat = path.stat()
        revisions[name] = (str(path), file_stat.st_size, file_stat.st_mtime_ns, file_stat.st_ctime_ns, None)
    for name, source in tokenizer_sources.items():
        # A directory has no meaningful digest; a pinned file does, and a swapped tokenizer has to
        # invalidate the cached runtime the same way a swapped weight would.
        if Path(source).is_dir():
            revisions[f"tokenizer_{name}"] = (str(source), 0, 0, 0, None)
        else:
            revisions[f"tokenizer_{name}"] = (
                str(source), Path(source).stat().st_size, 0, 0, file_sha256(Path(source))
            )
    lora_revisions = []
    measured_lora_bytes = 0
    for path, multiplier in lora_descriptors:
        file_stat = path.stat()
        measured_lora_bytes += file_stat.st_size
        lora_revisions.append((str(path), file_stat.st_size, file_stat.st_mtime_ns, file_stat.st_ctime_ns, multiplier))
    lora_bytes = measured_lora_bytes if lora_bytes is None else max(int(lora_bytes), measured_lora_bytes)
    revisions["loras"] = tuple(lora_revisions)

    with pipeline_lock:
        if not torch.cuda.is_available():
            raise RuntimeError(f"Native {engine} generation requires CUDA")
        if not hasattr(torch.cuda, "is_bf16_supported") or not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"Native {engine} requires a CUDA GPU with BF16 support")
        configure_anima_cuda_math()
        if (
            loaded_pipeline is not None
            and loaded_engine == engine
            and loaded_family == family
            and loaded_model_assets == identity
            and loaded_model_revisions == revisions
        ):
            strategy = choose_flux_memory_strategy(
                diffusion_model,
                width,
                height,
                guidance_scale,
                images_per_batch,
                int(loaded_pipeline.weight_sizes.get("total", 0)),
                max(
                    (int(size) for name, size in loaded_pipeline.weight_sizes.items() if name != "total"),
                    default=0,
                ),
                include_loaded=True,
                lora_bytes=lora_bytes,
                transformer_group_bytes=(
                    int(loaded_pipeline.weight_sizes.get("transformer_max_block", 0))
                    + int(loaded_pipeline.weight_sizes.get("transformer_unmatched", 0))
                ),
                force_group_offload=bool(getattr(loaded_pipeline, "transformer_group_offload_enabled", False)),
                family=family,
                guidance_copies=guidance_copies,
            )
            if strategy.get("transformer_group_offload", False) and not getattr(loaded_pipeline, "transformer_group_offload_enabled", False):
                loaded_pipeline.configure_transformer_residency(False)
                loaded_pipeline.enable_transformer_group_offload(strategy["transformer_blocks_per_group"])
            elif not strategy.get("keep_transformer_resident", False):
                loaded_pipeline.to_cpu()
            active_memory_strategy = strategy
            configure_anima_attention_backend(loaded_pipeline)
            loaded_pipeline.configure_transformer_residency(strategy.get("keep_transformer_resident", False))
            configure_decode_residency(loaded_pipeline)
            active_compute_dtype = "bf16"
            active_vae_mode = configure_flux_vae(loaded_pipeline, strategy)
            pipeline_cpu_parked = False
            record_mount_decision(job_id, family, engine, width, height, reused=True)
            update_job(
                job_id,
                phase=f"Reusing loaded {engine} runtime",
                progress=10,
                attention_backend=active_attention_backend,
                compute_dtype=active_compute_dtype,
                vae_mode=active_vae_mode,
                loaded_checkpoint=diffusion_model.name,
                loaded_checkpoint_path=loaded_checkpoint_reference(),
                loaded_model_assets=loaded_model_asset_references(),
                loaded_engine=engine,
                model_reused=True,
                **memory_job_fields(strategy),
            )
            return loaded_pipeline

        clear_pipeline()
        strategy = flux_memory_strategy(
            diffusion_model, assets, width, height, guidance_scale, images_per_batch,
            lora_bytes=lora_bytes, family=family, guidance_copies=guidance_copies,
        )
        update_job(
            job_id,
            phase=f"Loading {engine} components · {strategy['label']}",
            progress=4,
            **memory_job_fields(strategy, model_cached=False),
        )
        runtime = None
        try:
            if engine == "Flux2":
                runtime = load_flux2_runtime(
                    diffusion_model,
                    text_encoder,
                    vae,
                    tokenizer_sources.get("qwen"),
                    dtype=torch.bfloat16,
                    loras=lora_descriptors,
                )
            elif engine == "Krea2":
                runtime = load_krea2_runtime(
                    diffusion_model,
                    text_encoder,
                    vae,
                    tokenizer_sources.get("qwen"),
                    dtype=torch.bfloat16,
                    loras=lora_descriptors,
                )
            else:
                runtime = load_flux_runtime(
                    diffusion_model,
                    text_encoder,
                    text_encoder_2,
                    vae,
                    tokenizer_sources["clip"],
                    tokenizer_sources["t5"],
                    dtype=torch.bfloat16,
                    loras=lora_descriptors,
                )
            if performance_settings["calculate_model_hash"]:
                stat = diffusion_model.stat()
                update_job(job_id, phase="Calculating model identity", progress=15)
                loaded_checkpoint_hash = checkpoint_sha256(str(diffusion_model), stat.st_size, stat.st_mtime_ns)
            else:
                loaded_checkpoint_hash = None
            actual_total = int(runtime.weight_sizes.get("total", 0))
            actual_largest = max(
                (int(size) for name, size in runtime.weight_sizes.items() if name != "total"), default=0
            )
            if actual_total and actual_largest:
                # The measured sizes replace the header estimate: a fused LoRA and a dequantised
                # fp8 checkpoint both make the loaded model larger than the file suggested.
                strategy = choose_flux_memory_strategy(
                    diffusion_model,
                    width,
                    height,
                    guidance_scale,
                    images_per_batch,
                    actual_total,
                    actual_largest,
                    lora_bytes=lora_bytes,
                    transformer_group_bytes=(
                        int(runtime.weight_sizes.get("transformer_max_block", 0))
                        + int(runtime.weight_sizes.get("transformer_unmatched", 0))
                    ),
                    force_group_offload=strategy.get("transformer_group_offload", False),
                    family=family,
                    guidance_copies=guidance_copies,
                )
            if strategy.get("transformer_group_offload", False):
                runtime.enable_transformer_group_offload(strategy["transformer_blocks_per_group"])
            configure_anima_attention_backend(runtime)
            runtime.configure_transformer_residency(strategy.get("keep_transformer_resident", False))
            configure_decode_residency(runtime)
            loaded_pipeline = runtime
            loaded_checkpoint = str(diffusion_model)
            loaded_family = family
            loaded_engine = engine
            loaded_model_assets = identity
            loaded_model_revisions = revisions
            pipeline_cpu_parked = False
            active_memory_strategy = strategy
            active_compute_dtype = "bf16"
            active_vae_mode = configure_flux_vae(runtime, strategy)
            record_mount_decision(job_id, family, engine, width, height, reused=False)
        except Exception:
            if runtime is not None:
                runtime.close()
            clear_pipeline()
            raise
        update_job(
            job_id,
            phase=f"{engine} ready · {strategy['label']}",
            progress=18,
            attention_backend=active_attention_backend,
            compute_dtype=active_compute_dtype,
            vae_mode=active_vae_mode,
            loaded_checkpoint=diffusion_model.name,
            loaded_checkpoint_path=loaded_checkpoint_reference(),
            loaded_model_assets=loaded_model_asset_references(),
            loaded_engine=engine,
            model_reused=False,
            **memory_job_fields(strategy),
        )
        return runtime


def configure_scheduler(pipeline, sampler_name: str, scheduler_name: str):
    config = pipeline.scheduler.config
    normalized = sampler_name.lower()
    warnings = []
    supported_schedulers = {
        "simple",
        "sgm_uniform",
        "karras",
        "exponential",
        "ddim_uniform",
        "beta",
        "normal",
        "linear_quadratic",
        "kl_optimal",
    }
    if scheduler_name not in supported_schedulers:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    scheduler_options = {
        "use_karras_sigmas": scheduler_name == "karras",
        "use_exponential_sigmas": scheduler_name == "exponential",
        "use_beta_sigmas": scheduler_name == "beta",
        "timestep_spacing": "trailing" if scheduler_name in {"sgm_uniform", "ddim_uniform"} else "linspace",
    }
    if scheduler_name in {"sgm_uniform", "ddim_uniform"}:
        warnings.append(f"{scheduler_name} uses Diffusers trailing timesteps, not ComfyUI's exact schedule")
    elif scheduler_name in {"linear_quadratic", "kl_optimal"}:
        warnings.append(f"{scheduler_name} is approximated with Diffusers linspace timesteps")

    if normalized in {"euler", "euler_cfg_pp"}:
        scheduler = EulerDiscreteScheduler.from_config(config, **scheduler_options)
        if normalized.endswith("cfg_pp"):
            warnings.append(f"{sampler_name} uses standard CFG; ComfyUI CFG++ guidance is not available in this backend")
    elif normalized in {"euler_ancestral", "euler_ancestral_cfg_pp"}:
        scheduler = EulerAncestralDiscreteScheduler.from_config(config, **scheduler_options)
        if normalized.endswith("cfg_pp"):
            warnings.append(f"{sampler_name} uses standard CFG; ComfyUI CFG++ guidance is not available in this backend")
    elif normalized == "heun":
        scheduler = HeunDiscreteScheduler.from_config(config, **scheduler_options)
    elif normalized == "dpm_2":
        scheduler = KDPM2DiscreteScheduler.from_config(config, **scheduler_options)
    elif normalized == "dpm_2_ancestral":
        scheduler = KDPM2AncestralDiscreteScheduler.from_config(config, **scheduler_options)
    elif normalized == "lms":
        scheduler = LMSDiscreteScheduler.from_config(config, **scheduler_options)
    elif normalized == "ddim":
        scheduler = DDIMScheduler.from_config(config, timestep_spacing=scheduler_options["timestep_spacing"])
    elif normalized == "ddpm":
        scheduler = DDPMScheduler.from_config(config, timestep_spacing=scheduler_options["timestep_spacing"])
    elif normalized == "lcm":
        scheduler = LCMScheduler.from_config(config)
    elif normalized in {"uni_pc", "uni_pc_bh2"}:
        scheduler = UniPCMultistepScheduler.from_config(
            config, solver_type="bh1" if normalized == "uni_pc" else "bh2", **scheduler_options
        )
    elif normalized == "deis":
        scheduler = DEISMultistepScheduler.from_config(config, **scheduler_options)
    elif normalized in {"ipndm", "ipndm_v"}:
        scheduler = IPNDMScheduler.from_config(config)
        if normalized == "ipndm_v":
            warnings.append("ipndm_v is approximated with standard IPNDM")
    elif normalized in {"dpm_fast", "dpm_adaptive"}:
        scheduler = DPMSolverSinglestepScheduler.from_config(
            config,
            algorithm_type="dpmsolver++",
            solver_order=2,
            **scheduler_options,
        )
        warnings.append(f"{sampler_name} is approximated with DPM++ 2S; adaptive step control is unavailable")
    elif normalized.startswith("dpmpp_"):
        order = 3 if normalized.startswith("dpmpp_3m") else 2
        algorithm_type = "sde-dpmsolver++" if "sde" in normalized else "dpmsolver++"
        scheduler_class = DPMSolverSinglestepScheduler if normalized.startswith("dpmpp_2s") else DPMSolverMultistepScheduler
        scheduler = scheduler_class.from_config(
            config,
            solver_order=order,
            algorithm_type=algorithm_type,
            solver_type="heun" if "heun" in normalized else "midpoint",
            **scheduler_options,
        )
        if normalized.startswith("dpmpp_2s_ancestral"):
            warnings.append("DPM++ 2S ancestral noise is approximated with a deterministic DPM++ 2S solver")
        elif normalized in {"dpmpp_sde", "dpmpp_sde_gpu"}:
            warnings.append("DPM++ SDE is approximated with Diffusers DPM++ 2M SDE")
        if normalized.endswith("_gpu"):
            warnings.append("ComfyUI's GPU-specific sampler path is not available in this backend")
        if "cfg_pp" in normalized:
            warnings.append(f"{sampler_name} uses standard CFG; ComfyUI CFG++ guidance is not available in this backend")
    elif normalized in {
        "heunpp2",
        "exp_heun_2_x0",
        "exp_heun_2_x0_sde",
        "res_multistep",
        "res_multistep_cfg_pp",
        "res_multistep_ancestral",
        "res_multistep_ancestral_cfg_pp",
        "gradient_estimation",
        "gradient_estimation_cfg_pp",
        "er_sde",
        "seeds_2",
        "seeds_3",
        "sa_solver",
        "sa_solver_pece",
    }:
        scheduler = DPMSolverMultistepScheduler.from_config(config, solver_order=2, algorithm_type="dpmsolver++", **scheduler_options)
        warnings.append(f"{sampler_name} is a ComfyUI k-diffusion sampler and is approximated with DPM++ 2M")
        if normalized.endswith("cfg_pp"):
            warnings.append(f"{sampler_name} uses standard CFG; ComfyUI CFG++ guidance is not available in this backend")
    else:
        raise ValueError(f"Unsupported sampler: {sampler_name}")

    pipeline.scheduler = scheduler
    return "; ".join(warnings) or None


def report_mounted_loras(job_id: str, entries):
    """Record what was mounted on the model, as ``(path, weight)`` pairs.

    Written once the adapters are actually on the pipeline rather than when they were requested,
    so the record — and the console line the record produces — describes what the sampler will
    run with. A file whose size cannot be read still counts as mounted; only the size is dropped.
    """
    mounted = []
    for path, weight in entries:
        path = Path(path)
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        mounted.append({"name": path.name, "weight": float(weight), "bytes": size})
    if mounted:
        update_job(job_id, mounted_loras=mounted)


def lora_signature(loras, lora_root: Path):
    """Lossless ordered identity for an irreversible adapter combination.

    Paths are first resolved through the trusted model-root boundary, then stored
    only as their relative wire reference.  File replacement invalidates reuse
    through size and nanosecond mtime without putting a local absolute path in a
    job, PNG or pipeline signature. Creation time and the platform's stable file
    identity close the same-size/mtime-restored replacement hole without hashing
    a potentially multi-hundred-megabyte adapter on every task.
    """
    signature = []
    for lora in loras or ():
        path = resolve_model_path(lora.path, lora_root, LORA_EXTENSIONS, "LoRA")
        try:
            stat = path.stat()
        except Exception as error:
            raise readable_lora_error(path, "文件修订", error) from None
        if is_shared_ref(lora.path):
            reference = shared_model_reference(lora.path)
            if not reference:
                raise ValueError("LoRA shared reference is invalid")
        else:
            reference = path.relative_to(lora_root.resolve()).as_posix()
        signature.append((
            reference,
            float(lora.weight).hex(),
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            int(getattr(stat, "st_dev", 0)),
            int(getattr(stat, "st_ino", 0)),
        ))
    return tuple(signature)


def lora_signature_summary(signature):
    encoded = json.dumps(tuple(signature or ()), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def mark_pipeline_lora_state(pipeline, state, signature, report=None):
    pipeline._xiriai_lora_fusion_state = {
        "schema": 1,
        "state": state,
        "signature": tuple(signature or ()),
    }
    pipeline._xiriai_fused_loras = tuple(signature or ()) if state == "clean" else None
    if report is not None:
        pipeline._xiriai_lora_fusion_report = dict(report)


def pipeline_lora_modules(pipeline):
    return {
        "unet": getattr(pipeline, "unet", None),
        "text_encoder": getattr(pipeline, "text_encoder", None),
        "text_encoder_2": getattr(pipeline, "text_encoder_2", None),
    }


def _lora_file_report(classification, groups, path):
    return {
        "name": Path(path).name,
        "format": classification.family,
        "advanced": classification.advanced,
        "targets": len(groups),
        "dora": classification.dora,
    }


def _pipeline_has_unmanaged_offload(pipeline):
    for component in pipeline_lora_modules(pipeline).values():
        if not isinstance(component, torch.nn.Module):
            continue
        for module in component.modules():
            if getattr(module, "_hf_hook", None) is not None:
                return True
            registry = getattr(module, "_diffusers_hook", None)
            if registry is not None:
                return True
    return False


def prepare_pipeline_for_direct_lora_patch(pipeline):
    """Materialize advanced-patch targets through the residency owner.

    Standard Diffusers adapters never enter this boundary. A production pipeline
    with offload installed must have its matching ``ResidencyController``; this
    function never tears hooks down behind that owner's back.
    """

    if loaded_pipeline is not None and pipeline is not loaded_pipeline:
        raise LoraPatchError("LoRA 应用失败（驻留状态）：拒绝补丁非当前或已停放的旧 pipeline")
    controller = (
        active_residency
        if active_residency is not None and active_residency.pipeline is pipeline
        else None
    )
    if pipeline is loaded_pipeline and pipeline_cpu_parked:
        try:
            restore_parked_pipeline()
        except Exception as error:
            raise readable_lora_error("LoRA", "驻留恢复", error) from None
        if pipeline_cpu_parked:
            raise LoraPatchError("LoRA 应用失败（驻留状态）：pipeline 仍处于 CPU 停放状态，必须重建")
    if controller is not None and controller.parked:
        try:
            controller.reinstall(controller.tier)
        except Exception as error:
            raise readable_lora_error("LoRA", "驻留恢复", error) from None
        if controller.parked:
            raise LoraPatchError("LoRA 应用失败（驻留状态）：ResidencyController 未能恢复停放 pipeline")
    if controller is None:
        if _pipeline_has_unmanaged_offload(pipeline):
            raise LoraPatchError("LoRA 应用失败（驻留状态）：检测到无 ResidencyController 管理的 offload hook")
        return None
    tier = controller.tier
    if tier is None:
        raise LoraPatchError("LoRA 应用失败（驻留状态）：ResidencyController 没有可恢复的驻留层级")
    try:
        disable_sequential_batch_forward(getattr(pipeline, "unet", None))
        controller.release(empty_cache=False)
    except Exception as error:
        raise readable_lora_error("LoRA", "驻留准备", error) from None
    return controller, tier


# The rungs that hold real weights where Diffusers folds a standard adapter: resident on the card, or
# whole components on the host behind `CpuOffload`. Anything below them streams or offloads per
# submodule and has to be released through the residency owner before a standard fold.
LORA_FOLD_IN_PLACE_TIERS = frozenset({"T0", "T1"})


def standard_lora_fold_needs_release(pipeline):
    """Whether a standard-only fold must go through the residency owner's release.

    `load_lora_weights` removes accelerate's hooks and installs them again for every component of
    every file. On the per-submodule rung (the LOW_VRAM admission for SD/iL) that reinstall moves
    every weight to the meta device, so the `fuse_lora` that follows folds into placeholders and is
    lost while the record says it was fused — and the three strips and reinstalls per file are most
    of what a standard-only mount cost there (QPlan P5 Phase A). The streaming rungs re-pin every
    group per load instead. Only a pipeline this process's controller owns is ever released.
    """
    controller = (
        active_residency
        if active_residency is not None and active_residency.pipeline is pipeline
        else None
    )
    if controller is None or controller.tier is None:
        return False
    return controller.parked or controller.tier not in LORA_FOLD_IN_PLACE_TIERS


def restore_pipeline_after_direct_lora_patch(pipeline, placement, job_id=None):
    if placement is None:
        return
    controller, tier = placement
    try:
        if not controller.reinstall(tier):
            raise RuntimeError("ResidencyController did not reinstall the saved tier")
        if pipeline is loaded_pipeline and active_memory_strategy is not None:
            apply_cfg_execution_mode(pipeline, active_memory_strategy)
            publish_residency(job_id)
    except Exception as error:
        raise readable_lora_error("LoRA", "驻留恢复", error) from None


def planned_target_weights(plan, modules):
    """The live weights one advanced plan will write."""
    weights = []
    for planned in plan.groups:
        component = modules.get(planned.target.component)
        if isinstance(component, torch.nn.Module):
            weights.append(component.get_parameter(planned.target.parameter))
    return weights


def peft_base_weights(modules):
    """The base weights under every PEFT adapter layer mounted so far — what a standard fold writes."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    weights = []
    for component in modules.values():
        if not isinstance(component, torch.nn.Module):
            continue
        for module in component.modules():
            if isinstance(module, BaseTunerLayer):
                weight = getattr(module.get_base_layer(), "weight", None)
                if isinstance(weight, torch.Tensor):
                    weights.append(weight)
    return weights


# What an accelerator must have free, beyond the largest single workspace a mount may hold on it at
# once (one advanced group, capped at MAX_GROUP_WORKSPACE_BYTES; one standard layer is far smaller),
# before it is used for the dense part of a host-resident mount.
LORA_COMPUTE_RESERVE_BYTES = 512 * 1024**2


def lora_compute_device():
    """The accelerator that takes the dense, element-wise part of a host-resident LoRA mount.

    Measured on the reference seven-LoRA job (QPlan P5 Phase A), the host spent about 44 s folding
    six standard adapters and patching 788 LoKr groups — memory-bound FP16/FP32 passes over
    billions of weights. The weights stay where the residency owner put them; only each layer's
    arithmetic visits the card and comes back. None — the established host path — when there is no
    CUDA device or it has too little free for one workspace.
    """
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, _total = torch.cuda.mem_get_info()
    except Exception:
        return None
    if free_bytes < MAX_GROUP_WORKSPACE_BYTES + LORA_COMPUTE_RESERVE_BYTES:
        return None
    return torch.device("cuda")


# An isolated benchmark may ask for one device synchronisation per mount stage, so that queued CUDA
# work is not billed to whichever stage happens to read a result first. Never per target, and never
# in production, where the stage clocks below cost nothing but `perf_counter` calls.
LORA_BENCHMARK_SYNC = os.environ.get("XIRAI_LORA_BENCHMARK_SYNC") == "1"


class LoraMountClock:
    """Stage clocks for one `configure_loras` call (QPlan P5 Phase A).

    Holds stage names, seconds, basenames and counts only — never a path, a key list or a tensor —
    so the result can ride on the job record, the PNG and a failure record alike. A stage that
    raises is still billed, which is what a dirty report needs to say where the time went.
    """

    def __init__(self):
        self.started = time.perf_counter()
        self.seconds = {}
        self.counts = {}
        self.standard_files = []
        self.advanced_files = []

    def _synchronize(self):
        if LORA_BENCHMARK_SYNC and torch.cuda.is_available():
            torch.cuda.synchronize()

    def add(self, name, seconds):
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds

    @contextmanager
    def stage(self, name):
        self._synchronize()
        begin = time.perf_counter()
        failed = True
        try:
            yield
            failed = False
        finally:
            if not failed:
                self._synchronize()
            self.add(name, time.perf_counter() - begin)

    def total_seconds(self):
        return round(time.perf_counter() - self.started, 3)

    def report(self):
        timings = {name: round(value, 3) for name, value in self.seconds.items()}
        timings.update(self.counts)
        if self.standard_files:
            timings["standard_files"] = [dict(item) for item in self.standard_files]
        if self.advanced_files:
            timings["advanced_files"] = [dict(item) for item in self.advanced_files]
        timings["total_seconds"] = self.total_seconds()
        return timings


def configure_loras(pipeline, loras: list[LoraInput], lora_root: Path, job_id: str):
    """Mount the adapters and fold them into the weights.

    ComfyUI applies LoRA by patching the model's weights; Diffusers mounts PEFT adapters, so every
    linear in the UNet computes `base(x) + sum_i s_i * B_i(A_i(x))` at every step. With six adapters
    that is twelve extra matmuls per linear, twenty-eight steps, two forwards a step — and on the
    reference job it is the single largest remaining cost. Measured on the reference card, same
    seed, 8 steps with the warm-up excluded:

        768x1280    mounted 1.917 s/it   fused 1.169 s/it   (streaming: 1.781 -> 0.718)
        1152x1920   mounted 4.975 s/it   fused 2.352 s/it   (streaming: 4.606 -> 2.050)

    This is the plan's P1-6, and the order it prescribes: load and fuse first, then let the
    controller measure and place what is left. Fusing changes the arithmetic — the adapter deltas
    are added into fp16 weights once instead of being applied per forward — so a fixed seed draws a
    slightly different picture than the mounted path did. It is the same operation ComfyUI performs,
    and it is not a quality reduction, but it is a visible change and it is recorded on the job.

    The fused set is remembered on the pipeline, because a fuse cannot be undone by
    `unload_lora_weights`: a later job whose adapters differ has to rebuild the pipeline rather than
    stack a second set on top of the first. A later job with the *same* set skips both the load and
    the fuse, which is most of what a cached model is worth.
    """
    clock = LoraMountClock()
    with clock.stage("signature_seconds"):
        requested = lora_signature(loras, lora_root)
        resolved = [
            (resolve_model_path(lora.path, lora_root, LORA_EXTENSIONS, "LoRA"), float(lora.weight))
            for lora in (loras or ())
        ]
    with clock.stage("reuse_lookup_seconds"):
        reusable = pipeline_lora_cache_compatible(pipeline, requested)
    if reusable:
        previous = getattr(pipeline, "_xiriai_lora_fusion_report", None)
        report = dict(previous) if isinstance(previous, dict) else {
            "schema": 1,
            "state": "clean",
            "signature": lora_signature_summary(requested),
            "standard": len(requested),
            "lycoris": 0,
            "files": [],
            "targets": 0,
            "dora": False,
        }
        report.update({"state": "clean", "reused": True, "count": len(requested), "fused": bool(requested)})
        # The reused record keeps the cold fold's `elapsed_seconds` (its established meaning), but
        # the clocks describe this call: a hot reuse reads no header and folds nothing.
        report["timings"] = clock.report()
        report["total_seconds"] = report["timings"]["total_seconds"]
        pipeline._xiriai_lora_fusion_report = report
        update_job(job_id, lora_fusion=report)
        report_mounted_loras(job_id, resolved)
        return report

    prior_state = getattr(pipeline, "_xiriai_lora_fusion_state", None)
    if isinstance(prior_state, dict):
        if prior_state.get("state") == "dirty":
            raise LoraPatchError("LoRA 应用失败（事务）：当前 pipeline 已标记为 dirty，必须先重建")
        if prior_state.get("state") == "clean" and prior_state.get("signature") != requested:
            # Normally load_pipeline has already rebuilt.  Reaching this branch
            # means a file revision changed between load admission and fusion;
            # patching the newly observed set over the old fused set is unsafe.
            mark_pipeline_lora_state(pipeline, "dirty", requested)
            changed_name = resolved[0][0].name if resolved else "LoRA"
            raise LoraPatchError(
                f"LoRA 应用失败（文件修订）：{changed_name} 在加载期间发生变化，请重试以重建模型"
            )

    # Only advanced markers opt a Safetensors file into strict Kohya/LyCORIS
    # parsing. Diffusers owns its native standard namespace (including PEFT
    # unet/text_encoder/text_encoder_2 keys), so pure standard files must never
    # be rejected by our narrower parser before load_lora_weights sees them.
    standard_files = []
    advanced_files = []
    file_reports = []
    modules = pipeline_lora_modules(pipeline)
    target_index = None
    for index, ((path, weight), lora) in enumerate(zip(resolved, loras or ())):
        if path.suffix.lower() != ".safetensors":
            # Preserve the established Diffusers path for legacy standard LoRA
            # containers.  They cannot be safely header-inspected without
            # deserializing pickle, so they are never admitted to direct patching.
            file_reports.append({
                "name": path.name,
                "format": "standard-legacy",
                "advanced": False,
                "targets": None,
                "dora": False,
            })
            standard_files.append({
                "index": index, "path": path, "weight": weight, "groups": (),
            })
            continue
        with clock.stage("header_seconds"):
            header = read_lora_header(path)
            direct = lora_header_requires_direct_patch(header)
        if not direct:
            file_reports.append({
                "name": path.name,
                "format": "standard-native",
                "advanced": False,
                "targets": None,
                "dora": False,
            })
            standard_files.append({
                "index": index, "path": path, "weight": weight, "groups": (),
            })
            continue
        with clock.stage("header_seconds"):
            classification = classify_lora_keys(header)
            groups = build_lora_groups(header)
        # Structural mapping (including every advanced file) completes against
        # the live parameter table before any hook, adapter, or base is changed.
        # One target index serves every advanced file of this transaction.
        if target_index is None:
            with clock.stage("target_index_seconds"):
                target_index = build_target_index(modules)
        preflight_started = time.perf_counter()
        with clock.stage("advanced_preflight_seconds"):
            plan = preflight_lora_patch(
                path,
                modules,
                loaded_family or ("sdxl" if getattr(pipeline, "text_encoder_2", None) is not None else "sd"),
                header=header,
                groups=groups,
                require_materialized=False,
                index=target_index,
            )
        file_reports.append(_lora_file_report(classification, groups, path))
        advanced_files.append({
            "index": index,
            "path": path,
            "weight": weight,
            "groups": groups,
            "plan": plan,
            "clock": {
                "name": path.name,
                "targets": len(groups),
                "preflight_seconds": round(time.perf_counter() - preflight_started, 3),
            },
        })

    if target_index is not None:
        clock.counts.update(target_index.counts())
        # Transaction-local by construction: nothing may resolve against it once the adapters and
        # the residency hooks start changing what the components hold.
        target_index = None

    base_report = {
        "schema": 1,
        "state": "clean",
        "signature": lora_signature_summary(requested),
        "reused": False,
        "standard": len(standard_files),
        "lycoris": len(advanced_files),
        "files": file_reports,
        "targets": sum(len(item["groups"]) for item in advanced_files),
        "dora": any(item["plan"].dora for item in advanced_files),
        "count": len(requested),
    }
    started = time.time()

    def clocked(report):
        report["timings"] = clock.report()
        report["total_seconds"] = report["timings"]["total_seconds"]
        return report

    def publish_dirty():
        dirty_report = clocked({
            **base_report,
            "state": "dirty",
            "fused": False,
            "elapsed_seconds": round(time.time() - started, 2),
        })
        mark_pipeline_lora_state(pipeline, "dirty", requested, dirty_report)
        update_job(job_id, lora_fusion=dirty_report)
        return dirty_report

    # unload/load/set/fuse and direct-patch residency preparation can all leave
    # mutable adapter/base state. Dirty precedes the first one, never follows it.
    publish_dirty()
    placement = None
    prefault = None

    def finish_prefault():
        # Joined before the first write to any base weight, and on every failure exit.
        if prefault is not None:
            clock.counts["host_prefault_busy_seconds"] = prefault.finish()
            clock.counts["host_prefault_bytes"] = prefault.bytes

    def restore_after_failure():
        # Every exit restores the rung this transaction released. The pipeline is already dirty and
        # will be rebuilt, so a restore that fails too must not replace the error that brought us here.
        finish_prefault()
        if placement is not None:
            try:
                restore_pipeline_after_direct_lora_patch(pipeline, placement, job_id)
            except Exception:
                clock.counts["residency_restore_failed"] = True

    # The residency owner releases the pipeline once for the whole transaction whenever a direct
    # patch needs materialized weights, or whenever Diffusers' own per-load offload handling would
    # otherwise run: on a streaming or per-submodule rung it strips and reinstalls the hooks for
    # every component of every file, and on the per-submodule rung the reinstall leaves every weight
    # on the meta device, where the fold that follows would be silently lost.
    if advanced_files or (standard_files and standard_lora_fold_needs_release(pipeline)):
        try:
            with clock.stage("residency_prepare_seconds"):
                placement = prepare_pipeline_for_direct_lora_patch(pipeline)
                for entry in advanced_files:
                    validate_lora_plan_materialized(entry["plan"], modules)
        except Exception:
            restore_after_failure()
            publish_dirty()
            raise
    if loras:
        # Take the copy-on-write faults of every weight this mount will write while Diffusers maps
        # the standard files: the advanced targets now, each standard file's layers as it lands.
        prefault = HostWeightPrefault()
        for entry in advanced_files:
            if entry["weight"] != 0.0:
                prefault.add(planned_target_weights(entry["plan"], modules))
    try:
        with clock.stage("standard_unload_seconds"):
            pipeline.unload_lora_weights()
    except Exception as error:
        restore_after_failure()
        publish_dirty()
        source = resolved[0][0] if resolved else "LoRA"
        raise readable_lora_error(source, "适配器清理", error) from None
    if not loras:
        base_report["fused"] = False
        clocked(base_report)
        mark_pipeline_lora_state(pipeline, "clean", requested, base_report)
        update_job(job_id, lora_fusion=base_report)
        return base_report

    update_job(job_id, phase="Loading LoRA adapters", progress=21)
    adapter_names = []
    adapter_weights = []
    for entry in standard_files:
        path = entry["path"]
        adapter_name = f"lora_{entry['index']}"
        load_started = time.perf_counter()
        try:
            with clock.stage("standard_load_seconds"), quiet_lora_loading(path):
                pipeline.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=adapter_name)
        except Exception as error:
            restore_after_failure()
            publish_dirty()
            raise readable_lora_error(path, "标准 LoRA", error) from None
        finally:
            clock.standard_files.append({
                "name": path.name, "load_seconds": round(time.perf_counter() - load_started, 3),
            })
        if prefault is not None:
            prefault.add(peft_base_weights(modules))
        adapter_names.append(adapter_name)
        adapter_weights.append(entry["weight"])
    if adapter_names:
        try:
            with clock.stage("standard_set_seconds"):
                pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
        except Exception as error:
            restore_after_failure()
            publish_dirty()
            raise readable_lora_error(standard_files[0]["path"], "标准 LoRA 组合", error) from None

    # Nothing below may run while the prefault thread still rewrites a queued weight.
    with clock.stage("host_prefault_wait_seconds"):
        finish_prefault()
    # Chosen after the residency release, when the card holds none of this pipeline's weights.
    compute_device = lora_compute_device()
    clock.counts["lora_compute"] = "cuda" if compute_device is not None else "host"
    fused = not adapter_names
    if adapter_names:
        update_job(job_id, phase="Folding LoRA into weights", progress=22)
        try:
            with clock.stage("standard_fuse_seconds"):
                if compute_device is None:
                    pipeline.fuse_lora(adapter_names=adapter_names)
                else:
                    # Diffusers has already mapped, injected and scaled every adapter; only the fold
                    # moves, and PEFT's own `merge` still folds every layer this does not replicate.
                    folded = fold_peft_adapters(modules, adapter_names, compute_device)
                    clock.counts["standard_fold_layers"] = folded["folded_layers"]
                    clock.counts["standard_fold_delegated"] = folded["delegated_layers"]
                    if folded["device_fallbacks"]:
                        # The card ran out part-way; the rest of this transaction stays on the host.
                        clock.counts["lora_compute_fallbacks"] = folded["device_fallbacks"]
                        compute_device = None
                    pipeline._merged_adapters = set(getattr(pipeline, "_merged_adapters", ())) | folded["merged_adapters"]
            with clock.stage("standard_unload_seconds"):
                pipeline.unload_lora_weights()
            fused = True
        except Exception as error:
            fused = False
            # The mounted fallback samples this job with the adapters still attached, so the rung
            # has to be back on the pipeline before it returns, not only before it raises.
            restore_after_failure()
            dirty_report = publish_dirty()
            report_mounted_loras(job_id, resolved)
            if advanced_files:
                raise readable_lora_error(standard_files[0]["path"], "标准+LyCORIS", error) from None
            # Preserve the established standard-only fallback: adapters remain
            # mounted for this job, but the dirty pipeline is never reusable.
            return dirty_report

    if advanced_files:
        update_job(job_id, phase="Applying LyCORIS patches", progress=23)
        try:
            for entry in advanced_files:
                apply_started = time.perf_counter()
                try:
                    with clock.stage("advanced_apply_seconds"):
                        apply_stats = {}
                        apply_lora_plan(
                            entry["path"], entry["plan"], modules, entry["weight"],
                            compute_device=compute_device, stats=apply_stats,
                        )
                        if apply_stats.get("device_fallbacks"):
                            clock.counts["lora_compute_fallbacks"] = (
                                clock.counts.get("lora_compute_fallbacks", 0) + apply_stats["device_fallbacks"]
                            )
                            compute_device = None
                finally:
                    clock.advanced_files.append({
                        **entry["clock"], "apply_seconds": round(time.perf_counter() - apply_started, 3),
                    })
        except Exception:
            restore_after_failure()
            publish_dirty()
            raise
    if clock.counts.get("lora_compute") == "cuda":
        # Hand the mount's workspaces back before the rung is reinstalled and the controller next
        # reads free memory to place the base stage.
        torch.cuda.empty_cache()
    if placement is not None:
        try:
            with clock.stage("residency_restore_seconds"):
                restore_pipeline_after_direct_lora_patch(pipeline, placement, job_id)
        except Exception:
            publish_dirty()
            raise

    # A replacement racing this task must poison the irreversible result rather
    # than publish a clean cache identity for bytes that were not one revision.
    with clock.stage("revision_recheck_seconds"):
        revision_changed = lora_signature(loras, lora_root) != requested
    if revision_changed:
        publish_dirty()
        changed_name = resolved[0][0].name if resolved else "LoRA"
        raise LoraPatchError(
            f"LoRA 应用失败（文件修订）：{changed_name} 在应用期间发生变化，请重试以重建模型"
        )

    report = clocked({
        **base_report,
        "state": "clean",
        "fused": bool(requested) and (fused or bool(advanced_files)),
        "elapsed_seconds": round(time.time() - started, 2),
    })
    mark_pipeline_lora_state(pipeline, "clean", requested, report)
    update_job(job_id, lora_fusion=report)
    report_mounted_loras(job_id, resolved)
    return report


def append_warning(current: str | None, message: str):
    return f"{current}; {message}" if current else message


def sampling_pipeline(pipeline, guidance: str, pag: PagInput | None = None):
    if guidance == "none":
        return pipeline, None
    if guidance == "cfg_zero_star":
        raise RuntimeError("CFG-Zero* Flow Matching pipeline adapter is not available for the current model engine")
    settings = pag or PagInput()
    try:
        guided = AutoPipelineForText2Image.from_pipe(
            pipeline,
            enable_pag=True,
            pag_applied_layers=pag_layer_pattern(settings.applied_layers),
        )
    except (ImportError, ValueError, TypeError) as error:
        raise RuntimeError(
            "PAG 初始化失败。请重新运行环境配置器，确保项目 Diffusers/PyTorch 运行库完整。"
        ) from error
    guided.set_progress_bar_config(disable=True)
    return guided, {"pag_scale": settings.scale}


def derived_sampling_pipeline(pipeline, factory, guidance: str, pag: PagInput | None = None):
    kwargs = {}
    guidance_kwargs = None
    if guidance == "pag":
        settings = pag or PagInput()
        kwargs = {"enable_pag": True, "pag_applied_layers": pag_layer_pattern(settings.applied_layers)}
        guidance_kwargs = {"pag_scale": settings.scale}
    try:
        active_pipeline = factory.from_pipe(pipeline, **kwargs)
    except (ImportError, ValueError, TypeError) as error:
        if guidance == "pag":
            raise RuntimeError(
                "PAG 初始化失败。请重新运行环境配置器，确保项目 Diffusers/PyTorch 运行库完整。"
            ) from error
        raise
    active_pipeline.set_progress_bar_config(disable=True)
    return active_pipeline, guidance_kwargs


def run_with_offload_escalation(pipeline, pipeline_kwargs, call):
    """Answer a mid-sampling OOM by offloading harder and re-running, rather than ending the job.

    Every native engine already has this answer: park the transformer, stream it a block at a time,
    restore the generator state and sample again, so the picture is unchanged and only the clock
    suffers (`anima_pipeline.py` sampling and refinement both do it). The Diffusers path had none —
    an OOM ended the run with "GPU memory exhausted" — and that is why its admission has to budget
    the whole chain's worst stage before the first step runs. On a card where the UNet is already
    4.9 GB that lands jobs on `enable_sequential_cpu_offload`, which moves every submodule per
    forward, when component offload would have held them.

    Restoring the generator state is what makes the retry honest. Diffusers cannot resume a
    half-finished call, so the second attempt starts over; without the restore it would start over
    from noise the first attempt had already drawn, and the retry would not be the same picture.
    """
    global active_memory_strategy
    requested = pipeline_kwargs.get("generator")
    generators = requested if isinstance(requested, list) else [requested] if requested is not None else []
    states = [generator.get_state() for generator in generators]
    try:
        return call()
    except BaseException as error:
        if not is_oom_error(error) or isinstance(error, StageMemoryError):
            raise
        # With a controller the next rung is the next one down *its* ladder. The mode ladder cannot
        # express that — T3 and T4 both report `low_vram` — and asking `reconfigure_memory_strategy`
        # for it would have the controller reinstall the rung that has just run out of memory.
        controlled = active_residency is not None and active_residency.pipeline is pipeline
        lower_tier = active_residency.next_lower_tier() if controlled else None
        dynamic = bool(controlled and active_residency.dynamic)
        if dynamic and lower_tier is None:
            # AUTO's floor is the per-submodule rung. The serial-CFG survival mode below it stays a
            # manual choice (QPlan P8 §3.3), so an OOM here is a readable stage error instead.
            raise stage_oom_error("采样期间显存不足，已处于最低驻留路径") from error
        heavier = (
            RESIDENCY_TIER_MEMORY_MODES[lower_tier] if lower_tier is not None
            else SD_OFFLOAD_ESCALATION.get((active_memory_strategy or {}).get("mode"))
        )
        if heavier is None:
            raise
        offload_to = (
            RESIDENCY_TIER_OFFLOAD_MODES[lower_tier] if lower_tier is not None
            else MEMORY_OFFLOAD_MODES[heavier]
        )
        torch.cuda.empty_cache()
        # An OOM is also the moment to stop batching the guidance branches, when this job is allowed
        # to split them: it halves the activation peak without touching the arithmetic, which is a
        # cheaper answer than the next offload rung and arrives with it rather than after it.
        serial_now = active_memory_strategy.get("cfg_serial_eligible", False)
        previous_tier = active_residency.tier if controlled else None
        if dynamic:
            # The legacy rung follows the escalation when it was where the real one was: the pre-P8
            # scheduler would have run out of memory in the same place and stepped down the same way.
            if active_residency.legacy_tier == previous_tier:
                active_residency.legacy_tier = lower_tier
            legacy_mode = RESIDENCY_TIER_MEMORY_MODES.get(active_residency.legacy_tier, heavier)
        else:
            legacy_mode = heavier
        escalated = {
            **active_memory_strategy,
            "mode": heavier,
            "legacy_mode": legacy_mode,
            "memory_path": path_for_tier(lower_tier) if lower_tier is not None else MEMORY_MODE_PATHS.get(heavier),
            "label": strategy_label(
                active_memory_strategy, heavier,
                path_for_tier(lower_tier) if lower_tier is not None else MEMORY_MODE_PATHS.get(heavier),
            ),
            "offload_mode": offload_to,
            "model_resident": False,
            "residency_tier": lower_tier or active_memory_strategy.get("residency_tier"),
            "cfg_mode": "serial" if serial_now else active_memory_strategy.get("cfg_mode", "batched"),
            "reason": active_memory_strategy.get("reason", "") + "；采样期间显存不足，已自动改用更重的卸载策略",
        }
        admission = escalated.get("admission")
        if isinstance(admission, dict):
            escalated["admission"] = {
                **admission,
                "actual_offload_mode": offload_to,
                "cfg_mode": escalated["cfg_mode"],
                "fallback": {
                    "stage": "sampling",
                    "reason": "cuda_oom",
                    "from": active_memory_strategy.get("offload_mode"),
                    "to": offload_to,
                    "cfg_mode_from": active_memory_strategy.get("cfg_mode", "batched"),
                    "cfg_mode_to": escalated["cfg_mode"],
                    "attempts": 1,
                    "generator_states_restored": True,
                },
            }
        if lower_tier is not None:
            active_residency.set_tier(lower_tier, reason="oom_escalation", stage=active_residency.stage or "sampling")
            if dynamic:
                active_residency.cap_allocator()
            apply_cfg_execution_mode(pipeline, escalated)
        else:
            # The bottom of the controller's ladder, or no controller at all: the existing rung
            # stays and ultra-low adds what it adds — serial CFG and a tiled autoencoder.
            reconfigure_memory_strategy(pipeline, escalated)
            if controlled:
                active_residency._record(
                    active_residency.stage or "sampling", active_residency.tier, active_residency.tier,
                    active_residency.observe(), "oom_escalation_ultra_low",
                )
        active_memory_strategy = escalated
        if dynamic:
            record_stage_decision(decision_record(
                stage=active_residency.stage or "sampling", engine=loaded_engine, family=loaded_family,
                path=path_for_tier(lower_tier), tier=lower_tier, reason="oom_retry_one_tier_lower",
                changed=True, legacy_tier=active_residency.legacy_tier,
            ), residency_job_id)
        for generator, state in zip(generators, states):
            generator.set_state(state)
        if not dynamic:
            return call()
        try:
            return call()
        except BaseException as retry_error:
            # One retry, one rung lower, the same noise (QPlan P8 §3.6). A second failure is the
            # stage's answer: say so plainly rather than keep stepping down.
            if is_oom_error(retry_error) and not isinstance(retry_error, StageMemoryError):
                raise stage_oom_error("降级一次重试后仍显存不足") from retry_error
            raise


def stage_oom_error(detail):
    """A mid-stage OOM under AUTO, restated as the stage error QPlan P8 §3.6 asks for."""
    decision = (active_residency.last_decision if active_residency is not None else None) or {}
    try:
        available = read_stage_budget(active_residency.resident_bytes() if active_residency else 0).ambient_free_bytes
    except Exception:
        available = 0
    return StageMemoryError(
        (active_residency.stage if active_residency is not None else None) or "sampling",
        decision.get("canvas"),
        decision.get("required_bytes") or 0,
        available,
        detail=detail,
    )


def keep_prompt_lease_seated(callback, prompt_lease):
    """Re-seat the prompt lease after every step, once the job's own callback has run.

    Residency work between steps (a pause, a park, a hook rebuild) can hand the UNet a new
    `forward`; the lease has to be back on top before Diffusers makes the next call, or its
    sentinel refuses that call. Diffusers' callback objects carry their own tensor list and are
    left alone — the lease still re-seats itself after each of its own calls.
    """
    if callback is not None and hasattr(callback, "tensor_inputs"):
        return callback

    def step_end(active, step, timestep, callback_kwargs):
        result = callback(active, step, timestep, callback_kwargs) if callback is not None else callback_kwargs
        prompt_lease.ensure_bound()
        return result

    return step_end


def run_guided_pipeline(
    pipeline,
    active_pipeline,
    pipeline_kwargs,
    guidance_kwargs=None,
    prompt_conditioning: PreparedConditioning | None = None,
):
    original_attn_processors = (
        dict(pipeline.unet.attn_processors)
        if guidance_kwargs and hasattr(pipeline, "unet") else None
    )
    try:
        with static_conditioning_lease(
            pipeline,
            prompt_conditioning,
            pipeline_kwargs.get("guidance_scale", 1.0),
            (guidance_kwargs or {}).get("pag_scale", 0.0),
        ) as prompt_lease:
            def invoke():
                # An OOM escalation can rebuild an Accelerate hook and replace
                # UNet.forward between attempts. Rebind only to the new live
                # layer, under the same pipeline-local lease token.
                call_kwargs = {**pipeline_kwargs, **(guidance_kwargs or {})}
                if prompt_lease is not None:
                    prompt_lease.begin_invocation(active_pipeline)
                    prompt_lease.ensure_bound()
                    call_kwargs["callback_on_step_end"] = keep_prompt_lease_seated(
                        call_kwargs.get("callback_on_step_end"), prompt_lease
                    )
                return active_pipeline(**call_kwargs)

            return run_with_offload_escalation(pipeline, pipeline_kwargs, invoke)
    finally:
        if original_attn_processors is not None:
            pipeline.unet.set_attn_processor(original_attn_processors)


def decode_source_image(image_data: str) -> Image.Image:
    """Decode an image-to-image source into an opaque RGB image.

    Transparency is flattened onto white rather than dropped: ``convert("RGB")`` on an RGBA source
    keeps whatever colour sits under a fully transparent pixel, which for a cut-out is usually
    black, and the sampler would then treat that black halo as content to be preserved.
    """
    if not isinstance(image_data, str) or not image_data:
        raise ValueError("Image-to-image requires a source image")
    encoded = image_data.split(",", 1)[-1]
    try:
        raw = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (ValueError, binascii.Error, OSError) as error:
        raise ValueError("无法读取来源图片，请改用 PNG、JPEG 或 WebP 格式") from error
    if image.mode == "RGB":
        return image
    if "A" in image.getbands() or image.mode == "P":
        rgba = image.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, (255, 255, 255))
        flattened.paste(rgba, mask=rgba.split()[-1])
        return flattened
    return image.convert("RGB")


def fit_source_image(image: Image.Image, size, resize_mode: str) -> Image.Image:
    """Resample a source image onto the requested canvas.

    ``cover`` scales up until both edges are covered and centre-crops the overflow, ``contain``
    scales down until the whole picture fits and centres it on white, ``stretch`` ignores the aspect
    ratio. Only the sampler's own canvas is negotiable here — the latent grid demands the exact
    requested size, so every mode has to end at it.
    """
    target_width, target_height = int(size[0]), int(size[1])
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Image-to-image canvas must have positive dimensions")
    if image.size == (target_width, target_height):
        return image
    if resize_mode == "stretch":
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    source_width, source_height = image.size
    ratios = (target_width / source_width, target_height / source_height)
    scale = max(ratios) if resize_mode == "cover" else min(ratios)
    scaled = image.resize(
        (max(1, round(source_width * scale)), max(1, round(source_height * scale))),
        Image.Resampling.LANCZOS,
    )
    if resize_mode == "cover":
        left = max(0, (scaled.width - target_width) // 2)
        top = max(0, (scaled.height - target_height) // 2)
        return scaled.crop((left, top, left + target_width, top + target_height))
    canvas = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    canvas.paste(scaled, ((target_width - scaled.width) // 2, (target_height - scaled.height) // 2))
    return canvas


def prepare_source_image(request: GenerateInput) -> Image.Image | None:
    """The source image a request samples from, already on the requested canvas.

    Post-processing mode is the exception: there is no sampler canvas to land on, so the picture is
    handed to the first stage exactly as the user supplied it. Resampling it would be a silent edit
    to the one thing that mode promises not to touch.
    """
    if not request.source_image.enabled:
        return None
    image = decode_source_image(request.source_image.image_data)
    if request.postprocess_only:
        return image
    return fit_source_image(image, (request.width, request.height), request.source_image.resize_mode)


def validate_postprocess_source_size(size) -> None:
    """The envelope a post-processing source has to fit in, checked wherever a real size appears."""
    width, height = int(size[0]), int(size[1])
    if width < 64 or height < 64:
        raise ValueError("Post-processing source must be at least 64 pixels on each edge")
    if max(width, height) > MAX_POSTPROCESS_SOURCE_EDGE or width * height > MAX_POSTPROCESS_SOURCE_PIXELS:
        raise ValueError("Post-processing source exceeds the safe 8192-edge / 32-megapixel limit")


def adopt_postprocess_canvas(request: GenerateInput, image: Image.Image) -> bool:
    """Make the request's canvas the picture's own size, and report whether it had to move.

    The client measures the source in the browser and the backend decodes it with Pillow. The two
    almost always agree, but an EXIF-rotated JPEG is reported one way round by an `<img>` element and
    the other way round by `Image.open`. Memory admission reads `width`/`height`, so the decoded
    picture — not the declared size — has to be what it reads.
    """
    if not request.postprocess_only:
        return False
    validate_postprocess_source_size(image.size)
    if (request.width, request.height) == image.size:
        return False
    request.width, request.height = image.size
    return True


def uses_diffusers_img2img_fix_steps(request: GenerateInput) -> bool:
    """Whether this request has a real SD/iL base image-to-image pass to expand."""
    return (
        request.img2img_fix_steps
        and request.engine in {"SD", "iL"}
        and request.source_image.enabled
        and not request.postprocess_only
    )


def img2img_schedule_steps(request: GenerateInput) -> int | None:
    """The expanded base schedule when fixed-step image-to-image is active."""
    if not uses_diffusers_img2img_fix_steps(request):
        return None
    return diffusers_schedule_total(request.steps, request.denoise)


def source_image_pipeline_kwargs(request: GenerateInput, source_image, generators, on_step_end, conditioning):
    """Diffusers image-to-image kwargs.

    ``width``/``height`` are deliberately absent: an image-to-image pipeline derives the canvas from
    the image it is handed, and ``prepare_source_image`` has already put it on the requested one.
    """
    if len(generators) != request.images_per_batch:
        raise ValueError("Generator count must match images_per_batch")
    return {
        "image": source_image,
        "strength": request.denoise,
        "num_inference_steps": img2img_schedule_steps(request) or request.steps,
        "num_images_per_prompt": request.images_per_batch,
        "guidance_scale": request.cfg,
        "generator": generators,
        "callback_on_step_end": on_step_end,
        "callback_on_step_end_tensor_inputs": ["latents"],
        **conditioning,
    }


def base_sampling_steps(request: GenerateInput, family: str) -> int:
    """How many denoise updates the base pass actually performs.

    Diffusers image-to-image runs ``int(steps * strength)`` of them, so reporting the requested
    count would leave the progress bar permanently short of its own total. Anima's refinement path
    executes every requested step regardless of denoise, and text-to-image has no strength at all.
    Post-processing mode performs none: `steps` survives only as what the stages inherit.
    """
    if request.postprocess_only:
        return 0
    if uses_diffusers_img2img_fix_steps(request):
        return request.steps
    if not request.source_image.enabled or is_native_family(family):
        return request.steps
    return max(1, int(request.steps * request.denoise))


def run_guided_pipeline_batch(
    pipeline,
    guidance: str,
    pipeline_kwargs,
    pag: PagInput | None = None,
    factory=None,
    prompt_conditioning: PreparedConditioning | None = None,
):
    active_pipeline, guidance_kwargs = (
        derived_sampling_pipeline(pipeline, factory, guidance, pag)
        if factory is not None
        else sampling_pipeline(pipeline, guidance, pag)
    )
    try:
        outputs = run_guided_pipeline(
            pipeline,
            active_pipeline,
            pipeline_kwargs,
            guidance_kwargs,
            prompt_conditioning,
        ).images
        expected = pipeline_kwargs["num_images_per_prompt"]
        if len(outputs) != expected:
            raise RuntimeError(f"Pipeline returned {len(outputs)} images for a requested batch of {expected}")
        return outputs
    finally:
        if active_pipeline is not pipeline:
            del active_pipeline


def postprocessing_stages(request: GenerateInput):
    """The chain entries that will actually run, in order."""
    return [stage for stage in request.postprocess_chain if stage.enabled]


def postprocess_stage_types(request: GenerateInput):
    """The same list as stage type names, for the record blocks that describe a run."""
    return [stage.type for stage in postprocessing_stages(request)]


def resample_unit_sizes(size, unit: ResampleUnitInput, family: str):
    """The three sizes one unit deals in.

    `target` is what the user asked for: `round(edge x scale)` and nothing else. `redraw` is what the
    engine will accept, which is `target` rounded to its latent grid — the same number for most
    scales and a few pixels away for the rest. `output` is what leaves the unit, which is the redraw
    canvas unless the unit asked for its exact target back.
    """
    target = scaled_size(size, unit.scale)
    if unit.denoise <= 0:
        # No redraw, so no grid to satisfy: the resize lands on the exact target.
        return {"target": target, "redraw": target, "output": target}
    redraw = redraw_canvas_size(target, family)
    return {"target": target, "redraw": redraw, "output": target if unit.exact_output else redraw}


DIFFUSERS_SCHEDULE_STEP_LIMIT = 1000


def diffusers_schedule_total(steps: int, denoise: float) -> int:
    """Return the smallest Diffusers schedule whose strength suffix runs ``steps`` updates.

    Diffusers truncates image-to-image and inpaint schedules with
    ``int(num_inference_steps * strength)``. Resample and ADetailer expose executed updates instead,
    so SD/iL expand the schedule while leaving ``strength`` unchanged. The narrow search around the
    mathematical quotient also uses Diffusers' exact floating-point operation, avoiding an off-by-one
    at a representational boundary.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("Diffusers schedule steps must be a positive integer")
    if steps > DIFFUSERS_SCHEDULE_STEP_LIMIT:
        raise ValueError(
            f"Diffusers schedule cannot exceed {DIFFUSERS_SCHEDULE_STEP_LIMIT} steps; lower the steps"
        )
    if isinstance(denoise, bool):
        raise ValueError("重绘强度必须是大于 0 的有限数值")
    try:
        strength = float(denoise)
    except (TypeError, ValueError) as error:
        raise ValueError("重绘强度必须是大于 0 的有限数值") from error
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("重绘强度必须是大于 0 的有限数值")
    if strength >= 1:
        return steps

    try:
        estimated_total = math.ceil(steps / strength)
    except (OverflowError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"重绘强度过低：调度总步数不能超过 {DIFFUSERS_SCHEDULE_STEP_LIMIT}，请提高重绘强度或降低步数"
        ) from error
    first = max(steps, estimated_total - 2)
    last = min(DIFFUSERS_SCHEDULE_STEP_LIMIT, estimated_total + 2)
    for total in range(first, last + 1):
        if int(total * strength) == steps:
            return total
    if estimated_total > DIFFUSERS_SCHEDULE_STEP_LIMIT:
        raise ValueError(
            f"重绘强度过低：调度总步数不能超过 {DIFFUSERS_SCHEDULE_STEP_LIMIT}，请提高重绘强度或降低步数"
        )
    raise ValueError("无法为当前步数与重绘强度构造 Diffusers 调度，请调整后重试")


def validate_diffusers_postprocess_schedules(request: GenerateInput) -> None:
    """Reject expanded SD/iL schedules before a job is queued or loads a model."""
    if uses_diffusers_img2img_fix_steps(request):
        try:
            diffusers_schedule_total(request.steps, request.denoise)
        except ValueError as error:
            raise ValueError(f"图生图基础采样：{error}") from error
    if is_native_family(request_family(request.engine)):
        return
    for stage in postprocessing_stages(request):
        if stage.type == "resample":
            if stage.denoise <= 0:
                continue
            label = f"重采样 {request.resample_unit_number(stage)}"
            steps = stage.resolved_steps(request.steps)
            schedules = ((label, steps, stage.denoise),)
        elif stage.type == "adetailer":
            schedules = tuple(
                (
                    f"ADetailer {index}" if len(stage.active_units) > 1 else "ADetailer",
                    unit.steps if unit.use_steps else request.steps,
                    unit.denoise,
                )
                for index, unit in enumerate(stage.active_units, start=1)
            )
        else:
            continue
        for label, steps, denoise in schedules:
            try:
                diffusers_schedule_total(steps, denoise)
            except ValueError as error:
                raise ValueError(f"{label}：{error}") from error


def validate_latent_resample_request(request: GenerateInput) -> None:
    """Defensive admission gate, including callers that bypass normal Pydantic construction."""
    for number, unit in enumerate(request.resample_units, start=1):
        if not unit.enabled or unit.latent_algorithm is None:
            continue
        if unit.upscaler is not None:
            raise ValueError(f"Resample unit {number}: latent resampling cannot use an upscaler model")
        if unit.execution_mode != "full_frame":
            raise ValueError(f"Resample unit {number}: latent resampling supports full_frame only; USDU is unavailable")
        if request.engine not in {"SD", "iL"}:
            raise ValueError(f"Resample unit {number}: 潜空间放大目前仅支持 SD/iL；{request.engine} 暂不支持")


def resample_chain_plan(request: GenerateInput, base_size=None):
    """Walk the enabled chain, accumulating sizes, without running anything.

    One walk serves three readers — request validation, memory admission and the stage headers the
    job reports — so a chain can never be admitted against one set of sizes and executed at another.
    """
    family = request_family(request.engine)
    size = tuple(base_size) if base_size is not None else (request.width, request.height)
    plan = []
    for stage in request.postprocess_chain:
        if not stage.enabled:
            continue
        entry = {"type": stage.type, "stage": stage, "input_size": size, "unit_number": 0}
        if stage.type == "resample":
            sizes = resample_unit_sizes(size, stage, family)
            entry.update({
                "unit_number": request.resample_unit_number(stage),
                "target_size": sizes["target"],
                "redraw_size": sizes["redraw"],
                "output_size": sizes["output"],
                "redraws": stage.denoise > 0,
                "upscales": bool(stage.upscaler),
            })
        elif stage.type == "rtx":
            entry["output_size"] = rtx_vsr.target_size(size, stage.scale)
        else:
            # ADetailer inpaints in place; the canvas it is handed is the canvas it returns.
            entry["output_size"] = size
        plan.append(entry)
        size = entry["output_size"]
    return plan


def chain_output_size(request: GenerateInput, base_size=None):
    plan = resample_chain_plan(request, base_size)
    if plan:
        return plan[-1]["output_size"]
    return tuple(base_size) if base_size is not None else (request.width, request.height)


def merge_runtime_metrics(job_id: str, metrics, namespace=None, previous=None):
    if not metrics:
        return
    previous = previous or {}
    selected = dict(metrics)
    if namespace:
        selected = {
            f"{namespace}.{name}": value
            for name, value in metrics.items()
            if name.startswith("refinement.") or previous.get(name) != value
        }
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        merged = dict(job.get("runtime_metrics") or {})
        merged.update(selected)
        job["runtime_metrics"] = merged


def logical_diffusion_step(pipeline, callback_index: int, logical_total: int):
    callback_total = max(1, int(getattr(pipeline, "num_timesteps", logical_total) or logical_total))
    return min(logical_total, max(1, math.ceil((callback_index + 1) * logical_total / callback_total)))


def generation_memory_workload(request: GenerateInput, family: str):
    return generation_memory_workload_diagnostics(request, family)["admission"]


USDU_AGGREGATE_SAMPLING_FIELDS = (
    "schedule_construction_steps", "executed_denoise_updates",
    "sequential_transformer_invocations", "actual_transformer_invocations",
)


def aggregate_usdu_sampling_metrics(tile_metrics):
    """Aggregate only concrete per-tile metrics; never invent missing values."""
    aggregate = {"tile_count": len(tile_metrics)}
    for field in USDU_AGGREGATE_SAMPLING_FIELDS:
        values = [item.get("sampling_metrics", {}).get(field) for item in tile_metrics
                  if isinstance(item.get("sampling_metrics"), dict) and isinstance(item["sampling_metrics"].get(field), int)]
        aggregate[field] = sum(values) if values else None
        aggregate[f"{field}_known_count"] = len(values)
        aggregate[f"{field}_complete"] = len(values) == len(tile_metrics)
    copies = [item.get("sampling_metrics", {}).get("peak_batch_copies") for item in tile_metrics
              if isinstance(item.get("sampling_metrics"), dict) and isinstance(item["sampling_metrics"].get("peak_batch_copies"), int)]
    aggregate["peak_batch_copies"] = max(copies) if copies else None
    aggregate["peak_batch_copies_known_count"] = len(copies)
    aggregate["peak_batch_copies_complete"] = len(copies) == len(tile_metrics)
    return aggregate


def generation_memory_workload_diagnostics(request: GenerateInput, family: str):
    # What ranks the stages is the *physical* forward count, because that is what the sampler will
    # allocate for. The native engines have always run their branches one after another; SD/iL now
    # does the same whenever serial CFG is eligible, so every stage of a diffusers job is compared
    # on one forward too. Where batched CFG is later admitted instead, it was admitted by clearing
    # the batched peak by half again, and that margin covers the difference in ranking.
    cfg_serial_eligible = serial_cfg_eligible(family, request.guidance)

    def physical_copies(cfg_value):
        if is_native_family(family) or cfg_serial_eligible:
            return 1
        return guidance_prediction_copies(request.guidance, cfg_value)

    guidance_copies = physical_copies(request.cfg)
    base = (request.width, request.height)
    diagnostics = {
        "guidance_copies": guidance_copies,
        "cfg_serial_eligible": cfg_serial_eligible,
        "base_dimensions": base,
        "target_dimensions": None,
        "core_dimensions": None,
        "processing_dimensions": None,
        "admission_dimensions": base,
        "stage": "base",
        "reason": "base_canvas",
        "target_excluded": False,
    }
    base_workload = (request.width, request.height, request.cfg, request.images_per_batch, guidance_copies, "base")
    # Post-processing mode never runs the base pass, so its canvas is not a workload — but a run
    # whose only stage is RTX (or ADetailer on a family that plans no admission entry for it) would
    # then have nothing to size the loaded pipeline against, so the base entry stands in.
    workloads = [] if request.postprocess_only else [base_workload]
    # Units run one after another, so the chain's demand is its most demanding single unit rather
    # than the sum: only one canvas is resident at a time.
    for entry in resample_chain_plan(request):
        stage = entry["stage"]
        if stage.type == "rtx":
            continue
        if stage.type == "resample":
            if not entry["redraws"]:
                # A resize with no redraw never touches the sampler, so it adds no inference bytes.
                continue
            resample_source_size = entry["input_size"]
            redraw_size = entry["redraw_size"]
            unit_cfg = stage.resolved_cfg(request.cfg)
            unit_copies = physical_copies(unit_cfg)
            admission_size = redraw_size
            reason = "full_frame_target"
            if supports_usdu_tiled(family) and stage.execution_mode == "usdu_tiled":
                # Match apply_resample_unit: Auto core is the canvas entering the unit, which an
                # earlier stage may already have changed.
                core = resample_source_size
                plan = plan_tiles(
                    redraw_size, core, padding=stage.padding,
                    seam_mode=stage.seam_mode, alignment=usdu_pixel_alignment(family),
                )
                admission_size = tuple(max(base, processed) for base, processed in zip(resample_source_size, plan.processing_size))
                reason = "usdu_tile_processing_target_excluded"
                diagnostics.update({
                    "target_dimensions": redraw_size,
                    "core_dimensions": core,
                    "processing_dimensions": plan.processing_size,
                    "admission_dimensions": admission_size,
                    "stage": "resample",
                    "unit_number": entry["unit_number"],
                    "reason": reason,
                    "target_excluded": True,
                })
            else:
                diagnostics.update({
                    "target_dimensions": redraw_size,
                    "admission_dimensions": admission_size,
                    "stage": "resample",
                    "unit_number": entry["unit_number"],
                    "reason": reason,
                })
            workloads.append((*admission_size, unit_cfg, 1, unit_copies, reason))
        elif stage.type == "adetailer" and is_native_family(family):
            # Units run one after another, never together, so admission is the
            # most demanding single unit rather than their sum.
            for unit in stage.active_units:
                detail_cfg = unit.cfg if unit.use_cfg else request.cfg
                detail_copies = physical_copies(detail_cfg)
                workloads.append((1024, 1024, detail_cfg, 1, detail_copies, "adetailer"))
    selected = max(workloads or [base_workload], key=lambda item: estimate_inference_bytes(family, *item[:5]))
    diagnostics["admission"] = selected[:4]
    diagnostics["admission_copies"] = selected[4]
    if selected[5] != "usdu_tile_processing_target_excluded":
        diagnostics["stage"] = selected[5]
    return diagnostics


# What streaming a sampler one submodule at a time still keeps on the card beside its activations:
# the block in flight, its hooks and the allocator's granularity. It is the floor of the lowest
# rung, which is the rung an oversized canvas would be pushed onto.
EXPANDED_CANVAS_STREAMED_MODULE_BYTES = 768 * 1024**2
# A tiled autoencoder decode: one tile's working set (measured at ~3.1 GiB for an SDXL tile, see
# `estimate_vae_decode_bytes`) plus the full-resolution float picture it assembles and blends.
EXPANDED_CANVAS_TILED_DECODE_BYTES = int(3.2 * GIB)
EXPANDED_CANVAS_DECODE_BYTES_PER_PIXEL = 24
# Host memory that grows with the picture rather than the model: the decoded float copies, the
# 8-bit picture, post-processing copies and the PNG encoder, per image held in one batch.
EXPANDED_CANVAS_HOST_BYTES_PER_PIXEL = 32
EXPANDED_CANVAS_HOST_FRACTION = 0.5


def expanded_canvas_resource_issue(request: GenerateInput) -> str | None:
    """Why a canvas past the historical 2048 cannot run on this machine, or `None`.

    Asked before a job exists. A canvas inside 2048 keeps the admission it always had — the
    runtime's own rungs and offload escalation — so nothing about an old request changes. Past it,
    the lowest rung still needs the sampler's activations on the card at once, and on Windows an
    allocation that does not fit is served out of host memory instead of failing, so the job would
    crawl or die long after it was accepted. The estimates are the same functions the runtime
    admission uses; nothing is reduced to make a request fit.
    """
    if request.postprocess_only or max(request.width, request.height) <= LEGACY_MAX_SAMPLING_EDGE:
        return None
    family = request_family(request.engine)
    if family == "sd":
        # iL is always SDXL. An SD checkpoint is only known to be SD 1.x once it is loaded, so until
        # then it is sized as SDXL, the more demanding of the two, rather than read from disk here.
        family = loaded_family if request.engine == "SD" and loaded_family in {"sd", "sdxl"} and loaded_checkpoint and Path(loaded_checkpoint).name == Path(request.checkpoint or "").name else "sdxl"
    diagnostics = generation_memory_workload_diagnostics(request, family)
    width, height, cfg, batch = diagnostics["admission"]
    inference_bytes = estimate_inference_bytes(family, width, height, cfg, batch, diagnostics["admission_copies"])
    canvas = f"{request.width}×{request.height}"
    if diagnostics["stage"] != "base":
        canvas += f"（后处理阶段 {width}×{height}）"
    if request.images_per_batch > 1:
        canvas += f"，单批 {request.images_per_batch} 张"
    if torch.cuda.is_available():
        physical_total = installed_vram_bytes()
        fixed_wall = performance_settings.get("vram_limit_gb", 0.0) > 0
        limit = effective_vram_limit_bytes(
            physical_total,
            performance_settings.get("vram_limit_gb", 0.0),
            allow_shared_memory=performance_settings["allow_shared_memory"],
        )
        budget = limit if fixed_wall else physical_total
        reserve = 0 if fixed_wall else int(RESERVED_VRAM_GB * GIB) if RESERVED_VRAM_GB is not None else reserved_vram_bytes(
            physical_total, allow_shared_memory=performance_settings["allow_shared_memory"],
        )
        sampling_bytes = inference_bytes + EXPANDED_CANVAS_STREAMED_MODULE_BYTES + reserve
        largest_output = max([request.width * request.height] + [
            entry["output_size"][0] * entry["output_size"][1] for entry in resample_chain_plan(request)
        ])
        decode_bytes = (
            EXPANDED_CANVAS_TILED_DECODE_BYTES + EXPANDED_CANVAS_DECODE_BYTES_PER_PIXEL * request.width * request.height + reserve
            if family in {"sd", "sdxl"} else 0
        )
        required = max(sampling_bytes, decode_bytes)
        if required > budget:
            stage = "采样" if sampling_bytes >= decode_bytes else "VAE 分块解码"
            return (
                f"画布 {canvas} 在本机预计需要约 {required / GIB:.1f} GiB 显存（{stage}），"
                f"超过可用上限 {budget / GIB:.1f} GiB；请降低画布、批量或后处理尺寸后重试。任务未入队，也不会自动降低分辨率。"
            )
    else:
        largest_output = request.width * request.height
    psutil = psutil_module()
    if psutil is not None:
        host_total = int(psutil.virtual_memory().total)
        host_bytes = EXPANDED_CANVAS_HOST_BYTES_PER_PIXEL * largest_output * request.images_per_batch
        if host_bytes > host_total * EXPANDED_CANVAS_HOST_FRACTION:
            return (
                f"画布 {canvas} 的图像缓冲预计需要约 {host_bytes / GIB:.1f} GiB 内存，"
                f"超过本机内存 {host_total / GIB:.1f} GiB 的一半；请降低画布或批量后重试。任务未入队。"
            )
    return None


def apply_rtx_vsr(
    image: Image.Image,
    pipeline,
    request: GenerateInput,
    job_id: str,
    control: JobControl,
    started_at: float,
    progress_start=80,
    progress_end=96,
):
    settings = request.rtx
    if not settings.enabled:
        return image, None
    destination = rtx_vsr.target_size(image, settings.scale)
    control.checkpoint(job_id, "RTX VSR upscaling")
    update_job(
        job_id,
        phase="RTX VSR · Parking diffusion model",
        stage="sampler_offload",
        stage_step=0,
        stage_total=0,
        progress=progress_start,
    )
    if not pipeline_cpu_parked:
        park_pipeline_for_external_stage(
            pipeline, NATIVE_FAMILY_BY_ENGINE.get(request.engine) or (loaded_family or "sd")
        )
    update_job(
        job_id,
        phase=f"RTX VSR · Upscaling {destination[0]} x {destination[1]}",
        stage="rtx_upscale",
        stage_step=0,
        stage_total=1,
        step=0,
        total_steps=0,
        progress=progress_start,
    )

    def on_progress(completed, total):
        update_job(
            job_id,
            stage_step=completed,
            stage_total=total,
            progress=round(progress_start + (progress_end - progress_start) * completed / max(1, total)),
            elapsed_seconds=round(control.active_elapsed(started_at), 1),
        )

    result, diagnostics = rtx_vsr.upscale_image(
        image,
        settings.scale,
        settings.quality,
        progress=on_progress,
        checkpoint=lambda: control.checkpoint(job_id, "RTX VSR upscaling"),
        cancel_check=control.raise_if_cancelled,
    )
    return result, diagnostics


def resample_unit_label(settings: ResampleUnitInput, unit_number: int):
    """What the job's phase text calls this unit. The user's name wins when they gave one."""
    name = (settings.name or "").strip()
    return f"Resample {unit_number} · {name}" if name else f"Resample {unit_number}"


def resample_stage_facts(
    settings: ResampleUnitInput,
    unit_number: int,
    base_seed: int,
    unit_seed,
    *,
    source_size,
    target_size,
    executed_size,
    steps,
    cfg,
    sampler_resolved,
    scheduler_resolved,
    execution_mode: str,
    effective_steps=None,
):
    """The facts every unit reports, whichever of the three paths produced the picture.

    `target_size` and `executed_size` are both recorded even when they agree. A reader that only
    saw one of them could not tell a chain that landed where it was asked from one an engine grid
    moved, and that difference is the whole reason `exact_output` exists.
    """
    facts = {
        "unit": unit_number,
        "name": settings.name or None,
        "upscaler": settings.upscaler,
        "latent_algorithm": settings.latent_algorithm,
        "scale": settings.scale,
        "resize_algorithm": settings.resize_algorithm,
        "exact_output": settings.exact_output,
        "execution_mode": execution_mode,
        "denoise": settings.denoise,
        "steps": steps,
        "effective_steps": effective_steps if effective_steps is not None else steps,
        "cfg": cfg,
        "sampler_requested": settings.sampler,
        "scheduler_requested": settings.scheduler,
        "sampler_resolved": sampler_resolved,
        "scheduler_resolved": scheduler_resolved,
        "source_size": list(source_size),
        "target_size": list(target_size),
        "executed_size": list(executed_size),
        "size_matches_target": tuple(target_size) == tuple(executed_size),
        "base_seed": str(base_seed),
        "seed_mode": settings.seed_mode,
        "seed": str(unit_seed) if unit_seed is not None else None,
        # The two names the single Hires block used. Kept so a reader written against the old
        # record — the gallery's seed restore, the artifact capture — still finds its fields.
        "hires_seed_mode": settings.seed_mode,
        "hires_seed": str(unit_seed) if unit_seed is not None else None,
    }
    return facts


def exact_resample_output(image, settings: ResampleUnitInput, target_size, diagnostics, control, job_id, label):
    """Put the picture back on the unit's exact target when it asked for that, and say so either way."""
    executed = tuple(image.size)
    target = tuple(target_size)
    diagnostics["executed_size"] = list(executed)
    diagnostics["size_matches_target"] = executed == target
    if not settings.exact_output or executed == target:
        return image
    control.checkpoint(job_id, f"{label} exact resize")
    corrected = resize_image(materialize_image(image), target[0], target[1], settings.resize_algorithm)
    diagnostics.update({
        "exact_output_applied": True,
        "pre_exact_size": list(executed),
        "executed_size": list(target),
        "size_matches_target": True,
    })
    return corrected


def apply_resample_unit(
    image: Image.Image | LatentImageArtifact,
    pipeline,
    family: str,
    request: GenerateInput,
    settings: ResampleUnitInput,
    job_id: str,
    control: JobControl,
    started_at: float,
    unit_number=1,
    image_seed=None,
    effective_seed=None,
    progress_start=80,
    progress_end=96,
):
    """Public resample entry point; benchmark stage capture stays a no-op unless its isolated gate passes."""
    if not settings.enabled:
        return image, None
    if settings.redraws and effective_seed is None:
        raise ValueError("A resample unit's effective seed must be resolved before the unit runs")
    base_seed = request.seed if image_seed is None else image_seed
    capture, capture_gate_facts = open_hires_artifact_capture(job_id, base_seed)
    stages = lambda: _apply_resample_unit_stages(
        image, pipeline, family, request, settings, job_id, control, started_at,
        unit_number=unit_number, image_seed=image_seed, effective_seed=effective_seed,
        progress_start=progress_start, progress_end=progress_end, capture=capture,
    )
    if not capture.enabled:
        return stages()
    try:
        result, diagnostics = stages()
        capture.stage("final", materialize_image(result))
        capture.finish(
            status="complete",
            protocol=INFERENCE_PROTOCOL,
            engine=request.engine,
            execution_mode=diagnostics.get("execution_mode"),
            base_seed=str(base_seed),
            hires_seed_mode=settings.seed_mode,
            effective_hires_seed=str(effective_seed) if effective_seed is not None else None,
            base_dimensions=list(image.size),
            post_sr_dimensions=list(result.size),
            tile_plan={
                "rows": diagnostics.get("rows"), "cols": diagnostics.get("cols"),
                "tile_count": diagnostics.get("tile_count"), "core_tile": list(diagnostics.get("core_tile") or []),
                "processing_size": list(diagnostics.get("model_size") or []),
                "padding": settings.padding, "mask_blur": settings.mask_blur,
                "uniform_tiles": settings.uniform_tiles, "seam_mode": settings.seam_mode,
            },
            canonical_parameter_sha256=canonical_parameter_digest({
                "engine": request.engine, "width": request.width, "height": request.height,
                "steps": request.steps, "cfg": request.cfg, "denoise": request.denoise,
                "sampler": request.sampler, "scheduler": request.scheduler,
                "resample": {
                    "unit": unit_number,
                    "scale": settings.scale, "steps": diagnostics.get("steps"), "cfg": diagnostics.get("cfg"),
                    "denoise": settings.denoise, "execution_mode": settings.execution_mode,
                    "resize_algorithm": settings.resize_algorithm,
                    "latent_algorithm": settings.latent_algorithm,
                    "exact_output": settings.exact_output,
                    "sampler": diagnostics.get("sampler_resolved"), "scheduler": diagnostics.get("scheduler_resolved"),
                    "padding": settings.padding, "mask_blur": settings.mask_blur,
                    "uniform_tiles": settings.uniform_tiles, "tiled_decode": settings.tiled_decode,
                },
            }),
            tiled_decode_requested=diagnostics.get("tiled_decode_requested"),
            tiled_decode_resolved=diagnostics.get("tiled_decode_resolved"),
            gate=capture_gate_facts,
            **prompt_facts(request.prompt, request.negative_prompt),
        )
        return result, diagnostics
    except BaseException:
        capture.abort()
        raise


def _apply_resample_unit_stages(
    image: Image.Image | LatentImageArtifact,
    pipeline,
    family: str,
    request: GenerateInput,
    settings: ResampleUnitInput,
    job_id: str,
    control: JobControl,
    started_at: float,
    unit_number=1,
    image_seed=None,
    effective_seed=None,
    progress_start=80,
    progress_end=96,
    capture=NULL_CAPTURE,
):
    if not settings.enabled:
        return image, None
    if settings.redraws and effective_seed is None:
        raise ValueError("A resample unit's effective seed must be resolved before the unit runs")
    if settings.latent_algorithm is not None:
        if family not in SD_COMPONENT_FAMILIES:
            raise ValueError(
                f"潜空间放大运行时仅支持 SD/SDXL family；当前 family 为 {family}"
            )
        if settings.upscaler is not None:
            raise ValueError("Latent resampling cannot be combined with an upscaler model")
        if settings.execution_mode != "full_frame":
            raise ValueError("Latent resampling supports full_frame execution only; USDU is unavailable")
    # A latent artifact is an internal chain value, not a public image boundary. Its logical size is
    # sufficient for planning and compatible latent units reuse the scaled tensor directly. Only a
    # pixel/model/callback boundary below is allowed to materialize it.
    base_seed = request.seed if image_seed is None else image_seed
    label = resample_unit_label(settings, unit_number)
    # USDU Auto follows the pre-SR image, even though planning happens after
    # pixel SR has produced the diffusion canvas.
    resample_source_size = tuple(image.size)
    unit_steps = settings.resolved_steps(request.steps)
    unit_cfg = settings.resolved_cfg(request.cfg)
    # Every refinement reports and executes the requested updates. Native engines build their own
    # Comfy-style suffix; SD/iL expand Diffusers' schedule below so strength only selects its start.
    effective_steps = unit_steps
    schedule_steps = (
        unit_steps
        if is_native_family(family) or not settings.redraws
        else diffusers_schedule_total(unit_steps, settings.denoise)
    )
    sizes = resample_unit_sizes(resample_source_size, settings, family)
    # `target` is the exact logical size the chain asked for; `destination` is the sampler canvas
    # when a redraw follows and otherwise the same logical target. A no-redraw latent may occupy the
    # contained VAE grid and defer its few-pixel correction until a real PIL boundary.
    target_size = sizes["target"]
    destination = sizes["redraw"]
    if settings.redraws and max(destination) > engine_max_refinement_edge(family):
        raise ValueError(f"Refinement dimensions cannot exceed {engine_max_refinement_edge(family)}")
    staged_decode = performance_settings["staged_vae_decode"] and not is_native_family(family)
    resizes = destination != resample_source_size
    latent_resample = settings.latent_algorithm is not None
    upscale_end = (
        progress_start + round((progress_end - progress_start) * 0.2)
        if resizes or latent_resample else progress_start
    )
    deferred_decode = staged_decode or latent_resample
    sampling_end = (
        progress_end - round((progress_end - progress_start) * 0.15)
        if deferred_decode else progress_end
    )
    control.checkpoint(job_id, f"{label} upscaling")
    reported_upscale_tiles = [0]

    def on_upscale_tile(completed, total):
        reported_upscale_tiles[0] = max(reported_upscale_tiles[0], completed)
        update_job(
            job_id,
            phase=f"{label} · Upscaling {destination[0]} x {destination[1]}",
            stage="resample_upscale",
            stage_step=reported_upscale_tiles[0],
            stage_total=total,
            progress=progress_start + round((upscale_end - progress_start) * reported_upscale_tiles[0] / max(1, total)),
            elapsed_seconds=round(control.active_elapsed(started_at), 1),
        )
        control.checkpoint(job_id, f"{label} upscaling")

    latent_generator = None
    if latent_resample:
        latent_mode = LATENT_UPSCALE_MODES[settings.latent_algorithm]
        latent_generator = torch.Generator(device="cpu").manual_seed(
            effective_seed if effective_seed is not None else base_seed
        )
        factor = pipeline_vae_scale_factor(pipeline)
        reuses_latents = (
            isinstance(image, LatentImageArtifact)
            and image.family == family
            and image.vae_scale_factor == factor
        )
        if not reuses_latents:
            plan_residency_for_stage(
                f"hires_latent_encode_{unit_number}", family, tuple(destination), unit_cfg, 1, job_id
            )
        preparation = "Reusing scaled latents" if reuses_latents else "Encoding image to latents"
        control.checkpoint(job_id, f"{label} latent {'reuse' if reuses_latents else 'encode'}")
        update_job(
            job_id,
            phase=f"{label} · {preparation} for {latent_mode['label']}",
            stage="resample_latent_encode",
            stage_step=0,
            stage_total=1,
            progress=progress_start,
        )
        if reuses_latents:
            source_latents = image.latents
        else:
            source_latents = encode_image_to_latents(
                pipeline, materialize_image(image).convert("RGB"), latent_generator
            )
        control.checkpoint(job_id, f"{label} latent interpolation")
        update_job(
            job_id,
            phase=f"{label} · {latent_mode['label']} {destination[0]} x {destination[1]}",
            stage="resample_latent_upscale",
            stage_step=0,
            stage_total=1,
            progress=progress_start,
        )
        interpolated, factor = interpolate_image_latents(
            pipeline, source_latents, destination, settings.latent_algorithm
        )
        latent_decode_size = (
            int(interpolated.shape[-1]) * factor,
            int(interpolated.shape[-2]) * factor,
        )
        upscaled = make_latent_image_artifact(
            interpolated,
            latent_decode_size,
            pipeline,
            family,
            job_id,
            control,
            staged_decode=staged_decode,
            start_progress=upscale_end,
            final_progress=progress_end,
            phase_prefix=label,
            output_size=target_size if not settings.redraws else None,
            output_resize_algorithm=settings.resize_algorithm if not settings.redraws else None,
        )
        diagnostics = {
            "model": None,
            "scale": settings.scale,
            "source_size": list(resample_source_size),
            "target_size": list(destination),
            "resize_algorithm": settings.resize_algorithm,
            "latent_algorithm": settings.latent_algorithm,
            "latent_mode": latent_mode["mode"],
            "latent_antialias": latent_mode["antialias"],
            "vae_scale_factor": factor,
            "latent_decode_size": list(latent_decode_size),
            "upscaler_applied": False,
            "device": interpolated.device.type,
        }
        update_job(
            job_id,
            phase=f"{label} · {latent_mode['label']} {destination[0]} x {destination[1]}",
            stage="resample_latent_upscale",
            stage_step=1,
            stage_total=1,
            progress=upscale_end,
        )
        upscale_warning = None
    else:
        image = materialize_image(image)
        use_cuda_upscaler = bool(settings.upscaler) and any(
            edge > source for edge, source in zip(destination, image.size)
        ) and torch.cuda.is_available()
        upscale_warning = None
        def run_upscaler(device: str, running_label: str):
            return upscale_image(
                image,
                UPSCALER_MODEL_DIRECTORY,
                settings.upscaler,
                settings.scale,
                settings.tile_size,
                settings.tile_overlap,
                on_upscale_tile,
                device=device,
                checkpoint=lambda: control.checkpoint(job_id, running_label),
                method=settings.resize_algorithm,
                destination=destination,
            )

        if settings.upscaler:
            update_job(job_id, phase=f"{label} · Loading upscaler", stage="resample_upscale", stage_step=0, stage_total=0, progress=progress_start)
            if use_cuda_upscaler:
                plan_upscale_stage(
                    f"hires_upscale_{unit_number}", pipeline, family, image.size, destination, settings, job_id,
                )
            try:
                try:
                    # The upscaler used to be given the card unconditionally, which meant parking the
                    # diffusion model before it and fetching it back for the refinement that follows —
                    # 10.1s out and 2.4s back on a 24 GB card, around an upscale that takes 2.0s. It never
                    # needed the room: the pass is tiled, so its footprint is bounded by one tile rather
                    # than by the picture, and an ESRGAN at a 192-pixel tile peaks far below what sampling
                    # just used with the model resident. So the parking is demand-driven now. Running out
                    # of memory is the signal, and it is answered the way it always was, one step later.
                    upscaled, diagnostics = run_upscaler(
                        "cuda" if use_cuda_upscaler else "cpu", f"{label} upscaling")
                except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                    if not use_cuda_upscaler or not is_oom_error(error):
                        raise
                    torch.cuda.empty_cache()
                    if not pipeline_cpu_parked:
                        # The room the tiled pass turned out to need is exactly what parking frees, and
                        # the GPU pass is worth another attempt before falling back to the CPU one.
                        update_job(job_id, phase=f"{label} · Parking diffusion model", stage="sampler_offload", progress=progress_start)
                        park_pipeline_for_external_stage(pipeline, family)
                        try:
                            upscaled, diagnostics = run_upscaler("cuda", f"{label} upscaling")
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as retry_error:
                            if not is_oom_error(retry_error):
                                raise
                            torch.cuda.empty_cache()
                            upscale_warning = f"{label} upscaler exceeded GPU memory and used CPU fallback"
                            update_job(job_id, phase=f"{label} · CPU fallback", stage="resample_upscale", warning=append_warning(None, upscale_warning))
                            upscaled, diagnostics = run_upscaler("cpu", f"{label} CPU upscaling")
                    else:
                        upscale_warning = f"{label} upscaler exceeded GPU memory and used CPU fallback"
                        update_job(job_id, phase=f"{label} · CPU fallback", stage="resample_upscale", warning=append_warning(None, upscale_warning))
                        upscaled, diagnostics = run_upscaler("cpu", f"{label} CPU upscaling")
            finally:
                if not is_native_family(family) and pipeline_cpu_parked and not control.cancelled:
                    update_job(job_id, phase=f"{label} · Restoring diffusion model", stage="model_restore")
                    restore_parked_pipeline()
        else:
            # No model means no model pass: the unit resizes and that is the whole pixel stage. It is
            # the same call the model path finishes with, so a unit whose model would have had nothing
            # to add lands on the same pixels either way.
            update_job(
                job_id,
                phase=f"{label} · Resizing {destination[0]} x {destination[1]}",
                stage="resample_resize", stage_step=0, stage_total=1, progress=progress_start,
            )
            control.checkpoint(job_id, f"{label} resizing")
            upscaled = resize_image(image, destination[0], destination[1], settings.resize_algorithm)
            diagnostics = {
                "model": None,
                "scale": settings.scale,
                "source_size": list(image.size),
                "target_size": list(destination),
                "resize_algorithm": settings.resize_algorithm,
                "upscaler_applied": False,
                "device": "cpu",
            }
            update_job(
                job_id,
                phase=f"{label} · Resizing {destination[0]} x {destination[1]}",
                stage="resample_resize", stage_step=1, stage_total=1, progress=upscale_end,
            )

    if settings.redraws and is_native_family(family) and max(upscaled.size) > NATIVE_MAX_REFINEMENT_EDGE:
        raise ValueError(f"Native refinement dimensions cannot exceed {NATIVE_MAX_REFINEMENT_EDGE}")
    # Denoise 0 asks for the enlargement without the redraw, so the unit is finished here: there is
    # no sampler to configure, no seed to spend and no schedule to build.
    if not settings.redraws:
        result = upscaled if latent_resample else materialize_image(upscaled).convert("RGB")
        if latent_resample and latent_decode_size != target_size:
            diagnostics.update({
                "latent_output_resize_applied": True,
                "pre_output_resize_size": list(latent_decode_size),
            })
        if capture.enabled:
            capture.stage("base", materialize_image(image))
            capture.stage("post_sr", materialize_image(result))
        diagnostics.update(resample_stage_facts(
            settings, unit_number, base_seed, None,
            source_size=resample_source_size, target_size=target_size,
            executed_size=result.size, steps=None, cfg=None,
            sampler_resolved=None, scheduler_resolved=None, execution_mode="resize_only",
        ))
        if upscale_warning:
            diagnostics["warning"] = upscale_warning
        return result, diagnostics
    if capture.enabled:
        capture.stage("base", materialize_image(image))
        capture.stage("post_sr", materialize_image(upscaled))
    control.checkpoint(job_id, f"{label} diffusion refinement")
    phase = (
        f"{label} · {LATENT_UPSCALE_MODES[settings.latent_algorithm]['label']} diffusion "
        f"{destination[0]} x {destination[1]}"
        if latent_resample
        else f"{label} · Diffusion {destination[0]} x {destination[1]}"
    )
    resolved_sampler = settings.sampler or request.sampler
    resolved_scheduler = settings.scheduler or request.scheduler
    if not is_native_family(family):
        validate_resample_sampling_override(resolved_sampler, resolved_scheduler, family, unit_number)
    hires_seed = effective_seed
    if is_native_family(family):
        if supports_usdu_tiled(family) and settings.execution_mode == "usdu_tiled":
            core_tile = (
                resample_source_size[0] if settings.tile_width == "auto" else settings.tile_width,
                resample_source_size[1] if settings.tile_height == "auto" else settings.tile_height,
            )
            plan = plan_tiles(
                upscaled.size, core_tile, padding=settings.padding,
                seam_mode=settings.seam_mode, alignment=usdu_pixel_alignment(family),
            )
            tile_count = len(plan.regions)
            stage_total = tile_count * unit_steps
            plan_native_stage(
                f"hires_sampling_{unit_number}", family, tuple(plan.processing_size), unit_cfg, 1, pipeline, job_id,
            )
            update_job(
                job_id,
                phase=f"{label} · Tile 1/{tile_count}",
                stage="resample_sampling", stage_step=0, stage_total=stage_total,
                stage_unit_index=1, stage_unit_total=tile_count,
                stage_unit_step=0, stage_unit_steps=unit_steps,
                progress=upscale_end, step=0, total_steps=stage_total,
            )
            previous_metrics = dict(getattr(pipeline, "last_generation_metrics", {}) or {})
            preparation_started = time.monotonic()
            prepared_conditioning = pipeline.prepare_refinement_conditioning(
                request.prompt, request.negative_prompt, unit_cfg, request.guidance
            )
            prepared_sigmas = pipeline.prepare_refinement_sigmas(unit_steps, settings.denoise, resolved_scheduler)
            preparation_metrics = serializable_runtime_metrics(
                (getattr(pipeline, "last_generation_metrics", {}) or {}).get("refinement.prompt_encode")
            )
            preparation_elapsed = round(time.monotonic() - preparation_started, 4)
            compositor = TileCompositor(upscaled.convert("RGB"), plan, mask_blur=settings.mask_blur)
            tile_metrics = []
            for tile_index, region in enumerate(plan.regions):
                control.checkpoint(job_id, f"{label} tile {tile_index + 1}/{tile_count}")
                tile = prepare_tile(upscaled, region)
                generator = torch.Generator(device="cpu").manual_seed(hires_seed)
                tile_started = time.monotonic()

                def on_anima_tile_step(step, total, _latents, index=tile_index):
                    stage_step = index * unit_steps + step
                    progress = upscale_end + round((progress_end - upscale_end) * stage_step / max(1, stage_total))
                    update_job(
                        job_id, phase=f"{label} · Tile {index + 1}/{tile_count}", stage="resample_sampling",
                        stage_step=stage_step,
                        # The tile is the unit a reader waits on, so it is reported as one rather
                        # than left to be recovered from the phase text: the console gives each
                        # tile a line of its own and times it separately.
                        stage_unit_index=index + 1, stage_unit_total=tile_count,
                        stage_unit_step=step, stage_unit_steps=total,
                        stage_total=stage_total, progress=min(progress, progress_end), step=stage_step,
                        total_steps=stage_total, elapsed_seconds=round(control.active_elapsed(started_at), 1),
                        paused_seconds=round(control.total_paused(), 1),
                    )
                def on_anima_tile_checkpoint(_step, _total, _latents, index=tile_index):
                    control.checkpoint(job_id, f"{label} tile {index + 1}/{tile_count}")

                refined = pipeline.refine_batch(
                    images=[tile], prompt=request.prompt, negative_prompt=request.negative_prompt,
                    steps=unit_steps, denoise=settings.denoise, cfg=unit_cfg,
                    sampler=resolved_sampler, scheduler=resolved_scheduler, generators=[generator],
                    guidance=request.guidance, pag_scale=request.pag.scale, pag_applied_layers=request.pag.applied_layers,
                    masks=None, on_step=on_anima_tile_step, prepared_conditioning=prepared_conditioning,
                    prepared_sigmas=prepared_sigmas, force_tiled_decode=True,
                    on_step_checkpoint=on_anima_tile_checkpoint,
                )
                if len(refined) != 1:
                    raise RuntimeError(f"USDU resample tile returned {len(refined)} images instead of 1")
                metrics = serializable_runtime_metrics(getattr(pipeline, "last_generation_metrics", {}) or {})
                sampling_metrics = metrics.get("refinement.sampling")
                decode_metrics = metrics.get("refinement.vae_decode")
                capture.tile(tile_index, refined[0])
                compositor.composite(refined[0], region)
                # Snapshotting the canvas costs a full RGBA->RGB copy, so never build it when capture is off.
                if capture.enabled:
                    capture.composite(tile_index, compositor.finish())
                control.checkpoint(job_id, f"{label} tile {tile_index + 1}/{tile_count} composite")
                row, col = divmod(tile_index, plan.cols)
                tile_metrics.append({
                    "index": tile_index, "row": row, "col": col, "hires_seed": str(hires_seed),
                    "elapsed_seconds": round(time.monotonic() - tile_started, 4),
                    "schedule_requested": {"sampler": resolved_sampler, "scheduler": resolved_scheduler, "steps": unit_steps},
                    "schedule_executed": {key: sampling_metrics.get(key) if isinstance(sampling_metrics, dict) else None
                                          for key in ("schedule_steps", "schedule_mode", "start_sigma")},
                    "conditioning_reused": sampling_metrics.get("conditioning_reused") if isinstance(sampling_metrics, dict) else None,
                    "sigmas_reused": sampling_metrics.get("sigmas_reused") if isinstance(sampling_metrics, dict) else None,
                    "latent_state_dtype": sampling_metrics.get("latent_state_dtype") if isinstance(sampling_metrics, dict) else None,
                    "transformer_input_dtype": sampling_metrics.get("transformer_input_dtype") if isinstance(sampling_metrics, dict) else None,
                    "tiled_decode_requested": decode_metrics.get("requested_tiled_decode") if isinstance(decode_metrics, dict) else {"tile": 512, "overlap": 64},
                    "tiled_decode_resolved": decode_metrics.get("resolved_tiled_decode") if isinstance(decode_metrics, dict) else None,
                    "actual_vae_mode": decode_metrics.get("actual_vae_mode") if isinstance(decode_metrics, dict) else "unknown",
                    "vae_encode_metrics": metrics.get("refinement.vae_encode"),
                    "sampling_metrics": sampling_metrics,
                    "vae_decode_metrics": decode_metrics,
                })
            merge_runtime_metrics(job_id, getattr(pipeline, "last_generation_metrics", None), "hires", previous_metrics)
            refinement_sampling = (getattr(pipeline, "last_generation_metrics", {}) or {}).get("refinement.sampling", {})
            refinement_decode = (getattr(pipeline, "last_generation_metrics", {}) or {}).get("refinement.vae_decode", {})
            diagnostics.update(resample_stage_facts(
                settings, unit_number, base_seed, hires_seed,
                source_size=resample_source_size, target_size=target_size, executed_size=destination,
                steps=unit_steps, cfg=unit_cfg, sampler_resolved=resolved_sampler,
                scheduler_resolved=resolved_scheduler, execution_mode="usdu_tiled",
            ))
            diagnostics.update({
                "core_tile": core_tile, "model_size": plan.processing_size, "rows": plan.rows,
                "cols": plan.cols, "tile_count": tile_count, "padding": settings.padding,
                "mask_blur": settings.mask_blur, "seam_mode": settings.seam_mode,
                "uniform_tiles": settings.uniform_tiles, "tiled_decode_requested": {"tile": 512, "overlap": 64},
                "tiled_decode_resolved": refinement_decode.get("resolved_tiled_decode", {"mode": "unknown"}),
                "steps_per_tile": unit_steps, "total_steps": stage_total,
                "conditioning_reused": True, "sigmas_reused": True, "tile_metrics": tile_metrics,
                "preparation_metrics": {"prompt_encode": preparation_metrics, "elapsed_seconds": preparation_elapsed},
                "aggregate": {
                    "tile_elapsed_seconds": round(sum(item["elapsed_seconds"] for item in tile_metrics), 4),
                    "tile_count": tile_count,
                    "conditioning_reused_count": sum(item["conditioning_reused"] is True for item in tile_metrics),
                    "sigmas_reused_count": sum(item["sigmas_reused"] is True for item in tile_metrics),
                    **aggregate_usdu_sampling_metrics(tile_metrics),
                },
                "latent_state_dtype": refinement_sampling.get("latent_state_dtype", "unknown"),
                "transformer_input_dtype": refinement_sampling.get("transformer_input_dtype", "unknown"),
            })
            for key in ("schedule_steps", "schedule_mode", "start_sigma"):
                if key in refinement_sampling:
                    diagnostics[key] = refinement_sampling[key]
            if upscale_warning:
                diagnostics["warning"] = upscale_warning
            return exact_resample_output(compositor.finish(), settings, target_size, diagnostics, control, job_id, label), diagnostics

        generator = torch.Generator(device="cpu").manual_seed(hires_seed)
        update_job(
            job_id,
            phase=phase,
            stage="resample_sampling",
            stage_step=0,
            stage_total=effective_steps,
            progress=upscale_end,
            step=0,
            total_steps=effective_steps,
        )

        def on_anima_hires_step(step, total, _latents):
            progress = upscale_end + round((progress_end - upscale_end) * step / max(1, total))
            update_job(
                job_id,
                phase=phase,
                stage="resample_sampling",
                stage_step=min(step, total),
                stage_total=total,
                progress=min(progress, progress_end),
                step=min(step, total),
                total_steps=total,
                elapsed_seconds=round(control.active_elapsed(started_at), 1),
                paused_seconds=round(control.total_paused(), 1),
            )
        def on_anima_hires_checkpoint(_step, _total, _latents):
            control.checkpoint(job_id, phase)

        plan_native_stage(
            f"hires_sampling_{unit_number}", family, tuple(upscaled.size), unit_cfg, 1, pipeline, job_id,
        )
        previous_metrics = dict(getattr(pipeline, "last_generation_metrics", {}) or {})
        refined = pipeline.refine_batch(
            images=[upscaled.convert("RGB")],
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=unit_steps,
            denoise=settings.denoise,
            cfg=unit_cfg,
            sampler=resolved_sampler,
            scheduler=resolved_scheduler,
            generators=[generator],
            guidance=request.guidance,
            pag_scale=request.pag.scale,
            pag_applied_layers=request.pag.applied_layers,
            masks=None,
            on_step=on_anima_hires_step,
            on_step_checkpoint=on_anima_hires_checkpoint,
        )
        if len(refined) != 1:
            raise RuntimeError(f"Native resample refinement returned {len(refined)} images instead of 1")
        merge_runtime_metrics(
            job_id, getattr(pipeline, "last_generation_metrics", None), "hires", previous_metrics
        )
        diagnostics.update(resample_stage_facts(
            settings, unit_number, base_seed, hires_seed,
            source_size=resample_source_size, target_size=target_size, executed_size=refined[0].size,
            steps=unit_steps, cfg=unit_cfg, sampler_resolved=resolved_sampler,
            scheduler_resolved=resolved_scheduler, execution_mode="full_frame",
            effective_steps=effective_steps,
        ))
        refinement_sampling = (getattr(pipeline, "last_generation_metrics", {}) or {}).get("refinement.sampling", {})
        for key in ("schedule_steps", "schedule_mode", "start_sigma"):
            if key in refinement_sampling:
                diagnostics[key] = refinement_sampling[key]
        if upscale_warning:
            diagnostics["warning"] = upscale_warning
        return exact_resample_output(refined[0].convert("RGB"), settings, target_size, diagnostics, control, job_id, label), diagnostics

    # The redraw is its own stage on its own canvas. By now the base decode has given its memory
    # back and the upscaler has been parked, so where the diffusion model should live is a different
    # question from the one the base pass answered — and on a 1152x1920 redraw it is the question
    # that used to be answered once, for the whole chain, by whichever stage needed the most.
    plan_residency_for_stage(
        f"hires_sampling_{unit_number}", family, tuple(destination), unit_cfg, 1, job_id,
        decode_in_call=not deferred_decode,
    )
    image_pipeline, guidance_kwargs = derived_sampling_pipeline(
        pipeline, AutoPipelineForImage2Image, request.guidance, request.pag
    )
    if settings.sampler is not None or settings.scheduler is not None:
        configure_scheduler(image_pipeline, resolved_sampler, resolved_scheduler)
    # A redraw after a base pass continues the base pass's schedule timeline, as Forge's
    # Hires fix does; `pipeline` still carries the base pass's scheduler.
    base_timeline = None if request.postprocess_only else sampling_timeline(pipeline, request.steps)
    conditioning = stage_prompt_conditioning(
        image_pipeline, family, request.prompt, request.negative_prompt,
        timeline=sampling_timeline(image_pipeline, unit_steps, continues=base_timeline),
        clip_skip=effective_clip_skip(request),
    )
    last_hires_step = [0]

    def on_hires_step(_pipeline, step_index, _timestep, callback_kwargs):
        step = logical_diffusion_step(_pipeline, step_index, effective_steps)
        last_hires_step[0] = max(last_hires_step[0], step)
        progress = upscale_end + round((sampling_end - upscale_end) * step / effective_steps)
        update_job(
            job_id,
            phase=phase,
            stage="resample_sampling",
            stage_step=step,
            stage_total=effective_steps,
            progress=min(progress, progress_end),
            step=step,
            total_steps=effective_steps,
            elapsed_seconds=round(control.active_elapsed(started_at), 1),
            paused_seconds=round(control.total_paused(), 1),
        )
        control.checkpoint(job_id, phase)
        return callback_kwargs

    update_job(job_id, phase=phase, stage="resample_sampling", stage_step=0, stage_total=effective_steps, progress=upscale_end, step=0, total_steps=effective_steps)
    try:
        pipeline_input = upscaled.latents if isinstance(upscaled, LatentImageArtifact) else upscaled
        refinement_generator = latent_generator or torch.Generator(device="cpu").manual_seed(hires_seed)
        pipeline_output = run_guided_pipeline(
            pipeline,
            image_pipeline,
            {
                "image": pipeline_input,
                "strength": settings.denoise,
                "num_inference_steps": schedule_steps,
                "guidance_scale": unit_cfg,
                "generator": refinement_generator,
                "callback_on_step_end": on_hires_step,
                "callback_on_step_end_tensor_inputs": ["latents"],
                "output_type": "latent" if deferred_decode else "pil",
                **conditioning,
            },
            guidance_kwargs,
            conditioning,
        ).images
        if latent_resample:
            result = make_latent_image_artifact(
                pipeline_output,
                destination,
                pipeline,
                family,
                job_id,
                control,
                staged_decode=staged_decode,
                start_progress=sampling_end,
                final_progress=progress_end,
                phase_prefix=label,
            )
        elif staged_decode:
            pipeline_output = pipeline_output.to("cpu")
            decoded = decode_staged_latents(
                pipeline,
                pipeline_output,
                family,
                job_id,
                control,
                active_vae_mode == "tiled",
                start_progress=sampling_end,
                final_progress=progress_end,
                decision_stage=f"hires_decode_{unit_number}",
            )
            result = decoded[0].convert("RGB")
            del decoded, pipeline_output
        else:
            result = pipeline_output[0].convert("RGB")
            record_decode_decision(
                f"hires_decode_{unit_number}", family, tuple(destination), evicted=False, job_id=job_id,
                reason="decode_in_sampling_call",
                peak_bytes=component_decode_bytes(pipeline, destination[0], destination[1]),
            )
        diagnostics.update(resample_stage_facts(
            settings, unit_number, base_seed, hires_seed,
            source_size=resample_source_size, target_size=target_size, executed_size=result.size,
            steps=unit_steps, cfg=unit_cfg, sampler_resolved=resolved_sampler,
            scheduler_resolved=resolved_scheduler, execution_mode="full_frame",
            effective_steps=effective_steps,
        ))
        if upscale_warning:
            diagnostics["warning"] = upscale_warning
        return exact_resample_output(result, settings, target_size, diagnostics, control, job_id, label), diagnostics
    finally:
        try:
            image_pipeline.maybe_free_model_hooks()
        except Exception:
            pass
        del image_pipeline


def anima_detail_crop_box(mask: Image.Image, padding: int):
    box = mask.getbbox()
    if not box:
        return None
    image_width, image_height = mask.size
    left = max(0, box[0] - padding)
    top = max(0, box[1] - padding)
    right = min(image_width, box[2] + padding)
    bottom = min(image_height, box[3] + padding)
    side = max(right - left, bottom - top)
    crop_width = min(image_width, side)
    crop_height = min(image_height, side)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    left = max(0, min(image_width - crop_width, round(center_x - crop_width / 2)))
    top = max(0, min(image_height - crop_height, round(center_y - crop_height / 2)))
    return (left, top, left + crop_width, top + crop_height)


def anima_detail_processing_size(crop_box):
    width = crop_box[2] - crop_box[0]
    height = crop_box[3] - crop_box[1]
    scale = 1024 / max(width, height)
    return tuple(
        max(32, min(1024, round(dimension * scale / 32) * 32))
        for dimension in (width, height)
    )


def adetailer_effective_steps(settings: ADetailerUnitInput, request: GenerateInput, family: str) -> int:
    """How many denoise updates one detail pass actually performs.

    Every family now executes the requested count. Native refinement already builds a Comfy-style
    suffix; SD/iL receive an expanded Diffusers schedule in `apply_adetailer_unit` so strength selects
    the starting noise without shrinking this count. The family argument stays in the stable helper
    signature used by the accounting call sites.
    """
    detail_steps = settings.steps if settings.use_steps else request.steps
    return detail_steps


def apply_adetailer_unit(
    image: Image.Image,
    pipeline,
    family: str,
    request: GenerateInput,
    settings: ADetailerUnitInput,
    job_id: str,
    control: JobControl,
    started_at: float,
    image_seed=None,
    progress_start=94,
    progress_end=98,
    unit_index=0,
    unit_total=1,
):
    """One detail pass. `apply_adetailer` below runs the configured units in order."""

    label = f"ADetailer {unit_index + 1}/{unit_total}" if unit_total > 1 else "ADetailer"
    detail_steps = settings.steps if settings.use_steps else request.steps
    detail_cfg = settings.cfg if settings.use_cfg else request.cfg
    effective_steps = adetailer_effective_steps(settings, request, family)
    schedule_steps = (
        detail_steps
        if is_native_family(family)
        else diffusers_schedule_total(detail_steps, settings.denoise)
    )
    # Where the job-wide step counter starts. The base pass is behind us, so its steps are counted as
    # done — but only the ones it actually performed: post-processing mode runs none, and Diffusers
    # image-to-image runs `int(steps * strength)` rather than the requested count.
    base_steps = base_sampling_steps(request, family)

    detector = resolve_detector_model(ADETAILER_MODEL_DIRECTORY, settings.detector)
    control.checkpoint(job_id, "Detecting details")
    update_job(
        job_id,
        phase=f"{label} · Detecting",
        stage="adetailer_detect",
        stage_step=0,
        stage_total=0,
        progress=progress_start,
        step=base_steps,
        total_steps=base_steps,
        adetailer_state={
            "status": "detecting",
            "detector": detector.name,
            "source_size": list(image.size),
            "detected_count": 0,
            "selected_count": 0,
            "region_index": 0,
            "region_total": 0,
            "detections": [],
        },
    )
    raw_detections = run_detector(ADETAILER_PYTHON, ADETAILER_WORKER, detector, image, settings.confidence)
    control.checkpoint(job_id, "Preparing detail masks")
    detections = select_detections(
        raw_detections,
        image.size,
        settings.mask_min_ratio,
        settings.mask_max_ratio,
        settings.max_detections,
    )
    detection_states = [
        {
            "id": index,
            "box": list(detection.box),
            "crop_box": None,
            "confidence": round(detection.confidence, 4),
            "class_name": detection.class_name,
            "status": "pending",
        }
        for index, detection in enumerate(detections)
    ]
    detail_state = {
        "status": "detected" if detections else "no_detections",
        "detector": detector.name,
        "source_size": list(image.size),
        "detected_count": len(raw_detections),
        "selected_count": len(detections),
        "region_index": 0,
        "region_total": len(detections),
        "detections": detection_states,
    }
    diagnostics = {
        "detector": detector.name,
        "source_size": list(image.size),
        "confidence": settings.confidence,
        "detections": [
            {
                "box": list(detection.box),
                "confidence": round(detection.confidence, 4),
                "class": detection.class_name,
            }
            for detection in detections
        ],
        "denoise": settings.denoise,
        "steps": detail_steps,
        "cfg": detail_cfg,
        "dilate_erode": settings.dilate_erode,
        "mask_blur": settings.mask_blur,
        "padding": settings.padding,
    }
    update_job(
        job_id,
        stage="adetailer_detect",
        stage_step=len(detections),
        stage_total=len(detections),
        adetailer_state=detail_state,
    )
    if not detections:
        return image, diagnostics, f"{label} did not detect any matching regions"

    detail_prompt = expand_prompt(settings.prompt, request.prompt)
    detail_negative_prompt = expand_prompt(settings.negative_prompt, request.negative_prompt)
    total_steps = base_steps + len(detections) * effective_steps
    current_image = image.convert("RGB")
    update_job(job_id, step=base_steps, total_steps=total_steps)
    if is_native_family(family):
        for index, detection in enumerate(detections):
            control.checkpoint(job_id, f"Preparing ADetailer region {index + 1}")
            hard_mask, soft_mask = detection_mask(
                detection,
                current_image.size,
                settings.dilate_erode,
                settings.mask_blur,
            )
            crop_box = anima_detail_crop_box(hard_mask, settings.padding)
            if crop_box is None:
                detection_states[index]["status"] = "skipped"
                continue
            process_size = anima_detail_processing_size(crop_box)
            diagnostics["detections"][index]["crop_box"] = list(crop_box)
            diagnostics["detections"][index]["processing_size"] = list(process_size)
            detection_states[index].update(crop_box=list(crop_box), status="active")
            phase = f"{label} · Inpainting {index + 1}/{len(detections)}"
            update_job(
                job_id,
                phase=phase,
                stage="adetailer_inpaint",
                stage_step=0,
                stage_total=effective_steps,
                adetailer_state={
                    **detail_state,
                    "status": "inpainting",
                    "region_index": index + 1,
                    "detections": [dict(item) for item in detection_states],
                },
            )

            def on_anima_detail_step(detail_step, _total, _latents, *, detection_index=index, running_phase=phase):
                completed = base_steps + detection_index * effective_steps + detail_step
                progress = progress_start + round(
                    (completed - base_steps) / max(1, total_steps - base_steps)
                    * max(0, progress_end - progress_start)
                )
                update_job(
                    job_id,
                    phase=running_phase,
                    stage="adetailer_inpaint",
                    stage_step=min(detail_step, effective_steps),
                    stage_total=effective_steps,
                    progress=min(progress, progress_end),
                    step=completed,
                    total_steps=total_steps,
                    elapsed_seconds=round(control.active_elapsed(started_at), 1),
                    paused_seconds=round(control.total_paused(), 1),
                )
            def on_anima_detail_checkpoint(_step, _total, _latents, *, running_phase=phase):
                control.checkpoint(job_id, running_phase)

            crop = current_image.crop(crop_box).resize(process_size, Image.Resampling.LANCZOS)
            hard_crop = hard_mask.crop(crop_box).resize(process_size, Image.Resampling.NEAREST)
            region_seed = ((request.seed if image_seed is None else image_seed) + index + 1) & 0xFFFFFFFFFFFFFFFF
            # Stage 6: each region is placed for its own processing canvas; regions of one size
            # land on the same decision, so they do not move the model between them.
            plan_native_stage(
                f"adetailer_region_{unit_index + 1}_{index + 1}", family, tuple(process_size), detail_cfg, 1,
                pipeline, job_id,
            )
            previous_metrics = dict(getattr(pipeline, "last_generation_metrics", {}) or {})
            refined = pipeline.refine_batch(
                images=[crop.convert("RGB")],
                prompt=detail_prompt,
                negative_prompt=detail_negative_prompt,
                steps=detail_steps,
                denoise=settings.denoise,
                cfg=detail_cfg,
                sampler=request.sampler,
                scheduler=request.scheduler,
                generators=[torch.Generator(device="cpu").manual_seed(region_seed)],
                guidance=request.guidance,
                pag_scale=request.pag.scale,
                pag_applied_layers=request.pag.applied_layers,
                masks=[hard_crop],
                on_step=on_anima_detail_step,
                on_step_checkpoint=on_anima_detail_checkpoint,
            )
            if len(refined) != 1:
                raise RuntimeError(f"Anima ADetailer returned {len(refined)} images instead of 1")
            merge_runtime_metrics(
                job_id,
                getattr(pipeline, "last_generation_metrics", None),
                f"adetailer.region_{index + 1}",
                previous_metrics,
            )
            replacement = refined[0].convert("RGB").resize(
                (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]), Image.Resampling.LANCZOS
            )
            candidate = current_image.copy()
            candidate.paste(replacement, crop_box[:2])
            current_image = Image.composite(candidate, current_image, soft_mask)
            detection_states[index]["status"] = "complete"
            update_job(
                job_id,
                stage_step=effective_steps,
                adetailer_state={
                    **detail_state,
                    "status": "inpainting" if index + 1 < len(detections) else "complete",
                    "region_index": index + 1,
                    "detections": [dict(item) for item in detection_states],
                },
            )
        return current_image, diagnostics, None

    detail_pipeline, guidance_kwargs = derived_sampling_pipeline(
        pipeline, AutoPipelineForInpainting, request.guidance, request.pag
    )
    sample_size = detail_pipeline.unet.config.sample_size
    vae_scale = detail_pipeline.vae_scale_factor
    if isinstance(sample_size, (tuple, list)):
        detail_height = int(sample_size[0]) * vae_scale
        detail_width = int(sample_size[-1]) * vae_scale
    else:
        detail_width = detail_height = int(sample_size) * vae_scale

    def plan_detail_encode():
        # Only a unit with its own prompt reaches here; the others reuse the job's encoding.
        if stage_scheduling_active():
            plan_residency_for_stage(
                f"adetailer_encode_{unit_index + 1}", family, (detail_width, detail_height), detail_cfg, 1,
                job_id, kind="encode", inference_bytes=TEXT_ENCODE_TRANSIENT_BYTES, legacy_point=False,
            )

    conditioning = stage_prompt_conditioning(
        detail_pipeline, family, detail_prompt, detail_negative_prompt,
        timeline=sampling_timeline(detail_pipeline, detail_steps),
        clip_skip=effective_clip_skip(request),
        before_encode=plan_detail_encode,
    )

    try:
        for index, detection in enumerate(detections):
            hard_mask, soft_mask = detection_mask(
                detection,
                current_image.size,
                settings.dilate_erode,
                settings.mask_blur,
            )
            if not hard_mask.getbbox():
                detection_states[index]["status"] = "skipped"
                continue
            crop_box = detail_pipeline.mask_processor.get_crop_region(
                hard_mask,
                detail_width,
                detail_height,
                pad=settings.padding,
            )
            diagnostics["detections"][index]["crop_box"] = list(crop_box)
            detection_states[index].update(crop_box=list(crop_box), status="active")
            phase = f"{label} · Inpainting {index + 1}/{len(detections)}"
            control.checkpoint(job_id, phase)
            detail_last_step = [0]
            update_job(
                job_id,
                phase=phase,
                stage="adetailer_inpaint",
                stage_step=0,
                stage_total=effective_steps,
                adetailer_state={
                    **detail_state,
                    "status": "inpainting",
                    "region_index": index + 1,
                    "detections": [dict(item) for item in detection_states],
                },
            )

            def on_detail_step(_pipeline, step_index, _timestep, callback_kwargs, *, detection_index=index, running_phase=phase):
                detail_step = logical_diffusion_step(_pipeline, step_index, effective_steps)
                detail_last_step[0] = max(detail_last_step[0], detail_step)
                completed = base_steps + detection_index * effective_steps + detail_step
                progress = progress_start + round(
                    (completed - base_steps) / max(1, total_steps - base_steps)
                    * max(0, progress_end - progress_start)
                )
                update_job(
                    job_id,
                    phase=running_phase,
                    stage="adetailer_inpaint",
                    stage_step=detail_step,
                    stage_total=effective_steps,
                    progress=min(progress, progress_end),
                    step=completed,
                    total_steps=total_steps,
                    elapsed_seconds=round(control.active_elapsed(started_at), 1),
                    paused_seconds=round(control.total_paused(), 1),
                )
                control.checkpoint(job_id, running_phase)
                return callback_kwargs

            generator = torch.Generator(device="cpu").manual_seed(
                ((request.seed if image_seed is None else image_seed) + index + 1) & 0xFFFFFFFFFFFFFFFF
            )
            kwargs = {
                "image": current_image,
                "mask_image": hard_mask,
                "strength": settings.denoise,
                "num_inference_steps": schedule_steps,
                "guidance_scale": detail_cfg,
                "generator": generator,
                "callback_on_step_end": on_detail_step,
                "callback_on_step_end_tensor_inputs": ["latents"],
            }
            kwargs["padding_mask_crop"] = settings.padding
            kwargs.update(conditioning)
            if stage_scheduling_active():
                # Stage 6: the inpainting call samples and decodes its crop canvas before it returns.
                plan_residency_for_stage(
                    f"adetailer_region_{unit_index + 1}_{index + 1}", family, (detail_width, detail_height),
                    detail_cfg, 1, job_id, decode_in_call=True, legacy_point=False,
                )
            detailed = run_guided_pipeline(
                pipeline, detail_pipeline, kwargs, guidance_kwargs, conditioning
            ).images[0].convert("RGB")
            current_image = Image.composite(detailed, current_image, soft_mask)
            detection_states[index]["status"] = "complete"
            update_job(
                job_id,
                stage_step=effective_steps,
                adetailer_state={
                    **detail_state,
                    "status": "inpainting" if index + 1 < len(detections) else "complete",
                    "region_index": index + 1,
                    "detections": [dict(item) for item in detection_states],
                },
            )
        return current_image, diagnostics, None
    finally:
        try:
            detail_pipeline.maybe_free_model_hooks()
        except Exception:
            pass
        del detail_pipeline


def apply_adetailer(
    image: Image.Image,
    pipeline,
    family: str,
    request: GenerateInput,
    job_id: str,
    control: JobControl,
    started_at: float,
    image_seed=None,
    progress_start=94,
    progress_end=98,
):
    """Run every configured detail pass in order, each over the last one's output.

    A face model and a hand model repair different things and want different
    settings, so the stage is a list rather than one configuration. The passes
    are sequential by nature: a unit is given the picture the previous unit
    produced, because repairing a face inside a region a later pass will replace
    would be work thrown away.
    """

    units = request.adetailer.active_units
    if not units:
        return image, None, None

    span = max(0, progress_end - progress_start)
    current = image
    diagnostics: list[dict] = []
    warnings: list[str] = []
    for index, unit in enumerate(units):
        control.checkpoint(job_id, f"Preparing ADetailer pass {index + 1}")
        # Each unit owns its slice of the stage's progress band, so a two-unit
        # stage does not run the bar to the end twice.
        current, unit_diagnostics, warning = apply_adetailer_unit(
            current,
            pipeline,
            family,
            request,
            unit,
            job_id,
            control,
            started_at,
            image_seed=image_seed,
            progress_start=progress_start + round(span * index / len(units)),
            progress_end=progress_start + round(span * (index + 1) / len(units)),
            unit_index=index,
            unit_total=len(units),
        )
        if unit_diagnostics:
            diagnostics.append({"unit": index + 1, **unit_diagnostics})
        if warning:
            warnings.append(warning)

    # A unit that found nothing is a warning, not a failure: the passes after it
    # still ran, and the picture it passed along is the one they worked on.
    warning = "；".join(warnings) if warnings else None
    if not diagnostics:
        return current, None, warning
    return current, {
        "units": diagnostics,
        # What the older single-pass record exposed, kept so a gallery card and a
        # PNG parameter block still read a detector name off the stage itself.
        "detector": " · ".join(dict.fromkeys(item["detector"] for item in diagnostics)),
    }, warning


def remove_preview(path):
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


STAGE_PREVIEW_LABELS = {
    "base": "基础采样",
    "resample": "重采样",
    "adetailer": "ADetailer",
    "rtx": "RTX VSR",
    "background_removal": "透明背景",
}


def save_stage_preview(image: Image.Image, job_id: str, stage: str, index: int) -> Path:
    """Write one finished stage's picture where it will outlive the run."""
    preview = image.convert("RGB")
    scale = min(STAGE_PREVIEW_MAX_EDGE / max(preview.width, preview.height), 1.0)
    if scale < 1.0:
        preview = preview.resize(
            (max(1, round(preview.width * scale)), max(1, round(preview.height * scale))),
            Image.Resampling.LANCZOS,
        )
    stage_path = STAGE_PREVIEW_DIRECTORY / f"{job_id}-{index:02d}-{stage}-{uuid.uuid4().hex}.jpg"
    temporary_path = STAGE_PREVIEW_DIRECTORY / f"{job_id}.{uuid.uuid4().hex}.tmp"
    try:
        preview.save(temporary_path, format="JPEG", quality=90)
        os.replace(temporary_path, stage_path)
    finally:
        Path(temporary_path).unlink(missing_ok=True)
    return stage_path


def cleanup_job_stage_previews(job_id: str):
    for stage_path in STAGE_PREVIEW_DIRECTORY.glob(f"{job_id}-*"):
        if stage_path.is_file():
            remove_preview(stage_path)


def output_relative_path(path: Path) -> str:
    """Where a picture sits beneath the output root, as the only location a record may carry."""
    try:
        return path.resolve().relative_to(OUTPUT_DIRECTORY.resolve()).as_posix()
    except ValueError:
        return path.name


def output_seed(request: GenerateInput, batch_index: int, image_index: int):
    offset = batch_index * request.images_per_batch + image_index
    return (request.seed + offset) & 0xFFFFFFFFFFFFFFFF


def batch_seeds(request: GenerateInput, batch_index: int):
    return [output_seed(request, batch_index, image_index) for image_index in range(request.images_per_batch)]


def secure_random_uint64():
    """Return one OS-backed unsigned 64-bit value without shared PRNG state."""
    return secrets.randbits(64)


def resolve_resample_unit_seed(settings: ResampleUnitInput, base_seed: int, random_uint64=None):
    """Resolve one unit's seed for one output, after that output's base seed is known.

    A unit that does not redraw spends no seed, and says so with `None` rather than carrying one
    that never reaches a generator.
    """
    if not settings.redraws:
        return None
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or not 0 <= base_seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Base seed must be an unsigned 64-bit integer")
    if settings.seed_mode == "inherit":
        return base_seed
    if settings.seed_mode == "fixed":
        return int(settings.seed, 10)
    candidate = (random_uint64 or secure_random_uint64)()
    if isinstance(candidate, bool) or not isinstance(candidate, int) or not 0 <= candidate <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Random resample seed source returned an invalid unsigned 64-bit integer")
    return candidate


def resolve_chain_seeds(request: GenerateInput, base_seed: int, random_uint64=None):
    """Every enabled unit's seed for one output, keyed by its position in the chain.

    Resolved once per picture, before the chain runs: `random` must draw a fresh value per output
    rather than per unit invocation, and a unit that is reached twice in a retry must not move.
    """
    return {
        index: resolve_resample_unit_seed(stage, base_seed, random_uint64)
        for index, stage in enumerate(request.postprocess_chain)
        if stage.type == "resample" and stage.enabled
    }


def output_seed_facts(request: GenerateInput, base_seed: int, chain_seeds=None):
    """The seed block every output record carries.

    `hires_seed` and `hires_seed_mode` describe the first redrawing unit. They are the names the
    gallery restores a run from, and with one unit — which is what every chain migrated from the old
    Hires block has — they mean exactly what they always did.
    """
    seeds = chain_seeds or {}
    units = []
    for index, stage in enumerate(request.postprocess_chain):
        if stage.type != "resample" or not stage.enabled:
            continue
        seed_value = seeds.get(index)
        units.append({
            "unit": request.resample_unit_number(stage),
            "seed_mode": stage.seed_mode,
            "seed": str(seed_value) if seed_value is not None else None,
        })
    leading = next((unit for unit in units if unit["seed"] is not None), None)
    return {
        "seed": str(base_seed),
        "base_seed": str(base_seed),
        "hires_seed_mode": leading["seed_mode"] if leading else "inherit",
        "hires_seed": leading["seed"] if leading else None,
        "resample_seeds": units,
    }


def batch_pipeline_kwargs(request: GenerateInput, generators, on_step_end, conditioning):
    if len(generators) != request.images_per_batch:
        raise ValueError("Generator count must match images_per_batch")
    return {
        "width": request.width,
        "height": request.height,
        "num_inference_steps": request.steps,
        "num_images_per_prompt": request.images_per_batch,
        "guidance_scale": request.cfg,
        "generator": generators,
        "callback_on_step_end": on_step_end,
        "callback_on_step_end_tensor_inputs": ["latents"],
        **conditioning,
    }


def run_pipeline_batch(pipeline, pipeline_kwargs):
    outputs = run_with_offload_escalation(
        pipeline, pipeline_kwargs, lambda: pipeline(**pipeline_kwargs)
    ).images
    expected = pipeline_kwargs["num_images_per_prompt"]
    if len(outputs) != expected:
        raise RuntimeError(f"Pipeline returned {len(outputs)} images for a requested batch of {expected}")
    return outputs


# How much of a workflow failure reaches the job's warning line.
WORKFLOW_WARNING_LIMIT = 200


def comfy_workflow_facts(request: GenerateInput, *, seed, resample_results, identity, background_removal=False) -> WorkflowFacts:
    """What one picture executed, in the shape `build_workflow` reads.

    `request` is the runtime request: its prompt is the text the encoders were given and its LoRAs
    include the ones inline tags resolved to. Each resample unit's seed, steps and sizes come from
    the record the unit wrote while it ran, so a random seed is the one it actually drew.
    """
    family = request_family(request.engine)
    executed = {
        record["unit"]: record
        for record in resample_results or []
        if isinstance(record, dict) and isinstance(record.get("unit"), int)
    }
    stages = []
    for stage in request.postprocess_chain:
        if not stage.enabled:
            continue
        if stage.type != "resample":
            stages.append(stage.type)
            continue
        number = request.resample_unit_number(stage)
        record = executed.get(number)
        if record is None:
            raise RuntimeError(f"resample unit {number} left no execution record")
        source = tuple(int(edge) for edge in record["source_size"])
        sizes = resample_unit_sizes(source, stage, family)
        recorded_seed = record.get("seed")
        stages.append(ResampleStage(
            number=number,
            redraws=stage.redraws,
            tiled=stage.execution_mode == "usdu_tiled" and supports_usdu_tiled(family),
            upscaler=stage.upscaler,
            latent_algorithm=stage.latent_algorithm,
            resize_algorithm=stage.resize_algorithm,
            denoise=stage.denoise,
            steps=record.get("steps") if stage.redraws else None,
            cfg=record.get("cfg") if stage.redraws else None,
            sampler=record.get("sampler_resolved") or stage.sampler or request.sampler,
            scheduler=record.get("scheduler_resolved") or stage.scheduler or request.scheduler,
            seed=int(recorded_seed) if recorded_seed is not None else (int(seed) if stage.redraws else None),
            source_size=source,
            target_size=tuple(sizes["target"]),
            redraw_size=tuple(sizes["redraw"]),
            output_size=tuple(int(edge) for edge in (record.get("executed_size") or sizes["output"])),
        ))
    return WorkflowFacts(
        identity=identity,
        seed=int(seed),
        positive=request.prompt,
        negative=request.negative_prompt,
        loras=tuple((item.path, float(item.weight)) for item in request.loras),
        clip_skip=effective_clip_skip(request),
        stages=tuple(stages),
        background_removal=bool(background_removal),
    )


def embedded_workflow_text(request: GenerateInput, **facts):
    """The `workflow` chunk for one picture, and a warning; at most one of them is set.

    Never raises: a graph that cannot be built costs the chunk, never the picture.
    """
    try:
        workflow = build_workflow(request, comfy_workflow_facts(request, **facts))
    except Exception as error:
        traceback.print_exc()
        detail = f"{type(error).__name__}: {error}"
        return None, f"ComfyUI 工作流未嵌入：{detail[:WORKFLOW_WARNING_LIMIT]}"
    if workflow is None:
        return None, None
    # ASCII keeps the chunk a plain tEXt block that every ComfyUI frontend reads; JSON escapes carry
    # non-Latin prompt text through unchanged.
    return json.dumps(workflow, ensure_ascii=True, separators=(",", ":")), None


def save_image(
    image,
    job_id: str,
    request: GenerateInput,
    elapsed: float,
    wall_elapsed: float,
    paused_seconds: float,
    adetailer_result=None,
    resample_results=None,
    rtx_result=None,
    background_removal_result=None,
    *,
    conditioning_prompt=None,
    inline_loras=None,
    seed=None,
    chain_seeds=None,
    batch_index=1,
    image_index=1,
    before_commit=None,
    workflow_request=None,
    workflow_warnings=None,
):
    base_seed = request.seed if seed is None else seed
    resample_results = list(resample_results or [])
    if any(unit.redraws for unit in request.active_resample_units) and not chain_seeds:
        raise ValueError("Resample seeds must be resolved before saving an output that redraws")
    # The job's frozen template (or the historical scheme) names the folder and stem now; the file
    # name itself is only settled when the finished picture is published, exclusively.
    target = output_target(OUTPUT_DIRECTORY, request.output_template, moment=datetime.now(), engine=request.engine, create=True)
    temporary_path = temporary_output_path(target)
    metadata = PngInfo()
    metadata.add_text("prompt", request.prompt)
    metadata.add_text("negative_prompt", request.negative_prompt)
    parameters = {
        "job_id": job_id,
        "engine": request.engine,
        "checkpoint": Path(request.checkpoint).name if request.checkpoint else None,
        "width": request.width,
        "height": request.height,
        "output_width": image.width,
        "output_height": image.height,
        "steps": request.steps,
        "cfg": request.cfg,
        "denoise": request.denoise,
        "img2img_fix_steps": uses_diffusers_img2img_fix_steps(request),
        "img2img_schedule_steps": img2img_schedule_steps(request),
        **output_seed_facts(request, base_seed, chain_seeds),
        "images_per_batch": request.images_per_batch,
        "batch_count": request.batch_count,
        "batch_index": batch_index,
        "image_index": image_index,
        "sampler": request.sampler,
        "scheduler": request.scheduler,
        "guidance": guidance_diagnostics(
            request.guidance,
            request.steps,
            request.pag.scale,
            request.pag.applied_layers,
            engine=request.engine,
            guidance_scale=request.cfg,
        ),
        "background_removal_model": request.background_removal_model,
        # Facts about the source, never the source. A PNG parameter block is read back by the
        # gallery and the history list; embedding the picture would roughly double every output
        # file to store a copy of something the user already has.
        "source_image": {
            # `resize_mode` is recorded even in post-processing mode, where nothing resamples: the
            # block describes the request, and omitting a field there would read as "not sent".
            "resize_mode": request.source_image.resize_mode,
            "name": request.source_image.name or None,
            "denoise": request.denoise,
            "postprocess_only": request.postprocess_only,
        } if request.source_image.enabled else None,
        # The chain as it was configured, and what each stage actually did. The old single-block
        # `hires` key is still written when exactly one unit ran, so a reader from before the chain
        # — an older gallery card, someone's script — finds the record where it expects it.
        "postprocess_chain": [stage.model_dump() for stage in request.postprocess_chain],
        "resample": resample_results or None,
        "hires": resample_results[0] if len(resample_results) == 1 else None,
        "adetailer": adetailer_result if request.adetailer.enabled else None,
        "rtx": rtx_result if request.rtx.enabled else None,
        "postprocess_stages": postprocess_stage_types(request),
        "postprocess_only": request.postprocess_only,
        # `path` is the relative ID the job mounted (inside its engine's LoRA root, or a
        # `shared:<id>/...` reference), so a reader can match the exact file among same-named ones.
        "loras": [{"name": Path(item.path).name, "path": public_model_reference(item.path), "weight": item.weight} for item in request.loras],
        "elapsed_seconds": round(elapsed, 3),
        "wall_elapsed_seconds": round(wall_elapsed, 3),
        "paused_seconds": round(paused_seconds, 3),
    }
    clip_skip = effective_clip_skip(request)
    if clip_skip is not None:
        parameters["clip_skip"] = clip_skip
    if request.output_template is not None:
        parameters["output_template"] = request.output_template
    if inline_loras:
        parameters["inline_loras"] = copy.deepcopy(list(inline_loras))
    lora_fusion = getattr(loaded_pipeline, "_xiriai_lora_fusion_report", None)
    if request.engine in {"SD", "iL"} and isinstance(lora_fusion, dict):
        # The report contains only basenames, counts and a digest of the trusted
        # relative signature; never copy the private pipeline signature itself.
        parameters["lora_fusion"] = dict(lora_fusion)
    if request.engine == "Anima":
        parameters.update({
            "diffusion_model": public_model_reference(request.diffusion_model),
            "text_encoder": public_model_reference(request.text_encoder),
            "vae": public_model_reference(request.vae),
            "flow_shift": 3,
            "sampling": anima_sampling_diagnostics(request.sampler, request.scheduler),
        })
    elif request.engine == "Flux":
        # The shift is resolution-derived, so it belongs in the record: reproducing this image
        # needs the canvas the shift came from, not just the sampler name.
        flux_shift = flux_resolution_shift(request.width, request.height)
        parameters.update({
            "diffusion_model": public_model_reference(request.diffusion_model),
            "text_encoder": public_model_reference(request.text_encoder),
            "text_encoder_2": public_model_reference(request.text_encoder_2),
            "vae": public_model_reference(request.vae),
            "flow_shift": round(flux_shift, 4),
            "distilled_guidance": request.cfg,
            "sampling": flux_sampling_diagnostics(request.sampler, request.scheduler, flux_shift),
        })
    elif request.engine == "Flux2":
        # FLUX.2's shift is fitted from the canvas *and* the step count, so both belong in the
        # record: neither one alone reproduces the trajectory this image came from.
        flux2_shift = flux2_resolution_shift(request.width, request.height, request.steps)
        parameters.update({
            "diffusion_model": public_model_reference(request.diffusion_model),
            "text_encoder": public_model_reference(request.text_encoder),
            "vae": public_model_reference(request.vae),
            "flow_shift": round(flux2_shift, 4),
            "distilled_guidance": request.cfg,
            "sampling": flux2_sampling_diagnostics(request.sampler, request.scheduler, flux2_shift),
        })
    elif request.engine == "Krea2":
        # Krea 2's shift is the static one its model config declares, so the canvas does not
        # change it — but the canvas is still recorded through `sampling.sequence_length`, and the
        # CFG scale is a real guidance scale here rather than a distilled embedding.
        parameters.update({
            "diffusion_model": public_model_reference(request.diffusion_model),
            "text_encoder": public_model_reference(request.text_encoder),
            "vae": public_model_reference(request.vae),
            "flow_shift": round(KREA2_SHIFT, 4),
            "sampling": krea2_sampling_diagnostics(
                request.sampler, request.scheduler, request.width, request.height
            ),
        })
    if loaded_checkpoint_hash:
        parameters["checkpoint_sha256"] = loaded_checkpoint_hash
        if request.engine in NATIVE_ENGINES:
            parameters["diffusion_model_sha256"] = loaded_checkpoint_hash
        metadata.add_text("model_sha256", loaded_checkpoint_hash)
    if background_removal_result:
        parameters["transparent_background"] = background_removal_result
    if conditioning_prompt and conditioning_prompt != request.prompt:
        parameters["conditioning_prompt"] = conditioning_prompt
    if stage_scheduling_active():
        # A new key, written only under AUTO: which path each stage actually ran on. No existing
        # field changes meaning, and nothing machine-specific (bytes, free memory) is recorded.
        parameters["memory_schedule"] = png_schedule_note(active_stage_schedule)
    metadata.add_text("parameters", json.dumps(parameters, ensure_ascii=False))
    # The runtime request, when the caller has it, is what ran: inline tags resolved into LoRAs and
    # the prompt the encoders were given. The submitted one still describes the job above.
    workflow_text, workflow_warning = embedded_workflow_text(
        workflow_request or request,
        seed=base_seed,
        resample_results=resample_results,
        identity=f"{job_id}/{batch_index}/{image_index}",
        background_removal=bool(background_removal_result),
    )
    if workflow_text is not None:
        metadata.add_text("workflow", workflow_text)
    if workflow_warning and workflow_warnings is not None:
        workflow_warnings.append(workflow_warning)
    if background_removal_result and image.mode != "RGBA":
        raise RuntimeError("透明背景处理未生成 RGBA 图像，已停止保存")
    try:
        image.save(temporary_path, format="PNG", pnginfo=metadata)
        if before_commit:
            before_commit()
        # Exclusive publish: an existing picture is never replaced, even by a second process.
        output_path = publish_exclusive(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def cuda_memory_diagnostics():
    snapshot = {"available": torch.cuda.is_available()}
    if not snapshot["available"]:
        return snapshot
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        _budget_free, _physical_total, limit_bytes = effective_cuda_free_memory()
        snapshot.update(
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            effective_limit_bytes=limit_bytes,
            budget_free_bytes=_budget_free,
        )
    except Exception as error:
        snapshot["mem_get_info_error"] = str(error)
    # Two different scopes, reported side by side and named as such (QPlan P8 §1): the budget is the
    # device or the user's wall; the allocator cap is what PyTorch may actually hold right now. The
    # 12 GB report showed an 11.91 GiB budget beside a 6.62 GiB cap with nothing saying which was which.
    snapshot["budget_scope"] = (
        "vram_wall" if performance_settings.get("vram_limit_gb", 0.0) > 0 else "device"
    )
    if active_residency is not None:
        snapshot["allocator_cap_bytes"] = active_residency.allocator_cap_bytes
        snapshot["allocator_cap_scope"] = (
            "stage_cap" if active_residency.allocator_cap_bytes is not None else "configured_ceiling"
        )
    for name, reader in (
        ("allocated_bytes", torch.cuda.memory_allocated),
        ("reserved_bytes", torch.cuda.memory_reserved),
        ("peak_allocated_bytes", torch.cuda.max_memory_allocated),
        ("peak_reserved_bytes", torch.cuda.max_memory_reserved),
    ):
        try:
            snapshot[name] = reader()
        except Exception as error:
            snapshot[f"{name}_error"] = str(error)
    return snapshot


def cuda_device_name():
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception as error:
        return f"unavailable: {error}"


def system_memory_diagnostics():
    try:
        import psutil

        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "used_bytes": memory.used,
            "percent": memory.percent,
            "swap_total_bytes": swap.total,
            "swap_free_bytes": swap.free,
            "swap_used_bytes": swap.used,
            "swap_percent": swap.percent,
        }
    except Exception as error:
        return {"error": str(error)}


def pipeline_component_diagnostics():
    if loaded_pipeline is None:
        return {}
    components = {}
    try:
        pipeline_components = loaded_pipeline.components.items()
    except Exception as error:
        return {"error": str(error)}
    for name, component in pipeline_components:
        if not isinstance(component, torch.nn.Module):
            continue
        try:
            tensor = next(component.parameters(), None)
            if tensor is None:
                tensor = next(component.buffers(), None)
            hook = getattr(component, "_hf_hook", None)
            components[name] = {
                "device": str(tensor.device) if tensor is not None else None,
                "dtype": str(tensor.dtype).removeprefix("torch.") if tensor is not None else None,
                "offload_hook": hook is not None,
                "execution_device": str(getattr(hook, "execution_device", None)) if hook is not None else None,
            }
        except Exception as error:
            components[name] = {"error": str(error)}
    return components


def generation_failure_message(error: Exception) -> str:
    if isinstance(error, LoraPatchError):
        return " ".join(str(error).split())[:960]
    return str(error)


def generation_failure_trace(error: Exception) -> str:
    if isinstance(error, LoraPatchError):
        # The exception is already bounded and path-sanitized. Do not persist
        # source frames or chained dependency errors for this dedicated class.
        return "".join(traceback.format_exception_only(type(error), error)).strip()[:1100]
    return traceback.format_exc()


def write_generation_failure_log(job_id: str, request: GenerateInput, error: Exception, failure_kind=None):
    try:
        with jobs_lock:
            job = jobs.get(job_id, {})
        out_of_memory = is_oom_error(error)
        failure_kind = failure_kind or ("generation-memory-failure" if out_of_memory else "generation-failure")
        error_message = generation_failure_message(error)
        submitted_prompt = str(job.get("prompt", request.prompt))
        submitted_negative = str(job.get("negative_prompt", request.negative_prompt))
        details = {
            "job_id": job_id,
            "failure": {
                "type": type(error).__name__,
                "message": error_message,
                "phase": job.get("phase"),
                "stage": job.get("stage"),
                "stage_step": job.get("stage_step"),
                "stage_total": job.get("stage_total"),
                "out_of_memory": out_of_memory,
                "category": failure_kind,
            },
            "request": {
                "engine": request.engine,
                "checkpoint": public_model_reference(request.checkpoint),
                "diffusion_model": public_model_reference(request.diffusion_model),
                "text_encoder": public_model_reference(request.text_encoder),
                "text_encoder_2": public_model_reference(request.text_encoder_2),
                "vae": public_model_reference(request.vae),
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "cfg": request.cfg,
                "denoise": request.denoise,
                "img2img_fix_steps": uses_diffusers_img2img_fix_steps(request),
                "img2img_schedule_steps": img2img_schedule_steps(request),
                "seed": request.seed,
                "images_per_batch": request.images_per_batch,
                "batch_count": request.batch_count,
                "sampler": request.sampler,
                "scheduler": request.scheduler,
                "guidance": request.guidance,
                "pag": request.pag.model_dump(),
                "prompt_sha256": hashlib.sha256(submitted_prompt.encode("utf-8")).hexdigest(),
                "prompt_length": len(submitted_prompt),
                "negative_prompt_sha256": hashlib.sha256(submitted_negative.encode("utf-8")).hexdigest(),
                "negative_prompt_length": len(submitted_negative),
                "prompt_encoding": copy.deepcopy(job.get("prompt_encoding")),
                "source_image_enabled": request.source_image.enabled,
                "source_image_resize_mode": request.source_image.resize_mode if request.source_image.enabled else None,
                "postprocess_only": request.postprocess_only,
                "lora_count": len(request.loras),
                "loras": [
                    {"path": public_model_reference(item.path), "weight": item.weight}
                    for item in request.loras
                ],
                "lora_resolution": copy.deepcopy(job.get("inline_loras") or []),
                "adetailer_enabled": request.adetailer.enabled,
                "adetailer_units": len(request.adetailer.active_units),
                "adetailer_detector": " · ".join(unit.detector for unit in request.adetailer.active_units) or None,
                "resample_units": len(request.active_resample_units),
                "resample_chain": [
                    {
                        "unit": request.resample_unit_number(unit), "upscaler": unit.upscaler,
                        "scale": unit.scale, "denoise": unit.denoise,
                        "resize_algorithm": unit.resize_algorithm,
                        "latent_algorithm": unit.latent_algorithm,
                        "execution_mode": unit.execution_mode,
                    }
                    for unit in request.active_resample_units
                ],
                "rtx_enabled": request.rtx.enabled,
                "rtx_scale": request.rtx.scale if request.rtx.enabled else None,
                "rtx_quality": request.rtx.quality if request.rtx.enabled else None,
                "postprocess_chain": [stage.model_dump() for stage in request.postprocess_chain],
            },
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device": cuda_device_name(),
                "diffusers": installed_version("diffusers"),
                "transformers": installed_version("transformers"),
                "accelerate": installed_version("accelerate"),
                "xformers": installed_version("xformers"),
                "nvidia_vfx": installed_version("nvidia-vfx"),
            },
            "rtx_vsr": rtx_vsr.status(),
            "background_removal": background_removal_status(),
            "performance": {
                "configured": performance_settings.copy(),
                "requested_memory_mode": performance_settings["memory_mode"],
                "active_strategy": active_memory_strategy.copy() if active_memory_strategy else None,
                "attention_backend": current_attention_backend(),
                "compute_dtype": active_compute_dtype,
                "vae_mode": active_vae_mode,
                "pipeline_cpu_parked": pipeline_cpu_parked,
                "loaded_checkpoint": Path(loaded_checkpoint).name if loaded_checkpoint else None,
                "loaded_model_assets": loaded_model_asset_references(),
                "loaded_family": loaded_family,
                "loaded_engine": loaded_engine,
                # Where the model was and every move it made, and — under AUTO — every stage decision
                # with the cap it was checked against. The 12 GB report had none of this.
                "residency": active_residency.snapshot() if active_residency is not None else None,
                "residency_trace": list(active_residency.trace) if active_residency is not None else [],
                "stage_schedule": list(active_stage_schedule),
                "stage_error": error.diagnostics() if isinstance(error, StageMemoryError) else None,
                "conditioning_reuse": (
                    active_conditioning_cache.stats() if active_conditioning_cache is not None else None
                ),
            },
            "lora_fusion": job.get("lora_fusion") if isinstance(job.get("lora_fusion"), dict) else None,
            "cuda_memory": cuda_memory_diagnostics(),
            "system_memory": system_memory_diagnostics(),
            "pipeline_components": pipeline_component_diagnostics(),
        }
        created_at = datetime.now().astimezone()
        filename = f"{created_at.strftime('%Y%m%d-%H%M%S-%f')}-{failure_kind}-{job_id[:8]}.log"
        content = "\n".join([
            "XiriaCanvas AI diagnostic log",
            f"Time: {created_at.isoformat()}",
            f"Type: {failure_kind}",
            f"Message: {error_message}",
            "",
            "Details:",
            json.dumps(details, ensure_ascii=False, indent=2),
            "",
            "Traceback:",
            generation_failure_trace(error),
        ])
        LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        (LOG_DIRECTORY / filename).write_text(content, encoding="utf-8")
    except Exception:
        # The original generation failure is more important than a diagnostic write failure.
        pass


def write_cpu_mode_log():
    if torch.cuda.is_available():
        return
    try:
        created_at = datetime.now().astimezone()
        filename = f"{created_at.strftime('%Y%m%d-%H%M%S-%f')}-cpu-mode-active.log"
        content = "\n".join([
            "XiriaCanvas AI diagnostic log",
            f"Time: {created_at.isoformat()}",
            "Type: cpu-mode-active",
            "Message: The inference service started without a usable CUDA device.",
            "",
            "Details:",
            json.dumps({
                "python": sys.version,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": False,
                "platform": platform.platform(),
            }, ensure_ascii=False, indent=2),
            "",
        ])
        (LOG_DIRECTORY / filename).write_text(content, encoding="utf-8")
    except Exception:
        pass


def run_generation(
    job_id: str,
    request: GenerateInput,
    audit_request: GenerateInput | None = None,
    inline_lora_audit=None,
):
    global active_attention_backend, active_vae_mode, active_conditioning_cache
    control = job_controls.get(job_id)
    # A queued job can be cancelled before the worker ever reaches it. Checking here means it is
    # dropped without loading a model or claiming the pipeline, rather than running to completion
    # because nothing looked at the flag until the first sampler callback.
    if control is not None and control.cancelled:
        update_job(
            job_id, status="cancelled", stage="cancelled", phase="Cancelled before starting",
            progress=0, completed_at=time.time(),
        )
        return
    original_request = audit_request or request
    cleaned_prompt, directives = parse_prompt_directives(request.prompt)
    transparent_background = directives["transparent_background"]
    conditioning_prompt = transparent_conditioning_prompt(cleaned_prompt) if transparent_background else cleaned_prompt
    request = request.model_copy(update={"prompt": conditioning_prompt})
    started_at = time.time()
    control = job_controls[job_id]
    ultra_low_memory = performance_settings["memory_mode"] == "ultra_low_vram"
    pipeline = None
    adetailer_result = None
    resample_results = []
    rtx_result = None
    background_removal_result = None
    release_cuda_cache = False
    output_paths = []
    output_records = []

    def cleanup_partial_outputs():
        for output_path in output_paths:
            try:
                Path(output_path).unlink(missing_ok=True)
            except OSError:
                pass

    def publish_stage_preview(image, stage: str, image_index: int = 0, label: str | None = None):
        """Record one finished stage's picture on the job.

        This is the settled result of a stage that has just finished, published while the next
        stage is already running. It stays until that next stage finishes, which is what puts the
        base sample on screen for the whole of the first unit, and that unit's result on screen for
        the whole of whatever follows it. With several resample units, `label` is what tells two of
        them apart on screen -- they share a stage type and differ only by their number.
        """
        # A latent artifact's decode is part of generation, not a best-effort preview operation.
        # Materialize outside the preview-only exception boundary so NaN/OOM/cancellation cannot be
        # swallowed and retried later. The artifact memoizes this image for every downstream user.
        preview_image = materialize_image(image)
        try:
            with jobs_lock:
                job = jobs.get(job_id)
                index = len(job.get("stage_previews") or []) if job else 0
            stage_path = save_stage_preview(preview_image, job_id, stage, index)
        except Exception:
            # A stage picture is a convenience. Losing one must never take the generation with it.
            traceback.print_exc()
            return False
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                remove_preview(stage_path)
                return False
            entries = list(job.get("stage_previews") or [])
            paths = list(job.get("stage_preview_paths") or [])
            entries.append({
                "index": index,
                "stage": stage,
                "label": label or STAGE_PREVIEW_LABELS.get(stage, stage),
                "image_index": image_index,
                "url": f"/api/inference/jobs/{job_id}/stages/{index}",
                "width": preview_image.width,
                "height": preview_image.height,
                "created_at": time.time(),
            })
            paths.append(str(stage_path))
            job["stage_previews"] = entries
            job["stage_preview_paths"] = paths
        return True

    # One job, one stage schedule and one set of shared prompt encodings (QPlan P8).
    reset_stage_schedule()
    active_conditioning_cache = StaticConditioningCache()
    try:
        control.checkpoint(job_id, "Preparing model")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for local generation")
        if request.engine == "Anima":
            roots = anima_model_roots()
            diffusion_model = validate_anima_component(
                request.diffusion_model, roots["diffusion_model"], "Anima diffusion model"
            )
            text_encoder = validate_anima_component(
                request.text_encoder, roots["text_encoder"], "Anima text encoder"
            )
            vae = validate_anima_component(request.vae, roots["vae"], "Anima VAE")
            tokenizer_sources = anima_tokenizer_sources()
            checkpoint = diffusion_model
            resolved_anima_loras = [
                (validate_anima_lora(lora.path, roots["lora"]), lora.weight)
                for lora in request.loras
            ]
            anima_loras = [(path, multiplier) for path, multiplier in resolved_anima_loras if multiplier != 0.0]
            native_loras = anima_loras
            lora_bytes = sum(path.stat().st_size for path, _multiplier in anima_loras)
            family = "anima"
            if ultra_low_memory:
                raise ValueError("Native Anima does not support ultra-low-memory mode")
        elif request.engine == "Flux":
            roots = flux_model_roots()
            diffusion_model = validate_native_diffusion_model(
                request.diffusion_model, roots["diffusion_model"], "Flux diffusion model"
            )
            text_encoder = validate_native_component(
                request.text_encoder, roots["text_encoder"], "Flux text encoder"
            )
            text_encoder_2 = validate_native_component(
                request.text_encoder_2, roots["text_encoder"], "Flux second text encoder"
            )
            vae = validate_native_component(request.vae, roots["vae"], "Flux VAE")
            tokenizer_sources = flux_tokenizer_sources()
            checkpoint = diffusion_model
            anima_loras = []
            native_loras = [
                (path, multiplier)
                for path, multiplier in (
                    (validate_native_lora(lora.path, roots["lora"], "Flux LoRA"), lora.weight)
                    for lora in request.loras
                )
                if multiplier != 0.0
            ]
            lora_bytes = sum(path.stat().st_size for path, _multiplier in native_loras)
            family = "flux"
            if ultra_low_memory:
                raise ValueError("Native Flux does not support ultra-low-memory mode")
        elif request.engine == "Flux2":
            roots = flux2_model_roots()
            diffusion_model = validate_native_diffusion_model(
                request.diffusion_model, roots["diffusion_model"], "Flux2 diffusion model"
            )
            text_encoder = validate_native_component(
                request.text_encoder, roots["text_encoder"], "Flux2 text encoder"
            )
            text_encoder_2 = None
            vae = validate_native_component(request.vae, roots["vae"], "Flux2 VAE")
            tokenizer_sources = flux2_tokenizer_sources()
            checkpoint = diffusion_model
            anima_loras = []
            native_loras = [
                (path, multiplier)
                for path, multiplier in (
                    (validate_native_lora(lora.path, roots["lora"], "Flux2 LoRA"), lora.weight)
                    for lora in request.loras
                )
                if multiplier != 0.0
            ]
            lora_bytes = sum(path.stat().st_size for path, _multiplier in native_loras)
            family = "flux2"
            if ultra_low_memory:
                raise ValueError("Native Flux2 does not support ultra-low-memory mode")
        elif request.engine == "Krea2":
            roots = krea2_model_roots()
            diffusion_model = validate_native_diffusion_model(
                request.diffusion_model, roots["diffusion_model"], "Krea2 diffusion model"
            )
            text_encoder = validate_native_component(
                request.text_encoder, roots["text_encoder"], "Krea2 text encoder"
            )
            text_encoder_2 = None
            vae = validate_native_component(request.vae, roots["vae"], "Krea2 VAE")
            tokenizer_sources = krea2_tokenizer_sources()
            checkpoint = diffusion_model
            anima_loras = []
            native_loras = [
                (path, multiplier)
                for path, multiplier in (
                    (validate_native_lora(lora.path, roots["lora"], "Krea2 LoRA"), lora.weight)
                    for lora in request.loras
                )
                if multiplier != 0.0
            ]
            lora_bytes = sum(path.stat().st_size for path, _multiplier in native_loras)
            family = "krea2"
            if ultra_low_memory:
                raise ValueError("Native Krea2 does not support ultra-low-memory mode")
        else:
            anima_loras = []
            native_loras = []
            checkpoint_root, lora_root = model_roots(request.engine)
            checkpoint = resolve_model_path(request.checkpoint, checkpoint_root, CHECKPOINT_EXTENSIONS, "Checkpoint")
            family = detect_model_family(checkpoint, request.engine)
            if ultra_low_memory and family != "sdxl":
                raise ValueError("Ultra-low-memory mode is intended for SDXL or Illustrious checkpoints")
            if ultra_low_memory and postprocessing_stages(request):
                raise ValueError("Ultra-low-memory mode does not support Hires.fix, ADetailer, or RTX VSR; disable them and retry")
            if ultra_low_memory and request.loras:
                raise ValueError("Ultra-low-memory mode does not support LoRA loading on 16 GB system memory")
            if ultra_low_memory and request.guidance != "none":
                raise ValueError("Ultra-low-memory mode does not support guidance enhancement; select None and retry")
            lora_bytes = sum(resolve_model_path(lora.path, lora_root, LORA_EXTENSIONS, "LoRA").stat().st_size for lora in request.loras)
        update_job(
            job_id,
            status="running",
            phase="Preparing model",
            stage="model_load",
            progress=2,
            started_at=started_at,
            family=family,
            guidance=guidance_diagnostics(
                request.guidance,
                request.steps,
                request.pag.scale,
                request.pag.applied_layers,
                engine=request.engine,
                guidance_scale=request.cfg,
            ),
        )
        # Decoded and resampled once for the whole job: the same pixels start every batch, and every
        # image within a batch, so a per-batch decode would only repeat work and hold a second copy.
        # It happens before admission because in post-processing mode the decoded picture *is* the
        # canvas admission sizes against.
        source_image = prepare_source_image(request)
        canvas_adopted = source_image is not None and adopt_postprocess_canvas(request, source_image)
        workload_diagnostics = generation_memory_workload_diagnostics(request, family)
        memory_width, memory_height, memory_cfg, memory_batch = workload_diagnostics["admission"]
        update_job(job_id, memory_workload=workload_diagnostics)
        if family == "anima":
            pipeline = load_anima_pipeline(
                diffusion_model,
                text_encoder,
                vae,
                tokenizer_sources,
                memory_width,
                memory_height,
                memory_cfg,
                job_id,
                memory_batch,
                request.guidance,
                request.pag.scale,
                loras=anima_loras,
                lora_bytes=lora_bytes,
            )
        elif family in ("flux", "flux2", "krea2"):
            pipeline = load_flux_pipeline(
                diffusion_model,
                text_encoder,
                text_encoder_2,
                vae,
                tokenizer_sources,
                memory_width,
                memory_height,
                memory_cfg,
                job_id,
                memory_batch,
                loras=native_loras,
                lora_bytes=lora_bytes,
                engine=request.engine,
                # Reported, not budgeted: Krea 2 runs a second branch per step but runs it
                # sequentially, so this changes what the job says it is doing, not what it admits.
                guidance_copies=(
                    guidance_prediction_copies(request.guidance, memory_cfg) if family == "krea2" else 1
                ),
            )
        else:
            pipeline = load_pipeline(
                checkpoint,
                family,
                request.engine,
                memory_width,
                memory_height,
                memory_cfg,
                lora_bytes,
                job_id,
                memory_batch,
                request.guidance,
                loras=lora_signature(request.loras, lora_root),
            )
        # The native loaders fuse their adapters into the weights as the pipeline is built, so this
        # is the point at which they are mounted. The diffusers families report from
        # `configure_loras`, which runs a little further down for the same reason.
        report_mounted_loras(job_id, native_loras)
        control.checkpoint(job_id, "Preparing prompt")
        if family == "anima":
            sampling_diagnostics = anima_sampling_diagnostics(request.sampler, request.scheduler)
            warning = (
                f"Native Anima Flow Matching uses shift 3 with {sampling_diagnostics['sampler_implementation']} "
                f"and {sampling_diagnostics['scheduler_implementation']}"
            )
            if sampling_diagnostics["warning"]:
                warning = append_warning(warning, sampling_diagnostics["warning"])
            if request.guidance == "pag":
                layer_label = "Cosmos block 14 self-attention" if request.pag.applied_layers == "mid" else "all 28 Cosmos self-attention blocks"
                warning = append_warning(
                    warning,
                    f"Native PAG scale {request.pag.scale:g} is applied sequentially to {layer_label}",
                )
        elif family == "flux":
            flux_shift = flux_resolution_shift(request.width, request.height)
            sampling_diagnostics = flux_sampling_diagnostics(request.sampler, request.scheduler, flux_shift)
            warning = (
                f"Native FLUX.1 rectified flow uses shift {flux_shift:.3g} with "
                f"{sampling_diagnostics['sampler_implementation']} and {sampling_diagnostics['requested_scheduler']}"
            )
            if sampling_diagnostics["warning"]:
                warning = append_warning(warning, sampling_diagnostics["warning"])
            # `cfg` is the distilled guidance embedding here, not a classifier-free scale. Saying so
            # once is what stops the number being read as "how strongly the negative prompt applies".
            warning = append_warning(
                warning,
                f"FLUX.1 is guidance distilled: CFG {request.cfg:g} is the distilled guidance embedding and no negative branch runs",
            )
            if getattr(pipeline, "lora_report", None):
                skipped = sorted({
                    target for report in pipeline.lora_report for target in report.get("skipped_modules", ())
                })
                if skipped:
                    warning = append_warning(
                        warning,
                        f"{len(skipped)} LoRA target(s) outside the FLUX.1 transformer were skipped (text-encoder LoRA is not applied)",
                    )
        elif family == "flux2":
            flux2_shift = flux2_resolution_shift(request.width, request.height, request.steps)
            sampling_diagnostics = flux2_sampling_diagnostics(request.sampler, request.scheduler, flux2_shift)
            warning = (
                f"Native FLUX.2 rectified flow uses the empirical shift {flux2_shift:.3g} for this canvas "
                f"and step count, with {sampling_diagnostics['sampler_implementation']} and "
                f"{sampling_diagnostics['requested_scheduler']}"
            )
            if sampling_diagnostics["warning"]:
                warning = append_warning(warning, sampling_diagnostics["warning"])
            warning = append_warning(
                warning,
                f"FLUX.2 is guidance distilled: CFG {request.cfg:g} is the distilled guidance embedding and no negative branch runs",
            )
            # ComfyUI's FLUX.2 tokenizers pass `disable_weights=True`, so emphasis syntax is text.
            if prompt_carries_weight_syntax(request.prompt):
                warning = append_warning(
                    warning,
                    "FLUX.2 reads the prompt through a language model with weighting disabled; (word:1.2) is read as literal text",
                )
            if getattr(pipeline, "lora_report", None):
                skipped = sorted({
                    target for report in pipeline.lora_report for target in report.get("skipped_modules", ())
                })
                if skipped:
                    warning = append_warning(
                        warning,
                        f"{len(skipped)} LoRA target(s) outside the FLUX.2 transformer were skipped (text-encoder LoRA is not applied)",
                    )
        elif family == "krea2":
            sampling_diagnostics = krea2_sampling_diagnostics(
                request.sampler, request.scheduler, request.width, request.height
            )
            warning = (
                f"Native Krea 2 rectified flow uses the model's declared shift {KREA2_SHIFT:g} at every "
                f"canvas, with {sampling_diagnostics['sampler_implementation']} and "
                f"{sampling_diagnostics['requested_scheduler']}"
            )
            if sampling_diagnostics["warning"]:
                warning = append_warning(warning, sampling_diagnostics["warning"])
            if request.guidance == "cfg_zero_star":
                warning = append_warning(
                    warning,
                    f"CFG-Zero* rescales the unconditional branch and zeroes the first "
                    f"{cfg_zero_star_zero_steps(request.steps)} step(s)",
                )
            # `Qwen3VLTokenizer` passes `disable_weights=True`, so emphasis syntax is literal text.
            if prompt_carries_weight_syntax(request.prompt):
                warning = append_warning(
                    warning,
                    "Krea 2 reads the prompt through a language model with weighting disabled; (word:1.2) is read as literal text",
                )
            if getattr(pipeline, "lora_report", None):
                skipped = sorted({
                    target for report in pipeline.lora_report for target in report.get("skipped_modules", ())
                })
                if skipped:
                    warning = append_warning(
                        warning,
                        f"{len(skipped)} LoRA target(s) outside the Krea 2 transformer were skipped (text-encoder LoRA is not applied)",
                    )
        else:
            warning = configure_scheduler(pipeline, request.sampler, request.scheduler)
            if request.guidance == "pag":
                layer_label = "the UNet mid layer" if request.pag.applied_layers == "mid" else "all available self-attention layers"
                warning = append_warning(warning, f"PAG scale {request.pag.scale:g} is applied to {layer_label}")
            if ultra_low_memory:
                warning = append_warning(warning, "Ultra-low-memory mode uses serial CFG, disables process previews, and may take significantly longer")
        if request.denoise < 0.9999 and not request.source_image.enabled:
            denoise_warning = "Denoise below 1.0 is the image-to-image strength and is ignored for text-to-image."
            warning = f"{warning}; {denoise_warning}" if warning else denoise_warning
        if request.postprocess_only:
            warning = append_warning(
                warning,
                "Post-processing mode skips the base sampling pass; denoise, steps and CFG apply only where a stage inherits them",
            )
            if canvas_adopted:
                warning = append_warning(
                    warning,
                    f"Post-processing canvas follows the decoded source ({request.width} x {request.height})",
                )
        if family == "anima":
            prompt_diagnostics_anima = pipeline.token_diagnostics(request.prompt)
            negative_prompt_diagnostics_anima = pipeline.token_diagnostics(request.negative_prompt)
            prompt_info = prompt_diagnostics_anima["qwen"]
            negative_prompt_info = negative_prompt_diagnostics_anima["qwen"]
            prompt_weighted_tokens = prompt_diagnostics_anima["t5"]["weighted_token_count"]
            negative_prompt_weighted_tokens = negative_prompt_diagnostics_anima["t5"]["weighted_token_count"]
        elif family == "flux":
            # The T5 sequence is the conditioning; CLIP-L contributes only a pooled vector. Reporting
            # the T5 count is reporting the limit a long prompt actually runs into.
            prompt_diagnostics_flux = pipeline.token_diagnostics(request.prompt)
            prompt_info = prompt_diagnostics_flux["t5"]
            negative_prompt_info = {"token_count": 0, "weighted_token_count": 0}
            prompt_weighted_tokens = prompt_diagnostics_flux["t5"]["weighted_token_count"]
            negative_prompt_weighted_tokens = 0
        elif family == "flux2":
            # One language model is the whole conditioning, and it never reports weighted tokens:
            # its tokenizer runs with weighting disabled.
            prompt_info = pipeline.token_diagnostics(request.prompt)["llm"]
            negative_prompt_info = {"token_count": 0, "weighted_token_count": 0}
            prompt_weighted_tokens = 0
            negative_prompt_weighted_tokens = 0
        elif family == "krea2":
            # Also one language model, but with a real unconditional branch, so the negative
            # prompt is encoded and its token count is a number the user can act on.
            prompt_info = pipeline.token_diagnostics(request.prompt)["llm"]
            negative_prompt_info = pipeline.token_diagnostics(request.negative_prompt)["llm"]
            prompt_weighted_tokens = 0
            negative_prompt_weighted_tokens = 0
        else:
            configure_loras(pipeline, request.loras, lora_root, job_id)
            # Adapters change every figure the controller places the model by, and a block or leaf
            # carries its adapter submodules with it — so the measurement belongs after the mount,
            # which is the order the plan asks for (P1-6).
            if active_residency is not None:
                active_residency.measure()
            prompt_info = forge_prompt_diagnostics(pipeline.tokenizer, request.prompt)
            negative_prompt_info = forge_prompt_diagnostics(pipeline.tokenizer, request.negative_prompt)
            prompt_weighted_tokens = prompt_info["weighted_tokens"]
            negative_prompt_weighted_tokens = negative_prompt_info["weighted_tokens"]
        update_job(
            job_id,
            phase="Preparing prompt",
            stage="prompt_encode",
            progress=24,
            warning=warning,
            prompt_tokens=prompt_info["token_count"] if is_native_family(family) else prompt_info["tokens"],
            prompt_blocks=1 if is_native_family(family) else prompt_info["blocks"],
            prompt_weighted_tokens=prompt_weighted_tokens,
            negative_prompt_tokens=negative_prompt_info["token_count"] if is_native_family(family) else negative_prompt_info["tokens"],
            negative_prompt_blocks=1 if is_native_family(family) else negative_prompt_info["blocks"],
            negative_prompt_weighted_tokens=negative_prompt_weighted_tokens,
        )

        if not is_native_family(family) and stage_scheduling_active():
            # Stage 2 of the matrix: the text towers' own pass, placed before it runs.
            plan_residency_for_stage(
                "prompt_encode", family, (request.width, request.height), request.cfg,
                request.images_per_batch, job_id, kind="encode",
                inference_bytes=TEXT_ENCODE_TRANSIENT_BYTES, legacy_point=False,
            )
        conditioning = None if is_native_family(family) else stage_prompt_conditioning(
            pipeline, family, request.prompt, request.negative_prompt,
            timeline=sampling_timeline(pipeline, request.steps),
            clip_skip=effective_clip_skip(request),
        )
        if ultra_low_memory:
            update_job(job_id, phase="Releasing prompt encoders", progress=24)
            release_prompt_encoders(pipeline)
        total_images = request.images_per_batch * request.batch_count
        sampling_steps = base_sampling_steps(request, family)
        update_job(
            job_id,
            images_per_batch=request.images_per_batch,
            batch_count=request.batch_count,
            batch_index=0,
            completed_images=0,
            total_images=total_images,
            total_steps=sampling_steps,
        )

        for batch_index in range(request.batch_count):
            control.checkpoint(job_id, f"Preparing batch {batch_index + 1}")
            if pipeline_cpu_parked:
                update_job(job_id, phase=f"Restoring model · Batch {batch_index + 1}/{request.batch_count}", stage="model_restore")
                restore_parked_pipeline()

            seeds = batch_seeds(request, batch_index)
            generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in seeds]
            batch_start = 25 + 73 * batch_index / request.batch_count
            batch_end = 25 + 73 * (batch_index + 1) / request.batch_count
            enabled_post_stages = postprocessing_stages(request)
            # Where each of them sits in the whole chain. Two resample units can hold identical
            # settings, and `list.index` compares pydantic models by value, so the position has to
            # be carried rather than looked up again from the stage itself.
            enabled_post_stage_indices = [index for index, stage in enumerate(request.postprocess_chain) if stage.enabled]
            postprocessing_enabled = bool(enabled_post_stages)
            # Post-processing mode samples and decodes nothing, so both of those slices collapse onto
            # the start of the batch and the stages get the whole budget.
            sampling_end = batch_start if request.postprocess_only else batch_start + (batch_end - batch_start) * (0.68 if request.active_resample_units else 0.82 if postprocessing_enabled else 0.92)
            decode_end = batch_start if request.postprocess_only else batch_start + (batch_end - batch_start) * (0.74 if postprocessing_enabled else 1.0)
            phase = (
                f"Batch {batch_index + 1}/{request.batch_count} · Post-processing {request.images_per_batch} images"
                if request.postprocess_only
                else f"Batch {batch_index + 1}/{request.batch_count} · Sampling {request.images_per_batch} images"
            )
            last_base_step = [0]

            def on_step_end(_pipeline, step_index, _timestep, callback_kwargs, *, running_phase=phase, progress_start=batch_start, progress_end=sampling_end):
                step = logical_diffusion_step(_pipeline, step_index, sampling_steps)
                last_base_step[0] = max(last_base_step[0], step)
                batch_progress = progress_start + (step / sampling_steps) * (progress_end - progress_start)
                update_job(
                    job_id,
                    phase=running_phase,
                    stage="base_sampling",
                    stage_step=min(step, sampling_steps),
                    stage_total=sampling_steps,
                    progress=min(round(batch_progress), 98),
                    step=step,
                    total_steps=sampling_steps,
                    batch_index=batch_index + 1,
                    elapsed_seconds=round(control.active_elapsed(started_at), 1),
                    paused_seconds=round(control.total_paused(), 1),
                )
                control.checkpoint(job_id, running_phase)
                return callback_kwargs

            def on_anima_step(step, total_steps, _latents, *, running_phase=phase, progress_start=batch_start, progress_end=sampling_end):
                batch_progress = progress_start + (step / max(1, total_steps)) * (progress_end - progress_start)
                update_job(
                    job_id,
                    phase=running_phase,
                    stage="base_sampling",
                    stage_step=min(step, total_steps),
                    stage_total=total_steps,
                    progress=min(round(batch_progress), 98),
                    step=step,
                    total_steps=total_steps,
                    batch_index=batch_index + 1,
                    elapsed_seconds=round(control.active_elapsed(started_at), 1),
                    paused_seconds=round(control.total_paused(), 1),
                )
            def on_anima_step_checkpoint(_step, _total_steps, _latents, *, running_phase=phase):
                control.checkpoint(job_id, running_phase)

            update_job(
                job_id,
                phase=phase,
                stage="postprocess_source" if request.postprocess_only else "base_sampling",
                stage_step=0,
                stage_total=sampling_steps,
                progress=round(batch_start),
                step=0,
                total_steps=sampling_steps,
                batch_index=batch_index + 1,
            )
            if request.postprocess_only:
                # One copy per requested image rather than one shared object: each goes down its own
                # stage chain with its own seed, and the stages composite in place.
                images = [source_image.copy() for _ in range(request.images_per_batch)]
            elif family == "anima":
                plan_native_stage(
                    "base_sampling", family, (request.width, request.height), request.cfg,
                    request.images_per_batch, pipeline, job_id,
                )
                anima_batch_size = (active_memory_strategy or {}).get(
                    "sampling_batch_size", request.images_per_batch
                )
                if source_image is not None:
                    anima_chunk_size = max(1, int(anima_batch_size or request.images_per_batch))
                    # `refine_batch` is the same native path Hires.fix and ADetailer refine through,
                    # so image-to-image on Anima is the existing refinement contract with the user's
                    # picture in place of an upscaled one. It has no `sampling_batch_size` of its
                    # own — it was only ever handed a single image — so the memory strategy's
                    # subdivision is applied here, with the step offsets `generate_batch` uses
                    # internally so the progress bar still counts one run rather than N.
                    chunks = [
                        generators[index:index + anima_chunk_size]
                        for index in range(0, len(generators), anima_chunk_size)
                    ]
                    images = []
                    for chunk_index, chunk in enumerate(chunks):
                        chunk_offset = chunk_index * request.steps
                        chunk_total = len(chunks) * request.steps
                        images.extend(pipeline.refine_batch(
                            images=[source_image] * len(chunk),
                            prompt=request.prompt,
                            negative_prompt=request.negative_prompt,
                            steps=request.steps,
                            denoise=request.denoise,
                            cfg=request.cfg,
                            sampler=request.sampler,
                            scheduler=request.scheduler,
                            generators=chunk,
                            guidance=request.guidance,
                            pag_scale=request.pag.scale,
                            pag_applied_layers=request.pag.applied_layers,
                            on_step=lambda step, _total, latents, offset=chunk_offset, total=chunk_total: on_anima_step(offset + step, total, latents),
                            on_step_checkpoint=lambda step, _total, latents, offset=chunk_offset, total=chunk_total: on_anima_step_checkpoint(offset + step, total, latents),
                        ))
                else:
                    images = pipeline.generate_batch(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        width=request.width,
                        height=request.height,
                        steps=request.steps,
                        cfg=request.cfg,
                        sampler=request.sampler,
                        scheduler=request.scheduler,
                        generators=generators,
                        guidance=request.guidance,
                        pag_scale=request.pag.scale,
                        pag_applied_layers=request.pag.applied_layers,
                        on_step=on_anima_step,
                        on_step_checkpoint=on_anima_step_checkpoint,
                        sampling_batch_size=anima_batch_size,
                    )
                record_native_decode("base_decode", family, (request.width, request.height), job_id)
                actual_cfg_batch = bool(getattr(pipeline, "batch_cfg", False))
                actual_transformer_resident = bool(getattr(pipeline, "transformer_resident", False))
                runtime_attention_backend = getattr(pipeline, "attention_backend", None)
                if runtime_attention_backend:
                    active_attention_backend = anima_attention_backend_label(runtime_attention_backend)
                if bool(getattr(getattr(pipeline, "vae", None), "use_tiling", False)):
                    active_vae_mode = "tiled"
                if active_memory_strategy is not None:
                    active_memory_strategy["cfg_batch"] = actual_cfg_batch
                    actual_copies = 2 if actual_cfg_batch else 1
                    active_memory_strategy["actual_physical_forward_copies"] = actual_copies
                    active_memory_strategy["physical_forward_copies"] = actual_copies
                    admission = active_memory_strategy.get("admission")
                    if isinstance(admission, dict):
                        resident_probe = bool(admission.get("resident_cfg_batch_probe", False))
                        admission["resident_cfg_batch_actual"] = bool(resident_probe and actual_cfg_batch)
                        if resident_probe and not actual_cfg_batch:
                            fallback = getattr(pipeline, "last_generation_metrics", {}).get("cfg_batch_fallback")
                            admission["resident_cfg_batch_fallback"] = fallback or admission.get("resident_cfg_batch_fallback")
                        admission["actual_physical_forward_copies"] = actual_copies
                        admission["physical_forward_copies"] = actual_copies
                        admission["physical_inference_bytes"] = int(
                            admission["cfg_batch_inference_bytes"]
                            if actual_cfg_batch
                            else admission["serial_resident_inference_bytes"]
                        )
                        admission["sampling_stage_required_bytes"] = int(
                            admission["cfg_batch_sampling_required_bytes"]
                            if actual_cfg_batch
                            else admission["serial_resident_sampling_required_bytes"]
                        )
                    active_memory_strategy["physical_inference_bytes"] = int(
                        (admission or {}).get("physical_inference_bytes", active_memory_strategy.get("physical_inference_bytes", 0))
                    )
                if record_group_offload_fallback(pipeline, active_memory_strategy):
                    update_job(job_id, **memory_job_fields(active_memory_strategy))
                update_job(
                    job_id,
                    cfg_batch=actual_cfg_batch,
                    transformer_resident=actual_transformer_resident,
                    attention_backend=active_attention_backend,
                    vae_mode=active_vae_mode,
                )
                merge_runtime_metrics(job_id, getattr(pipeline, "last_generation_metrics", None))
            elif family in ("flux", "flux2", "krea2"):
                plan_native_stage(
                    "base_sampling", family, (request.width, request.height), request.cfg,
                    request.images_per_batch, pipeline, job_id,
                )
                flux_batch_size = (active_memory_strategy or {}).get(
                    "sampling_batch_size", request.images_per_batch
                )
                if source_image is not None:
                    # Same shape as the Anima branch: `refine_batch` takes one image per generator
                    # and has no subdivision of its own, so the memory strategy's chunk size is
                    # applied here with the step offsets that keep the bar counting one run.
                    chunk_size = max(1, int(flux_batch_size or request.images_per_batch))
                    chunks = [
                        generators[index:index + chunk_size]
                        for index in range(0, len(generators), chunk_size)
                    ]
                    images = []
                    for chunk_index, chunk in enumerate(chunks):
                        chunk_offset = chunk_index * request.steps
                        chunk_total = len(chunks) * request.steps
                        images.extend(pipeline.refine_batch(
                            images=[source_image] * len(chunk),
                            prompt=request.prompt,
                            negative_prompt=request.negative_prompt,
                            steps=request.steps,
                            denoise=request.denoise,
                            cfg=request.cfg,
                            sampler=request.sampler,
                            scheduler=request.scheduler,
                            generators=chunk,
                            guidance=request.guidance,
                            on_step=lambda step, _total, latents, offset=chunk_offset, total=chunk_total: on_anima_step(offset + step, total, latents),
                            on_step_checkpoint=lambda step, _total, latents, offset=chunk_offset, total=chunk_total: on_anima_step_checkpoint(offset + step, total, latents),
                        ))
                else:
                    images = pipeline.generate_batch(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        width=request.width,
                        height=request.height,
                        steps=request.steps,
                        cfg=request.cfg,
                        sampler=request.sampler,
                        scheduler=request.scheduler,
                        generators=generators,
                        guidance=request.guidance,
                        on_step=on_anima_step,
                        on_step_checkpoint=on_anima_step_checkpoint,
                        sampling_batch_size=flux_batch_size,
                    )
                record_native_decode("base_decode", family, (request.width, request.height), job_id)
                runtime_attention_backend = getattr(pipeline, "attention_backend", None)
                if runtime_attention_backend:
                    active_attention_backend = anima_attention_backend_label(runtime_attention_backend)
                if bool(getattr(getattr(pipeline, "vae", None), "use_tiling", False)):
                    active_vae_mode = "tiled"
                if record_group_offload_fallback(pipeline, active_memory_strategy):
                    update_job(job_id, **memory_job_fields(active_memory_strategy))
                update_job(
                    job_id,
                    # Always false: the Flux engines have no second branch at all, and Krea 2 runs
                    # its two one after the other rather than as a batch of two.
                    cfg_batch=False,
                    transformer_resident=bool(getattr(pipeline, "transformer_resident", False)),
                    attention_backend=active_attention_backend,
                    vae_mode=active_vae_mode,
                )
                merge_runtime_metrics(job_id, getattr(pipeline, "last_generation_metrics", None))
            else:
                # An image-to-image run is the same weights driven through the Diffusers
                # image-to-image pipeline — the one Hires.fix already derives for its refinement
                # pass — so the factory is the only difference between the two branches.
                sampling_factory = AutoPipelineForImage2Image if source_image is not None else None
                pipeline_kwargs = (
                    source_image_pipeline_kwargs(request, source_image, generators, on_step_end, conditioning)
                    if source_image is not None
                    else batch_pipeline_kwargs(request, generators, on_step_end, conditioning)
                )
                plan_residency_for_stage(
                    "base_sampling", family, (request.width, request.height),
                    request.cfg, request.images_per_batch, job_id,
                    decode_in_call=not performance_settings["staged_vae_decode"],
                )
                if performance_settings["staged_vae_decode"]:
                    pipeline_kwargs["output_type"] = "latent"
                    latents = run_guided_pipeline_batch(
                        pipeline, request.guidance, pipeline_kwargs, request.pag, sampling_factory, conditioning
                    )
                    latents = latents.to("cpu")
                    images = decode_staged_latents(
                        pipeline,
                        latents,
                        family,
                        job_id,
                        control,
                        active_vae_mode == "tiled",
                        start_progress=round(sampling_end),
                        final_progress=round(decode_end),
                        decision_stage="base_decode",
                    )
                    del latents
                else:
                    images = run_guided_pipeline_batch(
                        pipeline, request.guidance, pipeline_kwargs, request.pag, sampling_factory, conditioning
                    )
                    record_decode_decision(
                        "base_decode", family, (request.width, request.height), evicted=False, job_id=job_id,
                        reason="decode_in_sampling_call",
                        peak_bytes=component_decode_bytes(pipeline, request.width, request.height, request.images_per_batch),
                    )
            if len(images) != request.images_per_batch:
                raise RuntimeError(
                    f"Pipeline returned {len(images)} images for a requested batch of {request.images_per_batch}"
                )

            for image_index, image in enumerate(images):
                control.checkpoint(job_id, f"Processing image {image_index + 1}")
                base_seed = seeds[image_index]
                chain_seeds = resolve_chain_seeds(request, base_seed)
                seed_facts = output_seed_facts(request, base_seed, chain_seeds)
                update_job(job_id, **seed_facts)
                detail_start = decode_end + (batch_end - decode_end) * image_index / request.images_per_batch
                detail_end = decode_end + (batch_end - decode_end) * (image_index + 1) / request.images_per_batch
                adetailer_result = None
                resample_results = []
                rtx_result = None
                post_stages = enabled_post_stages
                # The base sample is a stage in its own right, and publishing it here is what puts a
                # finished picture on screen for the whole of the first post-processing stage rather
                # than leaving the panel on the last half-formed latent.
                if post_stages:
                    publish_stage_preview(image, "base", image_index)
                for stage_index, post_stage in enumerate(post_stages):
                    stage_start = detail_start + (detail_end - detail_start) * stage_index / len(post_stages)
                    stage_end = detail_start + (detail_end - detail_start) * (stage_index + 1) / len(post_stages)
                    chain_index = enabled_post_stage_indices[stage_index]
                    if post_stage.type == "adetailer" and pipeline_cpu_parked:
                        control.checkpoint(job_id, "Restoring sampler for ADetailer")
                        update_job(job_id, phase="Restoring model for ADetailer", stage="model_restore", progress=round(stage_start))
                        restore_parked_pipeline()
                    if post_stage.type == "resample" and pipeline_cpu_parked:
                        control.checkpoint(job_id, "Restoring sampler for resample")
                        update_job(job_id, phase="Restoring model for resample", stage="model_restore", progress=round(stage_start))
                        restore_parked_pipeline()
                    if post_stage.type == "resample":
                        unit_number = request.resample_unit_number(post_stage)
                        image, resample_result = apply_resample_unit(
                            image,
                            pipeline,
                            family,
                            request,
                            post_stage,
                            job_id,
                            control,
                            started_at,
                            unit_number=unit_number,
                            image_seed=base_seed,
                            effective_seed=chain_seeds.get(chain_index),
                            progress_start=round(stage_start),
                            progress_end=round(stage_end),
                        )
                        if resample_result:
                            resample_results.append(resample_result)
                        resample_warning = resample_result.get("warning") if resample_result else None
                        if resample_warning and resample_warning not in (warning or ""):
                            warning = append_warning(warning, resample_warning)
                            update_job(job_id, warning=warning)
                    elif post_stage.type == "adetailer":
                        image = materialize_image(image)
                        image, adetailer_result, adetailer_warning = apply_adetailer(
                            image,
                            pipeline,
                            family,
                            request,
                            job_id,
                            control,
                            started_at,
                            image_seed=base_seed,
                            progress_start=round(stage_start),
                            progress_end=round(stage_end),
                        )
                        if adetailer_warning and adetailer_warning not in (warning or ""):
                            warning = append_warning(warning, adetailer_warning)
                            update_job(job_id, warning=warning)
                    elif post_stage.type == "rtx":
                        image = materialize_image(image)
                        image, rtx_result = apply_rtx_vsr(
                            image,
                            pipeline,
                            request,
                            job_id,
                            control,
                            started_at,
                            progress_start=round(stage_start),
                            progress_end=round(stage_end),
                        )
                    else:
                        raise RuntimeError(f"Unsupported post-processing stage: {post_stage.type}")
                    # `image` is now this stage's result, and stays on screen until the next stage
                    # finishes — so the first unit is watched against the base sample, and whatever
                    # follows against that unit's result.
                    publish_stage_preview(
                        image, post_stage.type, image_index,
                        label=f"重采样 {request.resample_unit_number(post_stage)}" if post_stage.type == "resample" else None,
                    )
                # The ordered chain is allowed to carry a latent internally, but every external
                # consumer and the chain tail are PIL. This is a cache hit when the stage preview
                # already materialized the same artifact; it never performs a second full decode.
                image = materialize_image(image)
                background_removal_result = None
                if transparent_background:
                    control.checkpoint(job_id, "Removing background")
                    completed_images = len(output_paths) + 1
                    update_job(
                        job_id,
                        phase=f"正在提取前景并生成透明背景 {completed_images}/{total_images}",
                        stage="background_remove",
                        stage_step=completed_images,
                        stage_total=total_images,
                        progress=min(97, round(detail_end)),
                        transparent_background=True,
                    )
                    try:
                        image, background_removal_result = extract_foreground(image, request.background_removal_model)
                    except Exception as error:
                        raise RuntimeError(f"透明背景处理失败，未保存不透明结果：{error}") from error
                    background_warning = background_removal_result.get("warning")
                    if background_warning and background_warning not in (warning or ""):
                        warning = append_warning(warning, background_warning)
                    update_job(job_id, background_removal=background_removal_result, warning=warning)
                    control.checkpoint(job_id, "Saving transparent image")
                control.checkpoint(job_id, "Saving image")
                completed_images = len(output_paths) + 1
                update_job(
                    job_id,
                    phase=f"Saving image {completed_images}/{total_images}",
                    stage="save",
                    stage_step=completed_images,
                    stage_total=total_images,
                    progress=min(98, round(detail_end)),
                )
                record_final_decode(family, tuple(image.size), job_id)
                wall_elapsed = time.time() - started_at
                paused_seconds = control.total_paused()
                elapsed = max(0.0, wall_elapsed - paused_seconds)
                workflow_warnings = []
                output_path = save_image(
                    image,
                    job_id,
                    original_request,
                    elapsed,
                    wall_elapsed,
                    paused_seconds,
                    adetailer_result,
                    resample_results,
                    rtx_result,
                    background_removal_result,
                    conditioning_prompt=conditioning_prompt,
                    inline_loras=inline_lora_audit,
                    seed=base_seed,
                    chain_seeds=chain_seeds,
                    batch_index=batch_index + 1,
                    image_index=image_index + 1,
                    before_commit=lambda: control.checkpoint(job_id, "Committing image"),
                    workflow_request=request,
                    workflow_warnings=workflow_warnings,
                )
                for workflow_warning in workflow_warnings:
                    if workflow_warning not in (warning or ""):
                        warning = append_warning(warning, workflow_warning)
                        update_job(job_id, warning=warning)
                output_paths.append(output_path)
                control.checkpoint(job_id, "Recording saved image")
                output_records.append({
                    "index": completed_images - 1,
                    "asset_id": history_asset_token(output_path),
                    "batch_index": batch_index + 1,
                    "image_index": image_index + 1,
                    **seed_facts,
                    "width": image.width,
                    "height": image.height,
                    "output_name": output_path.name,
                    # Relative to the output root; the absolute path stays in the private fields.
                    "output_relative": output_relative_path(output_path),
                    "image_url": f"/api/inference/jobs/{job_id}/images/{completed_images - 1}",
                    "transparent_background": transparent_background,
                    "background_removal": background_removal_result,
                    "postprocess_stages": post_stages,
                    "resample": resample_results,
                    "rtx": rtx_result,
                })
                update_job(job_id, completed_images=completed_images)

            update_job(
                job_id,
                phase=f"Batch {batch_index + 1}/{request.batch_count} complete",
                progress=min(98, round(batch_end)),
                completed_images=len(output_paths),
            )

        if not is_native_family(family) and performance_settings["staged_vae_decode"] and performance_settings["keep_model_cached"] and postprocessing_stages(request) and not pipeline_cpu_parked:
            control.checkpoint(job_id, "Moving model to system memory")
            update_job(job_id, phase="Moving model to system memory", stage="sampler_offload", stage_step=0, stage_total=0, progress=98)
            park_pipeline_for_vae(pipeline)
        wall_elapsed = time.time() - started_at
        paused_seconds = control.total_paused()
        elapsed = max(0.0, wall_elapsed - paused_seconds)
        output_path = output_paths[0]
        if active_conditioning_cache is not None:
            update_job(job_id, conditioning_reuse=active_conditioning_cache.stats())
        release_pipeline_after_job()
        update_job(
            job_id,
            status="complete",
            phase="Complete",
            stage="complete",
            progress=100,
            output_path=str(output_path),
            output_paths=[str(path) for path in output_paths],
            outputs=output_records,
            output_name=output_path.name,
            output_relative=output_relative_path(output_path),
            image_url=f"/api/inference/jobs/{job_id}/image",
            elapsed_seconds=round(elapsed, 1),
            wall_elapsed_seconds=round(wall_elapsed, 1),
            paused_seconds=round(paused_seconds, 1),
            completed_at=time.time(),
            **pipeline_status_fields(),
        )
    except GenerationCancelled:
        cleanup_partial_outputs()
        if not restore_cached_pipeline_state(job_id, request):
            pipeline = None
            clear_pipeline()
        release_pipeline_after_job()
        update_job(
            job_id,
            status="cancelled",
            phase="Cancelled",
            stage="cancelled",
            error=None,
            completed_images=0,
            outputs=[],
            elapsed_seconds=round(control.active_elapsed(started_at), 1),
            paused_seconds=round(control.total_paused(), 1),
            completed_at=time.time(),
            **pipeline_status_fields(),
        )
    except Exception as error:
        cleanup_partial_outputs()
        if not isinstance(error, LoraPatchError):
            traceback.print_exc()
        write_generation_failure_log(job_id, request, error)
        if is_oom_error(error):
            release_cuda_cache = True
            pipeline = None
            clear_pipeline()
            update_job(
                job_id,
                status="error",
                phase="Failed",
                stage="error",
                # A stage AUTO could not place says which stage, on what canvas, needing how much
                # against how much (QPlan P8 §3.6); a bare allocator failure keeps the old advice.
                error=(
                    str(error) if isinstance(error, StageMemoryError)
                    else "GPU memory exhausted. Try a smaller canvas, fewer images per batch, or fewer LoRAs."
                ),
                progress=0,
                completed_images=0,
                outputs=[],
                **pipeline_status_fields(),
            )
        else:
            if loaded_pipeline is None or not restore_cached_pipeline_state(job_id, request):
                pipeline = None
                clear_pipeline()
            release_pipeline_after_job()
            update_job(job_id, status="error", phase="Failed", stage="error", error=generation_failure_message(error), progress=0, completed_images=0, outputs=[], **pipeline_status_fields())
    finally:
        # The shared prompt encodings belong to this job alone; nothing encoded here may reach the next.
        if active_conditioning_cache is not None:
            active_conditioning_cache.clear()
            active_conditioning_cache = None
        gc.collect()
        if release_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        with jobs_lock:
            job_controls.pop(job_id, None)


# The control plane calls health before proxying every /api/inference/* request,
# so this endpoint sits in front of every gallery image, job poll and catalog
# read. Three of its fields are filesystem walks over the model tree — tens of
# gigabytes of safetensors in a real installation — and on Windows, where the
# antivirus inspects those files on access, a cold walk costs seconds. Recomputing
# them per request made opening a gallery card wait on a directory scan.
#
# The scans are therefore memoized for a short window. Capability changes appear
# at most HEALTH_SCAN_TTL_SECONDS late, which is well inside the 2 s cadence the
# control plane already polls at, and every other caller of these functions --
# detector resolution, the upscaler catalog route -- still reads them live.
HEALTH_SCAN_TTL_SECONDS = 3.0
_health_scan_lock = threading.Lock()
_health_scan_cache = None
_health_scan_at = 0.0


def health_scan_fields(force=False):
    global _health_scan_cache, _health_scan_at
    with _health_scan_lock:
        fresh = _health_scan_cache is not None and (time.monotonic() - _health_scan_at) < HEALTH_SCAN_TTL_SECONDS
        if fresh and not force:
            return _health_scan_cache
    # Computed outside the lock: these are slow, and a concurrent burst should
    # queue on the filesystem rather than on each other holding the lock.
    scanned = {
        "detector_models": discover_detector_models(ADETAILER_MODEL_DIRECTORY),
        "anima": anima_health_fields(),
        "flux": flux_health_fields(),
        "flux2": flux2_health_fields(),
        "krea2": krea2_health_fields(),
        "upscalers": upscaler_status(UPSCALER_MODEL_DIRECTORY),
    }
    with _health_scan_lock:
        _health_scan_cache = scanned
        _health_scan_at = time.monotonic()
    return scanned


@app.get("/api/inference/health")
def health():
    scanned = health_scan_fields()
    detector_models = scanned["detector_models"]
    anima_status = scanned["anima"]
    flux_status = scanned["flux"]
    flux2_status = scanned["flux2"]
    krea2_status = scanned["krea2"]
    return {
        "status": "ready",
        "protocol": INFERENCE_PROTOCOL,
        "workspace_id": WORKSPACE_ID,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
        "adetailer_available": adetailer_runtime_available() and bool(detector_models),
        "adetailer_models": len(detector_models),
        "upscalers": scanned["upscalers"],
        "rtx_vsr": rtx_vsr.status(),
        "guidance": {
            "pag": {
                "available": True,
                "implementation": "diffusers_pag",
                "implementations": {
                    "SD": "diffusers_pag",
                    "iL": "diffusers_pag",
                    "Anima": "native_cosmos_identity_self_attention",
                },
                "engines": ["SD", "iL", "Anima"],
                "default_scale": PAG_SCALE,
                "scale_range": [0, 5],
                "default_applied_layers": PAG_APPLIED_LAYERS,
                "applied_layer_options": ["mid", "all"],
            },
            "cfg_zero_star": {
                "available": True,
                "implementation": "diffusers_native_formula",
                "engines": sorted(CFG_ZERO_STAR_ENGINES),
                "planned_engines": [],
                "zero_steps_ratio": 0.04,
                "reason": None,
            },
        },
        "engines": {
            "Anima": anima_status,
            "Flux": flux_status,
            "Flux2": flux2_status,
            "Krea2": krea2_status,
        },
        "background_removal": background_removal_status(),
        "pipeline_configs": {
            family: find_pipeline_config(family, os.environ["HF_HUB_CACHE"]) is not None
            for family in PIPELINE_CONFIG_REPOSITORIES
        },
        "ultralytics_version": installed_version("ultralytics"),
        "loaded_checkpoint": Path(loaded_checkpoint).name if loaded_checkpoint else None,
        "loaded_checkpoint_hash": loaded_checkpoint_hash,
        "loaded_checkpoint_path": loaded_checkpoint_reference(),
        "loaded_engine": loaded_engine,
        "loaded_model_assets": loaded_model_asset_references(),
        "model_cached": loaded_pipeline is not None,
        "memory_mode": reported_memory_mode(active_memory_strategy),
        "memory_label": active_memory_strategy["label"] if active_memory_strategy else None,
        "memory_reason": active_memory_strategy["reason"] if active_memory_strategy else None,
        # AUTO · 动态决策: the requested mode stays `auto`; these say which path the current stage is
        # actually on and every decision the job has made (QPlan P8 §4 step 5).
        **memory_path_fields(active_memory_strategy),
        "memory_mode_label": AUTO_LABEL if performance_settings["memory_mode"] == "auto" else None,
        "memory_mode_warning": (
            BELOW_RECOMMENDED_WARNING
            if performance_settings["memory_mode"] == "auto" and torch.cuda.is_available()
            and below_recommended_vram(torch.cuda.get_device_properties(0).total_memory)
            else None
        ),
        "memory_total_gb": active_memory_strategy["total_gb"] if active_memory_strategy else None,
        "memory_free_gb": active_memory_strategy["free_gb"] if active_memory_strategy else None,
        "memory_weight_gb": active_memory_strategy["weight_gb"] if active_memory_strategy else None,
        "memory_inference_gb": active_memory_strategy["inference_gb"] if active_memory_strategy else None,
        "memory_reserved_gb": active_memory_strategy["reserved_gb"] if active_memory_strategy else None,
        "offload_mode": "cpu_parked" if pipeline_cpu_parked else active_memory_strategy["offload_mode"] if active_memory_strategy else None,
        # Which order the guidance branches run in, and the arithmetic that chose the rung. Both are
        # what a run has to be checked against: the rung alone cannot say whether the activation
        # peak was halved, and the reason string rounds to one decimal place.
        "cfg_mode": active_memory_strategy.get("cfg_mode") if active_memory_strategy else None,
        "memory_admission": active_memory_strategy.get("admission") if active_memory_strategy else None,
        # Where the diffusion model is living, and every move it has made this job.
        "residency": active_residency.snapshot() if active_residency is not None else None,
        "residency_trace": list(active_residency.trace) if active_residency is not None else [],
        "attention_backend": current_attention_backend(),
        "compute_dtype": active_compute_dtype,
        "vae_mode": active_vae_mode,
        "model_resident": bool(active_memory_strategy and active_memory_strategy["model_resident"] and not pipeline_cpu_parked),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "diffusers_version": installed_version("diffusers"),
        "transformers_version": installed_version("transformers"),
        "xformers_version": installed_version("xformers"),
        "platform": platform.platform(),
        "memory_mode_request": performance_settings["memory_mode"],
        "performance_settings": performance_settings.copy(),
    }


def performance_payload():
    cuda_available = torch.cuda.is_available()
    capability = torch.cuda.get_device_capability() if cuda_available else (0, 0)
    try:
        import psutil
        ram_bytes = psutil.virtual_memory().total
        swap_bytes = psutil.swap_memory().total
    except Exception:
        ram_bytes = 0
        swap_bytes = 0
    if cuda_available:
        physical_vram = int(torch.cuda.get_device_properties(0).total_memory)
        minimum_vram, maximum_vram, platform_reserve = vram_limit_bounds(
            physical_vram,
            allow_shared_memory=performance_settings["allow_shared_memory"],
        )
        effective_limit = effective_vram_limit_bytes(
            physical_vram,
            performance_settings.get("vram_limit_gb", 0.0),
            allow_shared_memory=performance_settings["allow_shared_memory"],
        )
    else:
        physical_vram = minimum_vram = maximum_vram = platform_reserve = effective_limit = 0
    return {
        "settings": performance_settings.copy(),
        "active": {
            "memory_mode": reported_memory_mode(active_memory_strategy),
            "memory_label": active_memory_strategy["label"] if active_memory_strategy else None,
            "memory_path": (active_memory_strategy or {}).get("memory_path") if stage_scheduling_active() else None,
            "memory_path_label": (
                path_label(active_memory_strategy.get("memory_path"))
                if stage_scheduling_active() and active_memory_strategy.get("memory_path") else None
            ),
            "attention_backend": current_attention_backend(),
            "compute_dtype": active_compute_dtype,
            "vae_mode": active_vae_mode,
            "model_cached": loaded_pipeline is not None,
            "vram_limit_bytes": effective_limit,
            "transformer_resident": bool(
                loaded_pipeline is not None and getattr(loaded_pipeline, "transformer_resident", False)
            ),
        },
        "capabilities": {
            "cuda": cuda_available,
            "device": torch.cuda.get_device_name(0) if cuda_available else None,
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory if cuda_available else 0,
            "vram_limit": {
                "minimum_bytes": minimum_vram,
                "maximum_bytes": maximum_vram,
                "effective_bytes": effective_limit,
                "platform_reserve_bytes": platform_reserve,
                "automatic": performance_settings.get("vram_limit_gb", 0.0) <= 0,
            },
            "ram_bytes": ram_bytes,
            "swap_bytes": swap_bytes,
            "compute_capability": f"{capability[0]}.{capability[1]}" if cuda_available else None,
            "bf16": bool(cuda_available and torch.cuda.is_bf16_supported()),
            "sdpa": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
            "xformers": installed_version("xformers") is not None,
            # SageAttention 1.x is pure Triton, so `sage` is only selectable when
            # both are present; compilation needs Triton alone.
            "triton": find_spec("triton") is not None,
            "sage": installed_version("sageattention") is not None and find_spec("triton") is not None,
            "nvfp4": nvfp4_capabilities(torch),
            # P9: every ComfyUI layout the loaders read, and how each runs on this card.
            "quantization": quantization_capabilities(),
            # AUTO · 动态决策 stays selectable below the recommended 6 GB; this is only the notice.
            "auto_below_recommended": bool(
                cuda_available and below_recommended_vram(torch.cuda.get_device_properties(0).total_memory)
            ),
            "auto_recommended_minimum_bytes": RECOMMENDED_MINIMUM_VRAM_BYTES,
        },
    }


@app.get("/api/inference/performance")
def get_performance_settings():
    return performance_payload()


@app.put("/api/inference/performance")
def update_performance_settings(request: PerformanceInput):
    with pipeline_lock:
        with jobs_lock:
            busy = any(job["status"] in JOB_ACTIVE_STATUSES for job in jobs.values())
        if busy:
            raise HTTPException(status_code=409, detail="Cannot change performance settings while generation is active")
        requested = request.model_dump()
        if torch.cuda.is_available() and requested["vram_limit_gb"] > 0:
            total = int(torch.cuda.get_device_properties(0).total_memory)
            minimum, maximum, _reserve = vram_limit_bounds(
                total, allow_shared_memory=requested["allow_shared_memory"]
            )
            requested["vram_limit_gb"] = round(
                min(maximum, max(minimum, int(requested["vram_limit_gb"] * GIB))) / GIB,
                1,
            )
        saved = write_performance_settings(PERFORMANCE_SETTINGS_FILE, requested)
        performance_settings.clear()
        performance_settings.update(saved)
        clear_pipeline()
        configure_cuda_memory_limit()
    return performance_payload()


@app.get("/api/inference/adetailer/models")
def list_adetailer_models():
    models = discover_detector_models(ADETAILER_MODEL_DIRECTORY)
    builtins = builtin_yolo_models()
    builtin_by_name = {model["name"]: model for model in builtins}
    installed = set(models)
    preferred = next((name for name in models if name == "face_yolov8n.pt"), models[0] if models else "")
    return {
        "available": adetailer_runtime_available() and bool(models),
        "runtime_available": adetailer_runtime_available(),
        "models": [
            {
                "name": Path(name).name,
                "label": builtin_by_name[name]["label"] if name in builtin_by_name else Path(name).name,
                "value": name,
                "source": "official" if name in builtin_by_name else "community",
            }
            for name in models
        ],
        "builtins": [{**model, "installed": model["name"] in installed} for model in builtins],
        "default": preferred,
        "directory": str(ADETAILER_MODEL_DIRECTORY),
        "python": str(sys.executable),
    }


@app.get("/api/inference/upscalers")
def list_upscaler_models():
    return upscaler_status(UPSCALER_MODEL_DIRECTORY)


@app.post("/api/inference/shutdown", include_in_schema=False)
def shutdown(x_shutdown_token: str | None = Header(default=None)):
    global shutdown_requested
    expected_token = os.environ.get("INFERENCE_SHUTDOWN_TOKEN")
    if not expected_token or x_shutdown_token != expected_token:
        raise HTTPException(status_code=404, detail="Not found")
    shutdown_requested = True
    with jobs_lock:
        controls = list(job_controls.values())
    for control in controls:
        control.cancel()

    def stop_when_idle():
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with jobs_lock:
                if not job_controls:
                    break
            time.sleep(0.1)
        signal.raise_signal(signal.SIGINT)

    threading.Thread(target=stop_when_idle, name="shutdown", daemon=True).start()
    return {"status": "shutting_down"}


@app.delete("/api/inference/model-cache")
def unload_model_cache(engine: Literal["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"] | None = None, checkpoint: str | None = None):
    with pipeline_lock:
        with jobs_lock:
            busy = any(job["status"] in JOB_ACTIVE_STATUSES for job in jobs.values())
        if busy:
            raise HTTPException(status_code=409, detail="Cannot switch the loaded model while generation is active")
        if engine is None and checkpoint is None:
            clear_pipeline()
            return {"status": "released", "model_cached": False}
        if engine is None or not checkpoint:
            raise HTTPException(status_code=400, detail="Engine and checkpoint must be provided together")
        checkpoint_root = (
            native_model_roots(engine)["diffusion_model"] if engine in NATIVE_ENGINES else model_roots(engine)[0]
        )
        if is_shared_ref(checkpoint):
            # Releasing a shared checkpoint must resolve the same way loading
            # it did, or the cache could never be freed for that model.
            try:
                expected_path = resolve_shared_model(checkpoint, PROJECT_ROOT, CHECKPOINT_EXTENSIONS, "Checkpoint")
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
        else:
            relative_checkpoint = Path(checkpoint)
            if relative_checkpoint.is_absolute():
                raise HTTPException(status_code=400, detail="Checkpoint path must be relative")
            expected_path = (checkpoint_root / relative_checkpoint).resolve(strict=False)
            if expected_path == checkpoint_root or checkpoint_root not in expected_path.parents:
                raise HTTPException(status_code=400, detail="Checkpoint is outside its configured model directory")
        expected_checkpoint = str(expected_path)
        if loaded_engine != engine or loaded_checkpoint != expected_checkpoint:
            return {
                "status": "retained",
                "model_cached": loaded_pipeline is not None,
                "loaded_engine": loaded_engine,
                "loaded_checkpoint": loaded_checkpoint_reference(),
                "loaded_model_assets": loaded_model_asset_references(),
            }
        clear_pipeline()
    return {"status": "released", "model_cached": False}


def cuda_device_identity():
    """Identify the physical card torch is bound to, so an `nvidia-smi` row can be matched to it.

    ``CUDA_VISIBLE_DEVICES`` renumbers devices and ``CUDA_DEVICE_ORDER=FASTEST_FIRST`` reorders
    them, so torch's device 0 is not necessarily `nvidia-smi`'s row 0. The UUID settles it wherever
    torch exposes one; the remapped driver index is the fallback for builds that do not.
    """
    if not torch.cuda.is_available():
        return None, None
    device_uuid = None
    try:
        device_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    except Exception:
        device_uuid = None
    device_index = None
    try:
        logical = int(torch.cuda.current_device())
        visible = [part.strip() for part in (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",") if part.strip()]
        if not visible:
            device_index = str(logical)
        elif logical < len(visible):
            device_index = visible[logical]
        else:
            device_index = visible[0]
    except Exception:
        device_index = None
    return device_uuid, device_index


def psutil_module():
    try:
        import psutil

        return psutil
    except Exception:
        return None


@app.get("/api/inference/hardware")
def hardware():
    device_uuid, device_index = cuda_device_identity()
    stats = probe_hardware(
        torch_module=torch,
        psutil_module=psutil_module(),
        smi=gpu_sensor_sampler,
        cpu=cpu_sampler,
        device_uuid=device_uuid,
        device_index=device_index,
    )
    stats["cuda"] = torch.cuda.is_available()
    stats["rtx_vsr"] = rtx_vsr.status()
    if torch.cuda.is_available():
        if not stats.get("gpu_name"):
            stats["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            # The wall the allocator and the admission check both enforce. Without it the bar has
            # no way to show that a card is "full" at 20 GB of 24 because that is where the cap is.
            physical_total = int(torch.cuda.get_device_properties(0).total_memory)
            stats["vram_limit_mb"] = round(
                effective_vram_limit_bytes(
                    physical_total,
                    performance_settings.get("vram_limit_gb", 0.0),
                    allow_shared_memory=performance_settings["allow_shared_memory"],
                )
                / 1024**2
            )
        except Exception:
            pass
    if active_memory_strategy:
        stats.update({
            "memory_mode": reported_memory_mode(active_memory_strategy),
            "memory_label": active_memory_strategy["label"],
            "memory_reason": active_memory_strategy["reason"],
            "offload_mode": active_memory_strategy["offload_mode"],
            "model_resident": active_memory_strategy["model_resident"],
            "model_cached": loaded_pipeline is not None,
            "attention_backend": current_attention_backend(),
            "loaded_checkpoint": Path(loaded_checkpoint).name if loaded_checkpoint else None,
        })
    return stats


@app.post("/api/inference/jobs", status_code=202)
def create_job(request: GenerateInput):
    if shutdown_requested:
        raise HTTPException(status_code=503, detail="Inference service is shutting down")
    submitted_request = request
    try:
        request, inline_lora_audit = preflight_inline_lora_request(request)
    except (InlineLoraError, ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    try:
        validate_latent_resample_request(request)
        validate_diffusers_postprocess_schedules(request)
        prompt_encoding_audit = preflight_dynamic_prompt_request(request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if request.engine in NATIVE_ENGINES:
        if performance_settings["memory_mode"] == "ultra_low_vram":
            raise HTTPException(
                status_code=422, detail=f"Native {request.engine} does not support ultra-low-memory mode"
            )
    if performance_settings["memory_mode"] == "ultra_low_vram" and postprocessing_stages(request):
        raise HTTPException(status_code=422, detail="Ultra-low-memory mode does not support Hires.fix, ADetailer, or RTX VSR")
    if request.rtx.enabled:
        rtx_status = rtx_vsr.status()
        if rtx_status.get("probing"):
            raise HTTPException(status_code=503, detail="RTX VSR runtime verification is still in progress")
        if not rtx_status.get("available"):
            raise HTTPException(status_code=422, detail=rtx_status.get("reason") or "RTX VSR is unavailable")
    if request.adetailer.enabled:
        if not adetailer_runtime_available():
            raise HTTPException(status_code=422, detail="ADetailer requires the project's ultralytics package. Run npm run setup.")
        # Every unit is checked before any of them runs: finding a missing
        # detector three passes in would waste the two that already succeeded.
        for index, unit in enumerate(request.adetailer.active_units):
            try:
                resolve_detector_model(ADETAILER_MODEL_DIRECTORY, unit.detector)
            except ValueError as error:
                label = f"ADetailer {index + 1}" if len(request.adetailer.units) > 1 else "ADetailer"
                raise HTTPException(status_code=422, detail=f"{label}: {error}") from error
    for unit in request.active_resample_units:
        if not unit.upscaler:
            continue
        try:
            resolve_upscaler_model(UPSCALER_MODEL_DIRECTORY, unit.upscaler)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    if request.source_image.enabled:
        # Decoded here as well as during the run so an unreadable picture is a 422 on submit rather
        # than a job that queues, loads a model and only then fails. The decoded copy is discarded:
        # the request is the one carrier of the pixels, and holding a second one for the lifetime of
        # the job would double the cost of a large source.
        try:
            decoded_source = decode_source_image(request.source_image.image_data)
            # Post-processing keeps the picture at its own size, so the envelope has to be checked
            # against the decoded pixels rather than against the size the client declared.
            if request.postprocess_only:
                validate_postprocess_source_size(decoded_source.size)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    if request.output_template is not None:
        # The folders a template names are checked now, against the disk as it is: a segment that
        # is already a link, a junction or a file refuses the job before anything runs.
        try:
            output_target(OUTPUT_DIRECTORY, request.output_template, moment=datetime.now(), engine=request.engine, create=False)
        except OutputPathError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    resource_issue = expanded_canvas_resource_issue(request)
    if resource_issue:
        raise HTTPException(status_code=422, detail=resource_issue)
    _, directives = parse_prompt_directives(request.prompt)
    if directives["transparent_background"]:
        if not request.background_removal_model:
            raise HTTPException(status_code=422, detail="请选择透明背景模型")
        try:
            require_background_removal_model(request.background_removal_model)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    # This bearer is returned only to the submitting browser. The record holds its digest, so a
    # second LAN client that can list job IDs cannot read another client's saved Prompt snapshot.
    gallery_access_token = secrets.token_urlsafe(32) if request.gallery_settings is not None else None
    with jobs_lock:
        # Submitting while another job runs queues this one rather than refusing it. The request is
        # validated and captured now, above, so the job carries the parameters and the model the
        # user had selected at the moment they submitted — changing either afterwards has no effect
        # on what is already in the queue.
        job_id = uuid.uuid4().hex
        job_controls[job_id] = JobControl()
        trim_job_history()
        jobs[job_id] = {
            "id": job_id,
            "sequence": next(job_sequence),
            "status": "queued",
            "phase": "Queued",
            "stage": "queued",
            "stage_step": 0,
            "stage_total": 0,
            "progress": 0,
            "step": 0,
            "total_steps": request.steps,
            # Carried on the record so the queue list can describe a job that has not started and
            # has no diagnostics of its own yet.
            "width": request.width,
            "height": request.height,
            "images_per_batch": request.images_per_batch,
            "batch_count": request.batch_count,
            "batch_index": 0,
            "completed_images": 0,
            "total_images": request.images_per_batch * request.batch_count,
            "outputs": [],
            "stage_previews": [],
            "seed": str(request.seed),
            "base_seed": str(request.seed),
            "hires_seed_mode": request.resample_units[0].seed_mode if request.resample_units else "inherit",
            "hires_seed": None,
            "transparent_background": directives["transparent_background"],
            # The picture itself never reaches the job record: it is polled several times a second
            # by the workspace, and a megabyte of base64 in every response would be the dominant
            # cost of watching a run.
            "source_image_enabled": request.source_image.enabled,
            "source_image_name": request.source_image.name or None,
            "source_image_resize_mode": request.source_image.resize_mode if request.source_image.enabled else None,
            "img2img_fix_steps": uses_diffusers_img2img_fix_steps(request),
            "img2img_schedule_steps": img2img_schedule_steps(request),
            # The effective layer, never the raw field: an absent SD/iL value records what ran.
            "clip_skip": effective_clip_skip(request),
            # Relative to the output root and frozen with the job; `None` is the historical scheme.
            "output_template": request.output_template,
            "background_removal_model": request.background_removal_model,
            "background_removal": None,
            "resample_units": len(request.active_resample_units),
            "rtx_enabled": request.rtx.enabled,
            "rtx_scale": request.rtx.scale if request.rtx.enabled else None,
            "rtx_quality": request.rtx.quality if request.rtx.enabled else None,
            "postprocess_chain": [stage.model_dump() for stage in request.postprocess_chain],
            "postprocess_stages": postprocess_stage_types(request),
            "postprocess_only": request.postprocess_only,
            "created_at": time.time(),
            "error": None,
            "warning": None,
            # The submitted source is immutable audit data. Runtime sampling uses
            # the stripped request produced above, while PNG metadata and failed
            # jobs retain exactly what the client sent.
            "prompt": submitted_request.prompt,
            "negative_prompt": submitted_request.negative_prompt,
            "prompt_encoding": copy.deepcopy(prompt_encoding_audit),
            "prompt_tokens": None,
            "prompt_blocks": None,
            "prompt_weighted_tokens": None,
            "negative_prompt_tokens": None,
            "negative_prompt_blocks": None,
            "negative_prompt_weighted_tokens": None,
            "inline_loras": copy.deepcopy(inline_lora_audit),
            "adetailer_state": None,
            "elapsed_seconds": 0.0,
            "paused_seconds": 0.0,
            "memory_mode": None,
            "memory_label": None,
            "memory_reason": None,
            "attention_backend": None,
            "compute_dtype": None,
            "vae_mode": None,
            "offload_mode": None,
            "model_resident": False,
            "model_cached": loaded_pipeline is not None,
            "model_reused": False,
            "runtime_metrics": None,
            "loaded_checkpoint": Path(loaded_checkpoint).name if loaded_checkpoint else None,
            "loaded_model_assets": loaded_model_asset_references(),
            "loaded_engine": loaded_engine,
            "requested_engine": request.engine,
            "requested_checkpoint": public_model_reference(
                request.diffusion_model if request.engine in NATIVE_ENGINES else request.checkpoint
            ),
            "requested_model_assets": {
                key: value
                for key, value in {
                    "diffusion_model": public_model_reference(request.diffusion_model),
                    "text_encoder": public_model_reference(request.text_encoder),
                    "text_encoder_2": public_model_reference(request.text_encoder_2),
                    "vae": public_model_reference(request.vae),
                }.items()
                if value is not None
            } if request.engine in NATIVE_ENGINES else {},
            "gallery_settings": copy.deepcopy(request.gallery_settings),
            "gallery_access_token_digest": gallery_access_digest(gallery_access_token) if gallery_access_token else None,
        }
        # Snapshot before the worker can write progress fields, so response serialization never
        # races with `update_job()` on the shared record.
        response = public_job_view(jobs[job_id])
        if gallery_access_token:
            response["gallery_access_token"] = gallery_access_token
    executor.submit(run_generation, job_id, request, submitted_request, inline_lora_audit)
    return response


@app.get("/api/inference/jobs/active")
def get_active_job():
    with jobs_lock:
        active = [job for job in jobs.values() if job["status"] in JOB_ACTIVE_STATUSES]
        if not active:
            return {"job": None}
        # The job being worked on, not the newest submission. With a queue those differ, and a
        # client recovering its progress view wants the one that is actually moving.
        started = [job for job in active if job["status"] != "queued"]
        job = min(started or active, key=lambda item: item.get("sequence", 0))
        return {"job": public_job_view(job)}


@app.get("/api/inference/jobs")
def list_jobs():
    """Every job this session has seen, oldest first.

    The registry is in memory, so this is a session queue rather than a durable history: restarting
    the package empties it and numbering starts again at one.
    """
    with jobs_lock:
        ordered = sorted(jobs.values(), key=lambda item: item.get("sequence", 0))
        entries = []
        running_seen = False
        for job in ordered:
            entries.append(job_queue_entry(job, running_seen))
            if job["status"] in JOB_ACTIVE_STATUSES and job["status"] != "queued":
                running_seen = True
        counts = {"queued": 0, "running": 0, "complete": 0, "error": 0, "cancelled": 0}
        for job in ordered:
            status = job["status"]
            key = "running" if status in JOB_ACTIVE_STATUSES and status != "queued" else status
            if key in counts:
                counts[key] += 1
        return {"jobs": entries, "counts": counts, "total": len(entries)}


@app.get("/api/inference/jobs/{job_id}")
def get_job(job_id: str, x_xirai_gallery_access: str | None = Header(default=None)):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Generation job not found")
        payload = public_job_view(job)
        control = job_controls.get(job_id)
        completed_gallery_settings = (
            copy.deepcopy(job.get("gallery_settings"))
            if job["status"] in JOB_FINISHED_STATUSES
            and job.get("gallery_settings") is not None
            and has_gallery_access(job, x_xirai_gallery_access)
            else None
        )
    if control and job.get("started_at"):
        payload["elapsed_seconds"] = round(control.active_elapsed(job["started_at"]), 1)
        payload["paused_seconds"] = round(control.total_paused(), 1)
    if completed_gallery_settings is not None:
        payload["gallery_settings"] = completed_gallery_settings
    return payload


def get_controllable_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        control = job_controls.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Generation job not found")
        if not control or job["status"] in {"complete", "error", "cancelled"}:
            raise HTTPException(status_code=409, detail="Generation job is no longer controllable")
        return control, job["status"]


@app.post("/api/inference/jobs/{job_id}/pause")
def pause_job(job_id: str):
    control, status = get_controllable_job(job_id)
    if status in {"paused", "pausing"}:
        return {"status": status}
    control.request_pause()
    update_job(job_id, status="pausing", phase="Pausing after current step")
    return {"status": "pausing"}


@app.post("/api/inference/jobs/{job_id}/resume")
def resume_job(job_id: str):
    control, status = get_controllable_job(job_id)
    if status not in {"paused", "pausing"}:
        raise HTTPException(status_code=409, detail="Generation job is not paused")
    control.resume()
    update_job(job_id, status="running", phase="Resuming")
    return {"status": "running"}


@app.post("/api/inference/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    control, status = get_controllable_job(job_id)
    if status == "cancelling":
        return {"status": status}
    control.cancel()
    if status == "queued":
        # Nothing has started, so there is no current step to stop after. Reporting it as cancelled
        # straight away keeps a job removed from the back of the queue from sitting in "stopping"
        # until the worker finally reaches it — which, behind a long queue, could be many minutes.
        # `run_generation` sees the same cancelled flag and drops it without loading anything.
        update_job(
            job_id, status="cancelled", stage="cancelled", phase="Cancelled before starting",
            progress=0, completed_at=time.time(),
        )
        return {"status": "cancelled"}
    update_job(job_id, status="cancelling", phase="Stopping after current step")
    return {"status": "cancelling"}


@app.get("/api/inference/jobs/{job_id}/stages/{index}")
def get_job_stage_preview(job_id: str, index: int):
    """One finished stage's picture, still available after the job ends.

    This survives the run: the whole point is to compare what each stage did once they have all
    finished.
    """
    with jobs_lock:
        job = jobs.get(job_id)
        paths = (job.get("stage_preview_paths") or []) if job else []
        stage_path = paths[index] if 0 <= index < len(paths) else None
    if not stage_path or not Path(stage_path).is_file():
        raise HTTPException(status_code=404, detail="Stage preview is not available")
    return FileResponse(stage_path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/inference/jobs/{job_id}/image")
def get_job_image(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or job.get("status") != "complete" or not job.get("output_path"):
            raise HTTPException(status_code=404, detail="Generated image is not available")
        output_path = job["output_path"]
    return FileResponse(output_path, media_type="image/png", filename=Path(output_path).name, headers={"Cache-Control": "no-store"})


@app.get("/api/inference/jobs/{job_id}/images/{image_index}")
def get_job_image_at(job_id: str, image_index: int):
    with jobs_lock:
        job = jobs.get(job_id)
        output_paths = job.get("output_paths", []) if job else []
        if not job or job.get("status") != "complete" or image_index < 0 or image_index >= len(output_paths):
            raise HTTPException(status_code=404, detail="Generated image is not available")
        output_path = output_paths[image_index]
    if not Path(output_path).is_file():
        raise HTTPException(status_code=404, detail="Generated image file is missing")
    return FileResponse(output_path, media_type="image/png", filename=Path(output_path).name, headers={"Cache-Control": "no-store"})


@app.get("/api/inference/output-template")
def get_output_template(template: str | None = None, engine: str = "iL"):
    """What Settings → General shows: the active root, the placeholder legend and a path preview.

    The preview runs the same parser and directory walk a save does, without creating anything, so
    a template the page calls valid is one a job will accept. Only relative locations are returned
    for the preview; the root itself is named relative to the project when it lives inside it.
    """
    if engine not in {"SD", "iL", "Anima", "Flux", "Flux2", "Krea2"}:
        raise HTTPException(status_code=422, detail="Unsupported engine")
    payload = {
        "root": output_root_display(OUTPUT_DIRECTORY, PROJECT_ROOT),
        "default_template": DEFAULT_OUTPUT_TEMPLATE,
        "placeholders": [{"name": name, "label": label} for name, label in OUTPUT_TEMPLATE_PLACEHOLDERS.items()],
        "template": template,
        "valid": True,
        "legacy": True,
        "error": "",
        "preview": "",
    }
    try:
        normalized = normalize_output_template(template)
        target = output_target(OUTPUT_DIRECTORY, normalized, moment=datetime.now(), engine=engine, create=False)
        name = preview_output_name(target)
        payload.update({
            "legacy": normalized is None,
            "normalized": normalized,
            "preview": f"{target.relative_directory}/{name}" if target.relative_directory else name,
        })
    except OutputPathError as error:
        payload.update({"valid": False, "error": str(error)})
    return payload


@app.get("/api/inference/history")
def get_history(folder: str | None = None):
    selected_folder = history_folder_path(folder) if folder else None
    directory = history_directory_listing(selected_folder)
    return {
        "started_at": HISTORY_STARTED_AT,
        "scope": "folder" if selected_folder else "session",
        "selected_folder": folder or "",
        "directory": directory,
        "folders": directory["folders"],
        "cards": list_history_cards(selected_folder, session_only=selected_folder is None),
    }


def gallery_store():
    return GalleryStore(STATE_DIRECTORY, asset_resolver=history_asset_path)


def gallery_call(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except GalleryConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except GalleryNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except GalleryValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except GalleryStorageError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/inference/gallery")
def get_gallery(collection: str | None = None):
    store = gallery_store()
    return gallery_call(store.list_gallery, collection)


@app.post("/api/inference/gallery/collections", status_code=201)
def create_gallery_collection(input_data: GalleryCollectionCreateInput):
    store = gallery_store()
    return gallery_call(store.create_collection, input_data.id, input_data.description)


@app.patch("/api/inference/gallery/collections/{collection_id}")
def update_gallery_collection(collection_id: str, input_data: GalleryCollectionUpdateInput):
    store = gallery_store()
    fields = input_data.model_fields_set
    return gallery_call(
        store.update_collection,
        collection_id,
        new_id=input_data.id if "id" in fields else GALLERY_UNSET,
        description=input_data.description if "description" in fields else GALLERY_UNSET,
    )


@app.delete("/api/inference/gallery/collections/{collection_id}")
def delete_gallery_collection(collection_id: str):
    store = gallery_store()
    gallery_call(store.delete_collection, collection_id)
    return {"deleted": collection_id}


@app.post("/api/inference/gallery/cards", status_code=201)
def create_gallery_card(input_data: GalleryCardCreateInput):
    store = gallery_store()
    images = [image.model_dump(exclude_none=True) for image in input_data.images]
    return gallery_call(
        store.create_card,
        input_data.collection_id,
        input_data.title,
        input_data.settings,
        images,
    )


@app.post("/api/inference/gallery/cards/bulk", status_code=201)
def create_gallery_cards(input_data: GalleryCardsCreateInput):
    store = gallery_store()
    cards = [
        {
            "title": card.title,
            "settings": card.settings,
            "images": [image.model_dump(exclude_none=True) for image in card.images],
        }
        for card in input_data.cards
    ]
    return {"cards": gallery_call(store.create_cards, input_data.collection_id, cards)}


@app.patch("/api/inference/gallery/cards/{card_id}")
def update_gallery_card(card_id: str, input_data: GalleryCardUpdateInput):
    store = gallery_store()
    fields = input_data.model_fields_set
    images = (
        [image.model_dump(exclude_none=True) for image in input_data.images]
        if "images" in fields
        else GALLERY_UNSET
    )
    return gallery_call(
        store.update_card,
        card_id,
        collection_id=input_data.collection_id if "collection_id" in fields else GALLERY_UNSET,
        title=input_data.title if "title" in fields else GALLERY_UNSET,
        settings=input_data.settings if "settings" in fields else GALLERY_UNSET,
        images=images,
    )


@app.delete("/api/inference/gallery/cards/{card_id}")
def delete_gallery_card(card_id: str):
    store = gallery_store()
    gallery_call(store.delete_card, card_id)
    return {"deleted": card_id}


@app.put("/api/inference/gallery/collections/{collection_id}/card-order")
def reorder_gallery_cards(collection_id: str, input_data: GalleryCardOrderInput):
    store = gallery_store()
    return {"cards": gallery_call(store.reorder_cards, collection_id, input_data.card_ids)}


@app.get("/api/inference/gallery/prompts")
def get_gallery_prompts():
    store = gallery_store()
    return {"prompts": gallery_call(store.list_prompt_entries)}


@app.post("/api/inference/gallery/prompts", status_code=201)
def create_gallery_prompt(input_data: GalleryPromptCreateInput):
    store = gallery_store()
    return gallery_call(
        store.create_prompt_entry,
        input_data.title,
        input_data.positive_prompt,
        input_data.negative_prompt,
        input_data.notes,
    )


@app.patch("/api/inference/gallery/prompts/{prompt_id}")
def update_gallery_prompt(prompt_id: str, input_data: GalleryPromptUpdateInput):
    store = gallery_store()
    fields = input_data.model_fields_set
    return gallery_call(
        store.update_prompt_entry,
        prompt_id,
        title=input_data.title if "title" in fields else GALLERY_UNSET,
        positive_prompt=input_data.positive_prompt if "positive_prompt" in fields else GALLERY_UNSET,
        negative_prompt=input_data.negative_prompt if "negative_prompt" in fields else GALLERY_UNSET,
        notes=input_data.notes if "notes" in fields else GALLERY_UNSET,
    )


@app.delete("/api/inference/gallery/prompts/{prompt_id}")
def delete_gallery_prompt(prompt_id: str):
    store = gallery_store()
    gallery_call(store.delete_prompt_entry, prompt_id)
    return {"deleted": prompt_id}


@app.get("/api/inference/gallery/images/{image_id}")
def get_gallery_image(image_id: str, variant: str = "original"):
    store = gallery_store()
    image = gallery_call(store.get_image_file, image_id, variant)
    return FileResponse(
        image["path"],
        media_type=image["mime_type"],
        # No Content-Disposition: this is rendered by <img>, not downloaded, and
        # an attachment disposition only confuses caches and save dialogs.
        headers={
            "Cache-Control": "private, max-age=31536000, immutable",
            # A thumbnail that could not be produced falls back to the original,
            # so the client is told which one it actually received.
            "X-Gallery-Variant": image["variant"],
        },
    )


@app.get("/api/inference/history/assets/{asset_id}")
def get_history_asset(asset_id: str):
    path = history_asset_path(asset_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image asset is missing")
    return FileResponse(path, media_type="image/gif" if path.suffix.lower() == ".gif" else "image/png", filename=path.name, headers={"Cache-Control": "no-store"})


@app.delete("/api/inference/history")
def delete_history(input_data: HistoryDeleteInput):
    deleted = []
    missing = []
    for asset_id in input_data.asset_ids:
        try:
            path = history_asset_path(asset_id)
        except HTTPException:
            missing.append(asset_id)
            continue
        if not path.is_file():
            missing.append(asset_id)
            continue
        try:
            if input_data.delete_source:
                path.unlink()
                deleted.append(asset_id)
            else:
                deleted.append(asset_id)
        except OSError as error:
            raise HTTPException(status_code=500, detail=f"Unable to delete {path.name}: {error}") from error
    return {"deleted": deleted, "missing": missing, "cards": list_history_cards()}


@app.post("/api/inference/history/copy")
def copy_history_asset(input_data: dict):
    asset_id = input_data.get("asset_id")
    if not isinstance(asset_id, str):
        raise HTTPException(status_code=422, detail="asset_id is required")
    source = history_asset_path(asset_id)
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Image asset is missing")
    try:
        metrics = history_image_resource_metrics(source, COLLAGE_MAX_HISTORY_COPY_BYTES, "history copy source", decode_frames=True)
        if metrics["format"] == "GIF":
            if metrics["bytes"] > COLLAGE_MAX_ENCODED_OUTPUT_BYTES:
                raise CollageResourceLimitError("GIF copy exceeds the encoded output budget")
            return Response(content=source.read_bytes(), media_type="image/gif", headers={"Cache-Control": "no-store"})
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as image:
                image.load()
                copied = image.copy()
                copied.info.clear()
        raw_buffer, bounded_buffer = bounded_output_buffer()
        try:
            copied.save(bounded_buffer, format="PNG", optimize=False)
            content = raw_buffer.getvalue()
        finally:
            copied.close()
            raw_buffer.close()
    except (CollageResourceLimitError, Image.DecompressionBombWarning, Image.DecompressionBombError, MemoryError, OverflowError) as error:
        raise HTTPException(status_code=413, detail=f"History copy resource budget exceeded: {error}") from error
    except (OSError, SyntaxError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Unable to copy image") from error
    return Response(content=content, media_type="image/png", headers={"Cache-Control": "no-store"})


_COLLAGE_SAVE_LOCK = threading.Lock()
_COLLAGE_SAVE_RESULT_TTL_SECONDS = 3600
_COLLAGE_SAVE_RESULT_LIMIT = 256
_COLLAGE_SAVE_RESULTS: dict[str, tuple[float, str, dict]] = {}


def prune_collage_save_results(now: float | None = None):
    current = time.monotonic() if now is None else now
    for key, (expires_at, _digest, _result) in list(_COLLAGE_SAVE_RESULTS.items()):
        if expires_at <= current:
            _COLLAGE_SAVE_RESULTS.pop(key, None)
    while len(_COLLAGE_SAVE_RESULTS) > _COLLAGE_SAVE_RESULT_LIMIT:
        _COLLAGE_SAVE_RESULTS.pop(next(iter(_COLLAGE_SAVE_RESULTS)))


@app.post("/api/inference/collages", status_code=201)
def save_collage(input_data: CollageInput):
    image = None
    frames = []
    output_path = None
    completed = False
    try:
        raw = decode_collage_data_url(input_data.image_data, COLLAGE_MAX_DECODED_BYTES, "Collage image")
        digest = hashlib.sha256(raw)
        digest.update(input_data.name.encode("utf-8"))
        if input_data.manual_layout:
            digest.update(json.dumps(input_data.manual_layout, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        request_digest = digest.hexdigest()
        with _COLLAGE_SAVE_LOCK:
            now = time.monotonic()
            prune_collage_save_results(now)
            if input_data.idempotency_key:
                cached = _COLLAGE_SAVE_RESULTS.get(input_data.idempotency_key)
                if cached:
                    if cached[1] != request_digest:
                        raise HTTPException(status_code=409, detail="Idempotency key was already used for different collage data")
                    return dict(cached[2])
            image = open_budgeted_collage_image(raw, COLLAGE_MAX_PIXELS, "collage image")
            if image.format not in {"PNG", "GIF", "JPEG", "WEBP"}:
                raise ValueError("Unsupported collage image format")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                frame_count = max(1, int(getattr(image, "n_frames", 1)))
            if frame_count > COLLAGE_MAX_GIF_FRAMES or image.width * image.height * frame_count > COLLAGE_MAX_TOTAL_FRAME_PIXELS:
                raise CollageResourceLimitError("Collage GIF exceeds the frame or total frame-pixel budget")
            is_gif = image.format == "GIF"
            requested_name = input_data.name if is_gif else f"{Path(input_data.name).stem}.png"
            output_path = create_named_output_path(requested_name)
            if is_gif:
                durations = []
                for frame_index in range(frame_count):
                    image.seek(frame_index)
                    frames.append(image.convert("RGBA"))
                    durations.append(normalize_collage_gif_duration(image.info.get("duration")))
                comment = json.dumps({"manual_layout": input_data.manual_layout}, ensure_ascii=False).encode("utf-8") if input_data.manual_layout else None
                with output_path.open("w+b") as output_file:
                    frames[0].save(BoundedOutputWriter(output_file, COLLAGE_MAX_ENCODED_OUTPUT_BYTES), format="GIF", save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=2, optimize=False, comment=comment)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    image.load()
                metadata = PngInfo()
                if input_data.manual_layout:
                    metadata.add_text("parameters", json.dumps({"manual_layout": input_data.manual_layout}, ensure_ascii=False))
                converted = image.convert("RGBA")
                try:
                    with output_path.open("w+b") as output_file:
                        converted.save(BoundedOutputWriter(output_file, COLLAGE_MAX_ENCODED_OUTPUT_BYTES), format="PNG", optimize=False, pnginfo=metadata)
                finally:
                    converted.close()
            token = history_asset_token(output_path)
            result = {"name": output_path.name, "url": f"/api/inference/history/assets/{token}", "id": token}
            if input_data.idempotency_key:
                if len(_COLLAGE_SAVE_RESULTS) >= _COLLAGE_SAVE_RESULT_LIMIT:
                    _COLLAGE_SAVE_RESULTS.pop(next(iter(_COLLAGE_SAVE_RESULTS)))
                _COLLAGE_SAVE_RESULTS[input_data.idempotency_key] = (now + _COLLAGE_SAVE_RESULT_TTL_SECONDS, request_digest, dict(result))
            completed = True
            return result
    except HTTPException:
        raise
    except (CollageResourceLimitError, Image.DecompressionBombWarning, Image.DecompressionBombError, MemoryError, OverflowError) as error:
        raise HTTPException(status_code=413, detail=f"Collage resource budget exceeded: {error}") from error
    except (ValueError, binascii.Error, OSError, SyntaxError) as error:
        raise HTTPException(status_code=400, detail=f"Invalid collage image: {error}") from error
    finally:
        for frame in frames:
            frame.close()
        if image is not None:
            image.close()
        if output_path is not None and not completed:
            with suppress(OSError):
                output_path.unlink()


_COLLAGE_ANIMATED_RENDER_LOCK = threading.Lock()


@app.post("/api/inference/collages/animated")
def render_animated_collage(input_data: AnimatedCollageInput):
    if not _COLLAGE_ANIMATED_RENDER_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another animated collage is rendering; retry shortly", headers={"Retry-After": "1"})
    frames = []
    raw_buffer = None
    try:
        frames, durations = animated_collage_frames(input_data)
        raw_buffer, bounded_buffer = bounded_output_buffer()
        frames[0].save(bounded_buffer, format="GIF", save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=2, optimize=False)
        content = raw_buffer.getvalue()
    except (CollageResourceLimitError, Image.DecompressionBombWarning, Image.DecompressionBombError, MemoryError, OverflowError) as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except (ValueError, OSError, SyntaxError, binascii.Error) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        for frame in frames:
            frame.close()
        if raw_buffer is not None:
            raw_buffer.close()
        _COLLAGE_ANIMATED_RENDER_LOCK.release()
    return Response(content=content, media_type="image/gif", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    write_cpu_mode_log()
    uvicorn.run(app, host=INFERENCE_HOST, port=INFERENCE_PORT, log_level="info", access_log=False)
