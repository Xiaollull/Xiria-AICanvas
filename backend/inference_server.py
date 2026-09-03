import asyncio
import base64
import binascii
import copy
import gc
import io
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
from contextlib import asynccontextmanager
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as package_version
from importlib.util import find_spec
from math import ceil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal
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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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
        render_detection_preview,
        resolve_detector_model,
        run_detector,
        select_detections,
    )
    from .anima_pipeline import load_anima_runtime
    from .anima_sampling import ANIMA_SAMPLERS, ANIMA_SCHEDULERS, anima_sampling_diagnostics
    from .flux_pipeline import FLUX_MAX_EDGE, flux_component_bytes, load_flux_runtime
    from .flux_sampling import (
        FLUX_SAMPLERS,
        FLUX_SCHEDULERS,
        flux_resolution_shift,
        flux_sampling_diagnostics,
    )
    from .flux2_pipeline import FLUX2_MAX_EDGE, flux2_component_bytes, load_flux2_runtime
    from .flux2_sampling import (
        FLUX2_SAMPLERS,
        FLUX2_SCHEDULERS,
        flux2_resolution_shift,
        flux2_sampling_diagnostics,
    )
    from .krea2_pipeline import KREA2_MAX_EDGE, krea2_component_bytes, load_krea2_runtime
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
    from .prompt_encoding import parse_prompt_weights, prepare_prompt_conditioning, prompt_diagnostics
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
    from .performance_settings import (
        file_sha256,
        enable_sequential_batch_forward,
        memory_mode_for_family,
        nvfp4_capabilities,
        read_performance_settings,
        vae_decode_tile_count,
        write_performance_settings,
    )
    from .pipeline_configs import PIPELINE_CONFIG_REPOSITORIES, find_pipeline_config, require_pipeline_config
    from .upscaler import resolve_model as resolve_upscaler_model, status as upscaler_status, target_size as hires_target_size, upscale_image
    from .usdu_tiles import TileCompositor, plan_tiles, prepare_tile
except ImportError:
    from adetailer import (
        detection_mask,
        discover_detector_models,
        expand_prompt,
        render_detection_preview,
        resolve_detector_model,
        run_detector,
        select_detections,
    )
    from anima_pipeline import load_anima_runtime
    from anima_sampling import ANIMA_SAMPLERS, ANIMA_SCHEDULERS, anima_sampling_diagnostics
    from flux_pipeline import FLUX_MAX_EDGE, flux_component_bytes, load_flux_runtime
    from flux_sampling import (
        FLUX_SAMPLERS,
        FLUX_SCHEDULERS,
        flux_resolution_shift,
        flux_sampling_diagnostics,
    )
    from flux2_pipeline import FLUX2_MAX_EDGE, flux2_component_bytes, load_flux2_runtime
    from flux2_sampling import (
        FLUX2_SAMPLERS,
        FLUX2_SCHEDULERS,
        flux2_resolution_shift,
        flux2_sampling_diagnostics,
    )
    from krea2_pipeline import KREA2_MAX_EDGE, krea2_component_bytes, load_krea2_runtime
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
    from prompt_encoding import parse_prompt_weights, prepare_prompt_conditioning, prompt_diagnostics
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
    from performance_settings import (
        file_sha256,
        enable_sequential_batch_forward,
        memory_mode_for_family,
        nvfp4_capabilities,
        read_performance_settings,
        vae_decode_tile_count,
        write_performance_settings,
    )
    from pipeline_configs import PIPELINE_CONFIG_REPOSITORIES, find_pipeline_config, require_pipeline_config
    from upscaler import resolve_model as resolve_upscaler_model, status as upscaler_status, target_size as hires_target_size, upscale_image
    from usdu_tiles import TileCompositor, plan_tiles, prepare_tile


OUTPUT_DIRECTORY = configured_path("XIRAI_OUTPUT_DIR", "outputs")
PREVIEW_DIRECTORY = OUTPUT_DIRECTORY / ".previews"
LOG_DIRECTORY = PROJECT_ROOT / "logs"
STATE_DIRECTORY = configured_path("XIRAI_STATE_DIR", "state-cache")
PERFORMANCE_SETTINGS_FILE = STATE_DIRECTORY / "performance.json"
ADETAILER_MODEL_DIRECTORY = resolve_model_directory(PROJECT_ROOT, "yolo")
UPSCALER_MODEL_DIRECTORY = resolve_model_directory(PROJECT_ROOT, "upscalers")
ADETAILER_PYTHON = Path(sys.executable).resolve()
ADETAILER_WORKER = PROJECT_ROOT / "backend" / "adetailer_detector.py"
YOLO_CATALOG_PATH = PROJECT_ROOT / "models" / "yolo-models.json"
PREVIEW_MAX_FRAMES = configured_int("PREVIEW_MAX_FRAMES", 8, 0, 100)
PREVIEW_MIN_INTERVAL = configured_float("PREVIEW_MIN_INTERVAL", 0.7, 0.0)
PREVIEW_MAX_EDGE = configured_int("PREVIEW_MAX_EDGE", 384, 128, 4096)
performance_settings = read_performance_settings(PERFORMANCE_SETTINGS_FILE)
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
PREVIEW_DIRECTORY.mkdir(parents=True, exist_ok=True)
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


executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
preview_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview")
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
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
    preview_executor.shutdown(wait=False, cancel_futures=True)
    with jobs_lock:
        idle = not job_controls
    if idle:
        clear_pipeline()
    clear_background_removal_session()
    for preview_path in PREVIEW_DIRECTORY.glob("*"):
        if preview_path.is_file():
            remove_preview(preview_path)


app = FastAPI(title="XiriaCanvas AI Inference", docs_url=None, redoc_url=None, lifespan=lifespan)
INFERENCE_PROTOCOL = 34
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
    prompt: str = Field(default="", max_length=8000)
    negative_prompt: str = Field(default="", max_length=8000)

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.mask_min_ratio > self.mask_max_ratio:
            raise ValueError("ADetailer minimum mask ratio cannot exceed maximum mask ratio")
        return self


class ADetailerInput(BaseModel):
    """The stage: an ordered list of passes, each seeing what the last produced."""

    model_config = ConfigDict(extra="forbid")

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


class HiresInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: str = Field(default="", max_length=500)
    mode: Literal["inherit", "fixed", "random"] = "inherit"
    seed: str | None = None
    scale: float = Field(default=1.0, ge=1, le=4)
    denoise: float = Field(default=0.35, gt=0, le=1)
    steps: int = Field(default=20, ge=1, le=100)
    cfg: float = Field(default=7, ge=0, le=30)
    tile_size: int = Field(default=192, ge=32, le=2048)
    tile_overlap: int = Field(default=16, ge=0, le=512)
    execution_mode: Literal["full_frame", "usdu_tiled"] = "full_frame"
    sampler: str | None = None
    scheduler: str | None = None
    tile_width: Literal["auto"] = "auto"
    tile_height: Literal["auto"] = "auto"
    padding: int = Field(default=32, ge=0, le=256)
    mask_blur: int = Field(default=8, ge=0, le=64)
    seam_mode: Literal["none"] = "none"
    uniform_tiles: bool = True
    tiled_decode: bool = True

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed(cls, value):
        if value is None:
            return None
        if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
            raise ValueError("Hires.fix seed must be an unsigned 64-bit decimal string")
        parsed = int(value, 10)
        if parsed > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("Hires.fix seed must not exceed 18446744073709551615")
        return str(parsed)

    @field_validator("scale")
    @classmethod
    def normalize_scale(cls, value: float):
        return round(value, 1)

    @model_validator(mode="after")
    def validate_configuration(self):
        if not math.isfinite(self.scale):
            raise ValueError("Hires.fix scale must be finite")
        if self.enabled and not self.model:
            raise ValueError("Hires.fix upscaler model is required when enabled")
        if self.enabled and self.tile_overlap > self.tile_size // 2:
            raise ValueError("Hires.fix tile overlap cannot exceed half the tile size")
        if self.execution_mode == "usdu_tiled" and (not self.uniform_tiles or not self.tiled_decode):
            raise ValueError("USDU tiled Hires requires uniform_tiles and tiled_decode")
        if self.mode == "fixed" and self.seed is None:
            raise ValueError("Hires.fix fixed seed mode requires seed")
        if self.mode != "fixed" and self.seed is not None:
            raise ValueError("Hires.fix seed is only valid in fixed seed mode")
        return self


DIFFUSERS_HIRES_SAMPLERS = frozenset({
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2",
    "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast",
    "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm",
    "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp", "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde",
    "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2",
})
DIFFUSERS_HIRES_SCHEDULERS = frozenset({
    "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal",
    "linear_quadratic", "kl_optimal",
})


def validate_hires_sampling_override(sampler: str | None, scheduler: str | None, family: str):
    """Validate explicit Hires overrides without constructing a derived pipeline."""
    samplers, schedulers = (
        (ANIMA_SAMPLERS, ANIMA_SCHEDULERS) if family == "anima"
        else (FLUX_SAMPLERS, FLUX_SCHEDULERS) if family == "flux"
        else (FLUX2_SAMPLERS, FLUX2_SCHEDULERS) if family == "flux2"
        else (KREA2_SAMPLERS, KREA2_SCHEDULERS) if family == "krea2"
        else (DIFFUSERS_HIRES_SAMPLERS, DIFFUSERS_HIRES_SCHEDULERS)
    )
    if sampler is not None and sampler not in samplers:
        raise ValueError(f"Unsupported Hires sampler: {sampler}")
    if scheduler is not None and scheduler not in schedulers:
        raise ValueError(f"Unsupported Hires scheduler: {scheduler}")


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

    enabled: bool = False
    scale: float = Field(default=2.0, ge=1, le=4)
    quality: Literal["low", "medium", "high", "ultra"] = "ultra"

    @field_validator("scale")
    @classmethod
    def normalize_scale(cls, value: float):
        if not math.isfinite(value):
            raise ValueError("RTX VSR scale must be finite")
        return round(value, 2)


PostprocessStage = Literal["hires", "adetailer", "rtx"]
POSTPROCESS_STAGE_IDS = ("hires", "adetailer", "rtx")
# The sampler's canvas: a latent grid, so 64-aligned, and capped where the memory policy was
# measured. Post-processing has no latent grid of its own — every stage works on pixels it is
# handed — so a post-processing run is bounded by the RTX safety envelope instead and keeps the
# picture at whatever size the user's file actually is.
MAX_SAMPLING_EDGE = 2048
SAMPLING_EDGE_STEP = 64
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
    prompt: str = Field(min_length=1, max_length=8000)
    negative_prompt: str = Field(default="", max_length=8000)
    # The ceiling here is the post-processing one. A sampling run is held to 2048 and to 64-pixel
    # alignment by `validate_dimensions`, which is where the two contracts are told apart.
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
    preview_enabled: bool = True
    background_removal_model: str | None = Field(default=None, max_length=300)
    source_image: SourceImageInput = Field(default_factory=SourceImageInput)
    # Post-processing mode: run the enabled stages on the source picture and nothing else. The base
    # sampling pass is skipped outright rather than run at a denoise low enough to be invisible, so
    # the pixels the user brought are the pixels the first stage receives. Everything the stages
    # read — model, LoRAs, prompt, sampler, seed — is still the mounted configuration, which is why
    # this is a flag on the ordinary request rather than an endpoint of its own.
    postprocess_only: bool = False
    hires: HiresInput = Field(default_factory=HiresInput)
    adetailer: ADetailerInput = Field(default_factory=ADetailerInput)
    rtx: RtxInput = Field(default_factory=RtxInput)
    postprocess_order: list[PostprocessStage] = Field(
        default_factory=lambda: list(POSTPROCESS_STAGE_IDS), min_length=3, max_length=3
    )
    loras: list[LoraInput] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_dimensions(self):
        # A sampling canvas is a latent grid and keeps the alignment and the ceiling the memory
        # policy was measured against. A post-processing run has no latent grid of its own, so the
        # picture keeps its real size and is bounded by the same envelope the stages themselves are.
        if not self.postprocess_only:
            if self.width % SAMPLING_EDGE_STEP != 0 or self.height % SAMPLING_EDGE_STEP != 0:
                raise ValueError("Image dimensions must be divisible by 64")
            if self.width > MAX_SAMPLING_EDGE or self.height > MAX_SAMPLING_EDGE:
                raise ValueError(f"Image dimensions cannot exceed {MAX_SAMPLING_EDGE}")
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
            raise ValueError("Post-processing mode requires at least one of Hires.fix, ADetailer, or RTX VSR")
        return self

    @model_validator(mode="after")
    def validate_prompt_directives(self):
        cleaned_prompt, directives = parse_prompt_directives(self.prompt)
        if directives["transparent_background"] and not cleaned_prompt:
            raise ValueError("透明背景特殊标签之外还需要填写主体提示词")
        if tuple(sorted(self.postprocess_order)) != tuple(sorted(POSTPROCESS_STAGE_IDS)):
            raise ValueError("Post-processing order must contain Hires.fix, ADetailer, and RTX VSR exactly once")
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
            if self.hires.execution_mode == "usdu_tiled":
                raise ValueError("USDU tiled Hires is currently supported only by Anima")
            if self.hires.sampler is not None or self.hires.scheduler is not None:
                validate_hires_sampling_override(self.hires.sampler, self.hires.scheduler, "sd")
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
            if self.hires.sampler is not None and self.hires.sampler not in ANIMA_SAMPLERS:
                raise ValueError(f"Unsupported Anima Hires sampler: {self.hires.sampler}")
            if self.hires.scheduler is not None and self.hires.scheduler not in ANIMA_SCHEDULERS:
                raise ValueError(f"Unsupported Anima Hires scheduler: {self.hires.scheduler}")
            if self.preview_enabled:
                raise ValueError("Anima does not support process previews; set preview_enabled=false")
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
            if self.hires.sampler is not None and self.hires.sampler not in FLUX_SAMPLERS:
                raise ValueError(f"Unsupported Flux Hires sampler: {self.hires.sampler}")
            if self.hires.scheduler is not None and self.hires.scheduler not in FLUX_SCHEDULERS:
                raise ValueError(f"Unsupported Flux Hires scheduler: {self.hires.scheduler}")
            if self.preview_enabled:
                raise ValueError("Flux does not support process previews; set preview_enabled=false")
            if self.hires.execution_mode == "usdu_tiled":
                raise ValueError("USDU tiled Hires is currently supported only by Anima")
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
            if self.hires.sampler is not None and self.hires.sampler not in FLUX2_SAMPLERS:
                raise ValueError(f"Unsupported Flux2 Hires sampler: {self.hires.sampler}")
            if self.hires.scheduler is not None and self.hires.scheduler not in FLUX2_SCHEDULERS:
                raise ValueError(f"Unsupported Flux2 Hires scheduler: {self.hires.scheduler}")
            if self.preview_enabled:
                raise ValueError("Flux2 does not support process previews; set preview_enabled=false")
            if self.hires.execution_mode == "usdu_tiled":
                raise ValueError("USDU tiled Hires is currently supported only by Anima")
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
            if self.hires.sampler is not None and self.hires.sampler not in KREA2_SAMPLERS:
                raise ValueError(f"Unsupported Krea2 Hires sampler: {self.hires.sampler}")
            if self.hires.scheduler is not None and self.hires.scheduler not in KREA2_SCHEDULERS:
                raise ValueError(f"Unsupported Krea2 Hires scheduler: {self.hires.scheduler}")
            if self.preview_enabled:
                raise ValueError("Krea2 does not support process previews; set preview_enabled=false")
            if self.hires.execution_mode == "usdu_tiled":
                raise ValueError("USDU tiled Hires is currently supported only by Anima")
            # Unlike the Flux engines, Krea 2 has a real unconditional branch, so the negative
            # prompt and CFG-Zero* both work. PAG does not: it needs an identity-self-attention
            # override of the transformer's own blocks, which this runtime does not install.
            if self.guidance == "pag":
                raise ValueError("Krea2 does not support PAG; select None or CFG-Zero*")
        current_size = (self.width, self.height)
        for stage in self.postprocess_order:
            enabled = getattr(self, stage).enabled
            if not enabled:
                continue
            if stage == "hires":
                current_size = tuple(max(64, math.ceil(value * self.hires.scale / 64) * 64) for value in current_size)
            elif stage == "rtx":
                current_size = rtx_vsr.target_size(current_size, self.rtx.scale)
            if self.engine == "Anima" and stage == "hires" and max(current_size) > 4096:
                raise ValueError("Anima native refinement dimensions cannot exceed 4096")
            if self.engine == "Flux" and stage == "hires" and max(current_size) > FLUX_MAX_EDGE:
                raise ValueError(f"Flux native refinement dimensions cannot exceed {FLUX_MAX_EDGE}")
            if self.engine == "Flux2" and stage == "hires" and max(current_size) > FLUX2_MAX_EDGE:
                raise ValueError(f"Flux2 native refinement dimensions cannot exceed {FLUX2_MAX_EDGE}")
            if self.engine == "Krea2" and stage == "hires" and max(current_size) > KREA2_MAX_EDGE:
                raise ValueError(f"Krea2 native refinement dimensions cannot exceed {KREA2_MAX_EDGE}")
            if self.rtx.enabled and (max(current_size) > rtx_vsr.MAX_OUTPUT_EDGE or current_size[0] * current_size[1] > rtx_vsr.MAX_OUTPUT_PIXELS):
                raise ValueError("Post-processing output exceeds the safe 8192-edge / 32-megapixel limit")
        return self


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
    vram_limit_gb: float = Field(default=0.0, ge=0.0, le=1024.0)


class CollageInput(BaseModel):
    image_data: str = Field(min_length=32, max_length=200_000_000)
    name: str = Field(default="XirAI-collage.png", max_length=160)
    manual_layout: dict | None = None


class AnimatedCollageInput(BaseModel):
    layers: list[dict] = Field(min_length=1, max_length=100)
    width: int = Field(ge=1, le=4096)
    height: int = Field(ge=1, le=4096)
    edge_line: dict | None = None


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
    positive_prompt: str = Field(default="", max_length=8000)
    negative_prompt: str = Field(default="", max_length=8000)
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
    positive_prompt: str | None = Field(default=None, max_length=8000)
    negative_prompt: str | None = Field(default=None, max_length=8000)
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
    if (path != output_root and output_root not in path.parents) or not path.is_dir() or path == PREVIEW_DIRECTORY.resolve():
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
        "postprocess_order": parameters.get("postprocess_order") if isinstance(parameters.get("postprocess_order"), list) else None,
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
    if current == PREVIEW_DIRECTORY.resolve() or PREVIEW_DIRECTORY.resolve() in current.parents:
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
            if not entry.is_dir() or resolved == PREVIEW_DIRECTORY.resolve() or output_root not in resolved.parents:
                continue
            child_images = sum(1 for path in entry.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".gif"})
            child_folders = sum(1 for path in entry.iterdir() if path.is_dir() and path.resolve() != PREVIEW_DIRECTORY.resolve())
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
    preview_root = PREVIEW_DIRECTORY.resolve()
    candidates = [path for path in (folder.iterdir() if folder else OUTPUT_DIRECTORY.rglob("*")) if path.is_file() and path.suffix.lower() in {".png", ".gif"}]
    for path in candidates:
        try:
            resolved = path.resolve()
            if output_root not in resolved.parents or resolved == preview_root or preview_root in resolved.parents:
                continue
            if session_only and resolved.stat().st_mtime < HISTORY_STARTED_AT:
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
    color = options.get("color", "#D6FF3F")
    width = max(1, min(12, int(options.get("width", 2))))
    style = options.get("style", "solid")
    left, top = int(layer["x"]), int(layer["y"])
    right = left + max(1, int(layer["width"])) - 1
    bottom = top + max(1, int(layer["height"])) - 1
    draw = ImageDraw.Draw(image)
    if style == "dashed":
        for x in range(left, right + 1, width * 6):
            draw.line((x, top, min(right, x + width * 3), top), fill=color, width=width)
            draw.line((x, bottom, min(right, x + width * 3), bottom), fill=color, width=width)
        for y in range(top, bottom + 1, width * 6):
            draw.line((left, y, left, min(bottom, y + width * 3)), fill=color, width=width)
            draw.line((right, y, right, min(bottom, y + width * 3)), fill=color, width=width)
    elif style == "dotted":
        for x in range(left, right + 1, width * 3):
            draw.ellipse((x, top, x + width, top + width), fill=color)
            draw.ellipse((x, bottom - width, x + width, bottom), fill=color)
        for y in range(top, bottom + 1, width * 3):
            draw.ellipse((left, y, left + width, y + width), fill=color)
            draw.ellipse((right - width, y, right, y + width), fill=color)
    else:
        draw.rectangle((left, top, right, bottom), outline=color, width=width * (2 if style == "double" else 1))
        if style == "double":
            draw.rectangle((left + width * 3, top + width * 3, right - width * 3, bottom - width * 3), outline=color, width=width)


def animated_collage_frames(input_data: AnimatedCollageInput):
    sources = []
    output_frame_count = 1
    for layer in input_data.layers:
        source = layer.get("url") if isinstance(layer, dict) else None
        if not isinstance(source, str) or not source.startswith("data:image/"):
            raise ValueError("Animated collage source is invalid")
        try:
            raw = base64.b64decode(source.split(",", 1)[-1], validate=True)
            image = Image.open(io.BytesIO(raw))
            source_frame_count = max(1, int(getattr(image, "n_frames", 1)))
        except (ValueError, binascii.Error, OSError) as error:
            raise ValueError("Unable to read GIF collage source") from error
        sources.append((image, layer, source_frame_count))
        output_frame_count = max(output_frame_count, source_frame_count)
    if output_frame_count > 240 or input_data.width * input_data.height * output_frame_count > 120_000_000:
        raise ValueError("GIF collage is too large")
    frames, durations = [], []
    for frame_index in range(output_frame_count):
        output = Image.new("RGBA", (input_data.width, input_data.height), (0, 0, 0, 0))
        duration = 100
        for image, layer, source_frame_count in sources:
            image.seek(frame_index % source_frame_count)
            current = image.convert("RGBA")
            width = max(1, min(input_data.width, int(layer.get("width", current.width))))
            height = max(1, min(input_data.height, int(layer.get("height", current.height))))
            x, y = int(layer.get("x", 0)), int(layer.get("y", 0))
            placed = Image.new("RGBA", (input_data.width, input_data.height), (0, 0, 0, 0))
            placed.paste(current.resize((width, height), Image.Resampling.LANCZOS), (x, y), current.resize((width, height), Image.Resampling.LANCZOS))
            clip = layer.get("clip") if isinstance(layer.get("clip"), dict) else None
            if clip:
                clip_left = max(0, int(clip.get("x", 0)))
                clip_top = max(0, int(clip.get("y", 0)))
                clip_right = min(input_data.width, clip_left + max(1, int(clip.get("width", input_data.width))))
                clip_bottom = min(input_data.height, clip_top + max(1, int(clip.get("height", input_data.height))))
                clipped = placed.crop((clip_left, clip_top, clip_right, clip_bottom))
                output.alpha_composite(clipped, (clip_left, clip_top))
                draw_collage_edge(output, {"x": clip_left, "y": clip_top, "width": clip_right - clip_left, "height": clip_bottom - clip_top}, input_data.edge_line)
            else:
                output.alpha_composite(placed)
                draw_collage_edge(output, {"x": x, "y": y, "width": width, "height": height}, input_data.edge_line)
            duration = max(duration, max(20, int(image.info.get("duration", 100))))
        frames.append(output)
        durations.append(duration)
    return frames, durations


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
# quantisations are published in. The other three components stay safetensors: see
# `anima_pipeline._require_diffusion_weights` for why a GGUF text encoder is not the same problem.
NATIVE_DIFFUSION_EXTENSIONS = frozenset({".safetensors", ".gguf"})


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
            "process_preview": False,
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
            "process_preview": False,
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
            "process_preview": False,
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
            "process_preview": False,
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
            loaded_pipeline.remove_all_hooks()
        except Exception:
            pass
        del loaded_pipeline
    loaded_pipeline = None
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
    physical_forward_copies = max(1, int(guidance_copies if guidance_copies is not None else 2 if guidance_scale > 1 else 1))
    inference_bytes = estimate_inference_bytes(
        family, width, height, guidance_scale, images_per_batch, guidance_copies
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
    )
    mode = policy["mode"]
    if policy["requested_mode"] != "auto" and mode != policy["requested_mode"]:
        reason = (
            f"配置请求 {policy['requested_mode'].upper()}，但当前模型与画布需要约 "
            f"{policy['high_required_bytes'] / GIB:.1f} GB，已安全降级"
        )
    elif mode == "high_vram":
        reason = f"自动预算通过：{free_bytes / GIB:.1f} GB 可用，底模完整常驻 GPU"
    elif mode == "normal_vram":
        reason = f"自动预算：{free_bytes / GIB:.1f} GB 可用，按组件动态调度并缓存底模"
    elif mode == "ultra_low_vram":
        reason = "极限省存：逐层卸载、串行 CFG、VAE 分块，并在任务后释放全部模型"
    else:
        reason = f"自动预算：推理至少需要约 {policy['normal_required_bytes'] / GIB:.1f} GB，启用逐层低显存保护"
    return {
        **policy,
        "label": {
            "high_vram": "HIGH_VRAM 高速常驻",
            "normal_vram": "NORMAL_VRAM 智能调度",
            "low_vram": "LOW_VRAM 低显存",
            "ultra_low_vram": "ULTRA_LOW 极限省存",
        }[mode],
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
        "physical_forward_copies": physical_forward_copies,
        "reserved_bytes": reserved_bytes,
        "physical_total_bytes": physical_total_bytes,
        "vram_limit_bytes": limit_bytes,
        "offload_mode": {
            "high_vram": "none",
            "normal_vram": "model",
            "low_vram": "sequential",
            "ultra_low_vram": "sequential_cfg_split",
        }[mode],
        "model_resident": mode == "high_vram",
    }


def memory_job_fields(strategy, model_cached=True):
    parked = bool(model_cached and pipeline_cpu_parked)
    return {
        "memory_mode": strategy["mode"],
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
        "memory_admission": strategy.get("admission"),
        "acceleration": strategy.get("acceleration"),
    }


def configure_cuda_math():
    balanced = performance_settings["cuda_math"] == "balanced"
    torch.set_float32_matmul_precision("high" if balanced else "highest")
    torch.backends.cuda.matmul.allow_tf32 = balanced
    torch.backends.cudnn.allow_tf32 = balanced
    torch.backends.cudnn.benchmark = balanced


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
        mode = {
            "high_vram": "full",
            "normal_vram": "sliced",
            "low_vram": "tiled",
            "ultra_low_vram": "tiled",
        }[strategy["mode"]]
    pipeline.disable_vae_slicing()
    pipeline.disable_vae_tiling()
    if mode in {"sliced", "tiled"}:
        pipeline.enable_vae_slicing()
    if mode == "tiled":
        pipeline.enable_vae_tiling()
    active_vae_mode = mode


def resolve_compute_dtype():
    if performance_settings["compute_dtype"] == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def apply_memory_strategy(pipeline, strategy):
    mode = strategy["mode"]
    if mode == "high_vram":
        pipeline.to("cuda")
    elif mode == "normal_vram":
        pipeline.enable_model_cpu_offload(device="cuda")
    else:
        if active_attention_backend == "sliced attention":
            pipeline.enable_attention_slicing("max")
        pipeline.enable_sequential_cpu_offload(device="cuda")
        if mode == "ultra_low_vram":
            enable_sequential_batch_forward(pipeline.unet)
    configure_vae_mode(pipeline, strategy)


def reconfigure_memory_strategy(pipeline, strategy):
    pipeline.remove_all_hooks()
    pipeline.to("cpu", silence_dtype_warnings=True)
    pipeline.disable_attention_slicing()
    configure_cuda_math()
    configure_attention_backend(pipeline)
    apply_memory_strategy(pipeline, strategy)


def park_pipeline_for_vae(pipeline):
    global pipeline_cpu_parked
    try:
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


def decode_staged_latents(pipeline, latents, family, job_id, control, use_tiling, start_progress=94, final_progress=98):
    latents = latents.to("cpu")
    control.checkpoint(job_id, "Unloading sampler")
    update_job(job_id, phase="Moving sampler to system memory", stage="sampler_offload", stage_step=0, stage_total=0, progress=start_progress)
    park_pipeline_for_vae(pipeline)
    control.checkpoint(job_id, "Preparing VAE")
    vae = pipeline.vae
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
            phase="VAE tiled decode" if tiling_active else "VAE decode",
            stage="vae_decode",
            stage_step=0,
            stage_total=tile_total,
            progress=start_progress,
            **pipeline_status_fields(),
        )
        with torch.inference_mode():
            decoded = vae.decode(latents, return_dict=False)[0]
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
    try:
        if is_native_family(loaded_family):
            loaded_pipeline.to_cpu()
            return True
        if pipeline_cpu_parked:
            return True
        elif performance_settings["staged_vae_decode"]:
            park_pipeline_for_vae(loaded_pipeline)
        elif active_memory_strategy["mode"] == "normal_vram":
            loaded_pipeline.maybe_free_model_hooks()
        elif active_memory_strategy["mode"] in {"low_vram", "ultra_low_vram"}:
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
    if isinstance(error, torch.OutOfMemoryError):
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


def load_pipeline(checkpoint: Path, family: str, engine: str, width: int, height: int, guidance_scale: float, lora_bytes: int, job_id: str, images_per_batch=1, guidance="none"):
    global loaded_pipeline, loaded_checkpoint, loaded_checkpoint_hash, loaded_family, loaded_engine
    global pipeline_cpu_parked, active_memory_strategy, active_attention_backend, active_compute_dtype
    with pipeline_lock:
        guidance_copies = guidance_prediction_copies(guidance, guidance_scale)
        if loaded_pipeline is not None and loaded_checkpoint == str(checkpoint) and loaded_family == family and loaded_engine == engine:
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
            )
            if pipeline_cpu_parked or strategy["mode"] != active_memory_strategy["mode"]:
                update_job(job_id, phase=f"Adjusting VRAM policy · {strategy['label']}", progress=8)
                previous_strategy = active_memory_strategy
                try:
                    loaded_pipeline.unload_lora_weights()
                    reconfigure_memory_strategy(loaded_pipeline, strategy)
                    pipeline_cpu_parked = False
                except Exception:
                    try:
                        reconfigure_memory_strategy(loaded_pipeline, previous_strategy)
                    except Exception:
                        clear_pipeline()
                    raise
            active_memory_strategy = strategy
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
            checkpoint, family, width, height, guidance_scale, lora_bytes, images_per_batch, guidance_copies
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

        pipeline = pipeline_class.from_single_file(str(checkpoint), **kwargs)
        pipeline.set_progress_bar_config(disable=True)
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
        )
        configure_cuda_math()
        configure_attention_backend(pipeline)
        apply_memory_strategy(pipeline, strategy)
        loaded_pipeline = pipeline
        loaded_checkpoint = str(checkpoint)
        loaded_family = family
        loaded_engine = engine
        active_memory_strategy = strategy
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
        strategy.update(mode="low_vram", label="LOW_VRAM 低显存")
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
    # ComfyUI default auto behavior keeps models resident only for the current
    # job and returns them to RAM before the next stage; cross-job GPU residency
    # is a high-VRAM choice. Normal mode therefore stages per job.
    keep_transformer_resident = bool(
        performance_settings["keep_model_cached"]
        and not group_offload
        and strategy.get("mode") == "high_vram"
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
def flux_component_weight_bytes(component_path: str, file_size: int, modified_ns: int):
    """Loaded size of one Flux or Flux2 component, which is not its size on disk.

    Everything is expanded to BF16, so an fp8 file costs twice what it occupies. Budgeting from
    the file size would under-admit an fp8 checkpoint by half and let it OOM mid-load.
    """
    del modified_ns
    try:
        return flux_component_bytes([component_path])
    except Exception:
        return file_size


def flux_weight_estimates(assets):
    estimates = []
    for path in assets.values():
        stat = path.stat()
        estimates.append(flux_component_weight_bytes(str(path), stat.st_size, stat.st_mtime_ns))
    return sum(estimates), max(estimates)


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
        strategy.update(mode="low_vram", label="LOW_VRAM 低显存")
    keep_transformer_resident = bool(
        not group_offload
        and performance_settings["keep_model_cached"]
        and strategy["mode"] == "high_vram"
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
    total_weight_bytes, largest_component_bytes = flux_weight_estimates(assets)
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
            active_compute_dtype = "bf16"
            active_vae_mode = configure_flux_vae(loaded_pipeline, strategy)
            pipeline_cpu_parked = False
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


def configure_loras(pipeline, loras: list[LoraInput], lora_root: Path, job_id: str):
    pipeline.unload_lora_weights()
    if not loras:
        return

    update_job(job_id, phase="Loading LoRA adapters", progress=21)
    adapter_names = []
    adapter_weights = []
    mounted = []
    for index, lora in enumerate(loras):
        path = resolve_model_path(lora.path, lora_root, LORA_EXTENSIONS, "LoRA")
        adapter_name = f"lora_{index}"
        pipeline.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
        adapter_weights.append(lora.weight)
        mounted.append((path, lora.weight))
    pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
    report_mounted_loras(job_id, mounted)


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


def run_guided_pipeline(pipeline, active_pipeline, pipeline_kwargs, guidance_kwargs=None):
    original_attn_processors = (
        dict(pipeline.unet.attn_processors)
        if guidance_kwargs and hasattr(pipeline, "unet") else None
    )
    try:
        return active_pipeline(**{**pipeline_kwargs, **(guidance_kwargs or {})})
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
        "num_inference_steps": request.steps,
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
    if not request.source_image.enabled or is_native_family(family):
        return request.steps
    return max(1, int(request.steps * request.denoise))


def run_guided_pipeline_batch(pipeline, guidance: str, pipeline_kwargs, pag: PagInput | None = None, factory=None):
    active_pipeline, guidance_kwargs = (
        derived_sampling_pipeline(pipeline, factory, guidance, pag)
        if factory is not None
        else sampling_pipeline(pipeline, guidance, pag)
    )
    try:
        outputs = run_guided_pipeline(pipeline, active_pipeline, pipeline_kwargs, guidance_kwargs).images
        expected = pipeline_kwargs["num_images_per_prompt"]
        if len(outputs) != expected:
            raise RuntimeError(f"Pipeline returned {len(outputs)} images for a requested batch of {expected}")
        return outputs
    finally:
        if active_pipeline is not pipeline:
            del active_pipeline


def postprocessing_stages(request: GenerateInput):
    return [stage for stage in request.postprocess_order if getattr(request, stage).enabled]


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
    guidance_copies = (
        1 if is_native_family(family)
        else guidance_prediction_copies(request.guidance, request.cfg)
    )
    base = (request.width, request.height)
    diagnostics = {
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
    current_size = (request.width, request.height)
    for stage in postprocessing_stages(request):
        if stage == "rtx":
            current_size = rtx_vsr.target_size(current_size, request.rtx.scale)
        elif stage == "hires":
            hires_source_size = current_size
            current_size = tuple(math.ceil(value * request.hires.scale / 64) * 64 for value in current_size)
            hires_copies = (
                1 if is_native_family(family)
                else guidance_prediction_copies(request.guidance, request.hires.cfg)
            )
            admission_size = current_size
            reason = "full_frame_target"
            if family == "anima" and request.hires.execution_mode == "usdu_tiled":
                # Match apply_hires_fix: Auto core is the canvas entering Hires,
                # which may already have been changed by an earlier RTX stage.
                core = hires_source_size
                plan = plan_tiles(current_size, core, padding=request.hires.padding, seam_mode=request.hires.seam_mode)
                admission_size = tuple(max(base, processed) for base, processed in zip(hires_source_size, plan.processing_size))
                reason = "usdu_tile_processing_target_excluded"
                diagnostics.update({
                    "target_dimensions": current_size,
                    "core_dimensions": core,
                    "processing_dimensions": plan.processing_size,
                    "admission_dimensions": admission_size,
                    "stage": "hires",
                    "reason": reason,
                    "target_excluded": True,
                })
            elif stage == "hires":
                diagnostics.update({
                    "target_dimensions": current_size,
                    "admission_dimensions": admission_size,
                    "stage": "hires",
                    "reason": reason,
                })
            workloads.append((*admission_size, request.hires.cfg, 1, hires_copies, reason))
        elif stage == "adetailer" and is_native_family(family):
            # Units run one after another, never together, so admission is the
            # most demanding single unit rather than their sum.
            for unit in request.adetailer.active_units:
                detail_cfg = unit.cfg if unit.use_cfg else request.cfg
                detail_copies = (
                    1 if is_native_family(family)
                    else guidance_prediction_copies(request.guidance, detail_cfg)
                )
                workloads.append((1024, 1024, detail_cfg, 1, detail_copies, "adetailer"))
    selected = max(workloads or [base_workload], key=lambda item: estimate_inference_bytes(family, *item[:5]))
    diagnostics["admission"] = selected[:4]
    if selected[5] != "usdu_tile_processing_target_excluded":
        diagnostics["stage"] = selected[5]
    return diagnostics


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


def apply_hires_fix(
    image: Image.Image,
    pipeline,
    family: str,
    request: GenerateInput,
    job_id: str,
    control: JobControl,
    started_at: float,
    schedule_latent_preview=None,
    invalidate_preview=None,
    image_seed=None,
    effective_hires_seed=None,
    progress_start=80,
    progress_end=96,
):
    """Public Hires entry point; benchmark stage capture stays a no-op unless its isolated gate passes."""
    settings = request.hires
    if not settings.enabled:
        return image, None
    if effective_hires_seed is None:
        raise ValueError("Effective Hires.fix seed must be resolved before the Hires stage")
    base_seed = request.seed if image_seed is None else image_seed
    capture, capture_gate_facts = open_hires_artifact_capture(job_id, base_seed)
    stages = lambda: _apply_hires_fix_stages(
        image, pipeline, family, request, job_id, control, started_at,
        schedule_latent_preview=schedule_latent_preview, invalidate_preview=invalidate_preview,
        image_seed=image_seed, effective_hires_seed=effective_hires_seed,
        progress_start=progress_start, progress_end=progress_end, capture=capture,
    )
    if not capture.enabled:
        return stages()
    try:
        result, diagnostics = stages()
        capture.stage("final", result)
        capture.finish(
            status="complete",
            protocol=INFERENCE_PROTOCOL,
            engine=request.engine,
            execution_mode=diagnostics.get("execution_mode"),
            base_seed=str(base_seed),
            hires_seed_mode=settings.mode,
            effective_hires_seed=str(effective_hires_seed),
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
                "hires": {
                    "scale": settings.scale, "steps": settings.steps, "cfg": settings.cfg,
                    "denoise": settings.denoise, "execution_mode": settings.execution_mode,
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


def _apply_hires_fix_stages(
    image: Image.Image,
    pipeline,
    family: str,
    request: GenerateInput,
    job_id: str,
    control: JobControl,
    started_at: float,
    schedule_latent_preview=None,
    invalidate_preview=None,
    image_seed=None,
    effective_hires_seed=None,
    progress_start=80,
    progress_end=96,
    capture=NULL_CAPTURE,
):
    settings = request.hires
    if not settings.enabled:
        return image, None
    if effective_hires_seed is None:
        raise ValueError("Effective Hires.fix seed must be resolved before the Hires stage")
    base_seed = request.seed if image_seed is None else image_seed
    # USDU Auto follows the pre-SR image, even though planning happens after
    # pixel SR has produced the diffusion canvas.
    hires_source_size = image.size
    # SD/iL retain Diffusers' effective callback count. The native engines follow Comfy's
    # expanded-schedule suffix and always execute every requested step.
    effective_steps = settings.steps if is_native_family(family) else max(1, int(settings.steps * settings.denoise))
    destination = hires_target_size(image, settings.scale)
    if is_native_family(family) and max(destination) > NATIVE_MAX_REFINEMENT_EDGE:
        raise ValueError(f"Native refinement dimensions cannot exceed {NATIVE_MAX_REFINEMENT_EDGE}")
    staged_decode = performance_settings["staged_vae_decode"] and not is_native_family(family)
    upscale_end = progress_start + round((progress_end - progress_start) * 0.2) if settings.scale > 1 else progress_start
    sampling_end = progress_end - round((progress_end - progress_start) * 0.15) if staged_decode else progress_end
    control.checkpoint(job_id, "Hires.fix upscaling")
    reported_upscale_tiles = [0]

    def on_upscale_tile(completed, total):
        reported_upscale_tiles[0] = max(reported_upscale_tiles[0], completed)
        update_job(
            job_id,
            phase=f"Hires.fix · Upscaling {destination[0]} x {destination[1]}",
            stage="hires_upscale",
            stage_step=reported_upscale_tiles[0],
            stage_total=total,
            progress=progress_start + round((upscale_end - progress_start) * reported_upscale_tiles[0] / max(1, total)),
            elapsed_seconds=round(control.active_elapsed(started_at), 1),
        )
        control.checkpoint(job_id, "Hires.fix upscaling")

    update_job(job_id, phase="Hires.fix · Loading upscaler", stage="hires_upscale", stage_step=0, stage_total=0, progress=progress_start)
    use_cuda_upscaler = settings.scale > 1 and torch.cuda.is_available()
    upscale_warning = None
    try:
        if use_cuda_upscaler and not pipeline_cpu_parked:
            update_job(job_id, phase="Hires.fix · Parking diffusion model", stage="sampler_offload", progress=progress_start)
            park_pipeline_for_external_stage(pipeline, family)
        try:
            upscaled, diagnostics = upscale_image(
                image,
                UPSCALER_MODEL_DIRECTORY,
                settings.model,
                settings.scale,
                settings.tile_size,
                settings.tile_overlap,
                on_upscale_tile,
                device="cuda" if use_cuda_upscaler else "cpu",
                checkpoint=lambda: control.checkpoint(job_id, "Hires.fix upscaling"),
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if not use_cuda_upscaler or not is_oom_error(error):
                raise
            torch.cuda.empty_cache()
            upscale_warning = "Hires.fix upscaler exceeded GPU memory and used CPU fallback"
            update_job(job_id, phase="Hires.fix · CPU fallback", stage="hires_upscale", warning=append_warning(None, upscale_warning))
            upscaled, diagnostics = upscale_image(
                image,
                UPSCALER_MODEL_DIRECTORY,
                settings.model,
                settings.scale,
                settings.tile_size,
                settings.tile_overlap,
                on_upscale_tile,
                device="cpu",
                checkpoint=lambda: control.checkpoint(job_id, "Hires.fix CPU upscaling"),
            )
    finally:
        if not is_native_family(family) and pipeline_cpu_parked and not control.cancelled:
            update_job(job_id, phase="Hires.fix · Restoring diffusion model", stage="model_restore")
            restore_parked_pipeline()
    if is_native_family(family) and max(upscaled.size) > NATIVE_MAX_REFINEMENT_EDGE:
        raise ValueError(f"Native refinement dimensions cannot exceed {NATIVE_MAX_REFINEMENT_EDGE}")
    capture.stage("base", image)
    capture.stage("post_sr", upscaled)
    control.checkpoint(job_id, "Hires.fix diffusion refinement")
    if invalidate_preview:
        invalidate_preview()
    phase = f"Hires.fix · Diffusion {destination[0]} x {destination[1]}"
    resolved_sampler = settings.sampler or request.sampler
    resolved_scheduler = settings.scheduler or request.scheduler
    if not is_native_family(family):
        validate_hires_sampling_override(resolved_sampler, resolved_scheduler, family)
    hires_seed = effective_hires_seed
    if is_native_family(family):
        if family == "anima" and settings.execution_mode == "usdu_tiled":
            core_tile = (
                hires_source_size[0] if settings.tile_width == "auto" else settings.tile_width,
                hires_source_size[1] if settings.tile_height == "auto" else settings.tile_height,
            )
            plan = plan_tiles(upscaled.size, core_tile, padding=settings.padding, seam_mode=settings.seam_mode)
            tile_count = len(plan.regions)
            stage_total = tile_count * settings.steps
            update_job(
                job_id,
                phase=f"Hires.fix · Tile 1/{tile_count}",
                stage="hires_sampling", stage_step=0, stage_total=stage_total,
                stage_unit_index=1, stage_unit_total=tile_count,
                stage_unit_step=0, stage_unit_steps=settings.steps,
                progress=upscale_end, step=0, total_steps=stage_total,
            )
            previous_metrics = dict(getattr(pipeline, "last_generation_metrics", {}) or {})
            preparation_started = time.monotonic()
            prepared_conditioning = pipeline.prepare_refinement_conditioning(
                request.prompt, request.negative_prompt, settings.cfg, request.guidance
            )
            prepared_sigmas = pipeline.prepare_refinement_sigmas(settings.steps, settings.denoise, resolved_scheduler)
            preparation_metrics = serializable_runtime_metrics(
                (getattr(pipeline, "last_generation_metrics", {}) or {}).get("refinement.prompt_encode")
            )
            preparation_elapsed = round(time.monotonic() - preparation_started, 4)
            compositor = TileCompositor(upscaled.convert("RGB"), plan, mask_blur=settings.mask_blur)
            tile_metrics = []
            for tile_index, region in enumerate(plan.regions):
                control.checkpoint(job_id, f"Hires.fix tile {tile_index + 1}/{tile_count}")
                tile = prepare_tile(upscaled, region)
                generator = torch.Generator(device="cpu").manual_seed(hires_seed)
                tile_started = time.monotonic()

                def on_anima_tile_step(step, total, _latents, index=tile_index):
                    stage_step = index * settings.steps + step
                    progress = upscale_end + round((progress_end - upscale_end) * stage_step / max(1, stage_total))
                    update_job(
                        job_id, phase=f"Hires.fix · Tile {index + 1}/{tile_count}", stage="hires_sampling",
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
                    control.checkpoint(job_id, f"Hires.fix tile {index + 1}/{tile_count}")

                refined = pipeline.refine_batch(
                    images=[tile], prompt=request.prompt, negative_prompt=request.negative_prompt,
                    steps=settings.steps, denoise=settings.denoise, cfg=settings.cfg,
                    sampler=resolved_sampler, scheduler=resolved_scheduler, generators=[generator],
                    guidance=request.guidance, pag_scale=request.pag.scale, pag_applied_layers=request.pag.applied_layers,
                    masks=None, on_step=on_anima_tile_step, prepared_conditioning=prepared_conditioning,
                    prepared_sigmas=prepared_sigmas, force_tiled_decode=True,
                    on_step_checkpoint=on_anima_tile_checkpoint,
                )
                if len(refined) != 1:
                    raise RuntimeError(f"Anima USDU Hires tile returned {len(refined)} images instead of 1")
                metrics = serializable_runtime_metrics(getattr(pipeline, "last_generation_metrics", {}) or {})
                sampling_metrics = metrics.get("refinement.sampling")
                decode_metrics = metrics.get("refinement.vae_decode")
                capture.tile(tile_index, refined[0])
                compositor.composite(refined[0], region)
                # Snapshotting the canvas costs a full RGBA->RGB copy, so never build it when capture is off.
                if capture.enabled:
                    capture.composite(tile_index, compositor.finish())
                control.checkpoint(job_id, f"Hires.fix tile {tile_index + 1}/{tile_count} composite")
                row, col = divmod(tile_index, plan.cols)
                tile_metrics.append({
                    "index": tile_index, "row": row, "col": col, "hires_seed": str(hires_seed),
                    "elapsed_seconds": round(time.monotonic() - tile_started, 4),
                    "schedule_requested": {"sampler": resolved_sampler, "scheduler": resolved_scheduler, "steps": settings.steps},
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
            diagnostics.update({
                "execution_mode": "usdu_tiled", "sampler_requested": settings.sampler,
                "scheduler_requested": settings.scheduler, "sampler_resolved": resolved_sampler,
                "scheduler_resolved": resolved_scheduler, "source_size": hires_source_size, "target_size": destination,
                "base_seed": str(base_seed), "hires_seed_mode": settings.mode, "hires_seed": str(hires_seed),
                "core_tile": core_tile, "model_size": plan.processing_size, "rows": plan.rows,
                "cols": plan.cols, "tile_count": tile_count, "padding": settings.padding,
                "mask_blur": settings.mask_blur, "seam_mode": settings.seam_mode,
                "uniform_tiles": settings.uniform_tiles, "tiled_decode_requested": {"tile": 512, "overlap": 64},
                "tiled_decode_resolved": refinement_decode.get("resolved_tiled_decode", {"mode": "unknown"}),
                "steps_per_tile": settings.steps, "total_steps": stage_total,
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
            return compositor.finish(), diagnostics

        generator = torch.Generator(device="cpu").manual_seed(hires_seed)
        update_job(
            job_id,
            phase=phase,
            stage="hires_sampling",
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
                stage="hires_sampling",
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

        previous_metrics = dict(getattr(pipeline, "last_generation_metrics", {}) or {})
        refined = pipeline.refine_batch(
            images=[upscaled.convert("RGB")],
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=settings.steps,
            denoise=settings.denoise,
            cfg=settings.cfg,
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
            raise RuntimeError(f"Anima Hires.fix returned {len(refined)} images instead of 1")
        merge_runtime_metrics(
            job_id, getattr(pipeline, "last_generation_metrics", None), "hires", previous_metrics
        )
        diagnostics.update({
            "execution_mode": "full_frame",
            "sampler_requested": settings.sampler,
            "scheduler_requested": settings.scheduler,
            "sampler_resolved": resolved_sampler,
            "scheduler_resolved": resolved_scheduler,
            "base_seed": str(base_seed),
            "hires_seed_mode": settings.mode,
            "hires_seed": str(hires_seed),
            "denoise": settings.denoise,
            "steps": settings.steps,
            "effective_steps": effective_steps,
            "cfg": settings.cfg,
            "postprocess_order": list(request.postprocess_order),
        })
        refinement_sampling = (getattr(pipeline, "last_generation_metrics", {}) or {}).get("refinement.sampling", {})
        for key in ("schedule_steps", "schedule_mode", "start_sigma"):
            if key in refinement_sampling:
                diagnostics[key] = refinement_sampling[key]
        if upscale_warning:
            diagnostics["warning"] = upscale_warning
        return refined[0].convert("RGB"), diagnostics

    image_pipeline, guidance_kwargs = derived_sampling_pipeline(
        pipeline, AutoPipelineForImage2Image, request.guidance, request.pag
    )
    if settings.sampler is not None or settings.scheduler is not None:
        configure_scheduler(image_pipeline, resolved_sampler, resolved_scheduler)
    conditioning = prepare_prompt_conditioning(image_pipeline, family, request.prompt, request.negative_prompt)
    preview_interval = max(1, ceil(effective_steps / PREVIEW_MAX_FRAMES)) if PREVIEW_MAX_FRAMES > 0 else effective_steps + 1
    last_hires_step = [0]

    def on_hires_step(_pipeline, step_index, _timestep, callback_kwargs):
        step = logical_diffusion_step(_pipeline, step_index, effective_steps)
        step_advanced = step > last_hires_step[0]
        last_hires_step[0] = max(last_hires_step[0], step)
        progress = upscale_end + round((sampling_end - upscale_end) * step / effective_steps)
        update_job(
            job_id,
            phase=phase,
            stage="hires_sampling",
            stage_step=step,
            stage_total=effective_steps,
            progress=min(progress, progress_end),
            step=step,
            total_steps=effective_steps,
            elapsed_seconds=round(control.active_elapsed(started_at), 1),
            paused_seconds=round(control.total_paused(), 1),
        )
        control.checkpoint(job_id, phase)
        if schedule_latent_preview and step_advanced and (step == 1 or step == effective_steps or step % preview_interval == 0):
            schedule_latent_preview(
                callback_kwargs["latents"],
                family=family,
                step=step,
                width=destination[0],
                height=destination[1],
                kind="hires_sampling",
            )
        return callback_kwargs

    update_job(job_id, phase=phase, stage="hires_sampling", stage_step=0, stage_total=effective_steps, progress=upscale_end, step=0, total_steps=effective_steps)
    try:
        pipeline_output = run_guided_pipeline(
            pipeline,
            image_pipeline,
            {
                "image": upscaled,
                "strength": settings.denoise,
                "num_inference_steps": settings.steps,
                "guidance_scale": settings.cfg,
                "generator": torch.Generator(device="cpu").manual_seed(hires_seed),
                "callback_on_step_end": on_hires_step,
                "callback_on_step_end_tensor_inputs": ["latents"],
                "output_type": "latent" if staged_decode else "pil",
                **conditioning,
            },
            guidance_kwargs,
        ).images
        if staged_decode:
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
            )
            result = decoded[0].convert("RGB")
            del decoded, pipeline_output
        else:
            result = pipeline_output[0].convert("RGB")
        diagnostics.update({
            "execution_mode": "full_frame",
            "sampler_requested": settings.sampler,
            "scheduler_requested": settings.scheduler,
            "sampler_resolved": resolved_sampler,
            "scheduler_resolved": resolved_scheduler,
            "base_seed": str(base_seed),
            "hires_seed_mode": settings.mode,
            "hires_seed": str(hires_seed),
            "denoise": settings.denoise,
            "steps": settings.steps,
            "effective_steps": effective_steps,
            "cfg": settings.cfg,
            "postprocess_order": list(request.postprocess_order),
        })
        if upscale_warning:
            diagnostics["warning"] = upscale_warning
        return result, diagnostics
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

    The same family split `base_sampling_steps` and `_apply_hires_fix_stages` already make. Diffusers
    inpainting runs `int(steps * strength)` of them, so `steps` alone would overstate the pass. Native
    refinement follows Comfy: it builds a longer schedule and keeps the last `steps + 1` sigmas,
    so it executes **every** requested step whatever the denoise, and `int(steps * denoise)` would
    understate it — which is what drove the job's step counter past its own total.
    """
    detail_steps = settings.steps if settings.use_steps else request.steps
    if is_native_family(family):
        return detail_steps
    return int(detail_steps * settings.denoise)


def apply_adetailer_unit(
    image: Image.Image,
    pipeline,
    family: str,
    request: GenerateInput,
    settings: ADetailerUnitInput,
    job_id: str,
    control: JobControl,
    started_at: float,
    publish_image_preview=None,
    schedule_latent_preview=None,
    invalidate_preview=None,
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
    if effective_steps < 1:
        raise ValueError("ADetailer steps multiplied by denoise must be at least 1")
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
    if detections and publish_image_preview:
        annotated = render_detection_preview(image, raw_detections)
        publish_image_preview(
            annotated,
            kind="adetailer_detection",
            source_size=image.size,
            preserve_as_context=True,
            invalidate=True,
            hold_seconds=0.75,
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
            if invalidate_preview:
                invalidate_preview()
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
                preview_crop_box=list(crop_box),
                preview_region_index=index + 1,
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
            if publish_image_preview:
                publish_image_preview(
                    current_image.crop(crop_box),
                    kind="adetailer_crop_result",
                    source_size=(crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]),
                    crop_box=crop_box,
                    region_index=index + 1,
                    invalidate=True,
                )
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
    conditioning = prepare_prompt_conditioning(detail_pipeline, family, detail_prompt, detail_negative_prompt)
    sample_size = detail_pipeline.unet.config.sample_size
    vae_scale = detail_pipeline.vae_scale_factor
    if isinstance(sample_size, (tuple, list)):
        detail_height = int(sample_size[0]) * vae_scale
        detail_width = int(sample_size[-1]) * vae_scale
    else:
        detail_width = detail_height = int(sample_size) * vae_scale
    detail_preview_interval = max(1, ceil(effective_steps / PREVIEW_MAX_FRAMES)) if PREVIEW_MAX_FRAMES > 0 else effective_steps + 1

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
            if invalidate_preview:
                invalidate_preview()
            detail_last_preview_at = [0.0]
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
                preview_crop_box=list(crop_box),
                preview_region_index=index + 1,
            )

            def on_detail_step(_pipeline, step_index, _timestep, callback_kwargs, *, detection_index=index, running_phase=phase):
                detail_step = logical_diffusion_step(_pipeline, step_index, effective_steps)
                step_advanced = detail_step > detail_last_step[0]
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
                now = time.monotonic()
                scheduled_step = detail_step == 1 or detail_step == effective_steps or detail_step % detail_preview_interval == 0
                enough_time_passed = now - detail_last_preview_at[0] >= PREVIEW_MIN_INTERVAL
                if schedule_latent_preview and step_advanced and scheduled_step and enough_time_passed:
                    detail_last_preview_at[0] = now
                    schedule_latent_preview(
                        callback_kwargs["latents"],
                        family=family,
                        step=detail_step,
                        width=detail_width,
                        height=detail_height,
                        kind="adetailer_crop",
                        crop_box=crop_box,
                        region_index=detection_index + 1,
                    )
                return callback_kwargs

            generator = torch.Generator(device="cpu").manual_seed(
                ((request.seed if image_seed is None else image_seed) + index + 1) & 0xFFFFFFFFFFFFFFFF
            )
            kwargs = {
                "image": current_image,
                "mask_image": hard_mask,
                "strength": settings.denoise,
                "num_inference_steps": detail_steps,
                "guidance_scale": detail_cfg,
                "generator": generator,
                "callback_on_step_end": on_detail_step,
                "callback_on_step_end_tensor_inputs": ["latents"],
            }
            kwargs["padding_mask_crop"] = settings.padding
            kwargs.update(conditioning)
            detailed = run_guided_pipeline(
                pipeline, detail_pipeline, kwargs, guidance_kwargs
            ).images[0].convert("RGB")
            current_image = Image.composite(detailed, current_image, soft_mask)
            detection_states[index]["status"] = "complete"
            if publish_image_preview:
                publish_image_preview(
                    current_image.crop(crop_box),
                    kind="adetailer_crop_result",
                    source_size=(crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]),
                    crop_box=crop_box,
                    region_index=index + 1,
                    invalidate=True,
                )
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
    publish_image_preview=None,
    schedule_latent_preview=None,
    invalidate_preview=None,
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
            publish_image_preview=publish_image_preview,
            schedule_latent_preview=schedule_latent_preview,
            invalidate_preview=invalidate_preview,
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


def cleanup_job_previews(job_id: str):
    for preview_path in PREVIEW_DIRECTORY.glob(f"{job_id}-*"):
        remove_preview(preview_path)


def save_pil_preview(
    image: Image.Image,
    job_id: str,
    label: str,
    width: int | None = None,
    height: int | None = None,
    *,
    lossless=False,
):
    preview = image.convert("RGB")
    target_width = width or preview.width
    target_height = height or preview.height
    scale = 1.0 if lossless else min(PREVIEW_MAX_EDGE / max(target_width, target_height), 1.0)
    preview_size = (max(1, round(target_width * scale)), max(1, round(target_height * scale)))
    if preview.size != preview_size:
        preview = preview.resize(preview_size, Image.Resampling.BILINEAR)
    suffix = ".png" if lossless else ".jpg"
    preview_path = PREVIEW_DIRECTORY / f"{job_id}-{label}-{uuid.uuid4().hex}{suffix}"
    temporary_path = PREVIEW_DIRECTORY / f"{job_id}.{uuid.uuid4().hex}.tmp"
    try:
        preview.save(temporary_path, format="PNG", compress_level=3) if lossless else preview.save(temporary_path, format="JPEG", quality=82)
        os.replace(temporary_path, preview_path)
    finally:
        remove_preview(temporary_path)
    return preview_path


def save_latent_preview(latents: torch.Tensor, family: str, job_id: str, step: int, width: int, height: int, label: str = "sampling"):
    if family == "sdxl":
        factors = [
            [0.3651, 0.4232, 0.4341],
            [-0.2533, -0.0042, 0.1068],
            [0.1076, 0.1111, -0.0362],
            [-0.3165, -0.2492, -0.2188],
        ]
        bias = [0.1084, -0.0175, -0.0011]
    else:
        factors = [
            [0.3512, 0.2297, 0.3227],
            [0.3250, 0.4974, 0.2350],
            [-0.2829, 0.1762, 0.2721],
            [-0.2120, -0.2616, -0.7177],
        ]
        bias = None

    latent = latents[0].to(dtype=torch.float32).movedim(0, -1)
    weight = torch.tensor(factors, dtype=torch.float32).transpose(0, 1)
    bias_tensor = torch.tensor(bias, dtype=torch.float32) if bias else None
    rgb = torch.nn.functional.linear(latent, weight, bias_tensor)
    pixels = (((rgb + 1.0) / 2.0).clamp(0, 1) * 255).to(torch.uint8).numpy()
    preview = Image.fromarray(pixels, mode="RGB")
    return save_pil_preview(preview, job_id, f"{label}-{step}", width, height)


def create_output_path():
    now = datetime.now()
    directory = OUTPUT_DIRECTORY / now.strftime("%Y-%m-%d")
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"XirAI-{now:%Y-%m-%d-%H-%M}"
    output_path = directory / f"{stem}.png"
    index = 2
    while output_path.exists():
        output_path = directory / f"{stem}-{index:02d}.png"
        index += 1
    return output_path


def output_seed(request: GenerateInput, batch_index: int, image_index: int):
    offset = batch_index * request.images_per_batch + image_index
    return (request.seed + offset) & 0xFFFFFFFFFFFFFFFF


def batch_seeds(request: GenerateInput, batch_index: int):
    return [output_seed(request, batch_index, image_index) for image_index in range(request.images_per_batch)]


def secure_random_uint64():
    """Return one OS-backed unsigned 64-bit value without shared PRNG state."""
    return secrets.randbits(64)


def resolve_effective_hires_seed(settings: HiresInput, base_seed: int, random_uint64=None):
    """Resolve exactly one output's Hires seed after its base seed is known."""
    if not settings.enabled:
        return None
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or not 0 <= base_seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Base seed must be an unsigned 64-bit integer")
    if settings.mode == "inherit":
        return base_seed
    if settings.mode == "fixed":
        return int(settings.seed, 10)
    candidate = (random_uint64 or secure_random_uint64)()
    if isinstance(candidate, bool) or not isinstance(candidate, int) or not 0 <= candidate <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Random Hires seed source returned an invalid unsigned 64-bit integer")
    return candidate


def output_seed_facts(request: GenerateInput, base_seed: int, effective_hires_seed):
    return {
        "seed": str(base_seed),
        "base_seed": str(base_seed),
        "hires_seed_mode": request.hires.mode,
        "hires_seed": str(effective_hires_seed) if effective_hires_seed is not None else None,
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
    outputs = pipeline(**pipeline_kwargs).images
    expected = pipeline_kwargs["num_images_per_prompt"]
    if len(outputs) != expected:
        raise RuntimeError(f"Pipeline returned {len(outputs)} images for a requested batch of {expected}")
    return outputs


def save_image(
    image,
    job_id: str,
    request: GenerateInput,
    elapsed: float,
    wall_elapsed: float,
    paused_seconds: float,
    adetailer_result=None,
    hires_result=None,
    rtx_result=None,
    background_removal_result=None,
    *,
    conditioning_prompt=None,
    seed=None,
    effective_hires_seed=None,
    batch_index=1,
    image_index=1,
    before_commit=None,
):
    base_seed = request.seed if seed is None else seed
    if request.hires.enabled and effective_hires_seed is None:
        raise ValueError("Effective Hires.fix seed is required when saving a Hires output")
    output_path = create_output_path()
    temporary_path = output_path.with_name(f".{output_path.stem}-{uuid.uuid4().hex}.tmp")
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
        **output_seed_facts(request, base_seed, effective_hires_seed),
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
        "preview_enabled": request.preview_enabled,
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
        "hires": hires_result if request.hires.enabled else None,
        "adetailer": adetailer_result if request.adetailer.enabled else None,
        "rtx": rtx_result if request.rtx.enabled else None,
        "postprocess_order": list(request.postprocess_order),
        "postprocess_stages": postprocessing_stages(request),
        "postprocess_only": request.postprocess_only,
        "loras": [{"name": Path(item.path).name, "weight": item.weight} for item in request.loras],
        "elapsed_seconds": round(elapsed, 3),
        "wall_elapsed_seconds": round(wall_elapsed, 3),
        "paused_seconds": round(paused_seconds, 3),
    }
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
    metadata.add_text("parameters", json.dumps(parameters, ensure_ascii=False))
    if background_removal_result and image.mode != "RGBA":
        raise RuntimeError("透明背景处理未生成 RGBA 图像，已停止保存")
    try:
        image.save(temporary_path, format="PNG", pnginfo=metadata)
        if before_commit:
            before_commit()
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
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


def write_generation_failure_log(job_id: str, request: GenerateInput, error: Exception, failure_kind=None):
    try:
        with jobs_lock:
            job = jobs.get(job_id, {})
        out_of_memory = is_oom_error(error)
        failure_kind = failure_kind or ("generation-memory-failure" if out_of_memory else "generation-failure")
        details = {
            "job_id": job_id,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
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
                "seed": request.seed,
                "images_per_batch": request.images_per_batch,
                "batch_count": request.batch_count,
                "sampler": request.sampler,
                "scheduler": request.scheduler,
                "guidance": request.guidance,
                "pag": request.pag.model_dump(),
                "preview_enabled": request.preview_enabled,
                "source_image_enabled": request.source_image.enabled,
                "source_image_resize_mode": request.source_image.resize_mode if request.source_image.enabled else None,
                "postprocess_only": request.postprocess_only,
                "lora_count": len(request.loras),
                "loras": [{"path": item.path, "weight": item.weight} for item in request.loras],
                "adetailer_enabled": request.adetailer.enabled,
                "adetailer_units": len(request.adetailer.active_units),
                "adetailer_detector": " · ".join(unit.detector for unit in request.adetailer.active_units) or None,
                "hires_enabled": request.hires.enabled,
                "hires_model": request.hires.model if request.hires.enabled else None,
                "hires_scale": request.hires.scale if request.hires.enabled else None,
                "rtx_enabled": request.rtx.enabled,
                "rtx_scale": request.rtx.scale if request.rtx.enabled else None,
                "rtx_quality": request.rtx.quality if request.rtx.enabled else None,
                "postprocess_order": list(request.postprocess_order),
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
            },
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
            f"Message: {error}",
            "",
            "Details:",
            json.dumps(details, ensure_ascii=False, indent=2),
            "",
            "Traceback:",
            traceback.format_exc(),
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


def run_generation(job_id: str, request: GenerateInput):
    global active_attention_backend, active_vae_mode
    original_request = request
    cleaned_prompt, directives = parse_prompt_directives(request.prompt)
    transparent_background = directives["transparent_background"]
    conditioning_prompt = transparent_conditioning_prompt(cleaned_prompt) if transparent_background else cleaned_prompt
    request = request.model_copy(update={"prompt": conditioning_prompt})
    started_at = time.time()
    control = job_controls[job_id]
    preview_busy = threading.Event()
    preview_finished = threading.Event()
    preview_lock = threading.Lock()
    ultra_low_memory = performance_settings["memory_mode"] == "ultra_low_vram"
    previews_enabled = request.preview_enabled and not ultra_low_memory and PREVIEW_MAX_FRAMES > 0
    preview_interval = max(1, ceil(request.steps / PREVIEW_MAX_FRAMES)) if previews_enabled else request.steps + 1
    last_preview_at = [0.0]
    preview_revision = [0]
    preview_epoch = [0]
    preview_hold_until = [0.0]
    pipeline = None
    adetailer_result = None
    hires_result = None
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

    def invalidate_preview():
        with preview_lock:
            preview_epoch[0] += 1
            return preview_epoch[0]

    def publish_preview_path(
        preview_path: Path,
        *,
        expected_epoch: int,
        kind: str,
        source_size,
        crop_box=None,
        region_index=0,
        preserve_as_context=False,
    ):
        with preview_lock:
            if preview_finished.is_set() or expected_epoch != preview_epoch[0]:
                remove_preview(preview_path)
                return False
            preview_revision[0] += 1
            with jobs_lock:
                job = jobs.get(job_id)
                if not job:
                    remove_preview(preview_path)
                    return False
                previous_path = job.get("preview_path")
                job.update(
                    preview_path=str(preview_path),
                    preview_url=f"/api/inference/jobs/{job_id}/preview",
                    preview_version=preview_revision[0],
                    preview_kind=kind,
                    preview_source_size=list(source_size),
                    preview_crop_box=list(crop_box) if crop_box else None,
                    preview_region_index=region_index,
                )
                if preserve_as_context:
                    job.update(
                        context_preview_path=str(preview_path),
                        context_preview_url=f"/api/inference/jobs/{job_id}/preview?kind=context",
                        context_preview_version=preview_revision[0],
                    )
                context_path = job.get("context_preview_path")
            if previous_path and previous_path != str(preview_path) and previous_path != context_path:
                remove_preview(previous_path)
            return True

    def publish_image_preview(
        image: Image.Image,
        *,
        kind: str,
        source_size,
        crop_box=None,
        region_index=0,
        preserve_as_context=False,
        invalidate=False,
        hold_seconds=0.0,
    ):
        if not previews_enabled:
            return False
        expected_epoch = invalidate_preview() if invalidate else preview_epoch[0]
        preview_path = save_pil_preview(image, job_id, kind, lossless=kind == "adetailer_detection")
        published = publish_preview_path(
            preview_path,
            expected_epoch=expected_epoch,
            kind=kind,
            source_size=source_size,
            crop_box=crop_box,
            region_index=region_index,
            preserve_as_context=preserve_as_context,
        )
        if published and hold_seconds > 0:
            preview_hold_until[0] = time.monotonic() + hold_seconds
        return published

    def encode_preview(cpu_latents: torch.Tensor, family: str, step: int, width: int, height: int, kind: str, crop_box, region_index: int, expected_epoch: int):
        try:
            preview_path = save_latent_preview(cpu_latents, family, job_id, step, width, height, kind)
            publish_preview_path(
                preview_path,
                expected_epoch=expected_epoch,
                kind=kind,
                source_size=(width, height),
                crop_box=crop_box,
                region_index=region_index,
            )
        except Exception:
            traceback.print_exc()
        finally:
            preview_busy.clear()

    def schedule_latent_preview(latents: torch.Tensor, *, family: str, step: int, width: int, height: int, kind: str, crop_box=None, region_index=0):
        if not previews_enabled or preview_busy.is_set() or time.monotonic() < preview_hold_until[0]:
            return False
        preview_busy.set()
        with preview_lock:
            expected_epoch = preview_epoch[0]
        cpu_latents = latents[:1].detach().to(device="cpu", dtype=torch.float32)
        try:
            preview_executor.submit(
                encode_preview,
                cpu_latents,
                family,
                step,
                width,
                height,
                kind,
                crop_box,
                region_index,
                expected_epoch,
            )
            return True
        except RuntimeError:
            preview_busy.clear()
            return False

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
            text_encoder = validate_anima_component(
                request.text_encoder, roots["text_encoder"], "Flux text encoder"
            )
            text_encoder_2 = validate_anima_component(
                request.text_encoder_2, roots["text_encoder"], "Flux second text encoder"
            )
            vae = validate_anima_component(request.vae, roots["vae"], "Flux VAE")
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
            text_encoder = validate_anima_component(
                request.text_encoder, roots["text_encoder"], "Flux2 text encoder"
            )
            text_encoder_2 = None
            vae = validate_anima_component(request.vae, roots["vae"], "Flux2 VAE")
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
            text_encoder = validate_anima_component(
                request.text_encoder, roots["text_encoder"], "Krea2 text encoder"
            )
            text_encoder_2 = None
            vae = validate_anima_component(request.vae, roots["vae"], "Krea2 VAE")
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
            prompt_info = prompt_diagnostics(pipeline.tokenizer, request.prompt)
            negative_prompt_info = prompt_diagnostics(pipeline.tokenizer, request.negative_prompt)
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

        conditioning = None if is_native_family(family) else prepare_prompt_conditioning(
            pipeline, family, request.prompt, request.negative_prompt
        )
        if ultra_low_memory:
            update_job(job_id, phase="Releasing prompt encoders", progress=24)
            release_prompt_encoders(pipeline)
        total_images = request.images_per_batch * request.batch_count
        sampling_steps = base_sampling_steps(request, family)
        if sampling_steps != request.steps:
            preview_interval = max(1, ceil(sampling_steps / PREVIEW_MAX_FRAMES)) if previews_enabled else sampling_steps + 1
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
            postprocessing_enabled = bool(enabled_post_stages)
            # Post-processing mode samples and decodes nothing, so both of those slices collapse onto
            # the start of the batch and the stages get the whole budget.
            sampling_end = batch_start if request.postprocess_only else batch_start + (batch_end - batch_start) * (0.68 if request.hires.enabled else 0.82 if postprocessing_enabled else 0.92)
            decode_end = batch_start if request.postprocess_only else batch_start + (batch_end - batch_start) * (0.74 if postprocessing_enabled else 1.0)
            phase = (
                f"Batch {batch_index + 1}/{request.batch_count} · Post-processing {request.images_per_batch} images"
                if request.postprocess_only
                else f"Batch {batch_index + 1}/{request.batch_count} · Sampling {request.images_per_batch} images"
            )
            last_base_step = [0]

            def on_step_end(_pipeline, step_index, _timestep, callback_kwargs, *, running_phase=phase, progress_start=batch_start, progress_end=sampling_end):
                step = logical_diffusion_step(_pipeline, step_index, sampling_steps)
                step_advanced = step > last_base_step[0]
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

                now = time.monotonic()
                scheduled_step = step == 1 or step == sampling_steps or step % preview_interval == 0
                enough_time_passed = now - last_preview_at[0] >= PREVIEW_MIN_INTERVAL
                if step_advanced and scheduled_step and enough_time_passed and schedule_latent_preview(
                    callback_kwargs["latents"],
                    family=family,
                    step=step,
                    width=request.width,
                    height=request.height,
                    kind="base_sampling",
                ):
                    last_preview_at[0] = now
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
                if (
                    getattr(pipeline, "transformer_group_offload_enabled", False)
                    and active_memory_strategy is not None
                    and not active_memory_strategy.get("transformer_group_offload", False)
                ):
                    active_memory_strategy.update(
                        mode="low_vram",
                        label="LOW_VRAM 低显存",
                        offload_mode="staged_transformer_group_offload",
                        model_resident=False,
                        transformer_group_offload=True,
                        transformer_blocks_per_group=1,
                        reason=active_memory_strategy["reason"] + "；完整常驻显存不足，已自动回退单块动态装入",
                    )
                    admission = active_memory_strategy.get("admission")
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
                runtime_attention_backend = getattr(pipeline, "attention_backend", None)
                if runtime_attention_backend:
                    active_attention_backend = anima_attention_backend_label(runtime_attention_backend)
                if bool(getattr(getattr(pipeline, "vae", None), "use_tiling", False)):
                    active_vae_mode = "tiled"
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
                if performance_settings["staged_vae_decode"]:
                    pipeline_kwargs["output_type"] = "latent"
                    latents = run_guided_pipeline_batch(pipeline, request.guidance, pipeline_kwargs, request.pag, sampling_factory)
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
                    )
                    del latents
                else:
                    images = run_guided_pipeline_batch(pipeline, request.guidance, pipeline_kwargs, request.pag, sampling_factory)
            if len(images) != request.images_per_batch:
                raise RuntimeError(
                    f"Pipeline returned {len(images)} images for a requested batch of {request.images_per_batch}"
                )

            for image_index, image in enumerate(images):
                control.checkpoint(job_id, f"Processing image {image_index + 1}")
                base_seed = seeds[image_index]
                effective_hires_seed = resolve_effective_hires_seed(request.hires, base_seed)
                seed_facts = output_seed_facts(request, base_seed, effective_hires_seed)
                update_job(job_id, **seed_facts)
                detail_start = decode_end + (batch_end - decode_end) * image_index / request.images_per_batch
                detail_end = decode_end + (batch_end - decode_end) * (image_index + 1) / request.images_per_batch
                adetailer_result = None
                hires_result = None
                rtx_result = None
                post_stages = enabled_post_stages
                for stage_index, post_stage in enumerate(post_stages):
                    stage_start = detail_start + (detail_end - detail_start) * stage_index / len(post_stages)
                    stage_end = detail_start + (detail_end - detail_start) * (stage_index + 1) / len(post_stages)
                    if post_stage == "adetailer" and pipeline_cpu_parked:
                        control.checkpoint(job_id, "Restoring sampler for ADetailer")
                        update_job(job_id, phase="Restoring model for ADetailer", stage="model_restore", progress=round(stage_start))
                        restore_parked_pipeline()
                    if post_stage == "hires":
                        image, hires_result = apply_hires_fix(
                            image,
                            pipeline,
                            family,
                            request,
                            job_id,
                            control,
                            started_at,
                            schedule_latent_preview=schedule_latent_preview,
                            invalidate_preview=invalidate_preview,
                            image_seed=base_seed,
                            effective_hires_seed=effective_hires_seed,
                            progress_start=round(stage_start),
                            progress_end=round(stage_end),
                        )
                        hires_warning = hires_result.get("warning") if hires_result else None
                        if hires_warning and hires_warning not in (warning or ""):
                            warning = append_warning(warning, hires_warning)
                            update_job(job_id, warning=warning)
                    elif post_stage == "adetailer":
                        image, adetailer_result, adetailer_warning = apply_adetailer(
                            image,
                            pipeline,
                            family,
                            request,
                            job_id,
                            control,
                            started_at,
                            publish_image_preview=publish_image_preview,
                            schedule_latent_preview=schedule_latent_preview,
                            invalidate_preview=invalidate_preview,
                            image_seed=base_seed,
                            progress_start=round(stage_start),
                            progress_end=round(stage_end),
                        )
                        if adetailer_warning and adetailer_warning not in (warning or ""):
                            warning = append_warning(warning, adetailer_warning)
                            update_job(job_id, warning=warning)
                    elif post_stage == "rtx":
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
                        raise RuntimeError(f"Unsupported post-processing stage: {post_stage}")
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
                wall_elapsed = time.time() - started_at
                paused_seconds = control.total_paused()
                elapsed = max(0.0, wall_elapsed - paused_seconds)
                output_path = save_image(
                    image,
                    job_id,
                    original_request,
                    elapsed,
                    wall_elapsed,
                    paused_seconds,
                    adetailer_result,
                    hires_result,
                    rtx_result,
                    background_removal_result,
                    conditioning_prompt=conditioning_prompt,
                    seed=base_seed,
                    effective_hires_seed=effective_hires_seed,
                    batch_index=batch_index + 1,
                    image_index=image_index + 1,
                    before_commit=lambda: control.checkpoint(job_id, "Committing image"),
                )
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
                    "image_url": f"/api/inference/jobs/{job_id}/images/{completed_images - 1}",
                    "transparent_background": transparent_background,
                    "background_removal": background_removal_result,
                    "postprocess_stages": post_stages,
                    "hires": hires_result,
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
        release_pipeline_after_job()
        with preview_lock:
            preview_finished.set()
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
                preview_path=None,
                preview_url=None,
                context_preview_path=None,
                context_preview_url=None,
                image_url=f"/api/inference/jobs/{job_id}/image",
                elapsed_seconds=round(elapsed, 1),
                wall_elapsed_seconds=round(wall_elapsed, 1),
                paused_seconds=round(paused_seconds, 1),
                completed_at=time.time(),
                **pipeline_status_fields(),
            )
            cleanup_job_previews(job_id)
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
            preview_path=None,
            preview_url=None,
            context_preview_path=None,
            context_preview_url=None,
            elapsed_seconds=round(control.active_elapsed(started_at), 1),
            paused_seconds=round(control.total_paused(), 1),
            completed_at=time.time(),
            **pipeline_status_fields(),
        )
    except Exception as error:
        cleanup_partial_outputs()
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
                error="GPU memory exhausted. Try a smaller canvas, fewer images per batch, or fewer LoRAs.",
                progress=0,
                completed_images=0,
                outputs=[],
                preview_path=None,
                preview_url=None,
                context_preview_path=None,
                context_preview_url=None,
                **pipeline_status_fields(),
            )
        else:
            if loaded_pipeline is None or not restore_cached_pipeline_state(job_id, request):
                pipeline = None
                clear_pipeline()
            release_pipeline_after_job()
            update_job(job_id, status="error", phase="Failed", stage="error", error=str(error), progress=0, completed_images=0, outputs=[], preview_path=None, preview_url=None, context_preview_path=None, context_preview_url=None, **pipeline_status_fields())
    finally:
        with preview_lock:
            preview_finished.set()
            cleanup_job_previews(job_id)
            update_job(job_id, preview_path=None, preview_url=None, context_preview_path=None, context_preview_url=None)
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
        "memory_mode": active_memory_strategy["mode"] if active_memory_strategy else None,
        "memory_label": active_memory_strategy["label"] if active_memory_strategy else None,
        "memory_reason": active_memory_strategy["reason"] if active_memory_strategy else None,
        "memory_total_gb": active_memory_strategy["total_gb"] if active_memory_strategy else None,
        "memory_free_gb": active_memory_strategy["free_gb"] if active_memory_strategy else None,
        "memory_weight_gb": active_memory_strategy["weight_gb"] if active_memory_strategy else None,
        "memory_inference_gb": active_memory_strategy["inference_gb"] if active_memory_strategy else None,
        "memory_reserved_gb": active_memory_strategy["reserved_gb"] if active_memory_strategy else None,
        "offload_mode": "cpu_parked" if pipeline_cpu_parked else active_memory_strategy["offload_mode"] if active_memory_strategy else None,
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
            "memory_mode": active_memory_strategy["mode"] if active_memory_strategy else None,
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
        },
    }


@app.get("/api/inference/performance")
def get_performance_settings():
    return performance_payload()


@app.put("/api/inference/performance")
def update_performance_settings(request: PerformanceInput):
    with pipeline_lock:
        with jobs_lock:
            busy = any(job["status"] in {"queued", "running", "pausing", "paused", "cancelling"} for job in jobs.values())
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
            busy = any(job["status"] in {"queued", "running", "pausing", "paused", "cancelling"} for job in jobs.values())
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
            "memory_mode": active_memory_strategy["mode"],
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
    if request.hires.enabled:
        try:
            resolve_upscaler_model(UPSCALER_MODEL_DIRECTORY, request.hires.model)
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
    _, directives = parse_prompt_directives(request.prompt)
    if directives["transparent_background"]:
        if not request.background_removal_model:
            raise HTTPException(status_code=422, detail="请选择透明背景模型")
        try:
            require_background_removal_model(request.background_removal_model)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    with jobs_lock:
        if any(job["status"] in {"queued", "running", "pausing", "paused", "cancelling"} for job in jobs.values()):
            raise HTTPException(status_code=409, detail="Another image is already being generated")
        job_id = uuid.uuid4().hex
        job_controls[job_id] = JobControl()
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "phase": "Queued",
            "stage": "queued",
            "stage_step": 0,
            "stage_total": 0,
            "progress": 0,
            "step": 0,
            "total_steps": request.steps,
            "images_per_batch": request.images_per_batch,
            "batch_count": request.batch_count,
            "batch_index": 0,
            "completed_images": 0,
            "total_images": request.images_per_batch * request.batch_count,
            "outputs": [],
            "seed": str(request.seed),
            "base_seed": str(request.seed),
            "hires_seed_mode": request.hires.mode,
            "hires_seed": None,
            "transparent_background": directives["transparent_background"],
            # The picture itself never reaches the job record: it is polled several times a second
            # by the workspace, and a megabyte of base64 in every response would be the dominant
            # cost of watching a run.
            "source_image_enabled": request.source_image.enabled,
            "source_image_name": request.source_image.name or None,
            "source_image_resize_mode": request.source_image.resize_mode if request.source_image.enabled else None,
            "background_removal_model": request.background_removal_model,
            "background_removal": None,
            "hires_enabled": request.hires.enabled,
            "hires_model": request.hires.model if request.hires.enabled else None,
            "rtx_enabled": request.rtx.enabled,
            "rtx_scale": request.rtx.scale if request.rtx.enabled else None,
            "rtx_quality": request.rtx.quality if request.rtx.enabled else None,
            "postprocess_order": list(request.postprocess_order),
            "postprocess_stages": postprocessing_stages(request),
            "postprocess_only": request.postprocess_only,
            "created_at": time.time(),
            "error": None,
            "warning": None,
            "prompt_tokens": None,
            "prompt_blocks": None,
            "prompt_weighted_tokens": None,
            "negative_prompt_tokens": None,
            "negative_prompt_blocks": None,
            "negative_prompt_weighted_tokens": None,
            "preview_path": None,
            "preview_url": None,
            "preview_version": 0,
            "preview_kind": None,
            "preview_source_size": None,
            "preview_crop_box": None,
            "preview_region_index": 0,
            "context_preview_path": None,
            "context_preview_url": None,
            "context_preview_version": 0,
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
        }
    executor.submit(run_generation, job_id, request)
    return jobs[job_id]


@app.get("/api/inference/jobs/active")
def get_active_job():
    with jobs_lock:
        active = [job for job in jobs.values() if job["status"] in {"queued", "running", "pausing", "paused", "cancelling"}]
        if not active:
            return {"job": None}
        job = max(active, key=lambda item: item.get("created_at", 0))
        return {"job": {key: value for key, value in job.items() if key not in {"output_path", "output_paths", "preview_path", "context_preview_path"}}}


@app.get("/api/inference/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Generation job not found")
        payload = {
            key: value for key, value in job.items()
            if key not in {"output_path", "output_paths", "preview_path", "context_preview_path"}
        }
        control = job_controls.get(job_id)
    if control and job.get("started_at"):
        payload["elapsed_seconds"] = round(control.active_elapsed(job["started_at"]), 1)
        payload["paused_seconds"] = round(control.total_paused(), 1)
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
    update_job(job_id, status="cancelling", phase="Stopping after current step")
    return {"status": "cancelling"}


@app.get("/api/inference/jobs/{job_id}/preview")
def get_job_preview(job_id: str, kind: str | None = None):
    with jobs_lock:
        job = jobs.get(job_id)
        preview_path = job.get("context_preview_path" if kind == "context" else "preview_path") if job else None
        if not preview_path or not Path(preview_path).is_file():
            raise HTTPException(status_code=404, detail="Step preview is not available")
    media_type = "image/png" if Path(preview_path).suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(preview_path, media_type=media_type, headers={"Cache-Control": "no-store"})


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
    if source.suffix.lower() == ".gif":
        try:
            return Response(content=source.read_bytes(), media_type="image/gif", headers={"Cache-Control": "no-store"})
        except OSError as error:
            raise HTTPException(status_code=500, detail="Unable to copy GIF") from error
    try:
        with Image.open(source) as image:
            copied = image.copy()
            copied.info.clear()
            buffer = io.BytesIO()
            copied.save(buffer, format="PNG", optimize=False)
    except OSError as error:
        raise HTTPException(status_code=500, detail="Unable to copy image") from error
    return Response(content=buffer.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/api/inference/collages", status_code=201)
def save_collage(input_data: CollageInput):
    try:
        encoded = input_data.image_data.split(",", 1)[-1]
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 150_000_000:
            raise ValueError("collage is too large")
        image = Image.open(io.BytesIO(raw))
        image.load()
        if image.width < 1 or image.height < 1:
            raise ValueError("invalid collage dimensions")
        is_gif = image.format == "GIF" or input_data.name.lower().endswith(".gif")
        output_path = create_named_output_path(input_data.name if is_gif else input_data.name.removesuffix(".gif") + ".png")
        if is_gif:
            frames = []
            durations = []
            for frame_index in range(max(1, int(getattr(image, "n_frames", 1)))):
                image.seek(frame_index)
                frames.append(image.convert("RGBA"))
                durations.append(max(20, int(image.info.get("duration", 100))))
            comment = json.dumps({"manual_layout": input_data.manual_layout}, ensure_ascii=False).encode("utf-8") if input_data.manual_layout else None
            frames[0].save(output_path, format="GIF", save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=2, optimize=False, comment=comment)
        else:
            metadata = PngInfo()
            if input_data.manual_layout:
                metadata.add_text("parameters", json.dumps({"manual_layout": input_data.manual_layout}, ensure_ascii=False))
            image.convert("RGBA").save(output_path, format="PNG", optimize=False, pnginfo=metadata)
    except (ValueError, binascii.Error, OSError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid collage image: {error}") from error
    return {
        "name": output_path.name,
        "url": f"/api/inference/history/assets/{history_asset_token(output_path)}",
        "id": history_asset_token(output_path),
    }


@app.post("/api/inference/collages/animated")
def render_animated_collage(input_data: AnimatedCollageInput):
    try:
        frames, durations = animated_collage_frames(input_data)
        buffer = io.BytesIO()
        frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=2, optimize=False)
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return Response(content=buffer.getvalue(), media_type="image/gif", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    write_cpu_mode_log()
    uvicorn.run(app, host=INFERENCE_HOST, port=INFERENCE_PORT, log_level="info", access_log=False)
