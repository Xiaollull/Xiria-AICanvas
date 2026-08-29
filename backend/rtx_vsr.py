import json
import math
import os
import platform
import signal
import subprocess
import sys
import tempfile
import threading
import time
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import torch
from PIL import Image


WORKER_PATH = Path(__file__).with_name("rtx_vsr_worker.py")
RUNTIME_VERSION = "0.1.0.1"
WORKER_PROTOCOL = 1
QUALITY_LEVELS = {"low", "medium", "high", "ultra"}
MAX_OUTPUT_EDGE = 8192
MAX_OUTPUT_PIXELS = 32 * 1024 * 1024
_status_cache = None
_status_lock = threading.Lock()
_probe_thread = None


def target_size(image_or_size, scale):
    width, height = image_or_size.size if isinstance(image_or_size, Image.Image) else image_or_size
    if not math.isfinite(scale) or scale < 1 or scale > 4:
        raise ValueError("RTX VSR scale must be between 1.0 and 4.0")
    destination = (
        max(8, round(int(width * scale) / 8) * 8),
        max(8, round(int(height * scale) / 8) * 8),
    )
    if max(destination) > MAX_OUTPUT_EDGE or destination[0] * destination[1] > MAX_OUTPUT_PIXELS:
        raise ValueError("RTX VSR output exceeds the safe 8192-edge / 32-megapixel limit")
    return destination


def _package_version():
    try:
        return package_version("nvidia-vfx")
    except PackageNotFoundError:
        return None


def _driver_version():
    try:
        result = subprocess.run(
            [os.environ.get("NVIDIA_SMI_PATH") or "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        return result.stdout.splitlines()[0].strip() if result.returncode == 0 and result.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def _numeric_version(value):
    try:
        major, minor, *_rest = str(value).split(".")
        return int(major), int(minor)
    except (TypeError, ValueError):
        return None


def _driver_supported(system, value):
    driver = _numeric_version(value)
    if not driver:
        return False
    if system == "Windows":
        return driver >= (570, 65)
    major, minor = driver
    return (major == 570 and minor >= 190) or (major == 580 and minor >= 82) or major > 590 or (major == 590 and minor >= 44)


def _base_status():
    system = platform.system()
    if system not in {"Windows", "Linux"} or platform.machine().lower() not in {"amd64", "x86_64"}:
        return {"available": False, "supported": False, "probing": False, "reason": "RTX VSR only supports x64 Windows and Linux"}
    if os.environ.get("WSL_DISTRO_NAME") or "microsoft" in platform.release().lower():
        return {"available": False, "supported": False, "probing": False, "reason": "RTX VSR is disabled on WSL because the native runtime is not reliable"}
    if not torch.cuda.is_available():
        return {"available": False, "supported": False, "probing": False, "reason": "CUDA is unavailable"}
    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 5):
        return {"available": False, "supported": False, "probing": False, "reason": "The NVIDIA GPU does not provide supported RTX Tensor Cores"}
    driver = _driver_version()
    driver_warning = None
    if not _driver_supported(system, driver):
        minimum = "570.65+" if system == "Windows" else "570.190+, 580.82+, or 590.44+"
        driver_warning = f"NVIDIA documents driver {minimum}; availability is determined by the isolated runtime probe"
    runtime_version = _package_version()
    base = {
        "available": False,
        "supported": True,
        "probing": False,
        "runtime_version": runtime_version,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "driver_version": driver,
        "warning": driver_warning,
    }
    if not runtime_version:
        return {**base, "reason": "The optional NVIDIA VFX runtime is not installed"}
    if runtime_version != RUNTIME_VERSION:
        return {**base, "reason": f"RTX VSR requires nvidia-vfx {RUNTIME_VERSION}; found {runtime_version}"}
    return base


def _parse_result(stdout):
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                if value.get("protocol") != WORKER_PROTOCOL:
                    raise RuntimeError("RTX VSR worker protocol mismatch")
                return value
        except json.JSONDecodeError:
            continue
    raise RuntimeError("RTX VSR worker returned invalid output")


def _run_probe():
    global _status_cache, _probe_thread
    result = {
        "available": False,
        "supported": False,
        "probing": False,
        "reason": "RTX VSR runtime probe failed",
    }
    try:
        result = _base_status()
        if result.get("supported") and result.get("runtime_version") == RUNTIME_VERSION:
            probe = subprocess.run(
                [sys.executable, "-I", str(WORKER_PATH), "--probe"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                check=False,
            )
            if probe.returncode != 0:
                detail = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else f"exit code {probe.returncode}"
                raise RuntimeError(detail)
            worker = _parse_result(probe.stdout)
            result.update(worker, available=True, supported=True, probing=False, reason=None, checked_at=time.time())
    except Exception as error:
        result = {
            "available": False,
            "supported": False,
            "probing": False,
            "reason": f"RTX VSR runtime probe failed: {error}",
            "checked_at": time.time(),
        }
    finally:
        with _status_lock:
            _status_cache = result
            _probe_thread = None


def start_probe(refresh=False):
    global _status_cache, _probe_thread
    with _status_lock:
        if _probe_thread is not None and _probe_thread.is_alive():
            return dict(_status_cache)
        if _status_cache is not None and not refresh:
            return dict(_status_cache)
        _status_cache = {
            "available": False,
            "supported": None,
            "probing": True,
            "reason": "RTX VSR runtime is being verified",
        }
        _probe_thread = threading.Thread(target=_run_probe, name="rtx-vsr-probe", daemon=True)
        _probe_thread.start()
        return dict(_status_cache)


def status(refresh=False):
    return start_probe(refresh=refresh)


def require_available():
    current = status()
    if not current.get("available"):
        raise ValueError(current.get("reason") or "RTX VSR is unavailable")
    return current


def _terminate_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=5)


def _read_worker_log(path, limit=32_768):
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode("utf-8", "replace")
    except OSError:
        return ""


def upscale_image(image, scale, quality="ultra", progress=None, checkpoint=None, cancel_check=None, timeout=300):
    runtime = require_available()
    if quality not in QUALITY_LEVELS:
        raise ValueError("Unsupported RTX VSR quality level")
    destination = target_size(image, scale)
    if checkpoint:
        checkpoint()
    with tempfile.TemporaryDirectory(prefix="xirai-rtx-vsr-") as temporary:
        root = Path(temporary)
        input_path = root / "input.png"
        output_path = root / "output.png"
        stdout_path = root / "stdout.log"
        stderr_path = root / "stderr.log"
        image.convert("RGB").save(input_path)
        with stdout_path.open("w", encoding="utf-8") as stdout_stream, stderr_path.open("w", encoding="utf-8") as stderr_stream:
            process = subprocess.Popen(
                [
                    sys.executable, "-I", str(WORKER_PATH),
                    "--input", str(input_path), "--output", str(output_path),
                    "--width", str(destination[0]), "--height", str(destination[1]),
                    "--quality", quality,
                ],
                stdout=stdout_stream,
                stderr=stderr_stream,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    if cancel_check:
                        cancel_check()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("RTX VSR processing timed out")
                    time.sleep(0.1)
            except BaseException:
                _terminate_process(process)
                raise
        stdout = _read_worker_log(stdout_path)
        stderr = _read_worker_log(stderr_path)
        if process.returncode != 0:
            detail = stderr.strip().splitlines()[-1] if stderr.strip() else f"exit code {process.returncode}"
            raise RuntimeError(f"RTX VSR processing failed: {detail}")
        worker_result = _parse_result(stdout)
        if worker_result.get("width") != destination[0] or worker_result.get("height") != destination[1]:
            raise RuntimeError("RTX VSR worker reported an unexpected output size")
        if not output_path.is_file():
            raise RuntimeError("RTX VSR did not create an output image")
        with Image.open(output_path) as opened:
            output = opened.convert("RGB").copy()
    if output.size != destination:
        raise RuntimeError("RTX VSR returned an unexpected output size")
    if progress:
        progress(1, 1)
    if checkpoint:
        checkpoint()
    return output, {
        "provider": "NVIDIA RTX VSR",
        "runtime_version": runtime.get("runtime_version"),
        "device": runtime.get("device"),
        "quality": quality,
        "scale": scale,
        "source_size": list(image.size),
        "target_size": list(destination),
    }
