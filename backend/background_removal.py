import hashlib
import json
import os
import re
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .model_paths import resolve_model_directory
except ImportError:
    from model_paths import resolve_model_directory


TRANSPARENT_BACKGROUND_PATTERN = re.compile(
    r"\(\{\s*transparent\s+background\s*\}\)",
    re.IGNORECASE,
)
CONDITIONING_SUFFIX = "isolated foreground subject, clean unobstructed silhouette, solid plain background, no background objects touching the subject"
MODEL_CATALOG_PATH = Path(__file__).resolve().parents[1] / "models" / "background-removal-models.json"
MODEL_DIRECTORY = resolve_model_directory(Path(__file__).resolve().parents[1], "background_removal")

_session_lock = threading.Lock()
_inference_lock = threading.Lock()
_sessions = {}


def parse_prompt_directives(prompt: str):
    """Return diffusion conditioning text and normalized prompt directives."""
    enabled = bool(TRANSPARENT_BACKGROUND_PATTERN.search(prompt))
    if enabled:
        directive = TRANSPARENT_BACKGROUND_PATTERN.pattern
        cleaned = re.sub(rf",\s*{directive}", "", prompt, flags=re.IGNORECASE)
        cleaned = re.sub(rf"{directive}\s*,\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = TRANSPARENT_BACKGROUND_PATTERN.sub("", cleaned).strip()
    else:
        cleaned = prompt
    return cleaned, {"transparent_background": enabled}


def transparent_conditioning_prompt(cleaned_prompt: str):
    return f"{cleaned_prompt}, {CONDITIONING_SUFFIX}"


@lru_cache(maxsize=1)
def model_catalog():
    try:
        payload = json.loads(MODEL_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    models = payload.get("models")
    if payload.get("schema") != 3 or not isinstance(models, list) or not models:
        return None
    valid = []
    for model in models:
        if not isinstance(model, dict):
            return None
        filename = model.get("filename")
        sha256 = model.get("sha256")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or Path(filename).suffix.lower() != ".onnx"
            or not isinstance(sha256, str)
            or not re.fullmatch(r"[a-f0-9]{64}", sha256, re.IGNORECASE)
            or model.get("output") not in {"logits", "probability", "minmax"}
            or not isinstance(model.get("input_size"), int)
            or not 256 <= model["input_size"] <= 1536
        ):
            return None
        for key in ("modelscope_repository", "modelscope_revision"):
            if key in model and not isinstance(model[key], str):
                return None
        valid.append(model)
    return {**payload, "models": sorted(valid, key=lambda item: item.get("priority", 0), reverse=True)}


@lru_cache(maxsize=4)
def _file_hash(path: str, size: int, modified_ns: int):
    del size, modified_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def model_path(model=None):
    catalog = model_catalog()
    if not catalog:
        return None
    model = model or catalog["models"][0]
    root = MODEL_DIRECTORY.resolve()
    direct = (root / model["filename"]).resolve()
    candidates = [direct]
    try:
        candidates.extend(path.resolve() for path in sorted(root.rglob("*.onnx"), key=lambda path: path.as_posix().lower()) if path.resolve() != direct)
    except OSError:
        pass
    for candidate in candidates:
        try:
            candidate.relative_to(root)
            file_stat = candidate.stat()
        except (OSError, ValueError):
            continue
        expected_size = model.get("size")
        if isinstance(expected_size, int) and expected_size > 0 and file_stat.st_size != expected_size:
            continue
        if _file_hash(str(candidate), file_stat.st_size, file_stat.st_mtime_ns) == model["sha256"].lower():
            return candidate
    return None


def custom_models():
    """Discover compatible ONNX files recursively under the configured model root."""
    catalog = (model_catalog() or {"models": []})["models"]
    catalog_filenames = {model["filename"] for model in catalog}
    official_paths = {path.resolve() for model in catalog if (path := model_path(model)) is not None}
    discovered = []
    try:
        candidates = sorted(MODEL_DIRECTORY.rglob("*.onnx"), key=lambda path: path.as_posix().lower())
        for path in candidates:
            relative = path.resolve().relative_to(MODEL_DIRECTORY.resolve()).as_posix()
            if path.name in catalog_filenames or path.resolve() in official_paths or not path.is_file():
                continue
            discovered.append({
                "id": f"local:{relative}",
                "filename": relative,
                "label": str(Path(relative).with_suffix("")),
                "description": "本地兼容 ONNX 抠图模型",
                "size": path.stat().st_size,
                "input_size": 1024,
                "output": "auto",
                "priority": 0,
                "local": True,
                "selectable": True,
            })
    except OSError:
        return discovered
    return discovered


def available_models():
    catalog = model_catalog() or {"models": []}
    return [*catalog["models"], *custom_models()]


def model_by_id(model_id: str | None):
    if not model_id:
        return None
    return next((model for model in available_models() if model["id"] == model_id), None)


def resolved_model_path(model):
    if not model:
        return None
    candidate = (MODEL_DIRECTORY / model["filename"]).resolve()
    try:
        candidate.relative_to(MODEL_DIRECTORY.resolve())
    except ValueError:
        return None
    if model.get("local"):
        return candidate if candidate.is_file() else None
    return model_path(model)


def require_model(model_id: str):
    model = model_by_id(model_id)
    if not model:
        raise ValueError("所选透明背景模型不在模型目录中")
    path = resolved_model_path(model)
    if path is None:
        raise ValueError("所选透明背景模型尚未安装")
    return model, path


def installed_models():
    return [(model, path) for model in available_models() if (path := resolved_model_path(model)) is not None]


def background_removal_status():
    catalog = model_catalog() or {"models": []}
    models = available_models()
    installed = installed_models()
    active_model = installed[0][0] if installed else None
    preferred_model = catalog["models"][0] if catalog["models"] else None
    try:
        import onnxruntime as ort

        runtime_available = bool(ort.get_available_providers())
        runtime_version = getattr(ort, "__version__", None)
    except (ImportError, OSError):
        runtime_available = False
        runtime_version = None
    return {
        "available": True,
        "mode": active_model["id"] if installed and runtime_available else "algorithm",
        "algorithm_available": True,
        "runtime_available": runtime_available,
        "runtime_version": runtime_version,
        "model_available": bool(installed),
        "model": active_model.get("id") if active_model else None,
        "revision": active_model.get("revision") if active_model else None,
        "filename": active_model.get("filename") if active_model else None,
        "size": active_model.get("size") if active_model else None,
        "preferred_model": preferred_model.get("id") if preferred_model else None,
        "preferred_available": bool(preferred_model and model_path(preferred_model)),
        "preferred_size": preferred_model.get("size") if preferred_model else None,
        "directory": str(MODEL_DIRECTORY),
        "models": [{
            "id": model["id"],
            "label": model["label"],
            "description": model.get("description", ""),
            "size": model["size"],
            "license": model.get("license"),
            "installed": resolved_model_path(model) is not None,
            "selectable": model.get("selectable", True),
            "local": bool(model.get("local")),
        } for model in models],
        "device": "cpu",
    }


def _load_session(path: Path):
    file_stat = path.stat()
    identity = (str(path), file_stat.st_size, file_stat.st_mtime_ns)
    with _session_lock:
        if identity in _sessions:
            return _sessions[identity]
        try:
            import onnxruntime as ort
        except (ImportError, OSError) as error:
            raise RuntimeError("透明背景 ONNX 运行时不可用，请重新运行环境配置器") from error
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 1)))
        options.log_severity_level = 3
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        _sessions.clear()
        _sessions[identity] = session
        return session


def _resize_for_model(image: Image.Image, edge=320):
    width, height = image.size
    scale = min(edge / width, edge / height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    left = (edge - resized_size[0]) // 2
    top = (edge - resized_size[1]) // 2
    canvas = Image.new("RGB", (edge, edge))
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + resized_size[0], top + resized_size[1])


def _onnx_alpha(image: Image.Image, path: Path, model: dict):
    session = _load_session(path)
    input_shape = session.get_inputs()[0].shape
    input_size = model["input_size"]
    if len(input_shape) == 4 and isinstance(input_shape[2], int) and input_shape[2] == input_shape[3]:
        input_size = input_shape[2]
    prepared, content_box = _resize_for_model(image.convert("RGB"), input_size)
    pixels = np.asarray(prepared, dtype=np.float32) / 255.0
    pixels = (pixels - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    input_type = session.get_inputs()[0].type
    input_dtype = np.float16 if input_type == "tensor(float16)" else np.float32
    inputs = np.transpose(pixels, (2, 0, 1))[None, ...].astype(input_dtype, copy=False)
    with _inference_lock:
        prediction = np.asarray(session.run(None, {session.get_inputs()[0].name: inputs})[0], dtype=np.float32).squeeze()
    while prediction.ndim > 2:
        prediction = prediction[0]
    if model["output"] == "logits" or (model["output"] == "auto" and (prediction.min() < 0 or prediction.max() > 1)):
        prediction = 1.0 / (1.0 + np.exp(-np.clip(prediction, -30.0, 30.0)))
    elif model["output"] in {"probability", "auto"}:
        if not np.isfinite(prediction).all():
            raise RuntimeError("透明背景模型返回了无效蒙版")
        prediction = np.clip(prediction, 0.0, 1.0)
    else:
        minimum = float(np.nanmin(prediction))
        maximum = float(np.nanmax(prediction))
        if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum - minimum < 1e-6:
            raise RuntimeError("透明背景模型返回了无效蒙版")
        prediction = np.clip((prediction - minimum) / (maximum - minimum), 0.0, 1.0)
    left, top, right, bottom = content_box
    prediction = prediction[top:bottom, left:right]
    mask = Image.fromarray(np.round(prediction * 255).astype(np.uint8), mode="L")
    return np.asarray(mask.resize(image.size, Image.Resampling.LANCZOS), dtype=np.uint8)


def _background_prototypes(rgb: np.ndarray):
    height, width, _ = rgb.shape
    edge = max(1, min(12, round(min(width, height) * 0.015)))
    border = np.concatenate((
        rgb[:edge].reshape(-1, 3),
        rgb[-edge:].reshape(-1, 3),
        rgb[edge:-edge, :edge].reshape(-1, 3),
        rgb[edge:-edge, -edge:].reshape(-1, 3),
    ))
    quantized = np.floor_divide(border, 24).astype(np.int16)
    keys, inverse, counts = np.unique(quantized, axis=0, return_inverse=True, return_counts=True)
    del keys
    order = np.argsort(counts)[::-1]
    prototypes = []
    covered = 0
    target = max(1, round(len(border) * 0.86))
    for index in order[:16]:
        members = border[inverse == index]
        prototypes.append(np.median(members, axis=0))
        covered += len(members)
        if covered >= target and len(prototypes) >= 3:
            break
    return np.asarray(prototypes, dtype=np.float32)


def _algorithm_alpha(image: Image.Image):
    from scipy.ndimage import binary_propagation, gaussian_filter

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    prototypes = _background_prototypes(rgb)
    pixels = rgb.astype(np.float32)
    minimum_squared = np.full(rgb.shape[:2], np.inf, dtype=np.float32)
    for prototype in prototypes:
        delta = pixels - prototype
        squared = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
        np.minimum(minimum_squared, squared, out=minimum_squared)
    distances = np.sqrt(minimum_squared, out=minimum_squared)

    border_distances = np.concatenate((distances[0], distances[-1], distances[:, 0], distances[:, -1]))
    high = float(np.clip(np.percentile(border_distances, 82) + 26, 30, 78))
    low = max(8.0, high * 0.42)
    candidates = distances <= high
    seeds = np.zeros(candidates.shape, dtype=bool)
    seeds[0] = candidates[0]
    seeds[-1] = candidates[-1]
    seeds[:, 0] = candidates[:, 0]
    seeds[:, -1] = candidates[:, -1]
    connected = binary_propagation(seeds, mask=candidates)

    transition = np.clip((distances - low) / max(1.0, high - low), 0.0, 1.0)
    transition = transition * transition * (3.0 - 2.0 * transition)
    alpha = np.where(connected, transition, 1.0)
    alpha = gaussian_filter(alpha.astype(np.float32), sigma=0.65)
    alpha[alpha < 0.025] = 0.0
    alpha[alpha > 0.975] = 1.0
    return np.round(np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)


def _disk(radius: int):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def _suppress_border_structures(alpha: np.ndarray):
    """Remove thin structures entering from a canvas edge without dropping detached subjects."""
    from scipy.ndimage import binary_dilation, binary_opening, find_objects, gaussian_filter, label

    hard = alpha >= 128
    if not (hard[0].any() or hard[-1].any() or hard[:, 0].any() or hard[:, -1].any()):
        return alpha
    height, width = hard.shape
    radius = max(3, min(6, round(min(width, height) / 192)))
    opened = binary_opening(hard, structure=_disk(radius))
    opened_labels, count = label(opened)
    if not count:
        return alpha

    keep = np.zeros_like(hard)
    minimum_area = max(24, round(alpha.size * 0.0005))
    large_area = alpha.size * 0.01
    for index, region in enumerate(find_objects(opened_labels), 1):
        if region is None:
            continue
        y_slice, x_slice = region
        component = opened_labels[region] == index
        area = int(component.sum())
        component_width = x_slice.stop - x_slice.start
        component_height = y_slice.stop - y_slice.start
        touches_edge = x_slice.start == 0 or y_slice.start == 0 or x_slice.stop == width or y_slice.stop == height
        aspect = max(component_width, component_height) / max(1, min(component_width, component_height))
        substantial_edge_subject = area >= large_area or (
            min(component_width, component_height) >= radius * 4 and aspect <= 6
        )
        if area >= minimum_area and (not touches_edge or substantial_edge_subject):
            keep[region] |= component
    if not keep.any():
        return alpha

    # Preserve detached, non-edge elements that may be too fine to survive opening.
    original_labels, _ = label(hard)
    for index, region in enumerate(find_objects(original_labels), 1):
        if region is None:
            continue
        y_slice, x_slice = region
        if x_slice.start == 0 or y_slice.start == 0 or x_slice.stop == width or y_slice.stop == height:
            continue
        component = original_labels[region] == index
        if int(component.sum()) >= 8:
            keep[region] |= component

    allowed = binary_dilation(keep, structure=_disk(radius * 2))
    gate = gaussian_filter(allowed.astype(np.float32), sigma=max(0.8, radius / 2))
    cleaned = np.round(alpha.astype(np.float32) * np.clip(gate, 0.0, 1.0)).astype(np.uint8)
    original_visible = np.count_nonzero(hard)
    if np.count_nonzero(cleaned >= 128) < original_visible * 0.25:
        return alpha
    cleaned[cleaned <= 2] = 0
    return cleaned


def _validate_alpha(alpha: np.ndarray):
    if alpha.ndim != 2 or alpha.size == 0:
        raise RuntimeError("透明背景蒙版尺寸无效")
    transparent_ratio = float(np.count_nonzero(alpha <= 8) / alpha.size)
    opaque_ratio = float(np.count_nonzero(alpha >= 247) / alpha.size)
    visible_ratio = float(np.count_nonzero(alpha >= 128) / alpha.size)
    if transparent_ratio >= 0.995:
        raise RuntimeError("透明背景蒙版几乎完全透明，已停止保存以保护主体")
    if transparent_ratio <= 0.001 and opaque_ratio >= 0.995:
        raise RuntimeError("透明背景蒙版几乎完全不透明，请调整构图或关闭特殊标签")
    if transparent_ratio <= 0.001:
        raise RuntimeError("透明背景蒙版没有识别到可信透明区域，已停止保存半透明背景")
    if visible_ratio <= 0.002:
        raise RuntimeError("透明背景模型未识别到可信主体，请增加生成步数或调整主体构图")
    if opaque_ratio <= 0.001:
        raise RuntimeError("透明背景蒙版没有识别到可信不透明前景，已停止保存低对比度蒙版")
    return transparent_ratio, opaque_ratio, visible_ratio


def extract_foreground(image: Image.Image, model_id: str | None = None):
    """Return a same-size RGBA image and extraction diagnostics."""
    started_at = time.perf_counter()
    installed = installed_models()
    selected = model_by_id(model_id)
    if model_id and not selected:
        raise RuntimeError("所选透明背景模型已不在模型目录中")
    if selected:
        try:
            selected, selected_path = require_model(model_id)
        except ValueError as error:
            raise RuntimeError(f"{error}，请先完成下载或放入模型目录") from error
        installed = [
            (selected, selected_path),
            *[(model, path) for model, path in installed if model["id"] != selected["id"] and model.get("selectable") is False],
        ]
    status = background_removal_status()
    method = "algorithm"
    warning = None
    used_model = None
    errors = []
    if installed and status["runtime_available"]:
        for model, path in installed:
            try:
                alpha = _onnx_alpha(image, path, model)
                if model["id"].startswith(("birefnet", "bria-rmbg")):
                    alpha = _suppress_border_structures(alpha)
                _validate_alpha(alpha)
                method = model["id"]
                used_model = model
                if errors and selected and model["id"] != selected["id"]:
                    warning = f"所选模型运行失败，已使用内置轻量模型回退：{'; '.join(errors)}"
                break
            except Exception as error:
                errors.append(f"{model['id']}: {type(error).__name__}")
        else:
            alpha = _algorithm_alpha(image)
            warning = f"ONNX 精细抠图失败，已使用纯算法回退：{'; '.join(errors)}"
    else:
        alpha = _algorithm_alpha(image)
        if not installed:
            warning = "未安装轻量透明背景模型，已使用纯算法抠图"
        elif not status["runtime_available"]:
            warning = "ONNX 运行时不可用，已使用纯算法抠图"
    transparent_ratio, opaque_ratio, visible_ratio = _validate_alpha(alpha)
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    diagnostics = {
        "status": "complete",
        "method": method,
        "model": used_model["id"] if used_model else None,
        "revision": used_model["revision"] if used_model else None,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "transparent_ratio": round(transparent_ratio, 4),
        "opaque_ratio": round(opaque_ratio, 4),
        "visible_ratio": round(visible_ratio, 4),
    }
    if model_id:
        diagnostics["requested_model"] = model_id
    if warning:
        diagnostics["warning"] = warning
    return rgba, diagnostics


def clear_background_removal_session():
    with _session_lock:
        _sessions.clear()
