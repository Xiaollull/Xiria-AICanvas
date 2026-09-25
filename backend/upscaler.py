import gc
import hashlib
import json
import math
import re
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file

try:
    from .resize import DEFAULT_RESIZE_METHOD, resize_image, scaled_size
except ImportError:
    from resize import DEFAULT_RESIZE_METHOD, resize_image, scaled_size


UPSCALER_EXTENSIONS = {".pth", ".pt", ".ckpt", ".safetensors"}
_model_metadata_cache = {}
_extra_arches_registered = False
_builtin_hash_cache = {}
_builtin_catalog = None

_BUILTIN_UPSCALER_FAMILY = "hires-upscalers"
_BUILTIN_UPSCALER_MANIFEST = Path(__file__).resolve().parents[1] / "models" / "recommended-models.json"
_BUILTIN_ARTIFACT_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,99}\Z")
_BUILTIN_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def _read_builtin_catalog():
    """Project the trusted JSON manifest into immutable, download-safe upscaler facts.

    The inference service never receives a provider URL from this catalogue.  The public status only
    exposes the family/artifact identifiers accepted by the existing local download endpoint.
    """
    try:
        source = json.loads(_BUILTIN_UPSCALER_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ()
    if not isinstance(source, dict) or source.get("schema") != 1:
        return ()
    families = source.get("staticFamilies")
    artifacts = source.get("artifacts")
    if not isinstance(families, list) or not isinstance(artifacts, list):
        return ()
    family = next((item for item in families if isinstance(item, dict) and item.get("id") == _BUILTIN_UPSCALER_FAMILY), None)
    artifact_ids = family.get("models") if isinstance(family, dict) else None
    if not isinstance(artifact_ids, list) or not artifact_ids:
        return ()
    artifact_map = {}
    for artifact in artifacts:
        artifact_id = artifact.get("id") if isinstance(artifact, dict) else None
        if not isinstance(artifact_id, str) or artifact_id in artifact_map:
            return ()
        artifact_map[artifact_id] = artifact
    catalog = []
    seen_ids = set()
    seen_filenames = set()
    for artifact_id in artifact_ids:
        artifact = artifact_map.get(artifact_id)
        if (
            not isinstance(artifact_id, str)
            or artifact_id in seen_ids
            or not _BUILTIN_ARTIFACT_ID.fullmatch(artifact_id)
            or not isinstance(artifact, dict)
        ):
            return ()
        filename = artifact.get("filename")
        sha256 = artifact.get("sha256")
        size = artifact.get("size")
        scale = artifact.get("scale")
        compatible = artifact.get("compatible")
        if (
            artifact.get("role") != "upscaler"
            or not isinstance(artifact.get("label"), str) or not artifact["label"]
            or not isinstance(filename, str) or Path(filename).name != filename or Path(filename).suffix.lower() not in UPSCALER_EXTENSIONS
            or not isinstance(sha256, str) or not _BUILTIN_SHA256.fullmatch(sha256.lower())
            or not isinstance(size, int) or size < 1
            or not isinstance(scale, int) or scale < 1 or scale > 4
            or not isinstance(compatible, bool)
        ):
            return ()
        filename_key = filename.casefold()
        if filename_key in seen_filenames:
            return ()
        reason = artifact.get("reason") if isinstance(artifact.get("reason"), str) else ""
        if not compatible and not reason:
            return ()
        catalog.append(MappingProxyType({
            "id": artifact_id,
            "label": artifact["label"],
            "filename": filename,
            "size": size,
            "sha256": sha256.lower(),
            "scale": scale,
            "compatible": compatible,
            "reason": reason,
            "download": MappingProxyType({"family_id": _BUILTIN_UPSCALER_FAMILY, "artifact_id": artifact_id}),
        }))
        seen_ids.add(artifact_id)
        seen_filenames.add(filename_key)
    return tuple(catalog)


def builtin_catalog():
    global _builtin_catalog
    if _builtin_catalog is None:
        _builtin_catalog = _read_builtin_catalog()
    return _builtin_catalog


def _file_sha256(path: Path):
    stat = path.stat()
    key = str(path.resolve()).lower()
    cached = _builtin_hash_cache.get(key)
    if cached and cached["size"] == stat.st_size and cached["mtime_ns"] == stat.st_mtime_ns:
        return cached["sha256"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    sha256 = digest.hexdigest()
    _builtin_hash_cache[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256}
    return sha256


def _builtin_status(directory: Path, models):
    """Overlay actual local files onto the immutable directory without trusting their names."""
    root = directory.resolve()
    local_by_name = {}
    for model in models:
        name = model.get("name")
        if isinstance(name, str):
            local_by_name.setdefault(name.casefold(), model)
    builtins = []
    for builtin in builtin_catalog():
        local = local_by_name.get(builtin["filename"].casefold())
        installed = False
        compatible = builtin["compatible"]
        local_id = local.get("id") if isinstance(local, dict) and isinstance(local.get("id"), str) else None
        reason = builtin["reason"]
        if local is not None:
            try:
                path = (root / local_id).resolve() if local_id else None
                if path is None or root not in path.parents or not path.is_file():
                    raise OSError("local model path is unavailable")
                if local.get("size") != builtin["size"]:
                    compatible = False
                    reason = "；".join(filter(None, (reason, "同名本地文件大小与内置目录不一致")))
                elif _file_sha256(path) != builtin["sha256"]:
                    compatible = False
                    reason = "；".join(filter(None, (reason, "同名本地文件未通过内置目录 SHA-256 校验")))
                else:
                    installed = True
                    if not local.get("compatible") and builtin["compatible"]:
                        compatible = False
                        reason = "；".join(filter(None, (reason, local.get("error") or "同名本地文件的超分架构不兼容")))
            except OSError:
                installed = False
                compatible = False
                reason = "；".join(filter(None, (reason, "同名本地文件无法读取或已移除")))
        builtins.append({
            **{key: value for key, value in builtin.items() if key != "download"},
            "download": dict(builtin["download"]),
            "installed": installed,
            "compatible": compatible,
            "local_id": local_id,
            "reason": reason,
        })
    return builtins


def runtime_available():
    try:
        import spandrel  # noqa: F401
    except ImportError:
        return False
    return True


def _register_extra_arches():
    global _extra_arches_registered
    if _extra_arches_registered:
        return
    import spandrel
    try:
        import spandrel_extra_arches

        spandrel.MAIN_REGISTRY.add(*spandrel_extra_arches.EXTRA_REGISTRY)
    except ImportError:
        pass
    _extra_arches_registered = True


def _state_dict(path: Path):
    if path.suffix.lower() == ".safetensors":
        return load_safetensors_file(str(path), device="cpu")
    # weights_only prevents .pth/.pt files from executing pickled application code.
    return torch.load(str(path), map_location="cpu", weights_only=True)


def load_descriptor(path: Path, device="cpu"):
    import spandrel

    _register_extra_arches()
    descriptor = spandrel.ModelLoader(device="cpu").load_from_state_dict(_state_dict(path))
    scale = int(getattr(descriptor, "scale", 1) or 1)
    if str(getattr(descriptor, "purpose", "")).upper() != "SR" or scale <= 1:
        raise ValueError("The selected file is not a super-resolution image model")
    if int(getattr(descriptor, "input_channels", 0) or 0) != 3 or int(getattr(descriptor, "output_channels", 0) or 0) != 3:
        raise ValueError("Only three-channel RGB super-resolution models are supported")
    dtype = torch.float16 if str(device).startswith("cuda") and descriptor.supports_half else torch.float32
    descriptor.model.eval().to(device=device, dtype=dtype)
    return descriptor


def _model_metadata(path: Path, root: Path):
    stat = path.stat()
    key = str(path.resolve()).lower()
    cached = _model_metadata_cache.get(key)
    if cached and cached["size"] == stat.st_size and cached["mtime_ns"] == stat.st_mtime_ns:
        return dict(cached["metadata"])
    metadata = {
        "id": path.resolve().relative_to(root.resolve()).as_posix(),
        "name": path.name,
        "label": path.stem,
        "size": stat.st_size,
        "compatible": False,
        "architecture": "",
        "scale": 0,
        "error": "",
    }
    if not runtime_available():
        metadata["error"] = "Spandrel runtime is not installed"
    else:
        descriptor = None
        try:
            descriptor = load_descriptor(path)
            metadata.update({
                "compatible": True,
                "architecture": str(descriptor.architecture.name),
                "scale": int(descriptor.scale),
            })
        except Exception as error:
            metadata["error"] = str(error).splitlines()[0][:240]
        finally:
            if descriptor is not None:
                del descriptor
            gc.collect()
    _model_metadata_cache[key] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "metadata": dict(metadata),
    }
    return metadata


def discover_models(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    root = directory.resolve()
    models = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file() or path.suffix.lower() not in UPSCALER_EXTENSIONS:
            continue
        try:
            path.resolve().relative_to(root)
        except ValueError:
            continue
        models.append(_model_metadata(path, root))
    return models


def resolve_model(directory: Path, reference: str):
    if not reference:
        raise ValueError("An upscaler model reference is required")
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError("Upscaler model path must be relative")
    root = directory.resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents or path.suffix.lower() not in UPSCALER_EXTENSIONS:
        raise ValueError("Upscaler model is outside the configured directory")
    if not path.is_file():
        raise ValueError("Upscaler model does not exist")
    metadata = _model_metadata(path, root)
    if not metadata["compatible"]:
        raise ValueError(f"Upscaler model is incompatible: {metadata['error']}")
    return path, metadata


def status(directory: Path):
    models = discover_models(directory)
    return {
        "runtime_available": runtime_available(),
        "available": runtime_available() and any(model["compatible"] for model in models),
        "directory": str(directory),
        "models": models,
        "builtins": _builtin_status(directory, models),
    }


def target_size(image: Image.Image, scale: float):
    """`round(edge × scale)`, the size the chain asked for.

    This used to round every edge up to a multiple of 64, which made the requested scale an
    approximation: 1088 × 1.5 is 1632, and the snap turned it into 1664. A resample unit's target is
    exact, and whatever grid an engine needs is applied by the caller that knows which engine is
    mounted — not here, where only the picture is in hand.
    """
    if not math.isfinite(scale) or scale < 1 or scale > 4:
        raise ValueError("Resample scale must be between 1.0 and 4.0")
    return scaled_size(image, scale)


def _pil_to_bgr_tensor(image: Image.Image):
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)[:, :, ::-1].copy() / 255.0
    return torch.from_numpy(np.moveaxis(pixels, 2, 0)).unsqueeze(0)


def _tensor_to_pil(tensor: torch.Tensor):
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Upscaler returned an invalid batch")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Upscaler returned an invalid image tensor")
    pixels = tensor.detach().float().cpu().clamp_(0, 1).numpy()
    pixels = np.moveaxis(pixels, 0, 2)[:, :, ::-1]
    return Image.fromarray(np.rint(pixels * 255).astype(np.uint8), "RGB")


def _upscale_patch(descriptor, patch: Image.Image):
    parameter = next(descriptor.model.parameters(), None)
    tensor = _pil_to_bgr_tensor(patch)
    if parameter is not None:
        tensor = tensor.to(device=parameter.device, dtype=parameter.dtype)
    with torch.inference_mode():
        return _tensor_to_pil(descriptor(tensor))


def _tile_origins(length: int, tile_size: int, overlap: int):
    extent = min(length, tile_size)
    if length <= extent:
        return [0]
    stride = max(1, extent - min(overlap, extent - 1))
    count = math.ceil((length - extent) / stride) + 1
    distance = length - extent
    return sorted({round(index * distance / (count - 1)) for index in range(count)})


def _blend_mask(width: int, height: int, horizontal: bool):
    length = width if horizontal else height
    ramp = np.linspace(0, 255, length, endpoint=True, dtype=np.uint8)
    pixels = np.repeat(ramp.reshape(1, width), height, axis=0) if horizontal else np.repeat(ramp.reshape(height, 1), width, axis=1)
    return Image.fromarray(pixels, "L")


def _upscale_once(image: Image.Image, descriptor, tile_size: int, overlap: int, progress, checkpoint, completed: int, total: int):
    native_scale = int(descriptor.scale)
    width, height = image.size
    output = Image.new("RGB", (width * native_scale, height * native_scale))
    tile_width = min(width, tile_size)
    tile_height = min(height, tile_size)
    left_positions = _tile_origins(width, tile_width, overlap)
    top_positions = _tile_origins(height, tile_height, overlap)
    previous_row_bottom = 0
    for top in top_positions:
        row = Image.new("RGB", (width * native_scale, tile_height * native_scale))
        previous_tile_right = 0
        for left in left_positions:
            patch = image.crop((left, top, left + tile_width, top + tile_height))
            if checkpoint:
                checkpoint()
            enlarged = _upscale_patch(descriptor, patch)
            output_left = left * native_scale
            horizontal_overlap = max(0, previous_tile_right - output_left)
            if horizontal_overlap:
                blend = enlarged.crop((0, 0, horizontal_overlap, enlarged.height))
                row.paste(blend, (output_left, 0), _blend_mask(horizontal_overlap, enlarged.height, True))
                row.paste(enlarged.crop((horizontal_overlap, 0, enlarged.width, enlarged.height)), (output_left + horizontal_overlap, 0))
            else:
                row.paste(enlarged, (output_left, 0))
            previous_tile_right = output_left + enlarged.width
            completed += 1
            if progress:
                progress(completed, total)
        output_top = top * native_scale
        vertical_overlap = max(0, previous_row_bottom - output_top)
        if vertical_overlap:
            blend = row.crop((0, 0, row.width, vertical_overlap))
            output.paste(blend, (0, output_top), _blend_mask(row.width, vertical_overlap, False))
            output.paste(row.crop((0, vertical_overlap, row.width, row.height)), (0, output_top + vertical_overlap))
        else:
            output.paste(row, (0, output_top))
        previous_row_bottom = output_top + row.height
    return output, completed


def upscale_image(image: Image.Image, directory: Path, reference: str, scale: float, tile_size=192, overlap=16, progress=None, device="cpu", checkpoint=None, method=DEFAULT_RESIZE_METHOD, destination=None):
    """Run the model until the picture is at least `destination`, then resize onto it with `method`.

    `destination` defaults to the unit's exact target. A caller that has to land on an engine's
    latent grid passes the aligned size instead, so the model pass and the resize agree on one
    destination rather than resizing twice.
    """
    path, metadata = resolve_model(directory, reference)
    tile_size = max(32, int(tile_size))
    overlap = max(0, min(int(overlap), tile_size // 2))
    destination = tuple(destination) if destination is not None else target_size(image, scale)
    native_scale = int(metadata["scale"])
    planned_sizes = []
    planned = image.size
    while planned[0] < destination[0] or planned[1] < destination[1]:
        planned_sizes.append(planned)
        planned = (planned[0] * native_scale, planned[1] * native_scale)
    total_tiles = sum(
        len(_tile_origins(width, min(width, tile_size), overlap))
        * len(_tile_origins(height, min(height, tile_size), overlap))
        for width, height in planned_sizes
    )
    current = image.convert("RGB")
    completed = 0
    descriptor = None
    try:
        if not planned_sizes:
            # The model has nothing to add — the picture already covers the destination — but the
            # destination is still a promise: an engine grid can move it a few pixels off the source
            # even at scale 1.0, and the redraw that follows would then be handed the wrong canvas.
            if current.size != destination:
                if checkpoint:
                    checkpoint()
                current = resize_image(current, destination[0], destination[1], method)
            return current, {
                "model": reference,
                "model_name": path.name,
                "architecture": metadata["architecture"],
                "native_scale": native_scale,
                "scale": scale,
                "source_size": list(image.size),
                "target_size": list(destination),
                "resize_algorithm": method,
                "tile_size": tile_size,
                "tile_overlap": overlap,
                "upscaler_applied": False,
                "device": "cpu",
            }
        if checkpoint:
            checkpoint()
        descriptor = load_descriptor(path, device=device)
        for _ in planned_sizes:
            current, completed = _upscale_once(current, descriptor, tile_size, overlap, progress, checkpoint, completed, total_tiles)
        if current.size != destination:
            if checkpoint:
                checkpoint()
            current = resize_image(current, destination[0], destination[1], method)
        return current, {
            "model": reference,
            "model_name": path.name,
            "architecture": metadata["architecture"],
            "native_scale": native_scale,
            "scale": scale,
            "source_size": list(image.size),
            "target_size": list(destination),
            "resize_algorithm": method,
            "tile_size": tile_size,
            "tile_overlap": overlap,
            "upscaler_applied": True,
            "device": "cuda" if str(device).startswith("cuda") else "cpu",
        }
    finally:
        if descriptor is not None:
            del descriptor
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
