import gc
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file


UPSCALER_EXTENSIONS = {".pth", ".pt", ".ckpt", ".safetensors"}
_model_metadata_cache = {}
_extra_arches_registered = False


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
        raise ValueError("Hires.fix requires an upscaler model")
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
    }


def target_size(image: Image.Image, scale: float):
    if not math.isfinite(scale) or scale < 1 or scale > 4:
        raise ValueError("Hires.fix scale must be between 1.0 and 4.0")
    return tuple(max(64, int(math.ceil(dimension * scale / 64)) * 64) for dimension in image.size)


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


def upscale_image(image: Image.Image, directory: Path, reference: str, scale: float, tile_size=192, overlap=16, progress=None, device="cpu", checkpoint=None):
    path, metadata = resolve_model(directory, reference)
    tile_size = max(32, int(tile_size))
    overlap = max(0, min(int(overlap), tile_size // 2))
    destination = target_size(image, scale)
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
            return current, {
                "model": reference,
                "model_name": path.name,
                "architecture": metadata["architecture"],
                "native_scale": native_scale,
                "scale": scale,
                "source_size": list(image.size),
                "target_size": list(destination),
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
            current = current.resize(destination, Image.Resampling.LANCZOS)
        return current, {
            "model": reference,
            "model_name": path.name,
            "architecture": metadata["architecture"],
            "native_scale": native_scale,
            "scale": scale,
            "source_size": list(image.size),
            "target_size": list(destination),
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
