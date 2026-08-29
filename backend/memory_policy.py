import os


MIB = 1024**2
GIB = 1024**3

MODE_ALIASES = {
    "auto": "auto",
    "high": "high_vram",
    "high_vram": "high_vram",
    "balanced": "normal_vram",
    "normal": "normal_vram",
    "normal_vram": "normal_vram",
    "sdxl_balanced": "sdxl_balanced",
    "conservative": "low_vram",
    "low": "low_vram",
    "low_vram": "low_vram",
    "no_vram": "low_vram",
    "ultra_low_vram": "ultra_low_vram",
}


def normalize_memory_mode(value):
    mode = MODE_ALIASES.get((value or "auto").strip().lower())
    if mode is None:
        choices = ", ".join(sorted(MODE_ALIASES))
        raise ValueError(f"XIRAI_MEMORY_MODE must be one of: {choices}")
    return mode


def default_reserved_vram_bytes(total_bytes, platform_name=None):
    platform_name = platform_name or os.name
    reserved = 600 * MIB if platform_name == "nt" else 400 * MIB
    if platform_name == "nt" and total_bytes > 15 * GIB:
        reserved += 100 * MIB
    return reserved


def reserved_vram_bytes(total_bytes, platform_name=None, allow_shared_memory=True):
    base = default_reserved_vram_bytes(total_bytes, platform_name)
    if allow_shared_memory:
        return base
    return min(2 * GIB, max(base + 512 * MIB, int(total_bytes * 0.12)))


def vram_limit_bounds(total_bytes, platform_name=None, allow_shared_memory=True):
    """Return a deliberately modest slider floor and a driver-safe physical ceiling."""
    total_bytes = max(0, int(total_bytes))
    reserve = reserved_vram_bytes(total_bytes, platform_name, allow_shared_memory)
    maximum = max(0, total_bytes - reserve)
    minimum = min(maximum, max(512 * MIB, min(4 * GIB, int(total_bytes * 0.25))))
    return minimum, maximum, reserve


def effective_vram_limit_bytes(total_bytes, configured_gb=0.0, platform_name=None, allow_shared_memory=True):
    """Resolve an optional user wall without ever exceeding the device-safe ceiling."""
    total_bytes = max(0, int(total_bytes))
    _minimum, maximum, _reserve = vram_limit_bounds(total_bytes, platform_name, allow_shared_memory)
    try:
        configured = float(configured_gb or 0.0)
    except (TypeError, ValueError):
        configured = 0.0
    if configured <= 0.0:
        return total_bytes
    minimum, _maximum, _reserve = vram_limit_bounds(total_bytes, platform_name, allow_shared_memory)
    return min(total_bytes, maximum, max(minimum, int(configured * GIB)))


def estimate_inference_bytes(family, width, height, guidance_scale=7.5, images_per_batch=1, guidance_copies=None):
    megapixels = max(width * height, 64 * 64) / (1024 * 1024)
    copies = max(1, int(guidance_copies if guidance_copies is not None else 2 if guidance_scale > 1 else 1))
    batch_factor = copies * max(1, images_per_batch)
    if family == "sdxl":
        estimate_gb = 0.75 + 0.55 * megapixels * batch_factor
    elif family == "flux":
        # FLUX.1 packs a 1 MP canvas into a 4096-token sequence through a 3072-wide, 57-block DiT,
        # so its per-megapixel activation cost is well above a UNet's. The estimate is deliberately
        # generous: over-estimating moves a run onto block-level offload, which is slower but runs,
        # while under-estimating admits a resident pass that cannot fit.
        estimate_gb = 0.9 + 0.85 * megapixels * batch_factor
    elif family == "flux2":
        # FLUX.2 packs a 1 MP canvas into the same 4096-token sequence, but through a 6144-wide
        # DiT — twice FLUX.1's hidden size — so every activation along that sequence is twice as
        # wide. The slope doubles with it, and stays deliberately generous for the same reason.
        estimate_gb = 1.1 + 1.7 * megapixels * batch_factor
    elif family == "krea2":
        # Krea 2 is 6144 wide over the same 4096-token sequence, so its per-branch activation cost
        # tracks FLUX.2's. What differs is that a Krea 2 step runs two branches where FLUX.2 runs
        # one — and that already reaches this function through `batch_factor`, so the slope stays
        # per-branch rather than being doubled a second time. The text sequence adds a little on
        # top: twelve tapped hidden states are refined before the streams are concatenated.
        estimate_gb = 1.2 + 1.7 * megapixels * batch_factor
    else:
        estimate_gb = 0.65 + 0.40 * megapixels * batch_factor
    return max(int(0.8 * GIB), int(estimate_gb * GIB))


def estimate_largest_component_bytes(runtime_weight_bytes, family):
    # Flux's transformer is roughly 70% of a four-component mount, and FLUX.2's roughly 60% of a
    # three-component one because its language-model encoder is so much larger than a text tower.
    # Krea 2 sits between them: a 6144-wide, 28-block DiT beside a 4B-parameter encoder.
    # The native loaders normally pass the measured component size instead, and these ratios are
    # only the pre-load fallback.
    ratio = (
        0.74 if family == "sdxl"
        else 0.70 if family == "flux"
        else 0.60 if family == "flux2"
        else 0.75 if family == "krea2"
        else 0.82
    )
    return min(runtime_weight_bytes, max(GIB, int(runtime_weight_bytes * ratio)))


def error_looks_like_oom(error):
    return getattr(error, "error_code", None) == 2 or "out of memory" in str(error).lower()


def select_memory_policy(
    requested_mode,
    *,
    total_bytes,
    free_bytes,
    runtime_weight_bytes,
    inference_bytes,
    largest_component_bytes,
    reserved_bytes,
):
    requested_mode = normalize_memory_mode(requested_mode)
    if requested_mode == "sdxl_balanced":
        requested_mode = "normal_vram"
    minimum_inference_bytes = int(0.8 * GIB) + reserved_bytes
    maximum_weight_bytes = max(0, int(total_bytes * 0.88) - minimum_inference_bytes)
    high_required_bytes = int(runtime_weight_bytes * 1.08) + inference_bytes + reserved_bytes
    normal_required_bytes = int(largest_component_bytes * 1.10) + inference_bytes + reserved_bytes
    high_available = runtime_weight_bytes <= maximum_weight_bytes and free_bytes >= high_required_bytes
    normal_available = free_bytes >= normal_required_bytes

    automatic_mode = "high_vram" if high_available else "normal_vram" if normal_available else "low_vram"
    if requested_mode == "auto":
        mode = automatic_mode
    elif requested_mode == "high_vram":
        mode = "high_vram" if high_available else "normal_vram" if normal_available else "low_vram"
    elif requested_mode == "normal_vram":
        mode = "normal_vram" if normal_available else "low_vram"
    elif requested_mode == "low_vram":
        mode = "low_vram"
    else:
        mode = "ultra_low_vram"

    return {
        "mode": mode,
        "requested_mode": requested_mode,
        "automatic_mode": automatic_mode,
        "high_available": high_available,
        "normal_available": normal_available,
        "minimum_inference_bytes": minimum_inference_bytes,
        "maximum_weight_bytes": maximum_weight_bytes,
        "high_required_bytes": high_required_bytes,
        "normal_required_bytes": normal_required_bytes,
    }
