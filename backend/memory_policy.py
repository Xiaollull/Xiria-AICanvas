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
        # Krea 2 is 6144 wide over the same 4096-token sequence, and a step runs two branches where
        # FLUX.2 runs one — which reaches this function through `batch_factor`, so the slope stays
        # per-branch rather than being doubled a second time.
        #
        # The slope is measured rather than inherited. A resident transformer's peak allocation was
        # recorded on a 24 GB card at four canvases, with a 256-token conditioning sequence:
        #
        #     1024x1024 (1.0 MP)  0.69 GiB      2048x2048 (4.0 MP)  2.42 GiB
        #     1536x1536 (2.3 MP)  1.39 GiB      2048x2944 (5.8 MP)  3.46 GiB
        #
        # which is 0.11 + 0.58/MP. The figures below keep about 60% headroom over that for the
        # sampler's own latents, allocator fragmentation and the refinement stages. The slope this
        # replaced was 1.7/MP, three times the truth, and it was enough on its own to send a
        # 2048x2944 Hires pass — 15.7 GiB of a 23.5 GiB card, measured — down the block-streaming
        # path, where it took 690 seconds instead of running resident.
        estimate_gb = 0.35 + 0.90 * megapixels * batch_factor
    elif family == "moss":
        # MOSS-Image is 3072 wide like FLUX.1, but its single stream carries the whole conditioning
        # (up to 4096 Qwen3.5 tokens) beside the 4096 image tokens of a 1 MP canvas, and an INT8
        # linear's int32 accumulators are four bytes per output. The figure is a provisional upper
        # bound until the xiria-39 measurement replaces it; the branches run one after the other,
        # which reaches this function through `batch_factor` as for Krea 2.
        estimate_gb = 0.80 + 1.00 * megapixels * batch_factor
    elif family == "anima":
        # Anima was the last native engine still sized by the generic fallback below, which was
        # written for SD-era UNets. Its DiT is 2048 wide over 28 blocks and patchifies to (W/16 x
        # H/16), so what scales with the canvas is far cheaper than a UNet's feature pyramid, and
        # the fallback's 0.40/MP over-budgets it. That matters on a card where the transformer is
        # already half the device: a 1088x1408 base with a full-frame 2x resample unit is admitted
        # against the 2176x2816 redraw, and 0.40/MP put that at 2.99 GiB, which alone lifted
        # NORMAL past an 8 GB card's total and sent the whole job — the cheap base pass included —
        # onto block streaming.
        #
        # Measured on an RTX 4060 Laptop (8 GB) with anima_baseV10 BF16, at a 1120x1440 tile:
        #
        #     sampling, beside the resident DiT   0.24 GiB      VAE decode, DiT evicted  1.29 GiB
        #     VAE encode, beside the resident DiT 0.66 GiB
        #
        # The slope follows the sampling figure (0.156/MP) with headroom, because that is the only
        # part that grows with the canvas — the VAE runs tiled, so its working set is in the base.
        # The base also stays above the standalone decode peak, which never has to share the card
        # with the transformer but is the largest single number of the three. One canvas, one card:
        # this is an anchor, not the four-point fit `krea2` above got.
        estimate_gb = 0.90 + 0.26 * megapixels * batch_factor
    else:
        estimate_gb = 0.65 + 0.40 * megapixels * batch_factor
    return max(int(0.8 * GIB), int(estimate_gb * GIB))


def estimate_vae_decode_bytes(width, height, images_per_batch=1, sliced=True):
    """Peak allocation of one whole-image SD/SDXL autoencoder decode.

    The decoder works at full resolution and SDXL's is upcast to float32 (`force_upcast`), so this
    is the largest single allocation in a job — larger than the sampler's, and it grows with the
    canvas far faster. Measured on an RTX 4060 Laptop (8188 MiB) with `waiIllustriousSDXL_v170`'s
    VAE, as `torch.cuda.max_memory_allocated` around a batch-of-one decode:

        768x1280  (0.94 MP)   3.84 GiB whole-image    3.14 GiB tiled
        1152x1920 (2.11 MP)   8.24 GiB whole-image    4.09 GiB tiled

    which is 0.32 + 3.76/MP. The point of having it is that the second of those does not fit beside
    a 600 MiB platform reserve on an 8 GB card, and Windows answers an allocation it cannot serve by
    spilling into host memory rather than raising: the decode still returns a correct picture, and
    takes a hundred seconds instead of five. An OOM fallback never fires, so the size has to be
    asked about in advance.

    Slicing splits the batch, not the canvas, so it caps the peak at one image; tiling is what caps
    the canvas. `sliced` says which of the two the caller has enabled.
    """
    megapixels = max(width * height, 64 * 64) / (1024 * 1024)
    copies = 1 if sliced else max(1, int(images_per_batch))
    return int((0.32 + 3.76 * megapixels) * copies * GIB)


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
        # MOSS's largest component is the Qwen3.5-9B encoder, not the 5B DiT.
        else 0.60 if family == "moss"
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
    resident_first=False,
):
    """Pick the offload rung this workload runs on.

    `resident_first` changes what NORMAL has to prove, and only that. The default rule asks for
    `largest component x 1.10 + inference + reserve` to be *free before the run starts*, which is an
    "everything must fit up front" gate: on a card where the largest component is already most of
    the device it can exceed the whole card, and the request is then silently downgraded to the
    per-submodule path. Component offload never needs that — `enable_model_cpu_offload` holds one
    whole component at a time and the rest sit on the host — so the resident-first rule asks the two
    questions that path actually answers: does the largest component plus its activations plus the
    platform reserve fit in the *device*, and is enough free right now to move that component in
    while another process may own part of the card. `high` and `ultra_low` are untouched.
    """
    requested_mode = normalize_memory_mode(requested_mode)
    if requested_mode == "sdxl_balanced":
        requested_mode = "normal_vram"
    minimum_inference_bytes = int(0.8 * GIB) + reserved_bytes
    maximum_weight_bytes = max(0, int(total_bytes * 0.88) - minimum_inference_bytes)
    high_required_bytes = int(runtime_weight_bytes * 1.08) + inference_bytes + reserved_bytes
    if resident_first:
        # The reserve is already inside the sum, so the bound is the device total rather than a
        # fraction of it; the free-memory guard only refuses a move that cannot land.
        normal_required_bytes = largest_component_bytes + inference_bytes + reserved_bytes
        normal_residency_bytes = largest_component_bytes + reserved_bytes
        normal_available = normal_required_bytes <= total_bytes and free_bytes >= normal_residency_bytes
    else:
        normal_required_bytes = int(largest_component_bytes * 1.10) + inference_bytes + reserved_bytes
        normal_residency_bytes = normal_required_bytes
        normal_available = free_bytes >= normal_required_bytes
    high_available = runtime_weight_bytes <= maximum_weight_bytes and free_bytes >= high_required_bytes

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
        "normal_residency_bytes": normal_residency_bytes,
        "normal_admission_rule": "resident_first" if resident_first else "workspace_upfront",
    }
