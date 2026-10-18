"""MOSS-Image dense text-to-image DiT, ported term-for-term from the official implementation.

Reference: ``OpenMOSS-Team/MOSS-Image-Base@5f40f07 pipeline_moss_image.py::MOSSImageTransformer`` (the
same mathematics as the official ComfyUI plugin ``OpenMOSS/MOSS-Image@9d799c1
comfyui/moss-image/model.py``), Apache-2.0. Modifications: canonical fused ``proj.qkv`` (see below),
an explicit compute dtype instead of reading it off a weight that may be stored quantised, a
``skip_layers`` argument for negative-branch skipping, and Diffusers' ``ModelMixin`` as the base so
the server's group-offload machinery applies.

Architecture in one paragraph: 128-channel stride-16 latents (FLUX.2 VAE, 2x2-packed) are embedded
by one linear, the Qwen3.5-9B last hidden state (4096 wide) is projected and refined by two text
blocks, and the image-then-text sequence runs through 42 single-stream blocks with a *shared*
AdaLN modulation (one ``shared_block_modulation`` for every block), per-head QK RMSNorm, four-axis
split-half RoPE over ``(r, f, h, w)`` (text at ``(0, i, 0, 0)``, image at ``(0, L_text, h - (H-1)/2,
w - (W-1)/2)``), and the soft-capped SiTU gate ``4 tanh(g/4) sigmoid(g) * 25 tanh(h/25)``. The
timestep sinusoid is computed in float64 and fed as float32; RoPE bands are a CPU-float32 constant
cast to the activation dtype. Each sample runs at its true text length, so padding cannot reach
attention.

Layouts: Comfy-Org's repackages fuse each block's query/key/value into ``layers.N.proj.qkv``; the HF
original and the official plugin keep ``proj.q/k/v``. This module's canonical layout is the fused
one (one GEMM and, for INT8, one activation quantisation). HF files are fused at load by exact row
concatenation; the plugin's split naming is accepted only through that explicit conversion.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

MOSS_DIT_ARCHITECTURE = "moss_image_dit"
MOSS_TEXT_BLOCKS = 2
MOSS_EXPECTED_TEXT_DIM = 4096
MOSS_EXPECTED_LATENT_CHANNELS = 128
MOSS_ROPE_BASE = 10000.0
MOSS_FREQ_DIM = 256
MOSS_LATENT_STRIDE = 16
MOSS_PIXEL_ALIGNMENT = 16
MOSS_MAX_EDGE = 4096

# Explicit prefixes a MOSS DiT may arrive under. Anything else is refused rather than stripped by
# a fuzzy suffix match.
MOSS_DIT_PREFIXES = ("", "model.diffusion_model.", "diffusion_model.")

# `sha256(final_linear_video.weight bytes)`: Base and SFT share every key, shape and dtype, so the
# variant is a content signature, not a layout. Identical between each variant's BF16 and INT8
# builds (Comfy-Org/MOSS-Image@82920903, read by bounded range requests on 2026-10-16).
MOSS_VARIANT_FINGERPRINTS = {
    "ef69adce608244cd8866aaaa4b5f9a2135bb3c2983c793a0e4a88412613f2118": "base",
    "b2f3abd89f95c547e7c7a5159c9b308ded92c490d55bc45f9e47f0863cee8352": "sft",
}
MOSS_VARIANT_LABELS = {"base": "MOSS-Image Base", "sft": "MOSS-Image SFT", "unrecognised": "MOSS-Image (unrecognised variant)"}

# The signature `model_config` in the official plugin requires, under canonical names.
_REQUIRED_KEYS = (
    "video_embedder.weight",
    "video_embedder.bias",
    "text_embedder.proj.weight",
    "text_embedder.blocks.0.qkv.weight",
    "shared_block_modulation.weight",
    "layers.0.q_norm.weight",
    "layers.0.proj.ffn_up_gate.weight",
    "final_linear_video.weight",
    "video_head_modulation.weight",
    "time_embedding.0.weight",
    "time_embedding.0.bias",
)


def situ(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def rotate(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rope: torch.Tensor, heads: int) -> torch.Tensor:
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope).transpose(1, 2)
    k = rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)


class RMSNorm(nn.Module):
    """``_RMSNorm`` of the reference: normalise in float32, round, *then* apply the weight."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


class MossTextBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class MossTextProjection(nn.Module):
    def __init__(self, text_dim: int, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([MossTextBlock(dim, ffn_dim, heads) for _ in range(MOSS_TEXT_BLOCKS)])

    def forward(self, x: torch.Tensor, rope: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        x = self.proj(x.to(compute_dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class MossBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({
            "qkv": nn.Linear(dim, 3 * dim, bias=False),
            "o": nn.Linear(dim, dim, bias=False),
            "ffn_up_gate": nn.Linear(dim, 2 * ffn_dim, bias=False),
            "ffn_down": nn.Linear(ffn_dim, dim, bias=False),
        })

    def forward(self, x: torch.Tensor, rope: torch.Tensor, modulation: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        y = y.to(compute_dtype)
        q, k, v = self.proj["qkv"](y).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](attention(q, k, v, rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(compute_dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f[:, None]


def rope_bands(head_dim: int, base: float = MOSS_ROPE_BASE) -> torch.Tensor:
    """Training's CPU float32 construction; a stored copy is checked against it, never trusted."""
    num_bands = head_dim // 8
    exponent = torch.arange(num_bands, device="cpu", dtype=torch.float32) / num_bands
    return 1 / (base ** exponent)


def rope_table(height: int, width: int, text_length: int, bands: torch.Tensor, device) -> torch.Tensor:
    bands = bands.to(device)
    h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
    text[:, 1] = torch.arange(text_length, device=device)
    phase = torch.cat((image, text), 0).unsqueeze(-1) * bands
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


def _model_mixin_bases():
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin

    return ModelMixin, ConfigMixin, register_to_config


_ModelMixin, _ConfigMixin, _register_to_config = _model_mixin_bases()


class MossTransformer2DModel(_ModelMixin, _ConfigMixin):
    """The MOSS-Image DiT under the Comfy-Org (fused ``proj.qkv``) parameter names."""

    _supports_group_offloading = True
    _no_split_modules = ["MossBlock", "MossTextBlock"]

    @_register_to_config
    def __init__(
        self,
        dim: int = 3072,
        ffn_dim: int = 8192,
        num_heads: int = 24,
        num_layers: int = 42,
        text_dim: int = 4096,
        freq_dim: int = 256,
        latent_channels: int = 128,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("MOSS requires dim divisible by heads, head_dim divisible by 8, and an even freq_dim")
        self.latent_channels = latent_channels
        self.freq_dim = freq_dim
        self.head_dim = dim // num_heads
        self.compute_dtype = torch.bfloat16
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = MossTextProjection(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([MossBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def _bands(self) -> torch.Tensor:
        return rope_bands(self.head_dim, float(self.config.rope_base))

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        skip_layers=frozenset(),
    ) -> torch.Tensor:
        compute_dtype = self.compute_dtype
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError("hidden_states must be [B, latent_channels, H, W]")
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim]")
        skip = frozenset(skip_layers)
        if any(not 0 <= int(index) < len(self.layers) for index in skip):
            raise ValueError(f"skip layers must be block indices in [0, {len(self.layers)})")
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("encoder_attention_mask must be a binary [B, S] mask")
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1).expand(batch)
        # Training computes the sinusoid in float64 and feeds float32 into its MLP.
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(compute_dtype)).float()
        mod = self.shared_block_modulation(time.to(compute_dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(compute_dtype)).view(batch, 2, -1)
        bands = self._bands()
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index:index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError("Each sample needs at least one valid text token")
            video = hidden_states[index:index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(compute_dtype).contiguous())
            rope = rope_table(height, width, context.shape[1], bands, hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width:], compute_dtype)
            x = torch.cat((video, text), 1)
            for layer_index, layer in enumerate(self.layers):
                if layer_index in skip:
                    continue
                x = layer(x, rope, mod[index:index + 1], compute_dtype)
            x = self.final_norm_video(x[:, :height * width].to(compute_dtype))
            x = x * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
            x = self.final_linear_video(x.to(compute_dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        return torch.cat(outputs)


# -- checkpoint recognition and conversion -----------------------------------------------------


def _strip_prefix(state: Mapping[str, torch.Tensor]) -> tuple[dict, str]:
    """Remove the one explicit prefix every key shares; refuse mixed or unknown prefixes."""
    keys = list(state)
    for prefix in sorted(MOSS_DIT_PREFIXES, key=len, reverse=True):
        if not prefix:
            continue
        if all(key.startswith(prefix) for key in keys):
            return {key[len(prefix):]: value for key, value in state.items()}, prefix
        if any(key.startswith(prefix) for key in keys):
            raise ValueError(f"MOSS diffusion checkpoint mixes keys under {prefix!r} with keys outside it")
    return dict(state), ""


def _block_count(keys, pattern: str) -> int:
    expression = re.compile(pattern)
    indices = {int(match.group(1)) for key in keys if (match := expression.match(key))}
    if not indices:
        return 0
    if indices != set(range(max(indices) + 1)):
        raise ValueError(f"MOSS diffusion checkpoint has non-contiguous block indices for {pattern}")
    return max(indices) + 1


def detect_qkv_layout(keys) -> str:
    """``fused`` (Comfy-Org) or ``split`` (HF / official plugin). A block naming both is refused."""
    keys = set(keys)
    fused = any(re.match(r"^layers\.\d+\.proj\.qkv\.weight$", key) for key in keys)
    split = any(re.match(r"^layers\.\d+\.proj\.[qkv]\.weight$", key) for key in keys)
    if fused and split:
        raise ValueError("MOSS diffusion checkpoint carries both fused proj.qkv and split proj.q/k/v weights")
    if not fused and not split:
        raise ValueError("MOSS diffusion checkpoint has neither proj.qkv nor proj.q/k/v attention weights")
    return "fused" if fused else "split"


def shape_of(state, key: str) -> tuple[int, ...]:
    value = state.get(key)
    if value is None:
        raise ValueError(f"The selected file is not a complete MOSS-Image diffusion model: missing {key}")
    return tuple(int(v) for v in value.shape)


def infer_moss_config(state: Mapping[str, torch.Tensor]) -> dict:
    """Read the DiT geometry off canonical (fused) tensor shapes and validate every dimension."""
    missing = [key for key in _REQUIRED_KEYS if key not in state]
    if missing:
        raise ValueError(f"The selected file is not a complete MOSS-Image diffusion model; missing: {missing[:6]}")
    dim = shape_of(state, "video_embedder.bias")[0]
    latent = shape_of(state, "video_embedder.weight")[1]
    head = shape_of(state, "layers.0.q_norm.weight")[0]
    if head < 8 or dim % head or head % 8:
        raise ValueError(f"MOSS attention head width {head} does not divide the {dim}-wide model in multiples of 8")
    num_heads = dim // head
    layers = _block_count(state, r"^layers\.(\d+)\.")
    text_blocks = _block_count(state, r"^text_embedder\.blocks\.(\d+)\.")
    if text_blocks != MOSS_TEXT_BLOCKS:
        raise ValueError(f"MOSS text embedder has {text_blocks} refinement blocks; the architecture has {MOSS_TEXT_BLOCKS}")
    ffn = shape_of(state, "layers.0.proj.ffn_up_gate.weight")[0]
    if ffn % 2:
        raise ValueError("MOSS ffn_up_gate is not an even gate/up pair")
    ffn //= 2
    text_dim = shape_of(state, "text_embedder.proj.weight")[1]
    freq_dim = shape_of(state, "time_embedding.0.weight")[1]
    config = {
        "dim": dim,
        "ffn_dim": ffn,
        "num_heads": num_heads,
        "num_layers": layers,
        "text_dim": text_dim,
        "freq_dim": freq_dim,
        "latent_channels": latent,
        "rope_base": MOSS_ROPE_BASE,
    }
    expected = expected_shapes(config)
    for key, value in state.items():
        if key in expected and tuple(int(v) for v in value.shape) != expected[key]:
            raise ValueError(f"MOSS tensor {key} is {tuple(value.shape)}; the inferred architecture needs {expected[key]}")
    return config


def expected_shapes(config: Mapping) -> dict[str, tuple[int, ...]]:
    dim, ffn, heads = int(config["dim"]), int(config["ffn_dim"]), int(config["num_heads"])
    head = dim // heads
    text, freq, latent = int(config["text_dim"]), int(config["freq_dim"]), int(config["latent_channels"])
    shapes = {
        "video_embedder.weight": (dim, latent), "video_embedder.bias": (dim,),
        "text_embedder.proj.weight": (dim, text), "text_embedder.proj.bias": (dim,),
        "time_embedding.0.weight": (dim, freq), "time_embedding.0.bias": (dim,),
        "time_embedding.2.weight": (dim, dim), "time_embedding.2.bias": (dim,),
        "shared_block_modulation.weight": (6 * dim, dim), "shared_block_modulation.bias": (6 * dim,),
        "final_norm_video.weight": (dim,), "final_linear_video.weight": (latent, dim),
        "video_head_modulation.weight": (2 * dim, dim), "video_head_modulation.bias": (2 * dim,),
    }
    for index in range(MOSS_TEXT_BLOCKS):
        base = f"text_embedder.blocks.{index}"
        shapes.update({
            f"{base}.norm1.weight": (dim,), f"{base}.norm2.weight": (dim,),
            f"{base}.q_norm.weight": (head,), f"{base}.k_norm.weight": (head,),
            f"{base}.qkv.weight": (3 * dim, dim), f"{base}.qkv.bias": (3 * dim,),
            f"{base}.proj.weight": (dim, dim), f"{base}.proj.bias": (dim,),
            f"{base}.ffn_gate.weight": (ffn, dim), f"{base}.ffn_gate.bias": (ffn,),
            f"{base}.ffn_up.weight": (ffn, dim), f"{base}.ffn_up.bias": (ffn,),
            f"{base}.ffn_down.weight": (dim, ffn), f"{base}.ffn_down.bias": (dim,),
        })
    for index in range(int(config["num_layers"])):
        base = f"layers.{index}"
        shapes.update({
            f"{base}.norm1.weight": (dim,), f"{base}.norm2.weight": (dim,),
            f"{base}.q_norm.weight": (head,), f"{base}.k_norm.weight": (head,),
            f"{base}.proj.qkv.weight": (3 * dim, dim), f"{base}.proj.o.weight": (dim, dim),
            f"{base}.proj.ffn_up_gate.weight": (2 * ffn, dim), f"{base}.proj.ffn_down.weight": (dim, ffn),
        })
    return shapes


def linear_layer_names(config: Mapping) -> list[str]:
    names = ["video_embedder", "text_embedder.proj", "time_embedding.0", "time_embedding.2",
             "shared_block_modulation", "final_linear_video", "video_head_modulation"]
    for index in range(MOSS_TEXT_BLOCKS):
        names += [f"text_embedder.blocks.{index}.{name}" for name in ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down")]
    for index in range(int(config["num_layers"])):
        names += [f"layers.{index}.proj.{name}" for name in ("qkv", "o", "ffn_up_gate", "ffn_down")]
    return names


QKV_FUSION_VERSION = 1
_SCALE_SUFFIXES = (".weight_scale", ".scale_weight")


def fuse_split_qkv(state: Mapping[str, torch.Tensor], marker_reader=None) -> tuple[dict, int]:
    """Rewrite HF/plugin ``proj.q/k/v`` into the canonical fused ``proj.qkv``, exactly.

    BF16/FP32 parts concatenate on the output axis. Quantised parts fuse only when all three share
    one format and marker; per-row scales concatenate, scalar scales are broadcast to rows (exact,
    since a per-row scale equal to the scalar multiplies the same rows by the same number).
    Activation-side scales, block layouts and mismatched markers are refused.
    """
    state = dict(state)
    layers = sorted({int(m.group(1)) for key in state if (m := re.match(r"^layers\.(\d+)\.proj\.[qkv]\.", key))})
    fused = 0
    for index in layers:
        base = f"layers.{index}.proj"
        parts = [f"{base}.{name}" for name in ("q", "k", "v")]
        weights = []
        for part in parts:
            if f"{part}.weight" not in state:
                raise ValueError(f"MOSS split attention for block {index} is missing {part}.weight")
            weights.append(state[f"{part}.weight"])
        dtypes = {w.dtype for w in weights}
        if len(dtypes) != 1:
            raise ValueError(f"MOSS block {index} stores q/k/v in different dtypes ({sorted(map(str, dtypes))})")
        markers = [state.get(f"{part}.comfy_quant") for part in parts]
        scales = []
        for part in parts:
            found = [state.get(f"{part}{suffix}") for suffix in _SCALE_SUFFIXES if f"{part}{suffix}" in state]
            scales.append(found[0] if found else None)
        for part in parts:
            for suffix in (".input_scale", ".scale_input", ".bias", ".weight_scale_2", ".pre_quant_scale"):
                if f"{part}{suffix}" in state:
                    raise ValueError(f"MOSS split attention {part}{suffix} cannot be fused exactly; refused")
        if any(marker is not None for marker in markers):
            if any(marker is None for marker in markers):
                raise ValueError(f"MOSS block {index}: only some of q/k/v carry a quantisation marker")
            decoded = [bytes(m.to(torch.uint8).reshape(-1).tolist()) for m in markers]
            if len(set(decoded)) != 1:
                raise ValueError(f"MOSS block {index}: q/k/v quantisation markers differ; cannot fuse exactly")
        if any(scale is not None for scale in scales):
            if any(scale is None for scale in scales):
                raise ValueError(f"MOSS block {index}: only some of q/k/v carry a weight scale")
            rows = [int(w.shape[0]) for w in weights]
            fused_scales = []
            for scale, count in zip(scales, rows):
                scale = scale.to(torch.float32)
                if scale.numel() == 1:
                    fused_scales.append(scale.reshape(1, 1).expand(count, 1))
                elif scale.numel() == count:
                    fused_scales.append(scale.reshape(count, 1))
                else:
                    raise ValueError(f"MOSS block {index}: a {tuple(scale.shape)} scale is neither scalar nor per-row")
            state[f"{base}.qkv.weight_scale"] = torch.cat(fused_scales, 0).contiguous()
        elif weights[0].dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2) and markers[0] is not None:
            raise ValueError(f"MOSS block {index}: quantised q/k/v carry no scale")
        state[f"{base}.qkv.weight"] = torch.cat(weights, 0).contiguous()
        if markers[0] is not None:
            state[f"{base}.qkv.comfy_quant"] = markers[0]
        for part in parts:
            for suffix in (".weight", ".comfy_quant") + _SCALE_SUFFIXES:
                state.pop(f"{part}{suffix}", None)
        fused += 1
    return state, fused


def canonical_moss_state(state: Mapping[str, torch.Tensor]) -> tuple[dict, dict]:
    """Strip the explicit prefix and fuse split attention; return ``(state, layout report)``."""
    stripped, prefix = _strip_prefix(state)
    layout = detect_qkv_layout(stripped)
    fused_blocks = 0
    if layout == "split":
        stripped, fused_blocks = fuse_split_qkv(stripped)
    return stripped, {"prefix": prefix, "source_qkv": layout, "fused_blocks": fused_blocks, "qkv_fusion_version": QKV_FUSION_VERSION}


def check_rope_bands(state: Mapping[str, torch.Tensor], head_dim: int) -> str:
    """A stored ``rope.bands`` must equal the derived constant; it is retained only for compatibility."""
    stored = state.get("rope.bands")
    if stored is None:
        return "absent"
    expected = rope_bands(head_dim)
    if tuple(stored.shape) != tuple(expected.shape) or not torch.allclose(stored.float(), expected, rtol=0, atol=1e-7):
        raise ValueError("MOSS rope.bands does not match the base-10000 constant this architecture derives")
    return "verified"


def variant_fingerprint(state: Mapping[str, torch.Tensor]) -> tuple[str, str]:
    """``(variant, sha256)`` from ``final_linear_video.weight`` bytes; a name never decides it."""
    tensor = state["final_linear_video.weight"].detach().contiguous().cpu()
    digest = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
    return MOSS_VARIANT_FINGERPRINTS.get(digest, "unrecognised"), digest


def classify_moss_dit(keys_or_state) -> dict:
    """Architecture recognition without values: prefix, layout and a complete signature."""
    keys = list(keys_or_state)
    placeholder = {key: None for key in keys}
    stripped, prefix = _strip_prefix(placeholder)
    layout = detect_qkv_layout(stripped)
    canonical = set(stripped)
    if layout == "split":
        canonical = {re.sub(r"\.proj\.[qkv]\.", ".proj.qkv.", key) for key in canonical}
    missing = [key for key in _REQUIRED_KEYS if key not in canonical]
    if missing:
        raise ValueError(f"not a MOSS-Image diffusion model (missing {missing[:4]})")
    return {"architecture": MOSS_DIT_ARCHITECTURE, "prefix": prefix, "qkv_layout": layout}


# -- LoRA key mapping ------------------------------------------------------------------------


_LORA_PREFIXES = ("", "diffusion_model.", "transformer.", "model.diffusion_model.", "base_model.model.")
_QKV_PARTS = {"q": 0, "k": 1, "v": 2}


def moss_lora_targets(config: Mapping) -> dict[str, tuple[str, int | None]]:
    """Every LoRA module name a MOSS DiT accepts -> ``(canonical linear, qkv slice index or None)``."""
    targets = {name: (name, None) for name in linear_layer_names(config)}
    for index in range(int(config["num_layers"])):
        for part, slot in _QKV_PARTS.items():
            targets[f"layers.{index}.proj.{part}"] = (f"layers.{index}.proj.qkv", slot)
    return targets


def _lora_lookup(config: Mapping) -> dict[str, tuple[str, int | None]]:
    lookup = {}
    for name, target in moss_lora_targets(config).items():
        for prefix in _LORA_PREFIXES:
            lookup[f"{prefix}{name}"] = target
        flat = name.replace(".", "_")
        lookup[f"lora_unet_{flat}"] = target
        lookup[f"lycoris_{flat}"] = target
    return lookup


_LORA_SUFFIXES = (
    (".lora_A.weight", "down"), (".lora_down.weight", "down"), (".lora_B.weight", "up"), (".lora_up.weight", "up"),
    (".alpha", "alpha"), (".lokr_w1", "lokr_w1"), (".lokr_w2", "lokr_w2"), (".lokr_w2_a", "lokr_w2_a"),
    (".lokr_w2_b", "lokr_w2_b"),
)


def _lokr_entry(entry: Mapping, module: str, label: str) -> dict:
    """One LoKr group as ``w1``, ``w2`` and the scale ComfyUI applies to ``w1 ⊗ w2``.

    A full second factor carries no rank to normalise by, so the product is used as stored and the
    file's alpha is only checked (LyCORIS writes about 1e10 there); a low-rank ``w2_a @ w2_b`` takes
    ``alpha / rank``, or 1 without an alpha. ``comfy/weight_adapter/lokr.py`` and the Anima engine
    read both forms the same way.
    """
    w1 = entry.get("lokr_w1")
    full = "lokr_w2" in entry
    low_rank = "lokr_w2_a" in entry or "lokr_w2_b" in entry
    if w1 is None or full == low_rank:
        raise ValueError(f"{label} has an incomplete LoKr group for {module}")
    alpha = entry.get("alpha")
    if alpha is not None and not math.isfinite(float(alpha.reshape(()).item())):
        raise ValueError(f"{label} has a non-finite LoKr alpha for {module}")
    if full:
        w2, scale = entry["lokr_w2"].to(torch.float32), 1.0
    else:
        w2_a, w2_b = entry.get("lokr_w2_a"), entry.get("lokr_w2_b")
        if w2_a is None or w2_b is None or w2_a.ndim != 2 or w2_b.ndim != 2 or w2_a.shape[1] != w2_b.shape[0]:
            raise ValueError(f"{label} has a malformed low-rank LoKr factor for {module}")
        w2 = w2_a.to(torch.float32) @ w2_b.to(torch.float32)
        scale = float(alpha.reshape(()).item()) / int(w2_b.shape[0]) if alpha is not None else 1.0
    if w1.ndim != 2 or w2.ndim != 2:
        raise ValueError(f"{label} has a malformed LoKr pair for {module}")
    return {"module": module, "kind": "lokr", "w1": w1.to(torch.float32), "w2": w2, "scale": scale}


def convert_moss_lora_state_dict(state: Mapping[str, torch.Tensor], config: Mapping, label: str) -> dict:
    """Bring a MOSS DiT LoRA or LoKr into ``{target: [entry, ...]}`` canonical form.

    Accepts ``lora_A/lora_B`` and ``lora_down/lora_up`` with an optional ``alpha``, and LyCORIS LoKr
    with a full ``lokr_w1`` and a full or low-rank second factor (see :func:`_lokr_entry`). q/k/v
    targets map onto row slices of the fused ``proj.qkv``; a file naming a block's ``qkv`` *and* one
    of its q/k/v is refused (the slice would be patched twice). DoRA, LoHa, a decomposed or Tucker
    LoKr first factor and text-encoder LoRA are refused.
    """
    keys = list(state)
    if any("dora_scale" in key or ".hada_" in key or "lora_mid" in key for key in keys):
        raise ValueError(f"{label} is a DoRA/LoHa adapter; MOSS accepts standard LoRA and LoKr only")
    if any(key.endswith((".lokr_w1_a", ".lokr_w1_b", ".lokr_t1", ".lokr_t2")) for key in keys):
        raise ValueError(f"{label} decomposes its LoKr first factor or uses Tucker factors, which MOSS does not apply")
    if any(key.startswith(("lora_te", "text_encoder.", "text_encoders.", "lora_te1_", "lora_te2_")) for key in keys):
        raise ValueError(f"{label} carries text-encoder LoRA weights, which MOSS does not apply")
    lookup = _lora_lookup(config)
    grouped: dict[str, dict] = {}
    unknown = set()
    for key, value in state.items():
        for suffix, slot in _LORA_SUFFIXES:
            if key.endswith(suffix):
                module = key[: -len(suffix)]
                target = lookup.get(module)
                if target is None:
                    unknown.add(module)
                    break
                entry = grouped.setdefault(module, {"target": target})
                if slot in entry:
                    raise ValueError(f"{label} names {module} {slot} twice")
                entry[slot] = value
                break
        else:
            if not key.endswith((".dora_scale",)):
                unknown.add(key)
    if unknown:
        raise ValueError(f"{label} has tensors this MOSS DiT cannot place: {sorted(unknown)[:4]}")
    if not grouped:
        raise ValueError(f"{label} contains no LoRA pairs")
    result: dict[str, list] = {}
    whole_qkv, sliced = set(), set()
    for module, entry in grouped.items():
        target, slot = entry["target"]
        lokr = any(role.startswith("lokr_") for role in entry)
        if lokr and ("down" in entry or "up" in entry):
            raise ValueError(f"{label} mixes LoRA and LoKr tensors for {module}")
        if lokr:
            item = _lokr_entry(entry, module, label)
        else:
            down, up = entry.get("down"), entry.get("up")
            if down is None or up is None:
                raise ValueError(f"{label} has an unpaired LoRA tensor for {module}")
            if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
                raise ValueError(f"{label} has a malformed LoRA pair for {module}")
            alpha = entry.get("alpha")
            scale = 1.0
            if alpha is not None:
                rank = int(down.shape[0])
                scale = float(alpha.reshape(()).item()) / rank
            item = {"module": module, "kind": "lora", "down": down, "up": up, "scale": scale}
        if target.endswith(".proj.qkv"):
            (sliced if slot is not None else whole_qkv).add(target)
        result.setdefault(target, []).append({**item, "slot": slot})
    conflicts = whole_qkv & sliced
    if conflicts:
        raise ValueError(f"{label} patches {sorted(conflicts)[0]} both as fused qkv and as q/k/v slices")
    for target, entries in result.items():
        slots = [e["slot"] for e in entries if e["slot"] is not None]
        if len(slots) != len(set(slots)):
            raise ValueError(f"{label} patches one q/k/v slice of {target} twice")
    return result


def lora_pair_for(target_entries, out_features: int, in_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One ``(down, up)`` pair for a target, with q/k/v slices placed into the fused rows.

    Several slices stack on the rank axis; ``up`` is zero outside each slice's rows, so the product
    equals the sum of the per-slice deltas exactly.
    """
    downs, ups = [], []
    for entry in target_entries:
        down = entry["down"].to(torch.float32) * float(entry["scale"])
        up = entry["up"].to(torch.float32)
        if down.shape[1] != in_features:
            raise ValueError(f"LoRA {entry['module']} expects {down.shape[1]} inputs; the layer has {in_features}")
        if entry["slot"] is None:
            if up.shape[0] != out_features:
                raise ValueError(f"LoRA {entry['module']} produces {up.shape[0]} outputs; the layer has {out_features}")
            ups.append(up)
        else:
            width = out_features // 3
            if up.shape[0] != width:
                raise ValueError(f"LoRA {entry['module']} produces {up.shape[0]} outputs; one q/k/v slice is {width}")
            placed = up.new_zeros((out_features, up.shape[1]))
            placed[entry["slot"] * width:(entry["slot"] + 1) * width] = up
            ups.append(placed)
        downs.append(down)
    return torch.cat(downs, 0), torch.cat(ups, 1)


def lokr_terms_for(target_entries, out_features: int, in_features: int) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """``(w1 · scale, w2, row_offset)`` per LoKr entry, each product checked against its rows.

    A q/k/v entry covers one third of the fused ``proj.qkv`` rows and is placed at that slice.
    """
    terms = []
    for entry in target_entries:
        w1, w2 = entry["w1"].to(torch.float32), entry["w2"].to(torch.float32)
        rows, cols = w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1]
        if cols != in_features:
            raise ValueError(f"LoKr {entry['module']} expects {cols} inputs; the layer has {in_features}")
        width = out_features if entry["slot"] is None else out_features // 3
        if rows != width:
            raise ValueError(f"LoKr {entry['module']} produces {rows} outputs; its target rows number {width}")
        offset = 0 if entry["slot"] is None else entry["slot"] * width
        terms.append((w1 * float(entry["scale"]), w2, offset))
    return terms
