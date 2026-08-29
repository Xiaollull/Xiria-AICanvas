"""Krea 2 (K2) single-stream MMDiT, ported from ``comfy/ldm/krea2/model.py``.

Every other native engine in this project mounts a Diffusers model class and converts the ComfyUI
checkpoint into Diffusers' key layout.  Krea 2 has no Diffusers implementation, so the transformer
itself lives here and the checkpoint is loaded in ComfyUI's own naming — which also means a Krea 2
LoRA published against ``diffusion_model.blocks.N.attn.wq`` needs no key translation at all.

The architecture, in one paragraph: a Qwen3-VL-4B text encoder is tapped at twelve depths, and the
resulting ``(B, seq, 12, 2560)`` stack is refined by an internal ``txtfusion`` adapter — two blocks
attending *across the twelve taps* at each token position, a learned projection collapsing the tap
axis to one, then two blocks attending across the sequence.  The refined text tokens and the 2x2
patchified image tokens are concatenated into a single stream and run through ``layers`` shared
transformer blocks with AdaLN-single modulation (one shared parameter vector per block, added to a
timestep projection), GQA attention with per-head QK RMS-norm and a sigmoid gate on the attention
output, a SwiGLU MLP, and 3-axis RoPE where text sits at the origin and image tokens carry
``(0, row, column)``.

Two parts of the reference are deliberately absent:

* **Reference-latent conditioning** (``ref_latents`` / ``timestep_zero_index``).  ComfyUI drives it
  from ``default_ref_method``, which ``model_detection`` never sets for Krea 2 — it exists for a
  future edit-style checkpoint.  This runtime has no call site that could supply reference latents,
  so porting the branch would add a modulation split nothing can reach or test.
* **The temporal fold** (a five-dimensional ``x``).  Krea 2 uses Wan 2.1's video autoencoder, whose
  latents carry a frame axis, but a still-image run always holds exactly one frame.  The pipeline
  keeps four-dimensional latents and adds the frame axis only at the autoencoder boundary, so the
  fold would likewise be unreachable here.

Everything else is term-for-term the reference, including the ``(1 + scale)`` RMS-norm weight
convention, the cos-before-sin timestep embedding, and the float64 RoPE tables.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


# `TextFusionTransformer` hardcodes two layerwise and two refiner blocks; `krea2_to_diffusers`
# relies on the same constants when it builds a LoRA key map.
KREA2_TEXT_FUSION_LAYERWISE_BLOCKS = 2
KREA2_TEXT_FUSION_REFINER_BLOCKS = 2


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0):
    """``comfy/ldm/flux/layers.py::timestep_embedding``.

    Note the cosine half comes first.  Diffusers' equivalent emits sine first, so swapping the two
    would silently shift every timestep by a quarter period.
    """
    timesteps = time_factor * timesteps
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(timesteps):
        embedding = embedding.to(timesteps)
    return embedding


def rope(positions: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """``comfy/ldm/flux/math.py::rope``: (B, N) positions -> (B, N, dim/2, 2, 2) rotation pairs.

    The frequency ladder is built in float64 because ``theta ** scale`` spans three decades and a
    float32 ladder rounds the high-frequency end into visible banding at long sequences.
    """
    if dim % 2:
        raise ValueError("RoPE axis dimensions must be even")
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=positions.device)
    omega = 1.0 / (theta**scale)
    out = positions.to(dtype=torch.float32).unsqueeze(-1) * omega
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).to(dtype=torch.float32)


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """``comfy/ldm/flux/math.py::_apply_rope1``, applied to one of q or k."""
    values = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    rotated = freqs_cis[..., 0] * values[..., 0]
    rotated = rotated + freqs_cis[..., 1] * values[..., 1]
    return rotated.reshape(*x.shape).type_as(x)


def apply_rope(query: torch.Tensor, key: torch.Tensor, freqs_cis: torch.Tensor):
    return apply_rope1(query, freqs_cis), apply_rope1(key, freqs_cis)


class EmbedND(nn.Module):
    """``comfy/ldm/flux/layers.py::EmbedND``: one RoPE table per position axis, concatenated."""

    def __init__(self, dim: int, theta: int, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = int(theta)
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [rope(ids[..., axis], self.axes_dim[axis], self.theta) for axis in range(ids.shape[-1])],
            dim=-3,
        ).unsqueeze(1)


class RMSNorm(nn.Module):
    """RMSNorm with the reference ``(1 + scale)`` weight convention (the scale is zero-centred)."""

    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.empty(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        weight = self.scale.to(device=x.device, dtype=torch.float32) + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), weight=weight, eps=self.eps).to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, query, key):
        return self.qnorm(query), self.knorm(key)


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()
        hidden = int(2 * features / 3) * multiplier
        hidden = multiple * ((hidden + multiple - 1) // multiple)
        self.gate = nn.Linear(features, hidden, bias=bias)
        self.up = nn.Linear(features, hidden, bias=bias)
        self.down = nn.Linear(hidden, features, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    """GQA with per-head QK RMS-norm and a sigmoid gate applied to the attention output."""

    def __init__(self, dim: int, heads: int, kvheads=None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        if dim % self.heads:
            raise ValueError(f"Krea2 attention width {dim} is not divisible by {self.heads} heads")
        if self.heads % self.kvheads:
            raise ValueError(f"Krea2 attention has {self.heads} heads, not a multiple of {self.kvheads} key/value heads")
        self.headdim = dim // self.heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def _heads(self, tensor: torch.Tensor, heads: int) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, heads, self.headdim).transpose(1, 2)

    def forward(self, x, freqs=None):
        batch, length, _ = x.shape
        query = self._heads(self.wq(x), self.heads)
        key = self._heads(self.wk(x), self.kvheads)
        value = self._heads(self.wv(x), self.kvheads)
        gate = self.gate(x)
        query, key = self.qknorm(query, key)
        if freqs is not None:
            query, key = apply_rope(query, key, freqs)
        if self.kvheads != self.heads:
            repeats = self.heads // self.kvheads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        out = F.scaled_dot_product_attention(query, key, value)
        out = out.transpose(1, 2).reshape(batch, length, self.heads * self.headdim)
        return self.wo(out * F.sigmoid(gate))


class SimpleModulation(nn.Module):
    """The final layer's modulation: a shared (2, dim) offset added to the timestep projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.empty(2, dim))

    def forward(self, vec):
        out = vec + self.lin.to(device=vec.device, dtype=vec.dtype).unsqueeze(0)
        scale, shift = out.chunk(2, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    """AdaLN-single: one shared 6*dim vector per block, added to the projected timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.empty(6 * dim))

    def forward(self, vec):
        return (vec + self.lin.to(device=vec.device, dtype=vec.dtype)).chunk(6, dim=-1)


class TextFusionBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x):
        x = x + self.attn(self.prenorm(x))
        return x + self.mlp(self.postnorm(x))


class TextFusionTransformer(nn.Module):
    """Collapse the twelve encoder taps into one refined text sequence.

    The layerwise blocks see a sequence of *taps* — batch and token position are folded together,
    so each token attends across the twelve depths it was read at.  ``projector`` then contracts
    that axis to one and the refiner blocks attend across the token sequence as usual.
    """

    def __init__(self, num_txt_layers, txt_dim, heads, multiplier, bias=False, kvheads=None):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList([
            TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
            for _ in range(KREA2_TEXT_FUSION_LAYERWISE_BLOCKS)
        ])
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([
            TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
            for _ in range(KREA2_TEXT_FUSION_REFINER_BLOCKS)
        ])

    def forward(self, x):
        batch, sequence, taps, width = x.shape
        x = x.reshape(batch * sequence, taps, width)
        for block in self.layerwise_blocks:
            x = block(x.contiguous())
        x = x.reshape(batch, sequence, taps, width).permute(0, 1, 3, 2)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, vec, freqs):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs)
        return x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)


class LastLayer(nn.Module):
    def __init__(self, features, patch, channels):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        return self.linear((1 + scale) * self.norm(x) + shift)


def _model_mixin_bases():
    """Mount the transformer on Diffusers' ``ModelMixin`` so it shares the offload machinery.

    The server drives every native engine's transformer through ``enable_group_offload`` and the
    ``_diffusers_hook`` registry that removes it again.  Inheriting those is the difference between
    Krea 2 streaming a block at a time on an 8 GB card and Krea 2 needing its own offload
    implementation, so the dependency is load-bearing rather than decorative.
    """
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin

    return ModelMixin, ConfigMixin, register_to_config


_ModelMixin, _ConfigMixin, _register_to_config = _model_mixin_bases()


class Krea2Transformer2DModel(_ModelMixin, _ConfigMixin):
    """``comfy/ldm/krea2/model.py::SingleStreamDiT`` under ComfyUI's own parameter names."""

    _supports_group_offloading = True
    _no_split_modules = ["SingleStreamBlock", "TextFusionBlock"]

    @_register_to_config
    def __init__(
        self,
        features: int = 6144,
        tdim: int = 256,
        txtdim: int = 2560,
        heads: int = 48,
        kvheads: int = 12,
        multiplier: int = 4,
        layers: int = 28,
        patch: int = 2,
        channels: int = 16,
        bias: bool = False,
        theta: int = 1000,
        txtlayers: int = 12,
        txtheads: int = 20,
        txtkvheads: int = 20,
    ):
        super().__init__()
        self.patch = patch
        self.channels = channels
        self.tdim = tdim
        self.heads = heads
        self.txtdim = txtdim
        self.txtlayers = txtlayers

        if features % heads:
            raise ValueError(f"Krea2 hidden size {features} is not divisible by {heads} heads")
        headdim = features // heads
        # The 3-axis split ComfyUI computes: the index axis takes what the two spatial axes leave.
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        if sum(axes) != headdim:
            raise ValueError(f"Krea2 RoPE axes {axes} do not sum to the {headdim}-wide attention head")
        self.pe_embedder = EmbedND(dim=headdim, theta=int(theta), axes_dim=axes)

        self.first = nn.Linear(channels * patch**2, features, bias=True)
        self.blocks = nn.ModuleList([
            SingleStreamBlock(features, heads, multiplier, bias, kvheads) for _ in range(layers)
        ])
        self.tmlp = nn.Sequential(
            nn.Linear(tdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.txtfusion = TextFusionTransformer(txtlayers, txtdim, txtheads, multiplier, bias, txtkvheads)
        self.txtmlp = nn.Sequential(
            RMSNorm(txtdim),
            nn.Linear(txtdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.last = LastLayer(features, patch, channels)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features * 6),
        )

    # -- patching ------------------------------------------------------------------------------

    def process_img(self, x: torch.Tensor, index: int = 0):
        """Pad to the patch grid, fold each 2x2 cell into the channel axis, and number the tokens."""
        patch = self.patch
        pad_h = (patch - x.shape[-2] % patch) % patch
        pad_w = (patch - x.shape[-1] % patch) % patch
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        batch, channels, height, width = x.shape
        rows, columns = height // patch, width // patch
        img = x.view(batch, channels, rows, patch, columns, patch)
        img = img.permute(0, 2, 4, 1, 3, 5).reshape(batch, rows * columns, channels * patch * patch)

        ids = torch.zeros(rows, columns, 3, device=x.device, dtype=torch.float32)
        ids[..., 0] = index
        ids[..., 1] = torch.arange(rows, device=x.device, dtype=torch.float32)[:, None]
        ids[..., 2] = torch.arange(columns, device=x.device, dtype=torch.float32)[None, :]
        return img, ids.reshape(1, rows * columns, 3).repeat(batch, 1, 1), rows, columns

    def unprocess_img(self, tokens: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
        patch = self.patch
        batch = tokens.shape[0]
        out = tokens.view(batch, rows, columns, self.channels, patch, patch)
        return out.permute(0, 3, 1, 4, 2, 5).reshape(batch, self.channels, rows * patch, columns * patch)

    def unpack_context(self, context: torch.Tensor) -> torch.Tensor:
        """(B, seq, txtlayers*txtdim) -> (B, seq, txtlayers, txtdim).

        ComfyUI carries conditioning as a 3D tensor, so ``Krea2TEModel`` flattens the twelve taps
        into the feature axis and the model takes them apart again.  Keeping that contract means a
        conditioning tensor produced by ComfyUI and one produced here are the same object.
        """
        batch, sequence, fused = context.shape
        if fused != self.txtlayers * self.txtdim:
            raise ValueError(
                f"Krea2 expects conditioning with {self.txtlayers}x{self.txtdim}="
                f"{self.txtlayers * self.txtdim} features (a {self.txtlayers}-layer Qwen3-VL stack) but got {fused}"
            )
        return context.reshape(batch, sequence, self.txtlayers, self.txtdim)

    # -- forward -------------------------------------------------------------------------------

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Krea2 latents must be (batch, channels, height, width)")
        height, width = x.shape[-2], x.shape[-1]
        context = self.unpack_context(context)

        img, imgpos, rows, columns = self.process_img(x)
        img_tokens = img.shape[1]
        img = self.first(img)

        t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
        tvec = self.tproj(t)

        context = self.txtmlp(self.txtfusion(context))
        txtlen = context.shape[1]
        txtpos = torch.zeros(img.shape[0], txtlen, 3, device=context.device, dtype=torch.float32)

        combined = torch.cat((context, img), dim=1)
        del context, img
        freqs = self.pe_embedder(torch.cat((txtpos, imgpos), dim=1))
        del txtpos, imgpos

        for block in self.blocks:
            combined = block(combined, tvec, freqs)

        final = self.last(combined, t)
        del combined
        out = self.unprocess_img(final[:, txtlen:txtlen + img_tokens, :], rows, columns)
        return out[:, :, :height, :width]
