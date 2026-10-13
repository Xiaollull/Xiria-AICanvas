"""Static, inference-only adaptation of PixAI v1.0's SAM3 ViTDet configuration.

Derived architecture: PixAI Labs tagger_pipeline.py, revision
9fe10addf9326e292da8a85a98ea74cd91b41771; Meta SAM3 vitdet.py.
Attribution / terms: resources/pixai-tagger/NOTICE.md and SAM-LICENSE.txt.
Not a generic AutoModel, no downloaded Python imports, no timm dependency.
Only the released configuration is admitted by load_model; strict state load.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def axial_cis(dim, size, scale):
    frequencies = 1 / (10000.0 ** (torch.arange(0, dim, 4)[:dim // 4].float() / dim))
    positions = torch.arange(size * size, dtype=torch.float32)
    x = (positions % size).float() * scale
    y = torch.div(positions, size, rounding_mode="floor").float() * scale
    xf, yf = torch.outer(x, frequencies), torch.outer(y, frequencies)
    return torch.cat((torch.polar(torch.ones_like(xf), xf), torch.polar(torch.ones_like(yf), yf)), -1)


def rotary(x, frequencies):
    value = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(value * frequencies[None, None]).flatten(3).type_as(x)


class Mlp(nn.Module):
    """timm Mlp's active inference path (GELU, no norm, both drop rates zero)."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(0)

    def forward(self, x):
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=False)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)


class Attention(nn.Module):
    def __init__(self, dim, heads, grid, rope_size):
        super().__init__()
        self.num_heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.register_buffer("freqs_cis", axial_cis(dim // heads, grid, rope_size / grid), persistent=False)

    def forward(self, x):
        b, h, w, c = x.shape
        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = rotary(q, self.freqs_cis), rotary(k, self.freqs_cis)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.view(b, self.num_heads, h, w, -1).permute(0, 2, 3, 1, 4).reshape(b, h, w, c)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, heads, ratio, window, grid, rope_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = Attention(dim, heads, window or grid, rope_size)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = Mlp(dim, int(dim * ratio))
        self.window_size = window

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window_size:
            b, h, w, c = x.shape
            size = self.window_size
            ph, pw = (size - h % size) % size, (size - w % size) % size
            x = F.pad(x, (0, 0, 0, pw, 0, ph))
            hp, wp = h + ph, w + pw
            x = x.view(b, hp // size, size, wp // size, size, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, size, size, c)
        x = self.attn(x)
        if self.window_size:
            x = x.reshape(b, hp // size, wp // size, size, size, c).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, c)
            x = x[:, :h, :w]
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class MHAttnPool(nn.Module):
    def __init__(self, dim, heads, patches):
        super().__init__()
        self.num_heads = heads
        self.norm = nn.LayerNorm(dim)  # Official pool uses default eps, independently of block config.
        self.q = nn.Parameter(torch.empty(dim))
        self.kv = nn.Linear(dim, dim * 2)
        self.pos_embed = nn.Parameter(torch.empty(patches, dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        x = self.norm(x) + self.pos_embed[None]
        q = self.q[None, None].expand(b, -1, -1)
        k, v = self.kv(x).chunk(2, -1)
        heads, size = self.num_heads, d // self.num_heads
        q = q.view(b, 1, heads, size).transpose(1, 2)
        k = k.view(b, n, heads, size).transpose(1, 2)
        v = v.view(b, n, heads, size).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).contiguous().view(b, 1, d)).squeeze(1)


class PixAIModel(nn.Module):
    def __init__(self, dim=1024, depth=32, heads=16, patch=14, image_size=1008,
                 pretrain_size=336, window=24, globals=(7, 15, 23, 31), classes=30877, ratio=4.625):
        super().__init__()
        self.patch_embed = PatchEmbed(dim, patch)
        self.pos_embed = nn.Parameter(torch.empty(1, (pretrain_size // patch)**2 + 1, dim))
        grid = image_size // patch
        self.blocks = nn.ModuleList([Block(dim, heads, ratio, 0 if i in globals else window, grid, window) for i in range(depth)])
        self.ln_pre = nn.LayerNorm(dim, eps=1e-5)
        self.ln_post = nn.Identity()
        self.head_pool = MHAttnPool(dim, heads, grid**2)
        self.head = nn.Linear(dim, classes)

    def forward(self, x):
        x = self.patch_embed(x)
        h, w = x.shape[1:3]
        pos = self.pos_embed[:, 1:]
        size = math.isqrt(pos.shape[1])
        pos = pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        if size != h or size != w:
            pos = pos.tile(1, 1, h // size + 1, w // size + 1)[:, :, :h, :w]
        x = self.ln_pre(x + pos.permute(0, 2, 3, 1))
        for block in self.blocks:
            x = block(x)
        # Follow the original BCHW view/permute before pooling, not an altered flatten.
        x = self.ln_post(x).permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        return self.head(self.head_pool(x))


def load_model(path, config, device):
    expected = dict(model_type="cls_vitdet", img_size=1008, patch_size=14, embed_dim=1024, depth=32,
                    num_heads=16, mlp_ratio=4.625, num_classes=30877, window_size=24, pretrain_img_size=336,
                    global_att_blocks=[7, 15, 23, 31], rel_pos_blocks=[], retain_cls_token=False,
                    use_rope=True, use_interp_rope=True, tile_abs_pos=True, use_abs_pos=True,
                    ln_pre=True, ln_post=False, qkv_bias=True, bias_patch_embed=False, init_values=None,
                    dropout=0.0, drop_path_rate=0.0, act_layer="GELU", norm_layer="LayerNorm")
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("PixAI 配置不匹配固定官方 v1.0 架构")
    from safetensors.torch import load_file
    with torch.device("meta"):
        model = PixAIModel()
    state = load_file(str(path), device="cpu")
    # All keys, shapes and parameter types are checked. No arbitrary pickle or AutoModel.
    model.load_state_dict(state, strict=True, assign=True)
    del state
    # RoPE buffers were built on meta and aren't in the state: reconstruct from audited math.
    for index, block in enumerate(model.blocks):
        grid = 72 if index in (7, 15, 23, 31) else 24
        block.attn.freqs_cis = axial_cis(64, grid, 24 / grid)
    return model.eval().to(device)
