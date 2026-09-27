"""Built-in Triton kernels for ComfyUI's INT8 / ConvRot INT8 and ConvRot W4A4 matmuls (P9).

:mod:`comfy_quant` computes these layers with plain PyTorch: the activation is quantised in about
eight separate element-wise passes, the int8 matmul needs a transposed copy of the weight on every
call, W4A4 unpacks its whole weight to int8 in memory first, and the int32 result is rescaled in
another pass. That is correct and slow — a real Krea 2 block ran 1.6x slower than BF16.

Two kernels replace the passes, and nothing else changes:

* ``_quantize_rows_kernel`` — one program per row: the absolute maximum, then the quantised values,
  in two sweeps over the row instead of eight.
* ``_int_matmul_kernel`` — ``x @ W^T`` with int8 tensor-core dots accumulated in int32, reading the
  weight in its stored ``[N, K]`` layout (no transposed copy) and, for W4A4, unpacking the two
  signed nibbles of each byte in registers; the scale, bias and output cast are its epilogue.

**The arithmetic is the port's, bit for bit.** Every rounding step of
``comfy_quant.quantize_int8_rowwise`` / ``_int4_values_rowwise`` / ``_scale_accumulators`` is
reproduced in the same order and precision (division rounded to the activation dtype before
``rint``; ``(x_scale * w_scale)`` before the multiply; cast, then bias), and the int32 accumulation
is exact in any order. The ConvRot rotation stays the port's own cuBLAS matmul. So a picture is the
same with or without Triton, which is what lets these kernels stand in silently: on first use a
self-test compares them with the port on random data and they are switched off unless every output
is identical. ``XIRAI_QUANT_TRITON=0`` switches them off by hand.

Structure (row-wise quantiser, grouped-M int8 matmul with a fused epilogue) follows comfy-kitchen
0.2.31's Triton backend, ``backends/triton/quantization.py``, Apache-2.0, which credits
dxqb/OneTrainer; see ``backend/resources/comfy-quant/NOTICE.md``. Its arithmetic differs from
comfy-kitchen's eager reference and is not used here.
"""

from __future__ import annotations

import logging
import os

import torch

try:
    from . import comfy_quant as _cq
except ImportError:  # pragma: no cover - flat import when run from backend/
    import comfy_quant as _cq

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _IMPORT_ERROR = None
except Exception as error:  # noqa: BLE001 - any failure means "no Triton"; the port runs instead
    triton = None
    tl = None
    libdevice = None
    _IMPORT_ERROR = error

LOGGER = logging.getLogger(__name__)
MIN_CAPABILITY = (8, 0)
_ROUND_NONE, _ROUND_BF16, _ROUND_FP16 = 0, 1, 2
_ROUNDING = {torch.float32: _ROUND_NONE, torch.bfloat16: _ROUND_BF16, torch.float16: _ROUND_FP16}

if triton is not None:

    @triton.jit
    def _round_to(value, MODE: tl.constexpr):
        """Round float32 values to the activation dtype and back, as a torch op in that dtype does."""
        if MODE == 1:
            return value.to(tl.bfloat16).to(tl.float32)
        elif MODE == 2:
            return value.to(tl.float16).to(tl.float32)
        else:
            return value

    @triton.jit
    def _quantize_rows_kernel(
        x_ptr, q_ptr, s_ptr, K, stride_x, stride_q, tiny, reciprocal,
        BLOCK: tl.constexpr, INT4: tl.constexpr, MODE: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_row = x_ptr + row.to(tl.int64) * stride_x
        q_row = q_ptr + row.to(tl.int64) * stride_q
        peak = tl.zeros([BLOCK], dtype=tl.float32)
        for start in range(0, K, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            x = tl.load(x_row + offsets, mask=offsets < K, other=0.0).to(tl.float32)
            peak = tl.maximum(peak, tl.abs(x))
        amax = tl.max(peak, axis=0)
        if INT4:
            # _int4_values_rowwise: amax.clamp(min=1e-10) / 7, each step in the activation dtype.
            # PyTorch divides by a Python number as a multiply by its float32 reciprocal.
            scale = _round_to(tl.maximum(amax, 1e-10), MODE)
            scale = _round_to(libdevice.mul_rn(scale, reciprocal), MODE)
            divisor = scale
            low = -7.0
            high = 7.0
        else:
            # quantize_int8_rowwise: float32 amax / 127 clamped to 1e-30, divided in the activation
            # dtype after `_scale_for_math` cast it there (0 becomes that dtype's smallest normal).
            scale = tl.maximum(libdevice.mul_rn(amax, reciprocal), 1e-30)
            divisor = _round_to(scale, MODE)
            divisor = tl.where(divisor == 0.0, tiny, divisor)
            low = -128.0
            high = 127.0
        for start in range(0, K, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            mask = offsets < K
            x = tl.load(x_row + offsets, mask=mask, other=0.0).to(tl.float32)
            quantised = libdevice.rint(_round_to(libdevice.div_rn(x, divisor), MODE))
            quantised = tl.minimum(tl.maximum(quantised, low), high)
            tl.store(q_row + offsets, quantised.to(tl.int8), mask=mask)
        tl.store(s_ptr + row, scale)

    @triton.jit
    def _int_matmul_kernel(
        a_ptr, b_ptr, c_ptr, xs_ptr, ws_ptr, bias_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
        PACKED: tl.constexpr, WS_PER_ROW: tl.constexpr, HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        width = GROUP_M * grid_n
        group = pid // width
        first_m = group * GROUP_M
        group_rows = min(grid_m - first_m, GROUP_M)
        pid_m = first_m + (pid % width) % group_rows
        pid_n = (pid % width) // group_rows

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=mask_m[:, None] & (offs_k < K)[None, :], other=0,
            )
            if PACKED:
                # Two signed nibbles per byte, the even column in the low one (pack_int4).
                offs_p = k0 // 2 + tl.arange(0, BLOCK_K // 2)
                packed = tl.load(
                    b_ptr + offs_n[:, None] * stride_bn + offs_p[None, :] * stride_bk,
                    mask=mask_n[:, None] & (offs_p < K // 2)[None, :], other=0,
                )
                low = (packed << 4) >> 4
                high = packed >> 4
                b = tl.trans(tl.interleave(low, high))
            else:
                b = tl.load(
                    b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k < K)[:, None] & mask_n[None, :], other=0,
                )
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)

        # _scale_accumulators: acc * (x_scale * w_scale) in float32, cast, then the bias. The
        # explicitly rounded operations keep the compiler from fusing the multiply and the bias add
        # into one FMA, which rounds once where PyTorch rounds twice (visible on float32 outputs).
        xs = tl.load(xs_ptr + offs_m, mask=mask_m, other=0.0)
        if WS_PER_ROW:
            ws = tl.load(ws_ptr + offs_n, mask=mask_n, other=0.0)
            scale = libdevice.mul_rn(xs[:, None], ws[None, :])
        else:
            ws = tl.load(ws_ptr)
            scale = libdevice.mul_rn(xs[:, None], ws)
        out = libdevice.mul_rn(acc.to(tl.float32), scale).to(c_ptr.dtype.element_ty)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(c_ptr.dtype.element_ty)
            out = libdevice.add_rn(out.to(tl.float32), bias.to(tl.float32)[None, :]).to(c_ptr.dtype.element_ty)
        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            out, mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _e4m3_to_f32(bits):
        """An E4M3FN byte (int32 0..255) to its exact float32 value, as PyTorch's cast gives it. The
        sign goes straight into the bits: negating a float compiles to ``0 - x``, which loses -0.0."""
        exponent = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        normal = ((exponent + 120) << 23) | (mantissa << 20)
        subnormal = (mantissa.to(tl.float32) * 0.001953125).to(tl.int32, bitcast=True)  # m * 2**-9, exact
        magnitude = tl.where(exponent == 0, subnormal, normal)
        magnitude = tl.where((exponent == 15) & (mantissa == 7), 0x7FC00000, magnitude)
        return (magnitude | ((bits >> 7) << 31)).to(tl.float32, bitcast=True)

    @triton.jit
    def _tiled_scale_offset(row, block, tiles_per_row):
        """Where cuBLAS's 128x4 tiled layout (``comfy_quant.to_blocked``) keeps block ``block`` of
        row ``row``: whole 128x4 tiles in row-major order, each stored as 32 rows of four 4-wide
        groups, so row ``i`` of a tile is ``(i % 32) * 16 + (i // 32) * 4``."""
        tile = (row // 128) * tiles_per_row + block // 4
        return tile * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + block % 4

    @triton.jit
    def _nvfp4_decode_kernel(
        q_ptr, bs_ptr, gs_ptr, table_ptr, out_ptr, rows, cols, stride_q, stride_out, tiles_per_row,
        MODE: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        # comfy_quant.dequantize_nvfp4, element by element: the high nibble first; the global scale
        # cast to the output dtype, times the block scale, rounded there; then the E2M1 value times
        # that, rounded again — each product exact in float32 before its one rounding.
        r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
        mask = (r < rows) & (c < cols)
        byte = tl.load(q_ptr + r.to(tl.int64) * stride_q + c // 2, mask=mask, other=0).to(tl.int32) & 0xFF
        code = tl.where(c % 2 == 0, byte >> 4, byte & 0xF)
        value = tl.load(table_ptr + code, mask=mask, other=0.0)
        bits = tl.load(bs_ptr + _tiled_scale_offset(r, c // 16, tiles_per_row), mask=mask, other=0).to(tl.int32) & 0xFF
        total = _round_to(libdevice.mul_rn(_round_to(tl.load(gs_ptr), MODE), _e4m3_to_f32(bits)), MODE)
        out = _round_to(libdevice.mul_rn(value, total), MODE)
        tl.store(out_ptr + r.to(tl.int64) * stride_out + c, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _mxfp8_decode_kernel(
        q_ptr, bs_ptr, out_ptr, rows, cols, stride_q, stride_out, tiles_per_row,
        BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        # comfy_quant.dequantize_mxfp8: the fp8 value in float32 times 2**(e - 127) (E8M0; 0 is
        # zero), one float32 product, then the cast to the output dtype.
        r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
        mask = (r < rows) & (c < cols)
        bits = tl.load(q_ptr + r.to(tl.int64) * stride_q + c, mask=mask, other=0).to(tl.int32) & 0xFF
        e = tl.load(bs_ptr + _tiled_scale_offset(r, c // 32, tiles_per_row), mask=mask, other=0).to(tl.int32) & 0xFF
        out = libdevice.mul_rn(_e4m3_to_f32(bits), (e << 23).to(tl.float32, bitcast=True))
        tl.store(out_ptr + r.to(tl.int64) * stride_out + c, out.to(out_ptr.dtype.element_ty), mask=mask)


# PyTorch's CUDA true division by a CPU scalar multiplies by the float32 reciprocal instead
# (`div_true_kernel_cuda`); `amax / 127.0` and `amax / 7` in the port are therefore these products.
_RECIPROCAL_127 = (torch.tensor(1.0, dtype=torch.float32) / torch.tensor(127.0, dtype=torch.float32)).item()
_RECIPROCAL_7 = (torch.tensor(1.0, dtype=torch.float32) / torch.tensor(7.0, dtype=torch.float32)).item()


def _quantize_rows(x: torch.Tensor, int4: bool) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = x.shape
    q = torch.empty((rows, columns), dtype=torch.int8, device=x.device)
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    if rows:
        tiny = float(torch.finfo(x.dtype).tiny)
        reciprocal = _RECIPROCAL_7 if int4 else _RECIPROCAL_127
        block = min(4096, max(128, triton.next_power_of_2(columns)))
        _quantize_rows_kernel[(rows,)](
            x, q, scale, columns, x.stride(0), q.stride(0), tiny, reciprocal,
            BLOCK=block, INT4=int4, MODE=_ROUNDING[x.dtype], num_warps=8 if block >= 2048 else 4,
        )
    return q, scale


def _int_matmul(q: torch.Tensor, weight: torch.Tensor, x_scale, weight_scale, bias, out_dtype, packed: bool):
    rows, depth = q.shape
    out_features = weight.shape[0]
    out = torch.empty((rows, out_features), dtype=out_dtype, device=q.device)
    if rows == 0:
        return out
    per_row = weight_scale.numel() != 1
    bias_arg = bias.to(device=q.device).contiguous() if bias is not None else x_scale

    config = matmul_config(rows, packed)
    grid = (triton.cdiv(rows, config["BLOCK_M"]) * triton.cdiv(out_features, config["BLOCK_N"]),)
    _int_matmul_kernel[grid](
        q, weight, out, x_scale, weight_scale, bias_arg,
        rows, out_features, depth,
        q.stride(0), q.stride(1), weight.stride(0), weight.stride(1), out.stride(0), out.stride(1),
        PACKED=packed, WS_PER_ROW=per_row, HAS_BIAS=bias is not None, **config,
    )
    return out


# One tile shape per size class, chosen by measurement on an RTX 4060 (Temp/p9-claude/triton_tune.py)
# rather than autotuned: autotuning compiles every candidate on first use in every process.
MATMUL_CONFIGS = {
    "small": {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 4, "num_warps": 4, "num_stages": 4},
    "medium": {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
    "large": {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 8, "num_stages": 3},
}


def matmul_config(rows: int, packed: bool) -> dict:
    if rows <= 32:
        return MATMUL_CONFIGS["small"]
    if rows <= 256:
        return MATMUL_CONFIGS["medium"]
    return MATMUL_CONFIGS["large"]


def int8_linear(x, weight, weight_scale, bias=None, out_dtype=None, convrot=False,
                convrot_groupsize=_cq.CONVROT_DEFAULT_GROUPSIZE):
    """:func:`comfy_quant.int8_linear` on the Triton kernels; identical output."""
    out_dtype = out_dtype or x.dtype
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(f"Input and weight inner dimensions must match, got {x.shape[-1]} and {weight.shape[-1]}")
    weight = weight.to(device=x.device).contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1).contiguous()
    if weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {weight_scale.numel()} values "
            f"for weight shape {tuple(weight.shape)}"
        )
    if convrot:
        x = _cq.rotate_activation(x, _cq.build_hadamard(convrot_groupsize, x.device, x.dtype), convrot_groupsize)
    shape = x.shape
    q, x_scale = _quantize_rows(x.reshape(-1, shape[-1]).contiguous(), int4=False)
    out = _int_matmul(q, weight, x_scale, weight_scale, bias, out_dtype, packed=False)
    return out.reshape(*shape[:-1], weight.shape[0])


def convrot_w4a4_linear(x, qweight, wscales, bias=None, convrot_groupsize=_cq.CONVROT_DEFAULT_GROUPSIZE,
                        quant_group_size=_cq.W4A4_QUANT_GROUP_SIZE, linear_dtype="int4"):
    """:func:`comfy_quant.convrot_w4a4_linear` on the Triton kernels; identical output."""
    if quant_group_size != _cq.W4A4_QUANT_GROUP_SIZE:
        raise ValueError(f"int4 MMA kernel requires quant_group_size {_cq.W4A4_QUANT_GROUP_SIZE}")
    if x.shape[-1] != qweight.shape[-1] * 2:
        raise ValueError(f"Input K={x.shape[-1]} does not match qweight K={qweight.shape[-1] * 2}")
    shape = x.shape
    x2d = x.reshape(-1, shape[-1]).contiguous()
    x_rot = _cq.rotate_activation(x2d, _cq.build_hadamard(convrot_groupsize, x2d.device, x2d.dtype), convrot_groupsize)
    int4 = _cq.w4a4_activation_bits(x2d.device, linear_dtype, x2d.shape[0]) == 4
    q, x_scale = _quantize_rows(x_rot.contiguous(), int4=int4)
    weight = qweight.to(device=x.device).contiguous()
    scales = wscales.to(device=x.device, dtype=torch.float32).reshape(-1).contiguous()
    out = _int_matmul(q, weight, x_scale, scales, bias, x.dtype, packed=True)
    return out.reshape(*shape[:-1], qweight.shape[0])


# -- NVFP4 / MXFP8 decode ------------------------------------------------------------------------
# These layouts still run as "decode to the compute dtype, then a full-precision matmul" (as ComfyUI
# does below SM 10.0, and with a LoRA everywhere). The port decodes through an int64 table lookup, a
# stack, an unswizzle and two broadcast multiplies per layer and per step; one kernel writes the same
# weight directly. The matmul after it is unchanged, so outputs stay identical.

DECODE_DTYPES = {torch.bfloat16: _ROUND_BF16, torch.float16: _ROUND_FP16, torch.float32: _ROUND_NONE}
_DECODE_BLOCK_R, _DECODE_BLOCK_C = 16, 256
# Triton links libdevice with flush-to-zero on by default, so `mul_rn` would turn a float32 product
# below 2**-126 into zero where PyTorch keeps the subnormal (a tiny E8M0 scale reaches it).
_DECODE_OPTIONS = {"enable_reflect_ftz": False}
_FP4_TABLES: dict = {}


def _fp4_table(device) -> torch.Tensor:
    table = _FP4_TABLES.get(device)
    if table is None:
        table = torch.tensor(_cq.FP4_E2M1_VALUES, dtype=torch.float32, device=device)
        _FP4_TABLES[device] = table
    return table


def dequantize_nvfp4(qdata, global_scale, block_scale, dtype, rows, cols):
    """:func:`comfy_quant.dequantize_nvfp4` in one kernel; identical output. The caller has already
    checked the shapes; ``dtype`` is one of :data:`DECODE_DTYPES`."""
    qdata = qdata.view(torch.uint8).contiguous()
    storage_rows, storage_cols = qdata.shape
    _, padded_blocks = _cq.swizzled_scale_shape(storage_rows, storage_cols * 2 // _cq.NVFP4_GROUP_SIZE)
    out = torch.empty((rows, cols), dtype=dtype, device=qdata.device)
    if rows and cols:
        grid = (triton.cdiv(rows, _DECODE_BLOCK_R), triton.cdiv(cols, _DECODE_BLOCK_C))
        _nvfp4_decode_kernel[grid](
            qdata, block_scale.to(device=qdata.device).view(torch.uint8).contiguous().reshape(-1),
            global_scale.to(device=qdata.device, dtype=torch.float32).reshape(1), _fp4_table(qdata.device),
            out, rows, cols, qdata.stride(0), out.stride(0), padded_blocks // 4,
            MODE=DECODE_DTYPES[dtype], BLOCK_R=_DECODE_BLOCK_R, BLOCK_C=_DECODE_BLOCK_C, **_DECODE_OPTIONS,
        )
    return out


def dequantize_mxfp8(qdata, block_scale, dtype, rows, cols):
    """:func:`comfy_quant.dequantize_mxfp8` in one kernel; identical output."""
    qdata = qdata.view(torch.uint8).contiguous()
    storage_rows, storage_cols = qdata.shape
    _, padded_blocks = _cq.swizzled_scale_shape(storage_rows, storage_cols // _cq.MXFP8_GROUP_SIZE)
    out = torch.empty((rows, cols), dtype=dtype, device=qdata.device)
    if rows and cols:
        grid = (triton.cdiv(rows, _DECODE_BLOCK_R), triton.cdiv(cols, _DECODE_BLOCK_C))
        _mxfp8_decode_kernel[grid](
            qdata, block_scale.to(device=qdata.device).view(torch.uint8).contiguous().reshape(-1),
            out, rows, cols, qdata.stride(0), out.stride(0), padded_blocks // 4,
            BLOCK_R=_DECODE_BLOCK_R, BLOCK_C=_DECODE_BLOCK_C, **_DECODE_OPTIONS,
        )
    return out


# -- availability ------------------------------------------------------------------------------

_STATE: dict = {"checked": set(), "failed": {}}


def _self_test(device) -> str:
    """Empty when the kernels reproduce the port exactly on this device, else the reason."""
    generator = torch.Generator(device="cpu").manual_seed(20260927)

    def rand(*shape):
        return torch.randn(*shape, generator=generator).to(device)

    for dtype in (torch.bfloat16, torch.float16):
        x = (rand(3, 37, 512) * 2).to(dtype)
        x[..., ::41] *= 30
        weight = rand(72, 512) * 0.05
        rotated = _cq.rotate_weight(weight, _cq.build_hadamard(256, device=device), 256)
        q8, s8 = _cq.quantize_int8_rowwise(rotated)
        q4, s4 = _cq.quantize_int4_rowwise(rotated)
        bias = rand(72).to(dtype)
        cases = (
            ("int8 convrot", lambda f: f.int8_linear(x, q8, s8, bias, dtype, True, 256)),
            ("int8 scalar", lambda f: f.int8_linear(x, q8, torch.tensor(0.01, device=device), None, dtype)),
            ("w4a4", lambda f: f.convrot_w4a4_linear(x, q4, s4, bias, 256)),
            ("w4a8", lambda f: f.convrot_w4a4_linear(x, q4, s4, None, 256, linear_dtype="int8")),
        )
        port = _PortOps()
        for name, run in cases:
            expected, got = run(port), run(_TritonOps())
            if not torch.equal(expected, got):
                return f"{name} ({dtype}) differs from the port by up to {(expected.float() - got.float()).abs().max().item():.3g}"
    return ""


class _PortOps:
    """The port, with any installed comfy-kitchen and these kernels out of the way."""

    def int8_linear(self, *args, **kwargs):
        with _cq.backend_override("port"):
            return _cq.int8_linear(*args, **kwargs)

    def convrot_w4a4_linear(self, *args, **kwargs):
        with _cq.backend_override("port"):
            return _cq.convrot_w4a4_linear(*args, **kwargs)


class _TritonOps:
    int8_linear = staticmethod(int8_linear)
    convrot_w4a4_linear = staticmethod(convrot_w4a4_linear)


_DECODE_STATE: dict = {"checked": set(), "failed": {}}
_INTEGER_KERNELS = "Built-in Triton quantised kernels"
_DECODERS = "Built-in Triton NVFP4/MXFP8 decoders"


def _decode_self_test(device) -> str:
    """Empty when both decoders reproduce the port exactly on this device, else the reason. The data
    covers every E2M1 code, E4M3 zero/subnormal/normal scales, and rows and blocks that need the
    tiled layout's padding (200 rows of 256, 5 blocks of 8) with trailing columns trimmed."""
    generator = torch.Generator(device="cpu").manual_seed(20260928)

    def random_bytes(count, low=0, high=256):
        values = torch.randint(low, high, (count,), generator=generator, dtype=torch.int32)
        values[(values == 0x7F) | (values == 0xFF)] = 0x38  # no NaN scales; they never occur in files
        return values.to(torch.uint8).to(device)

    rows = 200
    padded_rows, padded_blocks = _cq.swizzled_scale_shape(rows, 5)
    qdata = random_bytes(rows * 40).reshape(rows, 40)
    block_scale = random_bytes(padded_rows * padded_blocks).view(torch.float8_e4m3fn).reshape(padded_rows, padded_blocks)
    global_scale = torch.tensor(0.0123, dtype=torch.float32, device=device)
    fp8 = random_bytes(rows * 128).view(torch.float8_e4m3fn).reshape(rows, 128)
    padded_rows, padded_blocks = _cq.swizzled_scale_shape(rows, 4)
    e8m0 = random_bytes(padded_rows * padded_blocks, 100, 150)
    for dtype in DECODE_DTYPES:
        with _cq.backend_override("port"):
            expected = (_cq.dequantize_nvfp4(qdata, global_scale, block_scale, dtype, rows, 72),
                        _cq.dequantize_mxfp8(fp8, e8m0, dtype, rows, 100))
        got = (dequantize_nvfp4(qdata, global_scale, block_scale, dtype, rows, 72),
               dequantize_mxfp8(fp8, e8m0, dtype, rows, 100))
        for name, want, have in zip(("NVFP4", "MXFP8"), expected, got):
            if not torch.equal(want, have):
                return f"{name} ({dtype}) differs from the port by up to {(want.float() - have.float()).abs().max().item():.3g}"
    return ""


def _device_index(device):
    """``(reason, index)``: why no kernel here can run on ``device``, or the CUDA index it names."""
    if os.environ.get("XIRAI_QUANT_TRITON", "1") == "0":
        return "disabled by XIRAI_QUANT_TRITON=0", None
    if triton is None:
        return f"Triton is not installed ({_IMPORT_ERROR})", None
    if not torch.cuda.is_available():
        return "no CUDA device", None
    index = torch.device(device if device is not None else "cuda").index
    index = index if index is not None else torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    if tuple(capability) < MIN_CAPABILITY:
        return f"SM {capability[0]}.{capability[1]} is below SM 8.0", None
    return "", index


def _status(state, self_test, what, device=None) -> dict:
    reason, index = _device_index(device)
    if reason:
        return {"available": False, "reason": reason}
    if index in state["failed"]:
        return {"available": False, "reason": state["failed"][index]}
    if index not in state["checked"]:
        try:
            reason = self_test(torch.device("cuda", index))
        except Exception as error:  # noqa: BLE001 - a kernel that cannot compile or run is off
            reason = f"self-test failed: {type(error).__name__}: {error}"
        state["checked"].add(index)
        if reason:
            state["failed"][index] = reason
            LOGGER.warning("%s disabled: %s", what, reason)
            return {"available": False, "reason": reason}
    return {"available": True, "reason": ""}


def _peek(state, device=None) -> dict:
    reason, index = _device_index(device)
    if reason:
        return {"available": False, "reason": reason}
    if index in state["failed"]:
        return {"available": False, "reason": state["failed"][index]}
    if index in state["checked"]:
        return {"available": True, "reason": ""}
    return {"available": None, "reason": "verified against the port on first use"}


def _disable(state, what, device, reason: str) -> None:
    index = torch.device(device).index
    index = index if index is not None else torch.cuda.current_device()
    state["failed"][index] = reason
    LOGGER.warning("%s disabled: %s", what, reason)


def status(device=None) -> dict:
    """Whether the integer kernels run on ``device`` (default: the current CUDA device), and why not."""
    return _status(_STATE, _self_test, _INTEGER_KERNELS, device)


def available(device=None) -> bool:
    return status(device)["available"]


def peek(device=None) -> dict:
    """:func:`status` without running the self-test: ``available`` is None when the kernels would be
    tried but have not been verified in this process yet (the health report must not compile)."""
    return _peek(_STATE, device)


def disable(device, reason: str) -> None:
    """Switch the integer kernels off for ``device`` after a failure at run time; the port takes over."""
    _disable(_STATE, _INTEGER_KERNELS, device, reason)


def decode_status(device=None) -> dict:
    """:func:`status` for the NVFP4/MXFP8 decoders, which are verified and switched off separately."""
    return _status(_DECODE_STATE, _decode_self_test, _DECODERS, device)


def decode_available(device=None) -> bool:
    return decode_status(device)["available"]


def decode_peek(device=None) -> dict:
    return _peek(_DECODE_STATE, device)


def disable_decode(device, reason: str) -> None:
    _disable(_DECODE_STATE, _DECODERS, device, reason)
