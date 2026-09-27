"""P9: ComfyUI's quantised layouts — markers, header metadata, kernels and capability gates.

The references in this file are written independently of :mod:`comfy_quant`: the cuBLAS scale
swizzle is checked against NVIDIA's index formula rather than against the reshape that implements
it, and the quantisers that build fixtures are transcriptions of ComfyUI's own (the product has no
quantiser). Evidence against comfy-kitchen 0.2.31 itself and against real Krea 2 layers is in
``D:\\.XAIG\\Temp\\p9-claude`` (``diff_kitchen.py``, ``verify_layers.py``).
"""

import functools
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .. import comfy_quant as cq
except ImportError:
    import comfy_quant as cq


def marker(**conf):
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


@functools.lru_cache(maxsize=None)
def reference_hadamard(size):
    """Sylvester-free construction: entry (i, j) of H4^{(x)n} is the product of H4 entries over
    the base-4 digits of i and j."""
    h4 = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]
    n = round(math.log(size, 4))
    out = torch.empty(size, size, dtype=torch.float64)
    for i in range(size):
        for j in range(size):
            value, a, b = 1, i, j
            for _ in range(n):
                value *= h4[a % 4][b % 4]
                a //= 4
                b //= 4
            out[i, j] = value
    # Digit order does not matter for orthogonality, but it does for equality: kron puts the
    # most significant digit first, and the product above is symmetric in digit order.
    return out / math.sqrt(size)


def swizzled_index(row, col, col_tiles):
    """NVIDIA cuBLAS 1D block-scaling layout: 128x4 tiles, each stored as 32 rows of 16 values."""
    tile = (row // 128) * col_tiles + col // 4
    return tile * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4


def reference_to_blocked(matrix):
    rows, cols = matrix.shape
    row_tiles, col_tiles = -(-rows // 128), -(-cols // 4)
    flat = torch.zeros(row_tiles * col_tiles * 512, dtype=matrix.dtype)
    for r in range(rows):
        for c in range(cols):
            flat[swizzled_index(r, c, col_tiles)] = matrix[r, c]
    return flat.reshape(row_tiles * 128, col_tiles * 4)


def reference_quantize_int8_rowwise(x):
    """ComfyUI eager quantize_int8_rowwise, transcribed."""
    scale = (x.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp(min=1e-30)
    s = scale.to(x.dtype)
    s = torch.where(s == 0, torch.full_like(s, torch.finfo(x.dtype).tiny), s)
    return (x / s).round().clamp(-128, 127).to(torch.int8), scale


def reference_convrot_rotate(w, group):
    h = reference_hadamard(group).to(w.dtype)
    rows, cols = w.shape
    return (w.reshape(rows, cols // group, group) @ h.T).reshape(rows, cols)


FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def reference_nvfp4(values, global_scale):
    """ComfyUI eager quantize_nvfp4 for already-representable inputs: returns packed (hi first)
    bytes and fp8 block scales in the tiled layout."""
    rows, cols = values.shape
    blocks = values.reshape(rows, cols // 16, 16)
    block_scale = (blocks.abs().amax(-1) / 6.0 / global_scale).to(torch.float8_e4m3fn)
    total = global_scale * block_scale.float()
    scaled = blocks / torch.where(total == 0, torch.ones_like(total), total).unsqueeze(-1)
    codes = torch.tensor([FP4.index(float(v)) if float(v) != 0 or math.copysign(1, float(v)) > 0 else 8
                          for v in scaled.reshape(-1)], dtype=torch.uint8).reshape(rows, cols)
    packed = (codes[:, 0::2] << 4) | codes[:, 1::2]
    return packed, reference_to_blocked(block_scale.float()).to(torch.float8_e4m3fn)


class MarkerTests(unittest.TestCase):
    def test_convrot_int8_reads_comfys_defaults(self):
        config = cq.parse_layer_config(marker(format="int8_tensorwise", convrot=True), "l")
        self.assertEqual((config.convrot, config.convrot_groupsize, config.label), (True, 256, "ConvRot INT8"))

    def test_the_rotation_may_sit_inside_params(self):
        # `_load_quantized_module` reads both spellings; the marker wins when both are present.
        config = cq.parse_layer_config(
            marker(format="int8_tensorwise", params={"convrot": True, "convrot_groupsize": 64}), "l"
        )
        self.assertEqual((config.convrot, config.convrot_groupsize), (True, 64))
        config = cq.parse_layer_config(
            marker(format="int8_tensorwise", convrot=True, convrot_groupsize=16, params={"convrot_groupsize": 64}), "l"
        )
        self.assertEqual(config.convrot_groupsize, 16)

    def test_w4a4_defaults(self):
        config = cq.parse_layer_config(marker(format="convrot_w4a4"), "l")
        self.assertEqual(
            (config.convrot_groupsize, config.quant_group_size, config.linear_dtype, config.rotated),
            (256, 64, "int4", True),
        )
        self.assertEqual(cq.parse_layer_config(marker(format="convrot_w4a4", linear_dtype="int8"), "l").linear_dtype, "int8")

    def test_markers_serialise_the_way_comfy_writes_them_back(self):
        for conf in (
            {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
            {"format": "convrot_w4a4", "convrot_groupsize": 256},
            {"format": "convrot_w4a4", "convrot_groupsize": 64, "linear_dtype": "int8"},
            {"format": "nvfp4", "full_precision_matrix_mult": True},
            {"format": "mxfp8"},
        ):
            self.assertEqual(cq.parse_layer_config(marker(**conf), "l").marker(), conf)

    def test_a_rotation_on_a_layout_that_cannot_carry_one_is_refused(self):
        # ComfyUI ignores the flag there, which multiplies a rotated weight by an unrotated input.
        for fmt in ("float8_e4m3fn", "nvfp4", "mxfp8"):
            with self.assertRaises(cq.UnsupportedQuantization) as raised:
                cq.parse_layer_config(marker(format=fmt, convrot=True), "l")
            self.assertIn("ConvRot", str(raised.exception))
        with self.assertRaises(cq.UnsupportedQuantization):
            cq.parse_layer_config(marker(format="int8_tensorwise", convrot_groupsize=256), "l")

    def test_a_group_size_without_a_hadamard_matrix_is_refused(self):
        for size in (0, 2, 8, 32, 128, 255):
            with self.assertRaises(cq.UnsupportedQuantization):
                cq.parse_layer_config(marker(format="convrot_w4a4", convrot_groupsize=size), "l")
        for size in (4, 16, 64, 256, 1024):
            cq.parse_layer_config(marker(format="convrot_w4a4", convrot_groupsize=size), "l")

    def test_unknown_fields_and_formats_are_refused_by_name(self):
        with self.assertRaises(cq.UnsupportedQuantization) as raised:
            cq.parse_layer_config(marker(format="int8_tensorwise", smooth=True), "l")
        self.assertIn("smooth", str(raised.exception))
        with self.assertRaises(cq.UnsupportedQuantization) as raised:
            cq.parse_layer_config(marker(format="asym_w4a8_int8"), "l")
        self.assertIn("W4A8", str(raised.exception))
        with self.assertRaises(cq.UnsupportedQuantization) as raised:
            cq.parse_layer_config(marker(format="int4_awq"), "l")
        self.assertIn("int4_awq", str(raised.exception))
        with self.assertRaises(cq.UnsupportedQuantization):
            cq.parse_layer_config(marker(format="convrot_w4a4", linear_dtype="fp8"), "l")

    def test_exporter_fields_are_accepted_only_when_they_agree_with_the_layout(self):
        # wikeeyang's FLUX.2 True V3 NVFP4/MXFP8/INT8 builds write these beside the format.
        config = cq.parse_layer_config(
            marker(format="nvfp4", group_size=16, orig_dtype="torch.bfloat16", orig_shape=[4096, 4096]), "l"
        )
        self.assertEqual(config.declared_shape, (4096, 4096))
        self.assertTrue(cq.parse_layer_config(marker(format="int8_tensorwise", per_row=True), "l").per_row)
        for bad in (
            {"format": "nvfp4", "group_size": 32},
            {"format": "mxfp8", "group_size": 16},
            {"format": "float8_e4m3fn", "group_size": 16},
            {"format": "nvfp4", "orig_shape": [4096]},
            {"format": "nvfp4", "orig_dtype": 16},
            {"format": "mxfp8", "per_row": True},
        ):
            with self.assertRaises(cq.UnsupportedQuantization, msg=bad):
                cq.parse_layer_config(marker(**bad), "l")
        shaped = cq.parse_layer_config(marker(format="nvfp4", orig_shape=[16, 32]), "l")
        good = {"weight_scale": (128, 4), "weight_scale_2": ()}
        self.assertEqual(cq.validate_layer(shaped, (16, 16), good, 16, 32, "l", "t"), (16, 32))
        with self.assertRaises(cq.UnsupportedQuantization):
            cq.validate_layer(shaped, (16, 16), good, 16, 16, "l", "t")
        row = cq.parse_layer_config(marker(format="int8_tensorwise", per_row=True), "l")
        with self.assertRaises(cq.UnsupportedQuantization):
            cq.validate_layer(row, (8, 64), {"weight_scale": ()}, None, None, "l", "t")

    def test_the_int8_fast_node_layout_is_refused_by_name(self):
        # `{"convrot": true, "convrot_groupsize": 256, "per_row": true}` with no format: ComfyUI core
        # raises "Unknown quantization format" on it as well.
        with self.assertRaises(cq.UnsupportedQuantization) as raised:
            cq.parse_layer_config(marker(convrot=True, convrot_groupsize=256, per_row=True), "l")
        self.assertIn("INT8-Fast", str(raised.exception))

    def test_malformed_markers_are_refused(self):
        for bad in (torch.tensor([123, 45], dtype=torch.uint8), marker(), b"[1, 2]"):
            with self.assertRaises(cq.UnsupportedQuantization):
                cq.parse_layer_config(bad, "l")
        with self.assertRaises(cq.UnsupportedQuantization):
            cq.parse_layer_config(marker(format="nvfp4", full_precision_matrix_mult="yes"), "l")
        with self.assertRaises(cq.UnsupportedQuantization):
            cq.parse_layer_config(marker(format="int8_tensorwise", convrot=True, convrot_groupsize=64.5), "l")


class HeaderMetadataTests(unittest.TestCase):
    TABLE = {"layers": {"a": {"format": "nvfp4", "full_precision_matrix_mult": False}, "b": {"format": "mxfp8"}}}

    def test_the_layer_table_becomes_marker_entries(self):
        state = {"a.weight": torch.zeros(1), "b.weight": torch.zeros(1)}
        converted = cq.apply_header_quant_metadata(state, {"_quantization_metadata": json.dumps(self.TABLE)})
        self.assertEqual(cq.decode_marker(converted["a.comfy_quant"], "a"), self.TABLE["layers"]["a"])
        self.assertEqual(cq.decode_marker(converted["b.comfy_quant"], "b"), {"format": "mxfp8"})
        self.assertNotIn("a.comfy_quant", state)  # the caller's dict is not mutated

    def test_a_header_entry_replaces_a_tensor_marker_as_in_comfy(self):
        state = {"a.comfy_quant": marker(format="float8_e4m3fn")}
        converted = cq.apply_header_quant_metadata(state, {"_quantization_metadata": json.dumps(self.TABLE)})
        self.assertEqual(cq.decode_marker(converted["a.comfy_quant"], "a")["format"], "nvfp4")

    def test_no_table_leaves_the_state_dict_alone(self):
        state = {"a.weight": torch.zeros(1)}
        self.assertIs(cq.apply_header_quant_metadata(state, {"format": "pt"}), state)

    def test_a_malformed_table_is_refused(self):
        for raw in ("{not json", json.dumps({"layers": []}), json.dumps({"format_version": "1.0"})):
            with self.assertRaises(cq.UnsupportedQuantization):
                cq.apply_header_quant_metadata({}, {"_quantization_metadata": raw})

    def test_a_real_file_is_read_with_its_header_markers(self):
        from safetensors.torch import load_file, save_file

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.safetensors"
            save_file({"a.weight": torch.zeros(2, 2)}, str(path), metadata={
                "_quantization_metadata": json.dumps({"layers": {"a": {"format": "float8_e4m3fn"}}}),
            })
            state = cq.load_safetensors_state_dict(path, load_file)
            self.assertIn("a.comfy_quant", state)
            self.assertEqual(cq.read_safetensors_metadata(path)["_quantization_metadata"][:10], '{"layers":')

    def test_a_stand_in_that_is_not_a_safetensors_file_passes_through(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fake.safetensors"
            path.write_bytes(b"")
            state = cq.load_safetensors_state_dict(path, lambda *_args, **_kwargs: {"x": torch.ones(1)})
            self.assertEqual(list(state), ["x"])
            missing = Path(folder) / "missing.safetensors"
            self.assertEqual(list(cq.load_safetensors_state_dict(missing, lambda *_a, **_k: {"y": 1})), ["y"])


class RotationTests(unittest.TestCase):
    def test_the_hadamard_matrix_matches_the_digit_formula_and_is_orthogonal(self):
        for size in (4, 16, 64, 256):
            h = cq.build_hadamard(size, dtype=torch.float64)
            self.assertTrue(torch.allclose(h, reference_hadamard(size), atol=1e-12))
            self.assertTrue(torch.allclose(h @ h.T, torch.eye(size, dtype=torch.float64), atol=1e-12))
            self.assertTrue(torch.equal(h, h.T))
        with self.assertRaises(ValueError):
            cq.build_hadamard(32)

    def test_rotating_twice_is_the_identity(self):
        w = torch.randn(8, 512, dtype=torch.float64)
        h = cq.build_hadamard(256, dtype=torch.float64)
        self.assertTrue(torch.allclose(cq.rotate_weight(cq.rotate_weight(w, h, 256), h, 256), w, atol=1e-12))

    def test_rotating_both_sides_leaves_the_product_unchanged(self):
        # (x H)(W H^T)^T = x H H W^T... with H symmetric orthogonal, x' W'^T = x W^T.
        x, w = torch.randn(3, 512, dtype=torch.float64), torch.randn(5, 512, dtype=torch.float64)
        h = cq.build_hadamard(64, dtype=torch.float64)
        rotated = cq.rotate_activation(x, h, 64) @ cq.rotate_weight(w, h, 64).T
        self.assertTrue(torch.allclose(rotated, x @ w.T, atol=1e-10))


class Int8Tests(unittest.TestCase):
    def test_the_integer_matmul_is_exact_for_any_shape(self):
        torch.manual_seed(0)
        for m, k, n in ((1, 1, 1), (3, 13, 17), (33, 64, 9), (40, 256, 48)):
            a = torch.randint(-128, 128, (m, k), dtype=torch.int8)
            b = torch.randint(-128, 128, (k, n), dtype=torch.int8)
            self.assertTrue(torch.equal(cq.int8_matmul_accumulate(a, b), a.int() @ b.int()))

    def test_rowwise_quantisation_matches_comfy(self):
        x = torch.randn(7, 64, dtype=torch.bfloat16) * 3
        got, got_scale = cq.quantize_int8_rowwise(x)
        want, want_scale = reference_quantize_int8_rowwise(x)
        self.assertTrue(torch.equal(got, want) and torch.equal(got_scale, want_scale))

    def build(self, rows=16, cols=512, group=256, seed=1):
        torch.manual_seed(seed)
        w = torch.randn(rows, cols) * 0.05
        q, scale = reference_quantize_int8_rowwise(reference_convrot_rotate(w, group))
        return w, q, scale

    def test_convrot_dequantisation_undoes_the_rotation(self):
        w, q, scale = self.build()
        back = cq.dequantize_int8(q, scale, torch.float32, True, 256)
        self.assertLess(((back - w).norm() / w.norm()).item(), 0.02)
        ignored = cq.dequantize_int8(q, scale, torch.float32, False)
        cosine = F.cosine_similarity(ignored.flatten().double(), w.flatten().double(), dim=0).item()
        self.assertLess(abs(cosine), 0.3)  # read as plain INT8, a rotated weight is noise

    def test_the_linear_equals_comfys_eager_arithmetic(self):
        w, q, scale = self.build()
        x = torch.randn(2, 5, 512, dtype=torch.bfloat16)
        bias = torch.randn(16, dtype=torch.bfloat16)
        x_rot = (x.reshape(-1, 2, 256) @ reference_hadamard(256).to(torch.bfloat16)).reshape(10, 512)
        xq, xs = reference_quantize_int8_rowwise(x_rot)
        want = ((xq.int() @ q.int().T).float() * (xs * scale.reshape(1, -1))).to(torch.bfloat16) + bias
        got = cq.int8_linear(x, q, scale, bias, torch.bfloat16, True, 256)
        self.assertTrue(torch.equal(got, want.reshape(2, 5, 16)))

    def test_a_scalar_scale_is_the_plain_tensorwise_layout(self):
        q = torch.randint(-127, 128, (4, 32), dtype=torch.int8)
        x = torch.randn(3, 32)
        got = cq.int8_linear(x, q, torch.tensor(0.01), None, torch.float32)
        self.assertLess(((got - x @ (q.float() * 0.01).T).norm() / got.norm()).item(), 0.02)
        with self.assertRaises(ValueError):
            cq.int8_linear(x, q, torch.ones(3), None, torch.float32)


class W4A4Tests(unittest.TestCase):
    def test_every_nibble_round_trips_signed(self):
        values = torch.arange(-8, 8, dtype=torch.int8).repeat(3, 2)
        packed = cq.pack_int4(values)
        self.assertEqual(tuple(packed.shape), (3, 16))
        self.assertTrue(torch.equal(cq.unpack_int4(packed), values))
        # The even column is the low nibble (svdquant._pack_int4_row_major).
        self.assertEqual(int(cq.pack_int4(torch.tensor([[3, -2]], dtype=torch.int8))[0, 0]) & 0xFF, 0xE3)

    def test_the_linear_is_the_exact_integer_product_of_comfys_quantisation(self):
        torch.manual_seed(3)
        w = torch.randn(24, 256) * 0.05
        w_rot = reference_convrot_rotate(w, 64)
        w_scale = w_rot.abs().amax(-1, keepdim=True).clamp(min=1e-10) / 7
        w_int = (w_rot / w_scale).round().clamp(-7, 7).to(torch.int8)
        qweight = cq.pack_int4(w_int)
        x = torch.randn(9, 256, dtype=torch.bfloat16)
        x_rot = (x.reshape(-1, 4, 64) @ reference_hadamard(64).to(torch.bfloat16)).reshape(9, 256)
        x_scale = x_rot.abs().amax(-1, keepdim=True).clamp(min=1e-10) / 7
        x_int = (x_rot / x_scale).round().clamp(-7, 7).to(torch.int8)
        want = ((x_int.int() @ w_int.int().T).float() * (x_scale.float().reshape(-1, 1) * w_scale.reshape(1, -1))).to(torch.bfloat16)
        got = cq.convrot_w4a4_linear(x, qweight, w_scale.reshape(-1), None, 64)
        self.assertTrue(torch.equal(got, want))
        back = cq.dequantize_convrot_w4a4(qweight, w_scale.reshape(-1), torch.float32, 64)
        self.assertLess(((back - w).norm() / w.norm()).item(), 0.2)

    def test_activation_width_follows_the_pinned_cuda_backend(self):
        self.assertEqual(cq.w4a4_activation_bits("cpu", "int4"), 4)
        self.assertEqual(cq.w4a4_activation_bits("cpu", "int8"), 8)
        for capability, rows, bits in (((8, 9), 1, 4), ((8, 0), 1, 4), ((9, 0), 64, 8), ((12, 0), 64, 8),
                                       ((7, 5), 64, 4), ((7, 5), 4, 8)):
            with mock.patch.object(torch.cuda, "get_device_capability", return_value=capability):
                self.assertEqual(cq.w4a4_activation_bits(torch.device("cuda", 0), "int4", rows), bits, capability)

    def test_only_comfys_quantisation_group_is_accepted(self):
        with self.assertRaises(ValueError):
            cq.convrot_w4a4_linear(torch.zeros(1, 256), torch.zeros(2, 128, dtype=torch.int8), torch.ones(2), None, 64, 32)


class BlockScaleTests(unittest.TestCase):
    def test_the_tile_layout_matches_nvidias_index_formula(self):
        for rows, cols in ((1, 1), (5, 3), (128, 4), (130, 9), (256, 12)):
            matrix = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols) + 1
            blocked = cq.to_blocked(matrix)
            self.assertTrue(torch.equal(blocked, reference_to_blocked(matrix)), (rows, cols))
            self.assertTrue(torch.equal(cq.from_blocked(blocked, rows, cols), matrix))

    def test_a_scale_table_of_the_wrong_size_is_refused(self):
        with self.assertRaises(cq.UnsupportedQuantization):
            cq.from_blocked(torch.zeros(8, 4), 8, 4)

    def test_nvfp4_reads_the_high_nibble_first_and_every_code(self):
        values = torch.tensor([FP4], dtype=torch.float32).repeat(128, 1)
        packed, scales = reference_nvfp4(values, torch.tensor(1.0))
        self.assertEqual(int(packed[0, 0]), (0 << 4) | 1)  # codes 0, 1 -> 0x01: hi first
        got = cq.dequantize_nvfp4(packed, torch.tensor(1.0), scales, torch.float32, 128, 16)
        self.assertTrue(torch.equal(got, values))

    def test_nvfp4_applies_block_and_global_scales_in_the_compute_dtype(self):
        torch.manual_seed(4)
        codes = torch.randint(0, 16, (130, 64))
        codes[:, ::16] = 7  # every block holds a 6.0, so its scale is exactly its factor
        base = torch.tensor(FP4)[codes]
        factors = torch.tensor([0.5, 2.0, 4.0, 1.5]).repeat_interleave(16).reshape(1, 64)
        values = base * factors
        packed, scales = reference_nvfp4(values * 0.25, torch.tensor(0.25))
        for dtype in (torch.float32, torch.bfloat16):
            got = cq.dequantize_nvfp4(packed, torch.tensor(0.25), scales, dtype, 130, 64)
            want = (torch.tensor(FP4, dtype=dtype)[codes] * (torch.tensor(0.25, dtype=dtype) * (factors / 1).to(dtype)))
            self.assertTrue(torch.equal(got, want.to(dtype)), dtype)

    def test_nvfp4_trims_padding_back_to_the_layer(self):
        values = torch.full((16, 32), 6.0)
        packed, scales = reference_nvfp4(values, torch.tensor(1.0))
        self.assertEqual(tuple(cq.dequantize_nvfp4(packed, torch.tensor(1.0), scales, torch.float32, 10, 30).shape), (10, 30))
        with self.assertRaises(cq.UnsupportedQuantization):
            cq.dequantize_nvfp4(packed, torch.tensor(1.0), scales, torch.float32, 17, 32)

    def test_mxfp8_decodes_e8m0_with_zero_as_zero(self):
        exponents = torch.tensor([[127, 128, 126, 0]], dtype=torch.uint8)
        self.assertEqual(cq.e8m0_to_f32(exponents).tolist(), [[1.0, 2.0, 0.5, 0.0]])
        weight = torch.full((1, 128), 3.0).to(torch.float8_e4m3fn)
        got = cq.dequantize_mxfp8(weight, cq.to_blocked(exponents), torch.float32, 1, 128)
        want = torch.tensor([3.0] * 32 + [6.0] * 32 + [1.5] * 32 + [0.0] * 32).reshape(1, 128)
        self.assertTrue(torch.equal(got, want))

    def test_mxfp8_refuses_float_block_scales(self):
        weight = torch.ones(1, 32).to(torch.float8_e4m3fn)
        with self.assertRaises(cq.UnsupportedQuantization):
            cq.dequantize_mxfp8(weight, cq.to_blocked(torch.ones(1, 1)), torch.float32, 1, 32)


class ValidationTests(unittest.TestCase):
    def test_layouts_check_their_scales_and_group_sizes(self):
        int8 = cq.LayerQuantConfig("int8_tensorwise", convrot=True)
        self.assertEqual(cq.validate_layer(int8, (8, 512), {"weight_scale": (8, 1)}, None, None, "l", "t"), (8, 512))
        self.assertEqual(cq.validate_layer(int8, (8, 512), {"weight_scale": ()}, None, None, "l", "t"), (8, 512))
        for scale, weight in (((4,), (8, 512)), ((8, 1), (8, 300))):
            with self.assertRaises(cq.UnsupportedQuantization):
                cq.validate_layer(int8, weight, {"weight_scale": scale}, None, None, "l", "t")
        w4 = cq.parse_layer_config(marker(format="convrot_w4a4"), "l")
        self.assertEqual(cq.validate_layer(w4, (8, 128), {"weight_scale": (8,)}, None, None, "l", "t"), (8, 256))
        with self.assertRaises(cq.UnsupportedQuantization):
            cq.validate_layer(w4, (8, 128), {"weight_scale": (8, 4)}, None, None, "l", "t")
        nvfp4 = cq.LayerQuantConfig("nvfp4")
        good = {"weight_scale": (128, 4), "weight_scale_2": ()}
        self.assertEqual(cq.validate_layer(nvfp4, (16, 16), good, 10, 30, "l", "t"), (10, 30))
        with self.assertRaises(cq.UnsupportedQuantization) as raised:
            cq.validate_layer(nvfp4, (16, 16), {"weight_scale": (16, 2), "weight_scale_2": ()}, None, None, "l", "t")
        self.assertIn("nvfp4", str(raised.exception))


class CapabilityTests(unittest.TestCase):
    def test_gates_follow_comfys_model_management(self):
        cases = {None: (False, False, False), (7, 5): (False, False, False), (8, 6): (False, False, False),
                 (8, 9): (True, False, False), (9, 0): (True, False, False), (10, 0): (True, True, True),
                 (12, 0): (True, True, True)}
        for capability, (fp8, nvfp4, mxfp8) in cases.items():
            with mock.patch.object(cq, "_cuda_capability", return_value=capability):
                self.assertEqual(
                    (cq.supports_fp8_compute(), cq.supports_nvfp4_compute(), cq.supports_mxfp8_compute()),
                    (fp8, nvfp4, mxfp8), capability,
                )

    def test_without_comfy_kitchen_the_ports_run_and_say_so(self):
        cq.kitchen_status.cache_clear()
        try:
            with mock.patch.dict(sys.modules, {"comfy_kitchen": None}), \
                    mock.patch.dict(os.environ, {"XIRAI_QUANT_TRITON": "0"}):
                status = cq.kitchen_status()
                self.assertFalse(status["available"])
                self.assertIn("not installed", status["reason"])
                self.assertIsNone(cq.kitchen_ops())
                report = cq.quantization_capabilities()
                self.assertIn("ported kernel", report["formats"]["ConvRot INT8"]["execution"])
                self.assertEqual(report["refused"], {"asym_w4a8_int8": "ConvRot W4A8 (asym_w4a8_int8)"})
        finally:
            cq.kitchen_status.cache_clear()

    def test_the_model_status_names_the_path_each_layout_takes(self):
        with mock.patch.object(cq, "_cuda_capability", return_value=(8, 9)), \
                mock.patch.object(cq, "kitchen_ops", return_value=None):
            self.assertIn("emulated", cq.execution_path(cq.LayerQuantConfig("nvfp4")))
            self.assertIn("emulated", cq.execution_path(cq.LayerQuantConfig("mxfp8")))
            self.assertIn("ConvRot W8A8", cq.execution_path(cq.LayerQuantConfig("int8_tensorwise", convrot=True)))
            self.assertIn("full-precision", cq.execution_path(cq.LayerQuantConfig("int8_tensorwise", True, True)))
            self.assertIn("full-precision", cq.execution_path(cq.LayerQuantConfig("int8_tensorwise", convrot=True), lora=True))


if __name__ == "__main__":
    unittest.main()
