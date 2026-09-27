"""Tests for running a quantised checkpoint without expanding it first."""

import json
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .. import comfy_quant
    from ..comfy_quant import to_blocked
    from ..quantized_linear import (
        FP4_E2M1_VALUES,
        MXFP8_GROUP_SIZE,
        NVFP4_GROUP_SIZE,
        QuantizedLinear,
        UnsupportedQuantization,
        apply_quantized_linears,
        dequantize_mxfp8,
        dequantize_nvfp4,
        dequantize_scalar_scale,
        expand_quantized_layers,
        logical_shape_view,
        quantized_state_dict,
        quantized_weight_bytes,
        scan_quantized_layers,
    )
except ImportError:
    import comfy_quant
    from comfy_quant import to_blocked
    from quantized_linear import (
        FP4_E2M1_VALUES,
        MXFP8_GROUP_SIZE,
        NVFP4_GROUP_SIZE,
        QuantizedLinear,
        UnsupportedQuantization,
        apply_quantized_linears,
        dequantize_mxfp8,
        dequantize_nvfp4,
        dequantize_scalar_scale,
        expand_quantized_layers,
        logical_shape_view,
        quantized_state_dict,
        quantized_weight_bytes,
        scan_quantized_layers,
    )


def marker(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def comfy_quant_marker(fmt: str) -> torch.Tensor:
    return marker({"format": fmt})


def fp8_layer(rows=8, columns=16, scale=0.0125, suffix=".weight_scale", prefix="block"):
    source = torch.randn(rows, columns)
    weight = (source / scale).to(torch.float8_e4m3fn)
    return {
        f"{prefix}.weight": weight,
        f"{prefix}{suffix}": torch.tensor(scale, dtype=torch.float32),
    }


class ScanTests(unittest.TestCase):
    def test_finds_a_legacy_fp8_layer_by_its_scale_and_storage(self):
        state = fp8_layer()
        layers = scan_quantized_layers(state)
        self.assertEqual(set(layers), {"block"})
        self.assertEqual(layers["block"].format, "float8_e4m3fn")
        self.assertEqual(layers["block"].scale_keys["scale_weight"], "block.weight_scale")

    def test_accepts_the_other_spelling_of_the_same_scale(self):
        layers = scan_quantized_layers(fp8_layer(suffix=".scale_weight"))
        self.assertEqual(layers["block"].scale_keys["scale_weight"], "block.scale_weight")

    def test_a_marker_names_the_format_even_when_the_storage_could_be_guessed(self):
        state = fp8_layer()
        state["block.comfy_quant"] = comfy_quant_marker("float8_e4m3fn")
        layers = scan_quantized_layers(state)
        self.assertEqual(layers["block"].format, "float8_e4m3fn")
        self.assertIn("block.comfy_quant", layers["block"].dropped_keys)

    def test_an_unscaled_fp8_weight_is_not_a_quantised_layer(self):
        # Without a scale there is nothing to defer, and treating it as quantised would invent one.
        state = {"block.weight": torch.randn(4, 4).to(torch.float8_e4m3fn)}
        self.assertEqual(scan_quantized_layers(state), {})

    def test_a_plain_checkpoint_scans_clean(self):
        self.assertEqual(scan_quantized_layers({"block.weight": torch.randn(4, 4)}), {})

    def test_an_unknown_format_is_refused_by_name(self):
        state = {"block.weight": torch.zeros(4, 4, dtype=torch.uint8), "block.comfy_quant": comfy_quant_marker("int4_awq")}
        with self.assertRaises(UnsupportedQuantization) as raised:
            scan_quantized_layers(state, "test checkpoint")
        self.assertIn("int4_awq", str(raised.exception))

    def test_a_rotated_int8_weight_is_read_with_its_rotation(self):
        """Krea 2's INT8 build: ``{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}``.

        Before P9 this was refused, because reading only ``format`` would multiply a weight stored
        in a rotated basis by an unrotated activation. The rotation is now carried into the layer,
        which rotates the activation the way comfy-kitchen's ``int8_linear`` does.
        """
        state = {
            "block.weight": torch.zeros(4, 256, dtype=torch.int8),
            "block.weight_scale": torch.full((4, 1), 0.02),
            "block.comfy_quant": marker({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}),
        }
        config = scan_quantized_layers(state, "test checkpoint")["block"].config
        self.assertEqual((config.format, config.convrot, config.convrot_groupsize), ("int8_tensorwise", True, 256))

    def test_a_rotation_the_input_width_cannot_hold_is_refused(self):
        state = {
            "block.weight": torch.zeros(4, 100, dtype=torch.int8),
            "block.weight_scale": torch.full((4, 1), 0.02),
            "block.comfy_quant": marker({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64}),
        }
        with self.assertRaises(UnsupportedQuantization) as raised:
            scan_quantized_layers(state, "test checkpoint")
        self.assertIn("ConvRot", str(raised.exception))

    def test_an_unknown_marker_field_is_refused_rather_than_ignored(self):
        state = fp8_layer()
        state["block.comfy_quant"] = marker({"format": "float8_e4m3fn", "smoothing": "awq"})
        with self.assertRaises(UnsupportedQuantization) as raised:
            scan_quantized_layers(state)
        self.assertIn("smoothing", str(raised.exception))

    def test_a_smoothing_scale_is_carried_where_comfy_reads_it_and_dropped_elsewhere(self):
        # ComfyUI multiplies an NVFP4 layer's activation by `pre_quant_scale`; dropping it there
        # changes every output. No other layout lists the parameter, so ComfyUI leaves it unread.
        codes = torch.zeros(16, 16, dtype=torch.uint8)
        state = {
            "block.weight": codes,
            "block.weight_scale": tiled(torch.ones(16, 2)).to(torch.float8_e4m3fn),
            "block.weight_scale_2": torch.tensor(1.0),
            "block.pre_quant_scale": torch.ones(32),
            "block.comfy_quant": comfy_quant_marker("nvfp4"),
        }
        layer = scan_quantized_layers(state)["block"]
        self.assertEqual(dict(layer.optional_keys), {"pre_quant_scale": "block.pre_quant_scale"})
        self.assertNotIn("block.pre_quant_scale", layer.dropped_keys)
        state = fp8_layer()
        state["block.pre_quant_scale"] = torch.ones(16)
        layer = scan_quantized_layers(state)["block"]
        self.assertEqual(dict(layer.optional_keys), {})
        self.assertIn("block.pre_quant_scale", layer.dropped_keys)

    def test_a_marker_that_only_describes_the_matmul_is_not_a_refusal(self):
        """`full_precision_matrix_mult` rides along on every layer of the Qwen3-VL fp8 encoder that
        this project loads today. It picks how the same value is computed, not what is stored."""
        state = fp8_layer()
        state["block.comfy_quant"] = torch.tensor(
            list(json.dumps({
                "format": "float8_e4m3fn", "full_precision_matrix_mult": False,
            }).encode("utf-8")),
            dtype=torch.uint8,
        )
        self.assertEqual(scan_quantized_layers(state)["block"].format, "float8_e4m3fn")

    def test_a_declared_format_must_match_the_stored_type(self):
        state = fp8_layer()
        state["block.comfy_quant"] = comfy_quant_marker("int8_tensorwise")
        with self.assertRaises(UnsupportedQuantization) as raised:
            scan_quantized_layers(state)
        self.assertIn("int8_tensorwise", str(raised.exception))

    def test_a_block_scale_on_a_scalar_format_is_refused(self):
        state = fp8_layer()
        state["block.weight_scale"] = torch.ones(8, dtype=torch.float32)
        with self.assertRaises(UnsupportedQuantization) as raised:
            scan_quantized_layers(state)
        self.assertIn("per-tensor", str(raised.exception))

    def test_an_activation_scale_is_recorded_as_dropped(self):
        state = fp8_layer()
        state["block.input_scale"] = torch.tensor(1.0)
        layers = scan_quantized_layers(state)
        self.assertIn("block.input_scale", layers["block"].dropped_keys)


class ScalarScaleTests(unittest.TestCase):
    def test_deferring_the_multiply_is_bit_identical_to_folding_it(self):
        # The whole design rests on this: folding at load and scaling at compute time land on the
        # same bits, so moving the multiply costs nothing but the time it used to take.
        torch.manual_seed(0)
        for _ in range(32):
            source = torch.randn(64, 128) * 10
            scale = torch.tensor(float(source.abs().max() / 448.0), dtype=torch.float32)
            weight = (source / scale).to(torch.float8_e4m3fn)
            folded = (weight.to(torch.float32) * scale.to(torch.float32).reshape(())).to(torch.bfloat16)
            deferred = dequantize_scalar_scale(weight, scale, torch.bfloat16)
            self.assertTrue(torch.equal(folded, deferred))

    def test_rounding_the_scale_first_is_what_would_change_the_numbers(self):
        # Recorded so the float32 scale is never "simplified" into the compute dtype.
        torch.manual_seed(1)
        source = torch.randn(64, 128) * 10
        scale = torch.tensor(float(source.abs().max() / 448.0), dtype=torch.float32)
        weight = (source / scale).to(torch.float8_e4m3fn)
        folded = (weight.to(torch.float32) * scale.reshape(())).to(torch.bfloat16)
        rounded = weight.to(torch.bfloat16) * scale.to(torch.bfloat16).reshape(())
        self.assertFalse(torch.equal(folded, rounded))

    def test_int8_recovers_its_weight(self):
        weight = torch.tensor([[-128, -1, 0, 1, 127]], dtype=torch.int8)
        scale = torch.tensor(0.5, dtype=torch.float32)
        recovered = dequantize_scalar_scale(weight, scale, torch.float32)
        self.assertTrue(torch.equal(recovered, torch.tensor([[-64.0, -0.5, 0.0, 0.5, 63.5]])))

    def test_the_result_carries_the_compute_dtype(self):
        state = fp8_layer()
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            recovered = dequantize_scalar_scale(state["block.weight"], state["block.weight_scale"], dtype)
            self.assertEqual(recovered.dtype, dtype)


def tiled(matrix: torch.Tensor) -> torch.Tensor:
    """A ``[rows, blocks]`` scale matrix in cuBLAS's 128x4 tile order, how ComfyUI stores it."""
    return to_blocked(matrix)


class BlockScaleTests(unittest.TestCase):
    """NVFP4 and MXFP8 pack or group their scales, so the arithmetic is checked against a
    hand-built tensor rather than against the loader that produced it.

    The layout is comfy-kitchen's: NVFP4 puts the **first** of two values in the **high** nibble,
    and both formats keep their block scales in cuBLAS's tiled order. The pre-P9 versions of these
    tests pinned the opposite nibble order and row-major scales; on a real Krea 2 layer that
    reading decoded to noise (see ``Temp/p9-claude/verify_layers.log``).
    """

    def test_nvfp4_unpacks_both_nibbles_in_order(self):
        # Codes 1 (+0.5) and 6 (+4.0) in the first byte of a block, high nibble first.
        weight = torch.zeros(1, NVFP4_GROUP_SIZE // 2, dtype=torch.uint8)
        weight[0, 0] = 0x16
        block = tiled(torch.ones(1, 1)).to(torch.float8_e4m3fn)
        recovered = dequantize_nvfp4(weight, block, torch.tensor(1.0), torch.float32, 1, NVFP4_GROUP_SIZE)
        self.assertTrue(torch.equal(recovered[:, :2], torch.tensor([[0.5, 4.0]])))
        with self.assertRaises(UnsupportedQuantization):
            dequantize_nvfp4(weight[:, :1], block, torch.tensor(1.0), torch.float32, 1, 2)

    def test_nvfp4_decodes_every_code_in_the_table(self):
        codes = torch.arange(16, dtype=torch.uint8)
        packed = (codes[0::2] << 4 | codes[1::2]).reshape(1, -1)
        block = tiled(torch.ones(1, 1)).to(torch.float8_e4m3fn)
        recovered = dequantize_nvfp4(packed, block, torch.tensor(1.0), torch.float32, 1, 16)
        self.assertTrue(torch.equal(recovered[0], torch.tensor(FP4_E2M1_VALUES)))

    def test_nvfp4_applies_the_block_scale_per_group_and_the_global_scale_once(self):
        codes = torch.full((1, NVFP4_GROUP_SIZE), 2, dtype=torch.uint8)  # code 2 is +1.0
        packed = (codes[:, 0::2] << 4 | codes[:, 1::2])
        packed = torch.cat([packed, packed], dim=1)
        block = tiled(torch.tensor([[2.0, 3.0]])).to(torch.float8_e4m3fn)
        recovered = dequantize_nvfp4(packed, block, torch.tensor(10.0), torch.float32, 1, 2 * NVFP4_GROUP_SIZE)
        self.assertTrue(torch.equal(recovered[0, :NVFP4_GROUP_SIZE], torch.full((NVFP4_GROUP_SIZE,), 20.0)))
        self.assertTrue(torch.equal(recovered[0, NVFP4_GROUP_SIZE:], torch.full((NVFP4_GROUP_SIZE,), 30.0)))

    def test_nvfp4_scales_are_read_in_tile_order_not_row_order(self):
        # Row 1's scale sits 16 entries into the first tile, not at index 1 (which is row 0's
        # second block). A row-major read would hand row 1 the scale of row 0's second block.
        scales = torch.tensor([[1.0], [4.0]])
        codes = torch.full((2, NVFP4_GROUP_SIZE), 2, dtype=torch.uint8)
        packed = codes[:, 0::2] << 4 | codes[:, 1::2]
        block = tiled(scales).to(torch.float8_e4m3fn)
        recovered = dequantize_nvfp4(packed, block, torch.tensor(1.0), torch.float32, 2, NVFP4_GROUP_SIZE)
        self.assertTrue(torch.equal(recovered[:, 0], torch.tensor([1.0, 4.0])))

    def test_nvfp4_refuses_a_weight_with_too_few_block_scales(self):
        packed = torch.zeros(1, NVFP4_GROUP_SIZE, dtype=torch.uint8)
        with self.assertRaises(UnsupportedQuantization):
            dequantize_nvfp4(packed, torch.ones(1, 1), torch.tensor(1.0), torch.float32, 1, 2 * NVFP4_GROUP_SIZE)

    def test_mxfp8_reads_a_power_of_two_exponent_byte(self):
        weight = torch.ones(1, MXFP8_GROUP_SIZE, dtype=torch.float32).to(torch.float8_e4m3fn)
        # 127 is 2^0, 128 is 2^1; 0 is how comfy-kitchen writes an all-zero block and reads as 0.
        for byte, expected in ((127, 1.0), (128, 2.0), (126, 0.5), (0, 0.0)):
            block = tiled(torch.tensor([[byte]], dtype=torch.uint8))
            recovered = dequantize_mxfp8(weight, block, torch.float32, 1, MXFP8_GROUP_SIZE)
            self.assertTrue(torch.equal(recovered, torch.full((1, MXFP8_GROUP_SIZE), expected)))

    def test_mxfp8_refuses_a_float_block_scale(self):
        # ComfyUI views the stored scale as E8M0 bytes; a float32 table is not that layout.
        weight = torch.full((1, MXFP8_GROUP_SIZE), 2.0).to(torch.float8_e4m3fn)
        with self.assertRaises(UnsupportedQuantization):
            dequantize_mxfp8(weight, tiled(torch.tensor([[3.0]])), torch.float32, 1, MXFP8_GROUP_SIZE)


class QuantizedLinearTests(unittest.TestCase):
    def build(self, rows=8, columns=16, bias=False):
        state = fp8_layer(rows, columns, prefix="net")
        if bias:
            state["net.bias"] = torch.randn(rows, dtype=torch.bfloat16)
        module = nn.Module()
        module.net = nn.Linear(columns, rows, bias=bias)
        layers = scan_quantized_layers(state)
        replaced, deferred = apply_quantized_linears(module, layers, state, torch.bfloat16)
        self.assertEqual(deferred, [])
        return module, state, layers, replaced

    def test_the_layer_is_replaced_and_keeps_its_shape(self):
        module, _, _, replaced = self.build()
        self.assertEqual(replaced, ["net"])
        self.assertIsInstance(module.net, QuantizedLinear)
        self.assertEqual((module.net.out_features, module.net.in_features), (8, 16))

    def test_it_is_still_a_linear_so_the_offload_machinery_recognises_it(self):
        module, _, _, _ = self.build()
        self.assertIsInstance(module.net, nn.Linear)

    def test_the_forward_matches_a_folded_weight_exactly(self):
        module, state, layers, replaced = self.build(bias=True)
        renamed = quantized_state_dict(state, layers, set(replaced))
        module.load_state_dict(renamed, strict=True, assign=True)
        folded = (state["net.weight"].to(torch.float32) * state["net.weight_scale"].reshape(())).to(torch.bfloat16)
        x = torch.randn(4, 16, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(module.net(x), F.linear(x, folded, state["net.bias"])))

    def test_a_device_move_does_not_expand_the_storage(self):
        module, state, layers, replaced = self.build()
        module.load_state_dict(quantized_state_dict(state, layers, set(replaced)), strict=True, assign=True)
        module.to(device="cpu", dtype=torch.bfloat16)
        self.assertEqual(module.net.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(module.net.scale_weight.dtype, torch.float32)

    def test_a_dtype_cast_leaves_the_scale_at_full_precision(self):
        # Rounding the scale is the one shortcut that changes the numbers, so a stray `.to(dtype)`
        # must not be able to do it.
        module, state, layers, replaced = self.build()
        module.load_state_dict(quantized_state_dict(state, layers, set(replaced)), strict=True, assign=True)
        before = module.net.scale_weight.clone()
        module.half()
        self.assertTrue(torch.equal(module.net.scale_weight, before))

    def test_a_non_linear_target_is_deferred_rather_than_refused(self):
        # A quantised embedding has no matmul to fold the scale into. Refusing it would stop a
        # checkpoint loading that used to load, so it is handed back for expansion instead.
        state = fp8_layer(prefix="embed")
        module = nn.Module()
        module.embed = nn.Embedding(8, 16)
        replaced, deferred = apply_quantized_linears(module, scan_quantized_layers(state), state, torch.bfloat16)
        self.assertEqual((replaced, deferred), ([], ["embed"]))
        self.assertIsInstance(module.embed, nn.Embedding)

    def test_a_layer_the_module_does_not_carry_is_skipped(self):
        state = fp8_layer(prefix="absent")
        module = nn.Module()
        self.assertEqual(apply_quantized_linears(module, scan_quantized_layers(state), state, torch.bfloat16), ([], []))

    def test_stored_bytes_count_the_scales_too(self):
        state = fp8_layer(8, 16, prefix="net")
        layers = scan_quantized_layers(state)
        self.assertEqual(quantized_weight_bytes(layers["net"], state), 8 * 16 + 4)


class StateDictTests(unittest.TestCase):
    def test_scales_move_to_the_buffer_name_and_markers_are_dropped(self):
        state = fp8_layer()
        state["block.comfy_quant"] = comfy_quant_marker("float8_e4m3fn")
        state["other.weight"] = torch.randn(2, 2)
        layers = scan_quantized_layers(state)
        renamed = quantized_state_dict(state, layers, {"block"})
        self.assertIn("block.scale_weight", renamed)
        self.assertNotIn("block.weight_scale", renamed)
        self.assertNotIn("block.comfy_quant", renamed)
        self.assertIn("other.weight", renamed)

    def test_a_layer_left_quantised_in_the_file_but_not_replaced_still_loses_its_marker(self):
        state = fp8_layer()
        state["block.comfy_quant"] = comfy_quant_marker("float8_e4m3fn")
        layers = scan_quantized_layers(state)
        renamed = quantized_state_dict(state, layers, set())
        self.assertNotIn("block.comfy_quant", renamed)


class ExpansionTests(unittest.TestCase):
    def test_only_the_named_layers_are_expanded(self):
        state = {**fp8_layer(prefix="a"), **fp8_layer(prefix="b")}
        layers = scan_quantized_layers(state)
        expanded = expand_quantized_layers(state, layers, {"a"}, torch.bfloat16)
        self.assertEqual(expanded["a.weight"].dtype, torch.bfloat16)
        self.assertEqual(expanded["b.weight"].dtype, torch.float8_e4m3fn)
        self.assertNotIn("a.weight_scale", expanded)
        self.assertIn("b.weight_scale", expanded)

    def test_expanding_nothing_returns_the_checkpoint_unchanged(self):
        state = fp8_layer()
        self.assertEqual(set(expand_quantized_layers(state, scan_quantized_layers(state), set(), torch.bfloat16)), set(state))

    def test_the_expansion_equals_the_folded_weight(self):
        state = fp8_layer(prefix="a")
        layers = scan_quantized_layers(state)
        expanded = expand_quantized_layers(state, layers, {"a"}, torch.bfloat16)
        folded = (state["a.weight"].to(torch.float32) * state["a.weight_scale"].reshape(())).to(torch.bfloat16)
        self.assertTrue(torch.equal(expanded["a.weight"], folded))


class LogicalShapeTests(unittest.TestCase):
    def test_a_packed_weight_reports_the_layer_shape_not_the_stored_one(self):
        state = {
            "block.weight": torch.zeros(16, 16, dtype=torch.uint8),
            "block.weight_scale": tiled(torch.ones(16, 2)).to(torch.float8_e4m3fn),
            "block.weight_scale_2": torch.tensor(1.0),
            "block.comfy_quant": comfy_quant_marker("nvfp4"),
            "w4.weight": torch.zeros(8, 128, dtype=torch.int8),
            "w4.weight_scale": torch.ones(8),
            "w4.comfy_quant": comfy_quant_marker("convrot_w4a4"),
        }
        layers = scan_quantized_layers(state)
        view = logical_shape_view(state, layers)
        self.assertEqual(tuple(view["block.weight"].shape), (16, 32))
        self.assertEqual(tuple(view["w4.weight"].shape), (8, 256))

    def test_an_nvfp4_block_scale_table_of_the_wrong_size_is_refused_at_scan(self):
        state = {
            "block.weight": torch.zeros(16, 16, dtype=torch.uint8),
            "block.weight_scale": torch.ones(16, 2).to(torch.float8_e4m3fn),
            "block.weight_scale_2": torch.tensor(1.0),
            "block.comfy_quant": comfy_quant_marker("nvfp4"),
        }
        with self.assertRaises(UnsupportedQuantization) as raised:
            scan_quantized_layers(state, "test checkpoint")
        self.assertIn("tiled", str(raised.exception))

    def test_an_unpacked_weight_is_passed_through_as_the_tensor_itself(self):
        state = fp8_layer(8, 16)
        view = logical_shape_view(state, scan_quantized_layers(state))
        self.assertIs(view["block.weight"], state["block.weight"])
        self.assertEqual(len(view), len(state))
        self.assertEqual(set(view), set(state))


class ComfyLayoutModuleTests(unittest.TestCase):
    """P9: each ported layout inside the module, against the arithmetic ComfyUI runs for it."""

    def build(self, state, rows, cols):
        module = nn.Module()
        module.net = nn.Linear(cols, rows, bias=False)
        layers = scan_quantized_layers(state)
        replaced, deferred = apply_quantized_linears(module, layers, state, torch.float32)
        self.assertEqual((replaced, deferred), (["net"], []))
        module.load_state_dict(quantized_state_dict(state, layers, set(replaced)), strict=True, assign=True)
        return module.net

    def convrot_int8_state(self, extra=None):
        torch.manual_seed(5)
        weight = torch.randn(8, 512) * 0.05
        rotated = comfy_quant.rotate_weight(weight, comfy_quant.build_hadamard(256), 256)
        q, scale = comfy_quant.quantize_int8_rowwise(rotated)
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256, **(extra or {})}
        return weight, {"net.weight": q, "net.weight_scale": scale, "net.comfy_quant": marker(conf)}

    def test_convrot_int8_runs_comfys_integer_matmul(self):
        weight, state = self.convrot_int8_state()
        layer = self.build(state, 8, 512)
        x = torch.randn(3, 512)
        self.assertTrue(layer.uses_integer_matmul)
        want = comfy_quant.int8_linear(x, state["net.weight"], state["net.weight_scale"], None, torch.float32, True, 256)
        self.assertTrue(torch.equal(layer(x), want))
        exact = x @ weight.T
        self.assertLess(((layer(x) - exact).norm() / exact.norm()).item(), 0.03)

    def test_a_full_precision_marker_or_a_lora_dequantises_instead(self):
        # ComfyUI's `_use_quantized` is false for both, and the weight is dequantised instead.
        x = torch.randn(3, 512)
        _, state = self.convrot_int8_state({"full_precision_matrix_mult": True})
        layer = self.build(state, 8, 512)
        dequantised = comfy_quant.dequantize_int8(state["net.weight"], state["net.weight_scale"], torch.float32, True, 256)
        self.assertFalse(layer.uses_integer_matmul)
        self.assertTrue(torch.equal(layer(x), F.linear(x, dequantised)))

        _, state = self.convrot_int8_state()
        layer = self.build(state, 8, 512)
        down, up = torch.randn(2, 512) * 0.1, torch.randn(8, 2) * 0.1
        layer.set_lora_adapter(down, up)
        self.assertFalse(layer.uses_integer_matmul)
        expected = F.linear(x, dequantised) + F.linear(F.linear(x, down), up)
        self.assertTrue(torch.allclose(layer(x), expected, atol=1e-5))

    def test_w4a4_runs_comfys_integer_matmul(self):
        torch.manual_seed(6)
        values = torch.randint(-7, 8, (8, 256), dtype=torch.int8)
        scale = torch.rand(8) * 0.01 + 0.001
        state = {
            "net.weight": comfy_quant.pack_int4(values),
            "net.weight_scale": scale,
            "net.comfy_quant": marker({"format": "convrot_w4a4", "convrot_groupsize": 64}),
        }
        layer = self.build(state, 8, 256)
        x = torch.randn(2, 3, 256)
        self.assertTrue(layer.uses_integer_matmul)
        want = comfy_quant.convrot_w4a4_linear(x, state["net.weight"], scale, None, 64)
        self.assertTrue(torch.equal(layer(x), want))
        self.assertEqual(tuple(layer(x).shape), (2, 3, 8))

    def test_nvfp4_and_mxfp8_multiply_the_dequantised_weight(self):
        torch.manual_seed(7)
        codes = torch.randint(0, 16, (16, 32), dtype=torch.uint8)
        nvfp4 = {
            "net.weight": codes[:, 0::2] << 4 | codes[:, 1::2],
            "net.weight_scale": tiled(torch.tensor([[0.5, 2.0]]).repeat(16, 1)).to(torch.float8_e4m3fn),
            "net.weight_scale_2": torch.tensor(0.25),
            "net.comfy_quant": comfy_quant_marker("nvfp4"),
        }
        layer = self.build(nvfp4, 16, 32)
        x = torch.randn(4, 32)
        weight = dequantize_nvfp4(nvfp4["net.weight"], nvfp4["net.weight_scale"], nvfp4["net.weight_scale_2"], torch.float32, 16, 32)
        self.assertTrue(torch.equal(layer(x), F.linear(x, weight)))
        self.assertEqual(layer.weight.dtype, torch.uint8)

        mxfp8 = {
            "net.weight": (torch.randn(16, 64) * 4).to(torch.float8_e4m3fn),
            "net.weight_scale": tiled(torch.tensor([[126, 128]], dtype=torch.uint8).repeat(16, 1)),
            "net.comfy_quant": comfy_quant_marker("mxfp8"),
        }
        layer = self.build(mxfp8, 16, 64)
        x = torch.randn(4, 64)
        weight = dequantize_mxfp8(mxfp8["net.weight"], mxfp8["net.weight_scale"], torch.float32, 16, 64)
        self.assertTrue(torch.equal(layer(x), F.linear(x, weight)))

    def test_the_smoothing_scale_multiplies_the_input(self):
        torch.manual_seed(8)
        codes = torch.randint(0, 16, (16, 32), dtype=torch.uint8)
        state = {
            "net.weight": codes[:, 0::2] << 4 | codes[:, 1::2],
            "net.weight_scale": tiled(torch.ones(16, 2)).to(torch.float8_e4m3fn),
            "net.weight_scale_2": torch.tensor(0.5),
            "net.pre_quant_scale": torch.linspace(0.5, 2.0, 32),
            "net.comfy_quant": comfy_quant_marker("nvfp4"),
        }
        layer = self.build(state, 16, 32)
        x = torch.randn(2, 32)
        weight = dequantize_nvfp4(state["net.weight"], state["net.weight_scale"], state["net.weight_scale_2"], torch.float32, 16, 32)
        self.assertTrue(torch.equal(layer(x), F.linear(x * state["net.pre_quant_scale"], weight)))
        expanded = expand_quantized_layers(state, scan_quantized_layers(state), {"net"}, torch.float32)
        self.assertNotIn("net.pre_quant_scale", expanded)
        self.assertTrue(torch.allclose(F.linear(x, expanded["net.weight"]), layer(x), atol=1e-5))

    def test_a_dtype_cast_keeps_integer_storage_and_float32_scales(self):
        _, state = self.convrot_int8_state()
        layer = self.build(state, 8, 512)
        layer.to(dtype=torch.bfloat16)
        self.assertEqual((layer.weight.dtype, layer.scale_weight.dtype), (torch.int8, torch.float32))

    def test_a_header_marked_fp8_file_loads_exactly_as_its_legacy_spelling(self):
        # Krea 2's fp8_scaled transformer keeps its markers in the header. Converted to marker
        # entries it must land on the same layers, scales and outputs as the scale-only reading.
        legacy = fp8_layer(prefix="net")
        table = {"layers": {"net": {"format": "float8_e4m3fn", "full_precision_matrix_mult": True}}}
        header = comfy_quant.apply_header_quant_metadata(legacy, {"_quantization_metadata": json.dumps(table)})
        a, b = scan_quantized_layers(legacy)["net"], scan_quantized_layers(header)["net"]
        self.assertEqual((a.format, dict(a.scale_keys)), (b.format, dict(b.scale_keys)))
        x = torch.randn(4, 16)
        self.assertTrue(torch.equal(self.build(legacy, 8, 16)(x), self.build(header, 8, 16)(x)))


if __name__ == "__main__":
    unittest.main()
