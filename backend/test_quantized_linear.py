"""Tests for running a quantised checkpoint without expanding it first."""

import json
import unittest

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .quantized_linear import (
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


def comfy_quant_marker(fmt: str) -> torch.Tensor:
    return torch.tensor(list(json.dumps({"format": fmt}).encode("utf-8")), dtype=torch.uint8)


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


class BlockScaleTests(unittest.TestCase):
    """NVFP4 and MXFP8 pack or group their scales, so the arithmetic is checked against a
    hand-built tensor rather than against the loader that produced it."""

    def test_nvfp4_unpacks_both_nibbles_in_order(self):
        # Codes 1 (+0.5) and 6 (+4.0) in one byte, low nibble first.
        weight = torch.tensor([[0x61]], dtype=torch.uint8)
        block = torch.tensor([[1.0]], dtype=torch.float32)
        recovered = dequantize_nvfp4(weight, block, torch.tensor(1.0), torch.float32, 1, 2)
        self.assertTrue(torch.equal(recovered, torch.tensor([[0.5, 4.0]])))

    def test_nvfp4_decodes_every_code_in_the_table(self):
        codes = torch.arange(16, dtype=torch.uint8)
        packed = (codes[1::2] << 4 | codes[0::2]).reshape(1, -1)
        block = torch.ones(1, 1, dtype=torch.float32)
        recovered = dequantize_nvfp4(packed, block, torch.tensor(1.0), torch.float32, 1, 16)
        self.assertTrue(torch.equal(recovered[0], torch.tensor(FP4_E2M1_VALUES)))

    def test_nvfp4_applies_the_block_scale_per_group_and_the_global_scale_once(self):
        codes = torch.full((1, NVFP4_GROUP_SIZE), 2, dtype=torch.uint8)  # code 2 is +1.0
        packed = (codes[:, 1::2] << 4 | codes[:, 0::2])
        packed = torch.cat([packed, packed], dim=1)
        block = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
        recovered = dequantize_nvfp4(packed, block, torch.tensor(10.0), torch.float32, 1, 2 * NVFP4_GROUP_SIZE)
        self.assertTrue(torch.equal(recovered[0, :NVFP4_GROUP_SIZE], torch.full((NVFP4_GROUP_SIZE,), 20.0)))
        self.assertTrue(torch.equal(recovered[0, NVFP4_GROUP_SIZE:], torch.full((NVFP4_GROUP_SIZE,), 30.0)))

    def test_nvfp4_refuses_a_weight_with_too_few_block_scales(self):
        packed = torch.zeros(1, NVFP4_GROUP_SIZE, dtype=torch.uint8)
        with self.assertRaises(UnsupportedQuantization):
            dequantize_nvfp4(packed, torch.ones(1, 1), torch.tensor(1.0), torch.float32, 1, 2 * NVFP4_GROUP_SIZE)

    def test_mxfp8_reads_a_power_of_two_exponent_byte(self):
        weight = torch.ones(1, MXFP8_GROUP_SIZE, dtype=torch.float32).to(torch.float8_e4m3fn)
        # 127 is 2^0, 128 is 2^1.
        for byte, expected in ((127, 1.0), (128, 2.0), (126, 0.5)):
            block = torch.tensor([[byte]], dtype=torch.uint8)
            recovered = dequantize_mxfp8(weight, block, torch.float32, 1, MXFP8_GROUP_SIZE)
            self.assertTrue(torch.equal(recovered, torch.full((1, MXFP8_GROUP_SIZE), expected)))

    def test_mxfp8_accepts_a_float_block_scale_unchanged(self):
        weight = torch.full((1, MXFP8_GROUP_SIZE), 2.0).to(torch.float8_e4m3fn)
        block = torch.tensor([[3.0]], dtype=torch.float32)
        recovered = dequantize_mxfp8(weight, block, torch.float32, 1, MXFP8_GROUP_SIZE)
        self.assertTrue(torch.equal(recovered, torch.full((1, MXFP8_GROUP_SIZE), 6.0)))


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
            "block.weight": torch.zeros(8, 16, dtype=torch.uint8),
            "block.weight_scale": torch.ones(8, 2, dtype=torch.float32),
            "block.weight_scale_2": torch.tensor(1.0),
            "block.comfy_quant": comfy_quant_marker("nvfp4"),
        }
        layers = scan_quantized_layers(state)
        view = logical_shape_view(state, layers)
        self.assertEqual(tuple(view["block.weight"].shape), (8, 32))

    def test_an_unpacked_weight_is_passed_through_as_the_tensor_itself(self):
        state = fp8_layer(8, 16)
        view = logical_shape_view(state, scan_quantized_layers(state))
        self.assertIs(view["block.weight"], state["block.weight"])
        self.assertEqual(len(view), len(state))
        self.assertEqual(set(view), set(state))


if __name__ == "__main__":
    unittest.main()
