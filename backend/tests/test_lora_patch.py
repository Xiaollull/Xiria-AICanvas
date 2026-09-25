import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from backend import lora_patch
from backend.lora_patch import (
    LoraPatchError,
    apply_lora_patch,
    apply_lora_plan,
    build_lora_groups,
    classify_lora_keys,
    lora_header_requires_direct_patch,
    lora_delta,
    preflight_lora_patch,
    read_lora_header,
    resolve_target,
)


def write_tensors(path, tensors):
    save_file({name: value.contiguous() for name, value in tensors.items()}, str(path))
    return path


def standard_tensors(name="lora_unet_linear", *, rank=1):
    return {
        f"{name}.lora_down.weight": torch.arange(1, rank * 2 + 1, dtype=torch.float32).reshape(rank, 2),
        f"{name}.lora_up.weight": torch.ones((2, rank), dtype=torch.float32),
        f"{name}.alpha": torch.tensor(float(rank)),
    }


class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.other = torch.nn.Linear(2, 2, bias=False)


class HeaderTests(unittest.TestCase):
    def test_header_uses_get_slice_and_never_get_tensor(self):
        class Slice:
            def get_shape(self):
                return [1, 2]

            def get_dtype(self):
                return "F32"

        class Handle:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def keys(self):
                return ["lora_unet_linear.lora_down.weight"]

            def get_slice(self, _key):
                return Slice()

            def get_tensor(self, _key):  # pragma: no cover - a call fails the test immediately
                raise AssertionError("header reader loaded a tensor")

        with mock.patch.object(lora_patch, "safe_open", return_value=Handle()):
            header = read_lora_header("only-header.safetensors")
        self.assertEqual(header.tensors[0].shape, (1, 2))
        self.assertEqual(header.tensors[0].dtype, "F32")

    def test_classifies_standard_loha_lokr_and_dora(self):
        cases = {
            "lora": ["lora_unet_x.lora_down.weight", "lora_unet_x.lora_up.weight"],
            "loha": [
                "lora_unet_x.hada_w1_a", "lora_unet_x.hada_w1_b",
                "lora_unet_x.hada_w2_a", "lora_unet_x.hada_w2_b",
            ],
            "lokr": ["lora_unet_x.lokr_w1", "lora_unet_x.lokr_w2"],
        }
        for family, keys in cases.items():
            with self.subTest(family=family):
                result = classify_lora_keys(keys, "fixture.safetensors")
                self.assertEqual(result.family, family)
        result = classify_lora_keys(
            cases["lora"] + ["lora_unet_x.dora_scale"], "dora.safetensors"
        )
        self.assertTrue(result.dora)
        self.assertTrue(result.advanced)

    def test_native_peft_and_kohya_standard_do_not_enter_strict_parser(self):
        native_keys = (
            "unet.down_blocks.0.attentions.0.to_q.lora_A.weight",
            "unet.down_blocks.0.attentions.0.to_q.lora_B.weight",
            "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight",
            "text_encoder_2.text_model.encoder.layers.0.self_attn.q_proj.lora_B.weight",
            "lora_unet_linear.lora_down.weight",
            "lora_unet_linear.lora_up.weight",
        )
        self.assertFalse(lora_header_requires_direct_patch(native_keys))
        for key in (
            "lora_unet_x.lokr_w2",
            "lora_unet_x.hada_w1_a",
            "lora_unet_x.dora_scale",
            "lora_unet_x.lora_mid.weight",
            "lora_unet_x.lora_magnitude_vector.weight",
        ):
            with self.subTest(key=key):
                self.assertTrue(lora_header_requires_direct_patch([key]))

    def test_rejects_every_declared_unsupported_role(self):
        bad_keys = (
            "lora_unet_x.lokr_t2",
            "lora_unet_x.lokr_w1_a",
            "lora_unet_x.lokr_w1_b",
            "lora_unet_x.hada_t1",
            "lora_unet_x.hada_t2",
            "lora_unet_x.lora_mid.weight",
            "lora_unet_x.lora_magnitude_vector.weight",
        )
        for key in bad_keys:
            with self.subTest(key=key):
                if key.endswith(("lokr_t2", "lokr_w1_a", "lokr_w1_b")):
                    keys = ["lora_unet_x.lokr_w1", "lora_unet_x.lokr_w2", key]
                elif ".hada_" in key:
                    keys = [
                        "lora_unet_x.hada_w1_a", "lora_unet_x.hada_w1_b",
                        "lora_unet_x.hada_w2_a", "lora_unet_x.hada_w2_b", key,
                    ]
                else:
                    keys = [key]
                with self.assertRaises(LoraPatchError):
                    classify_lora_keys(keys, "bad.safetensors")

    def test_build_rejects_missing_duplicate_mixed_unknown_nonfloat_and_bad_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = {
                "missing": {"lora_unet_linear.lora_down.weight": torch.ones(1, 2)},
                "duplicate": {
                    **standard_tensors(),
                    "lora_unet_linear.lora_A.weight": torch.ones(1, 2),
                },
                "mixed": {
                    **standard_tensors(),
                    "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                    "lora_unet_linear.lokr_w2": torch.ones(2, 2),
                },
                "nonfloat": {
                    "lora_unet_linear.lora_down.weight": torch.ones(1, 2, dtype=torch.int64),
                    "lora_unet_linear.lora_up.weight": torch.ones(2, 1),
                },
                "badshape": {
                    "lora_unet_linear.lora_down.weight": torch.ones(1),
                    "lora_unet_linear.lora_up.weight": torch.ones(2, 1),
                },
            }
            for label, tensors in fixtures.items():
                with self.subTest(label=label):
                    path = write_tensors(root / f"{label}.safetensors", tensors)
                    with self.assertRaises(LoraPatchError):
                        build_lora_groups(read_lora_header(path))
            with self.assertRaises(LoraPatchError):
                classify_lora_keys(["lora_unet_linear.unknown"], "unknown.safetensors")

    def test_rejects_full_and_lowrank_lokr_mix_and_loha_dora(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mixed = write_tensors(root / "mixed.safetensors", {
                "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                "lora_unet_linear.lokr_w2": torch.ones(2, 2),
                "lora_unet_linear.lokr_w2_a": torch.ones(2, 1),
                "lora_unet_linear.lokr_w2_b": torch.ones(1, 2),
            })
            loha = write_tensors(root / "loha-dora.safetensors", {
                "lora_unet_linear.hada_w1_a": torch.ones(2, 1),
                "lora_unet_linear.hada_w1_b": torch.ones(1, 2),
                "lora_unet_linear.hada_w2_a": torch.ones(2, 1),
                "lora_unet_linear.hada_w2_b": torch.ones(1, 2),
                "lora_unet_linear.dora_scale": torch.ones(2),
            })
            for path in (mixed, loha):
                with self.assertRaises(LoraPatchError):
                    build_lora_groups(read_lora_header(path))

    def test_header_key_group_workspace_and_error_limits(self):
        class Handle:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def keys(self):
                return ["a", "b", "c"]

            def get_slice(self, _key):  # pragma: no cover - count gate is earlier
                raise AssertionError("slice should not be read")

        with mock.patch.object(lora_patch, "MAX_LORA_KEYS", 2), mock.patch.object(
            lora_patch, "safe_open", return_value=Handle()
        ):
            with self.assertRaisesRegex(LoraPatchError, "键数量"):
                read_lora_header("too-many.safetensors")
        with mock.patch.object(lora_patch, "MAX_LORA_KEY_LENGTH", 8):
            with self.assertRaisesRegex(LoraPatchError, "键名"):
                classify_lora_keys(["lora_unet_far_too_long.lora_down.weight"], "long.safetensors")
        with mock.patch.object(lora_patch, "MAX_LORA_GROUPS", 1):
            with self.assertRaisesRegex(LoraPatchError, "目标组数量"):
                classify_lora_keys([
                    "lora_unet_a.lora_down.weight", "lora_unet_a.lora_up.weight",
                    "lora_unet_b.lora_down.weight", "lora_unet_b.lora_up.weight",
                ], "groups.safetensors")

        with tempfile.TemporaryDirectory() as directory:
            path = write_tensors(Path(directory) / "workspace.safetensors", {
                **standard_tensors(),
                "lora_unet_linear.dora_scale": torch.ones(2),
            })
            header = read_lora_header(path)
            workspace = build_lora_groups(header)[0].workspace_bytes
            with mock.patch.object(lora_patch, "MAX_GROUP_WORKSPACE_BYTES", workspace):
                self.assertEqual(build_lora_groups(header)[0].workspace_bytes, workspace)
            with mock.patch.object(lora_patch, "MAX_GROUP_WORKSPACE_BYTES", workspace - 1):
                with self.assertRaisesRegex(LoraPatchError, "workspace"):
                    build_lora_groups(header)
        error = lora_patch.readable_lora_error("limits.safetensors", "LoKr", "key," * 100_000)
        self.assertLessEqual(len(str(error)), lora_patch.MAX_LORA_ERROR_LENGTH)


class MathTests(unittest.TestCase):
    def group_and_tensors(self, tensors):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = write_tensors(Path(temporary.name) / "math.safetensors", tensors)
        group = build_lora_groups(read_lora_header(path))[0]
        role_tensors = {role: tensors[entry.key] for role, entry in group.roles}
        return group, role_tensors

    def test_standard_lora_alpha(self):
        tensors = standard_tensors(rank=2)
        tensors["lora_unet_linear.alpha"] = torch.tensor(1.0)
        group, roles = self.group_and_tensors(tensors)
        expected = roles["up"] @ roles["down"] * 0.5
        torch.testing.assert_close(lora_delta(group, roles), expected)

    def test_loha_four_factor_and_conv_flatten(self):
        tensors = {
            "lora_unet_conv.hada_w1_a": torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1),
            "lora_unet_conv.hada_w1_b": torch.arange(1, 9, dtype=torch.float32).reshape(1, 2, 2, 2),
            "lora_unet_conv.hada_w2_a": torch.tensor([3.0, 4.0]).reshape(2, 1, 1, 1),
            "lora_unet_conv.hada_w2_b": torch.ones(1, 2, 2, 2),
            "lora_unet_conv.alpha": torch.tensor(0.5),
        }
        group, roles = self.group_and_tensors(tensors)
        expected = (
            roles["hada_w1_a"].reshape(2, 1) @ roles["hada_w1_b"].reshape(1, 8)
        ) * (
            roles["hada_w2_a"].reshape(2, 1) @ roles["hada_w2_b"].reshape(1, 8)
        ) * 0.5
        torch.testing.assert_close(lora_delta(group, roles), expected)
        weight = torch.zeros(2, 2, 2, 2)
        apply_lora_patch(weight, expected, 1.0)
        torch.testing.assert_close(weight.reshape(2, 8), expected)

    def test_lokr_full_ignores_alpha_and_supports_conv(self):
        tensors = {
            "lora_unet_conv.lokr_w1": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "lora_unet_conv.lokr_w2": torch.arange(1, 5, dtype=torch.float32).reshape(1, 1, 2, 2),
            "lora_unet_conv.alpha": torch.tensor(999.0),
        }
        group, roles = self.group_and_tensors(tensors)
        w1 = roles["lokr_w1"].unsqueeze(-1).unsqueeze(-1)
        expected = torch.kron(w1, roles["lokr_w2"]).reshape(2, 8)
        torch.testing.assert_close(lora_delta(group, roles), expected)

    def test_lokr_lowrank_uses_alpha_over_rank(self):
        tensors = {
            "lora_unet_linear.lokr_w1": torch.tensor([[1.0, 2.0]]),
            "lora_unet_linear.lokr_w2_a": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "lora_unet_linear.lokr_w2_b": torch.tensor([[2.0], [4.0]]),
            "lora_unet_linear.alpha": torch.tensor(1.0),
        }
        group, roles = self.group_and_tensors(tensors)
        expected = torch.kron(roles["lokr_w1"], roles["lokr_w2_a"] @ roles["lokr_w2_b"]) * 0.5
        torch.testing.assert_close(lora_delta(group, roles), expected)

    def test_dora_row_norm_multiplier_zero_and_negative(self):
        base = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        delta = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        apply_lora_patch(base, delta, 0.5, torch.tensor([10.0, 8.0]))
        torch.testing.assert_close(base, torch.tensor([[6.5, 0.0], [0.0, 6.0]]))

        unchanged = torch.ones(2, 2)
        before = unchanged.clone()
        apply_lora_patch(unchanged, torch.full((2, 2), float("nan")), 0.0)
        torch.testing.assert_close(unchanged, before)

        negative = torch.zeros(2, 2)
        apply_lora_patch(negative, torch.ones(2, 2), -2.0)
        torch.testing.assert_close(negative, torch.full((2, 2), -2.0))

    def test_nan_shape_and_cast_overflow_are_rejected_without_write(self):
        base = torch.zeros(2, 2)
        for delta in (torch.full((2, 2), float("nan")), torch.ones(3, 2)):
            with self.subTest(shape=tuple(delta.shape)):
                before = base.clone()
                with self.assertRaises(LoraPatchError):
                    apply_lora_patch(base, delta, 1.0)
                torch.testing.assert_close(base, before)
        half = torch.zeros(1, 1, dtype=torch.float16)
        with self.assertRaises(LoraPatchError):
            apply_lora_patch(half, torch.tensor([[1e10]]), 1.0)
        self.assertEqual(half.item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available for the independent tiny-matrix probe")
    def test_tiny_matrix_patch_on_cuda(self):
        weight = torch.zeros((2, 2), device="cuda", dtype=torch.float16)
        delta = torch.tensor([[1.0, -2.0], [3.0, -4.0]], device="cuda")
        apply_lora_patch(weight, delta, -0.5)
        torch.testing.assert_close(weight.cpu(), torch.tensor([[-0.5, 1.0], [-1.5, 2.0]], dtype=torch.float16))


def mapping_parameters():
    tensor = lambda: torch.zeros(2, 2)  # noqa: E731 - compact fixture table
    return {
        "unet.down_blocks.0.resnets.0.conv1.weight": tensor(),
        "unet.down_blocks.0.resnets.1.conv1.weight": tensor(),
        "unet.down_blocks.0.downsamplers.0.conv.weight": tensor(),
        "unet.down_blocks.1.resnets.0.conv1.weight": tensor(),
        "unet.mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight": tensor(),
        "unet.up_blocks.0.resnets.0.conv_shortcut.weight": tensor(),
        "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.weight": tensor(),
        "text_encoder_2.text_model.encoder.layers.0.self_attn.k_proj.weight": tensor(),
    }


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.parameters = mapping_parameters()

    def test_mapping_goldens_include_comfy_and_lycoris(self):
        cases = {
            "lora_unet_input_blocks_1_0_in_layers_2": "unet.down_blocks.0.resnets.0.conv1.weight",
            "lycoris_input_blocks_1_0_in_layers_2": "unet.down_blocks.0.resnets.0.conv1.weight",
            "lora_unet_input_blocks_0_resnets_0_conv1": "unet.down_blocks.0.resnets.0.conv1.weight",
            "lora_unet_input_blocks_0_downsamplers_0_op": "unet.down_blocks.0.downsamplers.0.conv.weight",
            "lora_unet_middle_block_attentions_0_transformer_blocks_0_attn1_to_q": "unet.mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight",
            "lora_unet_output_blocks_0_0_skip_connection": "unet.up_blocks.0.resnets.0.conv_shortcut.weight",
            "lycoris_down_blocks_0_resnets_0_conv1": "unet.down_blocks.0.resnets.0.conv1.weight",
            "unet.down_blocks.0.resnets.0.conv1": "unet.down_blocks.0.resnets.0.conv1.weight",
            "lora_te_text_model_encoder_layers_0_self_attn_q_proj": "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.weight",
            "text_encoder.text_model.encoder.layers.0.self_attn.q_proj": "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.weight",
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj": "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.weight",
            "lora_te2_text_model_encoder_layers_0_self_attn_k_proj": "text_encoder_2.text_model.encoder.layers.0.self_attn.k_proj.weight",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                target = resolve_target(self.parameters, source, "sdxl", (2, 2))
                self.assertEqual(target.qualified_name, expected)

    def test_sd_te2_is_readably_rejected(self):
        for name in (
            "lora_te2_text_model_encoder_layers_0_self_attn_k_proj",
            "text_encoder_2.text_model.encoder.layers.0.self_attn.k_proj",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(LoraPatchError, "TE2"):
                resolve_target(self.parameters, name, "sd", (2, 2), file_name="te2.safetensors")

    def test_ambiguous_alias_fails_closed(self):
        parameters = dict(self.parameters)
        # Dots and underscores flatten to the same Kohya alias.
        parameters["unet.foo_bar.weight"] = torch.zeros(2, 2)
        parameters["unet.foo.bar.weight"] = torch.zeros(2, 2)
        with mock.patch.object(lora_patch, "_private_converter_candidate", return_value=None):
            with self.assertRaisesRegex(LoraPatchError, "歧义"):
                resolve_target(parameters, "lora_unet_foo_bar", "sdxl", (2, 2))

    def test_private_converter_missing_raising_and_wrong_version_use_fallback(self):
        source = "lora_unet_down_blocks_0_resnets_0_conv1"
        expected = "unet.down_blocks.0.resnets.0.conv1.weight"
        cases = (
            mock.patch.object(lora_patch, "import_module", side_effect=ImportError("missing")),
            mock.patch.object(
                lora_patch,
                "import_module",
                side_effect=[SimpleNamespace(__version__="0.39.0"), AssertionError("must not import private module")],
            ),
            mock.patch.object(
                lora_patch,
                "import_module",
                side_effect=[
                    SimpleNamespace(__version__="0.38.9"),
                    SimpleNamespace(
                        _convert_unet_lora_key=mock.Mock(side_effect=RuntimeError("converter changed")),
                        _convert_text_encoder_lora_key=mock.Mock(side_effect=RuntimeError("converter changed")),
                    ),
                ],
            ),
        )
        for patcher in cases:
            with self.subTest(patcher=repr(patcher)), patcher:
                target = resolve_target(self.parameters, source, "sdxl", (2, 2))
                self.assertEqual(target.qualified_name, expected)

    def test_private_converter_conflict_is_version_incompatibility(self):
        with mock.patch.object(
            lora_patch,
            "_private_converter_candidate",
            return_value=("unet", "down_blocks.0.resnets.1.conv1.weight"),
        ):
            with self.assertRaisesRegex(LoraPatchError, "Diffusers映射版本不兼容"):
                resolve_target(
                    self.parameters,
                    "lora_unet_down_blocks_0_resnets_0_conv1",
                    "sdxl",
                    (2, 2),
                )


class TargetIndexTests(unittest.TestCase):
    """QPlan P5 Phase B: one parameter/alias index per transaction, same answers as before."""

    def setUp(self):
        self.parameters = mapping_parameters()

    def test_shared_index_matches_per_lookup_path_for_every_golden_and_error(self):
        goldens = [
            "lora_unet_input_blocks_1_0_in_layers_2",
            "lycoris_input_blocks_1_0_in_layers_2",
            "lora_unet_input_blocks_0_downsamplers_0_op",
            "lora_unet_middle_block_attentions_0_transformer_blocks_0_attn1_to_q",
            "lora_unet_output_blocks_0_0_skip_connection",
            "unet.down_blocks.0.resnets.0.conv1",
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj",
            "lora_te2_text_model_encoder_layers_0_self_attn_k_proj",
        ]
        index = lora_patch.build_target_index(self.parameters)
        for name in goldens:
            with self.subTest(name=name):
                self.assertEqual(
                    resolve_target(self.parameters, name, "sdxl", (2, 2), index=index),
                    resolve_target(self.parameters, name, "sdxl", (2, 2)),
                )
        failures = (
            ("lora_unet_missing", "sdxl", (2, 2)),
            ("lora_te2_text_model_encoder_layers_0_self_attn_k_proj", "sd", (2, 2)),
            ("lora_unet_input_blocks_1_0_in_layers_2", "sdxl", (3, 3)),
        )
        for name, family, shape in failures:
            with self.subTest(failure=name):
                with self.assertRaises(LoraPatchError) as shared:
                    resolve_target(self.parameters, name, family, shape, file_name="x.safetensors", index=index)
                with self.assertRaises(LoraPatchError) as private:
                    resolve_target(self.parameters, name, family, shape, file_name="x.safetensors")
                self.assertEqual(str(shared.exception), str(private.exception))
        self.assertEqual(index.counts(), {"target_index_builds": 1, "target_lookups": len(goldens) + len(failures)})

    def test_ambiguity_and_converter_conflict_are_unchanged_through_an_index(self):
        parameters = dict(self.parameters)
        parameters["unet.foo_bar.weight"] = torch.zeros(2, 2)
        parameters["unet.foo.bar.weight"] = torch.zeros(2, 2)
        with mock.patch.object(lora_patch, "_private_converter_candidate", return_value=None):
            index = lora_patch.build_target_index(parameters)
            with self.assertRaisesRegex(LoraPatchError, "歧义"):
                resolve_target(parameters, "lora_unet_foo_bar", "sdxl", (2, 2), index=index)
        with mock.patch.object(
            lora_patch, "_private_converter_candidate", return_value=("unet", "down_blocks.0.resnets.1.conv1.weight")
        ):
            index = lora_patch.build_target_index(self.parameters)
            with self.assertRaisesRegex(LoraPatchError, "Diffusers映射版本不兼容"):
                resolve_target(
                    self.parameters, "lora_unet_down_blocks_0_resnets_0_conv1", "sdxl", (2, 2), index=index,
                )

    def test_one_alias_build_serves_every_group_and_converter_is_memoised_per_transaction(self):
        module = TinyUNet()
        components = {"unet": module, "text_encoder": None}
        with tempfile.TemporaryDirectory() as directory:
            path = write_tensors(Path(directory) / "two.safetensors", {
                "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                "lora_unet_linear.lokr_w2": torch.eye(2),
                "lora_unet_other.lokr_w1": torch.ones(1, 1),
                "lora_unet_other.lokr_w2": torch.eye(2),
            })
            original_aliases = lora_patch._unet_aliases
            original_converter = lora_patch._private_converter_candidate
            with mock.patch.object(lora_patch, "_unet_aliases", side_effect=original_aliases) as aliases, mock.patch.object(
                lora_patch, "_private_converter_candidate", side_effect=original_converter
            ) as converter:
                index = lora_patch.build_target_index(components)
                first = preflight_lora_patch(path, components, "sd", require_materialized=False, index=index)
                second = preflight_lora_patch(path, components, "sd", require_materialized=False, index=index)
            self.assertEqual(aliases.call_count, 1)
            self.assertEqual(converter.call_count, 2)
            self.assertEqual(first, second)
            self.assertEqual(index.counts(), {"target_index_builds": 1, "target_lookups": 4})

            # The pre-P5 behaviour is still what a caller without an index gets: one build per file.
            with mock.patch.object(lora_patch, "_unet_aliases", side_effect=original_aliases) as aliases:
                self.assertEqual(preflight_lora_patch(path, components, "sd", require_materialized=False), first)
            self.assertEqual(aliases.call_count, 1)

    def test_index_refuses_other_components_and_reads_live_parameters(self):
        module = TinyUNet()
        components = {"unet": module}
        index = lora_patch.build_target_index(components)
        # Another pipeline, or the same pipeline with a component replaced, is never answered.
        for other in ({"unet": TinyUNet()}, {"unet": module, "text_encoder": TinyUNet()}, {}):
            with self.subTest(components=sorted(other)), self.assertRaisesRegex(LoraPatchError, "目标索引"):
                resolve_target(other, "lora_unet_linear", "sd", (2, 2), index=index)
        # A fresh mapping holding the very same components is the same transaction's view.
        self.assertEqual(
            resolve_target({"unet": module}, "lora_unet_linear", "sd", (2, 2), index=index).qualified_name,
            "unet.linear.weight",
        )
        # The index keeps names, not tensors: a parameter swapped in after the build is what a lookup sees.
        module.linear.weight = torch.nn.Parameter(torch.zeros(2, 2, device="meta"))
        target = resolve_target(components, "lora_unet_linear", "sd", (2, 2), index=index)
        self.assertEqual(target.shape, (2, 2))
        with self.assertRaisesRegex(LoraPatchError, "materialize"):
            resolve_target(components, "lora_unet_linear", "sd", (2, 2), index=index, require_materialized=True)


class HostPrefaultTests(unittest.TestCase):
    def test_prefault_rewrites_every_queued_byte_unchanged_and_merges_ranges(self):
        # One storage, like a mapped checkpoint: adjacent views merge into one range.
        storage = torch.randn(4096).half()
        storage[7] = float("nan")
        storage.view(torch.int16)[9] = 0x7C01  # a signalling-NaN payload must survive too
        before = storage.clone()
        first, second, tail = storage[:1024], storage[1024:3000], storage[3500:]
        prefault = lora_patch.HostWeightPrefault()
        prefault.add([first, second])
        prefault.add([tail, first])  # a repeat is not queued twice
        prefault.add([storage[::2], torch.empty(0)])  # non-contiguous and empty tensors are skipped
        busy = prefault.finish()
        self.assertTrue(torch.equal(storage.view(torch.int16), before.view(torch.int16)))
        self.assertEqual(prefault.bytes, (3000 + (4096 - 3500)) * 2)
        self.assertGreaterEqual(busy, 0.0)
        self.assertEqual(
            [(start, length) for _storage, start, length in lora_patch._merged_storage_ranges([first, second, tail])],
            [(0, 6000), (7000, (4096 - 3500) * 2)],
        )
        # Finishing twice, or with nothing queued, is harmless.
        prefault.finish()
        lora_patch.HostWeightPrefault().finish()


class PlanTests(unittest.TestCase):
    def test_preflight_then_streaming_apply_and_zero_reads_no_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_tensors(Path(directory) / "advanced.safetensors", {
                "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                "lora_unet_linear.lokr_w2": torch.eye(2),
            })
            module = TinyUNet()
            module.linear.weight.data.zero_()
            plan = preflight_lora_patch(path, {"unet": module}, "sd")
            report = apply_lora_plan(path, plan, {"unet": module}, -1.0)
            torch.testing.assert_close(module.linear.weight, -torch.eye(2))
            self.assertEqual(report.target_count, 1)

            before = module.linear.weight.detach().clone()
            with mock.patch.object(lora_patch, "safe_open", side_effect=AssertionError("tensor file opened")):
                apply_lora_plan(path, plan, {"unet": module}, 0.0)
            torch.testing.assert_close(module.linear.weight, before)

    def test_meta_target_is_mappable_but_not_patchable(self):
        module = TinyUNet().to(device="meta")
        target = resolve_target({"unet": module}, "lora_unet_linear", "sd", (2, 2))
        self.assertEqual(target.shape, (2, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = write_tensors(Path(directory) / "meta.safetensors", {
                "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                "lora_unet_linear.lokr_w2": torch.eye(2),
            })
            with self.assertRaisesRegex(LoraPatchError, "materialize"):
                preflight_lora_patch(path, {"unet": module}, "sd")

    def test_same_file_standard_and_lokr_stays_one_direct_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_tensors(Path(directory) / "mixed-file.safetensors", {
                **standard_tensors("lora_unet_linear"),
                "lora_unet_other.lokr_w1": torch.ones(1, 1),
                "lora_unet_other.lokr_w2": torch.eye(2),
            })
            module = TinyUNet()
            module.linear.weight.data.zero_()
            module.other.weight.data.zero_()
            plan = preflight_lora_patch(path, {"unet": module}, "sd")
            self.assertEqual(plan.family, "lora+lokr")
            self.assertEqual(plan.target_count, 2)
            apply_lora_plan(path, plan, {"unet": module}, 1.0)
            self.assertFalse(torch.equal(module.linear.weight, torch.zeros_like(module.linear.weight)))
            torch.testing.assert_close(module.other.weight, torch.eye(2))

    def test_dora_header_group_preflight_and_lazy_apply_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_tensors(Path(directory) / "dora-chain.safetensors", {
                "lora_unet_linear.lora_down.weight": torch.tensor([[1.0, 0.0]]),
                "lora_unet_linear.lora_up.weight": torch.tensor([[1.0], [2.0]]),
                "lora_unet_linear.alpha": torch.tensor(1.0),
                "lora_unet_linear.dora_scale": torch.tensor([3.0, 4.0]),
            })
            module = TinyUNet()
            module.linear.weight.data.zero_()
            header = read_lora_header(path)
            self.assertTrue(lora_header_requires_direct_patch(header))
            groups = build_lora_groups(header)
            self.assertTrue(groups[0].dora)
            plan = preflight_lora_patch(path, {"unet": module}, "sd", header=header, groups=groups)
            self.assertTrue(plan.dora)
            before = module.linear.weight.detach().clone()
            self.assertTrue(torch.equal(before, module.linear.weight))
            report = apply_lora_plan(path, plan, {"unet": module}, 1.0)
            self.assertTrue(report.dora)
            torch.testing.assert_close(module.linear.weight, torch.tensor([[3.0, 0.0], [4.0, 0.0]]))


class IntegrationTests(unittest.TestCase):
    class Pipeline:
        def __init__(
            self,
            events,
            *,
            fuse_error=None,
            load_error_at=None,
            load_error=None,
            set_error=None,
            unload_error_at=None,
        ):
            self.unet = TinyUNet()
            self.text_encoder = None
            self.text_encoder_2 = None
            self.events = events
            self.fuse_error = fuse_error
            self.load_error_at = load_error_at
            self.load_error = load_error
            self.set_error = set_error
            self.unload_error_at = unload_error_at
            self.load_count = 0
            self.unload_count = 0
            self.dirty_before_actions = []

        def _record_state(self):
            state = getattr(self, "_xiriai_lora_fusion_state", {})
            self.dirty_before_actions.append(state.get("state"))

        def unload_lora_weights(self):
            self._record_state()
            self.unload_count += 1
            self.events.append("unload")
            if self.unload_error_at == self.unload_count:
                raise RuntimeError("partial unload")

        def load_lora_weights(self, _directory, *, weight_name, adapter_name):
            self._record_state()
            self.load_count += 1
            self.events.append(("load", weight_name, adapter_name))
            if self.load_error_at == self.load_count:
                raise self.load_error or RuntimeError(
                    "partial attach "
                    + ",".join(
                        f"base_model.unet.block.{index}.to_q.lora_A.adapter.weight"
                        for index in range(10_000)
                    )
                )

        def set_adapters(self, names, *, adapter_weights):
            self._record_state()
            self.events.append(("set", tuple(names), tuple(adapter_weights)))
            if self.set_error:
                raise self.set_error

        def fuse_lora(self, *, adapter_names):
            self._record_state()
            self.events.append(("fuse", tuple(adapter_names)))
            if self.fuse_error:
                raise self.fuse_error

    class Residency:
        def __init__(self, pipeline, events, tier="T1", parked=False):
            self.pipeline = pipeline
            self.events = events
            self.tier = tier
            self.parked = parked
            self.trace = []

        def release(self, *, empty_cache):
            self.events.append(("residency-release", empty_cache, self.tier))

        def reinstall(self, tier=None):
            self.events.append(("residency-reinstall", tier or self.tier))
            self.parked = False
            return True

    def setUp(self):
        from backend import inference_server

        self.server = inference_server
        self.old_family = inference_server.loaded_family
        self.old_checkpoint = inference_server.loaded_checkpoint
        self.old_engine = inference_server.loaded_engine
        self.old_pipeline = inference_server.loaded_pipeline
        self.old_residency = inference_server.active_residency
        self.old_parked = inference_server.pipeline_cpu_parked
        self.old_strategy = inference_server.active_memory_strategy
        inference_server.loaded_family = "sd"
        inference_server.loaded_pipeline = None
        inference_server.active_residency = None
        inference_server.pipeline_cpu_parked = False
        inference_server.active_memory_strategy = None
        # These fakes record Diffusers' call order, so they run on the host path whether or not
        # this machine has a card; the device paths have their own tests below.
        compute = mock.patch.object(inference_server, "lora_compute_device", return_value=None)
        compute.start()
        self.addCleanup(compute.stop)

    def tearDown(self):
        self.server.loaded_family = self.old_family
        self.server.loaded_checkpoint = self.old_checkpoint
        self.server.loaded_engine = self.old_engine
        self.server.loaded_pipeline = self.old_pipeline
        self.server.active_residency = self.old_residency
        self.server.pipeline_cpu_parked = self.old_parked
        self.server.active_memory_strategy = self.old_strategy

    def lora_input(self, name, weight):
        return self.server.LoraInput(path=name, weight=weight)

    def test_standard_only_keeps_original_numeric_call_order_and_zero_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "standard.safetensors", standard_tensors())
            events = []
            pipeline = self.Pipeline(events)
            report = self.server.configure_loras(
                pipeline, [self.lora_input("standard.safetensors", 0.0)], root, "missing-job"
            )
            self.assertEqual(
                events,
                [
                    "unload",
                    ("load", "standard.safetensors", "lora_0"),
                    ("set", ("lora_0",), (0.0,)),
                    ("fuse", ("lora_0",)),
                    "unload",
                ],
            )
            self.assertEqual(report["standard"], 1)
            self.assertEqual(report["lycoris"], 0)
            self.assertEqual(report["state"], "clean")
            self.assertTrue(pipeline.dirty_before_actions)
            self.assertEqual(set(pipeline.dirty_before_actions), {"dirty"})

    def test_real_temp_peft_standard_uses_only_native_diffusers_call_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "peft.safetensors", {
                "unet.linear.lora_A.weight": torch.ones(1, 2),
                "unet.linear.lora_B.weight": torch.ones(2, 1),
                "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1, 2),
                "text_encoder_2.text_model.encoder.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(2, 1),
            })
            events = []
            pipeline = self.Pipeline(events)
            with mock.patch.object(self.server, "classify_lora_keys", side_effect=AssertionError("strict parse")), mock.patch.object(
                self.server, "build_lora_groups", side_effect=AssertionError("strict build")
            ), mock.patch.object(self.server, "apply_lora_plan", side_effect=AssertionError("direct patch")):
                report = self.server.configure_loras(
                    pipeline, [self.lora_input("peft.safetensors", -0.25)], root, "missing-job"
                )
            self.assertEqual(events, [
                "unload",
                ("load", "peft.safetensors", "lora_0"),
                ("set", ("lora_0",), (-0.25,)),
                ("fuse", ("lora_0",)),
                "unload",
            ])
            self.assertEqual(report["files"][0]["format"], "standard-native")
            self.assertEqual(report["standard"], 1)
            self.assertEqual(report["targets"], 0)

    def test_partial_load_and_set_failures_are_dirty_before_action(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "one.safetensors", standard_tensors())
            write_tensors(root / "two.safetensors", standard_tensors("lora_unet_other"))
            loras = [self.lora_input("one.safetensors", 1), self.lora_input("two.safetensors", 1)]
            for pipeline in (
                self.Pipeline([], unload_error_at=1),
                self.Pipeline([], load_error_at=2),
                self.Pipeline([], set_error=RuntimeError("partial set")),
            ):
                with self.subTest(error=(pipeline.unload_error_at, pipeline.load_error_at, pipeline.set_error)):
                    with self.assertRaises(LoraPatchError) as captured:
                        self.server.configure_loras(pipeline, loras, root, "missing-job")
                    self.assertLessEqual(len(str(captured.exception)), lora_patch.MAX_LORA_ERROR_LENGTH)
                    self.assertEqual(pipeline._xiriai_lora_fusion_state["state"], "dirty")
                    self.assertEqual(set(pipeline.dirty_before_actions), {"dirty"})
                    signature = pipeline._xiriai_lora_fusion_state["signature"]
                    self.assertFalse(self.server.pipeline_lora_cache_compatible(pipeline, signature))
                    self.assertFalse(self.server.pipeline_lora_cache_compatible(pipeline, ()))

    def test_standard_only_fuse_failure_keeps_current_mounted_fallback_but_not_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "standard.safetensors", standard_tensors())
            pipeline = self.Pipeline([], fuse_error=RuntimeError("native fuse failed"))
            report = self.server.configure_loras(
                pipeline, [self.lora_input("standard.safetensors", 1)], root, "missing-job"
            )
            self.assertEqual(report["state"], "dirty")
            self.assertFalse(report["fused"])
            self.assertEqual(pipeline.events.count("unload"), 1)
            self.assertFalse(self.server.pipeline_lora_cache_compatible(
                pipeline, pipeline._xiriai_lora_fusion_state["signature"]
            ))

    def test_mixed_fuses_standard_before_advanced_and_clean_reuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "standard.safetensors", standard_tensors())
            write_tensors(root / "advanced.safetensors", {
                "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                "lora_unet_linear.lokr_w2": torch.eye(2),
            })
            events = []
            pipeline = self.Pipeline(events)
            original_apply = self.server.apply_lora_plan

            def recorded_apply(*args, **kwargs):
                events.append("advanced")
                return original_apply(*args, **kwargs)

            loras = [
                self.lora_input("advanced.safetensors", -0.5),
                self.lora_input("standard.safetensors", 1.0),
            ]
            with mock.patch.object(self.server, "apply_lora_plan", side_effect=recorded_apply):
                first = self.server.configure_loras(pipeline, loras, root, "missing-job")
            self.assertLess(events.index(("fuse", ("lora_1",))), events.index("advanced"))
            self.assertEqual(first["state"], "clean")
            self.assertEqual(first["standard"], 1)
            self.assertEqual(first["lycoris"], 1)

            prior = list(events)
            with mock.patch.object(self.server, "read_lora_header", side_effect=AssertionError("re-read")):
                reused = self.server.configure_loras(pipeline, loras, root, "missing-job")
            self.assertEqual(events, prior)
            self.assertTrue(reused["reused"])

    def test_one_target_index_per_transaction_serves_every_advanced_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "first.safetensors", {
                "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                "lora_unet_linear.lokr_w2": torch.eye(2),
            })
            write_tensors(root / "second.safetensors", {
                "lora_unet_other.lokr_w1": torch.ones(1, 1),
                "lora_unet_other.lokr_w2": torch.eye(2),
            })
            write_tensors(root / "standard.safetensors", standard_tensors())
            loras = [
                self.lora_input("first.safetensors", 1.0),
                self.lora_input("standard.safetensors", 1.0),
                self.lora_input("second.safetensors", -1.0),
            ]
            built = []
            original_build = self.server.build_target_index

            def recorded_build(components):
                index = original_build(components)
                built.append(index)
                return index

            with mock.patch.object(self.server, "build_target_index", side_effect=recorded_build):
                first = self.server.configure_loras(self.Pipeline([]), loras, root, "missing-job")
                # A second pipeline is a second transaction: it builds its own index.
                second = self.server.configure_loras(self.Pipeline([]), loras, root, "missing-job")
            self.assertEqual(len(built), 2)
            self.assertIsNot(built[0], built[1])
            for report in (first, second):
                self.assertEqual(report["state"], "clean")
                self.assertEqual(report["timings"]["target_index_builds"], 1)
                self.assertEqual(report["timings"]["target_lookups"], 2)
                self.assertEqual([item["name"] for item in report["timings"]["advanced_files"]],
                                 ["first.safetensors", "second.safetensors"])
                self.assertIn("target_index_seconds", report["timings"])
                self.assertEqual(report["total_seconds"], report["timings"]["total_seconds"])
                # Diagnostics carry basenames and numbers, never a local path.
                self.assertNotIn(str(root), json.dumps(report, ensure_ascii=False))

    def test_hot_reuse_reads_no_header_builds_no_index_and_folds_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "standard.safetensors", standard_tensors())
            write_tensors(root / "advanced.safetensors", {
                "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                "lora_unet_linear.lokr_w2": torch.eye(2),
            })
            loras = [self.lora_input("standard.safetensors", 0.5), self.lora_input("advanced.safetensors", 0.5)]
            events = []
            pipeline = self.Pipeline(events)
            cold = self.server.configure_loras(pipeline, loras, root, "missing-job")
            prior = list(events)
            refuse = AssertionError("hot reuse touched the mount path")
            with mock.patch.object(self.server, "read_lora_header", side_effect=refuse), mock.patch.object(
                self.server, "build_target_index", side_effect=refuse
            ), mock.patch.object(self.server, "preflight_lora_patch", side_effect=refuse), mock.patch.object(
                self.server, "apply_lora_plan", side_effect=refuse
            ):
                hot = self.server.configure_loras(pipeline, loras, root, "missing-job")
            self.assertEqual(events, prior)
            self.assertTrue(hot["reused"])
            self.assertEqual(hot["elapsed_seconds"], cold["elapsed_seconds"])
            self.assertEqual(
                set(hot["timings"]), {"signature_seconds", "reuse_lookup_seconds", "total_seconds"}
            )
            self.assertLess(hot["total_seconds"], 0.5)

    def test_one_file_standard_plus_lokr_is_wholly_direct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "combined.safetensors", {
                "unet.linear.lora_A.weight": torch.tensor([[1.0, 2.0]]),
                "unet.linear.lora_B.weight": torch.tensor([[3.0], [4.0]]),
                "lora_unet_other.lokr_w1": torch.ones(1, 1),
                "lora_unet_other.lokr_w2": torch.eye(2),
            })
            pipeline = self.Pipeline([])
            pipeline.unet.linear.weight.data.zero_()
            pipeline.unet.other.weight.data.zero_()
            report = self.server.configure_loras(
                pipeline, [self.lora_input("combined.safetensors", 1)], root, "missing-job"
            )
            self.assertEqual(pipeline.events, ["unload"])
            self.assertEqual(report["standard"], 0)
            self.assertEqual(report["lycoris"], 1)
            self.assertEqual(report["targets"], 2)
            self.assertEqual(report["files"][0]["format"], "lora+lokr")
            torch.testing.assert_close(pipeline.unet.other.weight, torch.eye(2))

    def test_file_revision_invalidates_signature_and_order_and_duplicates_are_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_tensors(root / "standard.safetensors", standard_tensors())
            loras = [
                self.lora_input("standard.safetensors", 0.1),
                self.lora_input("standard.safetensors", -0.0),
            ]
            first = self.server.lora_signature(loras, root)
            self.assertEqual(len(first), 2)
            self.assertEqual(first[0][1], float(0.1).hex())
            self.assertEqual(first[1][1], float(-0.0).hex())
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            second = self.server.lora_signature(loras, root)
            self.assertNotEqual(first, second)
            self.assertFalse(Path(first[0][0]).is_absolute())

            # Atomic same-size replacement with the original mtime must still
            # miss the cache through ctime and stable file identity.
            replacement = write_tensors(root / "replacement.safetensors", standard_tensors())
            self.assertEqual(replacement.stat().st_size, stat.st_size)
            time.sleep(0.01)
            os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            os.replace(replacement, path)
            restored = path.stat()
            self.assertEqual(restored.st_size, stat.st_size)
            self.assertEqual(restored.st_mtime_ns, stat.st_mtime_ns)
            third = self.server.lora_signature(loras, root)
            self.assertNotEqual(first, third)
            self.assertEqual(len(first[0]), 7)

    def test_advanced_failure_is_dirty_and_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "advanced.safetensors", {
                "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                "lora_unet_linear.lokr_w2": torch.eye(2),
            })
            pipeline = self.Pipeline([])
            with mock.patch.object(self.server, "apply_lora_plan", side_effect=LoraPatchError("injected")):
                with self.assertRaises(LoraPatchError):
                    self.server.configure_loras(
                        pipeline, [self.lora_input("advanced.safetensors", 1.0)], root, "missing-job"
                    )
            state = pipeline._xiriai_lora_fusion_state
            self.assertEqual(state["state"], "dirty")
            self.assertFalse(self.server.pipeline_lora_cache_compatible(pipeline, state["signature"]))

    def test_mixed_standard_fuse_failure_fails_and_marks_dirty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "standard.safetensors", standard_tensors())
            write_tensors(root / "advanced.safetensors", {
                "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                "lora_unet_linear.lokr_w2": torch.eye(2),
            })
            pipeline = self.Pipeline([], fuse_error=RuntimeError("fuse exploded"))
            with self.assertRaises(LoraPatchError):
                self.server.configure_loras(
                    pipeline,
                    [self.lora_input("standard.safetensors", 1), self.lora_input("advanced.safetensors", 1)],
                    root,
                    "missing-job",
                )
            self.assertEqual(pipeline._xiriai_lora_fusion_state["state"], "dirty")

    def test_direct_patch_prepares_and_restores_cpu_offload_and_8gb_tier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "advanced.safetensors", {
                "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                "lora_unet_linear.lokr_w2": torch.eye(2),
            })
            for tier in ("T1", "T4"):
                with self.subTest(tier=tier):
                    events = []
                    pipeline = self.Pipeline(events)
                    controller = self.Residency(pipeline, events, tier=tier)
                    self.server.loaded_pipeline = pipeline
                    self.server.active_residency = controller
                    self.server.active_memory_strategy = None
                    report = self.server.configure_loras(
                        pipeline, [self.lora_input("advanced.safetensors", 1)], root, "missing-job"
                    )
                    self.assertEqual(report["state"], "clean")
                    self.assertLess(events.index(("residency-release", False, tier)), events.index("unload"))
                    self.assertIn(("residency-reinstall", tier), events)
                    self.server.loaded_pipeline = None
                    self.server.active_residency = None

    def test_standard_only_on_streaming_rungs_releases_once_and_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "one.safetensors", standard_tensors())
            write_tensors(root / "two.safetensors", standard_tensors("lora_unet_other"))
            loras = [self.lora_input("one.safetensors", 1.0), self.lora_input("two.safetensors", 0.5)]
            for tier, released in (("T4", True), ("T3", True), ("T2", True), ("T1", False), ("T0", False)):
                with self.subTest(tier=tier):
                    events = []
                    pipeline = self.Pipeline(events)
                    self.server.loaded_pipeline = pipeline
                    self.server.active_residency = self.Residency(pipeline, events, tier=tier)
                    report = self.server.configure_loras(pipeline, loras, root, "missing-job")
                    self.assertEqual(report["state"], "clean")
                    residency = [event for event in events if isinstance(event, tuple) and event[0].startswith("residency")]
                    if released:
                        # One release before the first adapter operation, one reinstall after the last.
                        self.assertEqual(residency, [("residency-release", False, tier), ("residency-reinstall", tier)])
                        self.assertEqual(events[0], ("residency-release", False, tier))
                        self.assertEqual(events[-1], ("residency-reinstall", tier))
                        self.assertIn("residency_prepare_seconds", report["timings"])
                    else:
                        self.assertEqual(residency, [])
                    self.server.loaded_pipeline = None
                    self.server.active_residency = None

    def test_released_rung_is_restored_on_failure_and_before_mounted_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "one.safetensors", standard_tensors())
            write_tensors(root / "two.safetensors", standard_tensors("lora_unet_other"))
            write_tensors(root / "advanced.safetensors", {
                "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                "lora_unet_linear.lokr_w2": torch.eye(2),
            })
            standard = [self.lora_input("one.safetensors", 1.0), self.lora_input("two.safetensors", 1.0)]
            cases = (
                ("load", standard, {"load_error_at": 2}, None),
                ("set", standard, {"set_error": RuntimeError("partial set")}, None),
                ("fallback", standard, {"fuse_error": RuntimeError("fuse failed")}, None),
                ("advanced", standard[:1] + [self.lora_input("advanced.safetensors", 1.0)], {},
                 LoraPatchError("injected advanced failure")),
            )
            for label, loras, failures, apply_error in cases:
                with self.subTest(case=label):
                    events = []
                    pipeline = self.Pipeline(events, **failures)
                    self.server.loaded_pipeline = pipeline
                    self.server.active_residency = self.Residency(pipeline, events, tier="T4")
                    patch = mock.patch.object(
                        self.server, "apply_lora_plan", side_effect=apply_error
                    ) if apply_error else mock.patch.object(self.server, "apply_lora_plan", wraps=self.server.apply_lora_plan)
                    with patch:
                        if label == "fallback":
                            report = self.server.configure_loras(pipeline, loras, root, "missing-job")
                            self.assertEqual(report["state"], "dirty")
                            self.assertFalse(report["fused"])
                        else:
                            with self.assertRaises(LoraPatchError):
                                self.server.configure_loras(pipeline, loras, root, "missing-job")
                    self.assertEqual(pipeline._xiriai_lora_fusion_state["state"], "dirty")
                    self.assertEqual(events[0], ("residency-release", False, "T4"))
                    self.assertEqual(events[-1], ("residency-reinstall", "T4"))
                    self.assertEqual(events.count(("residency-reinstall", "T4")), 1)
                    self.server.loaded_pipeline = None
                    self.server.active_residency = None

    def test_host_oom_nth_group_failure_and_revision_race_stay_dirty_with_rung_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "standard.safetensors", standard_tensors())
            # Two groups; the second carries a NaN factor, so the first is written before it fails.
            write_tensors(root / "advanced.safetensors", {
                "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                "lora_unet_linear.lokr_w2": torch.eye(2),
                "lora_unet_other.lokr_w1": torch.ones(1, 1),
                "lora_unet_other.lokr_w2": torch.tensor([[1.0, float("nan")], [0.0, 1.0]]),
            })
            write_tensors(root / "clean.safetensors", {
                "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                "lora_unet_linear.lokr_w2": torch.eye(2),
            })
            loras = [self.lora_input("standard.safetensors", 1.0), self.lora_input("advanced.safetensors", 1.0)]
            clean = [self.lora_input("standard.safetensors", 1.0), self.lora_input("clean.safetensors", 1.0)]
            real_signature = self.server.lora_signature
            calls = []

            def racing_signature(items, lora_root):
                calls.append(True)
                value = real_signature(items, lora_root)
                # The second read is the post-apply recheck: a replacement landed in between.
                return value if len(calls) == 1 else value + (("replaced",),)

            cases = (
                ("host-oom", loras, mock.patch.object(
                    self.server, "apply_lora_plan", side_effect=torch.OutOfMemoryError("CPU out of memory")
                ), torch.OutOfMemoryError, None),
                ("nth-group", loras, mock.patch.object(self.server, "update_job"), LoraPatchError, "NaN/Inf"),
                ("revision-race", clean, mock.patch.object(
                    self.server, "lora_signature", side_effect=racing_signature
                ), LoraPatchError, "文件修订"),
            )
            for label, request, patch, error_type, text in cases:
                with self.subTest(case=label):
                    events = []
                    pipeline = self.Pipeline(events)
                    pipeline.unet.linear.weight.data.zero_()
                    self.server.loaded_pipeline = pipeline
                    self.server.active_residency = self.Residency(pipeline, events, tier="T4")
                    with patch, self.assertRaises(error_type) as captured:
                        self.server.configure_loras(pipeline, request, root, "missing-job")
                    if text:
                        self.assertIn(text, str(captured.exception))
                    state = pipeline._xiriai_lora_fusion_state
                    self.assertEqual(state["state"], "dirty")
                    self.assertFalse(self.server.pipeline_lora_cache_compatible(pipeline, state["signature"]))
                    self.assertEqual(events[0], ("residency-release", False, "T4"))
                    self.assertEqual(events.count(("residency-reinstall", "T4")), 1)
                    if label == "nth-group":
                        # Irreversible prefix: group one is in the weights, which is why dirty is final.
                        torch.testing.assert_close(pipeline.unet.linear.weight, torch.eye(2))
                    self.server.loaded_pipeline = None
                    self.server.active_residency = None

    def test_parked_pipeline_recovers_through_controller_and_old_pipeline_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "advanced.safetensors", {
                "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                "lora_unet_linear.lokr_w2": torch.eye(2),
            })
            events = []
            pipeline = self.Pipeline(events)
            controller = self.Residency(pipeline, events, parked=True)
            self.server.loaded_pipeline = pipeline
            self.server.active_residency = controller
            self.server.pipeline_cpu_parked = True

            def restore():
                events.append("parked-restore")
                self.server.pipeline_cpu_parked = False
                controller.parked = False

            with mock.patch.object(self.server, "restore_parked_pipeline", side_effect=restore):
                self.server.configure_loras(
                    pipeline, [self.lora_input("advanced.safetensors", 1)], root, "missing-job"
                )
            self.assertLess(events.index("parked-restore"), events.index(("residency-release", False, "T1")))

            old = self.Pipeline([])
            self.server.loaded_pipeline = self.Pipeline([])
            self.server.active_residency = None
            self.server.pipeline_cpu_parked = False
            with self.assertRaisesRegex(LoraPatchError, "旧 pipeline"):
                self.server.configure_loras(
                    old, [self.lora_input("advanced.safetensors", 1)], root, "missing-job"
                )
            self.assertEqual(old._xiriai_lora_fusion_state["state"], "dirty")

            stuck = self.Pipeline([])
            stuck_controller = self.Residency(stuck, [], parked=True)
            self.server.loaded_pipeline = stuck
            self.server.active_residency = stuck_controller
            self.server.pipeline_cpu_parked = True
            with mock.patch.object(self.server, "restore_parked_pipeline", return_value=None):
                with self.assertRaisesRegex(LoraPatchError, "CPU 停放"):
                    self.server.configure_loras(
                        stuck, [self.lora_input("advanced.safetensors", 1)], root, "missing-job"
                    )
            self.assertEqual(stuck._xiriai_lora_fusion_state["state"], "dirty")

    def test_outer_load_compatibility_rebuilds_dirty_pipeline_even_for_empty_request(self):
        pipeline = self.Pipeline([])
        checkpoint = Path("checkpoint.safetensors")
        self.server.loaded_pipeline = pipeline
        self.server.loaded_checkpoint = str(checkpoint)
        self.server.loaded_family = "sd"
        self.server.loaded_engine = "SD"
        self.server.mark_pipeline_lora_state(pipeline, "dirty", ())
        cleared = []

        def clear():
            cleared.append(True)
            self.server.loaded_pipeline = None

        with mock.patch.object(self.server, "clear_pipeline", side_effect=clear), mock.patch.object(
            self.server, "choose_memory_strategy", side_effect=RuntimeError("stop after rebuild decision")
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after rebuild"):
                self.server.load_pipeline(
                    checkpoint, "sd", "SD", 512, 512, 7.0, 0, "missing-job", loras=()
                )
        self.assertEqual(cleared, [True])

    def test_meta_target_fails_dirty_before_unload_or_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensors(root / "advanced.safetensors", {
                "lora_unet_linear.lokr_w1": torch.ones(1, 1),
                "lora_unet_linear.lokr_w2": torch.eye(2),
            })
            pipeline = self.Pipeline([])
            pipeline.unet = TinyUNet().to("meta")
            with mock.patch.object(self.server, "apply_lora_plan", side_effect=AssertionError("must not apply")):
                with self.assertRaisesRegex(LoraPatchError, "materialize"):
                    self.server.configure_loras(
                        pipeline, [self.lora_input("advanced.safetensors", 1)], root, "missing-job"
                    )
            self.assertEqual(pipeline.events, [])
            self.assertEqual(pipeline._xiriai_lora_fusion_state["state"], "dirty")

    def test_errors_are_bounded_and_hide_absolute_path(self):
        path = r"C:\private\models\secret.safetensors"
        error = lora_patch.readable_lora_error(path, "LoKr", path + " " + "key," * 10_000)
        self.assertLessEqual(len(str(error)), lora_patch.MAX_LORA_ERROR_LENGTH)
        self.assertNotIn(r"C:\private", str(error))
        self.assertIn("secret.safetensors", str(error))

        try:
            raise error
        except LoraPatchError as captured:
            message = self.server.generation_failure_message(captured)
            trace = self.server.generation_failure_trace(captured)
        self.assertLessEqual(len(message), 960)
        self.assertNotIn(r"C:\private", message + trace)
        self.assertNotIn("test_lora_patch.py", trace)
        self.assertNotIn("Traceback (most recent call last)", trace)
        injected = LoraPatchError(r"dependency failed at D:\private\source\loader.py " + "key," * 100_000)
        self.assertNotIn(r"D:\private", str(injected))
        self.assertLessEqual(len(str(injected)), lora_patch.MAX_LORA_ERROR_LENGTH)

    def test_standard_loader_key_dump_is_omitted_from_error_job_trace_and_temp_log(self):
        omitted = "上游标准 LoRA 加载失败（键列表已省略）"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_root = root / "logs"
            write_tensors(root / "standard.safetensors", standard_tensors())
            unique_keys = [
                f"base_model.unet.block.{index}.to_q.lora_A.adapter_{index}.weight"
                for index in range(5_000)
            ]
            dumped_keys = unique_keys + unique_keys
            upstream = RuntimeError("Adapter state mismatch:\n" + ",\n".join(dumped_keys))

            direct_readable = lora_patch.readable_lora_error(
                root / "standard.safetensors", "标准 LoRA", upstream
            )
            direct_exception = LoraPatchError(str(upstream))
            dense_readable = lora_patch.readable_lora_error(
                root / "standard.safetensors",
                "标准 LoRA",
                "Adapter state mismatch: " + ",".join(unique_keys[:20]),
            )
            exact_diffusers = lora_patch.readable_lora_error(
                root / "standard.safetensors",
                "标准 LoRA",
                "The following keys have not been correctly renamed: "
                + unique_keys[0],
            )
            short_oom = lora_patch.readable_lora_error(
                root / "standard.safetensors",
                "标准 LoRA",
                "CUDA out of memory while allocating 64 MiB",
            )
            self.assertIn("CUDA out of memory while allocating 64 MiB", str(short_oom))
            self.assertNotIn(omitted, str(short_oom))
            self.assertIn(omitted, str(exact_diffusers))
            self.assertNotIn(unique_keys[0], str(exact_diffusers))
            self.assertIn("standard.safetensors", str(direct_readable))
            self.assertIn("标准 LoRA", str(direct_readable))

            pipeline = self.Pipeline([], load_error_at=1, load_error=upstream)
            job_id = "lora-key-dump-test"
            with self.server.jobs_lock:
                self.server.jobs[job_id] = {"phase": "Loading LoRA adapters", "stage": "loading"}
            try:
                with self.assertRaises(LoraPatchError) as captured:
                    self.server.configure_loras(
                        pipeline,
                        [self.lora_input("standard.safetensors", 1.0)],
                        root,
                        job_id,
                    )
                error = captured.exception
                message = self.server.generation_failure_message(error)
                try:
                    raise error
                except LoraPatchError as raised:
                    trace = self.server.generation_failure_trace(raised)
                self.server.update_job(job_id, status="error", error=message)
                with self.server.jobs_lock:
                    job_json = json.dumps(self.server.jobs[job_id], ensure_ascii=False)

                request = self.server.GenerateInput(
                    engine="SD",
                    checkpoint="base.safetensors",
                    prompt="test",
                    width=512,
                    height=512,
                    steps=1,
                    cfg=7.0,
                    denoise=1.0,
                    seed=1,
                    sampler="euler",
                    scheduler="normal",
                    loras=[self.lora_input("standard.safetensors", 1.0)],
                )
                with mock.patch.object(self.server, "LOG_DIRECTORY", log_root), mock.patch.object(
                    self.server, "cuda_memory_diagnostics", return_value={}
                ), mock.patch.object(
                    self.server, "system_memory_diagnostics", return_value={}
                ), mock.patch.object(
                    self.server, "pipeline_component_diagnostics", return_value={}
                ), mock.patch.object(
                    self.server, "background_removal_status", return_value={}
                ), mock.patch.object(
                    self.server.rtx_vsr, "status", return_value={}
                ), mock.patch.object(
                    self.server, "installed_version", return_value=None
                ), mock.patch.object(
                    self.server, "cuda_device_name", return_value="test-device"
                ):
                    self.server.write_generation_failure_log(job_id, request, error)
                logs = list(log_root.glob("*.log"))
                self.assertEqual(len(logs), 1)
                log_text = logs[0].read_text(encoding="utf-8")
            finally:
                with self.server.jobs_lock:
                    self.server.jobs.pop(job_id, None)

        payloads = (
            str(direct_readable),
            str(direct_exception),
            str(dense_readable),
            str(error),
            message,
            trace,
            job_json,
            log_text,
        )
        for payload in payloads:
            self.assertNotIn("base_model.unet.block.", payload)
            self.assertNotIn("lora_A.adapter_", payload)
            self.assertIn(omitted, payload)
        for payload in payloads[:6]:
            self.assertLessEqual(len(payload), 1_100)
        self.assertLess(len(job_json), 8_000)
        self.assertLess(len(log_text), 32_000)


@unittest.skipUnless(torch.cuda.is_available(), "the device compute paths need an accelerator")
class DeviceComputeTests(unittest.TestCase):
    """QPlan P5 Phase C: the accelerator takes the dense part of a host-resident mount."""

    def setUp(self):
        # The server's balanced CUDA math: TF32 on, which neither device path may inherit.
        self.saved = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    def tearDown(self):
        torch.set_float32_matmul_precision(self.saved[0])
        torch.backends.cuda.matmul.allow_tf32 = self.saved[1]

    def test_advanced_apply_on_device_is_bit_for_bit_the_host_result_for_every_family(self):
        generator = torch.Generator().manual_seed(7)

        def rand(*shape):
            return torch.randn(*shape, generator=generator) * 0.05

        fixtures = {
            "lora": {"lora_unet_linear.lora_down.weight": rand(3, 8), "lora_unet_linear.lora_up.weight": rand(6, 3),
                     "lora_unet_linear.alpha": torch.tensor(2.0)},
            "loha": {"lora_unet_linear.hada_w1_a": rand(6, 2), "lora_unet_linear.hada_w1_b": rand(2, 8),
                     "lora_unet_linear.hada_w2_a": rand(6, 2), "lora_unet_linear.hada_w2_b": rand(2, 8),
                     "lora_unet_linear.alpha": torch.tensor(1.0)},
            "lokr_full": {"lora_unet_linear.lokr_w1": rand(2, 2), "lora_unet_linear.lokr_w2": rand(3, 4)},
            "lokr_lowrank": {"lora_unet_linear.lokr_w1": rand(2, 2), "lora_unet_linear.lokr_w2_a": rand(3, 2),
                             "lora_unet_linear.lokr_w2_b": rand(2, 4), "lora_unet_linear.alpha": torch.tensor(1.5)},
            "dora": {"lora_unet_linear.lora_down.weight": rand(3, 8), "lora_unet_linear.lora_up.weight": rand(6, 3),
                     "lora_unet_linear.dora_scale": rand(6, 1).abs() + 0.5},
        }
        for label, tensors in fixtures.items():
            for multiplier in (0.4, -1.25):
                with self.subTest(family=label, multiplier=multiplier), tempfile.TemporaryDirectory() as directory:
                    path = write_tensors(Path(directory) / f"{label}.safetensors", tensors)

                    class Module(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.linear = torch.nn.Linear(8, 6, bias=False)

                    host, device = Module().half(), Module().half()
                    device.load_state_dict(host.state_dict())
                    plan = preflight_lora_patch(path, {"unet": host}, "sd")
                    apply_lora_plan(path, plan, {"unet": host}, multiplier)
                    apply_lora_plan(path, plan, {"unet": device}, multiplier, compute_device="cuda")
                    self.assertEqual(device.linear.weight.device.type, "cpu")
                    self.assertTrue(torch.equal(host.linear.weight, device.linear.weight))
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)

    def test_queued_device_checks_report_the_same_first_failure_as_the_host(self):
        nan = float("nan")
        cases = {
            "factor": ({"lora_unet_linear.lokr_w1": torch.tensor([[nan]]), "lora_unet_linear.lokr_w2": torch.eye(2)}, None),
            "alpha": ({"lora_unet_linear.lokr_w1": torch.ones(1, 1), "lora_unet_linear.lokr_w2": torch.eye(2),
                       "lora_unet_linear.alpha": torch.tensor(float("inf"))}, None),
            "base": ({"lora_unet_linear.lokr_w1": torch.ones(1, 1), "lora_unet_linear.lokr_w2": torch.eye(2)}, nan),
            # Both a factor and the base are bad: the factor is checked first on either path.
            "double": ({"lora_unet_linear.lokr_w1": torch.tensor([[nan]]), "lora_unet_linear.lokr_w2": torch.eye(2)}, nan),
            "cast": ({"lora_unet_linear.lokr_w1": torch.tensor([[1e6]]), "lora_unet_linear.lokr_w2": torch.eye(2)}, None),
        }
        for label, (tensors, base_value) in cases.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                path = write_tensors(Path(directory) / f"{label}.safetensors", tensors)
                messages = []
                for compute_device in (None, "cuda"):
                    module = TinyUNet().half()
                    if base_value is not None:
                        module.linear.weight.data[0, 0] = base_value
                    before = module.linear.weight.detach().clone()
                    plan = preflight_lora_patch(path, {"unet": module}, "sd")
                    with self.assertRaises(LoraPatchError) as captured:
                        apply_lora_plan(path, plan, {"unet": module}, 1.0, compute_device=compute_device)
                    messages.append(str(captured.exception))
                    # Nothing is written when a check fails, on either path.
                    self.assertTrue(torch.equal(module.linear.weight.view(torch.int16), before.view(torch.int16)))
                self.assertEqual(messages[0], messages[1])

    def test_device_patch_refuses_before_writing_exactly_like_the_host(self):
        weight = torch.zeros(2, 2, dtype=torch.float16)
        before = weight.clone()
        for delta, text in ((torch.full((2, 2), float("nan")), "NaN/Inf"), (torch.tensor([[1e10, 0.0], [0.0, 0.0]]), "回写")):
            with self.subTest(text=text), self.assertRaisesRegex(LoraPatchError, text):
                apply_lora_patch(weight, delta.cuda(), 1.0, compute_device="cuda")
            self.assertTrue(torch.equal(weight, before))

    def peft_model(self, *, conv=False, bias=False):
        from peft import LoraConfig, inject_adapter_in_model

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Linear(16, 12)
                self.second = torch.nn.Linear(12, 10)
                self.conv = torch.nn.Conv2d(4, 6, 3)

        torch.manual_seed(3)
        model = Model()
        targets = ["first", "second"] + (["conv"] if conv else [])
        for index, (rank, alpha) in enumerate(((4, 8), (2, 2))):
            config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets, init_lora_weights=False,
                                lora_bias=bias and index == 1)
            inject_adapter_in_model(config, model, adapter_name=f"a{index}")
        model.half()
        for module in model.modules():
            if hasattr(module, "set_scale"):
                module.set_scale("a0", 0.6)
                module.set_scale("a1", -1.1)
        return model

    def test_standard_fold_matches_peft_merge_and_keeps_its_bookkeeping(self):
        reference = self.peft_model(conv=True)
        candidate = self.peft_model(conv=True)
        candidate.load_state_dict(reference.state_dict())
        for module in reference.modules():
            if hasattr(module, "merge") and hasattr(module, "lora_A"):
                module.merge(adapter_names=["a0", "a1"])
        counts = lora_patch.fold_peft_adapters({"unet": candidate, "text_encoder": None}, ["a0", "a1"], "cuda")
        self.assertEqual((counts["folded_layers"], counts["delegated_layers"]), (2, 1))
        self.assertEqual(counts["merged_adapters"], {"a0", "a1"})
        for name in ("first", "second", "conv"):
            with self.subTest(layer=name):
                ours = getattr(candidate, name)
                theirs = getattr(reference, name)
                self.assertEqual(ours.merged_adapters, theirs.merged_adapters)
                self.assertEqual(ours.get_base_layer().weight.device.type, "cpu")
                torch.testing.assert_close(ours.get_base_layer().weight, theirs.get_base_layer().weight, atol=0, rtol=2e-3)
        # The process's CUDA math mode is untouched once the fold returns.
        self.assertEqual(torch.get_float32_matmul_precision(), "high")
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)

    def test_device_out_of_memory_hands_the_rest_to_the_host_with_identical_bytes(self):
        tensors = {
            "lora_unet_linear.lokr_w1": torch.tensor([[-1.5]]),
            "lora_unet_linear.lokr_w2": torch.tensor([[0.25, -0.5], [1.5, 2.0]]),
            "lora_unet_other.lokr_w1": torch.tensor([[2.0]]),
            "lora_unet_other.lokr_w2": torch.tensor([[0.5, 1.0], [-1.0, 0.25]]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = write_tensors(Path(directory) / "oom.safetensors", tensors)
            host, device = TinyUNet().half(), TinyUNet().half()
            device.load_state_dict(host.state_dict())
            plan = preflight_lora_patch(path, {"unet": host}, "sd")
            apply_lora_plan(path, plan, {"unet": host}, 0.7)
            original = lora_patch.lora_delta
            device_calls = []

            def exhausted(group, values, **kwargs):
                if kwargs.get("product_device") is not None:
                    device_calls.append(group.name)
                    raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
                return original(group, values, **kwargs)

            stats = {}
            with mock.patch.object(lora_patch, "lora_delta", side_effect=exhausted):
                apply_lora_plan(path, plan, {"unet": device}, 0.7, compute_device="cuda", stats=stats)
            # The first group hit the wall; it and the second went to the host, and nothing was lost.
            self.assertEqual(device_calls, [plan.groups[0].group.name])
            self.assertEqual(stats, {"device_fallbacks": 1})
            self.assertTrue(torch.equal(host.linear.weight, device.linear.weight))
            self.assertTrue(torch.equal(host.other.weight, device.other.weight))

        model = self.peft_model()
        reference = self.peft_model()
        reference.load_state_dict(model.state_dict())
        for module in reference.modules():
            if hasattr(module, "merge") and hasattr(module, "lora_A"):
                module.merge(adapter_names=["a0", "a1"])
        with mock.patch.object(
            lora_patch, "_folded_on_device", side_effect=torch.OutOfMemoryError("CUDA out of memory")
        ) as folded:
            counts = lora_patch.fold_peft_adapters({"unet": model}, ["a0", "a1"], "cuda")
        self.assertEqual(folded.call_count, 1)
        self.assertEqual((counts["folded_layers"], counts["delegated_layers"], counts["device_fallbacks"]), (0, 2, 1))
        for name in ("first", "second"):
            self.assertTrue(torch.equal(getattr(model, name).get_base_layer().weight,
                                        getattr(reference, name).get_base_layer().weight))
            self.assertEqual(getattr(model, name).merged_adapters, ["a0", "a1"])
        # Anything that is not an out-of-memory condition is a real failure and still raises.
        with mock.patch.object(lora_patch, "_folded_on_device", side_effect=RuntimeError("device lost")):
            with self.assertRaisesRegex(RuntimeError, "device lost"):
                lora_patch.fold_peft_adapters({"unet": self.peft_model()}, ["a0", "a1"], "cuda")

    def test_fold_delegates_bias_and_device_resident_layers_to_peft(self):
        model = self.peft_model(bias=True)
        counts = lora_patch.fold_peft_adapters({"unet": model}, ["a0", "a1"], "cuda")
        self.assertEqual((counts["folded_layers"], counts["delegated_layers"]), (0, 2))
        resident = self.peft_model().cuda()
        counts = lora_patch.fold_peft_adapters({"unet": resident}, ["a0", "a1"], "cuda")
        self.assertEqual((counts["folded_layers"], counts["delegated_layers"]), (0, 2))
        self.assertEqual(resident.first.merged_adapters, ["a0", "a1"])


@unittest.skipUnless(torch.cuda.is_available(), "sequential CPU offload needs an accelerator")
class RealSequentialOffloadFoldTests(unittest.TestCase):
    """A standard-only fold on the LOW_VRAM per-submodule rung must reach the real weights.

    Before QPlan P5, Diffusers' per-load hook reinstall left every weight on the meta device and the
    fold was silently lost while the record said it was fused.
    """

    UNET_KEY = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
    TE_KEY = "text_model.encoder.layers.0.self_attn.q_proj"

    def tiny_pipeline(self):
        from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64), layers_per_block=1, sample_size=8, in_channels=4, out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), attention_head_dim=(2, 4),
            use_linear_projection=True, addition_embed_type="text_time", addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2), projection_class_embeddings_input_dim=80,
            cross_attention_dim=64, norm_num_groups=8,
        )
        vae = AutoencoderKL(
            block_out_channels=[32, 64], in_channels=3, out_channels=3, latent_channels=4, norm_num_groups=8,
            down_block_types=["DownEncoderBlock2D"] * 2, up_block_types=["UpDecoderBlock2D"] * 2,
        )
        config = CLIPTextConfig(
            bos_token_id=0, eos_token_id=2, hidden_size=32, intermediate_size=37, layer_norm_eps=1e-05,
            num_attention_heads=4, num_hidden_layers=2, pad_token_id=1, vocab_size=1000, hidden_act="gelu",
            projection_dim=32,
        )
        pipeline = StableDiffusionXLPipeline(
            vae=vae, text_encoder=CLIPTextModel(config), text_encoder_2=CLIPTextModelWithProjection(config),
            tokenizer=None, tokenizer_2=None, unet=unet, scheduler=EulerDiscreteScheduler(),
        )
        return pipeline.to(torch.float16)

    def test_per_submodule_rung_fold_reaches_real_weights_and_rung_is_reinstalled(self):
        from backend import inference_server as server

        for device_path in (True, False):
            with self.subTest(device_path=device_path):
                if device_path:
                    self.fold_on_per_submodule_rung(server)
                else:
                    with mock.patch.object(server, "lora_compute_device", return_value=None):
                        self.fold_on_per_submodule_rung(server)

    def fold_on_per_submodule_rung(self, server):
        from backend.residency import ResidencyController

        saved = (server.loaded_pipeline, server.active_residency, server.loaded_family,
                 server.pipeline_cpu_parked, server.active_memory_strategy)
        pipeline = self.tiny_pipeline()
        unet_base = pipeline.unet.get_parameter(self.UNET_KEY + ".weight").detach().float().clone()
        te_base = pipeline.text_encoder.get_parameter(self.TE_KEY + ".weight").detach().float().clone()
        generator = torch.Generator().manual_seed(1)
        unet_a, unet_b = torch.randn(4, 64, generator=generator) * 0.1, torch.randn(64, 4, generator=generator) * 0.1
        te_a, te_b = torch.randn(4, 32, generator=generator) * 0.1, torch.randn(32, 4, generator=generator) * 0.1
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_tensors(root / "native.safetensors", {
                    f"unet.{self.UNET_KEY}.lora_A.weight": unet_a,
                    f"unet.{self.UNET_KEY}.lora_B.weight": unet_b,
                    f"text_encoder.{self.TE_KEY}.lora_A.weight": te_a,
                    f"text_encoder.{self.TE_KEY}.lora_B.weight": te_b,
                })
                pipeline.enable_sequential_cpu_offload(device="cuda")
                controller = ResidencyController(pipeline, reserved_bytes=0, free_memory=lambda: 0, tier="T4")
                server.loaded_pipeline = pipeline
                server.active_residency = controller
                server.loaded_family = "sdxl"
                server.pipeline_cpu_parked = False
                server.active_memory_strategy = None
                report = server.configure_loras(
                    pipeline, [server.LoraInput(path="native.safetensors", weight=1.0)], root, "missing-job",
                )
            self.assertEqual(report["state"], "clean")
            self.assertTrue(report["fused"])
            self.assertEqual(report["timings"]["lora_compute"],
                             "host" if server.lora_compute_device() is None else "cuda")
            self.assertEqual(controller.tier, "T4")
            # The rung is back: weights sit on the meta device behind accelerate's hooks again.
            self.assertEqual(pipeline.unet.get_parameter(self.UNET_KEY + ".weight").device.type, "meta")
            pipeline.remove_all_hooks()
            pipeline.to("cpu", silence_dtype_warnings=True)
            unet_fused = pipeline.unet.get_parameter(self.UNET_KEY + ".weight").detach().float()
            te_fused = pipeline.text_encoder.get_parameter(self.TE_KEY + ".weight").detach().float()
            torch.testing.assert_close(unet_fused, unet_base + unet_b @ unet_a, atol=2e-3, rtol=0)
            torch.testing.assert_close(te_fused, te_base + te_b @ te_a, atol=2e-3, rtol=0)
        finally:
            (server.loaded_pipeline, server.active_residency, server.loaded_family,
             server.pipeline_cpu_parked, server.active_memory_strategy) = saved
            torch.cuda.empty_cache()


class RealMatchaAuditTests(unittest.TestCase):
    MATCHA = Path(r"D:\.XAIG\il\Style_九V6.0\matcha-000177.safetensors")
    CONFIG = Path(
        r"D:\.XAIG\XiriaCanvas AI\.cache\huggingface\hub\models--stabilityai--stable-diffusion-xl-base-1.0"
        r"\snapshots\462165984030d82259a11f4367a4eed129e94a7b\unet"
    )

    @unittest.skipUnless(MATCHA.is_file(), "real matcha header is not installed")
    def test_real_matcha_header_has_788_750_38_groups(self):
        header = read_lora_header(self.MATCHA)
        groups = build_lora_groups(header)
        self.assertEqual(len(groups), 788)
        self.assertEqual(sum(group.family == "lokr_full" for group in groups), 750)
        self.assertEqual(sum(group.family == "lokr_lowrank" for group in groups), 38)

    @unittest.skipUnless(MATCHA.is_file() and CONFIG.is_dir(), "real matcha/config is not installed")
    def test_real_matcha_all_788_targets_map_on_meta_sdxl(self):
        from accelerate import init_empty_weights
        from diffusers import UNet2DConditionModel

        config = UNet2DConditionModel.load_config(str(self.CONFIG))
        with init_empty_weights():
            unet = UNet2DConditionModel.from_config(config)
        header = read_lora_header(self.MATCHA)
        groups = build_lora_groups(header)
        targets = {
            resolve_target(
                {"unet": unet}, group.name, "sdxl", group.expected_shape,
                file_name=header.file_name,
            ).qualified_name
            for group in groups
        }
        self.assertEqual(len(targets), 788)


if __name__ == "__main__":
    unittest.main()
