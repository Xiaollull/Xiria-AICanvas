import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from backend import anima_pipeline
from backend.anima_pipeline import (
    ANIMA_MAX_SEQUENCE_LENGTH,
    AnimaCosmosAttnProcessor,
    _discard_module_storage,
    AnimaRuntime,
    CosmosPAGIdentitySelfAttnProcessor,
    PreparedAnimaConditioning,
    _GroupCfgBatchOom,
    _decoded_tensor_to_image,
    _fuse_anima_lora_state_dict,
    _fused_rms_norm,
    _lora_descriptors,
    _split_half_rotary,
    euler_ancestral_rf_step,
    euler_rf_step,
    install_anima_cosmos_attention_processor,
    normalize_checkpoint_keys,
    shifted_sigmas,
    split_llm_adapter_state_dict,
)
from backend.anima_sampling import (
    ANIMA_SAMPLERS,
    ANIMA_SCHEDULERS,
    anima_sigma_schedule,
    prepare_anima_refinement_sigmas,
)
from backend.inference_server import GenerationCancelled


class AnimaCheckpointTests(unittest.TestCase):
    def test_normalizes_exactly_one_known_prefix(self):
        tensor = torch.tensor(1.0)
        normalized = normalize_checkpoint_keys(
            {
                "net.blocks.0.weight": tensor,
                "model.diffusion_model.final.weight": tensor,
                "diffusion_model.net.nested.weight": tensor,
                "plain.weight": tensor,
            }
        )
        self.assertEqual(
            list(normalized),
            ["blocks.0.weight", "final.weight", "net.nested.weight", "plain.weight"],
        )

    def test_prefix_normalization_rejects_collisions(self):
        with self.assertRaisesRegex(ValueError, "collision"):
            normalize_checkpoint_keys({"net.foo": torch.tensor(1), "foo": torch.tensor(2)})

    def test_splits_adapter_before_transformer_conversion(self):
        transformer, adapter = split_llm_adapter_state_dict(
            {
                "blocks.0.weight": torch.tensor(1),
                "llm_adapter.embed.weight": torch.tensor(2),
                "llm_adapter.norm.weight": torch.tensor(3),
            }
        )
        self.assertEqual(list(transformer), ["blocks.0.weight"])
        self.assertEqual(list(adapter), ["embed.weight", "norm.weight"])


class AnimaLoraFusionTests(unittest.TestCase):
    def setUp(self):
        self.diffusion = {
            "blocks.0.attn.to_q.weight": torch.zeros((2, 3), dtype=torch.float16),
            "llm_adapter.blocks.0.cross_attn.q_proj.weight": torch.zeros((2, 3), dtype=torch.float16),
        }
        self.qwen = {"layers.0.self_attn.q_proj.weight": torch.zeros((2, 3), dtype=torch.float16)}
        self.down = torch.tensor([[1.0, 2.0, 3.0]])
        self.up = torch.tensor([[2.0], [4.0]])
        self.delta = self.up @ self.down

    def test_native_and_official_forms_each_target_dit_adapter_and_qwen(self):
        native_dit_official_adapter_native_qwen = {
            "lora_unet_blocks_0_attn_to_q.lora_down.weight": self.down,
            "lora_unet_blocks_0_attn_to_q.lora_up.weight": self.up,
            "diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.lora_A.weight": self.down,
            "diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.lora_B.weight": self.up,
            "lora_te_layers_0_self_attn_q_proj.lora_down.weight": self.down,
            "lora_te_layers_0_self_attn_q_proj.lora_up.weight": self.up,
        }
        diffusion, qwen = _fuse_anima_lora_state_dict(
            self.diffusion, self.qwen, native_dit_official_adapter_native_qwen, 1.0
        )
        self.assertTrue(torch.equal(diffusion["blocks.0.attn.to_q.weight"], self.delta.to(torch.float16)))
        self.assertTrue(
            torch.equal(diffusion["llm_adapter.blocks.0.cross_attn.q_proj.weight"], self.delta.to(torch.float16))
        )
        self.assertTrue(torch.equal(qwen["layers.0.self_attn.q_proj.weight"], self.delta.to(torch.float16)))

        official_dit_native_adapter_official_qwen = {
            "diffusion_model.blocks.0.attn.to_q.lora_A.weight": self.down,
            "diffusion_model.blocks.0.attn.to_q.lora_B.weight": self.up,
            "lora_unet_llm_adapter_blocks_0_cross_attn_q_proj.lora_down.weight": self.down,
            "lora_unet_llm_adapter_blocks_0_cross_attn_q_proj.lora_up.weight": self.up,
            "text_encoders.qwen3_06b.transformer.model.layers.0.self_attn.q_proj.lora_A.weight": self.down,
            "text_encoders.qwen3_06b.transformer.model.layers.0.self_attn.q_proj.lora_B.weight": self.up,
        }
        diffusion, qwen = _fuse_anima_lora_state_dict(
            self.diffusion, self.qwen, official_dit_native_adapter_official_qwen, 1.0
        )
        self.assertTrue(torch.equal(diffusion["blocks.0.attn.to_q.weight"], self.delta.to(torch.float16)))
        self.assertTrue(
            torch.equal(diffusion["llm_adapter.blocks.0.cross_attn.q_proj.weight"], self.delta.to(torch.float16))
        )
        self.assertTrue(torch.equal(qwen["layers.0.self_attn.q_proj.weight"], self.delta.to(torch.float16)))

    def test_analyze_and_gpu_apply_match_fuse_exactly(self):
        from torch import nn as torch_nn

        from backend.anima_pipeline import (
            _analyze_anima_lora_patch,
            _apply_anima_lora_groups_on_gpu,
        )

        lora = {
            "lora_unet_blocks_0_attn_to_q.lora_down.weight": self.down,
            "lora_unet_blocks_0_attn_to_q.lora_up.weight": self.up,
            "diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.lora_A.weight": self.down,
            "diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.lora_B.weight": self.up,
            "lora_te_layers_0_self_attn_q_proj.lora_down.weight": self.down,
            "lora_te_layers_0_self_attn_q_proj.lora_up.weight": self.up,
        }
        fused_diffusion, fused_qwen = _fuse_anima_lora_state_dict(self.diffusion, self.qwen, lora, 1.0)

        specs = _analyze_anima_lora_patch(lora, list(self.diffusion), list(self.qwen), 1.0)
        weights = {
            "diffusion|blocks.0.attn.to_q.weight": torch_nn.Parameter(
                torch.zeros((2, 3), dtype=torch.float16), requires_grad=False
            ),
            "diffusion|llm_adapter.blocks.0.cross_attn.q_proj.weight": torch_nn.Parameter(
                torch.zeros((2, 3), dtype=torch.float16), requires_grad=False
            ),
            "qwen|layers.0.self_attn.q_proj.weight": torch_nn.Parameter(
                torch.zeros((2, 3), dtype=torch.float16), requires_grad=False
            ),
        }

        def resolver(family, target):
            return weights[f"{family}|{target}"]

        _apply_anima_lora_groups_on_gpu(
            specs, resolver, 1.0, torch.device("cpu"), torch.float16, label="test"
        )
        self.assertTrue(torch.equal(weights["diffusion|blocks.0.attn.to_q.weight"], fused_diffusion["blocks.0.attn.to_q.weight"]))
        self.assertTrue(
            torch.equal(
                weights["diffusion|llm_adapter.blocks.0.cross_attn.q_proj.weight"],
                fused_diffusion["llm_adapter.blocks.0.cross_attn.q_proj.weight"],
            )
        )
        self.assertTrue(
            torch.equal(weights["qwen|layers.0.self_attn.q_proj.weight"], fused_qwen["layers.0.self_attn.q_proj.weight"])
        )

    def test_full_matrix_lokr_analyzes_without_borrowing_another_group_alpha(self):
        # A full-matrix LoKr has no rank to normalise by, so the branch that
        # handles it validates the file's alpha rather than using it as a scale.
        # It also never assigned `alpha`, which raised UnboundLocalError when
        # such a group was analyzed first and silently inherited the previous
        # group's alpha when it was not — the GPU streaming route only.
        from torch import nn as torch_nn

        from backend.anima_pipeline import (
            _analyze_anima_lora_patch,
            _apply_anima_lora_groups_on_gpu,
        )

        w1 = torch.tensor([[1.0]])
        w2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        full_first = {
            "lora_unet_blocks_0_attn_to_q.lokr_w1": w1,
            "lora_unet_blocks_0_attn_to_q.lokr_w2": w2,
            "lora_unet_blocks_0_attn_to_q.alpha": torch.tensor(999.0),
            "lora_te_layers_0_self_attn_q_proj.lora_down.weight": self.down,
            "lora_te_layers_0_self_attn_q_proj.lora_up.weight": self.up,
            "lora_te_layers_0_self_attn_q_proj.alpha": torch.tensor(4.0),
        }

        specs = _analyze_anima_lora_patch(full_first, list(self.diffusion), list(self.qwen), 1.0)
        by_kind = {spec["kind"]: spec for spec in specs}
        # The declared 999.0 is validated and then not used as a scale, so the
        # recorded alpha is the scale that will actually be applied.
        self.assertEqual(by_kind["lokr_full"]["alpha"], 1.0)

        # Order must not decide the value: the same group analyzed after a
        # group that does carry an alpha still records its own.
        reordered = {
            "lora_te_layers_0_self_attn_q_proj.lora_down.weight": self.down,
            "lora_te_layers_0_self_attn_q_proj.lora_up.weight": self.up,
            "lora_te_layers_0_self_attn_q_proj.alpha": torch.tensor(4.0),
            "lora_unet_blocks_0_attn_to_q.lokr_w1": w1,
            "lora_unet_blocks_0_attn_to_q.lokr_w2": w2,
            "lora_unet_blocks_0_attn_to_q.alpha": torch.tensor(999.0),
        }
        after = {spec["kind"]: spec for spec in _analyze_anima_lora_patch(reordered, list(self.diffusion), list(self.qwen), 1.0)}
        self.assertEqual(after["lokr_full"]["alpha"], 1.0)

        # And the streaming route still produces exactly what the fuse route does.
        fused_diffusion, _fused_qwen = _fuse_anima_lora_state_dict(self.diffusion, self.qwen, full_first, -0.5)
        weights = {
            "diffusion|blocks.0.attn.to_q.weight": torch_nn.Parameter(
                torch.zeros((2, 3), dtype=torch.float16), requires_grad=False
            ),
            "qwen|layers.0.self_attn.q_proj.weight": torch_nn.Parameter(
                torch.zeros((2, 3), dtype=torch.float16), requires_grad=False
            ),
        }
        _apply_anima_lora_groups_on_gpu(
            specs, lambda family, target: weights[f"{family}|{target}"], -0.5, torch.device("cpu"), torch.float16, label="test"
        )
        self.assertTrue(
            torch.equal(weights["diffusion|blocks.0.attn.to_q.weight"], fused_diffusion["blocks.0.attn.to_q.weight"])
        )

    def test_every_checkpoint_key_maps_to_a_live_parameter_through_the_real_converter(self):
        # The translation between the checkpoint's namespace and the modules that
        # hold the weights is the converter the loader ran. Deriving the map from
        # it rather than re-implementing a corner of it by hand is what closes the
        # whole class: only `cross_attn`, `self_attn` and `mlp` used to be mapped,
        # so `adaln_modulation_*`, the patch embedder and the final layer had no
        # live target and failed mid-sampling.
        from diffusers.loaders.single_file_utils import (
            convert_cosmos_transformer_checkpoint_to_diffusers as convert,
        )

        from backend.anima_pipeline import _lora_target_modules

        checkpoint_keys = [
            "blocks.0.self_attn.q_proj.weight",
            "blocks.0.cross_attn.k_proj.weight",
            "blocks.0.cross_attn.output_proj.weight",
            "blocks.0.adaln_modulation_self_attn.1.weight",
            "blocks.0.adaln_modulation_cross_attn.1.weight",
            "blocks.0.adaln_modulation_cross_attn.2.weight",
            "blocks.0.adaln_modulation_mlp.2.weight",
            "blocks.0.mlp.layer1.weight",
            "blocks.0.mlp.layer2.weight",
            "x_embedder.proj.1.weight",
            "final_layer.adaln_modulation.1.weight",
            "final_layer.linear.weight",
            "t_embedder.1.weight",
            "llm_adapter.blocks.0.cross_attn.k_proj.weight",
            "llm_adapter.out_proj.weight",
        ]
        targets = _lora_target_modules(checkpoint_keys, convert)

        # Every key resolves, including the kinds the hand-written map never had.
        self.assertEqual(sorted(targets), sorted(checkpoint_keys))
        self.assertEqual(targets["blocks.0.adaln_modulation_cross_attn.1.weight"], ("transformer", "transformer_blocks.0.norm2.linear_1.weight"))
        self.assertEqual(targets["blocks.0.cross_attn.k_proj.weight"], ("transformer", "transformer_blocks.0.attn2.to_k.weight"))
        self.assertEqual(targets["blocks.0.cross_attn.output_proj.weight"], ("transformer", "transformer_blocks.0.attn2.to_out.0.weight"))
        self.assertEqual(targets["blocks.0.mlp.layer1.weight"], ("transformer", "transformer_blocks.0.ff.net.0.proj.weight"))
        self.assertEqual(targets["final_layer.linear.weight"], ("transformer", "proj_out.weight"))
        # The bridge module keeps its own namespace and its own attribute.
        self.assertEqual(targets["llm_adapter.blocks.0.cross_attn.k_proj.weight"], ("llm_adapter", "blocks.0.cross_attn.k_proj.weight"))
        self.assertEqual(targets["llm_adapter.out_proj.weight"], ("llm_adapter", "out_proj.weight"))
        # A key the converter drops is not a weight of the model that runs.
        self.assertNotIn("pos_embedder.seq", _lora_target_modules(["pos_embedder.seq"], convert))

    def test_llm_adapter_lora_resolves_on_its_own_module_and_is_patched_with_the_text_pass(self):
        from torch import nn as torch_nn

        from diffusers.loaders.single_file_utils import (
            convert_cosmos_transformer_checkpoint_to_diffusers as convert,
        )

        from backend.anima_pipeline import AnimaRuntime, _lora_target_modules

        class Adapter(torch_nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch_nn.ModuleList([torch_nn.Module()])
                self.blocks[0].cross_attn = torch_nn.Module()
                self.blocks[0].cross_attn.k_proj = torch_nn.Linear(8, 8, bias=False)

        class Transformer(torch_nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = torch_nn.ModuleList([torch_nn.Module()])
                self.transformer_blocks[0].norm2 = torch_nn.Module()
                self.transformer_blocks[0].norm2.linear_1 = torch_nn.Linear(8, 8, bias=False)

        adapter_target = "llm_adapter.blocks.0.cross_attn.k_proj.weight"
        modulation_target = "blocks.0.adaln_modulation_cross_attn.1.weight"
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.transformer = Transformer()
        runtime.llm_adapter = Adapter()
        runtime.text_encoder = None
        runtime.lora_target_modules = _lora_target_modules([adapter_target, modulation_target], convert)

        resolved = runtime._lora_weight_resolver("diffusion", adapter_target)
        self.assertIsNotNone(resolved, "the adapter weight must resolve on self.llm_adapter")
        self.assertIs(resolved, runtime.llm_adapter.blocks[0].cross_attn.k_proj.weight)
        # The kind that produced the second report resolves on the transformer.
        self.assertIs(
            runtime._lora_weight_resolver("diffusion", modulation_target),
            runtime.transformer.transformer_blocks[0].norm2.linear_1.weight,
        )
        # A target the table does not know has no live parameter; guessing here is
        # what hid both faults.
        self.assertIsNone(runtime._lora_weight_resolver("diffusion", "blocks.0.nothing.weight"))

        # The adapter half must not ride with the transformer pass: patched there
        # it would be fused after the adapter had already produced this run's
        # conditioning, so the first image would miss it and every later one would not.
        specs = [
            {"family": "diffusion", "target": adapter_target},
            {"family": "diffusion", "target": modulation_target},
            {"family": "qwen", "target": "layers.0.self_attn.q_proj.weight"},
        ]
        runtime.lora_plans = [("a.safetensors", 1.0, specs)]
        by_pass = {name: self.__class__._targets(runtime, name) for name in ("diffusion", "adapter", "qwen")}
        self.assertEqual(by_pass["diffusion"], [modulation_target])
        self.assertEqual(by_pass["adapter"], [adapter_target])
        self.assertEqual(by_pass["qwen"], ["layers.0.self_attn.q_proj.weight"])
        # Every spec is claimed by exactly one pass, or a weight is fused twice.
        self.assertEqual(sum(len(targets) for targets in by_pass.values()), len(specs))

    @staticmethod
    def _targets(runtime, pass_name):
        return [spec["target"] for _path, _multiplier, specs in runtime._lora_family_plans(pass_name) for spec in specs]

    def test_unusable_lora_is_rejected_at_load_naming_the_file_and_the_key(self):
        from backend.anima_pipeline import _require_live_lora_targets

        table = {"blocks.0.cross_attn.k_proj.weight": ("transformer", "transformer_blocks.0.attn2.to_k.weight")}
        shapes = {"blocks.0.cross_attn.k_proj.weight": (8, 8)}
        good = [{"family": "diffusion", "target": "blocks.0.cross_attn.k_proj.weight", "group_name": "g", "expected": (8, 8)}]
        _require_live_lora_targets(good, table, shapes, {}, "Anima LoRA a.safetensors")

        # A target nothing in this model answers to.
        unknown = [{"family": "diffusion", "target": "blocks.0.gone.weight", "group_name": "g", "expected": (8, 8)}]
        with self.assertRaisesRegex(ValueError, r"a\.safetensors.*is not a weight of this model"):
            _require_live_lora_targets(unknown, table, shapes, {}, "Anima LoRA a.safetensors")

        # A real target the adapter reconstructs at another size. Caught before
        # the model reaches the GPU rather than from inside the first step.
        wrong = [{"family": "diffusion", "target": "blocks.0.cross_attn.k_proj.weight", "group_name": "g", "expected": (16, 8)}]
        with self.assertRaisesRegex(ValueError, r"is \(8, 8\), but the adapter reconstructs \(16, 8\)"):
            _require_live_lora_targets(wrong, table, shapes, {}, "Anima LoRA a.safetensors")

        # Text-encoder targets are checked against their own state dict.
        qwen = [{"family": "qwen", "target": "layers.0.self_attn.q_proj.weight", "group_name": "g", "expected": (4, 4)}]
        _require_live_lora_targets(qwen, table, shapes, {"layers.0.self_attn.q_proj.weight": (4, 4)}, "label")
        with self.assertRaisesRegex(ValueError, "is not a weight of this model"):
            _require_live_lora_targets(qwen, table, shapes, {}, "label")

    def test_unresolved_target_is_not_reported_as_a_shape_mismatch(self):
        from backend.anima_pipeline import _analyze_anima_lora_patch, _apply_anima_lora_groups_on_gpu

        lora = {
            "lora_unet_blocks_0_attn_to_q.lora_down.weight": self.down,
            "lora_unet_blocks_0_attn_to_q.lora_up.weight": self.up,
        }
        specs = _analyze_anima_lora_patch(lora, list(self.diffusion), list(self.qwen), 1.0)
        # "Not found" and "found but wrong" are different faults with different
        # fixes; folding them into one message sent the reader looking for a
        # shape problem that did not exist.
        with self.assertRaisesRegex(ValueError, "is not a weight of this model"):
            _apply_anima_lora_groups_on_gpu(
                specs, lambda _family, _target: None, 1.0, torch.device("cpu"), torch.float16, label="test"
            )
        wrong_shape = torch.nn.Parameter(torch.zeros((4, 5), dtype=torch.float16), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            _apply_anima_lora_groups_on_gpu(
                specs, lambda _family, _target: wrong_shape, 1.0, torch.device("cpu"), torch.float16, label="test"
            )

    def test_analyze_rejects_same_invalid_inputs_as_fuse(self):
        from backend.anima_pipeline import _analyze_anima_lora_patch

        tucker = {
            "lora_unet_blocks_0_attn_to_q.hada_w1_a": self.down,
            "lora_unet_blocks_0_attn_to_q.hada_t1": torch.ones((1, 1)),
        }
        with self.assertRaisesRegex(ValueError, "unsupported"):
            _analyze_anima_lora_patch(tucker, list(self.diffusion), list(self.qwen), 1.0)

        magnitude = {
            "lora_te_layers_0_self_attn_q_proj.magnitude_vector": torch.ones(2),
        }
        with self.assertRaisesRegex(ValueError, "magnitude"):
            _analyze_anima_lora_patch(magnitude, list(self.diffusion), list(self.qwen), 1.0)

        with self.assertRaisesRegex(ValueError, "multiplier"):
            _analyze_anima_lora_patch({}, list(self.diffusion), list(self.qwen), float("nan"))

    def test_scaling_missing_alpha_multiple_ordered_and_negative_multiplier(self):
        name = "text_encoders.qwen3_06b.transformer.model.layers.0.self_attn.q_proj"
        rank_two_down = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        rank_two_up = torch.eye(2)
        explicit = {
            f"{name}.lora_A.weight": rank_two_down,
            f"{name}.lora_B.weight": rank_two_up,
            f"{name}.alpha": torch.tensor(1.0),
        }
        diffusion, qwen = _fuse_anima_lora_state_dict(self.diffusion, self.qwen, explicit, 2.0)
        expected = rank_two_up @ rank_two_down
        self.assertTrue(torch.equal(qwen["layers.0.self_attn.q_proj.weight"], expected.to(torch.float16)))

        missing_alpha = {
            "lora_te_layers_0_self_attn_q_proj.lora_down.weight": self.down,
            "lora_te_layers_0_self_attn_q_proj.lora_up.weight": self.up,
        }
        diffusion, qwen = _fuse_anima_lora_state_dict(diffusion, qwen, missing_alpha, 0.5)
        diffusion, qwen = _fuse_anima_lora_state_dict(diffusion, qwen, missing_alpha, -0.25)
        self.assertTrue(
            torch.equal(qwen["layers.0.self_attn.q_proj.weight"], (expected + self.delta * 0.25).to(torch.float16))
        )

        unchanged_diffusion, unchanged_qwen = _fuse_anima_lora_state_dict(diffusion, qwen, missing_alpha, 0.0)
        self.assertTrue(torch.equal(unchanged_diffusion["blocks.0.attn.to_q.weight"], diffusion["blocks.0.attn.to_q.weight"]))
        self.assertTrue(torch.equal(unchanged_qwen["layers.0.self_attn.q_proj.weight"], qwen["layers.0.self_attn.q_proj.weight"]))

    def test_rejects_incomplete_shapes_unknown_advanced_aliases_and_nonfinite_values(self):
        prefix = "lora_unet_blocks_0_attn_to_q"
        valid_up = {f"{prefix}.lora_up.weight": self.up}
        invalid_cases = (
            ({f"{prefix}.lora_down.weight": self.down}, "incomplete"),
            ({f"{prefix}.lora_down.weight": torch.ones(3), **valid_up}, "2D"),
            ({f"{prefix}.lora_down.weight": torch.ones((1, 4)), **valid_up}, "product shape"),
            ({"not_an_anima_lora.weight": torch.ones(1)}, "unknown"),
            ({f"{prefix}.lora_down.weight": self.down, **valid_up, f"{prefix}.alpha": torch.tensor(float("inf"))}, "finite"),
        )
        for state, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _fuse_anima_lora_state_dict(self.diffusion, self.qwen, state, 1.0)
        for advanced_key in ("hada_w1_a", "lokr_w1", "lora_mid.weight", "dora_scale"):
            with self.subTest(advanced_key=advanced_key), self.assertRaises(ValueError):
                _fuse_anima_lora_state_dict(
                    self.diffusion, self.qwen, {f"{prefix}.{advanced_key}": torch.ones((1, 1))}, 1.0
                )
        with self.assertRaisesRegex(ValueError, "does not contain"):
            _fuse_anima_lora_state_dict(self.diffusion, self.qwen, {}, 1.0)
        with self.assertRaisesRegex(ValueError, "multiplier"):
            _fuse_anima_lora_state_dict(self.diffusion, self.qwen, {f"{prefix}.lora_down.weight": self.down, **valid_up}, float("nan"))

        aliases = {
            f"{prefix}.lora_down.weight": self.down,
            f"{prefix}.lora_up.weight": self.up,
            "diffusion_model.blocks.0.attn.to_q.lora_A.weight": self.down,
            "diffusion_model.blocks.0.attn.to_q.lora_B.weight": self.up,
        }
        with self.assertRaisesRegex(ValueError, "duplicate target"):
            _fuse_anima_lora_state_dict(self.diffusion, self.qwen, aliases, 1.0)

        ambiguous = {"a_b.weight": torch.zeros((1, 1)), "a.b.weight": torch.zeros((1, 1))}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _fuse_anima_lora_state_dict(
                ambiguous,
                {},
                {
                    "lora_unet_a_b.lora_down.weight": torch.ones((1, 1)),
                    "lora_unet_a_b.lora_up.weight": torch.ones((1, 1)),
                },
                1.0,
            )

    def test_rejects_non_linear_base_weight_and_validates_ordered_descriptors(self):
        embedding = {"embed_tokens.weight": torch.zeros((2, 3))}
        with self.assertRaisesRegex(ValueError, "not a linear"):
            _fuse_anima_lora_state_dict(
                embedding,
                {},
                {
                    "lora_unet_embed_tokens.lora_down.weight": self.down,
                    "lora_unet_embed_tokens.lora_up.weight": self.up,
                },
                1.0,
            )
        with patch("backend.anima_pipeline._require_safetensors", side_effect=lambda path, _label: path):
            self.assertEqual(
                _lora_descriptors([{"path": "one", "multiplier": 0.5}, ("two", -1)]),
                [("one", 0.5), ("two", -1.0)],
            )
            self.assertEqual(_lora_descriptors([("disabled", 0.0)]), [])
            with self.assertRaisesRegex(ValueError, "finite"):
                _lora_descriptors([("bad", float("nan"))])

    def test_loha_reconstructs_linear_hadamard_factors_and_rejects_unsafe_forms(self):
        prefix = "lora_unet_blocks_0_attn_to_q"
        w1a = torch.tensor([[1.0], [2.0]])
        w1b = torch.tensor([[1.0, 2.0, 3.0]])
        w2a = torch.tensor([[2.0], [1.0]])
        w2b = torch.tensor([[3.0, 2.0, 1.0]])
        state = {
            f"{prefix}.hada_w1_a": w1a,
            f"{prefix}.hada_w1_b": w1b,
            f"{prefix}.hada_w2_a": w2a,
            f"{prefix}.hada_w2_b": w2b,
            f"{prefix}.alpha": torch.tensor(0.5),
        }
        diffusion, _qwen = _fuse_anima_lora_state_dict(self.diffusion, self.qwen, state, 2.0)
        expected = (w1a @ w1b) * (w2a @ w2b)
        self.assertTrue(torch.equal(diffusion["blocks.0.attn.to_q.weight"], expected.to(torch.float16)))

        invalid = (
            ({key: value for key, value in state.items() if "hada_w2_b" not in key}, "incomplete"),
            ({**state, f"{prefix}.hada_t1": torch.ones((1, 1, 1, 1))}, "Tucker"),
            ({**state, f"{prefix}.hada_w1_a": torch.tensor([[float("nan")], [2.0]])}, "finite"),
            ({**state, f"{prefix}.alpha": torch.tensor(0.0)}, "zero alpha"),
        )
        for candidate, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _fuse_anima_lora_state_dict(self.diffusion, self.qwen, candidate, 1.0)

    def test_lokr_reconstructs_low_rank_and_full_kronecker_forms(self):
        prefix = "lora_unet_blocks_0_attn_to_q"
        w1 = torch.tensor([[1.0]])
        w2a = torch.tensor([[1.0], [2.0]])
        w2b = torch.tensor([[1.0, 2.0, 3.0]])
        low_rank = {
            f"{prefix}.lokr_w1": w1,
            f"{prefix}.lokr_w2_a": w2a,
            f"{prefix}.lokr_w2_b": w2b,
            f"{prefix}.alpha": torch.tensor(0.5),
        }
        diffusion, _qwen = _fuse_anima_lora_state_dict(self.diffusion, self.qwen, low_rank, 2.0)
        self.assertTrue(
            torch.equal(
                diffusion["blocks.0.attn.to_q.weight"],
                torch.kron(w1, w2a @ w2b).to(torch.float16),
            )
        )

        w2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        full = {
            f"{prefix}.lokr_w1": w1,
            f"{prefix}.lokr_w2": w2,
            f"{prefix}.alpha": torch.tensor(999.0),
        }
        diffusion, _qwen = _fuse_anima_lora_state_dict(self.diffusion, self.qwen, full, -0.5)
        self.assertTrue(
            torch.equal(
                diffusion["blocks.0.attn.to_q.weight"],
                (-0.5 * torch.kron(w1, w2)).to(torch.float16),
            )
        )

        invalid = (
            ({**low_rank, f"{prefix}.lokr_w2": w2}, "exactly one"),
            ({**low_rank, f"{prefix}.lokr_t2": torch.ones((1, 1, 1, 1))}, "unsupported"),
            ({**low_rank, f"{prefix}.lokr_w1_a": torch.ones((1, 1))}, "unsupported"),
        )
        for candidate, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _fuse_anima_lora_state_dict(self.diffusion, self.qwen, candidate, 1.0)

    def test_static_tlora_matches_ordinary_lora_and_middle_tensor_stays_rejected(self):
        prefix = "lora_unet_blocks_0_attn_to_q"
        state = {
            f"{prefix}.lora_down.weight": self.down,
            f"{prefix}.lora_up.weight": self.up,
            f"{prefix}.alpha": torch.tensor(0.5),
        }
        diffusion, _qwen = _fuse_anima_lora_state_dict(self.diffusion, self.qwen, state, 2.0)
        self.assertTrue(torch.equal(diffusion["blocks.0.attn.to_q.weight"], self.delta.to(torch.float16)))
        with self.assertRaisesRegex(ValueError, "middle tensor"):
            _fuse_anima_lora_state_dict(
                self.diffusion,
                self.qwen,
                {**state, f"{prefix}.lora_mid.weight": torch.ones((1, 1))},
                1.0,
            )

    def test_dora_uses_baseline_preserving_dense_interpolation_and_strict_magnitude(self):
        diffusion = {"blocks.0.attn.to_q.weight": torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float32)}
        prefix = "lora_unet_blocks_0_attn_to_q"
        down = torch.tensor([[1.0, -1.0]])
        up = torch.tensor([[2.0], [1.0]])
        magnitude = torch.tensor([[10.0], [4.0]])
        state = {
            f"{prefix}.lora_down.weight": down,
            f"{prefix}.lora_up.weight": up,
            f"{prefix}.dora_scale": magnitude,
        }
        baseline = diffusion["blocks.0.attn.to_q.weight"]
        unchanged, _qwen = _fuse_anima_lora_state_dict(diffusion, {}, state, 0.0)
        self.assertTrue(torch.equal(unchanged["blocks.0.attn.to_q.weight"], baseline))

        direction = baseline + up @ down
        target = direction * (
            magnitude[:, 0] / (torch.linalg.vector_norm(direction, dim=1) + torch.finfo(torch.float32).eps)
        ).unsqueeze(1)
        full, _qwen = _fuse_anima_lora_state_dict(diffusion, {}, state, 1.0)
        half, _qwen = _fuse_anima_lora_state_dict(diffusion, {}, state, 0.5)
        self.assertTrue(torch.allclose(full["blocks.0.attn.to_q.weight"], target))
        self.assertTrue(torch.allclose(half["blocks.0.attn.to_q.weight"], baseline + 0.5 * (target - baseline)))

        invalid = (
            ({**state, f"{prefix}.dora_scale": torch.ones(3)}, "shape"),
            ({**state, f"{prefix}.dora_scale": torch.tensor([float("inf"), 1.0])}, "finite"),
            ({**state, f"{prefix}.lora_magnitude_vector": torch.ones(2)}, "magnitude alias"),
        )
        for candidate, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _fuse_anima_lora_state_dict(diffusion, {}, candidate, 1.0)

        zero_direction = {
            f"{prefix}.lora_down.weight": down,
            f"{prefix}.lora_up.weight": up,
            f"{prefix}.dora_scale": magnitude,
        }
        with self.assertRaisesRegex(ValueError, "nonzero"):
            _fuse_anima_lora_state_dict({"blocks.0.attn.to_q.weight": -(up @ down)}, {}, zero_direction, 1.0)

    def test_lokr_dora_supports_low_rank_and_full_direction_with_strict_validation(self):
        diffusion = {"blocks.0.attn.to_q.weight": torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float32)}
        prefix = "lora_unet_blocks_0_attn_to_q"
        w1 = torch.tensor([[1.0]])
        w2a = torch.tensor([[2.0], [1.0]])
        w2b = torch.tensor([[1.0, -1.0]])
        magnitude = torch.tensor([[10.0], [4.0]])
        low_rank = {
            f"{prefix}.lokr_w1": w1,
            f"{prefix}.lokr_w2_a": w2a,
            f"{prefix}.lokr_w2_b": w2b,
            f"{prefix}.alpha": torch.tensor(0.5),
            f"{prefix}.dora_scale": magnitude,
        }
        baseline = diffusion["blocks.0.attn.to_q.weight"]
        delta = torch.kron(w1, w2a @ w2b) * 0.5
        direction = baseline + delta
        target = direction * (
            magnitude[:, 0] / (torch.linalg.vector_norm(direction, dim=1) + torch.finfo(torch.float32).eps)
        ).unsqueeze(1)

        full, _qwen = _fuse_anima_lora_state_dict(diffusion, {}, low_rank, 1.0)
        half, _qwen = _fuse_anima_lora_state_dict(diffusion, {}, low_rank, 0.5)
        self.assertTrue(torch.allclose(full["blocks.0.attn.to_q.weight"], target))
        self.assertTrue(torch.allclose(half["blocks.0.attn.to_q.weight"], baseline + 0.5 * (target - baseline)))

        full_matrix = {
            f"{prefix}.lokr_w1": w1,
            f"{prefix}.lokr_w2": w2a @ w2b,
            f"{prefix}.dora_scale": magnitude[:, 0],
        }
        reconstructed, _qwen = _fuse_anima_lora_state_dict(diffusion, {}, full_matrix, 1.0)
        full_direction = baseline + torch.kron(w1, w2a @ w2b)
        full_target = full_direction * (
            magnitude[:, 0] / (torch.linalg.vector_norm(full_direction, dim=1) + torch.finfo(torch.float32).eps)
        ).unsqueeze(1)
        self.assertTrue(torch.allclose(reconstructed["blocks.0.attn.to_q.weight"], full_target))

        invalid = (
            ({**low_rank, f"{prefix}.dora_scale": torch.ones(3)}, "shape"),
            ({**low_rank, f"{prefix}.dora_scale": torch.tensor([float("nan"), 1.0])}, "finite"),
            ({**low_rank, f"{prefix}.lokr_t2": torch.ones((1, 1, 1, 1))}, "unsupported"),
        )
        for candidate, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _fuse_anima_lora_state_dict(diffusion, {}, candidate, 1.0)

        zero_direction = {
            f"{prefix}.lokr_w1": w1,
            f"{prefix}.lokr_w2": w2a @ w2b,
            f"{prefix}.dora_scale": magnitude,
        }
        with self.assertRaisesRegex(ValueError, "nonzero"):
            _fuse_anima_lora_state_dict(
                {"blocks.0.attn.to_q.weight": -torch.kron(w1, w2a @ w2b)},
                {},
                zero_direction,
                1.0,
            )

    def test_invalid_lokr_dora_group_keeps_all_input_state_unchanged(self):
        original_diffusion = {key: value.clone() for key, value in self.diffusion.items()}
        state = {
            "lora_unet_blocks_0_attn_to_q.lokr_w1": torch.tensor([[1.0]]),
            "lora_unet_blocks_0_attn_to_q.lokr_w2": torch.ones((2, 3)),
            "lora_unet_blocks_0_attn_to_q.dora_scale": torch.ones(2),
            "lora_unet_llm_adapter_blocks_0_cross_attn_q_proj.lokr_w1": torch.tensor([[1.0]]),
            "lora_unet_llm_adapter_blocks_0_cross_attn_q_proj.lokr_w2": torch.ones((2, 3)),
            "lora_unet_llm_adapter_blocks_0_cross_attn_q_proj.dora_scale": torch.ones(3),
        }
        with self.assertRaisesRegex(ValueError, "shape"):
            _fuse_anima_lora_state_dict(self.diffusion, self.qwen, state, 1.0)
        for key, value in original_diffusion.items():
            self.assertTrue(torch.equal(self.diffusion[key], value))

    def test_invalid_group_does_not_mutate_any_input_state(self):
        original_diffusion = {key: value.clone() for key, value in self.diffusion.items()}
        state = {
            "lora_unet_blocks_0_attn_to_q.lora_down.weight": self.down,
            "lora_unet_blocks_0_attn_to_q.lora_up.weight": self.up,
            "lora_te_layers_0_self_attn_q_proj.lora_down.weight": torch.ones((1, 99)),
            "lora_te_layers_0_self_attn_q_proj.lora_up.weight": self.up,
        }
        with self.assertRaisesRegex(ValueError, "product shape"):
            _fuse_anima_lora_state_dict(self.diffusion, self.qwen, state, 1.0)
        for key, value in original_diffusion.items():
            self.assertTrue(torch.equal(self.diffusion[key], value))


class AnimaSamplerTests(unittest.TestCase):
    def test_cosmos_pag_identity_processor_uses_only_value_and_output_projection(self):
        processor = CosmosPAGIdentitySelfAttnProcessor()
        attention = SimpleNamespace(
            to_q=Mock(side_effect=AssertionError("Q projection used")),
            to_k=Mock(side_effect=AssertionError("K projection used")),
            to_v=Mock(side_effect=lambda value: value + 1),
            to_out=[Mock(side_effect=lambda value: value * 2), Mock(side_effect=lambda value: value - 3)],
        )
        result = processor(attention, torch.tensor([2.0]))
        self.assertTrue(torch.equal(result, torch.tensor([3.0])))
        attention.to_v.assert_called_once()
        attention.to_q.assert_not_called()
        attention.to_k.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "non-self-attention"):
            processor(attention, torch.tensor([2.0]), encoder_hidden_states=torch.tensor([1.0]))

    def test_pag_targets_and_processor_restoration_are_exact(self):
        class Attention:
            def __init__(self):
                self.processor = object()

            def set_processor(self, processor):
                self.processor = processor

        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._poisoned = False
        attentions = [Attention() for _ in range(28)]
        runtime.transformer = SimpleNamespace(
            transformer_blocks=[SimpleNamespace(attn1=attention) for attention in attentions]
        )
        mid = runtime._pag_targets("mid")
        all_targets = runtime._pag_targets("all")
        self.assertEqual([name for name, _attention in mid], ["transformer_blocks.14.attn1"])
        self.assertEqual(len(all_targets), 28)
        original = attentions[14].processor
        self.assertEqual(runtime._pag_prediction(mid, lambda: isinstance(attentions[14].processor, CosmosPAGIdentitySelfAttnProcessor)), True)
        self.assertIs(attentions[14].processor, original)

        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            runtime._pag_prediction(mid, lambda: (_ for _ in ()).throw(RuntimeError("forward failed")))
        self.assertIs(attentions[14].processor, original)
        self.assertFalse(runtime._poisoned)

    def test_pag_partial_install_or_restore_failure_poisons_runtime(self):
        class Attention:
            def __init__(self, fail_install=False, fail_restore=False):
                self.original = object()
                self.processor = self.original
                self.fail_install = fail_install
                self.fail_restore = fail_restore

            def set_processor(self, processor):
                if self.fail_install and processor is not self.original:
                    raise RuntimeError("install failed")
                if self.fail_restore and processor is self.original and self.processor is not self.original:
                    raise RuntimeError("restore failed")
                self.processor = processor

        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._poisoned = False
        first = Attention()
        second = Attention(fail_install=True)
        with self.assertRaisesRegex(RuntimeError, "install failed"):
            runtime._pag_prediction([("first", first), ("second", second)], lambda: None)
        self.assertTrue(runtime._poisoned)
        self.assertIs(first.processor, first.original)

        runtime._poisoned = False
        broken = Attention(fail_restore=True)
        with self.assertRaisesRegex(RuntimeError, "failed to restore"):
            runtime._pag_prediction([("broken", broken)], lambda: None)
        self.assertTrue(runtime._poisoned)

    def test_pag_sampler_runs_sequential_conditioned_perturbed_and_optional_cfg_branches(self):
        class Attention:
            def __init__(self):
                self.processor = object()

            def set_processor(self, processor):
                self.processor = processor

        class Transformer:
            def __init__(self):
                self.transformer_blocks = [SimpleNamespace(attn1=Attention()) for _ in range(28)]
                self.calls = []

            def to(self, *_args, **_kwargs):
                return self

            def __call__(self, hidden_states, encoder_hidden_states, **_kwargs):
                perturbed = isinstance(
                    self.transformer_blocks[14].attn1.processor,
                    CosmosPAGIdentitySelfAttnProcessor,
                )
                branch = "perturbed" if perturbed else "negative" if encoder_hidden_states[0, 0, 0].item() < 0 else "conditioned"
                self.calls.append(branch)
                value = {"conditioned": 2.0, "perturbed": 0.5, "negative": 1.0}[branch]
                return (torch.full_like(hidden_states, value),)

        def run(cfg, embeddings):
            runtime = AnimaRuntime.__new__(AnimaRuntime)
            runtime.dtype = torch.float32
            runtime._poisoned = False
            runtime.transformer = Transformer()
            originals = [block.attn1.processor for block in runtime.transformer.transformer_blocks]
            real_torch_device = torch.device
            with (
                patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)),
                patch("backend.anima_pipeline._empty_cuda_cache"),
            ):
                runtime._sample(
                    embeddings,
                    torch.ones(embeddings.shape[:2]),
                    32,
                    32,
                    1,
                    cfg,
                    "euler",
                    [torch.Generator(device="cpu").manual_seed(4)],
                    "pag",
                    None,
                    pag_scale=0.4,
                    pag_applied_layers="mid",
                )
            self.assertTrue(
                all(block.attn1.processor is original for block, original in zip(runtime.transformer.transformer_blocks, originals))
            )
            return runtime.transformer.calls

        self.assertEqual(run(1.0, torch.ones((1, 2, 3))), ["conditioned", "perturbed"])
        self.assertEqual(
            run(3.0, torch.tensor([[[1.0], [1.0]], [[-1.0], [-1.0]]])),
            ["conditioned", "perturbed", "negative"],
        )

    def test_zero_scale_pag_skips_processor_resolution_and_perturbed_forward(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.float32
        runtime.transformer = Mock()
        runtime.transformer.to = Mock(return_value=runtime.transformer)
        runtime.transformer.return_value = (torch.zeros((1, 16, 1, 4, 4)),)
        runtime._pag_targets = Mock(side_effect=AssertionError("PAG targets resolved at scale zero"))
        real_torch_device = torch.device
        with (
            patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)),
            patch("backend.anima_pipeline._empty_cuda_cache"),
        ):
            runtime._sample(
                torch.ones((1, 2, 3)),
                torch.ones((1, 2)),
                32,
                32,
                1,
                1.0,
                "euler",
                [torch.Generator(device="cpu")],
                "pag",
                None,
                pag_scale=0.0,
            )
        runtime._pag_targets.assert_not_called()
        self.assertEqual(runtime.transformer.call_count, 1)

    def test_cfg_batch_keeps_cosmos_padding_mask_singleton(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.float32
        runtime.batch_cfg = True
        runtime._transformer_group_offload = False
        runtime._transformer_resident = False
        runtime.keep_transformer_resident = False
        runtime.transformer = Mock()
        runtime.transformer.to = Mock(return_value=runtime.transformer)
        calls = []

        def predict(**kwargs):
            calls.append({key: tuple(value.shape) for key, value in kwargs.items()})
            return torch.zeros_like(kwargs["hidden_states"])

        runtime._transformer_prediction = predict
        real_torch_device = torch.device
        with (
            patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)),
            patch("backend.anima_pipeline._empty_cuda_cache"),
        ):
            runtime._sample(
                torch.tensor([[[1.0]], [[-1.0]]]),
                torch.ones((2, 1)),
                32,
                32,
                1,
                5.0,
                "euler",
                [torch.Generator(device="cpu")],
                "none",
                None,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["hidden_states"][0], 2)
        self.assertEqual(calls[0]["encoder_hidden_states"][0], 2)
        self.assertEqual(calls[0]["padding_mask"][0], 1)

    def test_flow_shift_three_values(self):
        expected = torch.tensor([1.0, 0.9, 0.75, 0.5, 0.0])
        self.assertTrue(torch.allclose(shifted_sigmas(4), expected))

    def test_euler_rf_update_is_deterministic_float32_before_runtime_dtype_cast(self):
        sample = torch.tensor([2.0], dtype=torch.bfloat16)
        velocity = torch.tensor([0.5], dtype=torch.bfloat16)
        first = euler_rf_step(sample, velocity, 1.0, 0.75)
        second = euler_rf_step(sample, velocity, 1.0, 0.75)
        self.assertEqual(first.dtype, torch.float32)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.allclose(first, torch.tensor([1.875])))

    def test_runtime_sampler_keeps_default_fp32_state_and_casts_only_transformer_input(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.bfloat16
        runtime.latent_state_mode = "fp32"
        runtime.transformer = Mock()
        runtime.transformer.to = Mock(return_value=runtime.transformer)
        runtime.transformer.return_value = (torch.full((1, 16, 1, 8, 8), 0.25, dtype=torch.bfloat16),)
        generator = torch.Generator(device="cpu").manual_seed(7)
        callback_dtypes = []
        real_torch_device = torch.device

        with (
            patch.object(
                torch,
                "device",
                side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value),
            ),
            patch("backend.anima_pipeline._empty_cuda_cache"),
        ):
            latents = runtime._sample(
                torch.zeros((1, 2, 4), dtype=torch.bfloat16),
                torch.ones((1, 2), dtype=torch.long),
                64,
                64,
                2,
                1.0,
                "euler",
                [generator],
                "none",
                lambda _step, _total, value: callback_dtypes.append(value.dtype),
            )

        self.assertEqual(latents.dtype, torch.float32)
        self.assertEqual(callback_dtypes, [torch.float32, torch.float32])
        self.assertEqual(runtime.transformer.call_count, 2)
        self.assertEqual(runtime.transformer.call_args.kwargs["hidden_states"].dtype, torch.bfloat16)

    def test_runtime_sampler_bf16_compatibility_mode_quantizes_state(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.bfloat16
        runtime.latent_state_mode = "bf16"
        runtime.transformer = Mock()
        runtime.transformer.to = Mock(return_value=runtime.transformer)
        runtime.transformer.return_value = (torch.zeros((1, 16, 1, 8, 8), dtype=torch.bfloat16),)
        real_torch_device = torch.device
        with (
            patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)),
            patch("backend.anima_pipeline._empty_cuda_cache"),
        ):
            latents = runtime._sample(
                torch.zeros((1, 2, 4)), torch.ones((1, 2)), 64, 64, 2, 1.0, "euler",
                [torch.Generator(device="cpu").manual_seed(7)], "none", None,
            )
        self.assertEqual(latents.dtype, torch.bfloat16)

    def test_default_fp32_state_covers_lcm_midpoint_and_multistep_solvers(self):
        real_torch_device = torch.device
        for sampler in ("lcm", "dpm_2", "lms"):
            with self.subTest(sampler=sampler):
                runtime = AnimaRuntime.__new__(AnimaRuntime)
                runtime.dtype = torch.bfloat16
                runtime.latent_state_mode = "fp32"
                runtime.transformer = Mock()
                runtime.transformer.to = Mock(return_value=runtime.transformer)
                runtime.transformer.return_value = (torch.full((1, 16, 1, 8, 8), .25, dtype=torch.bfloat16),)
                callbacks = []
                with (
                    patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)),
                    patch("backend.anima_pipeline._empty_cuda_cache"),
                ):
                    result = runtime._sample(
                        torch.zeros((1, 2, 4)), torch.ones((1, 2)), 64, 64, 2, 1.0, sampler,
                        [torch.Generator(device="cpu").manual_seed(7)], "none",
                        lambda _step, _total, value: callbacks.append(value.dtype),
                    )
                self.assertEqual(result.dtype, torch.float32)
                self.assertEqual(callbacks, [torch.float32, torch.float32])

    def test_reference_png_quantization_truncates_without_rounding(self):
        pixels = torch.tensor([
            [[-1.0, -0.5]],
            [[0.0, 0.5]],
            [[1.0, 0.01]],
        ])
        image = _decoded_tensor_to_image(pixels)
        self.assertEqual(image.tobytes(), bytes((0, 127, 255, 63, 191, 128)))

    def test_ancestral_terminal_step_returns_denoised_without_noise(self):
        sample = torch.tensor([[[2.0]]])
        denoised = torch.tensor([[[0.25]]])
        generator = torch.Generator(device="cpu").manual_seed(91)
        result = euler_ancestral_rf_step(sample, denoised, 0.5, 0.0, [generator])
        expected_next = torch.randn((1,), generator=torch.Generator(device="cpu").manual_seed(91))
        actual_next = torch.randn((1,), generator=generator)
        self.assertTrue(torch.equal(result, denoised))
        self.assertTrue(torch.equal(actual_next, expected_next))

    def test_ancestral_nonterminal_step_is_seeded_per_image(self):
        sample = torch.ones((2, 1, 1))
        denoised = torch.zeros_like(sample)

        def run():
            generators = [
                torch.Generator(device="cpu").manual_seed(10),
                torch.Generator(device="cpu").manual_seed(20),
            ]
            return euler_ancestral_rf_step(sample, denoised, 1.0, 0.75, generators)

        first = run()
        second = run()
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first[0], first[1]))
        self.assertTrue(torch.isfinite(first).all())

    def test_refinement_initialization_and_terminal_mask_preserve_source(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.bfloat16
        runtime.latent_state_mode = "fp32"
        source = torch.full((1, 16, 1, 4, 4), 0.25)
        generator = torch.Generator(device="cpu").manual_seed(17)
        initial, noise = runtime._refinement_start(source, [generator], torch.tensor(0.75))
        expected_noise = torch.randn(source.shape, generator=torch.Generator(device="cpu").manual_seed(17))
        self.assertTrue(torch.equal(noise, expected_noise))
        self.assertEqual(initial.dtype, torch.float32)
        self.assertTrue(torch.equal(initial, source * 0.25 + expected_noise * 0.75))

        runtime.transformer = Mock()
        runtime.transformer.to = Mock(return_value=runtime.transformer)
        runtime.transformer.return_value = (torch.zeros_like(source, dtype=torch.bfloat16),)
        latent_mask = torch.ones((1, 1, 1, 4, 4))
        latent_mask[..., 2:] = 0.0
        callbacks = []
        real_torch_device = torch.device
        with (
            patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)),
            patch("backend.anima_pipeline._empty_cuda_cache"),
        ):
            result = runtime._sample(
                torch.zeros((1, 2, 4)), torch.ones((1, 2)), 32, 32, 4, 1.0, "euler",
                [torch.Generator(device="cpu")], "none", lambda step, total, _value: callbacks.append((step, total)),
                initial_latents=torch.zeros_like(source), sigmas=shifted_sigmas(4), start_index=3,
                source_latents=source, source_noise=torch.ones_like(source), latent_mask=latent_mask,
            )
        self.assertTrue(torch.equal(result[..., 2:], source[..., 2:]))
        self.assertTrue(torch.equal(result[..., :2], torch.zeros_like(result[..., :2])))
        self.assertEqual(callbacks, [(1, 1)])

    def test_default_fp32_euler_trajectory_matches_independent_scalar_oracle(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.bfloat16
        runtime.latent_state_mode = "fp32"
        runtime.transformer = Mock()
        runtime.transformer.to = Mock(return_value=runtime.transformer)
        predictions = iter((0.12345, -0.23456, 0.34567))
        runtime.transformer.side_effect = lambda **kwargs: (
            torch.full_like(kwargs["hidden_states"], next(predictions)),
        )
        sigmas = anima_sigma_schedule(3, "normal")
        initial = torch.full((1, 16, 1, 4, 4), 0.1234567)
        expected = 0.1234567
        for sigma, next_sigma, prediction in zip(sigmas[:-1], sigmas[1:], (.12345, -.23456, .34567)):
            # Oracle intentionally uses scalar FP32 integration, with only
            # the mocked Transformer's BF16 output quantized at its boundary.
            boundary_prediction = float(torch.tensor(prediction, dtype=torch.bfloat16).float())
            expected += (float(next_sigma) - float(sigma)) * boundary_prediction
        real_torch_device = torch.device
        with (
            patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)),
            patch("backend.anima_pipeline._empty_cuda_cache"),
        ):
            result = runtime._sample(
                torch.zeros((1, 2, 4)), torch.ones((1, 2)), 32, 32, 3, 1.0, "euler",
                [torch.Generator(device="cpu")], "none", None, initial_latents=initial, sigmas=sigmas,
            )
        self.assertEqual(result.dtype, torch.float32)
        self.assertAlmostEqual(float(result[0, 0, 0, 0, 0]), expected, places=6)

    def test_default_fp32_euler_ancestral_trajectory_matches_independent_scalar_oracle(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.bfloat16
        runtime.latent_state_mode = "fp32"
        runtime.transformer = Mock()
        runtime.transformer.to = Mock(return_value=runtime.transformer)
        predictions = iter((0.125, -0.375))
        runtime.transformer.side_effect = lambda **kwargs: (
            torch.full_like(kwargs["hidden_states"], next(predictions)),
        )
        initial = torch.full((1, 16, 1, 4, 4), 0.2)
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        # Independent FP32 tensor oracle for the two ancestral RF updates;
        # it intentionally does not invoke any production solver helper.
        first_prediction = torch.full_like(initial, .125, dtype=torch.bfloat16).float()
        denoised = initial - first_prediction
        sigma_down = .25
        first = (.25 * initial + .75 * denoised) * (.5 / .75)
        renoise = (0.5 ** 2 - sigma_down ** 2 * .5 ** 2 / (.75 ** 2)) ** .5
        noise = torch.randn(initial.shape, generator=torch.Generator(device="cpu").manual_seed(31))
        first = first + noise * renoise
        expected = first - .5 * torch.full_like(initial, -.375, dtype=torch.bfloat16).float()
        real_torch_device = torch.device
        with (
            patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)),
            patch("backend.anima_pipeline._empty_cuda_cache"),
        ):
            result = runtime._sample(
                torch.zeros((1, 2, 4)), torch.ones((1, 2)), 32, 32, 2, 1.0, "euler_ancestral",
                [torch.Generator(device="cpu").manual_seed(31)], "none", None, initial_latents=initial, sigmas=sigmas,
            )
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6, rtol=0))

    def test_initial_latents_default_fp32_are_seed_deterministic(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.bfloat16
        first = runtime._initial_latents([torch.Generator(device="cpu").manual_seed(44)], 32, 32)
        second = runtime._initial_latents([torch.Generator(device="cpu").manual_seed(44)], 32, 32)
        self.assertEqual(first.dtype, torch.float32)
        self.assertTrue(torch.equal(first, second))


class AnimaRuntimeContractTests(unittest.TestCase):
    def _real_refine_runtime(self, batch_cfg=False):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False
        runtime._poisoned = False
        runtime.dtype = torch.float32
        runtime.latent_state_mode = "fp32"
        runtime.noise_device = "cpu"
        runtime.batch_cfg = batch_cfg
        runtime._transformer_group_offload = True
        runtime._transformer_blocks_per_group = 1
        runtime.last_generation_metrics = {}
        runtime._encode_prompts = Mock(return_value=(torch.tensor([[[1.0]], [[-1.0]]]), torch.ones((2, 1))) )
        runtime._encode_images = Mock(return_value=torch.zeros((1, 16, 1, 4, 4)))
        runtime._decode = Mock(side_effect=lambda latents, **_kwargs: [latents.detach().clone()])
        return runtime

    @contextmanager
    def _cpu_cuda_path(self):
        real_device = torch.device
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch, "device", side_effect=lambda value: real_device("cpu") if value == "cuda" else real_device(value)),
            patch("backend.anima_pipeline._empty_cuda_cache"),
        ):
            yield
    def test_optional_attention_failure_retries_inside_native_dispatch_scope(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._transformer_group_offload = False
        runtime.attention_backend = "_native_efficient"
        runtime.last_generation_metrics = {}
        runtime.transformer = Mock(side_effect=[RuntimeError("kernel rejected shape"), (torch.ones(1),)])
        entered = []

        @contextmanager
        def attention_scope():
            entered.append(runtime.attention_backend)
            yield

        runtime._attention_scope = attention_scope
        runtime.configure_attention_backend = Mock(
            side_effect=lambda backend: setattr(runtime, "attention_backend", backend) or backend
        )

        prediction = runtime._transformer_prediction(hidden_states=torch.zeros(1))

        self.assertTrue(torch.equal(prediction, torch.ones(1)))
        self.assertEqual(entered, ["_native_efficient"])
        self.assertEqual(
            runtime.last_generation_metrics["attention_fallback"],
            {"from": "_native_efficient", "to": "native", "reason": "RuntimeError"},
        )

    def test_high_vram_decode_oom_disables_co_residency_and_retries_staged(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.float32
        runtime._transformer_resident = True
        runtime._transformer_group_offload = False
        runtime.keep_transformer_resident = True
        runtime._co_residency_failed = False
        runtime.transformer = SimpleNamespace(to=Mock())
        pixels = torch.zeros((1, 3, 1, 1, 1))
        runtime.vae = SimpleNamespace(
            config=SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16),
            to=Mock(),
            decode=Mock(side_effect=[torch.cuda.OutOfMemoryError("CUDA out of memory"), (pixels,)]),
        )
        real_torch_device = torch.device
        with (
            patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)),
            patch("backend.anima_pipeline._empty_cuda_cache"),
        ):
            images = runtime._decode(torch.zeros((1, 16, 1, 1, 1)))

        self.assertEqual(len(images), 1)
        self.assertEqual(runtime.vae.decode.call_count, 2)
        runtime.transformer.to.assert_called_once_with("cpu")
        self.assertFalse(runtime.transformer_resident)
        self.assertFalse(runtime.keep_transformer_resident)
        self.assertTrue(runtime._co_residency_failed)

    def test_decode_evicts_resident_transformer_when_co_residency_does_not_fit(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.float32
        runtime._transformer_resident = True
        runtime._transformer_group_offload = False
        runtime.keep_transformer_resident = True
        runtime._co_residency_failed = False
        runtime.weight_sizes = {"transformer": 4 * 1024**3, "vae": 1024**3}
        runtime.transformer = SimpleNamespace(to=Mock())
        pixels = torch.zeros((1, 3, 1, 1, 1))
        runtime.vae = SimpleNamespace(
            config=SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16),
            to=Mock(),
            decode=Mock(return_value=(pixels,)),
        )
        real_torch_device = torch.device
        with (
            patch.object(
                torch,
                "device",
                side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value),
            ),
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(total_memory=8 * 1024**3),
            ),
            patch("backend.anima_pipeline._empty_cuda_cache"),
        ):
            images = runtime._decode(torch.zeros((1, 16, 1, 8, 8)))

        self.assertEqual(len(images), 1)
        runtime.transformer.to.assert_called_once_with("cpu")
        self.assertFalse(runtime.transformer_resident)
        self.assertTrue(runtime.keep_transformer_resident)
        self.assertFalse(runtime._co_residency_failed)

    def test_decode_keeps_transformer_resident_when_co_residency_fits(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.float32
        runtime._transformer_resident = True
        runtime._transformer_group_offload = False
        runtime.keep_transformer_resident = True
        runtime._co_residency_failed = False
        runtime.weight_sizes = {"transformer": 2 * 1024**3, "vae": 512 * 1024**2}
        runtime.transformer = SimpleNamespace(to=Mock())
        pixels = torch.zeros((1, 3, 1, 1, 1))
        runtime.vae = SimpleNamespace(
            config=SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16),
            to=Mock(),
            decode=Mock(return_value=(pixels,)),
        )
        real_torch_device = torch.device
        with (
            patch.object(
                torch,
                "device",
                side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value),
            ),
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(total_memory=16 * 1024**3),
            ),
            patch("backend.anima_pipeline._empty_cuda_cache"),
        ):
            images = runtime._decode(torch.zeros((1, 16, 1, 8, 8)))

        self.assertEqual(len(images), 1)
        runtime.transformer.to.assert_not_called()
        self.assertTrue(runtime.transformer_resident)
        self.assertTrue(runtime.keep_transformer_resident)
        self.assertFalse(runtime._co_residency_failed)

    def test_full_vae_oom_retries_tiled_and_remembers_runtime_requirement(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.float32
        runtime._transformer_resident = False
        runtime._transformer_group_offload = False
        runtime._vae_tiling_required = False
        runtime.last_generation_metrics = {}
        pixels = torch.zeros((1, 3, 1, 1, 1))
        runtime.vae = SimpleNamespace(
            config=SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16),
            use_tiling=False,
            to=Mock(),
            enable_tiling=Mock(),
            decode=Mock(side_effect=[torch.cuda.OutOfMemoryError("CUDA out of memory"), (pixels,)]),
        )
        runtime.vae.enable_tiling.side_effect = lambda: setattr(runtime.vae, "use_tiling", True)
        real_torch_device = torch.device
        with (
            patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)),
            patch("backend.anima_pipeline._empty_cuda_cache"),
        ):
            images = runtime._decode(torch.zeros((1, 16, 1, 1, 1)))

        self.assertEqual(len(images), 1)
        runtime.vae.enable_tiling.assert_called_once_with()
        self.assertTrue(runtime._vae_tiling_required)
        self.assertEqual(
            runtime.last_generation_metrics["vae_decode_fallback"],
            {"from": "full", "to": "tiled", "reason": "cuda_oom"},
        )

    def test_forced_tiled_decode_is_call_local_and_restores_full_vae_state(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.float32
        runtime._transformer_resident = False
        runtime._transformer_group_offload = False
        runtime._vae_tiling_required = False
        runtime.last_generation_metrics = {}
        pixels = torch.zeros((1, 3, 1, 1, 1))
        runtime.vae = SimpleNamespace(
            config=SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16), use_tiling=False,
            to=Mock(), decode=Mock(return_value=(pixels,)), enable_tiling=Mock(), disable_tiling=Mock(),
        )
        runtime.vae.enable_tiling.side_effect = lambda: setattr(runtime.vae, "use_tiling", True)
        runtime.vae.disable_tiling.side_effect = lambda: setattr(runtime.vae, "use_tiling", False)
        real_torch_device = torch.device
        with patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)), patch("backend.anima_pipeline._empty_cuda_cache"):
            runtime._decode(torch.zeros((1, 16, 1, 1, 1)), force_tiled_decode=True)
            runtime._decode(torch.zeros((1, 16, 1, 1, 1)))
        self.assertFalse(runtime.vae.use_tiling)
        runtime.vae.enable_tiling.assert_called_once_with()
        runtime.vae.disable_tiling.assert_called_once_with()
        self.assertFalse(runtime._vae_tiling_required)

    def test_forced_tiled_decode_keeps_existing_persistent_tiling_and_restores_on_error(self):
        def runtime_with_vae(use_tiling, required, decode):
            runtime = AnimaRuntime.__new__(AnimaRuntime)
            runtime.dtype = torch.float32
            runtime._transformer_resident = False
            runtime._transformer_group_offload = False
            runtime._vae_tiling_required = required
            runtime.last_generation_metrics = {}
            runtime.vae = SimpleNamespace(
                config=SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16), use_tiling=use_tiling,
                to=Mock(), decode=decode, enable_tiling=Mock(), disable_tiling=Mock(),
            )
            runtime.vae.enable_tiling.side_effect = lambda: setattr(runtime.vae, "use_tiling", True)
            runtime.vae.disable_tiling.side_effect = lambda: setattr(runtime.vae, "use_tiling", False)
            return runtime

        pixels = torch.zeros((1, 3, 1, 1, 1))
        real_torch_device = torch.device
        with patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)), patch("backend.anima_pipeline._empty_cuda_cache"):
            persistent = runtime_with_vae(True, True, Mock(return_value=(pixels,)))
            persistent._decode(torch.zeros((1, 16, 1, 1, 1)), force_tiled_decode=True)
            self.assertTrue(persistent.vae.use_tiling)
            persistent.vae.disable_tiling.assert_not_called()
            failing = runtime_with_vae(False, False, Mock(side_effect=RuntimeError("decode failed")))
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                failing._decode(torch.zeros((1, 16, 1, 1, 1)), force_tiled_decode=True)
            self.assertFalse(failing.vae.use_tiling)
            failing.vae.disable_tiling.assert_called_once_with()
            unavailable = runtime_with_vae(False, False, Mock(return_value=(pixels,)))
            del unavailable.vae.disable_tiling
            with self.assertRaisesRegex(RuntimeError, "reversible"):
                unavailable._decode(torch.zeros((1, 16, 1, 1, 1)), force_tiled_decode=True)

    def test_forced_tiled_decode_requests_comfy_geometry_and_reports_what_the_vae_resolved(self):
        """Comfy decodes each tile with VAEDecodeTiled(tile_size=512, overlap=64) -> stride 448."""
        def enable_tiling(tile_sample_min_height=None, tile_sample_min_width=None,
                          tile_sample_stride_height=None, tile_sample_stride_width=None):
            calls.append({
                "tile_sample_min_height": tile_sample_min_height, "tile_sample_min_width": tile_sample_min_width,
                "tile_sample_stride_height": tile_sample_stride_height, "tile_sample_stride_width": tile_sample_stride_width,
            })
            runtime.vae.use_tiling = True
            for key, value in calls[-1].items():
                if value is not None: setattr(runtime.vae, key, value)

        calls = []
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.float32
        runtime._transformer_resident = False
        runtime._transformer_group_offload = False
        runtime._vae_tiling_required = False
        runtime.last_generation_metrics = {}
        pixels = torch.zeros((1, 3, 1, 1, 1))
        runtime.vae = SimpleNamespace(
            config=SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16), use_tiling=False,
            tile_sample_min_height=256, tile_sample_min_width=256,
            tile_sample_stride_height=192, tile_sample_stride_width=192,
            to=Mock(), decode=Mock(return_value=(pixels,)), enable_tiling=enable_tiling,
            disable_tiling=Mock(side_effect=lambda: setattr(runtime.vae, "use_tiling", False)),
        )
        real_torch_device = torch.device
        with patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)), \
                patch("backend.anima_pipeline._empty_cuda_cache"):
            runtime._decode(torch.zeros((1, 16, 1, 1, 1)), force_tiled_decode=True)

        self.assertEqual(calls, [{"tile_sample_min_height": 512, "tile_sample_min_width": 512,
                                  "tile_sample_stride_height": 448, "tile_sample_stride_width": 448}])
        metrics = runtime.last_generation_metrics["refinement.vae_decode"]
        self.assertEqual(metrics["requested_tiled_decode"], {"tile": 512, "overlap": 64, "stride": 448})
        self.assertEqual(metrics["actual_vae_mode"], "tiled")
        self.assertEqual(metrics["resolved_tiled_decode"], {
            "mode": "diffusers_explicit_geometry",
            "tile_sample_min_height": 512, "tile_sample_min_width": 512,
            "tile_sample_stride_height": 448, "tile_sample_stride_width": 448,
            "overlap_height": 64, "overlap_width": 64, "matches_comfy_contract": True,
        })

    def test_resolved_tiled_decode_reports_unknown_when_the_vae_hides_its_geometry(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.vae = SimpleNamespace(enable_tiling=Mock())
        self.assertEqual(runtime._resolved_tiled_decode()["mode"], "unknown")
        self.assertFalse(runtime._resolved_tiled_decode().get("matches_comfy_contract", False))
        # A Mock exposes only (*args, **kwargs), so no geometry is forced onto VAEs that cannot take it.
        self.assertEqual(runtime._comfy_tiled_decode_kwargs(), {})

    def test_each_lora_weight_family_is_fused_exactly_once(self):
        """The diffusion and text passes must not fuse each other's specs.

        `_lora_weight_resolver` dispatches on `spec["family"]` alone, so a pass that received the
        full spec list would fuse every target a second time and silently double LoRA strength.
        """
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.float32
        runtime.transformer = SimpleNamespace(device=torch.device("cuda"))
        runtime.text_encoder = SimpleNamespace(device=torch.device("cuda"))
        runtime.lora_plans = [
            ("first.safetensors", 0.45, [{"family": "diffusion", "target": "a.weight"},
                                         {"family": "qwen", "target": "b.weight"}]),
            ("second.safetensors", 0.8, [{"family": "diffusion", "target": "c.weight"}]),
        ]
        calls = []
        with patch("backend.anima_pipeline._apply_anima_lora_groups_on_gpu",
                   side_effect=lambda specs, *args, **kwargs: calls.append(list(specs))):
            runtime._apply_text_lora_on_gpu()
            runtime._apply_diffusion_lora_on_gpu()

        fused = [(spec["family"], spec["target"]) for call in calls for spec in call]
        self.assertEqual(sorted(fused), [("diffusion", "a.weight"), ("diffusion", "c.weight"), ("qwen", "b.weight")])
        self.assertEqual(len(fused), len(set(fused)), "every LoRA target must be fused exactly once")

        # A LoRA with no spec in the requested family contributes no call at all.
        self.assertEqual([[spec["family"] for spec in call] for call in calls], [["qwen"], ["diffusion"], ["diffusion"]])

    def test_ancestral_noise_generator_matches_the_comfy_protocol(self):
        """Comfy seeds a fresh generator on the sampling device, it does not continue the CPU stream.

        `comfy/k_diffusion/sampling.py:default_noise_sampler` builds
        `torch.Generator(device=x.device).manual_seed(seed)` and only applies the +1 offset when the
        latent is on CPU; `comfy/samplers.py:outer_sample` moves the latent to CUDA before sampling.
        """
        seed = 1015878324182247
        base = torch.Generator(device="cpu").manual_seed(seed)
        torch.randn((1, 16, 1, 4, 4), generator=base, dtype=torch.float32, device="cpu")  # initial latent draw

        # Continuing the advanced stream is not the same as restarting from the base seed.
        advanced = torch.randn((1, 16, 1, 4, 4), generator=base, dtype=torch.float32, device="cpu")
        restarted = torch.randn(
            (1, 16, 1, 4, 4), generator=torch.Generator(device="cpu").manual_seed(seed),
            dtype=torch.float32, device="cpu",
        )
        self.assertFalse(torch.equal(advanced, restarted))

        # The derived ancestral generator restarts from the base seed and lives on the sampling device.
        self.assertEqual(base.initial_seed(), seed)
        if torch.cuda.is_available():
            derived = anima_pipeline._derive_cuda_generators([base])
            self.assertEqual(len(derived), 1)
            self.assertEqual(derived[0].initial_seed(), seed)
            self.assertEqual(torch.device(derived[0].device).type, "cuda")
        with self.assertRaises(ValueError):
            anima_pipeline._derive_cuda_generators([torch.Generator(device="cuda").manual_seed(seed)]
                                                   if torch.cuda.is_available() else ["not-a-generator"])

    def test_lora_family_plans_selects_only_the_requested_family(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.lora_plans = [
            ("only_diffusion.safetensors", 0.7, [{"family": "diffusion", "target": "a.weight"}]),
            ("mixed.safetensors", 0.6, [{"family": "diffusion", "target": "b.weight"},
                                        {"family": "qwen", "target": "c.weight"}]),
        ]
        diffusion = runtime._lora_family_plans("diffusion")
        qwen = runtime._lora_family_plans("qwen")
        self.assertEqual([(path, multiplier, [s["target"] for s in specs]) for path, multiplier, specs in diffusion],
                         [("only_diffusion.safetensors", 0.7, ["a.weight"]), ("mixed.safetensors", 0.6, ["b.weight"])])
        self.assertEqual([(path, multiplier, [s["target"] for s in specs]) for path, multiplier, specs in qwen],
                         [("mixed.safetensors", 0.6, ["c.weight"])])
        runtime.lora_plans = []
        self.assertEqual(runtime._lora_family_plans("diffusion"), [])

    def test_cuda_stage_timing_never_overwrites_contract_metrics_recorded_by_the_stage(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.last_generation_metrics = {"stage": {"stale": True}}

        def operation():
            runtime.last_generation_metrics["stage"] = {"requested_tiled_decode": {"tile": 512}}
            return "value"

        self.assertEqual(runtime._run_cuda_stage("stage", operation), "value")
        recorded = runtime.last_generation_metrics["stage"]
        self.assertEqual(recorded["requested_tiled_decode"], {"tile": 512})
        self.assertIn("seconds", recorded)
        self.assertNotIn("stale", recorded)

    def test_decode_casts_fp32_trajectory_to_model_dtype_at_vae_boundary(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.bfloat16
        runtime._transformer_resident = False
        runtime._transformer_group_offload = False
        runtime.vae = SimpleNamespace(
            config=SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16),
            to=Mock(),
            decode=Mock(return_value=(torch.zeros((1, 3, 1, 1, 1)),)),
        )
        real_torch_device = torch.device
        with (
            patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)),
            patch("backend.anima_pipeline._empty_cuda_cache"),
        ):
            runtime._decode(torch.zeros((1, 16, 1, 1, 1), dtype=torch.float32))
        self.assertEqual(runtime.vae.decode.call_args.args[0].dtype, torch.bfloat16)

    def test_request_validation_is_lightweight_and_strict(self):
        generator = torch.Generator(device="cpu")
        AnimaRuntime._validate_generation_request(512, 512, 20, 7.0, "euler", "simple", [generator], "none")
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            AnimaRuntime._validate_generation_request(513, 512, 20, 7.0, "euler", "simple", [generator], "none")
        with self.assertRaisesRegex(ValueError, "sampler"):
            AnimaRuntime._validate_generation_request(512, 512, 20, 7.0, "dpmpp", "simple", [generator], "none")
        AnimaRuntime._validate_generation_request(512, 512, 20, 7.0, "dpmpp_2m", "karras", [generator], "none")
        with self.assertRaisesRegex(ValueError, "scheduler"):
            AnimaRuntime._validate_generation_request(512, 512, 20, 7.0, "euler", "unknown", [generator], "none")
        AnimaRuntime._validate_generation_request(512, 512, 20, 7.0, "euler", "simple", [generator], "pag")
        with self.assertRaisesRegex(ValueError, "pag_scale"):
            AnimaRuntime._validate_generation_request(
                512, 512, 20, 7.0, "euler", "simple", [generator], "pag", pag_scale=float("nan")
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            AnimaRuntime._validate_generation_request(512, 512, 20, 7.0, "euler", "simple", [], "none")

    def test_all_visible_sampler_scheduler_pairs_pass_runtime_validation(self):
        generator = torch.Generator(device="cpu")
        for sampler in ANIMA_SAMPLERS:
            for scheduler in ANIMA_SCHEDULERS:
                with self.subTest(sampler=sampler, scheduler=scheduler):
                    AnimaRuntime._validate_generation_request(
                        512, 512, 20, 7.0, sampler, scheduler, [generator], "none"
                    )

    def test_group_offload_configuration_is_one_block_idempotent_and_strict(self):
        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(28)])
                self.enable_group_offload = Mock()

        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False
        runtime._poisoned = False
        runtime._transformer_group_offload = False
        runtime._transformer_blocks_per_group = 0
        runtime.transformer = Transformer()

        runtime.enable_transformer_group_offload(1)
        runtime.enable_transformer_group_offload(1)

        runtime.transformer.enable_group_offload.assert_called_once_with(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=1,
            non_blocking=False,
            use_stream=False,
            record_stream=False,
        )
        with self.assertRaisesRegex(RuntimeError, "different group size"):
            runtime.enable_transformer_group_offload(2)

    def test_group_offload_forward_failure_poisons_and_removes_hooks(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._poisoned = False
        runtime._transformer_group_offload = True
        runtime.transformer = Mock(side_effect=RuntimeError("forward failed"))
        runtime._remove_transformer_group_offload = Mock()

        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            runtime._transformer_prediction(hidden_states=torch.zeros(1))

        self.assertTrue(runtime._poisoned)
        runtime._remove_transformer_group_offload.assert_called_once_with()

    def test_group_offload_optional_attention_retry_failure_also_poisons_and_cleans_hooks(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.float32
        runtime.attention_backend = "efficient"
        runtime._transformer_group_offload = True
        runtime._poisoned = False
        runtime.transformer = Mock(side_effect=[RuntimeError("kernel rejected"), RuntimeError("native rejected")])
        runtime.configure_attention_backend = Mock(side_effect=lambda backend: setattr(runtime, "attention_backend", backend))
        @contextmanager
        def attention_scope():
            yield
        runtime._attention_scope = attention_scope
        runtime.last_generation_metrics = {}
        runtime._remove_transformer_group_offload = Mock()
        with self.assertRaisesRegex(RuntimeError, "native rejected"):
            runtime._transformer_prediction(hidden_states=torch.zeros(1))
        self.assertTrue(runtime._poisoned)
        runtime._remove_transformer_group_offload.assert_called_once_with()

    def test_cfg_zero_star_always_encodes_an_unconditional_branch(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False
        runtime._validate_generation_request = Mock()
        runtime._encode_prompts = Mock(return_value=(torch.empty(2, 1, 1), torch.empty(2, 1)))
        runtime._sample = Mock(return_value=torch.empty(1, 16, 1, 8, 8))
        runtime._decode = Mock(return_value=[])
        runtime.dtype = torch.float16
        generator = torch.Generator(device="cpu")
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "is_bf16_supported", return_value=True),
        ):
            runtime.generate_batch("prompt", "negative", 64, 64, 2, 1.0, "euler", "simple", [generator], "cfg_zero_star")
        runtime._encode_prompts.assert_called_once_with("prompt", "negative", True)

    def test_low_memory_sampling_microbatches_preserve_output_order(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False
        runtime.dtype = torch.float16
        runtime._validate_generation_request = Mock()
        runtime._encode_prompts = Mock(return_value=(torch.empty(1, 1, 1), torch.empty(1, 1)))
        sampled_batches = []

        def sample(_embeddings, _masks, _width, _height, _steps, _cfg, _sampler, generators, _guidance, callback, **_kwargs):
            values = torch.tensor([generator.initial_seed() for generator in generators], dtype=torch.float32)
            sampled_batches.append(list(values))
            if callback:
                callback(1, 1, values)
            return values[:, None]

        runtime._sample = Mock(side_effect=sample)
        runtime._decode = Mock(side_effect=lambda latents: list(latents[:, 0]))
        generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in (3, 4, 5)]
        steps = []
        with patch.object(torch.cuda, "is_available", return_value=True):
            images = runtime.generate_batch(
                "prompt", "", 64, 64, 1, 1.0, "euler", "simple", generators, "none",
                on_step=lambda step, total, _latents: steps.append((step, total)), sampling_batch_size=1,
            )

        self.assertEqual([float(value) for value in images], [3.0, 4.0, 5.0])
        self.assertEqual(sampled_batches, [[3.0], [4.0], [5.0]])
        self.assertEqual(steps, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual(set(runtime.last_generation_metrics), {"prompt_encode", "sampling", "vae_decode"})
        self.assertTrue(all(metric["seconds"] >= 0 for metric in runtime.last_generation_metrics.values()))

    def test_multichunk_sampling_metrics_aggregate_actual_execution_not_last_chunk(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False; runtime.dtype = torch.float16; runtime.batch_cfg = False
        runtime._validate_generation_request = Mock()
        runtime._encode_prompts = Mock(return_value=(torch.empty(2, 1, 1), torch.empty(2, 1)))
        runtime._decode = Mock(return_value=[])
        calls = []
        def sample(*args, **_kwargs):
            calls.append(1)
            runtime._last_sampling_execution = {
                "actual_transformer_invocations": 3 if len(calls) == 1 else 5,
                "peak_batch_copies": 2 if len(calls) == 1 else 1,
                "cfg_batch_attempts": 1 if len(calls) == 1 else 0,
            }
            return torch.zeros((1, 1))
        runtime._sample = Mock(side_effect=sample)
        generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in (1, 2)]
        with patch.object(torch.cuda, "is_available", return_value=True):
            runtime.generate_batch("p", "n", 64, 64, 2, 5.0, "euler", "simple", generators, "none", sampling_batch_size=1)
        metrics = runtime.last_generation_metrics["sampling"]
        self.assertEqual(metrics["actual_transformer_invocations"], 8)
        self.assertEqual(metrics["sequential_transformer_invocations"], 8)
        self.assertEqual(metrics["executed_denoise_updates"], 4)
        self.assertEqual(metrics["schedule_construction_steps"], 4)
        self.assertEqual(metrics["peak_batch_copies"], 2)
        self.assertEqual(metrics["chunk_execution_known_count"], 2)
        self.assertTrue(metrics["chunk_execution_complete"])

    def test_group_cfg_batch_oom_buffers_public_callbacks_but_keeps_checkpoints(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False
        runtime.dtype = torch.float32
        runtime.batch_cfg = True
        runtime._transformer_group_offload = True
        runtime._validate_generation_request = Mock()
        runtime._encode_prompts = Mock(return_value=(torch.empty(2, 1, 1), torch.empty(2, 1)))
        runtime._decode = Mock(side_effect=lambda latents: [latents.clone()])
        attempts, visible, checkpoints, stored = [], [], [], []
        def sample(*args, **kwargs):
            callback, checkpoint = args[9], kwargs["on_step_checkpoint"]
            latent = torch.tensor([1.0])
            checkpoint(1, 2, latent); callback(1, 2, latent)
            stored.append(latent)
            if len(attempts) == 0:
                attempts.append("batch")
                latent.fill_(99)
                raise _GroupCfgBatchOom("oom")
            callback(2, 2, torch.tensor([2.0])); checkpoint(2, 2, torch.tensor([2.0]))
            return torch.zeros((1, 1))
        runtime._sample = Mock(side_effect=sample)
        generator = torch.Generator(device="cpu").manual_seed(7)
        with patch.object(torch.cuda, "is_available", return_value=True):
            runtime.generate_batch("p", "n", 64, 64, 2, 5.0, "euler", "simple", [generator], "none",
                on_step=lambda step, _total, latent: visible.append((step, float(latent[0]))),
                on_step_checkpoint=lambda step, _total, _latent: checkpoints.append(step))
        self.assertEqual(checkpoints, [1, 1, 2])
        self.assertEqual(visible, [(1, 1.0), (2, 2.0)])
        self.assertEqual(float(stored[0][0]), 99.0)
        self.assertFalse(runtime.batch_cfg)

    def test_base_batch_then_resident_then_group_replay_is_callback_atomic_and_records_chunk_history(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False; runtime._poisoned = False; runtime.dtype = torch.float16; runtime.batch_cfg = True
        runtime._transformer_group_offload = False; runtime._transformer_blocks_per_group = 0
        runtime.transformer = SimpleNamespace(to=Mock()); runtime.keep_transformer_resident = False
        runtime._start_transformer_transfer = Mock(); runtime._wait_transformer_transfer = Mock()
        runtime._validate_generation_request = Mock(); runtime._encode_prompts = Mock(return_value=(torch.empty(2, 1, 1), torch.empty(2, 1)))
        runtime._decode = Mock(side_effect=lambda latents: [row.clone() for row in latents])
        runtime.enable_transformer_group_offload = Mock(side_effect=lambda _n: setattr(runtime, "_transformer_group_offload", True))
        attempts, public, checkpoints = [0, 0], [], []
        def sample(*args, **kwargs):
            chunk = args[7]; index = int(chunk[0].initial_seed() - 101); attempts[index] += 1
            callback, checkpoint = args[9], kwargs["on_step_checkpoint"]
            value = torch.randn((), generator=chunk[0]); checkpoint(1, 2, value); callback(1, 2, value)
            if attempts[index] == 1: raise _GroupCfgBatchOom("oom")
            if attempts[index] == 2: raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            value = torch.randn((), generator=chunk[0]); checkpoint(2, 2, value); callback(2, 2, value)
            return value.reshape(1, 1)
        runtime._sample = Mock(side_effect=sample)
        generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in (101, 102)]
        with patch.object(torch.cuda, "is_available", return_value=True):
            result = runtime.generate_batch("p", "n", 64, 64, 2, 5.0, "euler_ancestral", "simple", generators, "none",
                sampling_batch_size=1, on_step=lambda step, _total, latent: public.append((step, float(latent))),
                on_step_checkpoint=lambda step, _total, _latent: checkpoints.append(step))
        self.assertEqual(attempts, [3, 3]); self.assertEqual(public, [(1, public[0][1]), (2, public[1][1]), (3, public[2][1]), (4, public[3][1])])
        self.assertEqual(len(public), 4); self.assertGreater(len(checkpoints), len(public))  # checkpoints are control-only and may repeat.
        history = runtime.last_generation_metrics["sampling"]["fallback_history"]
        self.assertEqual([(item["from"], item["to"]) for item in history], [("resident_batched", "resident_sequential"), ("resident_sequential", "group_sequential")] * 2)
        self.assertEqual([item["chunk_index"] for item in history], [0, 0, 1, 1])
        self.assertTrue(all(item["callback_buffer_discarded"] and item["generator_restored"] for item in history))
        self.assertFalse(runtime.batch_cfg); self.assertTrue(runtime.transformer_group_offload_enabled)
        clean_generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in (101, 102)]
        expected = []
        for generator in clean_generators:
            torch.randn((), generator=generator); expected.append(torch.randn((), generator=generator))
        self.assertTrue(all(torch.equal(generator.get_state(), clean.get_state()) for generator, clean in zip(generators, clean_generators)))
        self.assertTrue(torch.equal(torch.cat(result).flatten(), torch.stack(expected)))

    def test_refine_batch_then_resident_then_group_replay_is_callback_atomic_and_records_history(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False; runtime._poisoned = False; runtime.dtype = torch.float32; runtime.latent_state_mode = "fp32"; runtime.noise_device = "cpu"
        runtime.batch_cfg = True; runtime._transformer_group_offload = False; runtime._transformer_blocks_per_group = 0
        runtime.transformer = SimpleNamespace(to=Mock()); runtime.keep_transformer_resident = False; runtime.last_generation_metrics = {}
        runtime._validate_generation_request = Mock(); runtime._encode_prompts = Mock(return_value=(torch.ones((2, 1, 1)), torch.ones((2, 1))))
        runtime._encode_images = Mock(return_value=torch.zeros((1, 16, 1, 4, 4))); runtime._decode = Mock(side_effect=lambda latents, **_k: [latents.clone()])
        runtime.enable_transformer_group_offload = Mock(side_effect=lambda _n: setattr(runtime, "_transformer_group_offload", True))
        starts, public, checkpoints, attempts = [], [], [], []
        def start(_source, generators, _sigma):
            initial = torch.randn((1, 1), generator=generators[0]); noise = torch.randn((1, 1), generator=generators[0]); starts.append((initial, noise)); return initial, noise
        runtime._refinement_start = start
        def sample(*args, **kwargs):
            attempts.append(1); callback, checkpoint = args[9], kwargs["on_step_checkpoint"]
            value = torch.randn((), generator=args[7][0]); checkpoint(1, 2, value); callback(1, 2, value)
            if len(attempts) == 1: raise _GroupCfgBatchOom("oom")
            if len(attempts) == 2: raise RuntimeError("CUDA out of memory")
            value = torch.randn((), generator=args[7][0]); checkpoint(2, 2, value); callback(2, 2, value); return value.reshape(1, 1)
        runtime._sample = Mock(side_effect=sample)
        generator = torch.Generator(device="cpu").manual_seed(77)
        with self._cpu_cuda_path():
            result = runtime.refine_batch([Image.new("RGB", (32, 32))], "p", "n", 2, 1.0, 5.0, "euler_ancestral", "simple", [generator], "none",
                on_step=lambda step, _total, latent: public.append((step, float(latent))), on_step_checkpoint=lambda step, _total, _latent: checkpoints.append(step))
        self.assertEqual(len(attempts), 3); self.assertEqual(len(starts), 3); self.assertEqual(len(public), 2); self.assertGreater(len(checkpoints), len(public))
        self.assertTrue(all(id(starts[0][part]) != id(starts[1][part]) != id(starts[2][part]) for part in (0, 1)))
        self.assertTrue(torch.equal(starts[0][0], starts[1][0]) and torch.equal(starts[1][0], starts[2][0]))
        history = runtime.last_generation_metrics["refinement.sampling"]["fallback_history"]
        self.assertEqual([(item["from"], item["to"]) for item in history], [("resident_batched", "resident_sequential"), ("resident_sequential", "group_sequential")])
        self.assertFalse(runtime.batch_cfg); self.assertTrue(runtime.transformer_group_offload_enabled)
        clean = torch.Generator(device="cpu").manual_seed(77)
        for _ in range(4): torch.randn((), generator=clean)
        self.assertTrue(torch.equal(generator.get_state(), clean.get_state()))
        self.assertEqual(len(result), 1)

    def test_real_refine_sequential_and_batch_metrics_are_exact_for_twelve_point_two(self):
        for batch_cfg, expected_actual, expected_peak in ((False, 24, 1), (True, 12, 2)):
            with self.subTest(batch_cfg=batch_cfg):
                runtime = self._real_refine_runtime(batch_cfg)
                calls = []
                def predict(**kwargs):
                    calls.append(kwargs["encoder_hidden_states"].detach().clone())
                    return torch.zeros_like(kwargs["hidden_states"])
                runtime._transformer_prediction = predict
                with self._cpu_cuda_path():
                    runtime.refine_batch([Image.new("RGB", (32, 32))], "p", "n", 12, .2, 5.0,
                        "euler", "simple", [torch.Generator(device="cpu").manual_seed(17)], "none")
                metrics = runtime.last_generation_metrics["refinement.sampling"]
                self.assertEqual(metrics["schedule_construction_steps"], 60)
                self.assertEqual(metrics["executed_denoise_updates"], 12)
                self.assertEqual(metrics["sequential_transformer_invocations"], 24)
                self.assertEqual(metrics["actual_transformer_invocations"], expected_actual)
                self.assertEqual(metrics["peak_batch_copies"], expected_peak)
                self.assertEqual(len(calls), expected_actual)
                self.assertTrue(all(call.shape[0] == (2 if batch_cfg else 1) for call in calls))
                if batch_cfg:
                    self.assertTrue(all(torch.equal(call[0], torch.ones((1, 1))) and torch.equal(call[1], -torch.ones((1, 1))) for call in calls))

    def test_real_refine_group_batch_oom_rebuilds_initial_and_replays_once(self):
        runtime = self._real_refine_runtime(True)
        starts, published, checkpoints, forwards = [], [], [], []
        original_start = runtime._refinement_start
        def start(*args):
            initial, noise = original_start(*args)
            starts.append((initial, noise))
            return initial, noise
        runtime._refinement_start = start
        def predict(**kwargs):
            forwards.append(kwargs["hidden_states"].shape[0])
            if len(forwards) == 2:
                raise torch.cuda.OutOfMemoryError("oom")
            return torch.zeros_like(kwargs["hidden_states"])
        runtime._transformer_prediction = predict
        generator = torch.Generator(device="cpu").manual_seed(123)
        with self._cpu_cuda_path():
            result = runtime.refine_batch([Image.new("RGB", (32, 32))], "p", "n", 12, .2, 5.0,
                "euler", "simple", [generator], "none",
                on_step=lambda step, _total, _latent: published.append(step),
                on_step_checkpoint=lambda step, _total, _latent: checkpoints.append(step))
        self.assertEqual(len(starts), 2)
        self.assertNotEqual(id(starts[0][0]), id(starts[1][0]))
        self.assertNotEqual(id(starts[0][1]), id(starts[1][1]))
        self.assertTrue(torch.equal(starts[0][0], starts[1][0]))
        self.assertTrue(torch.equal(starts[0][1], starts[1][1]))
        self.assertEqual(published, list(range(1, 13)))
        self.assertEqual(checkpoints, [1] + list(range(1, 13)))
        self.assertFalse(runtime.batch_cfg)
        self.assertEqual(forwards[0], 2)
        self.assertTrue(all(size == 1 for size in forwards[2:]))
        self.assertEqual(len(result), 1)
        clean = self._real_refine_runtime(False)
        clean._transformer_prediction = lambda **kwargs: torch.zeros_like(kwargs["hidden_states"])
        clean_generator = torch.Generator(device="cpu").manual_seed(123)
        with self._cpu_cuda_path():
            clean_result = clean.refine_batch([Image.new("RGB", (32, 32))], "p", "n", 12, .2, 5.0,
                "euler", "simple", [clean_generator], "none")
        self.assertTrue(torch.equal(result[0], clean_result[0]))
        self.assertTrue(torch.equal(generator.get_state(), clean_generator.get_state()))
        subsequent_shapes = []
        runtime._transformer_prediction = lambda **kwargs: subsequent_shapes.append(kwargs["hidden_states"].shape[0]) or torch.zeros_like(kwargs["hidden_states"])
        with self._cpu_cuda_path():
            runtime.refine_batch([Image.new("RGB", (32, 32))], "p", "n", 1, 1.0, 5.0,
                "euler", "simple", [torch.Generator(device="cpu").manual_seed(9)], "none")
        self.assertEqual(subsequent_shapes, [1, 1])

    def test_real_refine_speculative_cancel_does_not_publish_or_replay(self):
        runtime = self._real_refine_runtime(True)
        runtime._transformer_prediction = lambda **kwargs: torch.zeros_like(kwargs["hidden_states"])
        generator = torch.Generator(device="cpu").manual_seed(5)
        visible = []
        with self._cpu_cuda_path(), self.assertRaises(GenerationCancelled):
            runtime.refine_batch([Image.new("RGB", (32, 32))], "p", "n", 12, .2, 5.0,
                "euler", "simple", [generator], "none", on_step=lambda *_args: visible.append("published"),
                on_step_checkpoint=lambda *_args: (_ for _ in ()).throw(GenerationCancelled()))
        self.assertEqual(visible, [])
        self.assertTrue(runtime.batch_cfg)

    def test_resident_sampling_oom_restores_generators_and_retries_with_group_offload(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False
        runtime._poisoned = False
        runtime._transformer_group_offload = False
        runtime._transformer_blocks_per_group = 0
        runtime.dtype = torch.float16
        runtime.transformer = SimpleNamespace(to=Mock())
        runtime._validate_generation_request = Mock()
        runtime._encode_prompts = Mock(return_value=(torch.empty(1, 1, 1), torch.empty(1, 1)))
        runtime._decode = Mock(return_value=[])
        runtime.enable_transformer_group_offload = Mock(
            side_effect=lambda _blocks: setattr(runtime, "_transformer_group_offload", True)
        )
        observed_states = []

        def sample(_embeddings, _masks, _width, _height, _steps, _cfg, _sampler, generators, _guidance, _callback, **_kwargs):
            observed_states.append(generators[0].get_state().clone())
            if len(observed_states) == 1:
                torch.randn((1,), generator=generators[0])
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return torch.zeros((1, 1))

        runtime._sample = Mock(side_effect=sample)
        generator = torch.Generator(device="cpu").manual_seed(123)
        with patch.object(torch.cuda, "is_available", return_value=True):
            runtime.generate_batch(
                "prompt", "", 64, 64, 1, 1.0, "euler", "simple", [generator], "none"
            )

        self.assertEqual(runtime._sample.call_count, 2)
        self.assertTrue(torch.equal(observed_states[0], observed_states[1]))
        runtime.transformer.to.assert_any_call("cpu")
        runtime.enable_transformer_group_offload.assert_called_once_with(1)
        self.assertEqual(
            runtime.last_generation_metrics["sampling_fallback"],
            {
                "from": "staged_transformer_resident",
                "to": "staged_transformer_group_offload",
                "reason": "cuda_oom",
                "stage": "sampling",
                "attempts": 1,
                "generator_states_restored": True,
            },
        )

    def test_resident_oom_replay_is_once_only_for_multiple_cpu_generators(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False
        runtime._poisoned = False
        runtime._transformer_group_offload = False
        runtime._transformer_blocks_per_group = 0
        runtime.dtype = torch.float16
        runtime.transformer = SimpleNamespace(to=Mock())
        runtime._validate_generation_request = Mock()
        runtime._encode_prompts = Mock(return_value=(torch.empty(1, 1, 1), torch.empty(1, 1)))
        runtime._decode = Mock(return_value=[])
        runtime.enable_transformer_group_offload = Mock(side_effect=lambda _blocks: setattr(runtime, "_transformer_group_offload", True))
        generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in (11, 12)]
        initial_states = [generator.get_state().clone() for generator in generators]

        def sample(*args, **_kwargs):
            for generator in args[7]:
                torch.randn((1,), generator=generator)
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        runtime._sample = Mock(side_effect=sample)
        with patch.object(torch.cuda, "is_available", return_value=True):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                runtime.generate_batch("prompt", "", 64, 64, 1, 1.0, "euler", "simple", generators, "none")
        self.assertEqual(runtime._sample.call_count, 2)
        self.assertTrue(runtime.transformer_group_offload_enabled)
        for generator, state in zip(generators, initial_states):
            expected = torch.Generator(device="cpu")
            expected.set_state(state)
            torch.randn((1,), generator=expected)
            self.assertTrue(torch.equal(generator.get_state(), expected.get_state()))

    def test_ancestral_cfg_multichunk_resident_oom_replay_matches_clean_group_execution(self):
        def runtime(group_offload):
            value = AnimaRuntime.__new__(AnimaRuntime)
            value._closed = False
            value._poisoned = False
            value._transformer_group_offload = group_offload
            value._transformer_blocks_per_group = 1 if group_offload else 0
            value.dtype = torch.float16
            value.transformer = SimpleNamespace(to=Mock())
            value._validate_generation_request = Mock()
            value._encode_prompts = Mock(return_value=(torch.empty(1, 1, 1), torch.empty(1, 1)))
            value._decode = Mock(side_effect=lambda latents: [tuple(row.tolist()) for row in latents])
            value.enable_transformer_group_offload = Mock(
                side_effect=lambda _blocks: setattr(value, "_transformer_group_offload", True)
            )
            return value

        def deterministic_ancestral_cfg(*args, **_kwargs):
            generators = args[7]
            # One noise draw plus an ancestral draw per output models a CFG>1
            # Euler ancestral chunk without requiring CUDA kernels.
            values = torch.stack([
                torch.stack((torch.randn((), generator=generator), torch.randn((), generator=generator)))
                for generator in generators
            ])
            if not value.transformer_group_offload_enabled and not attempts:
                attempts.append("resident")
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            calls.append(("group" if value.transformer_group_offload_enabled else "resident", values.clone()))
            return values

        seeds = (101, 202)
        value = runtime(False)
        attempts, calls = [], []
        value._sample = Mock(side_effect=deterministic_ancestral_cfg)
        generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in seeds]
        with patch.object(torch.cuda, "is_available", return_value=True):
            replayed = value.generate_batch(
                "prompt", "negative", 64, 64, 2, 5.0, "euler_ancestral", "simple",
                generators, "none", sampling_batch_size=1,
            )

        clean = runtime(True)
        clean_calls = []
        def clean_sample(*args, **_kwargs):
            chunk = args[7]
            values = torch.stack([
                torch.stack((torch.randn((), generator=generator), torch.randn((), generator=generator)))
                for generator in chunk
            ])
            clean_calls.append(values.clone())
            return values
        clean._sample = Mock(side_effect=clean_sample)
        clean_generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in seeds]
        with patch.object(torch.cuda, "is_available", return_value=True):
            expected = clean.generate_batch(
                "prompt", "negative", 64, 64, 2, 5.0, "euler_ancestral", "simple",
                clean_generators, "none", sampling_batch_size=1,
            )

        self.assertEqual(value._sample.call_count, 3)  # resident OOM, group replay, next group chunk
        self.assertEqual([mode for mode, _values in calls], ["group", "group"])
        self.assertEqual(replayed, expected)
        self.assertEqual(len(clean_calls), 2)
        for generator, clean_generator in zip(generators, clean_generators):
            self.assertTrue(torch.equal(generator.get_state(), clean_generator.get_state()))
        self.assertEqual(value.last_generation_metrics["sampling_fallback"]["attempts"], 1)

    def test_token_diagnostics_dual_tokenizes_clean_prompt_with_t5_aligned_weights(self):
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def __init__(self, split=False):
                self.split = split
                self.segments = []

            def __call__(self, text, add_special_tokens=False, truncation=False):
                self.segments.append(text)
                ids = [ord(character) for character in text]
                if self.split:
                    ids = [value for token in ids for value in (token, token)]
                return {"input_ids": ids}

            def build_inputs_with_special_tokens(self, token_ids):
                return [1, *token_ids, 2]

            def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):
                if already_has_special_tokens:
                    return [int(token in {1, 2}) for token in token_ids]
                return [1, *([0] * len(token_ids)), 1]

        prompt = "  (raw:2) prompt  "
        qwen = Tokenizer()
        t5 = Tokenizer(split=True)
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False
        runtime.qwen_tokenizer = qwen
        runtime.t5_tokenizer = t5

        diagnostics = runtime.token_diagnostics(prompt)

        self.assertEqual(qwen.segments, ["  ", "raw", " prompt  "])
        self.assertEqual(t5.segments, ["  ", "raw", " prompt  "])
        self.assertEqual(diagnostics["qwen"]["token_count"], len("  raw prompt  ") + 2)
        self.assertEqual(diagnostics["qwen"]["weighted_token_count"], 0)
        self.assertEqual(diagnostics["t5"]["token_count"], len("  raw prompt  ") * 2 + 2)
        self.assertEqual(diagnostics["t5"]["weighted_token_count"], 6)

    def test_prompt_encoding_applies_t5_weights_once_after_adapter_and_masks_padding(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.float32
        runtime.text_encoder = Mock()
        runtime.text_encoder.to.return_value = runtime.text_encoder
        runtime.text_encoder.return_value = SimpleNamespace(last_hidden_state=torch.ones((1, 4, 2)))
        runtime.llm_adapter = Mock()
        runtime.llm_adapter.to.return_value = runtime.llm_adapter
        runtime.llm_adapter.return_value = torch.full((1, 4, 2), 3.0)
        runtime._tokenize_texts = Mock(return_value=(
            {
                "input_ids": torch.tensor([[1, 2, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0, 0]]),
                "weights": torch.ones((1, 4)),
            },
            {
                "input_ids": torch.tensor([[1, 2, 3, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]]),
                "weights": torch.tensor([[1.0, 2.0, 0.5, 1.0]]),
            },
        ))
        real_torch_device = torch.device
        with (
            patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)),
            patch("backend.anima_pipeline._empty_cuda_cache"),
        ):
            adapted, mask = runtime._encode_prompts("(prompt:2)", "", False)

        expected = torch.tensor([[[3.0, 3.0], [6.0, 6.0], [1.5, 1.5], [0.0, 0.0]]])
        self.assertTrue(torch.equal(adapted, expected))
        self.assertTrue(torch.equal(mask, torch.tensor([[1, 1, 1, 0]])))
        source = runtime.llm_adapter.call_args.kwargs["source_hidden_states"]
        self.assertTrue(torch.equal(source[0, 2:], torch.zeros((2, 2))))

    def test_vae_encode_uses_normalized_video_input_posterior_mode_and_latent_config(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.float32
        posterior = Mock()
        raw = torch.full((1, 16, 1, 4, 4), 5.0)
        posterior.mode.return_value = raw
        runtime.vae = Mock()
        runtime.vae.to.return_value = runtime.vae
        runtime.vae.config = SimpleNamespace(latents_mean=[1.0] * 16, latents_std=[2.0] * 16)
        runtime.vae.encode.return_value = SimpleNamespace(latent_dist=posterior)
        image = Image.new("RGB", (32, 32), (0, 128, 255))
        real_torch_device = torch.device
        with (
            patch.object(torch, "device", side_effect=lambda value: real_torch_device("cpu") if value == "cuda" else real_torch_device(value)),
            patch("backend.anima_pipeline._empty_cuda_cache"),
        ):
            latents = runtime._encode_images([image])

        vae_input = runtime.vae.encode.call_args.args[0]
        self.assertEqual(vae_input.shape, (1, 3, 1, 32, 32))
        self.assertTrue(
            torch.allclose(
                vae_input[0, :, 0, 0, 0], torch.tensor([-1.0, 128 / 127.5 - 1.0, 1.0]), atol=1e-6
            )
        )
        posterior.mode.assert_called_once_with()
        self.assertTrue(torch.equal(latents, torch.full_like(raw, 2.0)))

    def test_public_refine_contract_uses_full_schedule_suffix_masks_callbacks_and_namespaced_metrics(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False
        runtime.dtype = torch.float32
        runtime.last_generation_metrics = {"sampling": {"seconds": 3.0}}
        runtime._encode_prompts = Mock(return_value=(torch.empty(1, 1, 1), torch.empty(1, 1)))
        source = torch.zeros((1, 16, 1, 4, 4))
        runtime._encode_images = Mock(return_value=source)
        callback_values = []

        def sample(*args, **kwargs):
            callback = args[9]
            total = len(kwargs["sigmas"]) - 1 - kwargs["start_index"]
            callback(1, total, source)
            callback(2, total, source)
            return source

        runtime._sample = Mock(side_effect=sample)
        decoded = [Image.new("RGB", (32, 32))]
        runtime._decode = Mock(return_value=decoded)
        image = Image.new("RGB", (32, 32), "white")
        mask = Image.new("L", (32, 32), 255)
        with patch.object(torch.cuda, "is_available", return_value=True):
            result = runtime.refine_batch(
                [image], "prompt", "negative", 20, 0.35, 1.0, "euler", "simple",
                [torch.Generator(device="cpu").manual_seed(4)], "none", masks=[mask],
                on_step=lambda step, total, _value: callback_values.append((step, total)),
            )

        self.assertIs(result, decoded)
        kwargs = runtime._sample.call_args.kwargs
        self.assertEqual(kwargs["start_index"], 0)
        self.assertEqual(len(kwargs["sigmas"]), 21)
        self.assertAlmostEqual(float(kwargs["sigmas"][0]), 0.618683875, places=6)
        self.assertEqual(kwargs["latent_mask"].shape, (1, 1, 1, 32, 32))
        self.assertEqual(callback_values, [(1, 20), (2, 20)])
        self.assertIn("sampling", runtime.last_generation_metrics)
        self.assertTrue(
            {"refinement.prompt_encode", "refinement.vae_encode", "refinement.sampling", "refinement.vae_decode"}
            <= set(runtime.last_generation_metrics)
        )
        self.assertEqual(runtime.last_generation_metrics["refinement.sampling"]["schedule_steps"], 57)
        metrics = runtime.last_generation_metrics["refinement.sampling"]
        self.assertEqual(metrics["requested_steps"], 20)
        self.assertEqual(metrics["executed_denoise_updates"], 20)
        self.assertEqual(metrics["schedule_construction_steps"], 57)
        self.assertEqual(metrics["sequential_transformer_invocations"], 20)

    def test_partial_refinement_cfg_metrics_do_not_count_schedule_construction_as_updates(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False
        runtime.dtype = torch.float32
        runtime.last_generation_metrics = {}
        runtime._encode_prompts = Mock(return_value=(torch.empty(2, 1, 1), torch.empty(2, 1)))
        runtime._encode_images = Mock(return_value=torch.zeros((1, 16, 1, 4, 4)))
        runtime._sample = Mock(return_value=torch.zeros((1, 16, 1, 4, 4)))
        runtime._decode = Mock(return_value=[Image.new("RGB", (32, 32))])
        with patch.object(torch.cuda, "is_available", return_value=True):
            runtime.refine_batch(
                [Image.new("RGB", (32, 32))], "prompt", "negative", 1, 0.2, 5.0,
                "euler", "simple", [torch.Generator(device="cpu")], "none",
            )
        metrics = runtime.last_generation_metrics["refinement.sampling"]
        self.assertEqual(metrics["schedule_steps"], 5)
        self.assertEqual(metrics["schedule_construction_steps"], 5)
        self.assertEqual(metrics["requested_steps"], 1)
        self.assertEqual(metrics["executed_denoise_updates"], 1)
        self.assertEqual(metrics["branch_invocations_per_update"], 2)
        self.assertEqual(metrics["sequential_transformer_invocations"], 2)

    def test_prepared_refinement_conditioning_and_sigmas_reuse_without_generator_mutation(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False
        runtime.dtype = torch.bfloat16
        runtime.last_generation_metrics = {}
        runtime._encode_prompts = Mock(return_value=(torch.ones((1, 2, 3), dtype=torch.bfloat16), torch.ones((1, 2))))
        runtime._encode_images = Mock(return_value=torch.zeros((1, 16, 1, 4, 4)))
        runtime._sample = Mock(return_value=torch.zeros((1, 16, 1, 4, 4)))
        runtime._decode = Mock(return_value=[Image.new("RGB", (32, 32))])
        prepared = runtime.prepare_refinement_conditioning("prompt", "negative", 1.0, "none")
        self.assertIsInstance(prepared, PreparedAnimaConditioning)
        self.assertEqual(prepared.embeddings.device.type, "cpu")
        self.assertEqual(runtime._encode_prompts.call_count, 1)
        sigmas = prepare_anima_refinement_sigmas(2, 1.0, "normal")
        image = Image.new("RGB", (32, 32))
        final_states = []
        with patch.object(torch.cuda, "is_available", return_value=True):
            for _ in range(4):
                generator = torch.Generator(device="cpu").manual_seed(9)
                runtime.refine_batch(
                    [image], "prompt", "negative", 2, 1.0, 1.0, "euler", "normal", [generator], "none",
                    prepared_conditioning=prepared, prepared_sigmas=sigmas,
                )
                final_states.append(generator.get_state().clone())
        self.assertEqual(runtime._encode_prompts.call_count, 1)
        self.assertTrue(all(torch.equal(final_states[0], state) for state in final_states[1:]))
        self.assertTrue(runtime.last_generation_metrics["refinement.sampling"]["conditioning_reused"])
        self.assertTrue(runtime.last_generation_metrics["refinement.sampling"]["sigmas_reused"])
        self.assertEqual(runtime.last_generation_metrics["refinement.sampling"]["latent_state_dtype"], "float32")
        self.assertEqual(runtime.last_generation_metrics["refinement.sampling"]["transformer_input_dtype"], "bfloat16")

    def test_invalid_prepared_refinement_inputs_fail_closed(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False
        runtime.dtype = torch.bfloat16
        image = Image.new("RGB", (32, 32))
        generator = torch.Generator(device="cpu")
        bad_conditioning = PreparedAnimaConditioning("other", "negative", 1.0, "none", torch.ones((1, 2, 3), dtype=torch.bfloat16))
        valid_conditioning = PreparedAnimaConditioning("prompt", "negative", 1.0, "none", torch.ones((1, 2, 3), dtype=torch.bfloat16))
        with patch.object(torch.cuda, "is_available", return_value=True):
            with self.assertRaisesRegex(ValueError, "does not match"):
                runtime.refine_batch(
                    [image], "prompt", "negative", 2, 1.0, 1.0, "euler", "normal", [generator], "none",
                    prepared_conditioning=bad_conditioning,
                )
            with self.assertRaisesRegex(ValueError, "terminate at zero"):
                runtime.refine_batch(
                    [image], "prompt", "negative", 2, 1.0, 1.0, "euler", "normal", [generator], "none",
                    prepared_conditioning=valid_conditioning, prepared_sigmas=torch.tensor([1.0, 0.5, 0.1]),
                )

    def test_prepared_conditioning_requires_cpu_runtime_dtype_shape_contiguity_and_clone_safety(self):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime.dtype = torch.bfloat16
        base = PreparedAnimaConditioning("prompt", "negative", 1.0, "none", torch.ones((1, 2, 3), dtype=torch.bfloat16))
        cloned = runtime._validate_prepared_conditioning(base, "prompt", "negative", 1.0, "none")
        cloned.zero_()
        self.assertTrue(torch.equal(base.embeddings, torch.ones_like(base.embeddings)))
        invalid = (
            (PreparedAnimaConditioning("prompt", "negative", 1.0, "none", torch.ones((1, 2, 3))), "dtype"),
            (PreparedAnimaConditioning("prompt", "negative", 1.0, "none", torch.ones((2, 3), dtype=torch.bfloat16)), "shape"),
            (PreparedAnimaConditioning("prompt", "negative", 1.0, "none", torch.ones((1, 3, 2), dtype=torch.bfloat16).transpose(1, 2)), "contiguous"),
        )
        for prepared, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    runtime._validate_prepared_conditioning(prepared, "prompt", "negative", 1.0, "none")
        if torch.cuda.is_available():
            cuda_prepared = PreparedAnimaConditioning(
                "prompt", "negative", 1.0, "none", torch.ones((1, 2, 3), device="cuda", dtype=torch.bfloat16)
            )
            with self.assertRaisesRegex(ValueError, "CPU"):
                runtime._validate_prepared_conditioning(cuda_prepared, "prompt", "negative", 1.0, "none")


class AnimaFusedAttentionTests(unittest.TestCase):
    """The fused processor must be the same attention, only in fewer passes."""

    HEAD_DIM = 128

    def _rotary(self, tokens, dtype=torch.float32):
        torch.manual_seed(11)
        frequencies = torch.randn(tokens, self.HEAD_DIM, dtype=torch.float32) * 3.0
        return torch.cos(frequencies).to(dtype), torch.sin(frequencies).to(dtype)

    def test_split_half_rotary_is_bit_identical_to_diffusers(self):
        apply_rotary_emb = _require_diffusers("models.embeddings", "apply_rotary_emb")
        tokens = 37
        rotary = self._rotary(tokens)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(3)
                hidden = torch.randn(1, 16, tokens, self.HEAD_DIM, dtype=dtype)
                expected = apply_rotary_emb(hidden, rotary, use_real=True, use_real_unbind_dim=-2)
                self.assertTrue(torch.equal(expected, _split_half_rotary(hidden, rotary)))

    def test_fused_rms_norm_matches_diffusers_within_one_ulp(self):
        RMSNorm = _require_diffusers("models.normalization", "RMSNorm")
        for dtype, tolerance in ((torch.float32, 0.0), (torch.float16, 2e-3), (torch.bfloat16, 1.6e-2)):
            with self.subTest(dtype=dtype):
                torch.manual_seed(5)
                norm = RMSNorm(self.HEAD_DIM, eps=1e-6, elementwise_affine=True).to(dtype)
                norm.weight.data.normal_(mean=1.0, std=0.05)
                hidden = torch.randn(1, 16, 29, self.HEAD_DIM, dtype=dtype)
                expected = norm(hidden)
                fused = _fused_rms_norm(norm, hidden)
                self.assertEqual(fused.dtype, expected.dtype)
                self.assertEqual(fused.shape, expected.shape)
                difference = (fused.float() - expected.float()).abs().max().item()
                self.assertLessEqual(difference, tolerance * expected.float().abs().max().item())

    def test_fused_rms_norm_defers_to_the_module_when_it_carries_a_bias(self):
        RMSNorm = _require_diffusers("models.normalization", "RMSNorm")
        torch.manual_seed(7)
        norm = RMSNorm(self.HEAD_DIM, eps=1e-6, elementwise_affine=True, bias=True)
        norm.bias.data.normal_()
        hidden = torch.randn(1, 4, 6, self.HEAD_DIM)
        self.assertTrue(torch.equal(_fused_rms_norm(norm, hidden), norm(hidden)))

    def test_processor_matches_the_stock_cosmos_processor(self):
        Attention = _require_diffusers("models.attention_processor", "Attention")
        stock = _require_diffusers("models.transformers.transformer_cosmos", "CosmosAttnProcessor2_0")
        dispatch = _require_diffusers("models.attention_dispatch", "dispatch_attention_fn")
        tokens = 41
        torch.manual_seed(13)
        attention = Attention(
            query_dim=16 * self.HEAD_DIM, cross_attention_dim=None, heads=16, dim_head=self.HEAD_DIM,
            qk_norm="rms_norm", elementwise_affine=True, out_bias=False, processor=stock(),
        ).eval().requires_grad_(False)
        hidden = torch.randn(1, tokens, 16 * self.HEAD_DIM)
        rotary = self._rotary(tokens)
        with torch.inference_mode():
            expected = attention(hidden, image_rotary_emb=rotary)
            attention.set_processor(AnimaCosmosAttnProcessor(dispatch))
            fused = attention(hidden, image_rotary_emb=rotary)
        self.assertEqual(fused.shape, expected.shape)
        torch.testing.assert_close(fused, expected, rtol=2e-5, atol=2e-5)

    def test_processor_matches_the_stock_processor_without_rotary(self):
        Attention = _require_diffusers("models.attention_processor", "Attention")
        stock = _require_diffusers("models.transformers.transformer_cosmos", "CosmosAttnProcessor2_0")
        dispatch = _require_diffusers("models.attention_dispatch", "dispatch_attention_fn")
        torch.manual_seed(17)
        attention = Attention(
            query_dim=16 * self.HEAD_DIM, cross_attention_dim=64, heads=16, dim_head=self.HEAD_DIM,
            qk_norm="rms_norm", elementwise_affine=True, out_bias=False, processor=stock(),
        ).eval().requires_grad_(False)
        hidden = torch.randn(1, 23, 16 * self.HEAD_DIM)
        context = torch.randn(1, 9, 64)
        with torch.inference_mode():
            expected = attention(hidden, encoder_hidden_states=context)
            attention.set_processor(AnimaCosmosAttnProcessor(dispatch))
            fused = attention(hidden, encoder_hidden_states=context)
        torch.testing.assert_close(fused, expected, rtol=2e-5, atol=2e-5)

    def test_installer_replaces_every_stock_processor_and_refuses_an_empty_sweep(self):
        stock = _require_diffusers("models.transformers.transformer_cosmos", "CosmosAttnProcessor2_0")
        modules = [SimpleNamespace(processor=stock()) for _ in range(3)]
        for module in modules:
            module.set_processor = (lambda target: lambda value: setattr(target, "processor", value))(module)
        transformer = SimpleNamespace(modules=lambda: iter(modules))
        self.assertEqual(install_anima_cosmos_attention_processor(transformer), 3)
        for module in modules:
            self.assertIsInstance(module.processor, AnimaCosmosAttnProcessor)

        with self.assertRaisesRegex(RuntimeError, "CosmosAttnProcessor2_0"):
            install_anima_cosmos_attention_processor(SimpleNamespace(modules=lambda: iter(())))


class AnimaAccelerationTests(unittest.TestCase):
    """Inductor compilation and SageAttention are opt-in and must fail soft."""

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = torch.nn.Linear(2, 2, bias=False)

        def forward(self, value):
            return self.attn1(value)

    def _runtime_with_blocks(self, count=28):
        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False
        runtime._poisoned = False
        runtime._transformer_group_offload = False
        runtime._transformer_compiled = False
        runtime._transformer_compile_mode = None
        runtime.last_generation_metrics = {}
        transformer = torch.nn.Module()
        transformer.transformer_blocks = torch.nn.ModuleList(self._Block() for _ in range(count))
        runtime.transformer = transformer
        return runtime

    def test_compilation_leaves_every_parameter_path_addressable(self):
        runtime = self._runtime_with_blocks()
        path = "transformer_blocks.13.attn1.weight"
        self.assertIsNotNone(runtime.transformer.get_parameter(path))

        with patch.object(torch, "compile", side_effect=lambda fn, **_kwargs: fn) as compile_mock:
            self.assertTrue(runtime.configure_transformer_compilation(True, "default"))

        self.assertEqual(compile_mock.call_count, 28)
        self.assertTrue(runtime.transformer_compiled)
        self.assertTrue(all("forward" in vars(block) for block in runtime.transformer.transformer_blocks))
        # Wrapping the modules would have inserted `_orig_mod` here; assigning the
        # method must not, or the LoRA table and PAG target paths stop resolving.
        self.assertIsNotNone(runtime.transformer.get_parameter(path))
        self.assertIsInstance(runtime.transformer.transformer_blocks[13].attn1, torch.nn.Linear)

        self.assertFalse(runtime.configure_transformer_compilation(False))
        self.assertFalse(any("forward" in vars(block) for block in runtime.transformer.transformer_blocks))
        self.assertTrue(torch.is_tensor(runtime.transformer.transformer_blocks[0](torch.zeros(1, 2))))

    def test_compilation_is_idempotent_and_requires_the_full_block_stack(self):
        runtime = self._runtime_with_blocks()
        with patch.object(torch, "compile", side_effect=lambda fn, **_kwargs: fn) as compile_mock:
            runtime.configure_transformer_compilation(True)
            runtime.configure_transformer_compilation(True)
        self.assertEqual(compile_mock.call_count, 28)

        short = self._runtime_with_blocks(count=27)
        with self.assertRaisesRegex(RuntimeError, "28 Cosmos transformer blocks"):
            short.configure_transformer_compilation(True)

    def test_compilation_refuses_to_coexist_with_group_offload(self):
        runtime = self._runtime_with_blocks()
        runtime._transformer_group_offload = True
        with self.assertRaisesRegex(RuntimeError, "group offload"):
            runtime.configure_transformer_compilation(True)

    def test_a_failed_compilation_leaves_no_block_half_wrapped(self):
        runtime = self._runtime_with_blocks()
        calls = []

        def flaky(fn, **_kwargs):
            calls.append(fn)
            if len(calls) == 5:
                raise RuntimeError("inductor unavailable")
            return fn

        with patch.object(torch, "compile", side_effect=flaky):
            with self.assertRaisesRegex(RuntimeError, "inductor unavailable"):
                runtime.configure_transformer_compilation(True)
        self.assertFalse(runtime.transformer_compiled)
        self.assertFalse(any("forward" in vars(block) for block in runtime.transformer.transformer_blocks))

    def test_pag_scope_steps_outside_compilation_and_restores_the_mode(self):
        runtime = self._runtime_with_blocks()
        with patch.object(torch, "compile", side_effect=lambda fn, **_kwargs: fn):
            runtime.configure_transformer_compilation(True, "max-autotune")
            with runtime._eager_transformer_blocks(True):
                self.assertFalse(runtime.transformer_compiled)
            self.assertTrue(runtime.transformer_compiled)
            self.assertEqual(runtime._transformer_compile_mode, "max-autotune")
            with runtime._eager_transformer_blocks(False):
                self.assertTrue(runtime.transformer_compiled)

    def test_a_compiled_forward_failure_falls_back_to_eager_and_records_it(self):
        runtime = self._runtime_with_blocks()
        with patch.object(torch, "compile", side_effect=lambda fn, **_kwargs: fn):
            runtime.configure_transformer_compilation(True, "default")
        runtime.attention_backend = "native"
        runtime.dtype = torch.float32
        blocks = runtime.transformer.transformer_blocks
        runtime.transformer = Mock(side_effect=[RuntimeError("guard failure"), (torch.ones(1),)])
        runtime.transformer.transformer_blocks = blocks

        prediction = runtime._transformer_prediction(hidden_states=torch.zeros(1))

        self.assertTrue(torch.equal(prediction, torch.ones(1)))
        self.assertFalse(runtime.transformer_compiled)
        self.assertFalse(any("forward" in vars(block) for block in blocks))
        self.assertEqual(
            runtime.last_generation_metrics["compile_fallback"],
            {"from": "inductor:default", "to": "eager", "reason": "RuntimeError"},
        )

    def test_a_compiled_forward_oom_is_not_treated_as_a_compilation_fault(self):
        runtime = self._runtime_with_blocks()
        with patch.object(torch, "compile", side_effect=lambda fn, **_kwargs: fn):
            runtime.configure_transformer_compilation(True, "default")
        runtime.attention_backend = "native"
        runtime.dtype = torch.float32
        runtime.transformer = Mock(side_effect=torch.cuda.OutOfMemoryError("out of memory"))

        with self.assertRaises(torch.cuda.OutOfMemoryError):
            runtime._transformer_prediction(hidden_states=torch.zeros(1))
        # Memory pressure has its own staged fallback; disabling compilation here
        # would silently spend the retry budget on the wrong remedy.
        self.assertTrue(runtime.transformer_compiled)
        self.assertNotIn("compile_fallback", runtime.last_generation_metrics)

    def test_sage_is_declined_wherever_its_constraints_do_not_hold(self):
        dispatch = _require_diffusers("models.attention_dispatch", "dispatch_attention_fn")
        processor = AnimaCosmosAttnProcessor(dispatch, sage_attention=lambda *a, **k: None)
        cuda_query = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16, size=lambda _dim: 128)

        self.assertTrue(processor._use_sage(cuda_query, True, None))
        # Cross-attention is 5888x226; the quantisation win is noise there.
        self.assertFalse(processor._use_sage(cuda_query, False, None))
        self.assertFalse(processor._use_sage(cuda_query, True, torch.zeros(1)))
        self.assertFalse(processor._use_sage(SimpleNamespace(is_cuda=False, dtype=torch.bfloat16, size=lambda _d: 128), True, None))
        self.assertFalse(processor._use_sage(SimpleNamespace(is_cuda=True, dtype=torch.bfloat16, size=lambda _d: 32), True, None))
        self.assertFalse(processor._use_sage(SimpleNamespace(is_cuda=True, dtype=torch.int8, size=lambda _d: 128), True, None))
        self.assertFalse(
            AnimaCosmosAttnProcessor(dispatch)._use_sage(cuda_query, True, None)
        )

    def test_a_cpu_forward_never_reaches_sage_and_still_matches_the_stock_processor(self):
        Attention = _require_diffusers("models.attention_processor", "Attention")
        stock = _require_diffusers("models.transformers.transformer_cosmos", "CosmosAttnProcessor2_0")
        dispatch = _require_diffusers("models.attention_dispatch", "dispatch_attention_fn")
        sage = Mock(side_effect=AssertionError("Sage must not run on a CPU tensor"))
        torch.manual_seed(19)
        attention = Attention(
            query_dim=16 * 128, cross_attention_dim=None, heads=16, dim_head=128,
            qk_norm="rms_norm", elementwise_affine=True, out_bias=False, processor=stock(),
        ).eval().requires_grad_(False)
        hidden = torch.randn(1, 21, 16 * 128)
        with torch.inference_mode():
            expected = attention(hidden)
            attention.set_processor(AnimaCosmosAttnProcessor(dispatch, sage))
            fused = attention(hidden)
        sage.assert_not_called()
        torch.testing.assert_close(fused, expected, rtol=2e-5, atol=2e-5)

    def test_sage_can_be_switched_on_and_off_over_an_already_fused_transformer(self):
        """Regression: the strict installer failed every job after the first load.

        `configure_anima_acceleration` calls `configure_sage_attention(False)` on
        every load. By then the modules carry `AnimaCosmosAttnProcessor`, so an
        installer that only accepted `CosmosAttnProcessor2_0` found nothing and
        raised - which a real generation caught and the unit tests did not,
        because they all started from a stock transformer.
        """
        stock = _require_diffusers("models.transformers.transformer_cosmos", "CosmosAttnProcessor2_0")
        modules = [SimpleNamespace(processor=stock()) for _ in range(3)]
        for module in modules:
            module.set_processor = (lambda target: lambda value: setattr(target, "processor", value))(module)
        transformer = SimpleNamespace(modules=lambda: iter(modules))

        install_anima_cosmos_attention_processor(transformer)
        self.assertEqual(install_anima_cosmos_attention_processor(transformer), 3)

        runtime = AnimaRuntime.__new__(AnimaRuntime)
        runtime._closed = False
        runtime._poisoned = False
        runtime._sage_attention_enabled = False
        runtime.transformer = transformer

        # Off when already off is a no-op, not a re-sweep.
        self.assertFalse(runtime.configure_sage_attention(False))
        with patch("backend.anima_pipeline.sage_attention_callable", return_value=lambda *a, **k: None):
            self.assertTrue(runtime.configure_sage_attention(True))
            self.assertTrue(all(m.processor._sage_attention is not None for m in modules))
            self.assertFalse(runtime.configure_sage_attention(False))
        self.assertTrue(all(m.processor._sage_attention is None for m in modules))
        self.assertTrue(all(isinstance(m.processor, AnimaCosmosAttnProcessor) for m in modules))

    def test_inductor_template_reader_is_repaired_for_a_non_utf8_locale(self):
        """PyTorch reads its own Inductor templates with the locale encoding.

        `load_template` calls `open()` with no `encoding`, so on this cp936
        machine importing `torch._inductor` raised UnicodeDecodeError before any
        compilation began — `torch.compile` was unusable, and the acceleration
        record only said `UnicodeDecodeError` with no clue where from.
        """
        from backend.anima_pipeline import repair_inductor_template_encoding

        try:
            from torch._inductor import utils as inductor_utils
        except ImportError:  # pragma: no cover - environment without Inductor
            self.skipTest("torch._inductor is unavailable")

        def locale_reader(name, template_dir):
            with open(Path(template_dir) / f"{name}.py.jinja") as handle:
                return handle.read()

        original = inductor_utils.load_template
        try:
            # Any earlier test that compiled has already applied the repair, so
            # start from a known-unpatched reader rather than from whatever ran.
            inductor_utils.load_template = locale_reader
            self.assertTrue(repair_inductor_template_encoding())
            patched = inductor_utils.load_template
            self.assertIsNot(patched, locale_reader)
            # Idempotent: a second call must not stack another wrapper.
            self.assertFalse(repair_inductor_template_encoding())
            self.assertIs(inductor_utils.load_template, patched)

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                # 0x94 is the byte that made cp936 fail on the real template.
                (root / "sample.py.jinja").write_bytes("kernel “quoted”".encode("utf-8"))
                self.assertEqual(patched("sample", root), "kernel “quoted”")
                # `load_flex_template` calls the reader by keyword, so the second
                # parameter has to keep its name or the import raises TypeError.
                self.assertEqual(patched("sample", template_dir=root), "kernel “quoted”")
        finally:
            inductor_utils.load_template = original

    def test_the_pinned_host_mirror_is_allocated_once_and_reused_by_every_park(self):
        """Regression: pinning was claimed once and invalidated on the first park.

        `_pin_transformer` pinned the parameters in place, but `.to("cuda")`
        replaced every `parameter.data` and the later `.to("cpu")` allocated fresh
        pageable storage — freeing the pinned pages while the flag still claimed
        they existed. Every stage-in after the first then copied from pageable
        memory on a `non_blocking` path that cannot overlap.
        """
        runtime = self._runtime_with_blocks(count=2)
        runtime.dtype = torch.float32
        runtime.keep_transformer_resident = False
        runtime._pin_transformer()

        mirror = runtime._transformer_host_buffers
        self.assertTrue(runtime._transformer_pinned)
        self.assertEqual(sorted(mirror), sorted(name for name, _p in runtime.transformer.named_parameters()))
        # The parameters share the mirror's storage while parked, so nothing is
        # copied twice. `Tensor.data` returns a fresh object each access, which is
        # why this compares storage rather than object identity.
        for name, parameter in runtime.transformer.named_parameters():
            self.assertEqual(parameter.data.data_ptr(), mirror[name].data_ptr())

        original = {name: buffer.clone() for name, buffer in mirror.items()}
        # Simulate a stage-in that replaces `.data`, then a weight change on the
        # device — a fused LoRA — and confirm parking copies it back into the
        # same buffers rather than allocating new ones.
        for _name, parameter in runtime.transformer.named_parameters():
            parameter.data = parameter.data.clone() + 1.0
        runtime._park_transformer_on_cpu()

        self.assertIs(runtime._transformer_host_buffers, mirror)
        for name, parameter in runtime.transformer.named_parameters():
            self.assertEqual(parameter.data.data_ptr(), mirror[name].data_ptr())
            torch.testing.assert_close(parameter.data, original[name] + 1.0)
        self.assertFalse(runtime.transformer_resident)

        # Parking an already-parked Transformer copies nothing: the storage is
        # unchanged and the values are untouched.
        pointers = {name: buffer.data_ptr() for name, buffer in mirror.items()}
        runtime._park_transformer_on_cpu()
        for name, parameter in runtime.transformer.named_parameters():
            self.assertEqual(parameter.data.data_ptr(), pointers[name])
            torch.testing.assert_close(parameter.data, original[name] + 1.0)

        # A second pin is a no-op: one host allocation for the runtime's life.
        runtime._pin_transformer()
        self.assertIs(runtime._transformer_host_buffers, mirror)

    def test_a_mirror_that_stops_describing_the_weights_is_rebuilt_not_copied_into(self):
        runtime = self._runtime_with_blocks(count=2)
        runtime.dtype = torch.float32
        runtime._pin_transformer()
        for _name, parameter in runtime.transformer.named_parameters():
            parameter.data = parameter.data.clone().to(torch.float16)
        runtime._park_transformer_on_cpu()
        self.assertIsNone(runtime._transformer_host_buffers)
        self.assertFalse(runtime._transformer_pinned)

    def test_group_offload_releases_the_mirror_it_can_no_longer_describe(self):
        runtime = self._runtime_with_blocks()
        runtime.dtype = torch.float32
        runtime._pin_transformer()
        self.assertIsNotNone(runtime._transformer_host_buffers)
        runtime.transformer.enable_group_offload = lambda **_kwargs: None
        runtime.enable_transformer_group_offload(1)
        self.assertIsNone(runtime._transformer_host_buffers)
        self.assertFalse(runtime._transformer_pinned)

    def test_closing_releases_the_unswappable_pinned_pages(self):
        runtime = self._runtime_with_blocks(count=2)
        runtime.dtype = torch.float32
        runtime.text_encoder = runtime.llm_adapter = runtime.vae = None
        runtime.qwen_tokenizer = runtime.t5_tokenizer = None
        runtime.components = {}
        runtime._pin_transformer()
        runtime.close()
        self.assertIsNone(runtime._transformer_host_buffers)
        self.assertFalse(runtime._transformer_pinned)

    def test_sage_callable_is_reported_honestly(self):
        from backend.anima_pipeline import sage_attention_callable

        resolved = sage_attention_callable()
        if resolved is None:
            self.skipTest("sageattention is not installed in this environment")
        self.assertTrue(callable(resolved))


class AnimaRuntimeTeardownTests(unittest.TestCase):
    """Closing a runtime has to release its tensors without paying to move them anywhere."""

    def module(self):
        block = torch.nn.Linear(4, 4)
        block.register_buffer("scratch", torch.ones(4))
        wrapper = torch.nn.Sequential(block)
        wrapper.register_buffer("outer", torch.ones(2))
        return wrapper, block

    def test_storage_is_dropped_without_a_transfer(self):
        wrapper, block = self.module()
        moves = []
        original_to = torch.Tensor.to

        def record(self, *args, **kwargs):
            moves.append((tuple(args), tuple(sorted(kwargs))))
            return original_to(self, *args, **kwargs)

        with patch.object(torch.Tensor, "to", record):
            _discard_module_storage(wrapper)
        # Closing used to run `module.to("cpu")` over the whole runtime — for the transformer alone a
        # multi-gigabyte device-to-host copy of memory about to be freed.
        self.assertEqual(moves, [])
        self.assertEqual(block.weight.numel(), 0)
        self.assertEqual(block.scratch.numel(), 0)
        self.assertEqual(wrapper.outer.numel(), 0)

    def test_a_reference_held_elsewhere_cannot_keep_the_old_storage(self):
        # The property `.to()` provided and this has to keep: it reassigns `parameter.data` in place,
        # so a stray holder — a compiled block's closure, a hook, an attention processor — follows the
        # release rather than pinning the weights it captured.
        wrapper, block = self.module()
        captured = block.weight
        _discard_module_storage(wrapper)
        self.assertIs(captured, block.weight)
        self.assertEqual(captured.numel(), 0)

    def test_a_module_with_no_parameters_is_left_alone_rather_than_failing(self):
        _discard_module_storage(torch.nn.Identity())


def _require_diffusers(module_path: str, name: str):
    try:
        module = __import__(f"diffusers.{module_path}", fromlist=[name])
    except ImportError as error:  # pragma: no cover - environment without diffusers
        raise unittest.SkipTest(f"diffusers.{module_path} is unavailable: {error}") from error
    return getattr(module, name)


if __name__ == "__main__":
    unittest.main()
