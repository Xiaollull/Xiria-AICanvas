import json
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flux_pipeline
from flux_pipeline import (
    FLUX_ATTENTION_HEAD_DIM,
    FLUX_CLIP_CONFIG,
    FLUX_DISTILLED_T5_SEQUENCE_LENGTH,
    FLUX_GUIDED_T5_SEQUENCE_LENGTH,
    FLUX_LATENT_CHANNELS,
    FLUX_PIXEL_ALIGNMENT,
    FLUX_VAE_CONFIG,
    _classify_text_encoder,
    _load_flux_transformer,
    _runtime_dependencies,
    checkpoint_is_mixed_precision,
    checkpoint_is_scaled_fp8,
    convert_flux_lora_state_dict,
    convert_flux_transformer_state_dict,
    dequantize_scaled_fp8,
    flux_image_ids,
    flux_t5_sequence_length,
    fuse_flux_lora_state_dict,
    infer_flux_transformer_config,
    normalize_flux_checkpoint_keys,
    pack_flux_latents,
    resolve_quantized_state_dict,
    unmatched_lora_targets,
    unpack_flux_latents,
)


HIDDEN = FLUX_ATTENTION_HEAD_DIM  # one attention head keeps the fixture small but architecturally real
JOINT = 32
POOLED = 16
IN_CHANNELS = FLUX_LATENT_CHANNELS * 4
OUT_CHANNELS = FLUX_LATENT_CHANNELS * 4
MLP_HIDDEN = HIDDEN * 4


def _linear(out_features, in_features, prefix, state):
    state[f"{prefix}.weight"] = torch.randn(out_features, in_features) * 0.02
    state[f"{prefix}.bias"] = torch.zeros(out_features)


def comfy_flux_checkpoint(double_blocks=1, single_blocks=1, guidance=True):
    """A tiny checkpoint in ComfyUI's FLUX.1 layout.

    Every key and shape here is the one ``convert_flux_transformer_checkpoint_to_diffusers``
    consumes, so a strict load succeeding is real evidence that the conversion, the inferred
    configuration and the model geometry agree — without a 24 GB download.
    """
    state = {}
    _linear(HIDDEN, IN_CHANNELS, "img_in", state)
    _linear(HIDDEN, JOINT, "txt_in", state)
    _linear(HIDDEN, 256, "time_in.in_layer", state)
    _linear(HIDDEN, HIDDEN, "time_in.out_layer", state)
    _linear(HIDDEN, POOLED, "vector_in.in_layer", state)
    _linear(HIDDEN, HIDDEN, "vector_in.out_layer", state)
    if guidance:
        _linear(HIDDEN, 256, "guidance_in.in_layer", state)
        _linear(HIDDEN, HIDDEN, "guidance_in.out_layer", state)
    for index in range(double_blocks):
        prefix = f"double_blocks.{index}"
        for stream in ("img", "txt"):
            _linear(6 * HIDDEN, HIDDEN, f"{prefix}.{stream}_mod.lin", state)
            _linear(3 * HIDDEN, HIDDEN, f"{prefix}.{stream}_attn.qkv", state)
            _linear(HIDDEN, HIDDEN, f"{prefix}.{stream}_attn.proj", state)
            _linear(MLP_HIDDEN, HIDDEN, f"{prefix}.{stream}_mlp.0", state)
            _linear(HIDDEN, MLP_HIDDEN, f"{prefix}.{stream}_mlp.2", state)
            state[f"{prefix}.{stream}_attn.norm.query_norm.scale"] = torch.ones(FLUX_ATTENTION_HEAD_DIM)
            state[f"{prefix}.{stream}_attn.norm.key_norm.scale"] = torch.ones(FLUX_ATTENTION_HEAD_DIM)
    for index in range(single_blocks):
        prefix = f"single_blocks.{index}"
        _linear(3 * HIDDEN + MLP_HIDDEN, HIDDEN, f"{prefix}.linear1", state)
        _linear(HIDDEN, HIDDEN + MLP_HIDDEN, f"{prefix}.linear2", state)
        _linear(3 * HIDDEN, HIDDEN, f"{prefix}.modulation.lin", state)
        state[f"{prefix}.norm.query_norm.scale"] = torch.ones(FLUX_ATTENTION_HEAD_DIM)
        state[f"{prefix}.norm.key_norm.scale"] = torch.ones(FLUX_ATTENTION_HEAD_DIM)
    _linear(OUT_CHANNELS, HIDDEN, "final_layer.linear", state)
    _linear(2 * HIDDEN, HIDDEN, "final_layer.adaLN_modulation.1", state)
    return state


class CheckpointNormalisationTests(unittest.TestCase):
    def test_wrapper_prefixes_are_stripped(self):
        for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
            with self.subTest(prefix=prefix):
                normalized = normalize_flux_checkpoint_keys({f"{prefix}img_in.weight": torch.zeros(1)})
                self.assertEqual(list(normalized), ["img_in.weight"])

    def test_a_collision_between_prefixed_and_bare_keys_is_refused(self):
        with self.assertRaises(ValueError):
            normalize_flux_checkpoint_keys({
                "img_in.weight": torch.zeros(1),
                "model.diffusion_model.img_in.weight": torch.zeros(1),
            })


class ScaledFp8Tests(unittest.TestCase):
    def test_a_scaled_checkpoint_is_recognised(self):
        self.assertTrue(checkpoint_is_scaled_fp8({"scaled_fp8": torch.zeros(())}))
        self.assertTrue(checkpoint_is_scaled_fp8({"img_in.scale_weight": torch.ones(())}))
        self.assertFalse(checkpoint_is_scaled_fp8({"img_in.weight": torch.zeros(1)}))

    def test_the_scale_is_folded_into_the_weight(self):
        state = {
            "scaled_fp8": torch.zeros(()),
            "img_in.weight": torch.tensor([[2.0, 4.0]]),
            "img_in.scale_weight": torch.tensor(0.5),
            "img_in.bias": torch.tensor([1.0]),
        }
        resolved = dequantize_scaled_fp8(state, torch.float32)
        self.assertEqual(sorted(resolved), ["img_in.bias", "img_in.weight"])
        self.assertTrue(torch.equal(resolved["img_in.weight"], torch.tensor([[1.0, 2.0]])))

    def test_an_activation_scale_is_dropped_rather_than_carried_into_a_strict_load(self):
        state = {
            "scaled_fp8": torch.zeros(()),
            "img_in.weight": torch.tensor([[1.0]]),
            "img_in.scale_weight": torch.tensor(1.0),
            "img_in.scale_input": torch.tensor(1.0),
        }
        self.assertEqual(sorted(dequantize_scaled_fp8(state, torch.float32)), ["img_in.weight"])

    def test_a_scale_without_a_weight_is_refused(self):
        with self.assertRaises(ValueError):
            dequantize_scaled_fp8({"scaled_fp8": torch.zeros(()), "ghost.scale_weight": torch.tensor(1.0)}, torch.float32)

    def test_the_other_spelling_of_the_same_scale_is_recognised(self):
        # Comfy-Org's Krea 2 fp8 checkpoint writes `<layer>.weight_scale` where ComfyUI writes
        # `<layer>.scale_weight`, and carries no `scaled_fp8` marker. Read as unscaled, all 256 of
        # its scales fell out as unclaimed tensors and the transformer ran at the wrong scale —
        # which loads and samples perfectly happily, and produces a wrong image.
        self.assertTrue(checkpoint_is_scaled_fp8({"img_in.weight_scale": torch.ones(())}))

    def test_the_other_spelling_is_folded_in_the_same_way(self):
        state = {
            "img_in.weight": torch.tensor([[2.0, 4.0]]),
            "img_in.weight_scale": torch.tensor(0.5),
            "img_in.bias": torch.tensor([1.0]),
        }
        resolved = dequantize_scaled_fp8(state, torch.float32)
        self.assertEqual(sorted(resolved), ["img_in.bias", "img_in.weight"])
        self.assertTrue(torch.equal(resolved["img_in.weight"], torch.tensor([[1.0, 2.0]])))

    def test_a_folded_weight_is_narrowed_as_it_is_produced(self):
        # Holding every dequantised layer in float32 until a final pass costs four bytes per
        # parameter for the whole model at once. Krea 2's 12.2 GB fp8 transformer needs about 49 GB
        # that way, and the load was OOM-killed at 64.9 GB RSS on a 62 GB machine.
        state = {
            "img_in.weight": torch.tensor([[2.0, 4.0]]),
            "img_in.weight_scale": torch.tensor(0.5),
        }
        resolved = dequantize_scaled_fp8(state, torch.bfloat16)
        self.assertEqual(resolved["img_in.weight"].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(resolved["img_in.weight"], torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)))

    def test_a_plain_norm_scale_is_left_alone(self):
        # This format also carries ordinary bf16 norm parameters named `.scale`, such as
        # `last.norm.scale`. Treating those as quantisation state would strip real weights out of
        # the checkpoint and fail the strict load.
        state = {
            "img_in.weight": torch.tensor([[2.0]]),
            "img_in.weight_scale": torch.tensor(0.5),
            "last.norm.scale": torch.tensor([3.0]),
        }
        resolved = dequantize_scaled_fp8(state, torch.float32)
        self.assertEqual(sorted(resolved), ["img_in.weight", "last.norm.scale"])
        self.assertTrue(torch.equal(resolved["last.norm.scale"], torch.tensor([3.0])))


def comfy_quant_marker(quant_format, **extra):
    """The `comfy_quant` tensor ComfyUI writes: the layer's JSON configuration, stored as bytes."""
    payload = json.dumps({"format": quant_format, **extra}).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


class MixedPrecisionTests(unittest.TestCase):
    """ComfyUI's newer per-layer quantisation: `comfy_quant` names the format, scales sit beside it."""

    def test_a_mixed_precision_checkpoint_is_recognised_and_the_scale_is_folded_in(self):
        state = {
            "img_in.weight": torch.tensor([[2.0, 4.0]]),
            "img_in.weight_scale": torch.tensor([0.5]),
            "img_in.comfy_quant": comfy_quant_marker("float8_e4m3fn"),
            "img_in.bias": torch.tensor([1.0]),
        }
        self.assertTrue(checkpoint_is_mixed_precision(state))
        self.assertFalse(checkpoint_is_scaled_fp8(state))
        resolved = resolve_quantized_state_dict(state, torch.float32)
        self.assertEqual(sorted(resolved), ["img_in.bias", "img_in.weight"])
        self.assertTrue(torch.equal(resolved["img_in.weight"], torch.tensor([[1.0, 2.0]])))

    def test_an_unquantised_layer_in_a_mixed_checkpoint_is_left_alone(self):
        # "Mixed" is the point: only some layers carry a marker, and a layer without one is already
        # full precision. Scaling it too would silently change the model.
        state = {
            "img_in.weight": torch.tensor([[2.0]]),
            "img_in.weight_scale": torch.tensor(0.5),
            "img_in.comfy_quant": comfy_quant_marker("float8_e5m2"),
            "final_layer.linear.weight": torch.tensor([[3.0]]),
        }
        resolved = resolve_quantized_state_dict(state, torch.float32)
        self.assertTrue(torch.equal(resolved["img_in.weight"], torch.tensor([[1.0]])))
        self.assertTrue(torch.equal(resolved["final_layer.linear.weight"], torch.tensor([[3.0]])))

    def test_an_activation_scale_is_dropped_rather_than_carried_into_a_strict_load(self):
        state = {
            "img_in.weight": torch.tensor([[1.0]]),
            "img_in.weight_scale": torch.tensor(1.0),
            "img_in.input_scale": torch.tensor(1.0),
            "img_in.comfy_quant": comfy_quant_marker("float8_e4m3fn"),
        }
        self.assertEqual(sorted(resolve_quantized_state_dict(state, torch.float32)), ["img_in.weight"])

    def test_a_packed_format_is_refused_by_name_rather_than_loaded_wrongly(self):
        # nvfp4 stores two weights per byte, so its tensor is not the layer's matrix at all. The
        # failure has to say so: a shape mismatch thousands of tensors later does not.
        state = {
            "img_in.weight": torch.zeros(4, 2, dtype=torch.uint8),
            "img_in.weight_scale": torch.zeros(4),
            "img_in.weight_scale_2": torch.tensor(1.0),
            "img_in.comfy_quant": comfy_quant_marker("nvfp4"),
        }
        with self.assertRaises(ValueError) as error:
            resolve_quantized_state_dict(state, torch.float32, "FLUX.2 diffusion model")
        self.assertIn("nvfp4", str(error.exception))
        self.assertIn("FLUX.2 diffusion model", str(error.exception))

    def test_a_block_scale_is_refused_rather_than_broadcast(self):
        state = {
            "img_in.weight": torch.zeros(2, 64),
            "img_in.weight_scale": torch.zeros(2),
            "img_in.comfy_quant": comfy_quant_marker("float8_e4m3fn"),
        }
        with self.assertRaises(ValueError) as error:
            resolve_quantized_state_dict(state, torch.float32)
        self.assertIn("block scale", str(error.exception))

    def test_a_marker_without_a_weight_or_a_scale_is_refused(self):
        with self.assertRaises(ValueError):
            resolve_quantized_state_dict({"ghost.comfy_quant": comfy_quant_marker("float8_e4m3fn")}, torch.float32)
        with self.assertRaises(ValueError):
            resolve_quantized_state_dict({
                "img_in.weight": torch.zeros(1, 1),
                "img_in.comfy_quant": comfy_quant_marker("float8_e4m3fn"),
            }, torch.float32)

    def test_an_unreadable_marker_is_refused(self):
        with self.assertRaises(ValueError):
            resolve_quantized_state_dict({
                "img_in.weight": torch.zeros(1, 1),
                "img_in.comfy_quant": torch.tensor([123, 45], dtype=torch.uint8),
            }, torch.float32)

    def test_an_unquantised_checkpoint_passes_through_untouched(self):
        state = {"img_in.weight": torch.tensor([[7.0]])}
        resolved = resolve_quantized_state_dict(state, torch.float32)
        self.assertTrue(torch.equal(resolved["img_in.weight"], torch.tensor([[7.0]])))

    def test_a_mixed_precision_transformer_loads(self):
        state = comfy_flux_checkpoint()
        quantised = {}
        for key, value in state.items():
            quantised[key] = value
            if key.endswith(".weight") and value.dim() == 2:
                layer = key[: -len(".weight")]
                quantised[key] = value * 4.0
                quantised[f"{layer}.weight_scale"] = torch.tensor(0.25)
                quantised[f"{layer}.comfy_quant"] = comfy_quant_marker("float8_e4m3fn")
        transformer, _config, _report = _load_flux_transformer(
            Path("unused.safetensors"), torch.float32, _runtime_dependencies(), state_dict=quantised
        )
        self.assertTrue(torch.allclose(
            transformer.x_embedder.weight.detach(), state["img_in.weight"], atol=1e-6
        ))


class TransformerConfigTests(unittest.TestCase):
    def test_the_configuration_is_read_out_of_the_checkpoint(self):
        config = infer_flux_transformer_config(comfy_flux_checkpoint(double_blocks=3, single_blocks=5))
        self.assertEqual(config["num_layers"], 3)
        self.assertEqual(config["num_single_layers"], 5)
        self.assertEqual(config["in_channels"], IN_CHANNELS)
        self.assertEqual(config["out_channels"], OUT_CHANNELS)
        self.assertEqual(config["joint_attention_dim"], JOINT)
        self.assertEqual(config["pooled_projection_dim"], POOLED)
        self.assertEqual(config["attention_head_dim"], FLUX_ATTENTION_HEAD_DIM)
        self.assertEqual(config["num_attention_heads"], HIDDEN // FLUX_ATTENTION_HEAD_DIM)
        self.assertTrue(config["guidance_embeds"])

    def test_a_distilled_checkpoint_is_told_apart_by_its_missing_guidance_embedding(self):
        config = infer_flux_transformer_config(comfy_flux_checkpoint(guidance=False))
        self.assertFalse(config["guidance_embeds"])
        self.assertEqual(flux_t5_sequence_length(False), FLUX_DISTILLED_T5_SEQUENCE_LENGTH)
        self.assertEqual(flux_t5_sequence_length(True), FLUX_GUIDED_T5_SEQUENCE_LENGTH)

    def test_a_file_that_is_not_a_flux_model_is_refused_by_name_rather_than_by_shape_error(self):
        with self.assertRaises(ValueError) as error:
            infer_flux_transformer_config({"conditioner.embedders.0.weight": torch.zeros(1)})
        self.assertIn("not a FLUX.1 diffusion model", str(error.exception))

    def test_a_gap_in_the_block_numbering_is_refused(self):
        state = comfy_flux_checkpoint(double_blocks=2)
        for key in [key for key in state if key.startswith("double_blocks.0.")]:
            del state[key]
        with self.assertRaises(ValueError):
            infer_flux_transformer_config(state)


class LatentPackingTests(unittest.TestCase):
    def test_packing_round_trips(self):
        latents = torch.arange(2 * FLUX_LATENT_CHANNELS * 4 * 6, dtype=torch.float32).view(2, FLUX_LATENT_CHANNELS, 4, 6)
        packed = pack_flux_latents(latents)
        self.assertEqual(tuple(packed.shape), (2, (4 // 2) * (6 // 2), FLUX_LATENT_CHANNELS * 4))
        self.assertTrue(torch.equal(unpack_flux_latents(packed, 4, 6), latents))

    def test_an_odd_latent_grid_is_refused(self):
        with self.assertRaises(ValueError):
            pack_flux_latents(torch.zeros(1, FLUX_LATENT_CHANNELS, 3, 4))
        with self.assertRaises(ValueError):
            unpack_flux_latents(torch.zeros(1, 6, 64), 4, 4)

    def test_image_ids_carry_the_row_and_column_of_every_token(self):
        ids = flux_image_ids(4, 6, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(ids.shape), (6, 3))
        self.assertEqual(ids[..., 0].abs().sum().item(), 0.0)
        self.assertEqual([int(value) for value in ids[:, 1]], [0, 0, 0, 1, 1, 1])
        self.assertEqual([int(value) for value in ids[:, 2]], [0, 1, 2, 0, 1, 2])

    def test_the_canvas_alignment_is_the_vae_stride_times_the_latent_patch(self):
        self.assertEqual(FLUX_PIXEL_ALIGNMENT, 16)


class TextEncoderClassificationTests(unittest.TestCase):
    def test_clip_and_t5_are_told_apart_by_their_own_tensors(self):
        self.assertEqual(_classify_text_encoder(["text_model.encoder.layers.0.mlp.fc1.weight"]), "clip_l")
        self.assertEqual(_classify_text_encoder(["encoder.block.0.layer.0.SelfAttention.q.weight"]), "t5xxl")
        self.assertEqual(_classify_text_encoder(["shared.weight"]), "t5xxl")
        self.assertEqual(_classify_text_encoder(["diffusion_model.img_in.weight"]), "unknown")

    def test_the_clip_configuration_is_the_published_text_tower(self):
        self.assertEqual(FLUX_CLIP_CONFIG["hidden_size"], 768)
        self.assertEqual(FLUX_CLIP_CONFIG["num_hidden_layers"], 12)
        self.assertEqual(FLUX_CLIP_CONFIG["max_position_embeddings"], 77)
        self.assertEqual(FLUX_CLIP_CONFIG["hidden_act"], "quick_gelu")

    def test_the_vae_is_the_sixteen_channel_flux_autoencoder(self):
        self.assertEqual(FLUX_VAE_CONFIG["latent_channels"], FLUX_LATENT_CHANNELS)
        self.assertFalse(FLUX_VAE_CONFIG["use_quant_conv"])
        self.assertFalse(FLUX_VAE_CONFIG["use_post_quant_conv"])
        self.assertAlmostEqual(FLUX_VAE_CONFIG["scaling_factor"], 0.3611)
        self.assertAlmostEqual(FLUX_VAE_CONFIG["shift_factor"], 0.1159)


class LoraFusionTests(unittest.TestCase):
    def setUp(self):
        self.base = {"transformer_blocks.0.attn.to_q.weight": torch.zeros(4, 4)}

    def test_a_diffusers_layout_lora_is_fused_as_b_times_a(self):
        down = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        up = torch.tensor([[2.0], [0.0], [0.0], [0.0]])
        lora = {
            "transformer_blocks.0.attn.to_q.lora_A.weight": down,
            "transformer_blocks.0.attn.to_q.lora_B.weight": up,
        }
        patched = fuse_flux_lora_state_dict(self.base, lora, 0.5, "test")
        self.assertEqual(patched, 1)
        expected = torch.zeros(4, 4)
        expected[0, 0] = 1.0
        self.assertTrue(torch.equal(self.base["transformer_blocks.0.attn.to_q.weight"], expected))

    def test_a_shape_mismatch_is_refused_rather_than_broadcast(self):
        lora = {
            "transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(1, 8),
            "transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(4, 1),
        }
        with self.assertRaises(ValueError):
            fuse_flux_lora_state_dict(self.base, lora, 1.0, "test")

    def test_a_lora_for_a_different_base_is_refused(self):
        lora = {
            "somewhere.else.lora_A.weight": torch.zeros(1, 4),
            "somewhere.else.lora_B.weight": torch.zeros(4, 1),
        }
        with self.assertRaises(ValueError) as error:
            fuse_flux_lora_state_dict(self.base, lora, 1.0, "test")
        self.assertIn("different base", str(error.exception))

    def test_unmatched_targets_are_reported_before_they_are_skipped(self):
        lora = {
            "transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(1, 4),
            "transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(4, 1),
            "text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(1, 4),
            "text_model.encoder.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(4, 1),
        }
        self.assertEqual(
            unmatched_lora_targets(lora, self.base),
            ["text_model.encoder.layers.0.self_attn.q_proj"],
        )

    def test_an_unpaired_lora_tensor_is_refused(self):
        with self.assertRaises(ValueError):
            fuse_flux_lora_state_dict(
                self.base, {"transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(1, 4)}, 1.0, "test"
            )

    def test_a_diffusers_alpha_is_folded_in_so_every_layout_arrives_pre_scaled(self):
        deps = _runtime_dependencies()
        lora = {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
            "transformer.transformer_blocks.0.attn.to_q.alpha": torch.tensor(1.0),
        }
        converted = convert_flux_lora_state_dict(lora, deps, "test")
        # rank 2 with alpha 1 scales the down matrix by 0.5, and the `transformer.` prefix is gone.
        self.assertEqual(
            sorted(converted),
            ["transformer_blocks.0.attn.to_q.lora_A.weight", "transformer_blocks.0.attn.to_q.lora_B.weight"],
        )
        self.assertTrue(torch.equal(converted["transformer_blocks.0.attn.to_q.lora_A.weight"], torch.full((2, 4), 0.5)))

    def test_an_unrecognised_lora_layout_is_refused(self):
        with self.assertRaises(ValueError):
            convert_flux_lora_state_dict({"mystery.weight": torch.zeros(1)}, _runtime_dependencies(), "test")


class TransformerLoadTests(unittest.TestCase):
    """Load a tiny ComfyUI-layout checkpoint end to end and run one forward pass on the CPU."""

    def setUp(self):
        self.deps = _runtime_dependencies()
        self.state = comfy_flux_checkpoint(double_blocks=1, single_blocks=1)

    def _load(self, state=None, loras=()):
        return _load_flux_transformer(
            Path("unused.safetensors"), torch.float32, self.deps, loras=loras, state_dict=state or self.state
        )

    def test_a_comfy_layout_checkpoint_loads_strictly(self):
        transformer, config, report = self._load()
        self.assertEqual(config["num_layers"], 1)
        self.assertEqual(report["loras"], [])
        self.assertEqual(report["unclaimed_tensors"], [])
        self.assertFalse(any(parameter.is_meta for parameter in transformer.parameters()))

    def test_the_conversion_places_every_fused_projection_in_its_own_slot(self):
        # A qkv split that lands q into to_k would still load strictly and still run, so the
        # ordering is asserted directly: each third of the fused projection is filled with a
        # distinct value and checked where it is supposed to arrive.
        state = comfy_flux_checkpoint()
        marks = torch.cat([torch.full((HIDDEN, HIDDEN), value) for value in (1.0, 2.0, 3.0)])
        state["double_blocks.0.img_attn.qkv.weight"] = marks
        single = torch.cat([
            torch.full((HIDDEN, HIDDEN), 1.0),
            torch.full((HIDDEN, HIDDEN), 2.0),
            torch.full((HIDDEN, HIDDEN), 3.0),
            torch.full((MLP_HIDDEN, HIDDEN), 4.0),
        ])
        state["single_blocks.0.linear1.weight"] = single
        config = infer_flux_transformer_config(state)
        converted, unclaimed = convert_flux_transformer_state_dict(state, config)
        self.assertEqual(unclaimed, [])
        for name, value in (("to_q", 1.0), ("to_k", 2.0), ("to_v", 3.0)):
            self.assertEqual(converted[f"transformer_blocks.0.attn.{name}.weight"].unique().tolist(), [value])
        for name, value in (("attn.to_q", 1.0), ("attn.to_k", 2.0), ("attn.to_v", 3.0), ("proj_mlp", 4.0)):
            self.assertEqual(converted[f"single_transformer_blocks.0.{name}.weight"].unique().tolist(), [value])

    def test_the_final_adaln_projection_is_reordered_from_shift_scale_to_scale_shift(self):
        state = comfy_flux_checkpoint()
        state["final_layer.adaLN_modulation.1.weight"] = torch.cat([
            torch.full((HIDDEN, HIDDEN), 1.0), torch.full((HIDDEN, HIDDEN), 2.0)
        ])
        converted, _unclaimed = convert_flux_transformer_state_dict(state, infer_flux_transformer_config(state))
        reordered = converted["norm_out.linear.weight"]
        self.assertEqual(reordered[:HIDDEN].unique().tolist(), [2.0])
        self.assertEqual(reordered[HIDDEN:].unique().tolist(), [1.0])

    def test_an_unclaimed_tensor_is_reported_rather_than_swallowed(self):
        state = comfy_flux_checkpoint()
        state["something.unexpected"] = torch.zeros(1)
        _converted, unclaimed = convert_flux_transformer_state_dict(state, infer_flux_transformer_config(state))
        self.assertEqual(unclaimed, ["something.unexpected"])

    def test_a_missing_tensor_is_named_in_the_failure(self):
        state = comfy_flux_checkpoint()
        config = infer_flux_transformer_config(state)
        del state["double_blocks.0.img_mlp.2.bias"]
        with self.assertRaises(ValueError) as error:
            convert_flux_transformer_state_dict(state, config)
        self.assertIn("double_blocks.0.img_mlp.2.bias", str(error.exception))

    def test_a_non_standard_mlp_width_is_derived_rather_than_assumed(self):
        # Diffusers' own converter hard-codes a 4.0 MLP ratio. Reading the width off the fused
        # projection is what lets a variant that departs from it load at all.
        state = comfy_flux_checkpoint()
        narrow = HIDDEN * 2
        state["single_blocks.0.linear1.weight"] = torch.zeros(3 * HIDDEN + narrow, HIDDEN)
        state["single_blocks.0.linear1.bias"] = torch.zeros(3 * HIDDEN + narrow)
        state["single_blocks.0.linear2.weight"] = torch.zeros(HIDDEN, HIDDEN + narrow)
        converted, _unclaimed = convert_flux_transformer_state_dict(state, infer_flux_transformer_config(state))
        self.assertEqual(tuple(converted["single_transformer_blocks.0.proj_mlp.weight"].shape), (narrow, HIDDEN))

    def test_the_loaded_transformer_predicts_a_velocity_for_every_packed_token(self):
        transformer, _config, _report = self._load()
        tokens, text_tokens = 6, 4
        with torch.inference_mode():
            output = transformer(
                hidden_states=torch.zeros(1, tokens, IN_CHANNELS),
                encoder_hidden_states=torch.zeros(1, text_tokens, JOINT),
                pooled_projections=torch.zeros(1, POOLED),
                timestep=torch.full((1,), 0.5),
                img_ids=flux_image_ids(4, 6, torch.device("cpu"), torch.float32),
                txt_ids=torch.zeros(text_tokens, 3),
                guidance=torch.full((1,), 3.5),
                return_dict=False,
            )[0]
        self.assertEqual(tuple(output.shape), (1, tokens, OUT_CHANNELS))
        self.assertTrue(torch.isfinite(output).all())

    def test_a_distilled_checkpoint_loads_without_a_guidance_embedder(self):
        transformer, config, _report = self._load(comfy_flux_checkpoint(guidance=False))
        self.assertFalse(config["guidance_embeds"])
        self.assertFalse(hasattr(transformer.time_text_embed, "guidance_embedder"))

    def test_a_scaled_fp8_checkpoint_is_dequantised_on_the_way_in(self):
        state = dict(self.state)
        state["scaled_fp8"] = torch.zeros(())
        state["img_in.scale_weight"] = torch.tensor(2.0)
        expected = self.state["img_in.weight"] * 2.0
        transformer, _config, _report = self._load(state)
        self.assertTrue(torch.allclose(transformer.x_embedder.weight, expected, atol=1e-6))

    def test_a_fused_lora_changes_the_weights_it_targets(self):
        transformer, _config, _report = self._load()
        before = transformer.x_embedder.weight.clone()
        lora_state = {
            "x_embedder.lora_A.weight": torch.ones(1, IN_CHANNELS),
            "x_embedder.lora_B.weight": torch.ones(HIDDEN, 1),
        }
        converted, _unclaimed = convert_flux_transformer_state_dict(
            self.state, infer_flux_transformer_config(self.state)
        )
        patched = fuse_flux_lora_state_dict(converted, lora_state, 2.0, "test")
        self.assertEqual(patched, 1)
        self.assertTrue(torch.allclose(converted["x_embedder.weight"], before + 2.0))


if __name__ == "__main__":
    unittest.main()
