import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flux2_pipeline
from flux2_pipeline import (
    FLUX2_ATTENTION_HEAD_DIM,
    FLUX2_AXES_DIMS_ROPE,
    FLUX2_LATENT_CHANNELS,
    FLUX2_VAE_CONFIG,
    FLUX2_MISTRAL_TAP_LAYERS,
    FLUX2_PIXEL_ALIGNMENT,
    FLUX2_QWEN_TAP_LAYERS,
    FLUX2_ROPE_THETA,
    FLUX2_VAE_LATENT_CHANNELS,
    _load_flux2_text_encoder,
    _load_flux2_transformer,
    _load_flux2_vae,
    _runtime_dependencies,
    classify_flux2_text_encoder,
    convert_flux2_lora_state_dict,
    convert_flux2_transformer_state_dict,
    flux2_image_ids,
    flux2_text_ids,
    infer_flux2_text_encoder_config,
    infer_flux2_transformer_config,
    load_flux2_runtime,
    pack_flux2_latents,
    patchify_flux2_latents,
    unpack_flux2_latents,
    unpatchify_flux2_latents,
)
from test_flux_pipeline import comfy_flux_checkpoint

HIDDEN = FLUX2_ATTENTION_HEAD_DIM  # one attention head keeps the fixture small but real
LLM_HIDDEN = 32
JOINT = 3 * LLM_HIDDEN  # three tapped layers concatenated on the channel axis
IN_CHANNELS = FLUX2_LATENT_CHANNELS
OUT_CHANNELS = FLUX2_LATENT_CHANNELS
MLP_RATIO = 3.0
MLP_HIDDEN = int(HIDDEN * MLP_RATIO)


def _linear(out_features, in_features, prefix, state):
    # FLUX.2 carries no biases at all: `ops_bias=False` in `comfy/ldm/flux/model.py`.
    state[f"{prefix}.weight"] = torch.randn(out_features, in_features) * 0.02


def comfy_flux2_checkpoint(double_blocks=1, single_blocks=1, guidance=True, norm_suffix="scale"):
    """A tiny checkpoint in ComfyUI's FLUX.2 layout.

    The shared modulation projections, the gated SwiGLU MLP width and the absence of biases are
    what separate this layout from FLUX.1's, and each is reproduced here so a strict load
    succeeding is evidence the converter and the inferred configuration agree.
    """
    state = {}
    _linear(HIDDEN, IN_CHANNELS, "img_in", state)
    _linear(HIDDEN, JOINT, "txt_in", state)
    _linear(HIDDEN, 256, "time_in.in_layer", state)
    _linear(HIDDEN, HIDDEN, "time_in.out_layer", state)
    if guidance:
        _linear(HIDDEN, 256, "guidance_in.in_layer", state)
        _linear(HIDDEN, HIDDEN, "guidance_in.out_layer", state)
    _linear(6 * HIDDEN, HIDDEN, "double_stream_modulation_img.lin", state)
    _linear(6 * HIDDEN, HIDDEN, "double_stream_modulation_txt.lin", state)
    _linear(3 * HIDDEN, HIDDEN, "single_stream_modulation.lin", state)
    for index in range(double_blocks):
        prefix = f"double_blocks.{index}"
        for stream in ("img", "txt"):
            _linear(3 * HIDDEN, HIDDEN, f"{prefix}.{stream}_attn.qkv", state)
            _linear(HIDDEN, HIDDEN, f"{prefix}.{stream}_attn.proj", state)
            _linear(2 * MLP_HIDDEN, HIDDEN, f"{prefix}.{stream}_mlp.0", state)
            _linear(HIDDEN, MLP_HIDDEN, f"{prefix}.{stream}_mlp.2", state)
            state[f"{prefix}.{stream}_attn.norm.query_norm.{norm_suffix}"] = torch.ones(FLUX2_ATTENTION_HEAD_DIM)
            state[f"{prefix}.{stream}_attn.norm.key_norm.{norm_suffix}"] = torch.ones(FLUX2_ATTENTION_HEAD_DIM)
    for index in range(single_blocks):
        prefix = f"single_blocks.{index}"
        _linear(3 * HIDDEN + 2 * MLP_HIDDEN, HIDDEN, f"{prefix}.linear1", state)
        _linear(HIDDEN, HIDDEN + MLP_HIDDEN, f"{prefix}.linear2", state)
        state[f"{prefix}.norm.query_norm.{norm_suffix}"] = torch.ones(FLUX2_ATTENTION_HEAD_DIM)
        state[f"{prefix}.norm.key_norm.{norm_suffix}"] = torch.ones(FLUX2_ATTENTION_HEAD_DIM)
    _linear(OUT_CHANNELS, HIDDEN, "final_layer.linear", state)
    _linear(2 * HIDDEN, HIDDEN, "final_layer.adaLN_modulation.1", state)
    return state


def mistral_text_encoder_state(layers=max(FLUX2_MISTRAL_TAP_LAYERS), hidden=LLM_HIDDEN, vocab=64, qwen=False):
    """A miniature language model in the key layout ComfyUI publishes FLUX.2 text encoders in."""
    state = {"model.embed_tokens.weight": torch.randn(vocab, hidden) * 0.02}
    for index in range(layers):
        prefix = f"model.layers.{index}"
        for name, shape in (
            ("self_attn.q_proj", (FLUX2_ATTENTION_HEAD_DIM, hidden)),
            ("self_attn.k_proj", (FLUX2_ATTENTION_HEAD_DIM, hidden)),
            ("self_attn.v_proj", (FLUX2_ATTENTION_HEAD_DIM, hidden)),
            ("self_attn.o_proj", (hidden, FLUX2_ATTENTION_HEAD_DIM)),
            ("mlp.gate_proj", (hidden * 2, hidden)),
            ("mlp.up_proj", (hidden * 2, hidden)),
            ("mlp.down_proj", (hidden, hidden * 2)),
        ):
            state[f"{prefix}.{name}.weight"] = torch.randn(*shape) * 0.02
        state[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden)
        state[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(hidden)
        if qwen:
            state[f"{prefix}.self_attn.q_norm.weight"] = torch.ones(FLUX2_ATTENTION_HEAD_DIM)
            state[f"{prefix}.self_attn.k_norm.weight"] = torch.ones(FLUX2_ATTENTION_HEAD_DIM)
    state["model.norm.weight"] = torch.ones(hidden)
    return state


class TransformerConfigTests(unittest.TestCase):
    def test_the_configuration_is_read_off_the_checkpoint(self):
        config = infer_flux2_transformer_config(comfy_flux2_checkpoint(double_blocks=2, single_blocks=3))
        self.assertEqual(config["patch_size"], 1)
        self.assertEqual(config["in_channels"], IN_CHANNELS)
        self.assertEqual(config["out_channels"], OUT_CHANNELS)
        self.assertEqual(config["num_layers"], 2)
        self.assertEqual(config["num_single_layers"], 3)
        self.assertEqual(config["num_attention_heads"], 1)
        self.assertEqual(config["attention_head_dim"], FLUX2_ATTENTION_HEAD_DIM)
        self.assertEqual(config["joint_attention_dim"], JOINT)
        self.assertEqual(config["mlp_ratio"], MLP_RATIO)
        self.assertEqual(config["axes_dims_rope"], FLUX2_AXES_DIMS_ROPE)
        self.assertEqual(config["rope_theta"], FLUX2_ROPE_THETA)
        self.assertTrue(config["guidance_embeds"])

    def test_a_model_without_a_guidance_embedding_is_recognised(self):
        config = infer_flux2_transformer_config(comfy_flux2_checkpoint(guidance=False))
        self.assertFalse(config["guidance_embeds"])

    def test_a_flux1_checkpoint_is_refused(self):
        # FLUX.1 shares the double/single block naming; the shared modulation projection is what
        # tells the two families apart, exactly as `model_detection` does it.
        with self.assertRaises(ValueError) as error:
            infer_flux2_transformer_config(comfy_flux_checkpoint())
        self.assertIn("not a FLUX.2 diffusion model", str(error.exception))

    def test_the_rope_axes_sum_to_the_attention_head(self):
        self.assertEqual(sum(FLUX2_AXES_DIMS_ROPE), FLUX2_ATTENTION_HEAD_DIM)


class TransformerConversionTests(unittest.TestCase):
    def test_every_source_tensor_is_claimed(self):
        state = comfy_flux2_checkpoint(double_blocks=2, single_blocks=2)
        config = infer_flux2_transformer_config(state)
        _converted, unclaimed = convert_flux2_transformer_state_dict(state, config)
        self.assertEqual(unclaimed, [])

    def test_an_unknown_tensor_is_reported_rather_than_swallowed(self):
        state = comfy_flux2_checkpoint()
        state["mystery.weight"] = torch.zeros(1)
        config = infer_flux2_transformer_config(state)
        _converted, unclaimed = convert_flux2_transformer_state_dict(state, config)
        self.assertEqual(unclaimed, ["mystery.weight"])

    def test_the_fused_double_block_qkv_is_split_in_order(self):
        state = comfy_flux2_checkpoint()
        config = infer_flux2_transformer_config(state)
        converted, _unclaimed = convert_flux2_transformer_state_dict(state, config)
        fused = state["double_blocks.0.img_attn.qkv.weight"]
        for offset, name in enumerate(("to_q", "to_k", "to_v")):
            expected = fused[offset * HIDDEN:(offset + 1) * HIDDEN]
            self.assertTrue(torch.equal(converted[f"transformer_blocks.0.attn.{name}.weight"], expected))

    def test_the_single_block_projection_stays_fused(self):
        state = comfy_flux2_checkpoint()
        config = infer_flux2_transformer_config(state)
        converted, _unclaimed = convert_flux2_transformer_state_dict(state, config)
        self.assertTrue(torch.equal(
            converted["single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight"],
            state["single_blocks.0.linear1.weight"],
        ))

    def test_the_final_modulation_is_swapped_from_shift_scale_to_scale_shift(self):
        state = comfy_flux2_checkpoint()
        config = infer_flux2_transformer_config(state)
        converted, _unclaimed = convert_flux2_transformer_state_dict(state, config)
        shift, scale = state["final_layer.adaLN_modulation.1.weight"].chunk(2, dim=0)
        swapped = converted["norm_out.linear.weight"]
        self.assertTrue(torch.equal(swapped[:HIDDEN], scale))
        self.assertTrue(torch.equal(swapped[HIDDEN:], shift))

    def test_a_normalised_scale_key_is_accepted_as_a_weight(self):
        # `supported_models.Flux.process_unet_state_dict` renames `*_norm.scale` to `*_norm.weight`
        # before the model sees it, so both spellings reach a loader in practice.
        for suffix in ("scale", "weight"):
            with self.subTest(suffix=suffix):
                state = comfy_flux2_checkpoint(norm_suffix=suffix)
                config = infer_flux2_transformer_config(state)
                converted, unclaimed = convert_flux2_transformer_state_dict(state, config)
                self.assertEqual(unclaimed, [])
                self.assertIn("transformer_blocks.0.attn.norm_q.weight", converted)


class TransformerLoadTests(unittest.TestCase):
    def test_the_fixture_loads_strictly_and_runs_a_forward(self):
        deps = _runtime_dependencies()
        transformer, config, report = _load_flux2_transformer(
            Path("unused.safetensors"), torch.float32, deps, state_dict=comfy_flux2_checkpoint()
        )
        self.assertEqual(report["unclaimed_tensors"], [])
        latent_height, latent_width = 4, 4
        latents = torch.randn(1, IN_CHANNELS, latent_height, latent_width)
        with torch.inference_mode():
            output = transformer(
                hidden_states=pack_flux2_latents(latents),
                encoder_hidden_states=torch.randn(1, 8, JOINT),
                timestep=torch.full((1,), 0.5),
                img_ids=flux2_image_ids(latent_height, latent_width, torch.device("cpu"), torch.float32),
                txt_ids=flux2_text_ids(8, torch.device("cpu"), torch.float32),
                guidance=torch.full((1,), 4.0) if config["guidance_embeds"] else None,
                return_dict=False,
            )[0]
        self.assertEqual(tuple(output.shape), (1, latent_height * latent_width, OUT_CHANNELS))
        self.assertTrue(torch.isfinite(output).all())

    def test_a_missing_tensor_is_named(self):
        deps = _runtime_dependencies()
        state = comfy_flux2_checkpoint()
        del state["double_blocks.0.img_mlp.2.weight"]
        with self.assertRaises(ValueError) as error:
            _load_flux2_transformer(Path("unused.safetensors"), torch.float32, deps, state_dict=state)
        self.assertIn("img_mlp.2.weight", str(error.exception))


class LatentGeometryTests(unittest.TestCase):
    def test_the_autoencoder_pack_round_trips(self):
        latents = torch.randn(2, FLUX2_VAE_LATENT_CHANNELS, 8, 6)
        packed = patchify_flux2_latents(latents)
        self.assertEqual(tuple(packed.shape), (2, FLUX2_LATENT_CHANNELS, 4, 3))
        self.assertTrue(torch.equal(unpatchify_flux2_latents(packed), latents))

    def test_the_pack_is_channel_major_within_each_cell(self):
        # Channel index must be `c * 4 + row * 2 + column`, which is what both ComfyUI's rearrange
        # and Diffusers' permute produce. A transposed cell would decode to a scrambled image.
        latents = torch.zeros(1, FLUX2_VAE_LATENT_CHANNELS, 2, 2)
        latents[0, 1, 0, 1] = 5.0
        packed = patchify_flux2_latents(latents)
        self.assertEqual(packed[0, 1 * 4 + 0 * 2 + 1, 0, 0].item(), 5.0)

    def test_an_odd_autoencoder_latent_is_refused(self):
        with self.assertRaises(ValueError):
            patchify_flux2_latents(torch.zeros(1, FLUX2_VAE_LATENT_CHANNELS, 3, 4))

    def test_the_token_sequence_round_trips(self):
        latents = torch.randn(2, FLUX2_LATENT_CHANNELS, 5, 3)
        packed = pack_flux2_latents(latents)
        self.assertEqual(tuple(packed.shape), (2, 15, FLUX2_LATENT_CHANNELS))
        self.assertTrue(torch.equal(unpack_flux2_latents(packed, 5, 3), latents))

    def test_a_mismatched_sequence_is_refused(self):
        with self.assertRaises(ValueError):
            unpack_flux2_latents(torch.zeros(1, 15, FLUX2_LATENT_CHANNELS), 4, 3)

    def test_image_ids_carry_row_and_column_on_the_middle_axes(self):
        ids = flux2_image_ids(2, 3, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(ids.shape), (6, 4))
        self.assertTrue(torch.equal(ids[:, 0], torch.zeros(6)))
        self.assertTrue(torch.equal(ids[:, 1], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])))
        self.assertTrue(torch.equal(ids[:, 2], torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])))
        self.assertTrue(torch.equal(ids[:, 3], torch.zeros(6)))

    def test_text_ids_are_ordered_on_the_fourth_axis(self):
        ids = flux2_text_ids(4, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(ids.shape), (4, 4))
        self.assertTrue(torch.equal(ids[:, :3], torch.zeros(4, 3)))
        self.assertTrue(torch.equal(ids[:, 3], torch.tensor([0.0, 1.0, 2.0, 3.0])))

    def test_the_pixel_alignment_is_the_vae_stride_times_the_pack(self):
        self.assertEqual(FLUX2_PIXEL_ALIGNMENT, 16)
        self.assertEqual(FLUX2_LATENT_CHANNELS, FLUX2_VAE_LATENT_CHANNELS * 4)


class TextEncoderTests(unittest.TestCase):
    def test_a_mistral_encoder_is_told_from_a_qwen3_one(self):
        self.assertEqual(classify_flux2_text_encoder(mistral_text_encoder_state(layers=30).keys()), "mistral3")
        self.assertEqual(
            classify_flux2_text_encoder(mistral_text_encoder_state(layers=27, qwen=True).keys()), "qwen3"
        )
        self.assertEqual(classify_flux2_text_encoder({"shared.weight": None}.keys()), "unknown")

    def test_the_language_model_geometry_is_read_off_the_checkpoint(self):
        state = mistral_text_encoder_state(layers=30)
        weights = {key[len("model."):]: value for key, value in state.items()}
        config = infer_flux2_text_encoder_config(weights, "mistral3")
        self.assertEqual(config["hidden_size"], LLM_HIDDEN)
        self.assertEqual(config["num_hidden_layers"], 30)
        self.assertEqual(config["num_attention_heads"], 1)
        self.assertEqual(config["num_key_value_heads"], 1)
        self.assertEqual(config["intermediate_size"], LLM_HIDDEN * 2)
        self.assertEqual(config["vocab_size"], 64)
        self.assertEqual(config["rope_theta"], 1000000000.0)

    def test_a_qwen3_encoder_uses_its_own_rotary_base(self):
        state = mistral_text_encoder_state(layers=27, qwen=True)
        weights = {key[len("model."):]: value for key, value in state.items()}
        self.assertEqual(infer_flux2_text_encoder_config(weights, "qwen3")["rope_theta"], 1000000.0)

    def test_an_encoder_shallower_than_its_deepest_tap_is_refused(self):
        state = mistral_text_encoder_state(layers=8)
        weights = {key[len("model."):]: value for key, value in state.items()}
        with self.assertRaises(ValueError) as error:
            infer_flux2_text_encoder_config(weights, "mistral3")
        self.assertIn("too few", str(error.exception))

    def test_a_mistral_encoder_loads_strictly_and_reaches_its_deepest_tap(self):
        deps = _runtime_dependencies()
        state = mistral_text_encoder_state(layers=max(FLUX2_MISTRAL_TAP_LAYERS) + 2)
        encoder, family = _load_flux2_text_encoder(state, torch.float32, deps)
        self.assertEqual(family, "mistral3")
        ids = torch.tensor([[1, 5, 7, 9]])
        with torch.inference_mode():
            output = encoder(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                output_hidden_states=True, use_cache=False,
            )
        # One state per layer plus the embedding output, so the deepest tap has to be reachable.
        self.assertEqual(len(output.hidden_states), max(FLUX2_MISTRAL_TAP_LAYERS) + 3)
        self.assertEqual(tuple(output.hidden_states[max(FLUX2_MISTRAL_TAP_LAYERS)].shape), (1, 4, LLM_HIDDEN))

    def test_a_pruned_encoder_drops_its_final_norm_rather_than_neutralising_it(self):
        # ComfyUI prunes FLUX.2 [dev]'s Mistral to 30 layers and drops the final norm with it, so
        # the deepest tap is a raw layer output. An RMS norm with unit weights would still rescale
        # every position to norm 1, which is a different conditioning.
        deps = _runtime_dependencies()
        state = mistral_text_encoder_state(layers=max(FLUX2_MISTRAL_TAP_LAYERS))
        del state["model.norm.weight"]
        encoder, _family = _load_flux2_text_encoder(state, torch.float32, deps)
        self.assertIsInstance(encoder.norm, torch.nn.Identity)
        ids = torch.tensor([[1, 5, 7, 9]])
        with torch.inference_mode():
            output = encoder(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                output_hidden_states=True, use_cache=False,
            )
        deepest = output.hidden_states[max(FLUX2_MISTRAL_TAP_LAYERS)]
        root_mean_square = deepest.pow(2).mean(dim=-1).sqrt()
        self.assertFalse(torch.allclose(root_mean_square, torch.ones_like(root_mean_square), atol=1e-3))

    def test_a_qwen3_encoder_loads_strictly(self):
        deps = _runtime_dependencies()
        state = mistral_text_encoder_state(layers=max(FLUX2_QWEN_TAP_LAYERS) + 1, qwen=True)
        encoder, family = _load_flux2_text_encoder(state, torch.float32, deps)
        self.assertEqual(family, "qwen3")
        self.assertEqual(int(encoder.config.hidden_size), LLM_HIDDEN)

    def test_a_bundled_language_model_head_is_dropped_rather_than_refused(self):
        deps = _runtime_dependencies()
        state = mistral_text_encoder_state(layers=max(FLUX2_MISTRAL_TAP_LAYERS))
        state["lm_head.weight"] = torch.zeros(64, LLM_HIDDEN)
        state["tekken_model"] = torch.zeros(4, dtype=torch.uint8)
        encoder, _family = _load_flux2_text_encoder(state, torch.float32, deps)
        self.assertFalse(hasattr(encoder, "lm_head"))

    def test_the_tap_depths_match_comfyui(self):
        self.assertEqual(FLUX2_MISTRAL_TAP_LAYERS, (10, 20, 30))
        self.assertEqual(FLUX2_QWEN_TAP_LAYERS, (9, 18, 27))

    def test_a_quantised_encoder_is_expanded_rather_than_refused(self):
        # The published [klein] Qwen3 encoder ships as an FP8 Mixed file: every linear carries a
        # `comfy_quant` marker and a scale beside its weight. Loading it strictly without expanding
        # them first fails on the scales, and dropping them would load a mis-scaled encoder.
        deps = _runtime_dependencies()
        plain = mistral_text_encoder_state(layers=max(FLUX2_QWEN_TAP_LAYERS) + 1, qwen=True)
        quantised = {}
        for key, value in plain.items():
            quantised[key] = value
            if key.endswith(".weight") and value.dim() == 2:
                layer = key[: -len(".weight")]
                quantised[key] = value * 4.0
                quantised[f"{layer}.weight_scale"] = torch.tensor(0.25)
                quantised[f"{layer}.input_scale"] = torch.tensor(1.0)
                quantised[f"{layer}.comfy_quant"] = torch.tensor(
                    list(b'{"format": "float8_e4m3fn"}'), dtype=torch.uint8
                )
        encoder, family = _load_flux2_text_encoder(quantised, torch.float32, deps)
        self.assertEqual(family, "qwen3")
        self.assertTrue(torch.allclose(
            encoder.layers[0].self_attn.q_proj.weight.detach(),
            plain["model.layers.0.self_attn.q_proj.weight"],
            atol=1e-6,
        ))


TINY_VAE_CONFIG = {
    **{key: value for key, value in FLUX2_VAE_CONFIG.items() if key not in ("block_out_channels", "layers_per_block", "norm_num_groups", "sample_size")},
    "block_out_channels": (4, 4, 4, 4),
    "layers_per_block": 1,
    "norm_num_groups": 2,
    "sample_size": 64,
}


class AutoencoderLoadTests(unittest.TestCase):
    """The autoencoder is where FLUX.2's packing statistics live, so the load has to keep them."""

    def _write_vae(self, temporary, mutate=None):
        from safetensors.torch import save_file

        deps = _runtime_dependencies()
        vae = deps["AutoencoderKLFlux2"](**TINY_VAE_CONFIG)
        state = {key: value.clone() for key, value in vae.state_dict().items()}
        # Non-trivial statistics, so a dropped or truncated one is visible.
        state["bn.running_mean"] = torch.linspace(-0.5, 0.5, FLUX2_LATENT_CHANNELS)
        state["bn.running_var"] = torch.linspace(0.5, 1.5, FLUX2_LATENT_CHANNELS)
        # ComfyUI's published file nests the quantisation convolutions under the encoder and
        # decoder; `comfy/sd.py` lifts them before building the module, and so must this loader.
        for source, destination in (("quant_conv.", "encoder.quant_conv."), ("post_quant_conv.", "decoder.post_quant_conv.")):
            for key in [key for key in state if key.startswith(source)]:
                state[destination + key[len(source):]] = state.pop(key)
        state.pop("bn.num_batches_tracked", None)
        if mutate is not None:
            mutate(state)
        path = Path(temporary) / "flux2-vae.safetensors"
        save_file(state, str(path))
        return path

    def test_the_published_layout_loads_strictly_and_keeps_its_statistics(self):
        # The geometry is the module constant; only its width is shrunk here so the fixture loads
        # in milliseconds. Every rule under test — the key moves, the statistics, the strict
        # assign — is the production one.
        with tempfile.TemporaryDirectory() as temporary, patch.object(flux2_pipeline, "FLUX2_VAE_CONFIG", TINY_VAE_CONFIG):
            path = self._write_vae(temporary)
            vae, statistics = _load_flux2_vae(path, torch.float32, _runtime_dependencies())
        self.assertEqual(tuple(statistics["bn.running_mean"].shape), (FLUX2_LATENT_CHANNELS,))
        # Statistics stay float32 whatever the compute dtype: they rescale every latent.
        self.assertEqual(statistics["bn.running_var"].dtype, torch.float32)
        self.assertTrue(torch.allclose(vae.bn.running_mean, statistics["bn.running_mean"]))
        self.assertEqual(int(vae.config.latent_channels), FLUX2_VAE_LATENT_CHANNELS)
        self.assertIsNotNone(vae.quant_conv)
        self.assertIsNotNone(vae.post_quant_conv)

    def test_a_file_without_the_packing_statistics_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(flux2_pipeline, "FLUX2_VAE_CONFIG", TINY_VAE_CONFIG):
            path = self._write_vae(temporary, mutate=lambda state: state.pop("bn.running_var"))
            with self.assertRaises(ValueError) as error:
                _load_flux2_vae(path, torch.float32, _runtime_dependencies())
        self.assertIn("bn.running_var", str(error.exception))
        self.assertIn("not a FLUX.2 autoencoder", str(error.exception))


class RuntimeAssemblyTests(unittest.TestCase):
    """Three files on disk into one runtime, which is the only place the seams meet."""

    def _write_components(self, temporary, joint_hidden=LLM_HIDDEN):
        from safetensors.torch import save_file
        from test_tekken_tokenizer import tekken_blob

        directory = Path(temporary)
        deps = _runtime_dependencies()
        save_file(comfy_flux2_checkpoint(), str(directory / "flux2-dev.safetensors"))
        encoder = mistral_text_encoder_state(layers=max(FLUX2_MISTRAL_TAP_LAYERS), hidden=joint_hidden)
        encoder["tekken_model"] = torch.frombuffer(bytearray(tekken_blob()), dtype=torch.uint8).clone()
        save_file(encoder, str(directory / "flux2-text-encoder.safetensors"))
        vae = deps["AutoencoderKLFlux2"](**TINY_VAE_CONFIG)
        state = {key: value.clone() for key, value in vae.state_dict().items()}
        state.pop("bn.num_batches_tracked", None)
        save_file(state, str(directory / "flux2-vae.safetensors"))
        return (
            directory / "flux2-dev.safetensors",
            directory / "flux2-text-encoder.safetensors",
            directory / "flux2-vae.safetensors",
        )

    def test_three_component_files_assemble_into_one_runtime(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(flux2_pipeline, "FLUX2_VAE_CONFIG", TINY_VAE_CONFIG):
            paths = self._write_components(temporary)
            runtime = load_flux2_runtime(*paths, dtype=torch.float32)
            try:
                self.assertEqual(runtime.family, "mistral3")
                self.assertEqual(runtime.tap_layers, FLUX2_MISTRAL_TAP_LAYERS)
                # The tekken table came out of the checkpoint rather than off disk.
                self.assertEqual(runtime.tokenizer.encode("hello", add_bos=True)[0], 1)
                self.assertEqual(
                    set(runtime.weight_sizes) & {"transformer", "text_encoder", "vae"},
                    {"transformer", "text_encoder", "vae"},
                )
                self.assertFalse(runtime.batch_cfg)
            finally:
                runtime.close()

    def test_a_mismatched_encoder_and_transformer_are_refused_by_width(self):
        # Pairing a [dev] transformer with a [klein] encoder is the mistake this catches, and it
        # has to be caught by the numbers rather than thousands of tensors later.
        with tempfile.TemporaryDirectory() as temporary, patch.object(flux2_pipeline, "FLUX2_VAE_CONFIG", TINY_VAE_CONFIG):
            paths = self._write_components(temporary, joint_hidden=LLM_HIDDEN * 2)
            with self.assertRaises(ValueError) as error:
                load_flux2_runtime(*paths, dtype=torch.float32)
        self.assertIn("conditioning", str(error.exception))


class LoraConversionTests(unittest.TestCase):
    def test_a_diffusers_layout_lora_keeps_its_targets(self):
        deps = _runtime_dependencies()
        state = {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(2, HIDDEN),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(HIDDEN, 2),
        }
        converted = convert_flux2_lora_state_dict(state, deps, "test LoRA")
        self.assertEqual(sorted(converted), [
            "transformer_blocks.0.attn.to_q.lora_A.weight",
            "transformer_blocks.0.attn.to_q.lora_B.weight",
        ])

    def test_a_dora_checkpoint_is_refused_rather_than_silently_stripped(self):
        deps = _runtime_dependencies()
        with self.assertRaises(ValueError) as error:
            convert_flux2_lora_state_dict({"a.dora_scale": torch.zeros(1)}, deps, "test LoRA")
        self.assertIn("DoRA", str(error.exception))

    def test_an_unrecognised_layout_is_refused(self):
        deps = _runtime_dependencies()
        with self.assertRaises(ValueError):
            convert_flux2_lora_state_dict({"mystery": torch.zeros(1)}, deps, "test LoRA")


if __name__ == "__main__":
    unittest.main()
