import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import krea2_pipeline
from krea2_model import Krea2Transformer2DModel, rope, timestep_embedding
from krea2_pipeline import (
    KREA2_ATTENTION_HEAD_DIM,
    KREA2_IM_START_TOKEN,
    KREA2_LATENTS_MEAN,
    KREA2_LATENTS_STD,
    KREA2_MAX_EDGE,
    KREA2_PIXEL_ALIGNMENT,
    KREA2_TAP_COUNT,
    KREA2_TAP_LAYERS,
    KREA2_TEMPLATE,
    KREA2_VAE_LATENT_CHANNELS,
    KREA2_VAE_SCALE_FACTOR,
    Krea2Runtime,
    _load_krea2_text_encoder,
    _load_krea2_transformer,
    _load_krea2_vae,
    _runtime_dependencies,
    classify_krea2_text_encoder,
    convert_krea2_lora_state_dict,
    infer_krea2_text_encoder_config,
    infer_krea2_transformer_config,
    infer_krea2_vae_config,
    krea2_lora_key_map,
    krea2_template_end,
    krea2_vae_layout,
    load_krea2_runtime,
)
from gguf_loader import gguf_quantization_summary, read_gguf_header
from test_gguf_loader import write_checkpoint_gguf

# One 128-wide attention head is the narrowest fixture `model_detection`'s fixed head width allows,
# and it is the production width, so nothing about the attention is scaled down.
HEAD_DIM = KREA2_ATTENTION_HEAD_DIM
FEATURES = 2 * HEAD_DIM
HEADS = 2
KVHEADS = 1
TXTDIM = HEAD_DIM
TXTHEADS = 1
TXTKVHEADS = 1
TDIM = 256
CHANNELS = KREA2_VAE_LATENT_CHANNELS
PATCH = 2
MULTIPLIER = 1
BLOCK_MLP = 256  # `SwiGLU`: round_up(int(2 * 256 / 3) * 1, 128)
TEXT_MLP = 128  # round_up(int(2 * 128 / 3) * 1, 128)

LLM_HIDDEN = 32
LLM_HEAD_DIM = 16
LLM_LAYERS = max(KREA2_TAP_LAYERS) + 1

TINY_VAE_GEOMETRY = {
    "dim_mult": [1, 2],
    "num_res_blocks": 1,
    "attn_scales": [],
    "temperal_downsample": [False],
    "is_residual": False,
    "patch_size": None,
}


def _linear(state, prefix, out_features, in_features, bias=False):
    state[f"{prefix}.weight"] = torch.randn(out_features, in_features) * 0.02
    if bias:
        state[f"{prefix}.bias"] = torch.randn(out_features) * 0.02


def _attention(state, prefix, dim, heads, kvheads):
    _linear(state, f"{prefix}.wq", HEAD_DIM * heads, dim)
    _linear(state, f"{prefix}.wk", HEAD_DIM * kvheads, dim)
    _linear(state, f"{prefix}.wv", HEAD_DIM * kvheads, dim)
    _linear(state, f"{prefix}.gate", dim, dim)
    _linear(state, f"{prefix}.wo", dim, dim)
    state[f"{prefix}.qknorm.qnorm.scale"] = torch.zeros(HEAD_DIM)
    state[f"{prefix}.qknorm.knorm.scale"] = torch.zeros(HEAD_DIM)


def _fusion_block(state, prefix):
    state[f"{prefix}.prenorm.scale"] = torch.zeros(TXTDIM)
    state[f"{prefix}.postnorm.scale"] = torch.zeros(TXTDIM)
    _attention(state, f"{prefix}.attn", TXTDIM, TXTHEADS, TXTKVHEADS)
    _linear(state, f"{prefix}.mlp.gate", TEXT_MLP, TXTDIM)
    _linear(state, f"{prefix}.mlp.up", TEXT_MLP, TXTDIM)
    _linear(state, f"{prefix}.mlp.down", TXTDIM, TEXT_MLP)


def comfy_krea2_checkpoint(layers=2, txtlayers=KREA2_TAP_COUNT):
    """A tiny checkpoint in ComfyUI's Krea 2 layout.

    The transformer keeps ComfyUI's naming, so this fixture is simultaneously the checkpoint the
    loader reads and the key set the module declares — a strict load succeeding is evidence the
    two agree, which is exactly the guarantee the Flux engines get from their converters.
    """
    state = {}
    _linear(state, "first", FEATURES, CHANNELS * PATCH * PATCH, bias=True)
    for index in range(layers):
        prefix = f"blocks.{index}"
        state[f"{prefix}.mod.lin"] = torch.zeros(6 * FEATURES)
        state[f"{prefix}.prenorm.scale"] = torch.zeros(FEATURES)
        state[f"{prefix}.postnorm.scale"] = torch.zeros(FEATURES)
        _attention(state, f"{prefix}.attn", FEATURES, HEADS, KVHEADS)
        _linear(state, f"{prefix}.mlp.gate", BLOCK_MLP, FEATURES)
        _linear(state, f"{prefix}.mlp.up", BLOCK_MLP, FEATURES)
        _linear(state, f"{prefix}.mlp.down", FEATURES, BLOCK_MLP)
    _linear(state, "tmlp.0", FEATURES, TDIM, bias=True)
    _linear(state, "tmlp.2", FEATURES, FEATURES, bias=True)
    for index in range(2):
        _fusion_block(state, f"txtfusion.layerwise_blocks.{index}")
        _fusion_block(state, f"txtfusion.refiner_blocks.{index}")
    _linear(state, "txtfusion.projector", 1, txtlayers)
    state["txtmlp.0.scale"] = torch.zeros(TXTDIM)
    _linear(state, "txtmlp.1", FEATURES, TXTDIM, bias=True)
    _linear(state, "txtmlp.3", FEATURES, FEATURES, bias=True)
    state["last.norm.scale"] = torch.zeros(FEATURES)
    _linear(state, "last.linear", CHANNELS * PATCH * PATCH, FEATURES, bias=True)
    state["last.modulation.lin"] = torch.zeros(2, FEATURES)
    _linear(state, "tproj.1", 6 * FEATURES, FEATURES, bias=True)
    return state


def qwen3vl_text_encoder_state(layers=LLM_LAYERS, hidden=LLM_HIDDEN, vocab=256):
    """A miniature Qwen3-VL in the key layout a published Krea 2 text encoder carries.

    The vision tower and the language-model head are included because the real file has them and
    dropping them is the loader's job, not the fixture's.
    """
    state = {"model.language_model.embed_tokens.weight": torch.randn(vocab, hidden) * 0.02}
    for index in range(layers):
        prefix = f"model.language_model.layers.{index}"
        for name, shape in (
            ("self_attn.q_proj", (LLM_HEAD_DIM * 2, hidden)),
            ("self_attn.k_proj", (LLM_HEAD_DIM, hidden)),
            ("self_attn.v_proj", (LLM_HEAD_DIM, hidden)),
            ("self_attn.o_proj", (hidden, LLM_HEAD_DIM * 2)),
            ("mlp.gate_proj", (hidden * 2, hidden)),
            ("mlp.up_proj", (hidden * 2, hidden)),
            ("mlp.down_proj", (hidden, hidden * 2)),
        ):
            state[f"{prefix}.{name}.weight"] = torch.randn(*shape) * 0.02
        state[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden)
        state[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(hidden)
        state[f"{prefix}.self_attn.q_norm.weight"] = torch.ones(LLM_HEAD_DIM)
        state[f"{prefix}.self_attn.k_norm.weight"] = torch.ones(LLM_HEAD_DIM)
    state["model.language_model.norm.weight"] = torch.ones(hidden)
    state["model.visual.deepstack_merger_list.0.norm.weight"] = torch.ones(hidden)
    state["lm_head.weight"] = torch.randn(vocab, hidden) * 0.02
    return state


class TransformerConfigTests(unittest.TestCase):
    def test_the_configuration_is_read_off_the_checkpoint(self):
        config = infer_krea2_transformer_config(comfy_krea2_checkpoint(layers=3))
        self.assertEqual(config["features"], FEATURES)
        self.assertEqual(config["layers"], 3)
        self.assertEqual(config["heads"], HEADS)
        self.assertEqual(config["kvheads"], KVHEADS)
        self.assertEqual(config["channels"], CHANNELS)
        self.assertEqual(config["patch"], PATCH)
        self.assertEqual(config["tdim"], TDIM)
        self.assertEqual(config["txtdim"], TXTDIM)
        self.assertEqual(config["txtlayers"], KREA2_TAP_COUNT)
        self.assertEqual(config["txtheads"], TXTHEADS)
        self.assertEqual(config["multiplier"], MULTIPLIER)
        self.assertFalse(config["bias"])

    def test_a_file_without_the_text_fusion_projector_is_refused(self):
        # `model_detection` recognises Krea 2 by exactly this tensor; nothing else has it.
        state = comfy_krea2_checkpoint()
        state.pop("txtfusion.projector.weight")
        with self.assertRaises(ValueError) as error:
            infer_krea2_transformer_config(state)
        self.assertIn("not a Krea 2 diffusion model", str(error.exception))

    def test_a_projector_collapsing_the_wrong_number_of_taps_is_refused(self):
        # A twelve-tap conditioning is not negotiable: the text encoder produces twelve states and
        # a model expecting another count would silently read the wrong depths.
        with self.assertRaises(ValueError) as error:
            infer_krea2_transformer_config(comfy_krea2_checkpoint(txtlayers=3))
        self.assertIn("3 text-encoder taps", str(error.exception))

    def test_the_swiglu_multiplier_is_recovered_rather_than_assumed(self):
        # ComfyUI leaves the multiplier at its constructor default; reading it back off the MLP
        # width is what lets a differently-proportioned checkpoint load through the same path.
        state = comfy_krea2_checkpoint()
        for name in ("gate", "up"):
            state[f"blocks.0.mlp.{name}.weight"] = torch.zeros(384, FEATURES)
        state["blocks.0.mlp.down.weight"] = torch.zeros(FEATURES, 384)
        for name in ("gate", "up"):
            state[f"txtfusion.layerwise_blocks.0.mlp.{name}.weight"] = torch.zeros(256, TXTDIM)
        state["txtfusion.layerwise_blocks.0.mlp.down.weight"] = torch.zeros(TXTDIM, 256)
        self.assertEqual(infer_krea2_transformer_config(state)["multiplier"], 2)

    def test_a_model_whose_two_stacks_disagree_about_the_multiplier_is_refused(self):
        state = comfy_krea2_checkpoint()
        for name in ("gate", "up"):
            state[f"blocks.0.mlp.{name}.weight"] = torch.zeros(384, FEATURES)
        state["blocks.0.mlp.down.weight"] = torch.zeros(FEATURES, 384)
        with self.assertRaises(ValueError) as error:
            infer_krea2_transformer_config(state)
        self.assertIn("shares one multiplier", str(error.exception))

    def test_a_width_no_multiplier_reproduces_is_reported(self):
        state = comfy_krea2_checkpoint()
        state["blocks.0.mlp.gate.weight"] = torch.zeros(300, FEATURES)
        with self.assertRaises(ValueError) as error:
            infer_krea2_transformer_config(state)
        self.assertIn("no multiplier between 1 and 16", str(error.exception))


class Krea2GgufCheckpointTests(unittest.TestCase):
    def test_a_quantised_gguf_checkpoint_loads_through_the_same_conversion(self):
        state = comfy_krea2_checkpoint()
        deps = _runtime_dependencies()
        expected, expected_config, _report = _load_krea2_transformer(
            None, torch.float32, deps, state_dict=dict(state)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_checkpoint_gguf(Path(directory) / "krea2.gguf", state)
            self.assertIn("Q8_0", gguf_quantization_summary(read_gguf_header(path)))
            transformer, config, report = _load_krea2_transformer(path, torch.float32, deps)
        self.assertEqual(config, expected_config)
        self.assertEqual(report["unclaimed_tensors"], [])
        produced = dict(transformer.named_parameters())
        self.assertEqual(sorted(produced), sorted(name for name, _ in expected.named_parameters()))
        for name, parameter in expected.named_parameters():
            self.assertTrue(torch.allclose(produced[name], parameter, atol=1e-3), name)


class TransformerModuleTests(unittest.TestCase):
    def test_the_fixture_loads_strictly_and_runs_a_forward(self):
        state = comfy_krea2_checkpoint(layers=2)
        deps = _runtime_dependencies()
        transformer, config, report = _load_krea2_transformer(None, torch.float32, deps, state_dict=dict(state))
        self.assertEqual(report["unclaimed_tensors"], [])
        latents = torch.randn(1, CHANNELS, 4, 6)
        context = torch.randn(1, 5, KREA2_TAP_COUNT * TXTDIM)
        with torch.inference_mode():
            out = transformer(latents, torch.tensor([1.0]), context)
        self.assertEqual(tuple(out.shape), (1, CHANNELS, 4, 6))
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(config["layers"], 2)

    def test_a_tensor_the_module_does_not_declare_is_reported_rather_than_swallowed(self):
        state = comfy_krea2_checkpoint()
        state["mystery.weight"] = torch.zeros(1)
        _transformer, _config, report = _load_krea2_transformer(
            None, torch.float32, _runtime_dependencies(), state_dict=state
        )
        self.assertEqual(report["unclaimed_tensors"], ["mystery.weight"])

    def test_a_missing_tensor_is_named(self):
        state = comfy_krea2_checkpoint()
        state.pop("blocks.0.attn.wo.weight")
        with self.assertRaises(RuntimeError) as error:
            _load_krea2_transformer(None, torch.float32, _runtime_dependencies(), state_dict=state)
        self.assertIn("blocks.0.attn.wo.weight", str(error.exception))

    def test_an_odd_latent_is_padded_and_cropped_back(self):
        # `process_img` pads circularly to the 2x2 patch grid, and the forward crops the padding
        # off again, so an odd latent comes back at its own size rather than the padded one.
        transformer, _config, _report = _load_krea2_transformer(
            None, torch.float32, _runtime_dependencies(), state_dict=comfy_krea2_checkpoint()
        )
        with torch.inference_mode():
            out = transformer(torch.randn(1, CHANNELS, 3, 5), torch.tensor([1.0]), torch.randn(1, 2, KREA2_TAP_COUNT * TXTDIM))
        self.assertEqual(tuple(out.shape), (1, CHANNELS, 3, 5))

    def test_conditioning_of_the_wrong_width_is_refused_by_the_numbers(self):
        transformer, _config, _report = _load_krea2_transformer(
            None, torch.float32, _runtime_dependencies(), state_dict=comfy_krea2_checkpoint()
        )
        with self.assertRaises(ValueError) as error:
            transformer(torch.randn(1, CHANNELS, 4, 4), torch.tensor([1.0]), torch.randn(1, 2, TXTDIM))
        self.assertIn("Qwen3-VL stack", str(error.exception))

    def test_the_rope_axes_sum_to_the_attention_head(self):
        transformer, _config, _report = _load_krea2_transformer(
            None, torch.float32, _runtime_dependencies(), state_dict=comfy_krea2_checkpoint()
        )
        self.assertEqual(sum(transformer.pe_embedder.axes_dim), HEAD_DIM)
        self.assertEqual(transformer.pe_embedder.axes_dim, [32, 48, 48])

    def test_the_timestep_embedding_puts_cosine_first(self):
        # Diffusers' equivalent emits sine first; swapping the halves would shift every timestep
        # by a quarter period without changing a single shape.
        embedding = timestep_embedding(torch.tensor([0.5]), 8)
        self.assertEqual(tuple(embedding.shape), (1, 8))
        # `time_factor` is 1000, so t=0.5 enters the ladder at 500 and the lowest frequency is 1.
        self.assertTrue(torch.allclose(embedding[0, 0], torch.cos(torch.tensor(500.0))))
        self.assertTrue(torch.allclose(embedding[0, 4], torch.sin(torch.tensor(500.0))))

    def test_rope_pairs_are_orthonormal_rotations(self):
        table = rope(torch.arange(4, dtype=torch.float32).reshape(1, 4), 8, 1000)
        self.assertEqual(tuple(table.shape), (1, 4, 4, 2, 2))
        determinant = table[..., 0, 0] * table[..., 1, 1] - table[..., 0, 1] * table[..., 1, 0]
        self.assertTrue(torch.allclose(determinant, torch.ones_like(determinant), atol=1e-6))


class TextEncoderTests(unittest.TestCase):
    def test_a_qwen3vl_encoder_is_recognised_by_its_per_head_norms(self):
        self.assertEqual(classify_krea2_text_encoder(qwen3vl_text_encoder_state()), "qwen3vl")

    def test_a_file_without_qk_norms_is_not_a_krea2_encoder(self):
        state = qwen3vl_text_encoder_state()
        for key in [key for key in state if key.endswith("q_norm.weight")]:
            state.pop(key)
        self.assertEqual(classify_krea2_text_encoder(state), "unknown")

    def test_the_language_model_geometry_is_read_off_the_checkpoint(self):
        weights = krea2_pipeline.krea2_text_encoder_weights(qwen3vl_text_encoder_state())
        config = infer_krea2_text_encoder_config(weights)
        self.assertEqual(config["hidden_size"], LLM_HIDDEN)
        self.assertEqual(config["num_hidden_layers"], LLM_LAYERS)
        self.assertEqual(config["num_attention_heads"], 2)
        self.assertEqual(config["num_key_value_heads"], 1)
        self.assertEqual(config["head_dim"], LLM_HEAD_DIM)
        self.assertEqual(config["rope_theta"], 5000000.0)

    def test_the_vision_tower_and_head_are_dropped_rather_than_loaded(self):
        weights = krea2_pipeline.krea2_text_encoder_weights(qwen3vl_text_encoder_state())
        self.assertNotIn("lm_head.weight", weights)
        self.assertFalse(any(key.startswith("visual.") for key in weights))
        self.assertIn("embed_tokens.weight", weights)
        self.assertIn("norm.weight", weights)

    def test_an_encoder_shallower_than_its_deepest_tap_is_refused(self):
        weights = krea2_pipeline.krea2_text_encoder_weights(qwen3vl_text_encoder_state(layers=8))
        with self.assertRaises(ValueError) as error:
            infer_krea2_text_encoder_config(weights)
        self.assertIn("deepest is layer 35", str(error.exception))

    def test_the_encoder_loads_strictly_and_reaches_its_deepest_tap(self):
        encoder = _load_krea2_text_encoder(
            qwen3vl_text_encoder_state(), torch.float32, _runtime_dependencies()
        )
        ids = torch.tensor([[1, 2, 3, 4]])
        with torch.inference_mode():
            states = encoder(
                input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False
            ).hidden_states
        # hidden_states[k] is the input to layer k, which is exactly the tap ComfyUI's
        # `all_intermediate` list records; the deepest one must exist.
        self.assertGreater(len(states), max(KREA2_TAP_LAYERS))
        self.assertEqual(tuple(states[max(KREA2_TAP_LAYERS)].shape), (1, 4, LLM_HIDDEN))

    def test_the_tap_depths_match_comfyui(self):
        self.assertEqual(KREA2_TAP_LAYERS, (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35))
        self.assertEqual(KREA2_TAP_COUNT, 12)


class PromptTemplateTests(unittest.TestCase):
    def test_the_prefix_strip_lands_on_the_prompt(self):
        # `<|im_start|>system ... <|im_end|> <|im_start|> user \n` is dropped; the prompt and the
        # assistant opening are kept, because the model was trained reading them.
        ids = [KREA2_IM_START_TOKEN, 8948, 198, 99, KREA2_IM_START_TOKEN, 872, 198, 7, 8]
        self.assertEqual(krea2_template_end(ids), 7)

    def test_a_user_turn_without_the_newline_keeps_the_marker(self):
        ids = [KREA2_IM_START_TOKEN, 8948, KREA2_IM_START_TOKEN, 5, 6, 7]
        self.assertEqual(krea2_template_end(ids), 2)

    def test_a_template_with_one_marker_is_refused(self):
        with self.assertRaises(ValueError):
            krea2_template_end([KREA2_IM_START_TOKEN, 1, 2])

    def test_the_template_is_comfyuis(self):
        self.assertTrue(KREA2_TEMPLATE.startswith("<|im_start|>system\nDescribe the image by detailing"))
        self.assertTrue(KREA2_TEMPLATE.endswith("<|im_start|>assistant\n"))
        self.assertIn("<|im_start|>user\n{}<|im_end|>", KREA2_TEMPLATE)


class AutoencoderTests(unittest.TestCase):
    def _write_vae(self, temporary, mutate=None):
        from safetensors.torch import save_file

        deps = _runtime_dependencies()
        vae = deps["AutoencoderKLWan"](base_dim=8, z_dim=4, **TINY_VAE_GEOMETRY, in_channels=3, out_channels=3)
        state = {key: value.clone() for key, value in vae.state_dict().items()}
        if mutate is not None:
            mutate(state)
        path = Path(temporary) / "krea2-vae.safetensors"
        save_file(state, str(path))
        return path

    def test_the_wan_layout_geometry_is_read_the_way_comfyui_reads_it(self):
        state = {
            "decoder.middle.0.residual.0.gamma": torch.zeros(96),
            "decoder.head.0.gamma": torch.zeros(96),
            "encoder.head.2.weight": torch.zeros(32, 384, 3, 3, 3),
            "encoder.conv1.weight": torch.zeros(96, 3, 3, 3, 3),
            "decoder.head.2.weight": torch.zeros(3, 96, 3, 3, 3),
        }
        config = infer_krea2_vae_config(state)
        self.assertEqual(krea2_vae_layout(state), "wan")
        self.assertEqual(config["base_dim"], 96)
        self.assertEqual(config["z_dim"], KREA2_VAE_LATENT_CHANNELS)
        self.assertEqual(config["in_channels"], 3)
        self.assertEqual(config["out_channels"], 3)
        self.assertEqual(config["temperal_downsample"], [False, True, True])

    def test_a_wan22_autoencoder_is_refused_by_name(self):
        state = {
            "decoder.middle.0.residual.0.gamma": torch.zeros(160),
            "decoder.upsamples.0.upsamples.0.residual.2.weight": torch.zeros(1),
        }
        with self.assertRaises(ValueError) as error:
            infer_krea2_vae_config(state)
        self.assertIn("Wan 2.2", str(error.exception))

    def test_a_file_that_is_not_an_autoencoder_is_refused(self):
        with self.assertRaises(ValueError) as error:
            infer_krea2_vae_config({"first.weight": torch.zeros(1)})
        self.assertIn("not a Krea 2 autoencoder", str(error.exception))

    def test_a_diffusers_layout_file_loads_strictly_and_round_trips_one_frame(self):
        # The published file is in Wan's own layout and reaches `AutoencoderKLWan` through
        # Diffusers' converter; a file already in Diffusers naming skips that step, which is the
        # branch this exercises end to end. Only the block geometry is shrunk.
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            krea2_pipeline, "KREA2_VAE_GEOMETRY", TINY_VAE_GEOMETRY
        ):
            path = self._write_vae(temporary)
            vae, config = _load_krea2_vae(path, torch.float32, _runtime_dependencies())
        self.assertEqual(config["z_dim"], 4)
        self.assertEqual(config["base_dim"], 8)
        with torch.inference_mode():
            latents = vae.encode(torch.randn(1, 3, 1, 16, 16)).latent_dist.mode()
            decoded = vae.decode(latents, return_dict=False)[0]
        # The fixture keeps one downsampling stage rather than Wan 2.1's three, so the stride is 2
        # here; what is under test is that the load produced a working autoencoder, not its depth.
        self.assertEqual(tuple(latents.shape), (1, 4, 1, 8, 8))
        self.assertEqual(tuple(decoded.shape), (1, 3, 1, 16, 16))

    def test_a_truncated_file_fails_the_strict_load_rather_than_loading_half_a_model(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            krea2_pipeline, "KREA2_VAE_GEOMETRY", TINY_VAE_GEOMETRY
        ):
            path = self._write_vae(temporary, mutate=lambda state: state.pop("post_quant_conv.weight"))
            with self.assertRaises(RuntimeError) as error:
                _load_krea2_vae(path, torch.float32, _runtime_dependencies())
        self.assertIn("Krea 2 autoencoder", str(error.exception))

    def test_the_latent_statistics_are_wan21s(self):
        self.assertEqual(len(KREA2_LATENTS_MEAN), KREA2_VAE_LATENT_CHANNELS)
        self.assertEqual(len(KREA2_LATENTS_STD), KREA2_VAE_LATENT_CHANNELS)
        self.assertAlmostEqual(KREA2_LATENTS_MEAN[0], -0.7571)
        self.assertAlmostEqual(KREA2_LATENTS_STD[0], 2.8184)

    def test_the_pixel_alignment_is_the_vae_stride_times_the_pack(self):
        self.assertEqual(KREA2_PIXEL_ALIGNMENT, KREA2_VAE_SCALE_FACTOR * PATCH)
        self.assertEqual(KREA2_MAX_EDGE % KREA2_PIXEL_ALIGNMENT, 0)


class LoraConversionTests(unittest.TestCase):
    def _config(self):
        return infer_krea2_transformer_config(comfy_krea2_checkpoint())

    def test_the_key_map_matches_comfyuis_table(self):
        key_map = krea2_lora_key_map({"layers": 2})
        self.assertEqual(key_map["transformer_blocks.1.attn.to_q"], "blocks.1.attn.wq")
        self.assertEqual(key_map["transformer_blocks.0.attn.to_gate"], "blocks.0.attn.gate")
        self.assertEqual(key_map["transformer_blocks.0.attn.to_out.0"], "blocks.0.attn.wo")
        self.assertEqual(key_map["text_fusion.refiner_blocks.1.ff.down"], "txtfusion.refiner_blocks.1.mlp.down")
        self.assertEqual(key_map["time_mod_proj"], "tproj.1")
        self.assertEqual(key_map["txt_in.linear_1"], "txtmlp.1")
        self.assertEqual(key_map["final_layer.linear"], "last.linear")

    def test_a_diffusers_layout_lora_lands_on_comfy_names(self):
        lora = {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(4, FEATURES),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(FEATURES, 4),
        }
        converted = convert_krea2_lora_state_dict(lora, self._config(), "Krea2 LoRA test")
        self.assertIn("blocks.0.attn.wq.lora_A.weight", converted)
        self.assertIn("blocks.0.attn.wq.lora_B.weight", converted)

    def test_a_kohya_layout_lora_is_accepted_and_alpha_is_folded_in(self):
        lora = {
            "diffusion_model.transformer_blocks.0.ff.down.lora_down.weight": torch.ones(2, BLOCK_MLP),
            "diffusion_model.transformer_blocks.0.ff.down.lora_up.weight": torch.ones(FEATURES, 2),
            "diffusion_model.transformer_blocks.0.ff.down.alpha": torch.tensor(1.0),
        }
        converted = convert_krea2_lora_state_dict(lora, self._config(), "Krea2 LoRA test")
        # alpha 1 over rank 2 halves the down matrix.
        self.assertTrue(torch.allclose(converted["blocks.0.mlp.down.lora_A.weight"], torch.full((2, BLOCK_MLP), 0.5)))

    def test_a_lora_already_in_comfy_naming_maps_to_itself(self):
        lora = {
            "diffusion_model.blocks.1.attn.wv.lora_A.weight": torch.randn(4, FEATURES),
            "diffusion_model.blocks.1.attn.wv.lora_B.weight": torch.randn(HEAD_DIM, 4),
        }
        converted = convert_krea2_lora_state_dict(lora, self._config(), "Krea2 LoRA test")
        self.assertIn("blocks.1.attn.wv.lora_A.weight", converted)

    def test_a_dora_checkpoint_is_refused_rather_than_silently_stripped(self):
        with self.assertRaises(ValueError) as error:
            convert_krea2_lora_state_dict({"blocks.0.attn.wq.dora_scale": torch.zeros(1)}, self._config(), "Krea2 LoRA test")
        self.assertIn("DoRA", str(error.exception))

    def test_an_unrecognised_layout_is_refused_and_names_what_it_saw(self):
        lora = {
            "not_a_krea2_module.lora_A.weight": torch.randn(4, 4),
            "not_a_krea2_module.lora_B.weight": torch.randn(4, 4),
        }
        with self.assertRaises(ValueError) as error:
            convert_krea2_lora_state_dict(lora, self._config(), "Krea2 LoRA test")
        self.assertIn("not_a_krea2_module", str(error.exception))

    def test_a_fused_lora_changes_the_weight_it_targets(self):
        state = comfy_krea2_checkpoint()
        base = state["blocks.0.attn.wq.weight"].clone()
        with tempfile.TemporaryDirectory() as temporary:
            from safetensors.torch import save_file

            lora_path = Path(temporary) / "krea2-lora.safetensors"
            save_file(
                {
                    "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, FEATURES) * 0.01,
                    "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(HEAD_DIM * HEADS, 2) * 0.01,
                },
                str(lora_path),
            )
            transformer, _config, report = _load_krea2_transformer(
                None, torch.float32, _runtime_dependencies(), loras=[(lora_path, 1.0)], state_dict=state
            )
        self.assertEqual(report["loras"][0]["patched_modules"], 1)
        self.assertFalse(torch.allclose(transformer.blocks[0].attn.wq.weight, base))


class RuntimeAssemblyTests(unittest.TestCase):
    """Three files on disk into one runtime, which is the only place the seams meet."""

    def _write_components(self, temporary, llm_hidden=TXTDIM):
        from safetensors.torch import save_file

        directory = Path(temporary)
        deps = _runtime_dependencies()
        save_file(comfy_krea2_checkpoint(), str(directory / "krea2.safetensors"))
        save_file(
            qwen3vl_text_encoder_state(hidden=llm_hidden), str(directory / "krea2-text-encoder.safetensors")
        )
        vae = deps["AutoencoderKLWan"](
            base_dim=8, z_dim=CHANNELS, **TINY_VAE_GEOMETRY, in_channels=3, out_channels=3
        )
        save_file(
            {key: value.clone() for key, value in vae.state_dict().items()},
            str(directory / "krea2-vae.safetensors"),
        )
        return (
            directory / "krea2.safetensors",
            directory / "krea2-text-encoder.safetensors",
            directory / "krea2-vae.safetensors",
        )

    def _tokenizer_path(self):
        return Path(__file__).resolve().parent / "resources" / "anima-tokenizers" / "anima-qwen3-tokenizer.json"

    def test_three_component_files_assemble_into_one_runtime(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            krea2_pipeline, "KREA2_VAE_GEOMETRY", TINY_VAE_GEOMETRY
        ):
            paths = self._write_components(temporary)
            runtime = load_krea2_runtime(*paths, qwen_tokenizer_path=self._tokenizer_path(), dtype=torch.float32)
            try:
                self.assertIsInstance(runtime.transformer, Krea2Transformer2DModel)
                self.assertEqual(runtime.tap_layers, KREA2_TAP_LAYERS)
                self.assertEqual(runtime.latent_channels, CHANNELS)
                self.assertEqual(
                    set(runtime.weight_sizes) & {"transformer", "text_encoder", "vae"},
                    {"transformer", "text_encoder", "vae"},
                )
                self.assertGreater(runtime.weight_sizes["transformer_max_block"], 0)
                # Krea 2 is not guidance distilled, but the two branches still run sequentially.
                self.assertFalse(runtime.batch_cfg)
            finally:
                runtime.close()

    def test_a_mismatched_encoder_and_transformer_are_refused_by_width(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            krea2_pipeline, "KREA2_VAE_GEOMETRY", TINY_VAE_GEOMETRY
        ):
            paths = self._write_components(temporary, llm_hidden=TXTDIM * 2)
            with self.assertRaises(ValueError) as error:
                load_krea2_runtime(*paths, qwen_tokenizer_path=self._tokenizer_path(), dtype=torch.float32)
        self.assertIn("conditioning", str(error.exception))

    def test_a_missing_tokenizer_resource_is_named(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            krea2_pipeline, "KREA2_VAE_GEOMETRY", TINY_VAE_GEOMETRY
        ):
            paths = self._write_components(temporary)
            with self.assertRaises(FileNotFoundError) as error:
                load_krea2_runtime(*paths, qwen_tokenizer_path=Path(temporary) / "absent.json", dtype=torch.float32)
        self.assertIn("Qwen tokenizer", str(error.exception))

    def test_the_prompt_tokenises_and_the_conditioning_starts_after_the_template(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            krea2_pipeline, "KREA2_VAE_GEOMETRY", TINY_VAE_GEOMETRY
        ):
            paths = self._write_components(temporary)
            runtime = load_krea2_runtime(*paths, qwen_tokenizer_path=self._tokenizer_path(), dtype=torch.float32)
            try:
                diagnostics = runtime.token_diagnostics("a red fox")["llm"]
            finally:
                runtime.close()
        self.assertGreater(diagnostics["token_count"], diagnostics["conditioning_token_count"])
        self.assertEqual(diagnostics["template_tokens"], 34)
        self.assertEqual(diagnostics["tap_layers"], list(KREA2_TAP_LAYERS))


class LatentStatisticsTests(unittest.TestCase):
    def test_process_in_and_out_are_inverses(self):
        runtime = Krea2Runtime.__new__(Krea2Runtime)
        runtime._latent_mean = torch.tensor(KREA2_LATENTS_MEAN).reshape(1, -1, 1, 1)
        runtime._latent_std = torch.tensor(KREA2_LATENTS_STD).reshape(1, -1, 1, 1)
        latents = torch.randn(2, KREA2_VAE_LATENT_CHANNELS, 3, 3)
        restored = runtime._process_out(runtime._process_in(latents))
        self.assertTrue(torch.allclose(restored, latents, atol=1e-5))

    def test_process_in_matches_comfyuis_wan21_formula(self):
        runtime = Krea2Runtime.__new__(Krea2Runtime)
        runtime._latent_mean = torch.tensor(KREA2_LATENTS_MEAN).reshape(1, -1, 1, 1)
        runtime._latent_std = torch.tensor(KREA2_LATENTS_STD).reshape(1, -1, 1, 1)
        latents = torch.ones(1, KREA2_VAE_LATENT_CHANNELS, 1, 1)
        expected = (latents - runtime._latent_mean) / runtime._latent_std
        self.assertTrue(torch.allclose(runtime._process_in(latents), expected))


if __name__ == "__main__":
    unittest.main()
