"""Loading, LoRA, tokenizer and runtime contracts of the native MOSS-Image engine (QPlan2/MOSS/P1).

Everything runs on the CPU with small synthetic components; no real weights, no GPU. The numerical
oracles live in ``test_moss_numerics.py``.
"""

import gzip
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .. import comfy_quant, moss_model, moss_pipeline, moss_quant, moss_sampling, moss_text_encoder
    from ..quantized_linear import QuantizedLinear, UnsupportedQuantization, kron_adapter_output
    from .test_moss_numerics import TINY_DIT, TINY_TE, convrot_quantize, load_reference, randomize
except ImportError:
    import comfy_quant
    import moss_model
    import moss_pipeline
    import moss_quant
    import moss_sampling
    import moss_text_encoder
    from quantized_linear import QuantizedLinear, UnsupportedQuantization, kron_adapter_output
    from test_moss_numerics import TINY_DIT, TINY_TE, convrot_quantize, load_reference, randomize

BACKEND = Path(__file__).resolve().parents[1]
TOKENIZER = BACKEND / "resources" / "moss-tokenizers" / moss_text_encoder.TOKENIZER_GZ_NAME
INT8 = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}


def tiny_dit_state(seed=21, fused=True, **overrides):
    reference = load_reference()
    hf = randomize(reference.MOSSImageTransformer(**{**TINY_DIT, **overrides}), seed).eval()
    state = {key: value.detach().clone() for key, value in hf.state_dict().items()}
    if fused:
        state, _ = moss_model.canonical_moss_state(state)
    return state


def quantize_layers(state, layers):
    state = dict(state)
    for layer in layers:
        marker = comfy_quant.encode_marker(INT8)
        q, scale = convrot_quantize(state[f"{layer}.weight"].float(), 16)
        state[f"{layer}.weight"] = q
        state[f"{layer}.weight_scale"] = scale
        state[f"{layer}.comfy_quant"] = marker
    return state


def tiny_text_state(seed=31, layout="comfy", quantize=True):
    config = moss_text_encoder.Qwen35TextConfig(**TINY_TE)
    model = randomize(moss_text_encoder.Qwen35TextModel(config, compute_dtype=torch.float32), seed)
    with torch.no_grad():
        for layer in model.layers:
            if layer.layer_type == "linear_attention":
                layer.linear_attn.A_log.copy_(torch.log(torch.rand(4) * 2 + 0.5))
    language = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if quantize:
        linears = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
                   and name.rsplit(".", 1)[-1] not in ("in_proj_a", "in_proj_b")]
        language = quantize_layers(language, linears)
        q, scale = convrot_quantize(language["embed_tokens.weight"].float(), 16)
        language["embed_tokens.weight"], language["embed_tokens.weight_scale"] = q, scale
        language["embed_tokens.comfy_quant"] = comfy_quant.encode_marker(INT8)
    prefix = "model.language_model." if layout == "comfy" else "language_model."
    state = {prefix + key: value for key, value in language.items()}
    visual = "model.visual." if layout == "comfy" else "visual."
    state[visual + "blocks.0.attn.qkv.weight"] = torch.zeros(6, 2)
    state[visual + "blocks.0.norm1.weight"] = torch.zeros(2)
    state[visual + "pos_embed.weight"] = torch.zeros(4, 2)
    if layout == "comfy":
        state["lm_head.weight"] = torch.zeros(300, 32, dtype=torch.int8)
        state["lm_head.weight_scale"] = torch.ones(300, 1)
        state["lm_head.comfy_quant"] = comfy_quant.encode_marker(INT8)
        state["mtp.fc.weight"] = torch.zeros(32, 64)
        state["mtp.layers.0.self_attn.q_proj.weight"] = torch.zeros(64, 32)
    return state, model


class TokenizerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = moss_text_encoder.MossTokenizer.from_resource(TOKENIZER)

    def test_resource_is_the_pinned_upstream_file(self):
        text, record = moss_text_encoder.load_tokenizer_json(TOKENIZER)
        self.assertEqual(record["sha256"], moss_text_encoder.TOKENIZER_JSON_SHA256)
        self.assertEqual(record["size"], moss_text_encoder.TOKENIZER_JSON_SIZE)
        self.assertEqual(self.tokenizer.vocab_size, 248077)

    def test_template_tokens(self):
        ids, info = self.tokenizer.encode("")
        self.assertEqual(ids, [248045, 846, 198, 248046, 198, 248045, 74455, 198, 248068, 198])
        self.assertEqual(info["token_count"], moss_text_encoder.MOSS_TEMPLATE_TOKENS)
        self.assertFalse(info["truncated"])

    def test_prompt_is_trimmed_like_the_chat_template(self):
        self.assertEqual(self.tokenizer.encode("  A cat \n")[0], self.tokenizer.encode("A cat")[0])
        self.assertTrue(self.tokenizer.encode("  A cat \n")[1]["prompt_trimmed"])

    def test_comfy_escape_and_weight_syntax_are_literal(self):
        escaped, info = self.tokenizer.encode("\\(artist\\) (word:1.2)")
        plain, _ = self.tokenizer.encode("(artist) (word:1.2)")
        self.assertEqual(escaped, plain)
        self.assertTrue(info["escapes_removed"])
        self.assertEqual(info["weighted_token_count"], 0)

    def test_embedding_identifier_segments_like_comfy(self):
        ids, info = self.tokenizer.encode("a red embedding:fox here")
        templated = moss_text_encoder.MOSS_TEMPLATE.format("a red embedding:fox here")
        head, tail = templated.split("embedding:", 1)
        expected = self.tokenizer._ids(head) + self.tokenizer._ids("embedding:" + tail)
        self.assertEqual(ids, expected)
        self.assertEqual(info["segments"], 2)

    def test_truncation_includes_template_and_is_reported(self):
        ids, info = self.tokenizer.encode("word " * 5000)
        self.assertEqual(len(ids), 4096)
        self.assertTrue(info["truncated"])
        self.assertGreater(info["untruncated_token_count"], 4096)

    def test_corrupt_resource_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            bad = Path(root) / "bad.json.gz"
            bad.write_bytes(gzip.compress(b'{"version": "1.0"}'))
            with self.assertRaises(ValueError):
                moss_text_encoder.MossTokenizer.from_resource(bad)


class TextEncoderLoadingTests(unittest.TestCase):
    def test_comfy_layout_int8_loads_with_exact_discards(self):
        state, _model = tiny_text_state()
        model, config, report = moss_text_encoder.build_text_encoder(state, torch.bfloat16)
        self.assertEqual(report["language_prefix"], "model.language_model.")
        self.assertEqual(sorted(report["discarded_subgraphs"]), ["lm_head", "model.visual", "mtp"])
        self.assertIsInstance(model.embed_tokens, moss_quant.QuantizedEmbedding)
        self.assertIsInstance(model.layers[0].linear_attn.in_proj_qkv, QuantizedLinear)
        self.assertIsInstance(model.layers[0].linear_attn.in_proj_a, torch.nn.Linear)
        self.assertNotIsInstance(model.layers[0].linear_attn.in_proj_a, QuantizedLinear)
        self.assertEqual(model.layers[0].linear_attn.A_log.dtype, torch.float32)
        self.assertEqual(model.layers[0].linear_attn.norm.weight.dtype, torch.float32)
        self.assertEqual(model.layers[0].input_layernorm.weight.dtype, torch.bfloat16)
        self.assertEqual(model.layers[0].linear_attn.in_proj_qkv.weight.dtype, torch.int8)
        with torch.no_grad():
            out = model(torch.tensor([[1, 2, 3, 4, 5]]))
        self.assertTrue(torch.isfinite(out.float()).all())
        model.to(torch.float32)  # a dtype sweep must not expand quantised storage
        self.assertEqual(model.layers[0].linear_attn.in_proj_qkv.weight.dtype, torch.int8)
        self.assertEqual(model.embed_tokens.weight.dtype, torch.int8)

    def test_int8_encoder_tracks_float_encoder(self):
        state, float_model = tiny_text_state()
        model, _config, _report = moss_text_encoder.build_text_encoder(state, torch.float32)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        with torch.no_grad():
            quantized, reference = model(ids), float_model(ids)
        relative = float((quantized - reference).abs().max() / reference.abs().max())
        self.assertLess(relative, 0.1)

    def test_hf_layout_loads(self):
        state, _model = tiny_text_state(layout="hf", quantize=False)
        model, config, report = moss_text_encoder.build_text_encoder(state, torch.float32)
        self.assertEqual(report["language_prefix"], "language_model.")
        self.assertEqual(config.num_hidden_layers, 4)
        self.assertEqual(config.layer_types[3], "full_attention")

    def test_unknown_or_ambiguous_tensors_are_refused(self):
        state, _model = tiny_text_state()
        for extra in ({"model.something.weight": torch.zeros(1)}, {"visual.merger.unknown": torch.zeros(1)},
                      {"lm_head.bias": torch.zeros(1)}, {"language_model.norm.weight": torch.zeros(32)}):
            with self.subTest(extra=list(extra)), self.assertRaises(ValueError):
                moss_text_encoder.build_text_encoder({**state, **extra}, torch.float32)
        missing = dict(state)
        missing.pop("model.language_model.layers.2.mlp.up_proj.weight")
        missing.pop("model.language_model.layers.2.mlp.up_proj.weight_scale")
        missing.pop("model.language_model.layers.2.mlp.up_proj.comfy_quant")
        with self.assertRaises((RuntimeError, ValueError)):
            moss_text_encoder.build_text_encoder(missing, torch.float32)

    def test_non_qwen35_is_not_classified(self):
        state, _model = tiny_text_state(quantize=False)
        self.assertEqual(moss_text_encoder.classify_text_encoder(state.keys()), "qwen3_5")
        no_linear = {k: v for k, v in state.items() if ".linear_attn." not in k}
        self.assertEqual(moss_text_encoder.classify_text_encoder(no_linear.keys()), "unknown")

    def test_mxfp8_layer_is_refused_for_moss(self):
        state, _model = tiny_text_state(quantize=False)
        key = "model.language_model.layers.0.mlp.up_proj"
        state[f"{key}.weight"] = torch.zeros(48, 32, dtype=torch.float8_e4m3fn)
        state[f"{key}.weight_scale"] = torch.zeros(128, 4, dtype=torch.uint8)
        state[f"{key}.comfy_quant"] = comfy_quant.encode_marker({"format": "mxfp8"})
        with self.assertRaises(UnsupportedQuantization):
            moss_text_encoder.build_text_encoder(state, torch.float32)


class DiTLoadingTests(unittest.TestCase):
    def test_strict_refusals(self):
        state = tiny_dit_state()
        cases = {
            "unknown extra": {**state, "layers.0.extra.weight": torch.zeros(1)},
            "mixed prefix": {**state, "diffusion_model.layers.0.norm1.weight": torch.zeros(64)},
            "fused and split": {**state, "layers.0.proj.q.weight": torch.zeros(64, 64)},
            "wrong shape": {**state, "layers.1.proj.o.weight": torch.zeros(64, 32)},
            "rope bands": {**state, "rope.bands": torch.ones(2)},
            "unknown attention": {**state, "layers.0.comfy_attention.config": torch.tensor(list(b'{"attention": "x"}'), dtype=torch.uint8)},
            "marker on non-block": {**state, "text_embedder.comfy_attention.config": torch.tensor(list(b'{"attention": "comfy_kitchen_int8"}'), dtype=torch.uint8)},
        }
        for name, case in cases.items():
            with self.subTest(case=name), self.assertRaises((ValueError, RuntimeError)):
                moss_pipeline.load_moss_transformer(None, torch.float32, state=case)
        missing = dict(state)
        missing.pop("video_head_modulation.weight")
        with self.assertRaises(ValueError):
            moss_pipeline.load_moss_transformer(None, torch.float32, state=missing)
        gapped = {k: v for k, v in state.items() if not k.startswith("layers.1.")}
        with self.assertRaises(ValueError):
            moss_pipeline.load_moss_transformer(None, torch.float32, state=gapped)

    def test_prefixed_checkpoint_loads(self):
        state = {f"model.diffusion_model.{k}": v for k, v in tiny_dit_state().items()}
        _model, _config, report = moss_pipeline.load_moss_transformer(None, torch.float32, state=state)
        self.assertEqual(report["layout"]["prefix"], "model.diffusion_model.")

    def test_int8_convrot_dit_keeps_storage_and_reports_attention(self):
        state = tiny_dit_state()
        layers = [f"layers.{i}.proj.{n}" for i in range(3) for n in ("qkv", "o", "ffn_up_gate", "ffn_down")]
        state = quantize_layers(state, layers)
        marker = torch.tensor(list(b'{"attention": "comfy_kitchen_int8"}'), dtype=torch.uint8)
        for index in range(3):
            state[f"layers.{index}.comfy_attention.config"] = marker
        model, _config, report = moss_pipeline.load_moss_transformer(None, torch.bfloat16, state=state)
        self.assertEqual(len(report["quantized_layers"]), 12)
        self.assertEqual(report["attention_markers"], {"blocks": 3, "methods": ["comfy_kitchen_int8"]})
        self.assertEqual(model.layers[0].proj["qkv"].weight.dtype, torch.int8)
        self.assertEqual(model.layers[0].proj["qkv"].scale_weight.dtype, torch.float32)
        latents = torch.randn(1, 8, 2, 2)
        context = torch.randn(1, 3, 40)
        with torch.no_grad():
            out = model(latents, torch.tensor([500.0]), context)
        self.assertTrue(torch.isfinite(out.float()).all())

    def test_mixed_raw_and_scaled_fp8_match_dequantised_reference(self):
        state = tiny_dit_state()
        reference = dict(state)
        scale = torch.tensor(0.05)
        scaled = "layers.0.proj.o"
        raw = "layers.1.proj.ffn_down"
        stored = (state[f"{scaled}.weight"] / scale).to(torch.float8_e4m3fn)
        state[f"{scaled}.weight"] = stored
        state[f"{scaled}.weight_scale"] = scale
        state[f"{scaled}.comfy_quant"] = comfy_quant.encode_marker({"format": "float8_e4m3fn"})
        reference[f"{scaled}.weight"] = (stored.to(torch.float32) * scale).to(torch.float32)
        raw_stored = state[f"{raw}.weight"].to(torch.float8_e5m2)
        state[f"{raw}.weight"] = raw_stored
        reference[f"{raw}.weight"] = raw_stored.to(torch.float32)
        model, _c, report = moss_pipeline.load_moss_transformer(None, torch.float32, state=state)
        ref_model, _c2, _r2 = moss_pipeline.load_moss_transformer(None, torch.float32, state=reference)
        self.assertEqual(report["raw_fp8_layers"], [raw])
        self.assertIsInstance(model.layers[1].proj["ffn_down"], moss_quant.RawFP8Linear)
        latents, context = torch.randn(1, 8, 2, 4), torch.randn(1, 5, 40)
        with torch.no_grad():
            a = model(latents, torch.tensor([300.0]), context)
            b = ref_model(latents, torch.tensor([300.0]), context)
        self.assertTrue(torch.equal(a, b))

    def test_malformed_fp8_is_refused(self):
        state = tiny_dit_state()
        bad = dict(state)
        bad["layers.0.proj.o.weight"] = state["layers.0.proj.o.weight"].to(torch.float8_e4m3fn)
        bad["layers.0.proj.o.weight_scale"] = torch.ones(64, 1)  # row scales without a marker: undefined
        with self.assertRaises(UnsupportedQuantization):
            moss_pipeline.load_moss_transformer(None, torch.float32, state=bad)
        norm = dict(state)
        norm["layers.0.norm1.weight"] = state["layers.0.norm1.weight"].to(torch.float8_e4m3fn)
        with self.assertRaises(ValueError):
            moss_pipeline.load_moss_transformer(None, torch.float32, state=norm)

    def test_variant_fingerprint_is_content_not_name(self):
        state = tiny_dit_state()
        variant, digest = moss_model.variant_fingerprint(state)
        self.assertEqual(variant, "unrecognised")
        self.assertEqual(len(digest), 64)
        original = dict(moss_model.MOSS_VARIANT_FINGERPRINTS)
        try:
            moss_model.MOSS_VARIANT_FINGERPRINTS[digest] = "sft"
            self.assertEqual(moss_model.variant_fingerprint(state)[0], "sft")
        finally:
            moss_model.MOSS_VARIANT_FINGERPRINTS.clear()
            moss_model.MOSS_VARIANT_FINGERPRINTS.update(original)


class LoraTests(unittest.TestCase):
    def setUp(self):
        self.state = tiny_dit_state()
        self.config = moss_model.infer_moss_config(self.state)

    def _write(self, root, tensors, name="lora.safetensors"):
        path = Path(root) / name
        save_file(tensors, str(path))
        return path

    def test_qkv_slices_patch_the_fused_rows(self):
        generator = torch.Generator().manual_seed(4)
        down_q, up_q = torch.randn(4, 64, generator=generator), torch.randn(64, 4, generator=generator)
        down_v, up_v = torch.randn(2, 64, generator=generator), torch.randn(64, 2, generator=generator)
        with tempfile.TemporaryDirectory() as root:
            path = self._write(root, {
                "diffusion_model.layers.1.proj.q.lora_down.weight": down_q,
                "diffusion_model.layers.1.proj.q.lora_up.weight": up_q,
                "diffusion_model.layers.1.proj.q.alpha": torch.tensor(2.0),
                "transformer.layers.1.proj.v.lora_A.weight": down_v,
                "transformer.layers.1.proj.v.lora_B.weight": up_v,
            })
            model, _c, report = moss_pipeline.load_moss_transformer(None, torch.float32, loras=[(path, 0.75)], state=dict(self.state))
        weight = model.layers[1].proj["qkv"].weight
        base = self.state["layers.1.proj.qkv.weight"]
        expected = base.clone()
        expected[0:64] += 0.75 * (up_q @ (down_q * (2.0 / 4)))
        expected[128:192] += 0.75 * (up_v @ down_v)
        self.assertLessEqual(float((weight - expected).abs().max()), 1e-5)
        self.assertTrue(torch.equal(weight[64:128], base[64:128]))
        self.assertEqual(report["loras"][0]["fused_modules"], 1)

    def test_quantised_targets_carry_an_adapter(self):
        state = quantize_layers(self.state, ["layers.0.proj.o"])
        generator = torch.Generator().manual_seed(5)
        down, up = torch.randn(3, 64, generator=generator), torch.randn(64, 3, generator=generator)
        with tempfile.TemporaryDirectory() as root:
            path = self._write(root, {"lora_unet_layers_0_proj_o.lora_down.weight": down,
                                      "lora_unet_layers_0_proj_o.lora_up.weight": up})
            model, _c, report = moss_pipeline.load_moss_transformer(None, torch.float32, loras=[(path, -0.5)], state=state)
        layer = model.layers[0].proj["o"]
        self.assertTrue(layer.has_lora_adapter)
        self.assertFalse(layer.uses_integer_matmul)
        x = torch.randn(3, 64)
        expected = torch.nn.functional.linear(x, layer.dequantized_weight(torch.float32)) + (x @ (down * -0.5).T) @ up.T
        self.assertLessEqual(float((layer(x) - expected).abs().max()), 1e-5)
        self.assertEqual(report["loras"][0]["adapted_modules"], 1)

    def test_refusals(self):
        cases = {
            "both fused and slice": {"layers.0.proj.qkv.lora_A.weight": torch.zeros(1, 64), "layers.0.proj.qkv.lora_B.weight": torch.zeros(192, 1),
                                     "layers.0.proj.k.lora_A.weight": torch.zeros(1, 64), "layers.0.proj.k.lora_B.weight": torch.zeros(64, 1)},
            "text encoder": {"lora_te_layers_0.lora_down.weight": torch.zeros(1, 4), "lora_te_layers_0.lora_up.weight": torch.zeros(4, 1)},
            "dora": {"layers.0.proj.o.lora_A.weight": torch.zeros(1, 64), "layers.0.proj.o.lora_B.weight": torch.zeros(64, 1),
                     "layers.0.proj.o.dora_scale": torch.zeros(64)},
            "unknown target": {"layers.0.attn.to_q.lora_A.weight": torch.zeros(1, 64), "layers.0.attn.to_q.lora_B.weight": torch.zeros(64, 1)},
            "unpaired": {"layers.0.proj.o.lora_A.weight": torch.zeros(1, 64)},
            "wrong width": {"layers.0.proj.q.lora_A.weight": torch.zeros(1, 64), "layers.0.proj.q.lora_B.weight": torch.zeros(192, 1)},
        }
        for name, tensors in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                converted = moss_model.convert_moss_lora_state_dict(tensors, self.config, name)
                for target, entries in converted.items():
                    module_shape = {"layers.0.proj.qkv": (192, 64)}.get(target, (64, 64))
                    moss_model.lora_pair_for(entries, *module_shape)

    def test_zero_multiplier_is_not_mounted(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write(root, {"layers.0.proj.o.lora_A.weight": torch.ones(1, 64), "layers.0.proj.o.lora_B.weight": torch.ones(64, 1)})
            model, _c, report = moss_pipeline.load_moss_transformer(None, torch.float32, loras=[(path, 0.0)], state=dict(self.state))
        self.assertEqual(report["loras"], [])
        self.assertTrue(torch.equal(model.layers[0].proj["o"].weight, self.state["layers.0.proj.o.weight"]))


class LokrTests(unittest.TestCase):
    """LyCORIS LoKr on the DiT: ``delta = scale · (w1 ⊗ w2)``, fused or carried beside the weight."""

    # (w1, w2) shapes per tiny-DiT target. a ≠ b and c ≠ d everywhere, so a transposed or reordered
    # product cannot pass for the right one.
    SHAPES = {"proj.qkv": ((4, 2), (48, 32)), "proj.o": ((2, 4), (32, 16)),
              "proj.ffn_up_gate": ((3, 4), (64, 16)), "proj.ffn_down": ((4, 3), (16, 32))}
    # What LyCORIS writes as the alpha of a full-matrix LoKr (the real Nyalia file carries it).
    FULL_ALPHA = 9999220736.0

    def setUp(self):
        self.state = tiny_dit_state()
        self.config = moss_model.infer_moss_config(self.state)
        self.generator = torch.Generator().manual_seed(8)

    def _randn(self, *shape, scale=1.0):
        return torch.randn(*shape, generator=self.generator) * scale

    def _factors(self, block):
        return {f"layers.{block}.{name}": (self._randn(*a), self._randn(*b, scale=0.05)) for name, (a, b) in self.SHAPES.items()}

    def _write(self, root, factors, name="lokr.safetensors"):
        tensors = {}
        for module, (w1, w2) in factors.items():
            tensors[f"diffusion_model.{module}.lokr_w1"] = w1
            tensors[f"diffusion_model.{module}.lokr_w2"] = w2
            tensors[f"diffusion_model.{module}.alpha"] = torch.tensor(self.FULL_ALPHA)
        path = Path(root) / name
        save_file(tensors, str(path))
        return path

    def _close(self, actual, expected, tolerance=1e-4):
        self.assertLessEqual(float((actual - expected).abs().max()), tolerance)

    def test_the_adapter_output_is_the_kronecker_product(self):
        w1, w2 = self._randn(3, 2), self._randn(5, 7)
        x = self._randn(2, 4, 14)
        self._close(kron_adapter_output(x, w1, w2), x @ torch.kron(w1, w2).T, 1e-5)

    def test_quantised_targets_match_a_model_with_the_products_fused(self):
        layers = [f"layers.0.{name}" for name in self.SHAPES]
        state = quantize_layers(self.state, layers)
        factors = self._factors(0)
        with tempfile.TemporaryDirectory() as root:
            path = self._write(root, factors)
            model, _c, report = moss_pipeline.load_moss_transformer(None, torch.float32, loras=[(path, 0.8)], state=dict(state))
        plain, _c2, _r2 = moss_pipeline.load_moss_transformer(None, torch.float32, state=dict(state))
        reference = dict(self.state)
        for module, (w1, w2) in factors.items():
            layer = model.get_submodule(module)
            self.assertTrue(layer.has_lora_adapter)
            self.assertFalse(layer.uses_integer_matmul)
            weight = plain.get_submodule(module).dequantized_weight(torch.float32) + 0.8 * torch.kron(w1, w2)
            x = self._randn(5, layer.in_features)
            self._close(layer(x), torch.nn.functional.linear(x, weight, layer.bias))
            reference[f"{module}.weight"] = weight
        self.assertEqual(report["loras"][0]["adapted_modules"], 4)
        self.assertEqual(report["loras"][0]["fused_modules"], 0)
        self.assertEqual(report["adapted_layers"], sorted(layers))
        ref_model, _c3, _r3 = moss_pipeline.load_moss_transformer(None, torch.float32, state=reference)
        latents, context = torch.randn(1, 8, 2, 4), torch.randn(1, 5, 40)
        with torch.no_grad():
            self._close(model(latents, torch.tensor([300.0]), context), ref_model(latents, torch.tensor([300.0]), context))

    def test_full_precision_targets_fuse_slices_and_low_rank_factors(self):
        w1q, w2q = self._randn(4, 2), self._randn(48, 32)
        w1k, w2k = self._randn(2, 4), self._randn(32, 16)
        w1o, w2a, w2b = self._randn(2, 2), self._randn(32, 3), self._randn(3, 32)
        tensors = {
            "diffusion_model.layers.1.proj.qkv.lokr_w1": w1q, "diffusion_model.layers.1.proj.qkv.lokr_w2": w2q,
            "diffusion_model.layers.1.proj.qkv.alpha": torch.tensor(self.FULL_ALPHA),
            "lycoris_layers_2_proj_k.lokr_w1": w1k, "lycoris_layers_2_proj_k.lokr_w2": w2k,
            "layers.0.proj.o.lokr_w1": w1o, "layers.0.proj.o.lokr_w2_a": w2a, "layers.0.proj.o.lokr_w2_b": w2b,
            "layers.0.proj.o.alpha": torch.tensor(6.0),
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "lokr.safetensors"
            save_file(tensors, str(path))
            model, _c, report = moss_pipeline.load_moss_transformer(None, torch.float32, loras=[(path, 0.5)], state=dict(self.state))
        self._close(model.layers[1].proj["qkv"].weight, self.state["layers.1.proj.qkv.weight"] + 0.5 * torch.kron(w1q, w2q), 1e-5)
        fused_k = model.layers[2].proj["qkv"].weight
        base_k = self.state["layers.2.proj.qkv.weight"]
        self._close(fused_k[64:128], base_k[64:128] + 0.5 * torch.kron(w1k, w2k), 1e-5)
        self.assertTrue(torch.equal(fused_k[:64], base_k[:64]))
        self.assertTrue(torch.equal(fused_k[128:], base_k[128:]))
        # A low-rank second factor takes alpha / rank, as ComfyUI scales it.
        self._close(model.layers[0].proj["o"].weight, self.state["layers.0.proj.o.weight"] + 0.5 * (6.0 / 3) * torch.kron(w1o, w2a @ w2b), 1e-5)
        self.assertEqual(report["loras"][0]["fused_modules"], 3)

    def test_a_quantised_slice_changes_only_its_rows(self):
        state = quantize_layers(self.state, ["layers.0.proj.qkv"])
        w1, w2 = self._randn(2, 4), self._randn(32, 16)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "lokr.safetensors"
            save_file({"layers.0.proj.v.lokr_w1": w1, "layers.0.proj.v.lokr_w2": w2}, str(path))
            model, _c, _report = moss_pipeline.load_moss_transformer(None, torch.float32, loras=[(path, 1.0)], state=state)
        layer = model.layers[0].proj["qkv"]
        x = self._randn(3, 64)
        base = torch.nn.functional.linear(x, layer.dequantized_weight(torch.float32), layer.bias)
        out = layer(x)
        self.assertTrue(torch.equal(out[:, :128], base[:, :128]))
        self._close(out[:, 128:], base[:, 128:] + x @ torch.kron(w1, w2).T, 1e-5)

    def test_a_lora_and_a_lokr_stack_on_one_quantised_layer(self):
        state = quantize_layers(self.state, ["layers.0.proj.o"])
        down, up = self._randn(3, 64), self._randn(64, 3)
        w1, w2 = self._randn(2, 4), self._randn(32, 16)
        with tempfile.TemporaryDirectory() as root:
            lora, lokr = Path(root) / "lora.safetensors", Path(root) / "lokr.safetensors"
            save_file({"layers.0.proj.o.lora_A.weight": down, "layers.0.proj.o.lora_B.weight": up}, str(lora))
            save_file({"layers.0.proj.o.lokr_w1": w1, "layers.0.proj.o.lokr_w2": w2}, str(lokr))
            model, _c, report = moss_pipeline.load_moss_transformer(None, torch.float32, loras=[(lora, -0.5), (lokr, 0.7)], state=state)
        layer = model.layers[0].proj["o"]
        x = self._randn(4, 64)
        weight = layer.dequantized_weight(torch.float32) + 0.7 * torch.kron(w1, w2)
        self._close(layer(x), torch.nn.functional.linear(x, weight, layer.bias) + (x @ (down * -0.5).T) @ up.T)
        self.assertEqual([entry["adapted_modules"] for entry in report["loras"]], [1, 1])

    def test_a_raw_fp8_target_carries_the_product(self):
        state = dict(self.state)
        state["layers.1.proj.ffn_down.weight"] = self.state["layers.1.proj.ffn_down.weight"].to(torch.float8_e5m2)
        w1, w2 = self._randn(4, 3), self._randn(16, 32)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "lokr.safetensors"
            save_file({"layers.1.proj.ffn_down.lokr_w1": w1, "layers.1.proj.ffn_down.lokr_w2": w2}, str(path))
            model, _c, report = moss_pipeline.load_moss_transformer(None, torch.float32, loras=[(path, 1.0)], state=state)
        layer = model.layers[1].proj["ffn_down"]
        self.assertIsInstance(layer, moss_quant.RawFP8Linear)
        x = self._randn(3, 96)
        weight = state["layers.1.proj.ffn_down.weight"].to(torch.float32) + torch.kron(w1, w2)
        self._close(layer(x), torch.nn.functional.linear(x, weight, layer.bias))
        self.assertEqual(report["adapted_layers"], ["layers.1.proj.ffn_down"])

    def test_refusals(self):
        z = torch.zeros
        o = "layers.0.proj.o"
        cases = {
            "mixed families": {f"{o}.lokr_w1": z(2, 4), f"{o}.lokr_w2": z(32, 16), f"{o}.lora_down.weight": z(1, 64), f"{o}.lora_up.weight": z(64, 1)},
            "decomposed first factor": {f"{o}.lokr_w1_a": z(2, 1), f"{o}.lokr_w1_b": z(1, 4), f"{o}.lokr_w2": z(32, 16)},
            "tucker": {f"{o}.lokr_w1": z(2, 4), f"{o}.lokr_w2_a": z(32, 1), f"{o}.lokr_w2_b": z(1, 16), f"{o}.lokr_t2": z(1, 1, 1, 1)},
            "both second factors": {f"{o}.lokr_w1": z(2, 4), f"{o}.lokr_w2": z(32, 16), f"{o}.lokr_w2_a": z(32, 1), f"{o}.lokr_w2_b": z(1, 16)},
            "no first factor": {f"{o}.lokr_w2": z(32, 16)},
            "low-rank mismatch": {f"{o}.lokr_w1": z(2, 4), f"{o}.lokr_w2_a": z(32, 3), f"{o}.lokr_w2_b": z(2, 16)},
            "non-finite alpha": {f"{o}.lokr_w1": z(2, 4), f"{o}.lokr_w2": z(32, 16), f"{o}.alpha": torch.tensor(float("nan"))},
            "product too narrow": {f"{o}.lokr_w1": z(2, 4), f"{o}.lokr_w2": z(32, 8)},
            "slice product too tall": {"layers.0.proj.k.lokr_w1": z(4, 2), "layers.0.proj.k.lokr_w2": z(48, 32)},
            "fused and slice": {"layers.0.proj.qkv.lokr_w1": z(4, 2), "layers.0.proj.qkv.lokr_w2": z(48, 32),
                                "layers.0.proj.k.lokr_w1": z(2, 4), "layers.0.proj.k.lokr_w2": z(32, 16)},
        }
        for name, tensors in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                converted = moss_model.convert_moss_lora_state_dict(tensors, self.config, name)
                for target, entries in converted.items():
                    rows, cols = {"layers.0.proj.qkv": (192, 64)}.get(target, (64, 64))
                    moss_model.lokr_terms_for([e for e in entries if e["kind"] == "lokr"], rows, cols)
                    pairs = [e for e in entries if e["kind"] == "lora"]
                    if pairs:
                        moss_model.lora_pair_for(pairs, rows, cols)


class ComponentFileTests(unittest.TestCase):
    def test_hf_shards_load_and_mismatches_are_refused(self):
        state = tiny_dit_state(fused=False)
        keys = sorted(state)
        half = len(keys) // 2
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            save_file({k: state[k].contiguous() for k in keys[:half]}, str(root / "diffusion_pytorch_model-00001-of-00002.safetensors"))
            save_file({k: state[k].contiguous() for k in keys[half:]}, str(root / "diffusion_pytorch_model-00002-of-00002.safetensors"))
            mapping = {k: ("diffusion_pytorch_model-00001-of-00002.safetensors" if i < half else "diffusion_pytorch_model-00002-of-00002.safetensors")
                       for i, k in enumerate(keys)}
            index = root / "diffusion_pytorch_model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": mapping}), encoding="utf-8")
            loaded, source = moss_pipeline.load_component_state(root)
            self.assertTrue(source["sharded"])
            self.assertEqual(set(loaded), set(keys))
            _model, _config, report = moss_pipeline.load_moss_transformer(index, torch.float32)
            self.assertEqual(report["layout"]["source_qkv"], "split")
            wrong = dict(mapping)
            wrong[keys[0]] = "diffusion_pytorch_model-00002-of-00002.safetensors"
            index.write_text(json.dumps({"weight_map": wrong}), encoding="utf-8")
            with self.assertRaises(ValueError):
                moss_pipeline.load_component_state(index)
            escape = {keys[0]: "../outside.safetensors"}
            index.write_text(json.dumps({"weight_map": escape}), encoding="utf-8")
            with self.assertRaises((ValueError, FileNotFoundError)):
                moss_pipeline.load_component_state(index)

    def test_selective_text_encoder_read_skips_discarded_subgraphs(self):
        state, _model = tiny_text_state()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "te.safetensors"
            save_file({k: v.contiguous() for k, v in state.items()}, str(path))
            loaded, source = moss_pipeline.load_component_state(path, keep=moss_pipeline._keep_text_encoder_key)
            self.assertEqual(source["skipped_tensors"], 8)
            self.assertFalse(any(k.startswith(("lm_head", "mtp", "model.visual")) for k in loaded))
            full = moss_pipeline.moss_component_bytes(path, "text_encoder")
            expected = 0
            for key, value in loaded.items():
                if key.endswith(".comfy_quant"):
                    continue
                if value.dtype in (torch.int8,):
                    expected += value.numel()
                elif value.dtype == torch.float32 and key.endswith(("A_log", "linear_attn.norm.weight", "weight_scale")):
                    expected += value.numel() * 4
                else:
                    expected += value.numel() * 2
            self.assertEqual(full, expected)

    def test_vae_signature(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)

            def write(name, width, latent=32, bn=True):
                tensors = {"decoder.conv_in.weight": torch.zeros(width, latent, 3, 3)}
                if bn:
                    tensors.update({"bn.running_mean": torch.zeros(128), "bn.running_var": torch.ones(128)})
                path = root / name
                save_file(tensors, str(path))
                return path

            self.assertEqual(moss_pipeline.vae_signature(write("full.safetensors", 512))["signature"], "flux2_full_decoder")
            with self.assertRaises(ValueError):
                moss_pipeline.vae_signature(write("small.safetensors", 384))
            with self.assertRaises(ValueError):
                moss_pipeline.vae_signature(write("nobn.safetensors", 512, bn=False))
            with self.assertRaises(ValueError):
                moss_pipeline.vae_signature(write("wan.safetensors", 512, latent=16))
            gguf = root / "vae.gguf"
            gguf.write_bytes(b"GGUF")
            with self.assertRaises(ValueError):
                moss_pipeline.vae_signature(gguf)


class StubTokenizer:
    vocab_size = 300

    def encode(self, prompt):
        ids = [1 + (ord(c) % 250) for c in prompt.strip()][:20] or [2]
        ids = [3, 4] + ids + [5]
        return ids, {"token_count": len(ids), "untruncated_token_count": len(ids), "truncated": False, "max_tokens": 4096,
                     "template_tokens": 3, "prompt_trimmed": False, "escapes_removed": False, "segments": 1, "weighted_token_count": 0}


def tiny_vae():
    from diffusers import AutoencoderKLFlux2

    config = dict(moss_pipeline.FLUX2_VAE_CONFIG)
    config.update(block_out_channels=(16, 16, 16, 16), layers_per_block=1, norm_num_groups=8)
    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(**config).eval()
    statistics = {"bn.running_mean": torch.randn(128) * 0.1, "bn.running_var": torch.rand(128) + 0.5}
    return vae, statistics


def tiny_runtime():
    # The FLUX.2 VAE packs 32 x 2 x 2 = 128 channels, so the runtime fixture DiT is 128 wide in latents.
    transformer, config, report = moss_pipeline.load_moss_transformer(None, torch.float32, state=tiny_dit_state(latent_channels=128))
    text_state, _ = tiny_text_state(quantize=False)
    # The tiny DiT reads 40-wide conditioning; the tiny encoder emits 32. Project with a fixed matrix
    # inside a wrapper so the runtime contract (hidden states in, DiT conditioning out) is unchanged.
    text_model, text_config, text_report = moss_text_encoder.build_text_encoder(text_state, torch.float32)
    projector = torch.nn.Linear(32, 40, bias=False)
    randomize(projector, 9)

    class Projected(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = text_model
            self.embed_tokens = text_model.embed_tokens
            self.layers = text_model.layers
            self.norm = text_model.norm
            self.proj = projector

        def forward(self, ids, device=None, layer_hook=None):
            return self.proj(self.inner(ids, device=device, layer_hook=layer_hook))

    vae, statistics = tiny_vae()
    return moss_pipeline.MossRuntime(transformer, Projected().eval(), vae, StubTokenizer(), torch.float32, config, text_config,
                                     statistics, device="cpu", reports={"transformer": report, "text_encoder": text_report})


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = tiny_runtime()

    def _generate(self, **overrides):
        # The fixture DiT has 3 blocks, so the default skip index 22 does not exist; block 1 stands in.
        args = dict(prompt="a fox", negative_prompt="", width=32, height=32, steps=4, cfg=3.0, sampler="euler",
                    scheduler="sgm_uniform", generators=[torch.Generator().manual_seed(42)], moss={"uncond_skip_layers": [1]})
        args.update(overrides)
        return self.runtime.generate_batch(**args)

    def test_sampling_matches_hand_written_comfy_loop(self):
        captured = {}

        def on_step(step, total, latents):
            if step == total:
                captured["latents"] = latents.clone()

        images = self._generate(on_step=on_step, moss={"uncond_skip_layers": [1]})
        self.assertEqual(images[0].size, (32, 32))
        runtime = self.runtime
        pos = runtime._encode("a fox").embeddings
        neg = runtime._encode("").embeddings
        sigmas = moss_sampling.moss_sigma_schedule(4, "sgm_uniform", 5.0)
        x = torch.randn((1, 128, 2, 2), generator=torch.Generator().manual_seed(42), dtype=torch.float32) * float(sigmas[0])
        with torch.no_grad():
            for i in range(4):
                s, n = sigmas[i], sigmas[i + 1]
                t = (s * torch.ones(1) * 1000).float()
                vc = runtime.transformer(x, t, pos, torch.ones(pos.shape[:2], dtype=torch.bool)).float()
                vu = runtime.transformer(x, t, neg, torch.ones(neg.shape[:2], dtype=torch.bool), skip_layers=frozenset({1})).float()
                dc, du = x - vc * s, x - vu * s
                denoised = du + (dc - du) * 3.0
                x = x + (x - denoised) / s * (n - s)
        self.assertTrue(torch.equal(captured["latents"], x))
        metrics = runtime.last_generation_metrics["sampling"]
        self.assertEqual(metrics["actual_transformer_invocations"], 8)
        self.assertEqual(metrics["uncond_skip_invocations"], 4)
        self.assertEqual(metrics["uncond_skip_layers"], [1])
        self.assertEqual(metrics["shift"], 5.0)

    def test_seed_reproducibility_and_cfg_one(self):
        a = self._generate(generators=[torch.Generator().manual_seed(7)])
        b = self._generate(generators=[torch.Generator().manual_seed(7)])
        self.assertEqual(a[0].tobytes(), b[0].tobytes())
        self._generate(cfg=1.0)
        metrics = self.runtime.last_generation_metrics["sampling"]
        self.assertEqual(metrics["actual_transformer_invocations"], 4)
        self.assertEqual(metrics["cfg_branches"], 1)
        self.assertFalse(metrics["uncond_skip_active"])

    def test_batch_uses_one_generator_per_image(self):
        # Each image draws its noise from its own generator, so image 2 of a batch starts exactly where
        # seed 2 starts alone.
        sigma = torch.tensor(0.7)
        batch_noise = self.runtime._initial_latents([torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)], 32, 32, sigma)
        alone_noise = self.runtime._initial_latents([torch.Generator().manual_seed(2)], 32, 32, sigma)
        self.assertTrue(torch.equal(batch_noise[1:2], alone_noise))
        self.assertFalse(torch.equal(batch_noise[0:1], alone_noise))
        # Sampled one image per chunk, the whole run is the single run, pixel for pixel.
        chunked = self._generate(generators=[torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)], sampling_batch_size=1)
        alone = self._generate(generators=[torch.Generator().manual_seed(2)])
        self.assertEqual(chunked[1].tobytes(), alone[0].tobytes())
        self.assertNotEqual(chunked[0].tobytes(), alone[0].tobytes())
        # A two-image chunk runs the timestep MLP and the modulation projections on two rows at once,
        # as the official transformer does; a CPU GEMM may round those differently from one row, so
        # the batched image is close to the single one but is not promised to be bit-identical.
        batched = self._generate(generators=[torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)])
        difference = torch.tensor(list(batched[1].tobytes()), dtype=torch.int16) - torch.tensor(list(alone[0].tobytes()), dtype=torch.int16)
        self.assertLessEqual(int(difference.abs().max()), 2)

    def test_conditioning_cache_and_diagnostics(self):
        self.runtime._conditioning_cache.clear()
        self.runtime._encode("cache me")
        self.assertFalse(self.runtime.last_conditioning_reused)
        self.runtime._encode("cache me")
        self.assertTrue(self.runtime.last_conditioning_reused)
        self.assertEqual(self.runtime._encode("cache me").diagnostics["placement"], "cpu")

    def test_refusals(self):
        with self.assertRaises(ValueError):
            self._generate(guidance="pag")
        with self.assertRaises(ValueError):
            self._generate(guidance="cfg_zero_star")
        with self.assertRaises(ValueError):
            self._generate(moss={"standard_slg": True})
        with self.assertRaises(ValueError):
            self._generate(width=40)
        with self.assertRaises(ValueError):
            self._generate(sampler="dpmpp_2m")
        with self.assertRaises(ValueError):
            self._generate(moss={"uncond_skip_layers": [3]})
        # USDU's shared inputs are not prepared: MOSS does not run the tiled path.
        with self.assertRaises(ValueError):
            self.runtime.prepare_refinement_conditioning("a fox", "", 3.0, "none")
        with self.assertRaises(ValueError):
            self.runtime.prepare_refinement_sigmas(4, 0.5, "sgm_uniform")

    def test_capabilities_and_sizes(self):
        self._generate()
        capabilities = self.runtime.last_generation_metrics["capabilities"]["components"]
        self.assertEqual([c["role"] for c in capabilities], ["diffusion_model", "text_encoder", "vae"])
        self.assertEqual(capabilities[2]["compute_dtype"], "float32")
        for key in ("transformer", "text_encoder", "vae", "total", "transformer_max_block", "text_encoder_max_layer"):
            self.assertGreater(self.runtime.weight_sizes[key], 0)


def rgb_image(width, height, seed):
    from PIL import Image

    pixels = torch.randint(0, 256, (height, width, 3), generator=torch.Generator().manual_seed(seed), dtype=torch.uint8)
    return Image.frombytes("RGB", (width, height), pixels.numpy().tobytes())


def mask_image(width, height, columns):
    """White (redraw) on the pixel columns in ``columns``, black (keep) elsewhere."""
    from PIL import Image

    values = torch.zeros((height, width), dtype=torch.uint8)
    values[:, columns] = 255
    return Image.frombytes("L", (width, height), values.numpy().tobytes())


class RefinementScheduleTests(unittest.TestCase):
    def test_partial_denoise_is_comfys_suffix_of_the_longer_schedule(self):
        for scheduler in ("sgm_uniform", "simple", "karras", "normal"):
            with self.subTest(scheduler=scheduler):
                sigmas, facts = moss_sampling.moss_refinement_sigma_schedule(20, 0.4, scheduler, 5.0)
                self.assertTrue(torch.equal(sigmas, moss_sampling.moss_sigma_schedule(50, scheduler, 5.0)[-21:]))
                self.assertEqual(facts, {"schedule_steps": 50, "schedule_mode": "comfy_suffix", "start_sigma": float(sigmas[0])})
                self.assertEqual(float(sigmas[-1]), 0.0)
        # `int(steps / denoise)` truncates exactly as KSampler.set_steps does: 7 / 0.3 = 23.33 -> 23.
        _sigmas, facts = moss_sampling.moss_refinement_sigma_schedule(7, 0.3, "sgm_uniform", 5.0)
        self.assertEqual(facts["schedule_steps"], 23)

    def test_full_denoise_is_the_base_schedule(self):
        for denoise in (1.0, 1, 0.99995):
            sigmas, facts = moss_sampling.moss_refinement_sigma_schedule(12, denoise, "sgm_uniform", 3.0)
            self.assertTrue(torch.equal(sigmas, moss_sampling.moss_sigma_schedule(12, "sgm_uniform", 3.0)))
            self.assertEqual(facts["schedule_mode"], "full")

    def test_refusals(self):
        for denoise in (0, 0.0, -0.1, 1.01, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(denoise=denoise), self.assertRaises(ValueError):
                moss_sampling.moss_refinement_sigma_schedule(10, denoise, "sgm_uniform")
        with self.assertRaises(ValueError):
            moss_sampling.moss_refinement_sigma_schedule(100, 0.01, "sgm_uniform")
        with self.assertRaises(ValueError):
            moss_sampling.moss_refinement_sigma_schedule(10, 0.5, "align_your_steps")


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = tiny_runtime()

    def _refine(self, **overrides):
        args = dict(images=[rgb_image(32, 32, 5)], prompt="a fox", negative_prompt="", steps=4, denoise=0.5, cfg=3.0,
                    sampler="euler", scheduler="sgm_uniform", generators=[torch.Generator().manual_seed(42)],
                    moss={"uncond_skip_layers": [1]})
        args.update(overrides)
        return self.runtime.refine_batch(**args)

    def test_encode_packs_and_normalises_as_comfys_flux2_latent_format(self):
        image = rgb_image(32, 32, 11)
        latents = self.runtime._encode_images([image])
        self.assertEqual(tuple(latents.shape), (1, 128, 2, 2))
        vae, statistics = tiny_vae()
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(32, 32, 3).permute(2, 0, 1)
        with torch.no_grad():
            z = vae.encode(pixels.unsqueeze(0).float() / 127.5 - 1.0).latent_dist.mode()
        # `b c (h ph) (w pw) -> b (c ph pw) h w`, then the BN statistics, channel by packed channel.
        std = torch.sqrt(statistics["bn.running_var"].float() + float(moss_pipeline.FLUX2_VAE_CONFIG["batch_norm_eps"]))
        expected = torch.empty_like(latents)
        for c in range(32):
            for ph in range(2):
                for pw in range(2):
                    k = c * 4 + ph * 2 + pw
                    expected[0, k] = (z[0, c, ph::2, pw::2] - statistics["bn.running_mean"][k]) / std[k]
        torch.testing.assert_close(latents, expected, rtol=0, atol=1e-6)
        # And the decode path undoes it exactly, so a redraw starts from the picture it was handed.
        self.assertTrue(torch.allclose(moss_pipeline.unpatchify_flux2_latents(self.runtime._denormalize_latents(latents)), z, atol=1e-5))

    def test_refinement_matches_hand_written_comfy_img2img_loop(self):
        captured = {}

        def on_step(step, total, latents):
            if step == total:
                captured["latents"] = latents.clone()

        image = rgb_image(32, 32, 5)
        refined = self._refine(images=[image], on_step=on_step)
        self.assertEqual(refined[0].size, (32, 32))
        runtime = self.runtime
        source = runtime._encode_images([image])
        pos = runtime._encode("a fox").embeddings
        neg = runtime._encode("").embeddings
        # KSampler with denoise 0.5 builds int(4 / 0.5) = 8 sigmas and keeps the last 5.
        sigmas = moss_sampling.moss_sigma_schedule(8, "sgm_uniform", 5.0)[-5:]
        noise = torch.randn((1, 128, 2, 2), generator=torch.Generator().manual_seed(42), dtype=torch.float32)
        s0 = float(sigmas[0])
        x = (1.0 - s0) * source + s0 * noise
        with torch.no_grad():
            for i in range(4):
                s, n = sigmas[i], sigmas[i + 1]
                t = (s * torch.ones(1) * 1000).float()
                vc = runtime.transformer(x, t, pos, torch.ones(pos.shape[:2], dtype=torch.bool)).float()
                vu = runtime.transformer(x, t, neg, torch.ones(neg.shape[:2], dtype=torch.bool), skip_layers=frozenset({1})).float()
                dc, du = x - vc * s, x - vu * s
                denoised = du + (dc - du) * 3.0
                x = x + (x - denoised) / s * (n - s)
        self.assertTrue(torch.equal(captured["latents"], x))
        metrics = runtime.last_generation_metrics
        sampling = metrics["refinement.sampling"]
        self.assertEqual(sampling["schedule_steps"], 8)
        self.assertEqual(sampling["schedule_mode"], "comfy_suffix")
        self.assertEqual(sampling["start_sigma"], s0)
        self.assertEqual(sampling["executed_denoise_updates"], 4)
        self.assertEqual(sampling["actual_transformer_invocations"], 8)
        self.assertEqual(sampling["uncond_skip_invocations"], 4)
        self.assertFalse(sampling["masked"])
        for stage in ("refinement.prompt_encode", "refinement.negative_prompt_encode", "refinement.vae_encode", "refinement.vae_decode"):
            self.assertIn("seconds", metrics[stage], stage)
        self.assertEqual(metrics["refinement.vae_encode"]["vae_dtype"], "float32")
        self.assertEqual(metrics["refinement.vae_decode"]["actual_vae_mode"], "full")

    def test_refinement_follows_the_moss_block(self):
        self._refine(moss={"shift": 3.0, "uncond_skip_layers": []}, steps=3, denoise=0.6)
        sampling = self.runtime.last_generation_metrics["refinement.sampling"]
        self.assertEqual(sampling["shift"], 3.0)
        self.assertEqual(sampling["uncond_skip_invocations"], 0)
        self.assertEqual(sampling["start_sigma"], float(moss_sampling.moss_sigma_schedule(5, "sgm_uniform", 3.0)[-4]))
        # cfg 1 drops the negative branch, as the base pass does.
        self._refine(cfg=1.0)
        sampling = self.runtime.last_generation_metrics["refinement.sampling"]
        self.assertEqual(sampling["actual_transformer_invocations"], 4)
        self.assertEqual(sampling["cfg_branches"], 1)

    def test_the_pag_slider_alone_changes_nothing(self):
        # The server forwards the request's PAG scale (3.0 by default) with guidance "none"; PAG is
        # decided by the guidance mode, so the redraw is the one without the slider.
        plain = self._refine()
        forwarded = self._refine(guidance="none", pag_scale=3.0, pag_applied_layers="all")
        self.assertEqual(plain[0].tobytes(), forwarded[0].tobytes())

    def test_seed_reproducibility_and_full_denoise(self):
        a = self._refine(generators=[torch.Generator().manual_seed(9)], sampler="euler_ancestral")
        b = self._refine(generators=[torch.Generator().manual_seed(9)], sampler="euler_ancestral")
        self.assertEqual(a[0].tobytes(), b[0].tobytes())
        self._refine(denoise=1.0, sampler="heun")
        sampling = self.runtime.last_generation_metrics["refinement.sampling"]
        self.assertEqual(sampling["schedule_mode"], "full")
        self.assertEqual(sampling["start_sigma"], 1.0)

    def test_mask_keeps_the_unmasked_area_on_the_source(self):
        captured = {}

        def on_step(step, total, latents):
            if step == total:
                captured["latents"] = latents.clone()

        image = rgb_image(64, 32, 3)
        source = self.runtime._encode_images([image])
        # Pixel columns 0-31 are the left latent column pair (16 px per latent cell).
        for sampler in ("euler", "euler_ancestral", "heun"):
            with self.subTest(sampler=sampler):
                self._refine(images=[image], masks=[mask_image(64, 32, slice(0, 32))], on_step=on_step, sampler=sampler)
                latents = captured["latents"]
                self.assertTrue(torch.equal(latents[..., 2:], source[..., 2:]), "the kept area ends on the source exactly")
                self.assertFalse(torch.equal(latents[..., :2], source[..., :2]), "the masked area is redrawn")
                self.assertTrue(self.runtime.last_generation_metrics["refinement.sampling"]["masked"])
        # An all-white mask is the unmasked redraw, and an all-black one hands back the source.
        self._refine(images=[image], on_step=on_step)
        unmasked = captured["latents"]
        self._refine(images=[image], masks=[mask_image(64, 32, slice(0, 64))], on_step=on_step)
        self.assertTrue(torch.equal(captured["latents"], unmasked))
        self._refine(images=[image], masks=[mask_image(64, 32, slice(0, 0))], on_step=on_step)
        self.assertTrue(torch.equal(captured["latents"], source))

    def test_refusals(self):
        image = rgb_image(32, 32, 1)
        cases = [
            dict(prepared_conditioning=object()),
            dict(prepared_sigmas=torch.tensor([0.5, 0.0])),
            dict(denoise=0),
            dict(denoise=1.2),
            dict(guidance="pag", pag_scale=1.0),
            dict(guidance="cfg_zero_star"),
            dict(sampler="dpmpp_2m"),
            dict(moss={"standard_slg": True}),
            dict(moss={"uncond_skip_layers": [3]}),
            dict(images=[]),
            dict(images=[image.convert("L")]),
            dict(images=[image, rgb_image(48, 32, 2)], generators=[torch.Generator(), torch.Generator()]),
            dict(images=[image, image]),
            dict(images=[rgb_image(40, 32, 2)]),
            dict(masks=[mask_image(16, 16, slice(0, 8))]),
            dict(masks=[]),
        ]
        for overrides in cases:
            with self.subTest(overrides=sorted(overrides)), self.assertRaises(ValueError):
                self._refine(**overrides)


if __name__ == "__main__":
    unittest.main()
