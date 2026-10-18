"""Numerical oracles for the native MOSS-Image engine (QPlan2/MOSS/P1, Phase B/C).

Every comparison here is against an implementation written independently of the module under test:
the official HF ``MOSSImageTransformer`` (vendored, hash-checked), a token-by-token gated-delta
recurrence, a naive Qwen3.5 written from the formulas, an explicit Hadamard/ConvRot emulation, and
closed-form flow schedules. Tolerances are the ones Phase A predeclared
(``README/QPlan2/MOSS/P1.1-MOSS-PhaseA-Baseline.md`` §A5) and are not loosened here.
"""

import hashlib
import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .. import comfy_quant, moss_model, moss_quant, moss_sampling, moss_text_encoder
    from ..anima_sampling import _comfy_simple_refinement_schedule, anima_sigma_schedule
    from ..moss_pipeline import load_moss_transformer
    from ..quantized_linear import QuantizedLinear
except ImportError:
    import comfy_quant
    import moss_model
    import moss_quant
    import moss_sampling
    import moss_text_encoder
    from anima_sampling import _comfy_simple_refinement_schedule, anima_sigma_schedule
    from moss_pipeline import load_moss_transformer
    from quantized_linear import QuantizedLinear

REFERENCE = Path(__file__).resolve().parent / "fixtures" / "moss_reference" / "pipeline_moss_image.py"
REFERENCE_SHA256 = "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31"


def load_reference():
    data = REFERENCE.read_bytes()
    if hashlib.sha256(data).hexdigest() != REFERENCE_SHA256:
        raise AssertionError("vendored MOSS reference pipeline does not match its pinned SHA-256")
    spec = importlib.util.spec_from_file_location("moss_reference_pipeline", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TINY_DIT = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=40, freq_dim=32, latent_channels=8, rope_base=10000.0)


def randomize(module, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)
    return module


def hadamard_independent(size):
    """The regular Hadamard ConvRot uses: Kronecker powers of the regular 4x4, built in float64."""
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float64)
    h = h4
    while h.shape[0] < size:
        h = np.kron(h, h4)
    return h / math.sqrt(size)


def convrot_quantize(weight: torch.Tensor, group: int):
    """Offline ConvRot: rotate each column group by H, then per-row absmax INT8 (the exporter's step)."""
    h = torch.from_numpy(hadamard_independent(group)).to(torch.float32)
    out_features, in_features = weight.shape
    rotated = (weight.reshape(out_features, in_features // group, group) @ h.T).reshape(out_features, in_features)
    scale = rotated.abs().amax(dim=1, keepdim=True) / 127.0
    q = torch.round(rotated / scale).clamp(-128, 127).to(torch.int8)
    return q, scale.to(torch.float32)


class SamplingOracleTests(unittest.TestCase):
    def test_sgm_uniform_reproduces_comfy_double_shift_endpoint(self):
        shift, steps = 5.0, 50
        sigmas = moss_sampling.moss_sigma_schedule(steps, "sgm_uniform", shift)
        # ComfyUI: timestep(sigma_min) = sigma_min * 1000 -- the *shifted* sigma -- so the last
        # interpolated timestep is shifted a second time. Closed form in float64:
        shift_fn = lambda t: shift * t / (1 + (shift - 1) * t)
        start, end = 1000.0, 1000.0 * shift_fn(0.001)
        ts = np.linspace(start, end, steps + 1)[:-1] / 1000.0
        expected = np.append(shift_fn(ts), 0.0)
        self.assertEqual(len(sigmas), steps + 1)
        self.assertEqual(float(sigmas[0]), 1.0)
        self.assertEqual(float(sigmas[-1]), 0.0)
        self.assertLess(float(np.abs(sigmas.double().numpy() - expected).max()), 2e-7)

    def test_schedules_match_anima_where_both_follow_comfy(self):
        # Anima runs ModelSamplingDiscreteFlow at shift 3 with multiplier 1; these schedulers do not
        # touch `timestep()` and must agree exactly with this module's table at the same multiplier,
        # at a step count that divides the 1000-entry table.
        ms = moss_sampling.MossModelSampling(3.0, multiplier=1)
        for scheduler in ("simple", "ddim_uniform", "karras", "exponential", "kl_optimal", "linear_quadratic"):
            with self.subTest(scheduler=scheduler):
                ours = moss_sampling._calculate_sigmas(ms, scheduler, 20)
                # Anima's *base* `simple` is a linspace by design; its ComfyUI-exact table walk is
                # the refinement helper, which is the one ComfyUI's simple_scheduler is.
                theirs = _comfy_simple_refinement_schedule(20) if scheduler == "simple" else anima_sigma_schedule(20, scheduler)
                self.assertTrue(torch.equal(ours, theirs), f"{scheduler}: {ours[:4]} vs {theirs[:4]}")

    def test_multiplier_round_trip_is_comfy_float32(self):
        # MOSS's multiplier is 1000: ComfyUI evaluates sigma((i / 1000 * 1000) / 1000) in float32.
        ms = moss_sampling.MossModelSampling(5.0)
        t = (torch.arange(1, 1001, 1) / 1000) * 1000 / 1000
        self.assertTrue(torch.equal(ms.sigmas, 5.0 * t / (1 + 4.0 * t)))

    def test_every_scheduler_is_monotone_and_terminal(self):
        for scheduler in moss_sampling.MOSS_SCHEDULERS:
            for steps in (1, 7, 50):
                if scheduler == "kl_optimal" and steps == 1:
                    with self.assertRaises(ValueError):
                        moss_sampling.moss_sigma_schedule(steps, scheduler, 5.0)
                    continue
                sigmas = moss_sampling.moss_sigma_schedule(steps, scheduler, 5.0)
                self.assertEqual(float(sigmas[-1]), 0.0)
                self.assertTrue(torch.all(sigmas[:-1] >= sigmas[1:]))
                self.assertEqual(sigmas.dtype, torch.float32)

    def test_model_sampling_table_and_percent_to_sigma(self):
        ms = moss_sampling.MossModelSampling(5.0)
        self.assertEqual(len(ms.sigmas), 1000)
        self.assertAlmostEqual(float(ms.sigma_min), 5 * 0.001 / (1 + 4 * 0.001), places=7)
        self.assertEqual(float(ms.sigma_max), 1.0)
        self.assertEqual(ms.percent_to_sigma(0.0), 1.0)
        self.assertEqual(ms.percent_to_sigma(1.0), 0.0)
        self.assertAlmostEqual(ms.percent_to_sigma(0.5), 5 * 0.5 / (1 + 4 * 0.5))
        self.assertEqual(float(ms.timestep(torch.tensor(0.25))), 250.0)

    def test_shift_is_applied_once(self):
        # shift 1 is the identity; the same steps at shift 5 must equal the shifted identity schedule
        # for `simple`, which reads the table and nothing else.
        base = moss_sampling.moss_sigma_schedule(10, "simple", 1.0).double()
        shifted = moss_sampling.moss_sigma_schedule(10, "simple", 5.0).double()
        expected = 5 * base / (1 + 4 * base)
        self.assertLess(float((shifted - expected).abs().max()), 2e-7)

    def test_euler_and_heun_match_hand_written_k_diffusion(self):
        torch.manual_seed(0)
        target = torch.randn(1, 4, 3, 3)

        def denoise(x, sigma, _step):
            return 0.7 * x + 0.3 * target  # any deterministic denoiser

        sigmas = moss_sampling.moss_sigma_schedule(6, "sgm_uniform", 5.0)
        x0 = torch.randn(1, 4, 3, 3)
        for sampler in ("euler", "heun"):
            with self.subTest(sampler=sampler):
                ours, evaluations = moss_sampling.sample_moss(sampler, denoise, x0.clone(), sigmas)
                x = x0.clone()
                count = 0
                for i in range(len(sigmas) - 1):
                    s, n = sigmas[i], sigmas[i + 1]
                    d = (x - denoise(x, s, i)) / s
                    count += 1
                    if sampler == "euler" or n == 0:
                        x = x + d * (n - s)
                    else:
                        x2 = x + d * (n - s)
                        d2 = (x2 - denoise(x2, n, i)) / n
                        count += 1
                        x = x + (d + d2) / 2 * (n - s)
                self.assertTrue(torch.equal(ours, x))
                self.assertEqual(evaluations, count)

    def test_ancestral_is_seed_reproducible_and_rf(self):
        sigmas = moss_sampling.moss_sigma_schedule(5, "simple", 5.0)
        denoise = lambda x, s, i: 0.5 * x
        runs = []
        for _ in range(2):
            generators = [torch.Generator().manual_seed(7)]
            x = torch.randn(1, 2, 2, 2, generator=torch.Generator().manual_seed(1))
            runs.append(moss_sampling.sample_moss("euler_ancestral", denoise, x, sigmas, generators)[0])
        self.assertTrue(torch.equal(runs[0], runs[1]))

    def test_cfg_combines_denoised_estimates(self):
        x = torch.randn(2, 3, 2, 2)
        sigma = torch.tensor([0.6, 0.6])
        cond, uncond = torch.randn(2, 3, 2, 2), torch.randn(2, 3, 2, 2)
        out = moss_sampling.cfg_denoised(x, sigma, cond, uncond, 3.0)
        dc, du = x - cond * 0.6, x - uncond * 0.6
        self.assertTrue(torch.allclose(out, du + (dc - du) * 3.0, atol=1e-6))
        self.assertTrue(torch.equal(moss_sampling.cfg_denoised(x, sigma, cond, None, 1.0), x - cond * sigma.reshape(-1, 1, 1, 1)))

    def test_guidance_plan_windows_and_refusals(self):
        plan = moss_sampling.build_guidance_plan(3.0)
        self.assertEqual(plan.uncond_skip_layers, (22,))
        self.assertEqual(plan.skip_layers_at(torch.tensor([0.5])), frozenset({22}))
        self.assertEqual(plan.skip_layers_at(torch.tensor([1.0])), frozenset({22}))
        narrow = moss_sampling.build_guidance_plan(3.0, [5, 6], 0.2, 0.6, 5.0)
        window = narrow.skip_window()
        self.assertEqual(narrow.skip_layers_at(torch.tensor([window[1] + 1e-4])), frozenset())
        self.assertEqual(narrow.skip_layers_at(torch.tensor([(window[0] + window[1]) / 2])), frozenset({5, 6}))
        self.assertFalse(moss_sampling.build_guidance_plan(1.0).skip_active)
        self.assertEqual(moss_sampling.build_guidance_plan(1.0).skip_layers_at(torch.tensor([0.5])), frozenset())
        self.assertFalse(moss_sampling.build_guidance_plan(3.0, []).skip_active)
        for bad in ([42], [-1], [3, 3], [True], "22"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                moss_sampling.build_guidance_plan(3.0, bad)
        with self.assertRaises(ValueError):
            moss_sampling.build_guidance_plan(3.0, [22], 0.8, 0.2)
        with self.assertRaises(ValueError):
            moss_sampling.build_guidance_plan(3.0, [22], shift=0.0)
        with self.assertRaises(ValueError):
            moss_sampling.moss_sigma_schedule(10, "align_your_steps")
        with self.assertRaises(ValueError):
            moss_sampling.sample_moss("dpmpp_2m", lambda x, s, i: x, torch.zeros(1, 1, 1, 1), torch.tensor([1.0, 0.0]))


class DiTOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = load_reference()

    def _models(self, dtype=torch.float32):
        hf = randomize(self.reference.MOSSImageTransformer(**TINY_DIT), 11).to(dtype).eval()
        state = {key: value.detach().clone() for key, value in hf.state_dict().items()}
        ours, config, report = load_moss_transformer(None, dtype=dtype, state=state)
        return hf, ours, config, report

    def _inputs(self, dtype=torch.float32):
        generator = torch.Generator().manual_seed(3)
        latents = torch.randn(2, 8, 4, 6, generator=generator)
        context = torch.randn(2, 7, 40, generator=generator).to(dtype)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
        timestep = torch.tensor([999.0, 123.4])
        return latents, timestep, context, mask

    def test_float32_forward_matches_official_transformer(self):
        hf, ours, config, report = self._models()
        self.assertEqual(report["layout"]["source_qkv"], "split")
        self.assertEqual(report["layout"]["fused_blocks"], 3)
        self.assertEqual(report["rope_bands"], "verified")
        latents, timestep, context, mask = self._inputs()
        with torch.no_grad():
            expected = hf(latents, timestep, context, mask).sample
            actual = ours(latents, timestep, context, mask)
        self.assertEqual(actual.shape, expected.shape)
        self.assertLessEqual(float((actual - expected).abs().max()), 1e-5)

    def test_skip_layers_match_official_skip(self):
        hf, ours, _config, _report = self._models()
        latents, timestep, context, mask = self._inputs()
        with torch.no_grad():
            expected = hf(latents, timestep, context, mask, skip_layers=[1]).sample
            actual = ours(latents, timestep, context, mask, skip_layers=frozenset({1}))
            unskipped = ours(latents, timestep, context, mask)
        self.assertLessEqual(float((actual - expected).abs().max()), 1e-5)
        self.assertGreater(float((actual - unskipped).abs().max()), 1e-4)

    def test_bfloat16_forward_tracks_official_transformer(self):
        hf, ours, _config, _report = self._models(torch.bfloat16)
        latents, timestep, context, mask = self._inputs(torch.bfloat16)
        with torch.no_grad():
            expected = hf(latents, timestep, context, mask).sample.float()
            actual = ours(latents, timestep, context, mask).float()
        scale = float(expected.abs().max())
        error = float((actual - expected).abs().max())
        # One bfloat16 ulp of the output's magnitude: the fused q/k/v GEMM may round differently.
        self.assertLessEqual(error, scale * 2 ** -7, f"max error {error} at scale {scale}")

    def test_padding_never_reaches_attention(self):
        _hf, ours, _config, _report = self._models()
        latents, timestep, context, mask = self._inputs()
        garbage = context.clone()
        garbage[0, 5:] = 1e3
        with torch.no_grad():
            a = ours(latents, timestep, context, mask)
            b = ours(latents, timestep, garbage, mask)
        self.assertTrue(torch.equal(a, b))

    def test_fused_layout_loads_identically(self):
        hf, ours, _config, _report = self._models()
        state = {key: value.detach().clone() for key, value in hf.state_dict().items()}
        fused, _layout = moss_model.canonical_moss_state(state)
        for index in range(TINY_DIT["num_layers"]):
            expected = torch.cat([state[f"layers.{index}.proj.{part}.weight"] for part in "qkv"], 0)
            self.assertTrue(torch.equal(fused[f"layers.{index}.proj.qkv.weight"], expected))
        again, _config, report = load_moss_transformer(None, dtype=torch.float32, state=fused)
        self.assertEqual(report["layout"]["source_qkv"], "fused")
        latents, timestep, context, mask = self._inputs()
        with torch.no_grad():
            self.assertTrue(torch.equal(ours(latents, timestep, context, mask), again(latents, timestep, context, mask)))


class ConvRotOracleTests(unittest.TestCase):
    def test_int8_convrot_dequantise_matches_independent_hadamard(self):
        torch.manual_seed(5)
        weight = torch.randn(24, 64)
        q, scale = convrot_quantize(weight, 16)
        ours = comfy_quant.dequantize_int8(q, scale, torch.float32, convrot=True, convrot_groupsize=16)
        h = hadamard_independent(16)
        values = q.double().numpy() * scale.double().numpy()
        expected = (values.reshape(24, 4, 16) @ h.T).reshape(24, 64)  # H is symmetric orthogonal: its own inverse
        self.assertLess(float(np.abs(ours.double().numpy() - expected).max()), 1e-5)
        # The exporter's rotation is undone: the dequantised weight approximates the original.
        self.assertLess(float((ours - weight).abs().max()), float(weight.abs().max()) * 0.02)

    def test_int8_linear_matches_integer_oracle_exactly(self):
        torch.manual_seed(6)
        weight = torch.randn(32, 64)
        q, scale = convrot_quantize(weight, 16)
        x = torch.randn(5, 64)
        out = comfy_quant.int8_linear(x, q, scale, None, torch.float32, convrot=True, convrot_groupsize=16)
        # Independent from the activation rotation on: float32 grouped product with the separately
        # built Hadamard, per-row absmax quantisation in numpy float32, an exact int64 GEMM, then the
        # float32 rescale in the documented order (accumulator x (row scale x column scale)).
        # The rotated activation is the module's own float32 product (summation order is a BLAS
        # detail); it is checked against the float64 independent rotation, and everything after it
        # -- the part that makes this W8A8 -- is recomputed independently and compared exactly.
        xr = comfy_quant.rotate_activation(x, comfy_quant.build_hadamard(16, "cpu", torch.float32), 16).numpy()
        xr64 = (x.double().numpy().reshape(5, 4, 16) @ hadamard_independent(16)).reshape(5, 64)
        self.assertLess(float(np.abs(xr - xr64).max()), 1e-5)
        # PyTorch divides a float32 tensor by a Python number as a multiply by the float32 reciprocal.
        xs = (np.abs(xr).max(axis=1, keepdims=True).astype(np.float32) * np.float32(1.0 / 127.0)).astype(np.float32)
        xq = np.clip(np.round(xr / xs), -128, 127).astype(np.int64)
        acc = xq @ q.numpy().astype(np.int64).T
        expected = (acc.astype(np.float32) * (xs * scale.numpy().reshape(1, -1).astype(np.float32))).astype(np.float32)
        self.assertTrue(np.array_equal(out.numpy(), expected), f"max diff {np.abs(out.numpy() - expected).max()}")
        # And the W8A8 product stays close to the unquantised one.
        reference = x @ weight.T
        self.assertLess(float((out - reference).abs().max()), float(reference.abs().max()) * 5e-2)

    def test_fused_int8_qkv_is_bit_identical_to_split(self):
        torch.manual_seed(7)
        parts = [convrot_quantize(torch.randn(16, 64), 16) for _ in range(3)]
        marker = comfy_quant.encode_marker({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16})
        state = {}
        for name, (q, scale) in zip("qkv", parts):
            state[f"layers.0.proj.{name}.weight"] = q
            state[f"layers.0.proj.{name}.weight_scale"] = scale
            state[f"layers.0.proj.{name}.comfy_quant"] = marker
        fused, count = moss_model.fuse_split_qkv(state)
        self.assertEqual(count, 1)
        config = comfy_quant.parse_layer_config(marker, "x")
        x = torch.randn(9, 64).to(torch.bfloat16)
        fused_out = comfy_quant.int8_linear(x, fused["layers.0.proj.qkv.weight"], fused["layers.0.proj.qkv.weight_scale"],
                                            None, torch.bfloat16, convrot=True, convrot_groupsize=16)
        split_out = torch.cat([comfy_quant.int8_linear(x, q, s, None, torch.bfloat16, convrot=True, convrot_groupsize=16)
                               for q, s in parts], -1)
        self.assertTrue(torch.equal(fused_out, split_out))
        self.assertTrue(config.convrot)

    def test_split_fusion_refuses_inexact_cases(self):
        marker = comfy_quant.encode_marker({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16})
        other = comfy_quant.encode_marker({"format": "int8_tensorwise"})
        q = torch.zeros(16, 64, dtype=torch.int8)
        base = {f"layers.0.proj.{n}.weight": q for n in "qkv"}
        with self.assertRaises(ValueError):
            moss_model.fuse_split_qkv({**base, **{f"layers.0.proj.{n}.comfy_quant": marker for n in "qk"},
                                       **{f"layers.0.proj.{n}.weight_scale": torch.ones(16, 1) for n in "qkv"}})
        with self.assertRaises(ValueError):
            moss_model.fuse_split_qkv({**base, "layers.0.proj.q.comfy_quant": marker, "layers.0.proj.k.comfy_quant": marker,
                                       "layers.0.proj.v.comfy_quant": other,
                                       **{f"layers.0.proj.{n}.weight_scale": torch.ones(16, 1) for n in "qkv"}})
        with self.assertRaises(ValueError):
            moss_model.fuse_split_qkv({**base, **{f"layers.0.proj.{n}.comfy_quant": marker for n in "qkv"},
                                       **{f"layers.0.proj.{n}.weight_scale": torch.ones(3, 1) for n in "qkv"}})
        with self.assertRaises(ValueError):
            moss_model.fuse_split_qkv({**base, **{f"layers.0.proj.{n}.input_scale": torch.ones(()) for n in "qkv"}})
        # Scalar scales broadcast to rows exactly.
        fused, _ = moss_model.fuse_split_qkv({**base, **{f"layers.0.proj.{n}.comfy_quant": other for n in "qkv"},
                                              **{f"layers.0.proj.{n}.weight_scale": torch.tensor(0.5) for n in "qkv"}})
        self.assertEqual(tuple(fused["layers.0.proj.qkv.weight_scale"].shape), (48, 1))
        self.assertTrue(torch.all(fused["layers.0.proj.qkv.weight_scale"] == 0.5))


class EmbeddingAndFP8Tests(unittest.TestCase):
    def _int8_table(self):
        torch.manual_seed(8)
        table = torch.randn(50, 64)
        q, scale = convrot_quantize(table, 16)
        marker = comfy_quant.encode_marker({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16})
        return table, {"embed_tokens.weight": q, "embed_tokens.weight_scale": scale, "embed_tokens.comfy_quant": marker}

    def _build(self, state, dtype=torch.float32):
        module = moss_quant.embedding_from_state("embed_tokens", state, dtype, "test")
        module.load_state_dict(moss_quant.embedding_state_for("embed_tokens", state), strict=True, assign=True)
        return module

    def test_int8_convrot_gather_equals_full_table_dequantise(self):
        table, state = self._int8_table()
        module = self._build(state)
        ids = torch.tensor([[0, 7, 49, 7, 3]])
        rows = module(ids)
        q, scale = state["embed_tokens.weight"], state["embed_tokens.weight_scale"]
        full = comfy_quant.dequantize_int8(q, scale, torch.float32, convrot=True, convrot_groupsize=16)
        # Dequantising only the gathered rows is the same arithmetic on the same rows ...
        gathered = comfy_quant.dequantize_int8(q[ids[0]], scale[ids[0]], torch.float32, convrot=True, convrot_groupsize=16)
        self.assertTrue(torch.equal(rows[0], gathered))
        # ... and agrees with the full-table expansion up to the GEMM's batch-size-dependent rounding.
        self.assertLessEqual(float((rows[0] - full[ids[0]]).abs().max()), 1e-6)
        self.assertLess(float((rows[0] - table[ids[0]]).abs().max()), float(table.abs().max()) * 0.02)

    def test_embedding_refuses_bad_ids_and_layouts(self):
        _table, state = self._int8_table()
        module = self._build(state)
        with self.assertRaises(IndexError):
            module(torch.tensor([[50]]))
        with self.assertRaises(TypeError):
            module(torch.tensor([[1.0]]))
        bad = dict(state)
        bad["embed_tokens.weight_scale"] = torch.ones(3, 1)
        with self.assertRaises(moss_quant.UnsupportedQuantization):
            moss_quant.embedding_from_state("embed_tokens", bad, torch.float32, "test")
        mx = dict(state)
        mx["embed_tokens.comfy_quant"] = comfy_quant.encode_marker({"format": "mxfp8"})
        with self.assertRaises(moss_quant.UnsupportedQuantization):
            moss_quant.embedding_from_state("embed_tokens", mx, torch.float32, "test")

    def test_fp8_embedding_matches_comfy_cast_order(self):
        table = (torch.randn(20, 16) * 0.5).to(torch.float8_e4m3fn)
        state = {"embed_tokens.weight": table, "embed_tokens.weight_scale": torch.tensor(0.25)}
        module = self._build(state, torch.bfloat16)
        ids = torch.tensor([[1, 2, 19]])
        expected = table[ids[0]].to(torch.bfloat16) * torch.tensor(0.25).to(torch.bfloat16)
        self.assertTrue(torch.equal(module(ids)[0], expected))
        self.assertEqual(module.weight.dtype, torch.float8_e4m3fn)
        module.to(torch.float32)  # a dtype sweep must not expand storage or round the scale
        self.assertEqual(module.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(module.scale_weight.dtype, torch.float32)

    def test_raw_fp8_linear_is_a_per_use_cast(self):
        weight = (torch.randn(8, 16) * 0.3).to(torch.float8_e5m2)
        layer = moss_quant.RawFP8Linear(16, 8, False, torch.float8_e5m2, torch.float32, device="meta")
        layer.load_state_dict({"weight": weight}, strict=True, assign=True)
        x = torch.randn(3, 16)
        self.assertTrue(torch.equal(layer(x), x @ weight.to(torch.float32).T))
        layer.to(torch.bfloat16)
        self.assertEqual(layer.weight.dtype, torch.float8_e5m2)
        with self.assertRaises(ValueError):
            moss_quant.RawFP8Linear(16, 8, False, torch.bfloat16)

    def test_attention_markers(self):
        good = torch.tensor(list(b'{"attention": "comfy_kitchen_int8"}'), dtype=torch.uint8)
        self.assertEqual(moss_quant.parse_attention_config(good, "layers.0")["methods"], ["comfy_kitchen_int8"])
        listed = torch.tensor(list(b'[{"attention": "comfy_kitchen_int8"}]'), dtype=torch.uint8)
        self.assertEqual(moss_quant.parse_attention_config(listed, "layers.0")["methods"], ["comfy_kitchen_int8"])
        for raw in (b'{"attention": "comfy_kitchen_sol"}', b'{"attention": "comfy_kitchen_int8", "tau": 1}', b'[]', b'not json'):
            with self.subTest(raw=raw), self.assertRaises(moss_quant.UnsupportedQuantization):
                moss_quant.parse_attention_config(torch.tensor(list(raw), dtype=torch.uint8), "layers.0")
        execution = moss_quant.attention_execution(["comfy_kitchen_int8"])
        self.assertEqual(execution["backend"], "sdpa")
        self.assertIn("unavailable", execution["fallback_reason"])


def _recurrent_gated_delta(query, key, value, g, beta):
    """Token-by-token gated delta rule, written from the recurrence (not the chunked algorithm)."""
    q = query / torch.sqrt((query * query).sum(-1, keepdim=True) + 1e-6)
    k = key / torch.sqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    q, k, v, b, gg = [t.transpose(1, 2).double() for t in (q, k, value, beta, g)]
    batch, heads, length, dk = k.shape
    dv = v.shape[-1]
    q = q / math.sqrt(dk)
    state = torch.zeros(batch, heads, dk, dv, dtype=torch.float64)
    out = torch.zeros(batch, heads, length, dv, dtype=torch.float64)
    for t in range(length):
        state = state * gg[:, :, t].exp()[..., None, None]
        memory = torch.einsum("bhk,bhkv->bhv", k[:, :, t], state)
        delta = (v[:, :, t] - memory) * b[:, :, t][..., None]
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, :, t], delta)
        out[:, :, t] = torch.einsum("bhk,bhkv->bhv", q[:, :, t], state)
    return out.transpose(1, 2)


TINY_TE = dict(vocab_size=300, hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=2,
               num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
               linear_key_head_dim=8, linear_value_head_dim=8,
               layer_types=["linear_attention"] * 3 + ["full_attention"])


def _naive_qwen35(model, ids):
    """Qwen3.5 written from the formulas in float64: explicit causal conv, recurrent DeltaNet, softmax attention."""
    config = model.config

    def rms(x, w, add=True, eps=1e-6):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return y * ((1.0 + w) if add else w)

    def lin(x, layer):
        return x @ layer.weight.double().T

    x = model.embed_tokens.weight.double()[ids[0]][None]
    length = x.shape[1]
    dim = config.rotary_dim
    inv = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    freqs = torch.arange(length, dtype=torch.float64)[:, None] * inv[None]
    cos, sin = torch.cat((freqs, freqs), -1).cos(), torch.cat((freqs, freqs), -1).sin()
    for layer in model.layers:
        h = rms(x, layer.input_layernorm.weight.double())
        if layer.layer_type == "linear_attention":
            la = layer.linear_attn
            mixed = lin(h, la.in_proj_qkv)[0]  # [L, C]
            w = la.conv1d.weight.double()[:, 0]  # [C, 4]
            conv = torch.zeros_like(mixed)
            for t in range(length):
                for j in range(4):
                    src = t - 3 + j
                    if src >= 0:
                        conv[t] += w[:, j] * mixed[src]
            conv = F.silu(conv)
            kd, vd = la.key_dim, la.value_dim
            q = conv[:, :kd].reshape(1, length, -1, la.head_k_dim)
            k = conv[:, kd:2 * kd].reshape(1, length, -1, la.head_k_dim)
            v = conv[:, 2 * kd:].reshape(1, length, -1, la.head_v_dim)
            rep = la.num_v_heads // la.num_k_heads
            q, k = q.repeat_interleave(rep, 2), k.repeat_interleave(rep, 2)
            beta = lin(h, la.in_proj_b).sigmoid()
            g = -la.A_log.double().exp() * F.softplus(lin(h, la.in_proj_a) + la.dt_bias.double())
            core = _recurrent_gated_delta(q, k, v, g, beta)
            z = lin(h, la.in_proj_z).reshape(1, length, -1, la.head_v_dim)
            normed = core * torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + 1e-6) * la.norm.weight.double() * F.silu(z)
            h = lin(normed.reshape(1, length, -1), la.out_proj)
        else:
            at = layer.self_attn
            qg = lin(h, at.q_proj).view(1, length, -1, at.head_dim * 2)
            q, gate = qg[..., :at.head_dim], qg[..., at.head_dim:].reshape(1, length, -1)
            q = rms(q, at.q_norm.weight.double())
            k = rms(lin(h, at.k_proj).view(1, length, -1, at.head_dim), at.k_norm.weight.double())
            v = lin(h, at.v_proj).view(1, length, -1, at.head_dim)
            rot = cos.shape[-1]

            def rope(t):
                a, rest = t[..., :rot], t[..., rot:]
                half = rot // 2
                rotated = torch.cat((-a[..., half:], a[..., :half]), -1)
                return torch.cat((a * cos[None, :, None] + rotated * sin[None, :, None], rest), -1)

            q, k = rope(q), rope(k)
            groups = at.num_heads // at.num_kv_heads
            k, v = k.repeat_interleave(groups, 2), v.repeat_interleave(groups, 2)
            scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * at.scaling
            scores = scores.masked_fill(torch.triu(torch.ones(length, length, dtype=torch.bool), 1), float("-inf"))
            out = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v).reshape(1, length, -1)
            h = lin(out * gate.sigmoid(), at.o_proj)
        x = x + h
        m = layer.mlp
        hm = rms(x, layer.post_attention_layernorm.weight.double())
        x = x + lin(F.silu(lin(hm, m.gate_proj)) * lin(hm, m.up_proj), m.down_proj)
    return rms(x, model.norm.weight.double())


class TextEncoderOracleTests(unittest.TestCase):
    def test_chunked_delta_rule_matches_recurrence(self):
        torch.manual_seed(9)
        batch, length, heads, dk, dv = 1, 70, 3, 8, 6  # 70 > 64: two chunks plus padding
        q, k = torch.randn(batch, length, heads, dk), torch.randn(batch, length, heads, dk)
        v = torch.randn(batch, length, heads, dv)
        g = -torch.rand(batch, length, heads) * 0.5
        beta = torch.rand(batch, length, heads)
        ours = moss_text_encoder.chunk_gated_delta_rule(q, k, v, g, beta)
        expected = _recurrent_gated_delta(q, k, v, g, beta)
        self.assertLessEqual(float((ours.double() - expected).abs().max()), 2e-5)

    def _tiny_model(self):
        config = moss_text_encoder.Qwen35TextConfig(**TINY_TE)
        model = moss_text_encoder.Qwen35TextModel(config, compute_dtype=torch.float32)
        randomize(model, 12)
        with torch.no_grad():
            for layer in model.layers:
                if layer.layer_type == "linear_attention":
                    layer.linear_attn.A_log.copy_(torch.log(torch.rand(4) * 2 + 0.5))
        return model.eval()

    def test_tiny_qwen35_matches_naive_reference(self):
        model = self._tiny_model()
        ids = torch.tensor([[5, 17, 2, 299, 0, 42, 42, 8, 100, 3, 77, 1]])
        with torch.no_grad():
            ours = model(ids)
        expected = _naive_qwen35(model, ids)
        self.assertLessEqual(float((ours.double() - expected).abs().max()), 5e-5)

    def test_hidden_state_is_causal(self):
        model = self._tiny_model()
        ids = torch.tensor([[5, 17, 2, 299, 0, 42]])
        longer = torch.tensor([[5, 17, 2, 299, 0, 42, 9, 9]])
        with torch.no_grad():
            a, b = model(ids), model(longer)
        self.assertLessEqual(float((a - b[:, :6]).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
