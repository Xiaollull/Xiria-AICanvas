"""Krea 2 prepares its refinement conditioning and sigma schedule once and reuses them per tile.

USDU refines one tile at a time, so without a prepared pair every tile would re-encode the prompt
and rebuild the schedule. Both are validated rather than trusted on the way back in: conditioning
or sigmas belonging to a different request would not fail loudly, they would render a different
picture on one side of a seam, which is exactly the artefact tiled refinement exists to avoid.

`refine_batch` used to accept both parameters and discard them with `del`, so a caller that passed
them got no error and no effect. These tests pin that they now take effect and that a mismatch is
refused.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from krea2_pipeline import Krea2Runtime, PreparedKrea2Conditioning  # noqa: E402
from krea2_sampling import (  # noqa: E402
    prepare_krea2_refinement_sigmas,
    validate_prepared_krea2_refinement_sigmas,
)

WIDTH = 5  # twelve taps folded into the feature axis; the exact width does not matter here


def make_runtime(dtype=torch.bfloat16):
    runtime = Krea2Runtime.__new__(Krea2Runtime)
    runtime._closed = False
    runtime._poisoned = False
    runtime.dtype = dtype
    runtime.device = torch.device("cpu")
    runtime.last_generation_metrics = {}
    runtime._last_sampling_execution = {}
    runtime._conditioning_cache = {}
    runtime._encode_prompt = Mock(side_effect=lambda text: torch.ones((1, 2, WIDTH), dtype=dtype))
    runtime._encode_images = Mock(return_value=torch.zeros((1, 16, 4, 4)))
    runtime._sample = Mock(return_value=torch.zeros((1, 16, 4, 4)))
    runtime._decode = Mock(return_value=[Image.new("RGB", (32, 32))])
    return runtime


def conditioning(prompt="prompt", negative_prompt="negative", cfg=1.0, guidance="none",
                 dtype=torch.bfloat16, negative=None, positive=None):
    return PreparedKrea2Conditioning(
        prompt=prompt,
        negative_prompt=negative_prompt,
        cfg=cfg,
        guidance=guidance,
        positive=torch.ones((1, 2, WIDTH), dtype=dtype) if positive is None else positive,
        negative=negative,
    )


class PrepareTests(unittest.TestCase):
    def test_preparing_encodes_once_and_lands_on_the_cpu(self):
        runtime = make_runtime()
        prepared = runtime.prepare_refinement_conditioning("prompt", "negative", 1.0, "none")
        self.assertIsInstance(prepared, PreparedKrea2Conditioning)
        self.assertEqual(prepared.positive.device.type, "cpu")
        # CFG 1 evaluates no unconditional branch, so encoding one would be a language-model pass
        # for a tensor nothing reads.
        self.assertIsNone(prepared.negative)
        self.assertEqual(runtime._encode_prompt.call_count, 1)

    def test_a_cfg_request_prepares_both_branches(self):
        runtime = make_runtime()
        prepared = runtime.prepare_refinement_conditioning("prompt", "negative", 4.0, "none")
        self.assertIsNotNone(prepared.negative)
        self.assertEqual(runtime._encode_prompt.call_count, 2)

    def test_cfg_zero_star_prepares_an_unconditional_branch_even_at_cfg_one(self):
        runtime = make_runtime()
        prepared = runtime.prepare_refinement_conditioning("prompt", "negative", 1.0, "cfg_zero_star")
        self.assertIsNotNone(prepared.negative)

    def test_the_request_contract_is_checked(self):
        runtime = make_runtime()
        with self.assertRaises(TypeError):
            runtime.prepare_refinement_conditioning(None, "negative", 1.0, "none")
        for cfg in (float("nan"), float("inf"), True):
            with self.subTest(cfg=cfg):
                with self.assertRaises(ValueError):
                    runtime.prepare_refinement_conditioning("prompt", "negative", cfg, "none")
        # PAG needs an identity-self-attention override this runtime does not install.
        for guidance in ("pag", "unknown"):
            with self.subTest(guidance=guidance):
                with self.assertRaises(ValueError):
                    runtime.prepare_refinement_conditioning("prompt", "negative", 1.0, guidance)


class ValidateTests(unittest.TestCase):
    def test_matching_conditioning_is_returned_as_a_copy(self):
        runtime = make_runtime()
        prepared = conditioning()
        positive, negative = runtime._validate_prepared_conditioning(
            prepared, "prompt", "negative", 1.0, "none"
        )
        self.assertTrue(torch.equal(positive, prepared.positive))
        self.assertIsNot(positive, prepared.positive, "a tile must not be able to edit the shared copy")
        self.assertIsNone(negative)

    def test_a_different_request_is_refused(self):
        runtime = make_runtime()
        prepared = conditioning()
        for args in (
            ("other", "negative", 1.0, "none"),
            ("prompt", "other", 1.0, "none"),
            ("prompt", "negative", 4.0, "none"),
            ("prompt", "negative", 1.0, "cfg_zero_star"),
        ):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    runtime._validate_prepared_conditioning(prepared, *args)

    def test_a_branch_that_does_not_match_the_guidance_is_refused(self):
        runtime = make_runtime()
        # Prepared without an unconditional branch, but the request evaluates one.
        with self.assertRaises(ValueError):
            runtime._validate_prepared_conditioning(
                conditioning(cfg=4.0), "prompt", "negative", 4.0, "none"
            )
        # Prepared with one the request will never read.
        with self.assertRaises(ValueError):
            runtime._validate_prepared_conditioning(
                conditioning(negative=torch.ones((1, 2, WIDTH), dtype=torch.bfloat16)),
                "prompt", "negative", 1.0, "none",
            )

    def test_a_malformed_tensor_is_refused(self):
        runtime = make_runtime()
        cases = {
            "type": conditioning(positive="not a tensor"),
            "dtype": conditioning(positive=torch.ones((1, 2, WIDTH), dtype=torch.float32)),
            "rank": conditioning(positive=torch.ones((2, WIDTH), dtype=torch.bfloat16)),
            "batch": conditioning(positive=torch.ones((2, 2, WIDTH), dtype=torch.bfloat16)),
            "contiguous": conditioning(
                positive=torch.ones((1, WIDTH, 2), dtype=torch.bfloat16).transpose(1, 2)
            ),
            "finite": conditioning(
                positive=torch.full((1, 2, WIDTH), float("nan"), dtype=torch.bfloat16)
            ),
        }
        for name, prepared in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    runtime._validate_prepared_conditioning(prepared, "prompt", "negative", 1.0, "none")

    def test_something_else_entirely_is_refused(self):
        runtime = make_runtime()
        for value in (None, object(), (torch.ones((1, 2, WIDTH)), None)):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(ValueError):
                    runtime._validate_prepared_conditioning(value, "prompt", "negative", 1.0, "none")


class SigmaTests(unittest.TestCase):
    def test_a_prepared_schedule_matches_the_one_built_inline(self):
        prepared = prepare_krea2_refinement_sigmas(5, 0.12, "normal")
        self.assertEqual(prepared.device.type, "cpu")
        self.assertEqual(prepared.dtype, torch.float32)
        accepted = validate_prepared_krea2_refinement_sigmas(prepared, 5, 0.12, "normal")
        self.assertTrue(torch.equal(accepted, prepared))
        self.assertIsNot(accepted, prepared)

    def test_a_schedule_from_a_different_request_is_refused(self):
        prepared = prepare_krea2_refinement_sigmas(5, 0.12, "normal")
        for args in ((6, 0.12, "normal"), (5, 0.35, "normal"), (5, 0.12, "simple")):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    validate_prepared_krea2_refinement_sigmas(prepared, *args)

    def test_a_malformed_schedule_is_refused(self):
        good = prepare_krea2_refinement_sigmas(5, 0.12, "normal")
        cases = {
            "type": [0.0] * 6,
            "rank": good.unsqueeze(0),
            "dtype": good.to(torch.float64),
            "length": good[:-1].clone(),
            "terminal": torch.cat([good[:-1], torch.tensor([0.5])]),
            "monotonic": good.flip(0).contiguous(),
        }
        for name, candidate in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    validate_prepared_krea2_refinement_sigmas(candidate, 5, 0.12, "normal")


class RefineBatchTests(unittest.TestCase):
    """The parameters used to be accepted and discarded. These pin that they take effect."""

    def refine(self, runtime, **kwargs):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "is_bf16_supported", return_value=True),
            patch.object(Krea2Runtime, "_run_cuda_stage", lambda self, name, op: op()),
            patch.object(Krea2Runtime, "_require_cuda", lambda self: None),
        ):
            return runtime.refine_batch(
                images=[Image.new("RGB", (64, 64))],
                prompt="prompt", negative_prompt="negative",
                steps=5, denoise=0.12, cfg=1.0,
                sampler="euler", scheduler="normal",
                generators=[torch.Generator(device="cpu").manual_seed(1)],
                **kwargs,
            )

    def test_prepared_conditioning_replaces_the_encode(self):
        runtime = make_runtime()
        prepared = conditioning()
        self.refine(runtime, prepared_conditioning=prepared)
        self.assertEqual(runtime._encode_prompt.call_count, 0, "the prepared pair must be used as-is")
        metrics = runtime.last_generation_metrics["refinement.sampling"]
        self.assertIs(metrics["conditioning_reused"], True)

    def test_without_a_prepared_pair_the_stage_encodes_and_says_so(self):
        runtime = make_runtime()
        self.refine(runtime)
        self.assertEqual(runtime._encode_prompt.call_count, 1)
        metrics = runtime.last_generation_metrics["refinement.sampling"]
        self.assertIs(metrics["conditioning_reused"], False)
        self.assertIs(metrics["sigmas_reused"], False)

    def test_prepared_sigmas_are_used_and_reported(self):
        runtime = make_runtime()
        sigmas = prepare_krea2_refinement_sigmas(5, 0.12, "normal")
        self.refine(runtime, prepared_sigmas=sigmas)
        metrics = runtime.last_generation_metrics["refinement.sampling"]
        self.assertIs(metrics["sigmas_reused"], True)
        used = runtime._sample.call_args.args[2]
        self.assertTrue(torch.equal(used, sigmas))

    def test_a_mismatched_prepared_pair_stops_the_tile(self):
        runtime = make_runtime()
        with self.assertRaises(ValueError):
            self.refine(runtime, prepared_conditioning=conditioning(prompt="different"))
        with self.assertRaises(ValueError):
            self.refine(runtime, prepared_sigmas=prepare_krea2_refinement_sigmas(6, 0.12, "normal"))


class FluxRefusalTests(unittest.TestCase):
    """The Flux engines share the signature but have neither surface, so they refuse rather than
    accept and discard — which is what they used to do, making a prepared pair look effective."""

    def test_both_flux_engines_refuse_a_prepared_pair(self):
        from flux_pipeline import refuse_prepared_refinement

        for engine in ("FLUX.1", "FLUX.2"):
            with self.subTest(engine=engine, argument="conditioning"):
                with self.assertRaises(ValueError) as error:
                    refuse_prepared_refinement(engine, conditioning(), None)
                self.assertIn("prepared_conditioning", str(error.exception))
            with self.subTest(engine=engine, argument="sigmas"):
                with self.assertRaises(ValueError) as error:
                    refuse_prepared_refinement(engine, None, torch.zeros(3))
                self.assertIn("prepared_sigmas", str(error.exception))

    def test_passing_neither_is_the_ordinary_case(self):
        from flux_pipeline import refuse_prepared_refinement

        self.assertIsNone(refuse_prepared_refinement("FLUX.1", None, None))


if __name__ == "__main__":
    unittest.main()
