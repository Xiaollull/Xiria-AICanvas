import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flux_sampling import FLUX_SAMPLERS, FLUX_SCHEDULERS, flux_resolution_shift, flux_sigma_schedule
from krea2_sampling import (
    KREA2_MAX_REFINEMENT_SCHEDULE_STEPS,
    KREA2_MULTIPLIER,
    KREA2_SAMPLERS,
    KREA2_SCHEDULERS,
    KREA2_SEQUENCE_DIVISOR,
    KREA2_SHIFT,
    krea2_refinement_sigma_schedule,
    krea2_resolution_shift,
    krea2_sampling_diagnostics,
    krea2_sequence_length,
    krea2_sigma_schedule,
    resolve_krea2_sampler,
)


def comfy_model_sampling_flux(shift, steps):
    """``ModelSamplingFlux`` plus the ``simple`` scheduler, reimplemented for comparison.

    ``set_parameters`` builds a 10 000-entry table of ``exp(mu) / (exp(mu) + (1/t - 1))`` and the
    simple scheduler walks it backwards at a fixed stride, appending zero.
    """
    timesteps = torch.arange(1, 10001, 1) / 10000
    table = (math.exp(shift) / (math.exp(shift) + (1 / timesteps - 1))).to(torch.float32)
    stride = len(table) / steps
    selected = [table[-(1 + int(index * stride))] for index in range(steps)]
    return torch.cat((torch.stack(selected), torch.zeros(1)))


class VocabularyTests(unittest.TestCase):
    def test_krea2_exposes_the_same_ksampler_vocabulary_as_the_flux_engines(self):
        # One ComfyUI node drives every native engine; a name that exists for one and not another
        # would be a UI that lies about what the backend accepts.
        self.assertEqual(KREA2_SAMPLERS, FLUX_SAMPLERS)
        self.assertEqual(KREA2_SCHEDULERS, FLUX_SCHEDULERS)

    def test_the_refinement_ceiling_is_shared(self):
        self.assertEqual(KREA2_MAX_REFINEMENT_SCHEDULE_STEPS, 4096)


class StaticShiftTests(unittest.TestCase):
    def test_the_shift_is_the_one_krea2s_model_config_declares(self):
        # `comfy/supported_models.py::Krea2.sampling_settings`.
        self.assertEqual(KREA2_SHIFT, 1.15)
        self.assertEqual(KREA2_MULTIPLIER, 1.0)

    def test_the_shift_does_not_move_with_the_canvas(self):
        # ComfyUI ships no Krea 2 scheduler node, so a bare KSampler graph runs one shift at every
        # resolution. Following FLUX.1's canvas interpolation instead would triple it at 2048.
        for width, height in ((512, 512), (1024, 1024), (2048, 2048), (1024, 1536)):
            with self.subTest(size=(width, height)):
                diagnostics = krea2_sampling_diagnostics("euler", "simple", width, height)
                self.assertEqual(diagnostics["shift"], round(KREA2_SHIFT, 4))
                self.assertEqual(diagnostics["shift_source"], "comfy_sampling_settings")

    def test_the_token_geometry_is_flux1s(self):
        # Patch 2 over a stride-8 latent, which is why FLUX.1's interpolation agrees at 1024.
        self.assertEqual(KREA2_SEQUENCE_DIVISOR, 256)
        self.assertEqual(krea2_sequence_length(1024, 1024), 4096)
        self.assertEqual(krea2_sequence_length(1024, 1536), 6144)

    def test_the_reported_flux_comparison_agrees_at_1024_and_diverges_above_it(self):
        self.assertAlmostEqual(krea2_resolution_shift(1024, 1024), KREA2_SHIFT, places=6)
        self.assertEqual(krea2_resolution_shift(1024, 1024), flux_resolution_shift(1024, 1024))
        self.assertGreater(krea2_resolution_shift(2048, 2048), 3.0)

    def test_the_diagnostics_report_both_the_used_and_the_comparison_shift(self):
        diagnostics = krea2_sampling_diagnostics("euler", "simple", 2048, 2048)
        self.assertEqual(diagnostics["shift"], round(KREA2_SHIFT, 4))
        self.assertNotEqual(diagnostics["flux_resolution_shift"], diagnostics["shift"])
        self.assertEqual(diagnostics["sequence_length"], 16384)
        self.assertEqual(diagnostics["scheduler_implementation"], "rf_simple_shift_1.15")

    def test_an_invalid_canvas_is_refused(self):
        for width, height in ((0, 512), (512, -1)):
            with self.subTest(size=(width, height)):
                with self.assertRaises(ValueError):
                    krea2_sequence_length(width, height)
        with self.assertRaises(ValueError):
            krea2_sequence_length(512.0, 512)


class ScheduleTests(unittest.TestCase):
    def test_the_simple_schedule_matches_comfyuis_table(self):
        for steps in (1, 8, 20, 50):
            with self.subTest(steps=steps):
                self.assertTrue(torch.allclose(
                    krea2_sigma_schedule(steps, "simple"), comfy_model_sampling_flux(KREA2_SHIFT, steps)
                ))

    def test_every_scheduler_produces_a_descending_trajectory_that_reaches_zero(self):
        for scheduler in KREA2_SCHEDULERS:
            with self.subTest(scheduler=scheduler):
                sigmas = krea2_sigma_schedule(20, scheduler)
                self.assertGreaterEqual(len(sigmas), 2)
                self.assertEqual(float(sigmas[-1]), 0.0)
                self.assertTrue(torch.all(sigmas[:-1] >= sigmas[1:]))

    def test_the_schedule_is_flux1s_table_evaluated_at_krea2s_shift(self):
        # The engines share `ModelSamplingFlux`; only where the shift comes from differs.
        for scheduler in KREA2_SCHEDULERS:
            with self.subTest(scheduler=scheduler):
                self.assertTrue(torch.equal(
                    krea2_sigma_schedule(12, scheduler), flux_sigma_schedule(12, scheduler, KREA2_SHIFT)
                ))

    def test_an_unknown_scheduler_is_refused_by_name(self):
        with self.assertRaises(ValueError) as error:
            krea2_sigma_schedule(10, "not_a_scheduler")
        self.assertIn("Krea2 scheduler", str(error.exception))


class RefinementScheduleTests(unittest.TestCase):
    def test_full_denoise_returns_the_base_schedule(self):
        sigmas, diagnostics = krea2_refinement_sigma_schedule(10, 1.0, "simple")
        self.assertEqual(diagnostics["schedule_mode"], "full")
        self.assertTrue(torch.equal(sigmas, krea2_sigma_schedule(10, "simple")))

    def test_partial_denoise_keeps_the_tail_of_a_longer_trajectory(self):
        # Comfy KSampler semantics: build int(steps / denoise) then keep the final steps + 1, so a
        # refinement performs every requested step whatever the denoise.
        sigmas, diagnostics = krea2_refinement_sigma_schedule(10, 0.5, "simple")
        self.assertEqual(diagnostics["schedule_mode"], "comfy_suffix")
        self.assertEqual(diagnostics["schedule_steps"], 20)
        self.assertEqual(len(sigmas), 11)
        self.assertTrue(torch.equal(sigmas, krea2_sigma_schedule(20, "simple")[-11:]))
        self.assertLess(diagnostics["start_sigma"], 1.0)

    def test_a_denoise_that_would_build_an_unbounded_schedule_is_refused(self):
        with self.assertRaises(ValueError) as error:
            krea2_refinement_sigma_schedule(100, 0.001, "simple")
        self.assertIn("schedule steps", str(error.exception))

    def test_an_unknown_scheduler_is_refused_by_name(self):
        with self.assertRaises(ValueError) as error:
            krea2_refinement_sigma_schedule(10, 0.5, "not_a_scheduler")
        self.assertIn("Krea2 scheduler", str(error.exception))


class SamplerResolutionTests(unittest.TestCase):
    def test_every_name_resolves_to_an_implemented_update(self):
        implementations = {"euler", "euler_ancestral", "heun", "midpoint", "multistep", "flow_lcm"}
        for sampler in KREA2_SAMPLERS:
            with self.subTest(sampler=sampler):
                implementation, _warning = resolve_krea2_sampler(sampler)
                self.assertIn(implementation, implementations)

    def test_the_exact_samplers_carry_no_warning(self):
        for sampler in ("euler", "euler_ancestral", "heun", "dpm_2", "lms"):
            with self.subTest(sampler=sampler):
                self.assertIsNone(resolve_krea2_sampler(sampler)[1])

    def test_the_distillation_note_does_not_claim_krea2_is_guidance_distilled(self):
        # FLUX.1's wording is a statement about *guidance* distillation and would be false here:
        # Krea 2 runs a real unconditional branch.
        _implementation, warning = resolve_krea2_sampler("lcm")
        self.assertIn("LCM-distilled", warning)
        self.assertNotIn("guidance-distilled", warning)

    def test_cfg_pp_is_explained_by_implementation_rather_than_by_architecture(self):
        _implementation, warning = resolve_krea2_sampler("euler_cfg_pp")
        self.assertIn("CFG++", warning)
        self.assertIn("after the branches", warning)

    def test_an_unknown_sampler_is_refused_by_name(self):
        with self.assertRaises(ValueError) as error:
            resolve_krea2_sampler("not_a_sampler")
        self.assertIn("Krea2 sampler", str(error.exception))


if __name__ == "__main__":
    unittest.main()
