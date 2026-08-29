import math
import unittest
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anima_sampling import ANIMA_SAMPLERS, ANIMA_SCHEDULERS
from flux_sampling import (
    FLUX_MAX_REFINEMENT_SCHEDULE_STEPS,
    FLUX_SAMPLERS,
    FLUX_SCHEDULERS,
    FLUX_SIGMA_TABLE_SIZE,
    flux_model_sigmas,
    flux_refinement_sigma_schedule,
    flux_resolution_shift,
    flux_sampling_diagnostics,
    flux_sigma_schedule,
    flux_time_shift,
    resolve_flux_sampler,
)


class FluxVocabularyTests(unittest.TestCase):
    def test_the_two_native_engines_expose_the_same_comfyui_ksampler_names(self):
        # Both engines are driven by ComfyUI's KSampler node, so the names have to stay identical
        # even though the schedules behind them differ. Divergence here would present the user
        # with an engine-dependent sampler list nothing in the UI explains.
        self.assertEqual(FLUX_SAMPLERS, ANIMA_SAMPLERS)
        self.assertEqual(FLUX_SCHEDULERS, ANIMA_SCHEDULERS)


class FluxSigmaTableTests(unittest.TestCase):
    def test_the_table_matches_comfyui_model_sampling_flux(self):
        sigmas = flux_model_sigmas(1.15)
        self.assertEqual(len(sigmas), FLUX_SIGMA_TABLE_SIZE)
        self.assertEqual(sigmas.dtype, torch.float32)
        self.assertAlmostEqual(float(sigmas[-1]), 1.0, places=6)
        expected_min = flux_time_shift(1.15, 1.0, 1.0 / FLUX_SIGMA_TABLE_SIZE)
        self.assertAlmostEqual(float(sigmas[0]), expected_min, places=7)
        self.assertTrue(torch.all(sigmas[1:] >= sigmas[:-1]))

    def test_the_default_resolution_shift_is_the_stock_static_shift(self):
        # ComfyUI's blueprint runs without a ModelSamplingFlux node, which means shift 1.15. The
        # node's own interpolation resolves to 1.15 at 1024 x 1024, so the resolution-aware value
        # this engine uses reproduces the blueprint exactly at its default canvas.
        self.assertAlmostEqual(flux_resolution_shift(1024, 1024), 1.15, places=4)
        self.assertLess(flux_resolution_shift(512, 512), 1.15)
        self.assertGreater(flux_resolution_shift(1536, 1536), 1.15)

    def test_the_resolution_shift_rejects_impossible_canvases(self):
        with self.assertRaises(ValueError):
            flux_resolution_shift(0, 1024)
        with self.assertRaises(ValueError):
            flux_resolution_shift(1024.0, 1024)


class FluxScheduleTests(unittest.TestCase):
    def test_every_scheduler_produces_a_descending_trajectory_terminating_at_zero(self):
        for scheduler in FLUX_SCHEDULERS:
            with self.subTest(scheduler=scheduler):
                sigmas = flux_sigma_schedule(20, scheduler, 1.15)
                self.assertGreaterEqual(len(sigmas), 2)
                self.assertEqual(float(sigmas[-1]), 0.0)
                self.assertTrue(torch.all(sigmas[:-1] >= sigmas[1:]))
                self.assertTrue(torch.isfinite(sigmas).all())

    def test_the_requested_step_count_is_the_trajectory_length_for_the_default_schedulers(self):
        for scheduler in ("simple", "normal", "sgm_uniform", "karras", "exponential", "beta", "linear_quadratic", "kl_optimal"):
            with self.subTest(scheduler=scheduler):
                self.assertEqual(len(flux_sigma_schedule(20, scheduler, 1.15)), 21)

    def test_simple_selects_the_table_entries_comfyui_selects(self):
        table = flux_model_sigmas(1.15)
        sigmas = flux_sigma_schedule(4, "simple", 1.15)
        stride = FLUX_SIGMA_TABLE_SIZE / 4
        expected = [float(table[-(1 + int(index * stride))]) for index in range(4)] + [0.0]
        self.assertEqual([float(value) for value in sigmas], expected)

    def test_the_first_sigma_of_a_full_run_is_pure_noise(self):
        for scheduler in ("simple", "normal", "karras", "exponential", "kl_optimal"):
            with self.subTest(scheduler=scheduler):
                self.assertAlmostEqual(float(flux_sigma_schedule(20, scheduler, 1.15)[0]), 1.0, places=5)

    def test_a_single_step_schedule_is_valid_for_every_scheduler(self):
        for scheduler in FLUX_SCHEDULERS:
            with self.subTest(scheduler=scheduler):
                sigmas = flux_sigma_schedule(1, scheduler, 1.15)
                self.assertGreaterEqual(len(sigmas), 2)
                self.assertEqual(float(sigmas[-1]), 0.0)

    def test_a_larger_shift_keeps_more_noise_in_the_early_trajectory(self):
        low = flux_sigma_schedule(20, "simple", 0.5)
        high = flux_sigma_schedule(20, "simple", 3.0)
        self.assertTrue(torch.all(high[1:-1] >= low[1:-1]))

    def test_invalid_requests_are_rejected(self):
        with self.assertRaises(ValueError):
            flux_sigma_schedule(0, "simple", 1.15)
        with self.assertRaises(ValueError):
            flux_sigma_schedule(True, "simple", 1.15)
        with self.assertRaises(ValueError):
            flux_sigma_schedule(20, "not_a_scheduler", 1.15)
        with self.assertRaises(ValueError):
            flux_sigma_schedule(20, "simple", 0.0)
        with self.assertRaises(ValueError):
            flux_sigma_schedule(20, "simple", float("nan"))


class FluxRefinementScheduleTests(unittest.TestCase):
    def test_full_denoise_returns_the_base_schedule(self):
        sigmas, diagnostics = flux_refinement_sigma_schedule(20, 1.0, "simple", 1.15)
        self.assertTrue(torch.equal(sigmas, flux_sigma_schedule(20, "simple", 1.15)))
        self.assertEqual(diagnostics["schedule_mode"], "full")
        self.assertEqual(diagnostics["schedule_steps"], 20)

    def test_partial_denoise_keeps_the_tail_of_a_longer_comfy_schedule(self):
        sigmas, diagnostics = flux_refinement_sigma_schedule(10, 0.5, "simple", 1.15)
        self.assertEqual(diagnostics["schedule_mode"], "comfy_suffix")
        self.assertEqual(diagnostics["schedule_steps"], 20)
        self.assertTrue(torch.equal(sigmas, flux_sigma_schedule(20, "simple", 1.15)[-11:]))

    def test_a_refinement_pass_runs_every_requested_step_whatever_the_denoise(self):
        # This is the property that makes `int(steps * denoise)` the wrong step count for a native
        # Comfy-style refinement: the denoise moves the starting sigma, it does not shorten the run.
        for denoise in (0.2, 0.45, 0.8):
            with self.subTest(denoise=denoise):
                sigmas, _diagnostics = flux_refinement_sigma_schedule(12, denoise, "simple", 1.15)
                self.assertEqual(len(sigmas) - 1, 12)

    def test_a_lower_denoise_starts_from_less_noise(self):
        weak, _ = flux_refinement_sigma_schedule(10, 0.2, "simple", 1.15)
        strong, _ = flux_refinement_sigma_schedule(10, 0.9, "simple", 1.15)
        self.assertLess(float(weak[0]), float(strong[0]))

    def test_a_near_zero_denoise_is_rejected_before_it_builds_an_unbounded_schedule(self):
        with self.assertRaises(ValueError) as error:
            flux_refinement_sigma_schedule(100, 0.001, "simple", 1.15)
        self.assertIn(str(FLUX_MAX_REFINEMENT_SCHEDULE_STEPS), str(error.exception))
        with self.assertRaises(ValueError):
            flux_refinement_sigma_schedule(20, 0.0, "simple", 1.15)
        with self.assertRaises(ValueError):
            flux_refinement_sigma_schedule(20, 1.5, "simple", 1.15)


class FluxSamplerResolutionTests(unittest.TestCase):
    def test_every_sampler_resolves_to_an_implemented_update(self):
        implemented = {"euler", "euler_ancestral", "heun", "midpoint", "multistep", "flow_lcm"}
        for sampler in FLUX_SAMPLERS:
            with self.subTest(sampler=sampler):
                implementation, _warning = resolve_flux_sampler(sampler)
                self.assertIn(implementation, implemented)

    def test_the_exact_samplers_carry_no_compatibility_warning(self):
        for sampler in ("euler", "euler_ancestral", "heun", "dpm_2", "lms"):
            with self.subTest(sampler=sampler):
                self.assertIsNone(resolve_flux_sampler(sampler)[1])

    def test_cfg_pp_names_explain_why_flux_cannot_honour_them(self):
        _implementation, warning = resolve_flux_sampler("euler_cfg_pp")
        self.assertIn("CFG++", warning)
        self.assertIn("guidance-distilled", warning)

    def test_an_unknown_sampler_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_flux_sampler("not_a_sampler")

    def test_diagnostics_report_the_shift_the_run_will_use(self):
        diagnostics = flux_sampling_diagnostics("euler", "simple", flux_resolution_shift(1024, 1024))
        self.assertEqual(diagnostics["requested_sampler"], "euler")
        self.assertEqual(diagnostics["requested_scheduler"], "simple")
        self.assertAlmostEqual(diagnostics["shift"], 1.15, places=3)
        self.assertIn("1.15", diagnostics["scheduler_implementation"])
        self.assertIsNone(diagnostics["warning"])

    def test_diagnostics_reject_an_unknown_scheduler(self):
        with self.assertRaises(ValueError):
            flux_sampling_diagnostics("euler", "not_a_scheduler", 1.15)


if __name__ == "__main__":
    unittest.main()
