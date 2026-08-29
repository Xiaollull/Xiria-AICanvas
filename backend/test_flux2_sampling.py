import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flux_sampling import FLUX_SAMPLERS, FLUX_SCHEDULERS
from flux2_sampling import (
    FLUX2_MAX_REFINEMENT_SCHEDULE_STEPS,
    FLUX2_MU_SEQUENCE_PLATEAU,
    FLUX2_SAMPLERS,
    FLUX2_SCHEDULERS,
    FLUX2_STATIC_SHIFT,
    flux2_empirical_mu,
    flux2_refinement_sigma_schedule,
    flux2_resolution_shift,
    flux2_sampling_diagnostics,
    flux2_sequence_length,
    flux2_sigma_schedule,
    resolve_flux2_sampler,
)


def comfy_flux2_schedule(steps, sequence_length):
    """`comfy_extras/nodes_flux.py::get_schedule`, reimplemented independently for comparison."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if sequence_length > 4300:
        mu = a2 * sequence_length + b2
    else:
        m_200 = a2 * sequence_length + b2
        m_10 = a1 * sequence_length + b1
        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * steps + b
    timesteps = torch.linspace(1, 0, steps + 1)
    return math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1))


class VocabularyTests(unittest.TestCase):
    def test_both_flux_engines_expose_the_same_ksampler_vocabulary(self):
        # One ComfyUI node drives all of them; a name that exists for one engine and not the other
        # would be a UI that lies about what the backend accepts.
        self.assertEqual(FLUX2_SAMPLERS, FLUX_SAMPLERS)
        self.assertEqual(FLUX2_SCHEDULERS, FLUX_SCHEDULERS)


class EmpiricalShiftTests(unittest.TestCase):
    def test_the_sequence_length_is_the_packed_token_count(self):
        self.assertEqual(flux2_sequence_length(1024, 1024), 4096)
        self.assertEqual(flux2_sequence_length(1024, 1536), 6144)
        self.assertEqual(flux2_sequence_length(16, 16), 1)

    def test_the_fit_matches_comfyui_below_the_plateau(self):
        for sequence, steps in ((4096, 20), (1024, 8), (256, 50)):
            with self.subTest(sequence=sequence, steps=steps):
                a1, b1 = 8.73809524e-05, 1.89833333
                a2, b2 = 0.00016927, 0.45666666
                m_200 = a2 * sequence + b2
                m_10 = a1 * sequence + b1
                slope = (m_200 - m_10) / 190.0
                expected = slope * steps + (m_200 - 200.0 * slope)
                self.assertAlmostEqual(flux2_empirical_mu(sequence, steps), expected, places=10)

    def test_past_the_plateau_the_step_count_drops_out(self):
        sequence = FLUX2_MU_SEQUENCE_PLATEAU + 1
        self.assertEqual(flux2_empirical_mu(sequence, 4), flux2_empirical_mu(sequence, 400))
        self.assertAlmostEqual(flux2_empirical_mu(sequence, 20), 0.00016927 * sequence + 0.45666666, places=10)

    def test_the_default_canvas_lands_near_the_static_shift_comfyui_registers(self):
        # `supported_models.Flux2` carries shift 2.02 for a bare KSampler; the fit's own answer at
        # the default canvas and step count has to be in the same place or one of them is wrong.
        self.assertAlmostEqual(flux2_resolution_shift(1024, 1024, 20), FLUX2_STATIC_SHIFT, delta=0.25)

    def test_the_shift_stays_positive_across_the_whole_supported_range(self):
        for width, height, steps in ((16, 16, 100), (4096, 4096, 1), (1024, 1024, 100), (256, 256, 100)):
            with self.subTest(width=width, height=height, steps=steps):
                self.assertGreater(flux2_resolution_shift(width, height, steps), 0.0)

    def test_a_degenerate_canvas_is_refused(self):
        with self.assertRaises(ValueError):
            flux2_resolution_shift(0, 1024, 20)
        with self.assertRaises(ValueError):
            flux2_resolution_shift(1024, 1024, 0)


class ScheduleParityTests(unittest.TestCase):
    def test_the_simple_scheduler_reproduces_the_flux2_scheduler_node(self):
        # `Flux2Scheduler` evaluates the same expression the sigma table holds, at the same uniform
        # timesteps `simple` selects — so `simple` at the fitted mu *is* the reference schedule.
        for width, height, steps in ((1024, 1024, 20), (1024, 1024, 25), (1536, 1024, 50)):
            with self.subTest(width=width, height=height, steps=steps):
                shift = flux2_resolution_shift(width, height, steps)
                actual = flux2_sigma_schedule(steps, "simple", shift)
                expected = comfy_flux2_schedule(steps, flux2_sequence_length(width, height))
                self.assertEqual(len(actual), steps + 1)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_a_step_count_the_table_cannot_divide_evenly_still_tracks_the_node(self):
        shift = flux2_resolution_shift(1024, 1024, 7)
        actual = flux2_sigma_schedule(7, "simple", shift)
        expected = comfy_flux2_schedule(7, 4096)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-4))

    def test_every_scheduler_produces_a_monotonic_trajectory_ending_at_zero(self):
        shift = flux2_resolution_shift(1024, 1024, 12)
        for scheduler in FLUX2_SCHEDULERS:
            with self.subTest(scheduler=scheduler):
                sigmas = flux2_sigma_schedule(12, scheduler, shift)
                self.assertGreaterEqual(len(sigmas), 2)
                self.assertEqual(float(sigmas[-1]), 0.0)
                self.assertTrue(torch.all(sigmas[:-1] >= sigmas[1:]))

    def test_an_unknown_scheduler_is_refused(self):
        with self.assertRaises(ValueError):
            flux2_sigma_schedule(10, "mystery", 2.0)
        with self.assertRaises(ValueError):
            flux2_refinement_sigma_schedule(10, 0.5, "mystery", 2.0)


class RefinementScheduleTests(unittest.TestCase):
    def test_a_partial_denoise_keeps_the_tail_of_a_longer_trajectory(self):
        shift = flux2_resolution_shift(1024, 1024, 10)
        sigmas, diagnostics = flux2_refinement_sigma_schedule(10, 0.5, "simple", shift)
        self.assertEqual(diagnostics["schedule_mode"], "comfy_suffix")
        self.assertEqual(diagnostics["schedule_steps"], 20)
        # Every requested step still runs; the denoise chose where on the trajectory it starts.
        self.assertEqual(len(sigmas) - 1, 10)
        self.assertLess(float(sigmas[0]), 1.0)

    def test_a_denoise_that_would_build_an_unbounded_schedule_is_refused(self):
        # The construction is `int(steps / denoise)` and runs on the CPU; FLUX.2 inherits FLUX.1's
        # ceiling rather than declaring a second one that could drift from it.
        shift = flux2_resolution_shift(1024, 1024, 100)
        with self.assertRaises(ValueError) as error:
            flux2_refinement_sigma_schedule(100, 0.001, "simple", shift)
        self.assertIn(str(FLUX2_MAX_REFINEMENT_SCHEDULE_STEPS), str(error.exception))

    def test_a_full_denoise_returns_the_base_schedule(self):
        shift = flux2_resolution_shift(1024, 1024, 10)
        sigmas, diagnostics = flux2_refinement_sigma_schedule(10, 1.0, "simple", shift)
        self.assertEqual(diagnostics["schedule_mode"], "full")
        self.assertTrue(torch.equal(sigmas, flux2_sigma_schedule(10, "simple", shift)))


class SamplerResolutionTests(unittest.TestCase):
    def test_the_compatibility_warnings_name_flux2(self):
        _implementation, warning = resolve_flux2_sampler("lcm")
        self.assertIn("FLUX.2", warning)
        self.assertNotIn("FLUX.1", warning)

    def test_euler_carries_no_warning(self):
        self.assertEqual(resolve_flux2_sampler("euler"), ("euler", None))

    def test_an_unknown_sampler_is_refused(self):
        with self.assertRaises(ValueError):
            resolve_flux2_sampler("mystery")

    def test_the_diagnostics_report_both_the_fitted_and_the_static_shift(self):
        diagnostics = flux2_sampling_diagnostics("euler", "simple", flux2_resolution_shift(1024, 1024, 20))
        self.assertEqual(diagnostics["requested_sampler"], "euler")
        self.assertEqual(diagnostics["requested_scheduler"], "simple")
        self.assertEqual(diagnostics["static_shift"], FLUX2_STATIC_SHIFT)
        self.assertGreater(diagnostics["shift"], 0.0)
        self.assertIsNone(diagnostics["warning"])


if __name__ == "__main__":
    unittest.main()
