import math
import unittest

import torch

from backend.anima_sampling import (
    ANIMA_SAMPLERS,
    ANIMA_SCHEDULERS,
    anima_sampling_diagnostics,
    anima_refinement_sigma_schedule,
    anima_sigma_schedule,
    prepare_anima_refinement_sigmas,
    validate_prepared_anima_refinement_sigmas,
)


def comfy_discrete_flow_refinement_oracle(steps: int, denoise: float, scheduler: str) -> torch.Tensor:
    """Independent scalar oracle for the relevant Comfy discrete-flow paths."""
    schedule_steps = int(steps / denoise)

    def sigma(timestep: float) -> float:
        value = timestep / 1000.0
        return 3.0 * value / (1.0 + 2.0 * value)

    if scheduler == "simple":
        model_sigmas = [sigma(index + 1) for index in range(1000)]
        full = [model_sigmas[-(1 + int(x * 1000 / schedule_steps))] for x in range(schedule_steps)] + [0.0]
    elif scheduler == "normal":
        full = [sigma(1000.0 + (1.0 - 1000.0) * index / (schedule_steps - 1)) for index in range(schedule_steps)] + [0.0]
    elif scheduler == "sgm_uniform":
        full = [sigma(1000.0 + (1.0 - 1000.0) * index / schedule_steps) for index in range(schedule_steps)] + [0.0]
    else:
        raise ValueError("oracle only covers Comfy discrete-flow schedulers")
    return torch.tensor(full[-(steps + 1):], dtype=torch.float32)


class AnimaScheduleTests(unittest.TestCase):
    def test_all_schedulers_match_four_step_flow_shift_three_references(self):
        expected = {
            "simple": [1.0, 0.9, 0.75, 0.5, 0.0],
            "sgm_uniform": [1.0, 0.9001200, 0.7503749, 0.5009990, 0.0],
            "karras": [1.0, 0.2327368, 0.0368317, 0.00299401, 0.0],
            "exponential": [1.0, 0.1441289, 0.0207731, 0.00299401, 0.0],
            "ddim_uniform": [0.9009585, 0.7514970, 0.5026596, 0.0059761, 0.0],
            "beta": [1.0, 0.9335347, 0.7507493, 0.3921713, 0.0],
            "normal": [1.0, 0.8573265, 0.6007186, 0.00299401, 0.0],
            "linear_quadratic": [1.0, 0.9875, 0.975, 0.725, 0.0],
            "kl_optimal": [1.0, 0.5786817, 0.2700896, 0.00299401, 0.0],
        }
        self.assertEqual(set(expected), set(ANIMA_SCHEDULERS))
        for scheduler, values in expected.items():
            with self.subTest(scheduler=scheduler):
                actual = anima_sigma_schedule(4, scheduler)
                self.assertTrue(torch.allclose(actual, torch.tensor(values), atol=1e-6, rtol=1e-6))

    def test_every_scheduler_is_finite_monotonic_and_fixed_length(self):
        for scheduler in ANIMA_SCHEDULERS:
            for steps in (1, 2, 28, 60, 100):
                with self.subTest(scheduler=scheduler, steps=steps):
                    sigmas = anima_sigma_schedule(steps, scheduler)
                    self.assertEqual(sigmas.device.type, "cpu")
                    self.assertEqual(sigmas.dtype, torch.float32)
                    self.assertEqual(len(sigmas), steps + 1)
                    self.assertTrue(torch.isfinite(sigmas).all())
                    self.assertTrue(torch.all(sigmas[:-1] >= sigmas[1:]))
                    self.assertEqual(sigmas[-1].item(), 0.0)

    def test_all_visible_sampler_scheduler_pairs_have_explicit_flow_diagnostics(self):
        self.assertEqual(len(ANIMA_SAMPLERS), 44)
        self.assertEqual(len(ANIMA_SCHEDULERS), 9)
        for sampler in ANIMA_SAMPLERS:
            for scheduler in ANIMA_SCHEDULERS:
                with self.subTest(sampler=sampler, scheduler=scheduler):
                    diagnostics = anima_sampling_diagnostics(sampler, scheduler)
                    self.assertEqual(diagnostics["requested_sampler"], sampler)
                    self.assertEqual(diagnostics["requested_scheduler"], scheduler)
                    self.assertTrue(diagnostics["sampler_implementation"])
                    self.assertEqual(diagnostics["scheduler_implementation"], f"rf_{scheduler}_shift_3")

        self.assertIn("standard sequential CFG", anima_sampling_diagnostics("euler_cfg_pp", "simple")["warning"])
        self.assertIn("deterministic CPU-seeded noise", anima_sampling_diagnostics("dpmpp_sde_gpu", "karras")["warning"])
        self.assertIn("maps to Euler", anima_sampling_diagnostics("ddim", "ddim_uniform")["warning"])
        self.assertIn("not LCM-distilled", anima_sampling_diagnostics("lcm", "normal")["warning"])

    def test_unknown_sampler_scheduler_and_invalid_steps_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "sampler"):
            anima_sampling_diagnostics("unknown", "simple")
        with self.assertRaisesRegex(ValueError, "scheduler"):
            anima_sigma_schedule(20, "unknown")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            anima_sigma_schedule(0, "simple")

    def test_comfy_discrete_flow_refinement_suffixes_match_independent_oracle_pointwise(self):
        for scheduler in ("simple", "normal", "sgm_uniform"):
            with self.subTest(scheduler=scheduler):
                sigmas, diagnostics = anima_refinement_sigma_schedule(20, 0.35, scheduler)
                self.assertTrue(torch.allclose(sigmas, comfy_discrete_flow_refinement_oracle(20, 0.35, scheduler), atol=1e-7, rtol=0))
                self.assertEqual((len(sigmas), diagnostics["schedule_steps"]), (21, 57))
                self.assertEqual(diagnostics["schedule_mode"], "comfy_suffix")
        simple, diagnostics = anima_refinement_sigma_schedule(20, 0.35, "simple")
        self.assertAlmostEqual(diagnostics["start_sigma"], 0.618683875, places=6)
        self.assertAlmostEqual(float((simple[:-1] - simple[1:]).max()), 0.05212355, places=6)

        for steps, denoise in ((20, 0.5), (25, 0.5)):
            with self.subTest(scheduler="simple", steps=steps, denoise=denoise):
                sigmas, diagnostics = anima_refinement_sigma_schedule(steps, denoise, "simple")
                self.assertTrue(torch.allclose(
                    sigmas, comfy_discrete_flow_refinement_oracle(steps, denoise, "simple"), atol=1e-7, rtol=0
                ))
                self.assertEqual(len(sigmas), steps + 1)
                self.assertEqual(diagnostics["schedule_steps"], int(steps / denoise))
        self.assertAlmostEqual(
            anima_refinement_sigma_schedule(20, 0.5, "simple")[1]["start_sigma"], 0.75, places=7
        )

    def test_refinement_normal_suffix_full_denoise_and_schedule_limit(self):
        sigmas, diagnostics = anima_refinement_sigma_schedule(20, 0.35, "normal")
        self.assertEqual((len(sigmas), diagnostics["schedule_steps"]), (21, 57))
        self.assertTrue(torch.isfinite(sigmas).all())
        self.assertTrue(torch.all(sigmas[:-1] >= sigmas[1:]))
        full, full_diagnostics = anima_refinement_sigma_schedule(20, 1, "simple")
        self.assertTrue(torch.equal(full, anima_sigma_schedule(20, "simple")))
        self.assertEqual(full_diagnostics["schedule_mode"], "full")
        accepted, accepted_diagnostics = anima_refinement_sigma_schedule(100, 100 / 4096, "normal")
        self.assertEqual((len(accepted), accepted_diagnostics["schedule_steps"]), (101, 4096))
        with self.assertRaisesRegex(ValueError, "4097 schedule steps"):
            anima_refinement_sigma_schedule(100, 100 / 4097, "normal")

    def test_prepared_refinement_sigmas_require_exact_cpu_fp32_contiguous_contract(self):
        expected = prepare_anima_refinement_sigmas(2, 1.0, "normal")
        accepted = validate_prepared_anima_refinement_sigmas(expected, 2, 1.0, "normal")
        self.assertTrue(torch.equal(accepted, expected))
        self.assertIsNot(accepted, expected)
        cases = (
            (expected.to(torch.float64), "dtype"),
            (expected.to(torch.float16), "dtype"),
            (expected.to(torch.bfloat16), "dtype"),
            (torch.stack((expected, expected)).t()[0], "contiguous"),
            (torch.tensor([1.0, float("nan"), 0.0]), "finite"),
            (torch.tensor([1.0, float("inf"), 0.0]), "finite"),
            (torch.tensor([0.5, 0.75, 0.0]), "monotonic"),
            (torch.tensor([1.0, 0.5, 0.1]), "terminate"),
            (prepare_anima_refinement_sigmas(2, 1.0, "simple"), "do not match"),
        )
        for candidate, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_prepared_anima_refinement_sigmas(candidate, 2, 1.0, "normal")
        below_fp32_ulp = expected.to(torch.float64)
        below_fp32_ulp[1] += 1e-12
        with self.assertRaisesRegex(ValueError, "dtype"):
            validate_prepared_anima_refinement_sigmas(below_fp32_ulp, 2, 1.0, "normal")
        if torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError, "CPU"):
                validate_prepared_anima_refinement_sigmas(expected.cuda(), 2, 1.0, "normal")


if __name__ == "__main__":
    unittest.main()
