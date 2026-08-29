import unittest

from diffusers import DDPMScheduler, EulerDiscreteScheduler, IPNDMScheduler

from backend.inference_server import configure_scheduler


SAMPLERS = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2",
    "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast",
    "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm",
    "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp", "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde",
    "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2",
]
SCHEDULERS = ["simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal"]


class Pipeline:
    def __init__(self):
        self.scheduler = DDPMScheduler()


class SchedulerConfigurationTests(unittest.TestCase):
    def configure(self, sampler, scheduler="karras"):
        pipeline = Pipeline()
        warning = configure_scheduler(pipeline, sampler, scheduler)
        return pipeline.scheduler, warning

    def test_all_visible_sampler_and_scheduler_combinations_configure(self):
        for sampler in SAMPLERS:
            for scheduler in SCHEDULERS:
                with self.subTest(sampler=sampler, scheduler=scheduler):
                    self.configure(sampler, scheduler)

    def test_cfg_pp_is_not_silently_treated_as_native(self):
        scheduler, warning = self.configure("euler_cfg_pp")
        self.assertIsInstance(scheduler, EulerDiscreteScheduler)
        self.assertIn("standard CFG", warning)

    def test_variant_and_k_diffusion_approximations_are_reported(self):
        scheduler, warning = self.configure("ipndm_v")
        self.assertIsInstance(scheduler, IPNDMScheduler)
        self.assertIn("approximated", warning)
        _, warning = self.configure("sa_solver_pece")
        self.assertIn("k-diffusion", warning)

    def test_unknown_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported sampler"):
            self.configure("not-a-sampler")
        with self.assertRaisesRegex(ValueError, "Unsupported scheduler"):
            self.configure("euler", "not-a-scheduler")


if __name__ == "__main__":
    unittest.main()
