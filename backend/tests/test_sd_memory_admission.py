"""SD/iL admission: which offload rung a job lands on, and how its guidance branches are run.

The workload behind every number here is the one the bug was reported against — an RTX 4060 Laptop
(8188 MiB), `waiIllustriousSDXL_v170` with six LoRAs, a 768x1280 base and a 1152x1920 hires redraw —
so these tests fail when the admission arithmetic moves, not when a round number is edited.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from backend import inference_server
from backend.inference_server import GenerateInput
from backend.memory_policy import GIB, estimate_vae_decode_bytes

CARD_BYTES = 8585216000          # 8188 MiB, as `torch.cuda.get_device_properties` reports it
WEIGHT_BYTES = int(6.461 * GIB)  # measured weight bytes of the checkpoint
UNET_BYTES = int(4.78 * GIB)     # its largest single component
LORA_BYTES = int(0.52 * GIB)     # the six adapters the reference request mounts
BASE_CANVAS = (768, 1280)
REDRAW_CANVAS = (1152, 1920)
# Any file will do: `choose_memory_strategy` only stats the checkpoint when it has to estimate the
# weight bytes, and every test here supplies the measured ones.
CHECKPOINT = Path(__file__)


def strategy(family="sdxl", canvas=REDRAW_CANVAS, cfg=5.0, free_gb=6.9, guidance="none",
             lora_bytes=LORA_BYTES, memory_mode="sdxl_balanced", total_bytes=CARD_BYTES):
    width, height = canvas
    copies = inference_server.guidance_prediction_copies(guidance, cfg)
    with patch.object(inference_server, "effective_cuda_free_memory",
                      return_value=(int(free_gb * GIB), total_bytes, total_bytes)), \
            patch.dict(inference_server.performance_settings, {"memory_mode": memory_mode}), \
            patch.object(inference_server, "RESERVED_VRAM_GB", None):
        return inference_server.choose_memory_strategy(
            CHECKPOINT, family, width, height, cfg, lora_bytes, 1, copies,
            base_weight_bytes=WEIGHT_BYTES,
            base_largest_component_bytes=UNET_BYTES,
            resident_first=family in inference_server.SD_COMPONENT_FAMILIES,
            cfg_serial_eligible=inference_server.serial_cfg_eligible(family, guidance),
        )


class SdAdmissionTests(unittest.TestCase):
    def test_the_reference_job_runs_component_offload_with_serial_guidance(self):
        # What the bug produced instead: `low_vram`, `enable_sequential_cpu_offload`, 8.11 s/it.
        selected = strategy()
        self.assertEqual(selected["mode"], "normal_vram")
        self.assertEqual(selected["offload_mode"], "model")
        self.assertEqual(selected["cfg_mode"], "serial")
        self.assertEqual(selected["physical_forward_copies"], 1)
        self.assertEqual(selected["logical_guidance_copies"], 2)
        self.assertFalse(selected["cfg_batch"])

    def test_the_base_canvas_lands_on_the_same_rung_as_the_redraw(self):
        # Admission happens once for the whole job, so the cheap pass must not choose differently
        # from the expensive one it will be loaded against.
        self.assertEqual(strategy(canvas=BASE_CANVAS)["mode"], "normal_vram")
        self.assertEqual(strategy(canvas=BASE_CANVAS)["cfg_mode"], "serial")

    def test_the_downgrade_reason_quotes_what_the_requested_mode_needs(self):
        # The old text always quoted HIGH's requirement, which is why a NORMAL request on this card
        # reported "about 11.2 GB" for a component path that wanted 7.8.
        squeezed = strategy(free_gb=2.0)
        self.assertEqual(squeezed["mode"], "low_vram")
        self.assertIn(f"{squeezed['normal_required_bytes'] / GIB:.1f} GB", squeezed["reason"])
        self.assertNotIn(f"{squeezed['high_required_bytes'] / GIB:.1f} GB", squeezed["reason"])

    def test_a_card_with_room_keeps_batching_the_guidance_branches(self):
        # Serial CFG is the answer to a margin that is not there; it must not become the rule on
        # hardware that can pay for one forward instead of two.
        roomy = strategy(free_gb=22.0, total_bytes=24 * GIB)
        self.assertEqual(roomy["cfg_mode"], "batched")
        self.assertEqual(roomy["physical_forward_copies"], 2)
        self.assertTrue(roomy["cfg_batch"])

    def test_a_job_without_classifier_free_guidance_is_neither_batched_nor_split(self):
        single = strategy(cfg=1.0)
        self.assertEqual(single["cfg_mode"], "batched")
        self.assertEqual(single["logical_guidance_copies"], 1)
        self.assertFalse(single["cfg_batch"])

    def test_pag_keeps_its_batch_because_its_attention_divides_it(self):
        # `PAGCFGIdentitySelfAttnProcessor2_0` chunks `hidden_states` into three inside the UNet, so
        # one sample per forward is a batch it cannot divide.
        self.assertFalse(inference_server.serial_cfg_eligible("sdxl", "pag"))
        self.assertTrue(inference_server.serial_cfg_eligible("sdxl", "none"))
        self.assertFalse(inference_server.serial_cfg_eligible("anima", "none"))
        self.assertEqual(strategy(guidance="pag")["cfg_mode"], "batched")

    def test_the_native_engines_keep_the_workspace_upfront_rule(self):
        # `choose_anima_memory_strategy`, `load_flux_pipeline` and `load_krea2_pipeline` call the
        # same function and pass neither flag; their staged runtimes are written against that rule.
        for family in ("flux", "flux2", "krea2", "anima"):
            native = strategy(family=family, lora_bytes=0)
            self.assertEqual(native["admission"]["normal_admission_rule"], "workspace_upfront")
            self.assertEqual(native["cfg_mode"], "batched")
            self.assertFalse(native["cfg_serial_eligible"])

    def test_the_admission_record_says_which_rule_answered(self):
        admission = strategy()["admission"]
        self.assertEqual(admission["normal_admission_rule"], "resident_first")
        self.assertEqual(admission["cfg_mode"], "serial")
        self.assertEqual(admission["physical_forward_copies"], 1)
        self.assertGreater(admission["cfg_batch_inference_bytes"], admission["serial_cfg_inference_bytes"])
        self.assertLessEqual(admission["normal_required_bytes"], admission["budget_total_bytes"])


class CfgExecutionModeTests(unittest.TestCase):
    """What `apply_memory_strategy` installs on the UNet, and what it takes back off."""

    class Unet:
        def __init__(self):
            self.batch_sizes = []

        def forward(self, sample, **kwargs):
            self.batch_sizes.append(sample.shape[0])
            return (sample,)

    class Pipeline:
        def __init__(self, unet):
            self.unet = unet

    def pipeline(self):
        unet = self.Unet()
        return self.Pipeline(unet), unet

    def test_a_serial_job_takes_one_forward_per_branch_and_a_batched_one_takes_them_together(self):
        pipe, unet = self.pipeline()
        inference_server.apply_cfg_execution_mode(pipe, {"mode": "normal_vram", "cfg_mode": "serial"})
        unet.forward(torch.zeros((2, 1)))
        inference_server.apply_cfg_execution_mode(pipe, {"mode": "normal_vram", "cfg_mode": "batched"})
        unet.forward(torch.zeros((2, 1)))
        self.assertEqual(unet.batch_sizes, [1, 1, 2])

    def test_ultra_low_still_splits_whatever_it_is_given_except_pag(self):
        pipe, unet = self.pipeline()
        inference_server.apply_cfg_execution_mode(
            pipe, {"mode": "ultra_low_vram", "cfg_mode": "batched", "cfg_serial_eligible": True}
        )
        unet.forward(torch.zeros((2, 1)))
        # PAG is the exception: splitting its batch is what breaks it, so survival mode leaves it.
        inference_server.apply_cfg_execution_mode(
            pipe, {"mode": "ultra_low_vram", "cfg_mode": "batched", "cfg_serial_eligible": False}
        )
        unet.forward(torch.zeros((3, 1)))
        self.assertEqual(unet.batch_sizes, [1, 1, 3])

    def test_a_pipeline_without_a_unet_is_left_alone(self):
        inference_server.apply_cfg_execution_mode(object(), {"mode": "normal_vram", "cfg_mode": "serial"})

    def test_a_rung_change_cannot_bury_the_previous_wrapper_under_a_new_hook(self):
        # A hook-less rung leaves the wrapper in place, and accelerate captures whatever `forward`
        # it finds. Installed in that order, the wrapper ends up *below* the hook, where removing it
        # is no longer possible — and a job admitted for batched CFG would keep splitting silently.
        unet = self.Unet()

        class Pipeline:
            def __init__(self, inner):
                self.unet = inner

            def to(self, *_args, **_kwargs):
                pass

            def enable_model_cpu_offload(self, device=None):
                # What `add_hook_to_module` does to `forward`, in one line.
                captured = self.unet.forward
                self.unet._old_forward = captured
                self.unet.forward = lambda *args, **kwargs: captured(*args, **kwargs)

        pipe = Pipeline(unet)
        pipe.vae = SimpleNamespace(
            disable_slicing=lambda: None, disable_tiling=lambda: None,
            enable_slicing=lambda: None, enable_tiling=lambda: None,
        )
        inference_server.apply_memory_strategy(pipe, {"mode": "high_vram", "cfg_mode": "serial"})
        unet.forward(torch.zeros((2, 1)))
        inference_server.apply_memory_strategy(pipe, {"mode": "normal_vram", "cfg_mode": "batched"})
        unet.forward(torch.zeros((2, 1)))
        self.assertEqual(unet.batch_sizes, [1, 1, 2])


class VaeDecodeFitTests(unittest.TestCase):
    """Whether a decode runs whole-image or tiled, measured rather than read off the offload rung.

    Both figures below were recorded on the reference card with this checkpoint's autoencoder, as
    `torch.cuda.max_memory_allocated` around a batch-of-one decode: 3.84 GiB at 768x1280 and 8.24
    GiB at 1152x1920. The second is what a Windows driver serves out of host memory rather than
    refusing, which is how a 27-second tiled decode became a 105-second whole-image one.
    """

    class Vae:
        config = SimpleNamespace(block_out_channels=(128, 256, 512, 512))
        use_slicing = False

    def latents(self, canvas):
        return torch.zeros((1, 4, canvas[1] // 8, canvas[0] // 8))

    def fits(self, canvas, free_gb):
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(inference_server, "effective_cuda_free_memory",
                             return_value=(int(free_gb * GIB), CARD_BYTES, CARD_BYTES)), \
                patch.object(inference_server, "RESERVED_VRAM_GB", None):
            return inference_server.whole_image_decode_fits(self.latents(canvas), self.Vae())

    def test_the_base_canvas_decodes_whole_and_the_redraw_does_not(self):
        self.assertTrue(self.fits(BASE_CANVAS, 6.9))
        self.assertFalse(self.fits(REDRAW_CANVAS, 6.9))

    def test_a_card_with_room_decodes_the_redraw_whole_too(self):
        self.assertTrue(self.fits(REDRAW_CANVAS, 22.0))

    def test_the_estimate_covers_both_measured_peaks(self):
        measured = {BASE_CANVAS: 3.84, REDRAW_CANVAS: 8.24}
        for canvas, peak in measured.items():
            estimate = estimate_vae_decode_bytes(*canvas) / GIB
            self.assertAlmostEqual(estimate, peak, delta=0.05)
        # Slicing caps the peak at one image; without it the batch multiplies. (The two differ by a
        # byte or two: one truncates the product, the other four times the truncated figure.)
        self.assertAlmostEqual(
            estimate_vae_decode_bytes(*BASE_CANVAS, images_per_batch=4, sliced=False),
            estimate_vae_decode_bytes(*BASE_CANVAS) * 4,
            delta=8,
        )
        self.assertEqual(
            estimate_vae_decode_bytes(*BASE_CANVAS, images_per_batch=4, sliced=True),
            estimate_vae_decode_bytes(*BASE_CANVAS),
        )

    def test_a_machine_without_a_device_keeps_whatever_was_configured(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertTrue(inference_server.whole_image_decode_fits(self.latents(REDRAW_CANVAS), self.Vae()))


class CudaMathTests(unittest.TestCase):
    def test_balanced_math_keeps_tf32_but_never_benchmarks_convolutions(self):
        # The benchmark chose each convolution's algorithm by timing it in the current process, so a
        # fixed seed drew a different SD/iL picture after every restart. Measured: two fresh
        # processes with it on produced different latents; with it off, identical ones — and the
        # off side was faster, because the benchmark was the cold-process tax on every new shape.
        original_setting = inference_server.performance_settings["cuda_math"]
        original_matmul_tf32 = inference_server.torch.backends.cuda.matmul.allow_tf32
        original_cudnn_tf32 = inference_server.torch.backends.cudnn.allow_tf32
        original_benchmark = inference_server.torch.backends.cudnn.benchmark
        try:
            inference_server.performance_settings["cuda_math"] = "balanced"
            inference_server.configure_cuda_math()
            self.assertTrue(inference_server.torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(inference_server.torch.backends.cudnn.allow_tf32)
            self.assertFalse(inference_server.torch.backends.cudnn.benchmark)
        finally:
            inference_server.performance_settings["cuda_math"] = original_setting
            inference_server.torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
            inference_server.torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
            inference_server.torch.backends.cudnn.benchmark = original_benchmark


class WorkloadDiagnosticsTests(unittest.TestCase):
    def request(self, guidance="none", **overrides):
        fields = dict(
            engine="iL", checkpoint="waiIllustriousSDXL_v170.safetensors", prompt="a fox",
            width=BASE_CANVAS[0], height=BASE_CANVAS[1], steps=28, cfg=5.0, denoise=1.0, seed=1,
            sampler="euler_ancestral", scheduler="simple", guidance=guidance,
            postprocess_chain=[{
                "type": "resample", "enabled": True, "upscaler": "RealESRGAN_x4plus_anime_6B.pth",
                "scale": 1.5, "resize_algorithm": "lanczos", "denoise": 0.3, "tiled_decode": True,
            }],
        )
        fields.update(overrides)
        return GenerateInput(**fields)

    def test_the_chain_is_ranked_on_one_physical_forward_when_the_branches_are_split(self):
        diagnostics = inference_server.generation_memory_workload_diagnostics(self.request(), "sdxl")
        self.assertTrue(diagnostics["cfg_serial_eligible"])
        self.assertEqual(diagnostics["guidance_copies"], 1)
        # The hires redraw is still the stage the whole chain is admitted against.
        self.assertEqual(diagnostics["admission"][:2], REDRAW_CANVAS)
        self.assertEqual(diagnostics["stage"], "full_frame_target")
        self.assertEqual(diagnostics["target_dimensions"], REDRAW_CANVAS)

    def test_pag_is_still_ranked_on_the_batch_it_will_actually_build(self):
        diagnostics = inference_server.generation_memory_workload_diagnostics(self.request(guidance="pag"), "sdxl")
        self.assertFalse(diagnostics["cfg_serial_eligible"])
        self.assertEqual(diagnostics["guidance_copies"], 3)


if __name__ == "__main__":
    unittest.main()
