import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from pydantic import ValidationError

from backend import inference_server
from backend.guidance import (
    PAG_APPLIED_LAYERS,
    PAG_SCALE,
    apply_cfg_zero_star,
    apply_pag,
    cfg_zero_star_scale,
    cfg_zero_star_zero_steps,
    guidance_diagnostics,
    guidance_prediction_copies,
    pag_layer_pattern,
)


def generation_request(**overrides):
    values = {
        "engine": "SD",
        "checkpoint": "model.safetensors",
        "prompt": "test",
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg": 7,
        "denoise": 1,
        "seed": 1,
        "sampler": "euler",
        "scheduler": "simple",
    }
    values.update(overrides)
    return inference_server.GenerateInput(**values)


class GuidanceTests(unittest.TestCase):
    def test_request_defaults_to_none_and_accepts_pag(self):
        self.assertEqual(generation_request().guidance, "none")
        request = generation_request(guidance="pag")
        self.assertEqual(request.guidance, "pag")
        self.assertEqual(request.pag.scale, 0.3)
        self.assertEqual(request.pag.applied_layers, "mid")
        custom = generation_request(guidance="pag", pag={"scale": 0.8, "applied_layers": "all"})
        self.assertEqual(custom.pag.scale, 0.8)
        self.assertEqual(custom.pag.applied_layers, "all")
        with self.assertRaises(ValidationError):
            generation_request(guidance="pag", pag={"scale": 5.1, "applied_layers": "mid"})

    def test_cfg_zero_star_rejects_current_non_flow_engines(self):
        with self.assertRaisesRegex(ValidationError, "Flow Matching"):
            generation_request(guidance="cfg_zero_star")

    def test_cfg_zero_star_uses_optimized_scale_and_four_percent_zero_init(self):
        self.assertEqual(cfg_zero_star_zero_steps(20), 1)
        self.assertEqual(cfg_zero_star_zero_steps(50), 2)
        conditioned = torch.tensor([[[[2.0, 4.0]]]])
        unconditioned = torch.tensor([[[[1.0, 1.0]]]])
        self.assertTrue(torch.equal(apply_cfg_zero_star(conditioned, unconditioned, 7, 0, 20), torch.zeros_like(conditioned)))
        scale = cfg_zero_star_scale(conditioned, unconditioned)
        self.assertTrue(torch.allclose(scale, torch.tensor([[[[3.0]]]])))
        expected = unconditioned * scale + 7 * (conditioned - unconditioned * scale)
        self.assertTrue(torch.allclose(apply_cfg_zero_star(conditioned, unconditioned, 7, 1, 20), expected))

    def test_pag_diagnostics_and_memory_budget_use_three_predictions(self):
        diagnostics = guidance_diagnostics("pag", 20, 0.8, "all")
        self.assertEqual(diagnostics["scale"], 0.8)
        self.assertEqual(diagnostics["applied_layers"], "all")
        self.assertEqual(diagnostics["attention_override"], "all")
        self.assertEqual(PAG_SCALE, 0.3)
        self.assertEqual(PAG_APPLIED_LAYERS, "mid")
        self.assertEqual(pag_layer_pattern("mid"), "mid")
        self.assertEqual(pag_layer_pattern("all"), ".*")
        self.assertEqual(guidance_prediction_copies("pag", 7), 3)
        self.assertEqual(guidance_prediction_copies("pag", 1), 2)
        self.assertEqual(guidance_prediction_copies("none", 1), 1)
        self.assertEqual(guidance_prediction_copies("cfg_zero_star", 1), 2)
        normal = inference_server.estimate_inference_bytes("sdxl", 1024, 1024, 7, 1, 2)
        pag = inference_server.estimate_inference_bytes("sdxl", 1024, 1024, 7, 1, 3)
        self.assertGreater(pag, normal)

    def test_native_pag_formula_and_diagnostics_cover_cfg_and_non_cfg(self):
        conditioned = torch.tensor([2.0])
        perturbed = torch.tensor([0.5])
        unconditioned = torch.tensor([1.0])
        self.assertTrue(torch.equal(apply_pag(conditioned, perturbed, 0.4), torch.tensor([2.6])))
        self.assertTrue(
            torch.equal(
                apply_pag(conditioned, perturbed, 0.4, unconditioned=unconditioned, guidance_scale=3.0),
                torch.tensor([4.6]),
            )
        )
        self.assertTrue(torch.equal(apply_pag(conditioned, perturbed, 0.0), conditioned))
        diagnostics = guidance_diagnostics(
            "pag", 20, 0.4, "mid", engine="Anima", guidance_scale=3.0
        )
        self.assertEqual(diagnostics["implementation"], "native_cosmos_identity_self_attention")
        self.assertEqual(diagnostics["resolved_layers"], ["transformer_blocks.14.attn1"])
        self.assertEqual(diagnostics["logical_prediction_branches"], 3)
        self.assertEqual(diagnostics["peak_forward_copies"], 1)
        zero_diagnostics = guidance_diagnostics(
            "pag", 20, 0.0, "mid", engine="Anima", guidance_scale=1.0
        )
        self.assertEqual(zero_diagnostics["logical_prediction_branches"], 1)
        self.assertEqual(zero_diagnostics["resolved_layers"], [])
        self.assertEqual(zero_diagnostics["resolved_layer_count"], 0)

    def test_pag_pipeline_uses_native_diffusers_adapter(self):
        base = SimpleNamespace()
        guided = SimpleNamespace(set_progress_bar_config=lambda **_kwargs: None)
        settings = inference_server.PagInput(scale=0.8, applied_layers="all")
        with patch.object(inference_server.AutoPipelineForText2Image, "from_pipe", return_value=guided) as from_pipe:
            result, kwargs = inference_server.sampling_pipeline(base, "pag", settings)
        self.assertIs(result, guided)
        self.assertEqual(kwargs, {"pag_scale": 0.8})
        from_pipe.assert_called_once_with(
            base,
            enable_pag=True,
            pag_applied_layers=".*",
        )

    def test_pag_derived_pipeline_uses_safe_default_configuration(self):
        base = SimpleNamespace()
        guided = SimpleNamespace(set_progress_bar_config=lambda **_kwargs: None)
        factory = SimpleNamespace(from_pipe=lambda *_args, **_kwargs: guided)
        with patch.object(factory, "from_pipe", return_value=guided) as from_pipe:
            result, kwargs = inference_server.derived_sampling_pipeline(base, factory, "pag")
        self.assertIs(result, guided)
        self.assertEqual(kwargs, {"pag_scale": PAG_SCALE})
        from_pipe.assert_called_once_with(
            base,
            enable_pag=True,
            pag_applied_layers=PAG_APPLIED_LAYERS,
        )

    def test_pag_restores_shared_attention_processors_after_failure(self):
        original = object()

        class Unet:
            def __init__(self):
                self.attn_processors = {"layer.processor": original}

            def set_attn_processor(self, processors):
                self.attn_processors = dict(processors)

        base = SimpleNamespace(unet=Unet())

        class Guided:
            def __call__(self, **_kwargs):
                base.unet.attn_processors = {"layer.processor": object()}
                raise RuntimeError("sampling failed")

        with patch.object(inference_server, "sampling_pipeline", return_value=(Guided(), {"pag_scale": PAG_SCALE})):
            with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                inference_server.run_guided_pipeline_batch(base, "pag", {"num_images_per_prompt": 1})
        self.assertIs(base.unet.attn_processors["layer.processor"], original)


if __name__ == "__main__":
    unittest.main()
