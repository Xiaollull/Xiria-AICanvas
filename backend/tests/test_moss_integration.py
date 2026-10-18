"""Server contract of the seventh engine, MOSS (QPlan2/MOSS/P1 Phase D): request validation, roots,
health, performance settings, PNG workflow, XYZ planning and memory estimates. CPU only."""

import json
import unittest

from backend import comfy_workflow, memory_policy, performance_settings, xyz_planner
from backend.comfy_workflow import WorkflowFacts, build_workflow, validate_workflow


def server():
    from backend import inference_server

    return inference_server


def moss_request(**overrides):
    fields = {
        "engine": "MOSS", "diffusion_model": "moss_image_sft_int8_convrot_attn.safetensors",
        "text_encoder": "qwen3.5_9b_int8_convrot.safetensors", "vae": "flux2-vae.safetensors",
        "prompt": "a fox in snow", "negative_prompt": "", "width": 1024, "height": 1024, "steps": 50, "cfg": 3,
        "denoise": 1, "seed": 42, "sampler": "euler", "scheduler": "sgm_uniform",
    }
    fields.update(overrides)
    return server().GenerateInput(**fields)


class MossRequestContractTests(unittest.TestCase):
    def test_defaults_are_filled_and_effective(self):
        request = moss_request()
        self.assertEqual(request.moss.version, 1)
        self.assertEqual(request.moss.shift, 5.0)
        self.assertEqual(request.moss.uncond_skip_layers, [22])
        self.assertEqual((request.moss.uncond_skip_start, request.moss.uncond_skip_end), (0.0, 1.0))
        self.assertFalse(request.moss.standard_slg)
        self.assertEqual(request.moss.runtime_options()["uncond_skip_layers"], [22])
        self.assertEqual(server().request_family("MOSS"), "moss")

    def test_explicit_block_round_trips(self):
        request = moss_request(moss={"shift": 3.5, "uncond_skip_layers": [], "uncond_skip_start": 0.1, "uncond_skip_end": 0.9})
        self.assertEqual(request.moss.uncond_skip_layers, [])
        self.assertEqual(server().GenerateInput.model_validate(request.model_dump()).moss, request.moss)

    def test_refusals(self):
        cases = [
            {"sampler": "dpmpp_2m"},
            {"scheduler": "align_your_steps"},
            {"guidance": "pag"},
            {"guidance": "cfg_zero_star"},
            {"moss": {"standard_slg": True}},
            {"moss": {"uncond_skip_layers": [42]}},
            {"moss": {"uncond_skip_layers": [3, 3]}},
            {"moss": {"uncond_skip_start": 0.8, "uncond_skip_end": 0.2}},
            {"moss": {"shift": 0}},
            {"moss": {"version": 2}},
            {"moss": {"unknown": 1}},
            {"checkpoint": "x.safetensors"},
            {"text_encoder_2": "t5.safetensors"},
            {"vae": None},
            {"clip_skip": 2},
            {"scheduler": "kl_optimal", "steps": 1},
            {"width": 1000},
            {"source_image": {"enabled": True, "image_data": "data:image/png;base64,AAAA"}},
            {"postprocess_only": True},
            {"postprocess_chain": [{"type": "resample", "enabled": True, "scale": 1.5, "denoise": 0.4, "execution_mode": "usdu_tiled"}]},
            {"postprocess_chain": [{"type": "resample", "enabled": True, "scale": 1.5, "denoise": 0.4, "latent_algorithm": "latent"}]},
        ]
        for overrides in cases:
            with self.subTest(overrides=list(overrides)), self.assertRaises(Exception):
                moss_request(**overrides)

    def test_refusals_name_their_reason(self):
        cases = {
            "MOSS supports plain CFG only": {"guidance": "pag"},
            "image-to-image": {"source_image": {"enabled": True, "image_data": "data:image/png;base64,AAAA"}},
            "supported only by Anima and Krea2": {"postprocess_chain": [
                {"type": "resample", "enabled": True, "scale": 1.5, "denoise": 0.4, "execution_mode": "usdu_tiled"},
            ]},
            "Unsupported MOSS sampler": {"sampler": "dpmpp_2m"},
            "standard_slg": {"moss": {"standard_slg": True}},
            "block indices": {"moss": {"uncond_skip_layers": [42]}},
        }
        for fragment, overrides in cases.items():
            with self.subTest(fragment=fragment):
                with self.assertRaises(Exception) as caught:
                    moss_request(**overrides)
                self.assertIn(fragment, str(caught.exception))

    def test_pixel_only_stages_stay_available(self):
        request = moss_request(postprocess_chain=[
            {"type": "resample", "enabled": True, "scale": 1.5, "denoise": 0},
            {"type": "rtx", "enabled": True, "scale": 2},
        ])
        self.assertEqual(len(request.postprocess_chain), 2)

    def test_full_frame_hires_and_adetailer_are_accepted(self):
        request = moss_request(postprocess_chain=[
            {"type": "resample", "enabled": True, "scale": 1.5, "denoise": 0.4, "steps": 20,
             "sampler": "euler_ancestral", "scheduler": "simple"},
            {"type": "adetailer", "enabled": True, "units": [{"detector": "face_yolov8n.pt", "denoise": 0.35}]},
        ], moss={"shift": 4.0, "uncond_skip_layers": [10, 22]})
        self.assertTrue(request.resample_units[0].redraws)
        self.assertEqual(request.resample_units[0].execution_mode, "full_frame")
        self.assertEqual(len(request.adetailer.active_units), 1)
        # The redraws read the request's own MOSS block; nothing in the units can drift from it.
        self.assertEqual(request.moss.runtime_options()["uncond_skip_layers"], [10, 22])
        # A unit's own sampler override is checked against the MOSS vocabulary.
        with self.assertRaises(Exception):
            moss_request(postprocess_chain=[{"type": "resample", "enabled": True, "scale": 1.5, "denoise": 0.4, "sampler": "dpmpp_2m"}])
        # The redraw canvas is held to the MOSS edge limit like the base pass.
        with self.assertRaises(Exception):
            moss_request(width=2048, height=2048, postprocess_chain=[{"type": "resample", "enabled": True, "scale": 2.5, "denoise": 0.4}])

    def test_moss_block_is_refused_on_other_engines(self):
        with self.assertRaises(Exception):
            server().GenerateInput(
                engine="Krea2", diffusion_model="d.safetensors", text_encoder="t.safetensors", vae="v.safetensors",
                prompt="x", width=512, height=512, steps=4, cfg=3, denoise=1, seed=1, sampler="euler",
                scheduler="simple", moss={"shift": 5},
            )

    def test_unknown_native_engine_does_not_borrow_anima_roots(self):
        with self.assertRaises(ValueError):
            server().native_model_roots("Unknown")
        roots = server().native_model_roots("MOSS")
        self.assertEqual(roots["lora"].name, "moss")
        self.assertEqual(roots["diffusion_model"], server().anima_model_roots()["diffusion_model"])


class MossHealthTests(unittest.TestCase):
    def test_health_fields_describe_the_contract(self):
        fields = server().moss_health_fields()
        self.assertTrue(fields["tokenizers_ready"], fields["reason"])
        self.assertEqual(fields["samplers"], ["euler", "euler_ancestral", "heun"])
        self.assertFalse(fields["features"]["img2img"])
        self.assertTrue(fields["features"]["hires"])
        self.assertTrue(fields["features"]["adetailer"])
        self.assertFalse(fields["features"]["usdu"])
        self.assertTrue(fields["features"]["resize_only_resample"])
        self.assertFalse(fields["features"]["pag"])
        self.assertTrue(fields["features"]["negative_prompt"])
        self.assertTrue(fields["features"]["lora"])
        self.assertEqual(fields["moss_sampling"]["uncond_skip_layers"]["default"], [22])
        self.assertIn("real-model verified", fields["validation"]["int8_convrot"])
        self.assertIn("not tested", fields["validation"]["bf16"])

    def test_tokenizer_status_detects_tampering(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as root:
            status = server().moss_tokenizer_status(root)
            self.assertFalse(status["installed"])
            (Path(root) / server().MOSS_TOKENIZER_GZ_NAME).write_bytes(b"not a tokenizer")
            self.assertFalse(server().moss_tokenizer_status(root)["installed"])


class MossRefinementRoutingTests(unittest.TestCase):
    """Hires and ADetailer reach MOSS through the native `refine_batch`, carrying the request's MOSS block."""

    class Control:
        cancelled = False

        def checkpoint(self, *_args):
            pass

        def active_elapsed(self, _started_at):
            return 0

        def total_paused(self):
            return 0

    class Runtime:
        def __init__(self):
            self.calls = []
            self.last_generation_metrics = {"sampling": {"seconds": 1.0}}

        def refine_batch(self, **kwargs):
            from PIL import Image

            self.calls.append(kwargs)
            kwargs["on_step"](kwargs["steps"], kwargs["steps"], None)
            self.last_generation_metrics = {
                **self.last_generation_metrics,
                "refinement.sampling": {"seconds": 0.1, "schedule_steps": 40, "schedule_mode": "comfy_suffix", "start_sigma": 0.6},
            }
            return [Image.new("RGB", kwargs["images"][0].size, "white")]

    def test_hires_redraw_passes_the_moss_block(self):
        from unittest.mock import patch

        from PIL import Image

        s = server()
        request = moss_request(width=512, height=512, moss={"shift": 4.0, "uncond_skip_layers": [10]}, postprocess_chain=[
            {"type": "resample", "enabled": True, "upscaler": "model.pth", "scale": 2, "denoise": 0.5, "steps": 20},
        ])
        runtime = self.Runtime()
        s.jobs["moss-hires"] = {"runtime_metrics": {}}
        try:
            with (
                patch.object(s, "pipeline_cpu_parked", False),
                patch.object(s.torch.cuda, "is_available", return_value=True),
                patch.object(s, "upscale_image", return_value=(Image.new("RGB", (128, 128), "green"), {"model": "model.pth"})),
                patch.object(s.AutoPipelineForImage2Image, "from_pipe", side_effect=AssertionError("Diffusers image pipeline used")),
            ):
                result, diagnostics = s.apply_resample_unit(
                    Image.new("RGB", (64, 64), "red"), runtime, "moss", request, request.resample_units[0], "moss-hires",
                    self.Control(), 0, image_seed=17, effective_seed=17,
                )
        finally:
            s.jobs.pop("moss-hires", None)
        (call,) = runtime.calls
        self.assertEqual(call["moss"], {"shift": 4.0, "uncond_skip_layers": [10], "uncond_skip_start": 0.0,
                                        "uncond_skip_end": 1.0, "standard_slg": False})
        self.assertEqual((call["steps"], call["denoise"], call["cfg"]), (20, 0.5, 3))
        self.assertEqual((call["sampler"], call["scheduler"], call["guidance"]), ("euler", "sgm_uniform", "none"))
        self.assertIsNone(call["masks"])
        self.assertNotIn("prepared_conditioning", call)
        # The PAG slider travels with every native redraw, whatever the guidance mode.
        self.assertEqual(call["pag_scale"], request.pag.scale)
        self.assertEqual(call["images"][0].size, (128, 128))
        self.assertEqual(result.size, (128, 128))
        self.assertEqual((diagnostics["schedule_steps"], diagnostics["schedule_mode"]), (40, "comfy_suffix"))

    def test_adetailer_regions_pass_the_moss_block(self):
        from pathlib import Path
        from unittest.mock import patch

        from PIL import Image

        from backend.adetailer import Detection

        s = server()
        request = moss_request(width=512, height=512, moss={"shift": 6.0, "uncond_skip_layers": []}, postprocess_chain=[
            {"type": "adetailer", "enabled": True,
             "units": [{"detector": "face_yolov8n.pt", "denoise": 0.35, "padding": 16, "dilate_erode": 0}]},
        ])
        runtime = self.Runtime()
        s.jobs["moss-detail"] = {"runtime_metrics": {}}
        try:
            with (
                patch.object(s, "resolve_detector_model", return_value=Path("face_yolov8n.pt")),
                patch.object(s, "run_detector", return_value=[Detection((20, 20, 100, 110), 0.9, "face")]),
                patch.object(s.AutoPipelineForInpainting, "from_pipe", side_effect=AssertionError("Diffusers inpaint pipeline used")),
            ):
                result, _diagnostics, warning = s.apply_adetailer(
                    Image.new("RGB", (256, 192), "black"), runtime, "moss", request, "moss-detail", self.Control(), 0,
                    image_seed=100,
                )
        finally:
            s.jobs.pop("moss-detail", None)
        self.assertIsNone(warning)
        (call,) = runtime.calls
        self.assertEqual(call["moss"], request.moss.runtime_options())
        self.assertEqual(call["moss"]["uncond_skip_layers"], [])
        self.assertEqual(call["denoise"], 0.35)
        self.assertEqual(call["generators"][0].initial_seed(), 101)
        crop, mask = call["images"][0], call["masks"][0]
        self.assertEqual(mask.size, crop.size)
        self.assertEqual((crop.width % 16, crop.height % 16), (0, 0))
        self.assertIsNotNone(mask.getbbox())
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))
        self.assertNotEqual(result.getpixel((60, 60)), (0, 0, 0))

    # The fakes above accept any value. These two hand the server's own arguments to the real runtime
    # (the tiny CPU fixture), so whatever the server forwards -- the PAG slider, the MOSS block, the
    # mask -- has to be something the runtime accepts and acts on.

    def test_hires_reaches_the_real_runtime(self):
        from unittest.mock import patch

        from backend.tests.test_moss_runtime import rgb_image, tiny_runtime

        s = server()
        runtime = tiny_runtime()
        request = moss_request(width=512, height=512, steps=4, moss={"shift": 4.0, "uncond_skip_layers": [1]}, postprocess_chain=[
            {"type": "resample", "enabled": True, "scale": 2, "denoise": 0.5, "steps": 3},
        ])
        self.assertGreater(request.pag.scale, 0)
        s.jobs["moss-hires-real"] = {"runtime_metrics": {}}
        try:
            with patch.object(s, "pipeline_cpu_parked", False), patch.object(s.torch.cuda, "is_available", return_value=False):
                result, diagnostics = s.apply_resample_unit(
                    rgb_image(32, 32, 4), runtime, "moss", request, request.resample_units[0], "moss-hires-real",
                    self.Control(), 0, image_seed=17, effective_seed=17,
                )
        finally:
            s.jobs.pop("moss-hires-real", None)
        self.assertEqual(result.size, (64, 64))
        sampling = runtime.last_generation_metrics["refinement.sampling"]
        self.assertEqual((sampling["shift"], sampling["uncond_skip_layers"]), (4.0, [1]))
        self.assertEqual((sampling["schedule_steps"], sampling["executed_denoise_updates"], sampling["masked"]), (6, 3, False))
        self.assertEqual((diagnostics["schedule_steps"], diagnostics["schedule_mode"]), (6, "comfy_suffix"))

    def test_adetailer_reaches_the_real_runtime(self):
        from pathlib import Path
        from unittest.mock import patch

        from backend.adetailer import Detection
        from backend.tests.test_moss_runtime import rgb_image, tiny_runtime

        s = server()
        runtime = tiny_runtime()
        source = rgb_image(64, 64, 6)
        request = moss_request(width=512, height=512, steps=4, moss={"uncond_skip_layers": [1]}, postprocess_chain=[
            {"type": "adetailer", "enabled": True,
             "units": [{"detector": "face_yolov8n.pt", "denoise": 0.4, "use_steps": True, "steps": 2, "padding": 8}]},
        ])
        s.jobs["moss-detail-real"] = {"runtime_metrics": {}}
        try:
            with (
                patch.object(s.torch.cuda, "is_available", return_value=False),
                patch.object(s, "resolve_detector_model", return_value=Path("face_yolov8n.pt")),
                patch.object(s, "run_detector", return_value=[Detection((24, 24, 40, 40), 0.9, "face")]),
            ):
                result, _diagnostics, warning = s.apply_adetailer(
                    source, runtime, "moss", request, "moss-detail-real", self.Control(), 0, image_seed=100,
                )
        finally:
            s.jobs.pop("moss-detail-real", None)
        self.assertIsNone(warning)
        self.assertEqual(result.size, (64, 64))
        sampling = runtime.last_generation_metrics["refinement.sampling"]
        self.assertTrue(sampling["masked"])
        self.assertEqual(sampling["executed_denoise_updates"], 2)
        # Outside the soft mask the picture is the source; inside, the region was redrawn.
        self.assertEqual(result.crop((0, 0, 64, 12)).tobytes(), source.crop((0, 0, 64, 12)).tobytes())
        self.assertNotEqual(result.crop((26, 26, 38, 38)).tobytes(), source.crop((26, 26, 38, 38)).tobytes())


class MossSettingsAndMemoryTests(unittest.TestCase):
    def test_text_encoder_placement_setting(self):
        # Sparse: a file without the key (every file written before MOSS) reads back unchanged, and
        # the absent key means `auto`.
        self.assertNotIn("moss_text_encoder_placement", performance_settings.normalize_performance_settings({}))
        self.assertEqual(performance_settings.MOSS_TEXT_ENCODER_PLACEMENT_DEFAULT, "auto")
        self.assertEqual(performance_settings.normalize_performance_settings({"moss_text_encoder_placement": "cpu"})["moss_text_encoder_placement"], "cpu")
        self.assertNotIn("moss_text_encoder_placement", performance_settings.normalize_performance_settings({"moss_text_encoder_placement": "bogus"}))
        self.assertEqual(server().PerformanceInput().moss_text_encoder_placement, "auto")

    def test_estimates_have_a_moss_entry(self):
        moss = memory_policy.estimate_inference_bytes("moss", 1024, 1024, 3.0, 1, 2)
        generic = memory_policy.estimate_inference_bytes("unknown-family", 1024, 1024, 3.0, 1, 2)
        self.assertNotEqual(moss, generic)
        self.assertEqual(memory_policy.estimate_largest_component_bytes(10 * 1024 ** 3, "moss"), int(10 * 1024 ** 3 * 0.60))


class MossWorkflowTests(unittest.TestCase):
    def facts(self, loras=()):
        return WorkflowFacts(identity="job/1/1", seed=42, positive="a fox in snow", negative="", loras=tuple(loras),
                             clip_skip=None, stages=())

    def test_graph_matches_the_official_order_and_names_its_requirement(self):
        workflow = build_workflow(moss_request(), self.facts([("moss\\style.safetensors", 0.8)]))
        validate_workflow(workflow)
        types = [node["type"] for node in workflow["nodes"]]
        for expected in ("UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly", "ModelSamplingSD3",
                         "SkipLayerGuidanceDiTSimple", "EmptyFlux2LatentImage", "KSampler", "Note"):
            self.assertIn(expected, types)
        self.assertNotIn("LoraLoader", types)
        clip = next(node for node in workflow["nodes"] if node["type"] == "CLIPLoader")
        self.assertIn("moss_image", clip["widgets_values"])
        skip = next(node for node in workflow["nodes"] if node["type"] == "SkipLayerGuidanceDiTSimple")
        self.assertEqual(skip["widgets_values"], ["", "22", 0.0, 1.0])
        shift = next(node for node in workflow["nodes"] if node["type"] == "ModelSamplingSD3")
        self.assertEqual(shift["widgets_values"], [5.0])
        sampler = next(node for node in workflow["nodes"] if node["type"] == "KSampler")
        self.assertEqual(sampler["widgets_values"][2:6], [50, 3.0, "euler", "sgm_uniform"])
        note = next(node for node in workflow["nodes"] if node["type"] == "Note")
        self.assertIn("PR #17095", note["widgets_values"][0])
        # ModelSamplingSD3 feeds the skip node, which feeds the sampler.
        links = {link[0]: link for link in workflow["links"]}
        by_id = {node["id"]: node for node in workflow["nodes"]}
        model_into_sampler = [links[i["link"]] for i in sampler["inputs"] if i["name"] == "model"][0]
        self.assertEqual(by_id[model_into_sampler[1]]["type"], "SkipLayerGuidanceDiTSimple")
        model_into_skip = [links[i["link"]] for i in skip["inputs"] if i["name"] == "model"][0]
        self.assertEqual(by_id[model_into_skip[1]]["type"], "ModelSamplingSD3")

    def test_empty_skip_list_drops_the_skip_node(self):
        workflow = build_workflow(moss_request(moss={"uncond_skip_layers": []}), self.facts())
        self.assertNotIn("SkipLayerGuidanceDiTSimple", [node["type"] for node in workflow["nodes"]])
        self.assertIn("ModelSamplingSD3", [node["type"] for node in workflow["nodes"]])


class MossXYZTests(unittest.TestCase):
    def draft(self, **parameters):
        base = {
            "engine": "MOSS", "diffusion_model": "d.safetensors", "text_encoder": "t.safetensors", "vae": "v.safetensors",
            "prompt": "fox", "negative_prompt": "", "width": 1024, "height": 1024, "steps": 20, "cfg": 3, "seed": "1",
            "sampler": "euler", "scheduler": "sgm_uniform", "moss": {"version": 1, "shift": 5.0, "uncond_skip_layers": [22]},
        }
        base.update(parameters)
        return {"engine": "MOSS", "baseline": {"parameters": base, "prompt_without_groups": "fox"},
                "axes": {"x": {"field": "cfg", "values": [{"value": 2}, {"value": 4}]}}}

    def catalog(self):
        return {("MOSS", "diffusion_model", "d.safetensors"), ("MOSS", "text_encoder", "t.safetensors"), ("MOSS", "vae", "v.safetensors")}

    def test_moss_plans_with_its_own_sampler_vocabulary(self):
        plan = xyz_planner.compile_xyz(self.draft(), model_catalog=self.catalog())
        self.assertEqual(plan.errors, [])
        self.assertEqual(plan.total_cells, 2)
        self.assertEqual(plan.cells[0].request["moss"]["uncond_skip_layers"], [22])
        refused = xyz_planner.compile_xyz(self.draft(sampler="dpmpp_2m"), model_catalog=self.catalog())
        self.assertTrue(refused.errors)

    def test_capability_entry_is_shared_with_the_frontend(self):
        from pathlib import Path

        capabilities = json.loads((Path(__file__).resolve().parents[2] / "src" / "xyz-capabilities.json").read_text(encoding="utf-8"))
        entry = capabilities["engines"]["MOSS"]
        self.assertEqual(entry["guidance"], ["none"])
        self.assertEqual(entry["samplers"], list(server().MOSS_SAMPLERS))
        self.assertEqual(entry["schedulers"], list(server().MOSS_SCHEDULERS))


if __name__ == "__main__":
    unittest.main()
