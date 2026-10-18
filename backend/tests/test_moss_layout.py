"""MOSS coordinate layouts (QPlan2/MOSS/P2): the official compiler port, the stage transforms, the request
contract and the routing of recompiled layouts into Hires and ADetailer. CPU only."""

import copy
import json
import unittest
from pathlib import Path

from backend import moss_layout
from backend.tests.test_moss_integration import MossRefinementRoutingTests, moss_request, server

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "moss_layout_parity.json").read_text(encoding="utf-8"))

README_LAYOUT = {
    "summary": {"meta": ["poster"], "canvas": "single",
                "caption": "A minimalist travel poster with a cream background, a blue mountain illustration, and the title EXPLORE at the top."},
    "items": [
        {"label": "text", "caption": "Large dark-blue uppercase title centered at the top.", "text": "EXPLORE",
         "bbox": [-416, -384, -288, 384]},
        {"label": "object", "caption": "A flat blue mountain illustration in the lower center.", "bbox": [-192, -288, 384, 288]},
    ],
    "bbox_coord_space": "abs_centralized",
    "bbox_format": "yxyx",
}


def draft(width=1024, height=1024, items=None, summary=None):
    return {
        "width": width, "height": height, "bbox_coord_space": "abs", "bbox_format": "xyxy",
        "summary": summary or {"caption": "A test canvas."},
        "items": items if items is not None else [{"label": "object", "caption": "A box.", "bbox": [0, 0, 512, 512]}],
    }


def readme_draft():
    return next(case["draft"] for case in FIXTURE["cases"] if case["name"] == "readme example")


class OfficialCompilerTests(unittest.TestCase):
    """The vectors of OpenMOSS's `test_compile_prompt.py` and the README layout, byte for byte."""

    def test_readme_layout_is_reproduced_exactly(self):
        expected = json.dumps(README_LAYOUT, ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(moss_layout.compile_draft(readme_draft()), expected)

    def test_landscape_conversion_preserves_content(self):
        source = draft(1536, 1024, [{"label": "text", "caption": "Dark-green lettering.",
                                     "text": '把日常，过成诗。\n"MOSS" 🪴', "bbox": [928, 272, 1408, 448]}],
                       {"meta": ["poster"], "canvas": "single", "caption": "A ceramic campaign."})
        before = copy.deepcopy(source)
        serialized = moss_layout.compile_draft(source)
        result = json.loads(serialized)
        self.assertEqual(result["items"][0]["bbox"], [-240, 160, -64, 640])
        self.assertEqual(result["items"][0]["text"], source["items"][0]["text"])
        self.assertEqual(result["summary"], source["summary"])
        self.assertEqual((result["bbox_coord_space"], result["bbox_format"]), ("abs_centralized", "yxyx"))
        self.assertNotIn("width", result)
        self.assertNotIn("\n", serialized)
        self.assertEqual(source, before)

    def test_portrait_edges_and_fractional_enclosure(self):
        cases = [([0, 0, 768, 1344], [-672, -384, 672, 384]), ([10.9, 20.1, 100.01, 200.99], [-652, -374, -471, -283]),
                 ([1.00000000001, 2, 100, 200], [-670, -383, -472, -284])]
        for box, expected in cases:
            with self.subTest(box=box):
                compiled = json.loads(moss_layout.layout_json(draft(768, 1344, [{"label": "object", "caption": "x", "bbox": box}])))
                self.assertEqual(compiled["items"][0]["bbox"], expected)

    def test_invalid_geometry_is_refused(self):
        cases = [[-1, 0, 100, 100], [0, 0, 1537, 100], [0, 0, 100, 1025], [100, 0, 50, 100], [0, 0, 0, 100],
                 [0, 0, 100], [True, 0, 100, 100], [0, 0, float("nan"), 100], [0, 0, float("inf"), 100],
                 [0, 0, "100", 100], [0, 0, 0.00000000001, 100]]
        for box in cases:
            with self.subTest(box=box), self.assertRaises(moss_layout.LayoutError) as caught:
                moss_layout.compile_draft(draft(1536, 1024, [{"label": "object", "caption": "x", "bbox": box}]))
            self.assertEqual(caught.exception.path, "items[0].bbox")

    def test_invalid_dimensions_and_conventions_are_refused(self):
        for field, value in (("width", 1023), ("height", 0), ("width", True), ("height", 1024.0),
                             ("bbox_coord_space", "abs_centralized"), ("bbox_coord_space", "normalized_1000"),
                             ("bbox_format", "yxyx")):
            source = draft()
            source[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(moss_layout.LayoutError):
                moss_layout.compile_draft(source)

    def test_unrepresentable_content_is_not_dropped(self):
        for field, value in (("children", [{"bbox": [0, 0, 20, 20]}]), ("font_size", 32), ("parent", "cup"), ("id", "a")):
            source = draft()
            source["items"][0][field] = value
            with self.subTest(field=field), self.assertRaises(moss_layout.LayoutError):
                moss_layout.compile_draft(source)
        source = draft(items=[{"label": "text", "caption": "x", "bbox": [0, 0, 10, 10]}])
        with self.assertRaisesRegex(moss_layout.LayoutError, "literal text"):
            moss_layout.compile_draft(source)

    def test_xiria_labels_and_text_rules(self):
        accepted = draft(items=[{"label": label, "caption": "x", "bbox": [0, 0, 64, 64]} for label in ("object", "character", "panel")])
        moss_layout.compile_draft(accepted)
        for label in ("subject", "Object", ""):
            with self.subTest(label=label), self.assertRaisesRegex(moss_layout.LayoutError, "label"):
                moss_layout.compile_draft(draft(items=[{"label": label, "caption": "x", "bbox": [0, 0, 64, 64]}]))
        for caption in ("a\u0001b", "a\rb", "a\ud800b", " 　 ", "x" * (moss_layout.MAX_TEXT_CHARACTERS + 1)):
            with self.subTest(caption=caption[:8]), self.assertRaises(moss_layout.LayoutError):
                moss_layout.compile_draft(draft(items=[{"label": "object", "caption": caption, "bbox": [0, 0, 64, 64]}]))
        too_many = draft(items=[{"label": "object", "caption": "x", "bbox": [0, 0, 64, 64]}] * (moss_layout.MAX_ITEMS + 1))
        with self.assertRaisesRegex(moss_layout.LayoutError, "at most"):
            moss_layout.compile_draft(too_many)

    def test_issues_list_every_problem_with_its_path(self):
        source = draft(items=[{"label": "subject", "caption": "", "bbox": [0, 0, 2000, 10]}, {"label": "text", "caption": "ok", "bbox": [0, 0, 8, 8]}],
                       summary={"canvas": "grid"})
        paths = [issue["path"] for issue in moss_layout.draft_issues(source)]
        self.assertEqual(paths, ["summary.caption", "summary.canvas", "items[0].label", "items[0].caption", "items[0].bbox", "items[1].text"])
        self.assertEqual(moss_layout.draft_issues(draft()), [])

    def test_prefix_joins_on_its_own_line(self):
        body = moss_layout.layout_json(readme_draft())
        self.assertEqual(moss_layout.compile_draft(readme_draft(), "  @nyalia, best quality \n"), f"@nyalia, best quality\n{body}")
        self.assertEqual(moss_layout.compile_draft(readme_draft(), " \n "), body)


class ParityFixtureTests(unittest.TestCase):
    """`scripts/moss-layout-parity.mjs` compiled these with src/moss-layout.js; Python must agree byte for byte."""

    def test_every_case_matches_the_browser(self):
        self.assertGreaterEqual(len(FIXTURE["cases"]), 25)
        for case in FIXTURE["cases"]:
            with self.subTest(case=case["name"]):
                if case["prompt"] is None:
                    self.assertTrue(moss_layout.draft_issues(case["draft"]))
                    continue
                scaled = moss_layout.scale_draft(case["draft"], case["width"], case["height"])
                self.assertEqual(moss_layout.compile_draft(scaled, case["prefix"]), case["prompt"])


class StageTransformTests(unittest.TestCase):
    def test_scaling_doubles_boxes_and_keeps_edges_exact(self):
        compiled = json.loads(moss_layout.layout_json(moss_layout.scale_draft(readme_draft(), 2048, 2048)))
        self.assertEqual([item["bbox"] for item in compiled["items"]], [[-832, -768, -576, 768], [-384, -576, 768, 576]])
        edge = moss_layout.scale_draft(draft(768, 768, [{"label": "panel", "caption": "x", "bbox": [0, 0, 768, 768]}]), 1024, 1024)
        self.assertEqual(edge["items"][0]["bbox"], [0, 0, 1024, 1024])
        same = readme_draft()
        self.assertIs(moss_layout.scale_draft(same, 1024, 1024), same)

    def test_scaled_draft_keeps_the_key_order(self):
        source = readme_draft()
        self.assertEqual(list(moss_layout.scale_draft(source, 512, 512)), list(source))

    def test_aspect_mismatch(self):
        self.assertFalse(moss_layout.aspect_mismatch(readme_draft(), 2048, 2048))
        self.assertTrue(moss_layout.aspect_mismatch(readme_draft(), 2048, 1024))

    def test_region_keeps_the_items_that_belong_to_the_crop(self):
        source = draft(1024, 1024, [
            {"label": "panel", "caption": "Whole frame.", "bbox": [0, 0, 1024, 1024]},
            {"label": "character", "caption": "A woman on the left.", "bbox": [0, 128, 512, 1024]},
            {"label": "text", "caption": "Sign.", "text": "OPEN", "bbox": [700, 64, 960, 192]},
            # 61 % of the cup lies inside the crop below; it is kept and clipped.
            {"label": "object", "caption": "A cup that pokes in.", "bbox": [200, 380, 500, 420]},
            # 12 % of this shelf lies inside, and it covers 5 % of the crop: it belongs to another region.
            {"label": "object", "caption": "A long shelf.", "bbox": [300, 380, 1000, 420]},
        ])
        # The image went through Hires 2x: 2048 square. The face crop is (256, 512)-(768, 1024).
        region = moss_layout.region_draft(source, (2048, 2048), (256, 512, 768, 1024), (1024, 1024))
        self.assertEqual([item["caption"] for item in region["items"]], ["Whole frame.", "A woman on the left.", "A cup that pokes in."])
        self.assertEqual((region["width"], region["height"]), (1024, 1024))
        # The panel and the woman both fill the crop; the cup is clipped to the crop's left part.
        compiled = json.loads(moss_layout.layout_json(region))
        self.assertEqual(compiled["items"][0]["bbox"], [-512, -512, 512, 512])
        self.assertEqual(compiled["items"][1]["bbox"], [-512, -512, 512, 512])
        # On the 2048 image the cup is x 400..1000, y 760..840; the crop keeps x 400..768, scaled 2x from 256.
        self.assertEqual(compiled["items"][2]["bbox"], [-16, -224, 144, 512])
        self.assertEqual(compiled["summary"], source["summary"])

    def test_region_without_items_falls_back_to_the_caption(self):
        source = draft(1024, 1024, [{"label": "object", "caption": "Far away.", "bbox": [900, 900, 1024, 1024]}],
                       {"caption": "A quiet room."})
        self.assertIsNone(moss_layout.region_draft(source, (1024, 1024), (0, 0, 256, 256), (1024, 1024)))
        self.assertEqual(moss_layout.region_prompt(source, "@nyalia", (1024, 1024), (0, 0, 256, 256), (1024, 1024)), "@nyalia\nA quiet room.")

    def test_region_ratio_rule(self):
        source = draft(1024, 1024, [
            {"label": "object", "caption": "Mostly inside.", "bbox": [0, 0, 100, 100]},
            # 12 % of itself inside the 128 crop and 23 % of that crop: neither share reaches 25 %.
            {"label": "object", "caption": "Barely inside.", "bbox": [90, 0, 400, 100]},
            # Only 12.5 % of the band is inside, but it covers 78 % of the 128 crop and exactly 25 % of the tall one.
            {"label": "object", "caption": "Wide band.", "bbox": [0, 0, 1024, 100]},
        ])
        region = moss_layout.region_draft(source, (1024, 1024), (0, 0, 128, 128), (512, 512))
        self.assertEqual([item["caption"] for item in region["items"]], ["Mostly inside.", "Wide band."])
        region = moss_layout.region_draft(source, (1024, 1024), (0, 0, 100, 400), (256, 1024))
        self.assertEqual([item["caption"] for item in region["items"]], ["Mostly inside.", "Wide band."])


def layout_request(layout_draft=None, prefix="", **overrides):
    layout_draft = layout_draft or readme_draft()
    fields = {"moss_layout": {"version": 1, "prefix": prefix, "draft": layout_draft},
              "prompt": moss_layout.compile_draft(layout_draft, prefix),
              "width": layout_draft["width"], "height": layout_draft["height"]}
    fields.update(overrides)
    return moss_request(**fields)


class LayoutRequestContractTests(unittest.TestCase):
    def test_a_compiled_layout_is_accepted_and_round_trips(self):
        request = layout_request(prefix="@nyalia")
        self.assertEqual(request.moss_layout.prompt_at(1024, 1024), request.prompt)
        again = server().GenerateInput.model_validate(request.model_dump())
        self.assertEqual(again.moss_layout.draft, request.moss_layout.draft)
        self.assertEqual(list(again.moss_layout.draft["summary"]), ["meta", "canvas", "caption"])

    def test_refusals_name_their_reason(self):
        s = server()
        cases = [
            ({"prompt": "a fox"}, "not this layout compiled"),
            ({"width": 2048, "height": 2048}, "must equal the requested"),
            ({"moss_layout": {"version": 2, "draft": readme_draft()}}, "version"),
            ({"moss_layout": {"version": 1, "draft": draft(items=[])}}, "items"),
            ({"moss_layout": {"version": 1, "draft": readme_draft(), "extra": 1}}, "extra"),
        ]
        for overrides, fragment in cases:
            with self.subTest(fragment=fragment), self.assertRaisesRegex(Exception, fragment):
                layout_request(**overrides)
        with self.assertRaisesRegex(Exception, "MOSS engine only"):
            s.GenerateInput(
                engine="Krea2", diffusion_model="d.safetensors", text_encoder="t.safetensors", vae="v.safetensors",
                prompt=moss_layout.compile_draft(readme_draft()), width=1024, height=1024, steps=20, cfg=4, denoise=1,
                seed=1, sampler="euler", scheduler="simple", moss_layout={"version": 1, "draft": readme_draft()},
            )
        with self.assertRaisesRegex(Exception, "transparent background"):
            layout_request(prefix="({Transparent background})")

    def test_preflight_counts_base_and_every_hires_canvas(self):
        s = server()
        request = layout_request(postprocess_chain=[
            {"type": "resample", "enabled": True, "scale": 2, "denoise": 0.3, "steps": 10},
        ])
        audit = s.preflight_moss_layout_request(request)
        self.assertEqual(audit["compiler"], "moss_layout_v1")
        self.assertEqual(audit["items"], 2)
        self.assertEqual([(stage["stage"], stage["width"], stage["height"]) for stage in audit["stages"]],
                         [("base", 1024, 1024), ("hires_1", 2048, 2048)])
        self.assertEqual(audit["stages"][0]["tokens"], s.moss_prompt_tokens(request.prompt))
        self.assertEqual(s.moss_prompt_tokens(request.prompt), 142)
        self.assertIsNone(s.preflight_moss_layout_request(moss_request()))

    def test_an_over_window_layout_is_refused_before_a_job_exists(self):
        s = server()
        long_caption = "汉字的长描述。" * 560
        source = draft(items=[{"label": "object", "caption": long_caption, "bbox": [0, 0, 512, 512]},
                              {"label": "object", "caption": long_caption, "bbox": [512, 512, 1024, 1024]}])
        request = layout_request(source)
        with self.assertRaisesRegex(ValueError, "tokens; MOSS reads at most 4096"):
            s.preflight_moss_layout_request(request)

    def test_compile_endpoint(self):
        s = server()
        response = s.compile_moss_layout(s.MossLayoutCompileInput(
            layout={"version": 1, "prefix": "", "draft": readme_draft()}, stages=[[2048, 2048]],
        ))
        self.assertTrue(response["valid"])
        self.assertEqual(response["prompt"], moss_layout.compile_draft(readme_draft()))
        self.assertEqual((response["tokens"], response["max_tokens"]), (142, 4096))
        self.assertEqual(response["stages"][0]["width"], 2048)
        self.assertGreaterEqual(response["stages"][0]["tokens"], 142)
        invalid = s.compile_moss_layout(s.MossLayoutCompileInput(layout={"version": 1, "draft": draft(items=[])}))
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["issues"][0]["path"], "items")
        self.assertIsNone(invalid["prompt"])


class LayoutRoutingTests(unittest.TestCase):
    Control = MossRefinementRoutingTests.Control
    Runtime = MossRefinementRoutingTests.Runtime

    def test_hires_encodes_the_layout_recompiled_for_its_canvas(self):
        from unittest.mock import patch

        from PIL import Image

        s = server()
        request = layout_request(draft(512, 512, [{"label": "object", "caption": "A box.", "bbox": [128, 128, 384, 384]}]),
                                 postprocess_chain=[{"type": "resample", "enabled": True, "upscaler": "model.pth", "scale": 2,
                                                     "denoise": 0.5, "steps": 20}])
        runtime = self.Runtime()
        s.jobs["moss-layout-hires"] = {"runtime_metrics": {}}
        try:
            with (
                patch.object(s, "pipeline_cpu_parked", False),
                patch.object(s.torch.cuda, "is_available", return_value=True),
                patch.object(s, "upscale_image", return_value=(Image.new("RGB", (1024, 1024), "green"), {"model": "model.pth"})),
            ):
                s.apply_resample_unit(Image.new("RGB", (512, 512), "red"), runtime, "moss", request, request.resample_units[0],
                                      "moss-layout-hires", self.Control(), 0, image_seed=17, effective_seed=17)
        finally:
            s.jobs.pop("moss-layout-hires", None)
        (call,) = runtime.calls
        self.assertEqual(call["images"][0].size, (1024, 1024))
        self.assertNotEqual(call["prompt"], request.prompt)
        self.assertEqual(call["prompt"], request.moss_layout.prompt_at(1024, 1024))
        self.assertEqual(json.loads(call["prompt"])["items"][0]["bbox"], [-256, -256, 256, 256])

    def run_detail(self, request, image_size=(512, 512), detection=(64, 64, 192, 192)):
        from pathlib import Path
        from unittest.mock import patch

        from PIL import Image

        from backend.adetailer import Detection

        s = server()
        runtime = self.Runtime()
        s.jobs["moss-layout-detail"] = {"runtime_metrics": {}}
        try:
            with (
                patch.object(s, "resolve_detector_model", return_value=Path("face_yolov8n.pt")),
                patch.object(s, "run_detector", return_value=[Detection(detection, 0.9, "face")]),
            ):
                s.apply_adetailer(Image.new("RGB", image_size, "black"), runtime, "moss", request, "moss-layout-detail",
                                  self.Control(), 0, image_seed=100)
        finally:
            s.jobs.pop("moss-layout-detail", None)
        (call,) = runtime.calls
        return call

    def detail_request(self, unit_prompt=""):
        source = draft(512, 512, [
            {"label": "character", "caption": "A woman, top left.", "bbox": [0, 0, 256, 256]},
            {"label": "object", "caption": "A lamp, bottom right.", "bbox": [400, 400, 512, 512]},
        ], {"caption": "A cosy room."})
        return layout_request(source, prefix="@nyalia", postprocess_chain=[
            {"type": "adetailer", "enabled": True,
             "units": [{"detector": "face_yolov8n.pt", "prompt": unit_prompt, "denoise": 0.35, "padding": 0, "dilate_erode": 0}]},
        ])

    def test_adetailer_region_encodes_its_crop_local_layout(self):
        call = self.run_detail(self.detail_request())
        prefix, body = call["prompt"].split("\n", 1)
        self.assertEqual(prefix, "@nyalia")
        region = json.loads(body)
        self.assertEqual([item["caption"] for item in region["items"]], ["A woman, top left."])
        width, height = call["images"][0].size
        self.assertEqual(region["items"][0]["bbox"], [-height // 2, -width // 2, height // 2, width // 2])

    def test_adetailer_unit_prompts_keep_their_meaning(self):
        self.assertEqual(self.run_detail(self.detail_request("detailed face, sharp eyes"))["prompt"], "detailed face, sharp eyes")
        placed = self.run_detail(self.detail_request("[PROMPT]\ndetailed face"))["prompt"]
        self.assertTrue(placed.startswith("@nyalia\n{"))
        self.assertTrue(placed.endswith("\ndetailed face"))

    def test_text_requests_are_untouched(self):
        request = moss_request(width=512, height=512, postprocess_chain=[
            {"type": "adetailer", "enabled": True, "units": [{"detector": "face_yolov8n.pt", "denoise": 0.35, "padding": 0}]},
        ])
        self.assertEqual(self.run_detail(request)["prompt"], request.prompt)
        self.assertEqual(server().moss_stage_prompt(request, (1024, 1024)), request.prompt)


class LayoutWorkflowTests(unittest.TestCase):
    def notes(self, request):
        from backend.comfy_workflow import WorkflowFacts, build_workflow

        facts = WorkflowFacts(identity="job/1/1", seed=42, positive=request.prompt, negative="", loras=(),
                              clip_skip=None, stages=())
        workflow = build_workflow(request, facts)
        return " ".join(str(value) for node in workflow["nodes"] if node["type"] == "Note" for value in node["widgets_values"])

    def test_hires_layout_workflows_say_the_samplers_reuse_the_base_layout(self):
        request = layout_request(postprocess_chain=[{"type": "resample", "enabled": True, "scale": 2, "denoise": 0.3, "steps": 10}])
        self.assertIn("re-encoded it for each Hires", self.notes(request))
        self.assertNotIn("re-encoded it for each Hires", self.notes(layout_request()))


if __name__ == "__main__":
    unittest.main()
