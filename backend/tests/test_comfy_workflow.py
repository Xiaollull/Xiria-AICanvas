"""P7: the ComfyUI workflow embedded in every text-to-image PNG."""

import json
import re
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest import mock

from PIL import Image

from backend import comfy_workflow
from backend.comfy_workflow import (
    ResampleStage,
    WorkflowFacts,
    build_workflow,
    comfy_asset_name,
    lora_group_label,
    lora_rows,
    validate_workflow,
)

# Appendix A of the plan, verbatim.
APPENDIX_A = {
    1: (1,), 2: (2,), 3: (2, 1), 4: (2, 2), 5: (2, 2, 1), 6: (2, 2, 2), 7: (3, 2, 2), 8: (3, 3, 2),
    9: (3, 3, 3), 10: (3, 3, 3, 1), 11: (3, 3, 3, 2), 12: (3, 3, 3, 3), 13: (4, 4, 4, 1),
    14: (4, 4, 4, 2), 15: (4, 4, 4, 3), 16: (4, 4, 4, 4), 17: (4, 4, 4, 4, 1), 18: (4, 4, 4, 4, 2),
    19: (4, 4, 4, 4, 3), 20: (4, 4, 4, 4, 4),
}


def server():
    from backend import inference_server

    return inference_server


def request(**overrides):
    explicit = set(overrides)
    fields = {
        "engine": "iL", "checkpoint": "waiIllustriousSDXL_v170.safetensors", "prompt": "masterpiece, 1girl",
        "negative_prompt": "low quality", "width": 768, "height": 1152, "steps": 27, "cfg": 5, "denoise": 1,
        "seed": 1234567890, "sampler": "euler_ancestral", "scheduler": "normal",
    }
    fields.update(overrides)
    engine = fields["engine"]
    if engine not in ("SD", "iL"):
        fields.pop("checkpoint", None)
        fields.setdefault("diffusion_model", f"{engine.lower()}-model.safetensors")
        fields.setdefault("text_encoder", f"{engine.lower()}-te.safetensors")
        fields.setdefault("vae", f"{engine.lower()}-vae.safetensors")
        if engine == "Flux":
            fields.setdefault("text_encoder_2", "t5xxl_fp8.safetensors")
        if engine in ("Flux", "Flux2"):
            fields["negative_prompt"] = ""
            if "cfg" not in explicit:
                fields["cfg"] = 3.5
            fields["sampler"], fields["scheduler"] = "euler", "simple"
        if engine in ("Anima", "Krea2"):
            fields["sampler"], fields["scheduler"] = "euler_ancestral", "simple"
    return server().GenerateInput(**fields)


def latent_unit(number=1, **overrides):
    values = dict(
        number=number, redraws=True, tiled=False, upscaler=None, latent_algorithm="latent-bicubic",
        resize_algorithm="lanczos", denoise=0.35, steps=16, cfg=5.0, sampler="euler_ancestral",
        scheduler="normal", seed=987654321, source_size=(768, 1152), target_size=(1152, 1728),
        redraw_size=(1152, 1728), output_size=(1152, 1728),
    )
    values.update(overrides)
    return ResampleStage(**values)


def pixel_unit(number=2, **overrides):
    values = dict(
        number=number, redraws=True, tiled=False, upscaler="RealESRGAN_x4plus_anime_6B.pth", latent_algorithm=None,
        resize_algorithm="lanczos", denoise=0.3, steps=14, cfg=5.0, sampler="euler_ancestral", scheduler="normal",
        seed=1122334455, source_size=(1152, 1728), target_size=(1296, 1944), redraw_size=(1296, 1944),
        output_size=(1296, 1944),
    )
    values.update(overrides)
    return ResampleStage(**values)


def facts(count=0, stages=(), **overrides):
    """``count`` placeholder LoRAs, unless ``loras=`` names them."""
    values = dict(
        identity="job-1/1/1", seed=1234567890, positive="masterpiece, 1girl", negative="low quality",
        loras=tuple((f"Style9X/LoRA-{index + 1:02d}.safetensors", 1.0) for index in range(count)),
        clip_skip=2, stages=tuple(stages),
    )
    values.update(overrides)
    return WorkflowFacts(**values)


def titles(workflow):
    return [group["title"] for group in workflow["groups"]]


def nodes_of(workflow, node_type):
    return [node for node in workflow["nodes"] if node["type"] == node_type]


def subgraphs(workflow):
    return workflow.get("definitions", {}).get("subgraphs", [])


def inner(definition, node_type):
    return [node for node in definition["nodes"] if node["type"] == node_type]


def named(node):
    return node["widgets_values_named"]


def overlaps(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


class SectionTests(unittest.TestCase):
    def test_omitted_sections_do_not_consume_numbers(self):
        cases = {
            (0, ()): ["01 / Model & VAE", "02 / Prompts & CLIP", "03 / Base Sampler", "04 / Final Output"],
            (6, ()): ["01 / Model & VAE", "02 / LoRA · 6 (2 × 3)", "03 / Prompts & CLIP", "04 / Base Sampler", "05 / Final Output"],
            (0, (latent_unit(),)): ["01 / Model & VAE", "02 / Prompts & CLIP", "03 / Base Sampler", "04 / NSampler", "05 / Final Output"],
            (6, (latent_unit(), pixel_unit())): [
                "01 / Model & VAE", "02 / LoRA · 6 (2 × 3)", "03 / Prompts & CLIP", "04 / Base Sampler",
                "05 / NSampler", "06 / Final Output",
            ],
        }
        for (count, stages), expected in cases.items():
            with self.subTest(loras=count, cards=len(stages)):
                workflow = build_workflow(request(), facts(count, stages))
                self.assertEqual(titles(workflow), expected)
                self.assertEqual(len(nodes_of(workflow, "LoraLoader")), count)
                self.assertEqual(bool(subgraphs(workflow)), bool(stages))

    def test_the_reference_layout_is_reproduced(self):
        # P7-Workflow-iL.json: six LoRAs, one latent and one model-upscale card.
        workflow = build_workflow(request(), facts(6, (latent_unit(), pixel_unit())))
        bounds = [group["bounding"] for group in workflow["groups"]]
        self.assertEqual(bounds, [
            [0, 0, 360, 750], [360, 0, 690, 750], [1050, 0, 430, 750], [1480, 0, 400, 750],
            [1880, 0, 450, 750], [2330, 0, 400, 750],
        ])
        loras = nodes_of(workflow, "LoraLoader")
        self.assertEqual([node["pos"] for node in loras], [[390, 80], [720, 80], [390, 300], [720, 300], [390, 520], [720, 520]])
        self.assertEqual(nodes_of(workflow, "CLIPTextEncode")[0]["pos"], [1090, 210])
        self.assertEqual(nodes_of(workflow, "KSampler")[0]["pos"], [1520, 210])
        cards = [node for node in workflow["nodes"] if node["type"] in {item["id"] for item in subgraphs(workflow)}]
        self.assertEqual([node["pos"] for node in cards], [[1890, 70], [2100, 70]])
        self.assertEqual([node["title"] for node in cards], ["5.1 / 2Sampler", "5.2 / 3Sampler"])
        self.assertEqual(nodes_of(workflow, "VAEDecodeTiled")[0]["pos"], [2360, 70])

    def test_groups_are_contiguous_equal_height_and_hold_their_nodes(self):
        for count in (0, 1, 3, 7, 13, 19, 64):
            with self.subTest(loras=count):
                workflow = build_workflow(request(), facts(count, (latent_unit(), pixel_unit(3))))
                groups = workflow["groups"]
                self.assertEqual(len({group["bounding"][3] for group in groups}), 1)
                for left, right in zip(groups, groups[1:]):
                    self.assertEqual(left["bounding"][0] + left["bounding"][2], right["bounding"][0])
                boxes = [node["pos"] + node["size"] for node in workflow["nodes"]]
                for first, second in combinations(boxes, 2):
                    self.assertFalse(overlaps(first, second), (first, second))
                for box in boxes:
                    holders = [group for group in groups if overlaps(box, group["bounding"])]
                    self.assertEqual(len(holders), 1, box)
                    gx, gy, gw, gh = holders[0]["bounding"]
                    self.assertTrue(gx <= box[0] and box[0] + box[2] <= gx + gw and box[1] + box[3] <= gy + gh, box)


class LoraGridTests(unittest.TestCase):
    def test_appendix_a_and_its_extrapolation(self):
        for count, rows in APPENDIX_A.items():
            self.assertEqual(lora_rows(count), rows, count)
        self.assertEqual(lora_rows(21), (4, 4, 4, 4, 4, 1))
        self.assertEqual(lora_rows(24), (4,) * 6)
        self.assertEqual(lora_rows(64), (4,) * 16)
        self.assertEqual(lora_rows(0), ())
        for count in range(1, 70):
            self.assertLessEqual(max(lora_rows(count)), 4)
            self.assertEqual(sum(lora_rows(count)), count)

    def test_group_labels_follow_the_reference_files(self):
        self.assertEqual(lora_group_label(6), "LoRA · 6 (2 × 3)")
        self.assertEqual(lora_group_label(3), "LoRA · 3 (2 + 1)")
        self.assertEqual(lora_group_label(19), "LoRA · 19")
        self.assertEqual(lora_group_label(64), "LoRA · 64")
        self.assertEqual(lora_group_label(1), "LoRA · 1")

    def test_group_width_and_height_grow_with_the_grid(self):
        for count, width, height in ((1, 360, 750), (6, 690, 750), (19, 1350, 1180), (64, 1350, 3600)):
            with self.subTest(loras=count):
                workflow = build_workflow(request(), facts(count))
                lora_group = workflow["groups"][1]
                self.assertEqual(lora_group["bounding"][2:], [width, height])
                self.assertEqual(workflow["groups"][2]["bounding"][0], 360 + width)

    def test_cards_chain_in_mount_order_and_carry_the_weight(self):
        loras = (("Style9X/a.safetensors", 0.5), ("Style9X/b.safetensors", -1.25), ("c.safetensors", 1.0))
        workflow = build_workflow(request(), facts(0, loras=loras))
        cards = nodes_of(workflow, "LoraLoader")
        self.assertEqual([named(node)["lora_name"] for node in cards], ["Style9X\\a.safetensors", "Style9X\\b.safetensors", "c.safetensors"])
        self.assertEqual([(named(node)["strength_model"], named(node)["strength_clip"]) for node in cards], [(0.5, 0.5), (-1.25, -1.25), (1.0, 1.0)])
        links = {link[0]: link for link in workflow["links"]}
        checkpoint = nodes_of(workflow, "CheckpointLoaderSimple")[0]
        self.assertEqual(links[cards[0]["inputs"][0]["link"]][1], checkpoint["id"])
        self.assertEqual(links[cards[1]["inputs"][0]["link"]][1], cards[0]["id"])
        self.assertEqual(links[cards[2]["inputs"][1]["link"]][1], cards[1]["id"])
        # The last card feeds the CLIP skip node and the sampler; the middle ones only the next card.
        self.assertEqual(len(cards[1]["outputs"][0]["links"]), 1)
        skip = nodes_of(workflow, "CLIPSetLastLayer")[0]
        self.assertEqual(links[skip["inputs"][0]["link"]][1], cards[2]["id"])
        self.assertEqual(cards[0]["title"], "LoRA 1 · a")


class NSamplerTests(unittest.TestCase):
    def test_latent_and_pixel_cards_carry_the_executed_values(self):
        workflow = build_workflow(request(), facts(0, (latent_unit(), pixel_unit())))
        first, second = subgraphs(workflow)
        self.assertEqual(first["name"], "4.1 / 2Sampler")
        self.assertEqual({node["type"] for node in first["nodes"]}, {"LatentUpscale", "KSampler"})
        upscale = inner(first, "LatentUpscale")[0]
        self.assertEqual(upscale["widgets_values"], ["bicubic", 1152, 1728, "disabled"])
        sampler = inner(first, "KSampler")[0]
        self.assertEqual(sampler["widgets_values"], [987654321, "fixed", 16, 5.0, "euler_ancestral", "normal", 0.35])
        self.assertEqual(second["name"], "4.2 / 3Sampler")
        self.assertEqual(
            {node["type"] for node in second["nodes"]},
            {"VAEDecodeTiled", "UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale", "VAEEncodeTiled", "KSampler"},
        )
        self.assertEqual(named(inner(second, "UpscaleModelLoader")[0])["model_name"], "RealESRGAN_x4plus_anime_6B.pth")
        self.assertEqual(inner(second, "ImageScale")[0]["widgets_values"], ["lanczos", 1296, 1944, "disabled"])
        self.assertEqual(named(inner(second, "KSampler")[0])["seed"], 1122334455)
        # The seed is the card's one visible widget; everything else is inside it.
        instances = [node for node in workflow["nodes"] if node["type"] in (first["id"], second["id"])]
        self.assertEqual([node["widgets_values"] for node in instances], [[987654321], [1122334455]])
        self.assertTrue(all(entry["name"] != "seed" for node in instances for entry in node["inputs"]))
        self.assertEqual([entry["name"] for entry in second["inputs"]], ["samples", "model", "positive", "negative", "vae", "seed"])

    def test_a_pure_resize_skips_the_model_and_a_no_redraw_unit_skips_the_sampler(self):
        resize = pixel_unit(1, upscaler=None, resize_algorithm="bicubic")
        enlarge = pixel_unit(2, redraws=False, denoise=0.0, steps=None, cfg=None, seed=None, source_size=(1296, 1944),
                             target_size=(2593, 3889), redraw_size=(2593, 3889), output_size=(2593, 3889))
        workflow = build_workflow(request(), facts(0, (resize, enlarge)))
        first, second = subgraphs(workflow)
        self.assertEqual({node["type"] for node in first["nodes"]}, {"VAEDecodeTiled", "ImageScale", "VAEEncodeTiled", "KSampler"})
        self.assertEqual(inner(first, "ImageScale")[0]["widgets_values"][0], "bicubic")
        self.assertEqual(second["name"], "4.2 / Upscale")
        self.assertNotIn("KSampler", {node["type"] for node in second["nodes"]})
        self.assertEqual([entry["name"] for entry in second["inputs"]], ["samples", "vae"])
        # The re-encoded latent is on the grid, so the exact size comes back before saving.
        exact = [node for node in nodes_of(workflow, "ImageScale")]
        self.assertEqual(exact[0]["widgets_values"], ["lanczos", 2593, 3889, "disabled"])

    def test_eight_cards_tile_two_by_four_and_more_keep_stacking(self):
        stages = [pixel_unit(number, upscaler=None) for number in range(1, 11)]
        workflow = build_workflow(request(), facts(0, stages))
        definitions = {item["id"] for item in subgraphs(workflow)}
        cards = [node for node in workflow["nodes"] if node["type"] in definitions]
        left = workflow["groups"][3]["bounding"][0]
        self.assertEqual([(node["pos"][0] - left, node["pos"][1]) for node in cards[:8]], [
            (10, 70), (220, 70), (10, 240), (220, 240), (10, 410), (220, 410), (10, 580), (220, 580),
        ])
        self.assertEqual([(node["pos"][0] - left, node["pos"][1]) for node in cards[8:]], [(10, 750), (220, 750)])
        self.assertEqual([node["title"] for node in cards][-1], "4.10 / 11Sampler")
        self.assertGreater(workflow["groups"][0]["bounding"][3], 750)

    def test_every_card_has_its_own_definition(self):
        workflow = build_workflow(request(), facts(0, (latent_unit(1), latent_unit(2, steps=20, denoise=0.5))))
        first, second = subgraphs(workflow)
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(named(inner(second, "KSampler")[0])["steps"], 20)
        self.assertEqual(named(inner(second, "KSampler")[0])["denoise"], 0.5)


class EngineMappingTests(unittest.TestCase):
    def test_clip_skip_follows_the_effective_value(self):
        self.assertEqual(named(nodes_of(build_workflow(request(), facts()), "CLIPSetLastLayer")[0]), {"stop_at_clip_layer": -2})
        sd = build_workflow(request(engine="SD"), facts(clip_skip=1))
        self.assertEqual(nodes_of(sd, "CLIPSetLastLayer"), [])
        self.assertEqual(titles(sd)[1], "02 / Prompts")
        self.assertEqual(named(nodes_of(build_workflow(request(engine="SD"), facts(clip_skip=3)), "CLIPSetLastLayer")[0]), {"stop_at_clip_layer": -3})

    def test_base_sampler_and_canvas_take_the_executed_values(self):
        workflow = build_workflow(request(), facts(seed=18446744073709551615))
        self.assertEqual(named(nodes_of(workflow, "EmptyLatentImage")[0]), {"width": 768, "height": 1152, "batch_size": 1})
        self.assertEqual(nodes_of(workflow, "KSampler")[0]["widgets_values"], [18446744073709551615, "fixed", 27, 5.0, "euler_ancestral", "normal", 1.0])
        self.assertEqual(named(nodes_of(workflow, "CheckpointLoaderSimple")[0]), {"ckpt_name": "waiIllustriousSDXL_v170.safetensors"})
        self.assertEqual(named(nodes_of(workflow, "CLIPTextEncode")[0])["text"], "masterpiece, 1girl")
        self.assertEqual(named(nodes_of(workflow, "CLIPTextEncode")[1])["text"], "low quality")

    def test_anima_and_krea2_load_three_components(self):
        anima = build_workflow(request(engine="Anima"), facts(3, clip_skip=None))
        self.assertEqual(named(nodes_of(anima, "UNETLoader")[0]), {"unet_name": "anima-model.safetensors", "weight_dtype": "default"})
        self.assertEqual(named(nodes_of(anima, "CLIPLoader")[0])["type"], "stable_diffusion")
        self.assertEqual(named(nodes_of(anima, "VAELoader")[0]), {"vae_name": "anima-vae.safetensors"})
        self.assertEqual(len(nodes_of(anima, "LoraLoader")), 3)
        self.assertEqual(titles(anima)[:3], ["01 / Model & VAE", "02 / LoRA · 3 (2 + 1)", "03 / Prompts"])
        self.assertEqual(nodes_of(anima, "CheckpointLoaderSimple"), [])
        krea = build_workflow(request(engine="Krea2", cfg=4.5), facts(2, clip_skip=None))
        self.assertEqual(named(nodes_of(krea, "CLIPLoader")[0])["type"], "krea2")
        self.assertEqual(len(nodes_of(krea, "LoraLoaderModelOnly")), 2)
        self.assertEqual(nodes_of(krea, "LoraLoader"), [])
        self.assertEqual(named(nodes_of(krea, "KSampler")[0])["cfg"], 4.5)

    def test_flux_reproduces_the_resolution_shift_and_distilled_guidance(self):
        workflow = build_workflow(request(engine="Flux", width=1024, height=1536), facts(1, clip_skip=None, negative=""))
        encoders = named(nodes_of(workflow, "DualCLIPLoader")[0])
        self.assertEqual((encoders["clip_name1"], encoders["clip_name2"], encoders["type"]), ("flux-te.safetensors", "t5xxl_fp8.safetensors", "flux"))
        self.assertEqual(named(nodes_of(workflow, "ModelSamplingFlux")[0]), {"max_shift": 1.15, "base_shift": 0.5, "width": 1024, "height": 1536})
        self.assertEqual(named(nodes_of(workflow, "FluxGuidance")[0]), {"guidance": 3.5})
        self.assertEqual(named(nodes_of(workflow, "KSampler")[0])["cfg"], 1.0)
        self.assertEqual(len(nodes_of(workflow, "ConditioningZeroOut")), 1)
        self.assertEqual(len(nodes_of(workflow, "LoraLoaderModelOnly")), 1)
        self.assertEqual(len(nodes_of(workflow, "CLIPTextEncode")), 1)

    def test_flux2_pins_its_fitted_shift_through_both_anchors(self):
        from backend.flux2_sampling import flux2_resolution_shift

        unit = pixel_unit(1, upscaler=None, cfg=2.5, steps=12, source_size=(1024, 1024), target_size=(1536, 1536),
                          redraw_size=(1536, 1536), output_size=(1536, 1536))
        workflow = build_workflow(request(engine="Flux2", width=1024, height=1024, steps=28, cfg=4), facts(0, (unit,), clip_skip=None, negative=""))
        self.assertEqual(len(nodes_of(workflow, "EmptyFlux2LatentImage")), 1)
        self.assertEqual(named(nodes_of(workflow, "CLIPLoader")[0])["type"], "flux2")
        base = named(nodes_of(workflow, "ModelSamplingFlux")[0])
        mu = round(flux2_resolution_shift(1024, 1024, 28), 6)
        self.assertEqual((base["max_shift"], base["base_shift"]), (mu, mu))
        card = subgraphs(workflow)[0]
        unit_shift = named(inner(card, "ModelSamplingFlux")[0])
        self.assertEqual(unit_shift["max_shift"], round(flux2_resolution_shift(1536, 1536, 12), 6))
        self.assertEqual((unit_shift["width"], unit_shift["height"]), (1536, 1536))
        self.assertEqual(named(inner(card, "FluxGuidance")[0]), {"guidance": 2.5})
        self.assertEqual(named(inner(card, "KSampler")[0])["cfg"], 1.0)

    def test_guidance_patches_are_native_or_named(self):
        pag = build_workflow(request(guidance="pag", pag={"scale": 3, "applied_layers": "all"}), facts())
        self.assertEqual(named(nodes_of(pag, "PerturbedAttentionGuidance")[0]), {"scale": 3.0})
        self.assertIn("middle block only", named_note(pag))
        cfg_zero = build_workflow(request(engine="Krea2", guidance="cfg_zero_star"), facts(clip_skip=None))
        self.assertEqual(len(nodes_of(cfg_zero, "CFGZeroStar")), 1)
        anima_pag = build_workflow(request(engine="Anima", guidance="pag"), facts(clip_skip=None))
        self.assertEqual(nodes_of(anima_pag, "PerturbedAttentionGuidance"), [])
        self.assertIn("PAG ran in XirAI", named_note(anima_pag))


def named_note(workflow):
    notes = nodes_of(workflow, "Note")
    return notes[0]["widgets_values"][0] if notes else ""


class ContentTests(unittest.TestCase):
    def test_omitted_stages_are_named_not_drawn(self):
        tiled = pixel_unit(2, tiled=True)
        workflow = build_workflow(request(engine="Anima"), facts(0, (latent_unit(1, latent_algorithm=None), "adetailer", tiled, "rtx"), clip_skip=None, background_removal=True))
        self.assertEqual(len(subgraphs(workflow)), 1)
        note = named_note(workflow)
        for fragment in ("ADetailer", "Resample 2 ran as a USDU tiled redraw", "RTX VSR", "removed the background"):
            self.assertIn(fragment, note)
        text = json.dumps(workflow)
        self.assertNotRegex(text, r"ADetailer\w*\"|UltimateSDUpscale|USDU\w*\"")
        self.assertEqual(nodes_of(build_workflow(request(), facts()), "Note"), [])

    def test_schedules_keep_their_first_step_and_are_noted(self):
        workflow = build_workflow(request(), facts(positive="a [cat:dog:0.5] BREAK sky", negative="[ugly|bad]"))
        encoders = nodes_of(workflow, "CLIPTextEncode")
        self.assertEqual(named(encoders[0])["text"], "a cat BREAK sky")
        self.assertEqual(named(encoders[1])["text"], "ugly")
        note = named_note(workflow)
        self.assertIn("BREAK", note)
        self.assertIn("first step", note)

    def test_no_absolute_path_hash_or_escape_reaches_the_graph(self):
        self.assertEqual(comfy_asset_name("shared:0123456789ab/Style/x.safetensors"), "Style\\x.safetensors")
        self.assertEqual(comfy_asset_name("C:\\Users\\me\\models\\x.safetensors"), "x.safetensors")
        self.assertEqual(comfy_asset_name("/home/me/models/x.safetensors"), "x.safetensors")
        self.assertEqual(comfy_asset_name("../escape/x.safetensors"), "x.safetensors")
        workflow = build_workflow(
            request(checkpoint="D:\\secret\\ckpt.safetensors"),
            facts(loras=(("shared:0123456789ab/Style/y.safetensors", 1.0), ("/abs/z.safetensors", 0.5))),
        )
        text = json.dumps(workflow, ensure_ascii=False)
        self.assertNotIn("secret", text)
        self.assertNotIn("/abs/", text)
        self.assertNotIn("shared:", text)
        self.assertNotRegex(text, r"[0-9a-f]{64}")
        self.assertNotIn("sha256", text)
        with self.assertRaises(ValueError):
            broken = json.loads(json.dumps(workflow))
            nodes_of(broken, "CheckpointLoaderSimple")[0]["widgets_values_named"]["ckpt_name"] = "D:\\x.safetensors"
            validate_workflow(broken)

    def test_image_to_image_and_post_processing_carry_no_workflow(self):
        # Both start from a picture the graph would have to fake; an unknown engine has no nodes.
        for engine, postprocess_only, has_source in (("iL", False, True), ("iL", True, True), ("Unknown", False, False)):
            fake = mock.Mock(engine=engine, postprocess_only=postprocess_only, source_image=mock.Mock(enabled=has_source))
            self.assertIsNone(build_workflow(fake, facts()))

    def test_the_same_run_yields_the_same_json(self):
        first = build_workflow(request(), facts(3, (latent_unit(), pixel_unit())))
        second = build_workflow(request(), facts(3, (latent_unit(), pixel_unit())))
        self.assertEqual(json.dumps(first), json.dumps(second))
        other = build_workflow(request(), facts(3, (latent_unit(), pixel_unit()), identity="job-1/1/2"))
        self.assertNotEqual(first["id"], other["id"])
        self.assertNotEqual(subgraphs(first)[0]["id"], subgraphs(other)[0]["id"])

    def test_structure_matches_the_ui_format(self):
        workflow = build_workflow(request(), facts(19, (latent_unit(), pixel_unit())))
        validate_workflow(workflow)
        self.assertEqual(workflow["version"], 0.4)
        self.assertEqual(set(workflow), {"id", "revision", "last_node_id", "last_link_id", "nodes", "links", "groups", "definitions", "config", "extra", "version"})
        for definition in subgraphs(workflow):
            self.assertEqual(definition["inputNode"]["id"], -10)
            self.assertEqual(definition["outputNode"]["id"], -20)
            self.assertEqual(definition["state"]["lastNodeId"], workflow["last_node_id"])
            self.assertEqual(len(definition["groups"]), 1)
            for entry in definition["inputs"]:
                self.assertTrue(entry["linkIds"], entry["name"])
        for node in workflow["nodes"]:
            self.assertTrue(node["title"])
            if node["type"] not in {item["id"] for item in subgraphs(workflow)}:
                self.assertEqual(node["properties"].get("Node name for S&R"), node["type"])
        orders = sorted(node["order"] for node in workflow["nodes"])
        self.assertEqual(orders, list(range(len(workflow["nodes"]))))
        self.assertLess(len(json.dumps(workflow, separators=(",", ":"))), 150_000)


class PngChunkTests(unittest.TestCase):
    def test_saved_png_carries_an_ascii_workflow_beside_the_unchanged_chunks(self):
        svc = server()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(svc, "OUTPUT_DIRECTORY", Path(directory)):
            submitted = request(prompt="cat <lora:matcha:0.4> 猫", loras=[])
            runtime = request(prompt="cat  猫", loras=[{"path": "Style/matcha.safetensors", "weight": 0.4}])
            output = svc.save_image(Image.new("RGB", (8, 8)), "job", submitted, 1.0, 1.0, 0.0, workflow_request=runtime)
            with Image.open(output) as saved:
                chunks = dict(saved.text)
            raw = Path(output).read_bytes()
        self.assertEqual(chunks["prompt"], "cat <lora:matcha:0.4> 猫")
        self.assertIn("parameters", chunks)
        workflow = json.loads(chunks["workflow"])
        self.assertIn(b"tEXtworkflow\x00", raw)
        self.assertTrue(chunks["workflow"].isascii())
        self.assertEqual(named(nodes_of(workflow, "CLIPTextEncode")[0])["text"], "cat  猫")
        self.assertEqual(named(nodes_of(workflow, "LoraLoader")[0])["lora_name"], "Style\\matcha.safetensors")

    def test_a_graph_that_cannot_be_built_costs_the_chunk_not_the_picture(self):
        svc = server()
        warnings = []
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(svc, "OUTPUT_DIRECTORY", Path(directory)):
            with mock.patch.object(svc, "build_workflow", side_effect=RuntimeError("x" * 500)), mock.patch("traceback.print_exc"):
                output = svc.save_image(Image.new("RGB", (8, 8)), "job", request(), 1.0, 1.0, 0.0, workflow_warnings=warnings)
            with Image.open(output) as saved:
                chunks = dict(saved.text)
        self.assertNotIn("workflow", chunks)
        self.assertIn("parameters", chunks)
        self.assertEqual(len(warnings), 1)
        self.assertTrue(warnings[0].startswith("ComfyUI 工作流未嵌入：RuntimeError"))
        self.assertLessEqual(len(warnings[0]), 220)

    def test_image_to_image_pictures_have_no_workflow_chunk(self):
        svc = server()
        buffer = __import__("io").BytesIO()
        Image.new("RGB", (64, 64)).save(buffer, format="PNG")
        data = "data:image/png;base64," + __import__("base64").b64encode(buffer.getvalue()).decode()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(svc, "OUTPUT_DIRECTORY", Path(directory)):
            i2i = request(source_image={"enabled": True, "image_data": data, "name": "s.png"}, denoise=0.5)
            output = svc.save_image(Image.new("RGB", (8, 8)), "job", i2i, 1.0, 1.0, 0.0)
            with Image.open(output) as saved:
                self.assertNotIn("workflow", saved.text)

    def test_facts_follow_the_recorded_units(self):
        svc = server()
        chain = [
            {"type": "resample", "enabled": True, "latent_algorithm": "latent", "scale": 1.5, "denoise": 0.4, "seed_mode": "fixed", "seed": "77"},
            {"type": "adetailer", "enabled": True, "units": [{"detector": "face.pt"}]},
            {"type": "resample", "enabled": False, "scale": 2, "denoise": 0.3},
            {"type": "resample", "enabled": True, "upscaler": "x4.pth", "scale": 1.3, "denoise": 0.3},
        ]
        runtime = request(postprocess_chain=chain, width=768, height=1152)
        records = [
            {"unit": 1, "source_size": [768, 1152], "steps": 27, "cfg": 5, "sampler_resolved": "euler_ancestral",
             "scheduler_resolved": "normal", "seed": "77", "executed_size": [1152, 1728]},
            {"unit": 3, "source_size": [1152, 1728], "steps": 27, "cfg": 5, "sampler_resolved": "euler_ancestral",
             "scheduler_resolved": "normal", "seed": "123", "executed_size": [1496, 2248]},
        ]
        result = svc.comfy_workflow_facts(runtime, seed=5, resample_results=records, identity="j/1/1")
        first, middle, last = result.stages
        self.assertEqual((first.number, first.seed, first.redraw_size), (1, 77, (1152, 1728)))
        self.assertEqual(middle, "adetailer")
        self.assertEqual((last.number, last.upscaler, last.target_size), (3, "x4.pth", (1498, 2246)))
        self.assertEqual(last.redraw_size, (1496, 2248))
        workflow = build_workflow(runtime, result)
        self.assertEqual([item["name"] for item in subgraphs(workflow)], ["4.1 / 2Sampler", "4.2 / 3Sampler"])
        self.assertIn("ADetailer", named_note(workflow))


if __name__ == "__main__":
    unittest.main()
