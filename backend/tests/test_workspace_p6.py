"""P6: CLIP skip, the configurable sampling canvas and output templates."""

import json
import multiprocessing
import os
import tempfile
import threading
import types
import unicodedata
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

from backend import forge_prompt, output_paths
from backend.dynamic_prompt import PromptTimeline, prepare_dynamic_prompt_conditioning
from backend.forge_prompt import CHUNK_LENGTH, encode_forge_texts, prepare_prompt_conditioning, tower_layers
from backend.output_paths import (
    DEFAULT_OUTPUT_TEMPLATE,
    OutputPathError,
    normalize_output_template,
    output_target,
    preview_output_name,
    publish_exclusive,
    temporary_output_path,
)

MOMENT = datetime(2026, 9, 24, 17, 5, 9)


def _publish_many(root, template, count, marker, queue):
    """Worker for the multi-process race: publish `count` pictures and report their names."""
    names = []
    for index in range(count):
        target = output_target(Path(root), template, moment=MOMENT, engine="iL", create=True)
        temporary = temporary_output_path(target)
        temporary.write_bytes(f"{marker}:{index}".encode())
        names.append(publish_exclusive(temporary, target).name)
    queue.put(names)


class OutputTemplateParsingTests(unittest.TestCase):
    def test_default_and_absent_templates_mean_the_historical_scheme(self):
        self.assertIsNone(normalize_output_template(None))
        self.assertIsNone(normalize_output_template(""))
        self.assertIsNone(normalize_output_template("   "))
        self.assertIsNone(normalize_output_template(DEFAULT_OUTPUT_TEMPLATE))

    def test_custom_templates_are_normalised(self):
        self.assertEqual(normalize_output_template("Test/image"), "Test/image")
        self.assertEqual(normalize_output_template("Test\\image.png"), "Test/image")
        self.assertEqual(normalize_output_template("{engine}/{year}-{month}/shot-{hour}{minute}{second}"), "{engine}/{year}-{month}/shot-{hour}{minute}{second}")
        decomposed = unicodedata.normalize("NFD", "Café/图")
        self.assertEqual(normalize_output_template(decomposed), unicodedata.normalize("NFC", "Café/图"))

    def test_unsafe_templates_are_refused(self):
        cases = {
            "/abs/image": "绝对路径",
            "\\\\server\\share\\x": "绝对路径",
            "C:/Users/image": "盘符",
            "c:image": "盘符",
            "../image": r"\.\.",
            "a/../image": r"\.\.",
            "./image": r"\. 或",
            "a//image": "空的目录名",
            "image/": "空的目录名",
            "CON/image": "保留名称",
            "a/nul.txt": "保留名称",
            "a/COM1": "保留名称",
            "trail./image": "以点结尾",
            "space /image": "首尾不能有空格",
            ".hidden/image": "以点开头",
            "a/b:c": "不允许的字符",
            "a/b*c": "不允许的字符",
            "a/b\u0001c": "不允许的字符",
            "{month}/{minutes}": "未知占位符",
            "{date/image": "花括号",
            "a}/image": "花括号",
            "x" * 201: "200",
            "/".join(["a"] * 9): "最多",
            "a/" + "b" * 97: "过长",
        }
        for template, message in cases.items():
            with self.subTest(template=template):
                with self.assertRaisesRegex(OutputPathError, message):
                    normalize_output_template(template)

    def test_month_and_minute_are_distinct_placeholders(self):
        segments = output_paths.expand_output_template("{year}-{month}-{day}/{hour}-{minute}-{second}", MOMENT, "SD")
        self.assertEqual(segments, ["2026-09-24", "17-05-09"])


class OutputTargetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "outputs"
        self.root.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def publish(self, template, content=b"png"):
        target = output_target(self.root, template, moment=MOMENT, engine="iL", create=True)
        temporary = temporary_output_path(target)
        temporary.write_bytes(content)
        return publish_exclusive(temporary, target)

    def test_legacy_scheme_is_byte_for_byte_the_historical_one(self):
        first = self.publish(None)
        second = self.publish(None)
        third = self.publish(None)
        self.assertEqual(first.relative_to(self.root).as_posix(), "2026-09-24/XirAI-2026-09-24-17-05.png")
        self.assertEqual(second.name, "XirAI-2026-09-24-17-05-02.png")
        self.assertEqual(third.name, "XirAI-2026-09-24-17-05-03.png")

    def test_custom_template_counts_from_0001(self):
        first = self.publish("Test/image")
        second = self.publish("Test/image")
        self.assertEqual(first.relative_to(self.root).as_posix(), "Test/image_0001.png")
        self.assertEqual(second.relative_to(self.root).as_posix(), "Test/image_0002.png")
        target = output_target(self.root, "Test/image", moment=MOMENT, engine="iL", create=False)
        self.assertEqual(preview_output_name(target), "image_0003.png")

    def test_counter_continues_past_the_highest_existing_file_and_never_overwrites(self):
        directory = self.root / "Test"
        directory.mkdir()
        (directory / "image_0007.png").write_bytes(b"old")
        (directory / "image_0002.png").write_bytes(b"older")
        published = self.publish("Test/image", b"new")
        self.assertEqual(published.name, "image_0008.png")
        self.assertEqual((directory / "image_0007.png").read_bytes(), b"old")
        self.assertEqual((directory / "image_0002.png").read_bytes(), b"older")
        self.assertEqual(published.read_bytes(), b"new")

    def test_a_taken_name_is_skipped_rather_than_replaced(self):
        target = output_target(self.root, "Test/image", moment=MOMENT, engine="iL", create=True)
        temporary = temporary_output_path(target)
        temporary.write_bytes(b"late")
        # Another writer takes image_0001 after the counter was read but before publishing.
        with mock.patch.object(output_paths, "next_counter", return_value=1):
            (self.root / "Test" / "image_0001.png").write_bytes(b"winner")
            published = publish_exclusive(temporary, target)
        self.assertEqual(published.name, "image_0002.png")
        self.assertEqual((self.root / "Test" / "image_0001.png").read_bytes(), b"winner")
        self.assertFalse(temporary.exists())

    def test_volumes_without_hard_links_fall_back_to_exclusive_reservation(self):
        (self.root / "Test").mkdir()
        (self.root / "Test" / "image_0001.png").write_bytes(b"old")
        with mock.patch.object(output_paths.os, "link", side_effect=PermissionError(1, "Operation not permitted")), \
                mock.patch.object(output_paths, "next_counter", return_value=1):
            published = self.publish("Test/image", b"new")
        self.assertEqual(published.name, "image_0002.png")
        self.assertEqual((self.root / "Test" / "image_0001.png").read_bytes(), b"old")
        self.assertEqual(published.read_bytes(), b"new")
        self.assertEqual(sorted(path.name for path in (self.root / "Test").iterdir()), ["image_0001.png", "image_0002.png"])

    def test_threads_racing_for_one_template_get_distinct_names(self):
        names = []
        lock = threading.Lock()

        def worker(marker):
            for index in range(10):
                published = self.publish("Race/shot", f"{marker}:{index}".encode())
                with lock:
                    names.append(published.name)

        threads = [threading.Thread(target=worker, args=(marker,)) for marker in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(names), 60)
        self.assertEqual(len(set(names)), 60)
        contents = {path.read_bytes() for path in (self.root / "Race").iterdir()}
        self.assertEqual(len(contents), 60)

    def test_processes_racing_for_one_template_never_overwrite(self):
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        workers = [context.Process(target=_publish_many, args=(str(self.root), "Multi/shot", 8, marker, queue)) for marker in range(3)]
        for worker in workers:
            worker.start()
        names = [name for _ in workers for name in queue.get(timeout=120)]
        for worker in workers:
            worker.join(timeout=120)
        self.assertEqual(len(names), 24)
        self.assertEqual(len(set(names)), 24)
        files = sorted((self.root / "Multi").glob("*.png"))
        self.assertEqual(len(files), 24)
        self.assertEqual(len({path.read_bytes() for path in files}), 24)
        self.assertEqual([path for path in (self.root / "Multi").iterdir() if path.suffix == ".tmp"], [])

    def test_a_file_in_the_way_is_refused(self):
        (self.root / "Test").write_bytes(b"not a folder")
        with self.assertRaisesRegex(OutputPathError, "不是文件夹"):
            output_target(self.root, "Test/image", moment=MOMENT, engine="iL", create=False)

    def test_links_and_junctions_on_the_way_are_refused(self):
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        link = self.root / "Linked"
        created = False
        if os.name == "nt":
            import _winapi
            try:
                _winapi.CreateJunction(str(outside), str(link))
                created = True
            except OSError:
                created = False
        if not created:
            try:
                os.symlink(outside, link, target_is_directory=True)
                created = True
            except (OSError, NotImplementedError):
                self.skipTest("this machine can create neither junctions nor symlinks")
        for create in (False, True):
            with self.assertRaisesRegex(OutputPathError, "符号链接或目录联接"):
                output_target(self.root, "Linked/image", moment=MOMENT, engine="iL", create=create)
        self.assertEqual(list(outside.iterdir()), [])


class GenerateInputP6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend import inference_server

        cls.server = inference_server

    def request(self, **overrides):
        fields = {
            "engine": "SD", "checkpoint": "model.safetensors", "prompt": "cat", "width": 512, "height": 512,
            "steps": 4, "cfg": 7, "denoise": 1, "seed": 1, "sampler": "euler", "scheduler": "normal",
        }
        return self.server.GenerateInput(**{**fields, **overrides})

    def native(self, engine="Anima", **overrides):
        return self.request(
            engine=engine, checkpoint=None, diffusion_model="d.safetensors", text_encoder="t.safetensors",
            vae="v.safetensors", sampler="euler", scheduler="simple", **overrides,
        )

    def test_protocol_is_39(self):
        self.assertEqual(self.server.INFERENCE_PROTOCOL, 39)

    def test_clip_skip_defaults_and_ranges(self):
        self.assertEqual(self.server.effective_clip_skip(self.request()), 1)
        self.assertEqual(self.server.effective_clip_skip(self.request(engine="iL")), 2)
        self.assertIsNone(self.server.effective_clip_skip(self.native()))
        self.assertEqual(self.server.effective_clip_skip(self.request(clip_skip=12)), 12)
        self.assertEqual(self.server.effective_clip_skip(self.request(engine="iL", clip_skip=3)), 3)
        for engine, value, message in (("SD", 0, "between 1 and 12"), ("SD", 13, "between 1 and 12"), ("iL", 1, "minimum effective skip of 2"), ("iL", 13, "between 2 and 12")):
            with self.subTest(engine=engine, value=value):
                with self.assertRaisesRegex(ValueError, message):
                    self.request(engine=engine, clip_skip=value)

    def test_clip_skip_is_strict_and_refused_by_native_engines(self):
        for value in ("2", 2.0, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.request(clip_skip=value)
        for engine in ("Anima", "Krea2"):
            with self.assertRaisesRegex(ValueError, "does not support CLIP skip"):
                self.native(engine, clip_skip=2)

    def test_sampling_canvas_ceiling_is_8192_and_the_engine_edge(self):
        self.assertEqual(self.request(width=4096, height=2048).width, 4096)
        self.assertEqual(self.request(width=8192, height=1024).width, 8192)
        with self.assertRaisesRegex(ValueError, "divisible by 64"):
            self.request(width=2100, height=512)
        with self.assertRaisesRegex(ValueError, "cannot exceed 4096"):
            self.native(width=4160, height=512)
        self.assertEqual(self.native(width=4096, height=512).width, 4096)
        with self.assertRaises(ValueError):
            self.request(width=8256, height=512)

    def test_output_template_is_checked_on_the_request(self):
        self.assertIsNone(self.request(output_template=DEFAULT_OUTPUT_TEMPLATE).output_template)
        self.assertEqual(self.request(output_template="Test/image").output_template, "Test/image")
        with self.assertRaisesRegex(ValueError, "绝对路径"):
            self.request(output_template="C:/Windows/image")


class ExpandedCanvasAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend import inference_server

        cls.server = inference_server

    def setUp(self):
        self.created = []

    def tearDown(self):
        with self.server.jobs_lock:
            for job_id in self.created:
                self.server.jobs.pop(job_id, None)
                self.server.job_controls.pop(job_id, None)

    def request(self, **overrides):
        fields = {
            "engine": "iL", "checkpoint": "model.safetensors", "prompt": "cat", "width": 1024, "height": 1024,
            "steps": 4, "cfg": 7, "denoise": 1, "seed": 1, "sampler": "euler", "scheduler": "normal",
        }
        return self.server.GenerateInput(**{**fields, **overrides})

    def device(self, gib=8):
        settings = {**self.server.performance_settings, "vram_limit_gb": 0.0, "allow_shared_memory": False}
        return (
            mock.patch.object(self.server.torch.cuda, "is_available", return_value=True),
            mock.patch.object(self.server, "installed_vram_bytes", return_value=gib * 1024**3),
            mock.patch.object(self.server, "performance_settings", settings),
        )

    def test_legacy_canvases_are_not_asked_again(self):
        first, second, third = self.device(2)
        with first, second, third:
            self.assertIsNone(self.server.expanded_canvas_resource_issue(self.request(width=2048, height=2048, images_per_batch=10)))

    def test_an_8192_square_is_refused_before_queueing_on_an_8_gb_card(self):
        first, second, third = self.device(8)
        before = set(self.server.jobs)
        with first, second, third, mock.patch.object(self.server.executor, "submit") as submit:
            with self.assertRaises(self.server.HTTPException) as caught:
                self.server.create_job(self.request(width=8192, height=8192))
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("任务未入队", caught.exception.detail)
        self.assertIn("8192×8192", caught.exception.detail)
        submit.assert_not_called()
        self.assertEqual(set(self.server.jobs), before)

    def test_a_modest_expanded_canvas_is_admitted_and_batch_size_counts(self):
        first, second, third = self.device(8)
        with first, second, third:
            self.assertIsNone(self.server.expanded_canvas_resource_issue(self.request(width=2560, height=2560)))
            issue = self.server.expanded_canvas_resource_issue(self.request(width=2560, height=2560, images_per_batch=6, guidance="pag"))
        self.assertIn("单批 6 张", issue)

    def test_a_larger_card_admits_what_a_smaller_one_refuses(self):
        request = self.request(width=4096, height=4096)
        first, second, third = self.device(8)
        with first, second, third:
            self.assertIsNotNone(self.server.expanded_canvas_resource_issue(request))
        first, second, third = self.device(48)
        with first, second, third:
            self.assertIsNone(self.server.expanded_canvas_resource_issue(request))


class JobAndPngRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend import inference_server

        cls.server = inference_server

    def setUp(self):
        self.created = []

    def tearDown(self):
        with self.server.jobs_lock:
            for job_id in self.created:
                self.server.jobs.pop(job_id, None)
                self.server.job_controls.pop(job_id, None)

    def request(self, **overrides):
        fields = {
            "engine": "iL", "checkpoint": "model.safetensors", "prompt": "cat", "width": 512, "height": 512,
            "steps": 4, "cfg": 7, "denoise": 1, "seed": 1, "sampler": "euler", "scheduler": "normal",
            "loras": [{"path": "style/Style_九V6.0/matcha.safetensors", "weight": 0.4}],
        }
        return self.server.GenerateInput(**{**fields, **overrides})

    def test_job_records_effective_clip_skip_and_frozen_template(self):
        with mock.patch.object(self.server.executor, "submit"):
            response = self.server.create_job(self.request(output_template="Test/image"))
        self.created.append(response["id"])
        self.assertEqual(response["clip_skip"], 2)
        self.assertEqual(response["output_template"], "Test/image")
        with mock.patch.object(self.server.executor, "submit"):
            legacy = self.server.create_job(self.request(engine="SD"))
        self.created.append(legacy["id"])
        self.assertEqual(legacy["clip_skip"], 1)
        self.assertIsNone(legacy["output_template"])

    def test_a_template_through_a_file_is_refused_before_queueing(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "Test").write_bytes(b"x")
            before = set(self.server.jobs)
            with mock.patch.object(self.server, "OUTPUT_DIRECTORY", Path(directory)), mock.patch.object(self.server.executor, "submit") as submit:
                with self.assertRaises(self.server.HTTPException) as caught:
                    self.server.create_job(self.request(output_template="Test/image"))
            self.assertEqual(caught.exception.status_code, 422)
            submit.assert_not_called()
            self.assertEqual(set(self.server.jobs), before)

    def test_png_records_clip_skip_relative_lora_ids_and_uses_the_template(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(self.server, "OUTPUT_DIRECTORY", Path(directory)):
                request = self.request(clip_skip=3, output_template="Test/image")
                first = self.server.save_image(Image.new("RGB", (8, 8)), "job", request, 1.0, 1.0, 0.0)
                second = self.server.save_image(Image.new("RGB", (8, 8)), "job", request, 1.0, 1.0, 0.0)
                self.assertEqual(self.server.output_relative_path(first), "Test/image_0001.png")
                self.assertEqual(self.server.output_relative_path(second), "Test/image_0002.png")
                with Image.open(first) as saved:
                    parameters = json.loads(saved.text["parameters"])
            self.assertEqual(parameters["clip_skip"], 3)
            self.assertEqual(parameters["output_template"], "Test/image")
            self.assertEqual(parameters["loras"], [{"name": "matcha.safetensors", "path": "style/Style_九V6.0/matcha.safetensors", "weight": 0.4}])
            self.assertNotIn(directory, json.dumps(parameters))

    def test_native_png_has_no_clip_skip_and_legacy_naming_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(self.server, "OUTPUT_DIRECTORY", Path(directory)):
                request = self.server.GenerateInput(
                    engine="Anima", diffusion_model="d.safetensors", text_encoder="t.safetensors", vae="v.safetensors",
                    prompt="cat", width=512, height=512, steps=4, cfg=4, denoise=1, seed=1, sampler="euler", scheduler="simple",
                )
                output = self.server.save_image(Image.new("RGB", (8, 8)), "job", request, 1.0, 1.0, 0.0)
                self.assertRegex(self.server.output_relative_path(output), r"^\d{4}-\d{2}-\d{2}/XirAI-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}\.png$")
                with Image.open(output) as saved:
                    parameters = json.loads(saved.text["parameters"])
            self.assertNotIn("clip_skip", parameters)
            self.assertNotIn("output_template", parameters)

    def test_cancelled_commit_leaves_no_file_or_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(self.server, "OUTPUT_DIRECTORY", Path(directory)):
                def cancel():
                    raise self.server.GenerationCancelled()

                with self.assertRaises(self.server.GenerationCancelled):
                    self.server.save_image(Image.new("RGB", (8, 8)), "job", self.request(output_template="Test/image"), 1.0, 1.0, 0.0, before_commit=cancel)
            self.assertEqual([path for path in Path(directory).rglob("*") if path.is_file()], [])

    def test_output_template_endpoint_previews_and_explains(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(self.server, "OUTPUT_DIRECTORY", Path(directory)):
                preview = self.server.get_output_template("Test/image", "iL")
                legacy = self.server.get_output_template(None, "iL")
                invalid = self.server.get_output_template("../escape", "iL")
        self.assertTrue(preview["valid"])
        self.assertFalse(preview["legacy"])
        self.assertEqual(preview["preview"], "Test/image_0001.png")
        self.assertTrue(legacy["legacy"])
        self.assertRegex(legacy["preview"], r"^\d{4}-\d{2}-\d{2}/XirAI-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}\.png$")
        self.assertFalse(invalid["valid"])
        self.assertIn("..", invalid["error"])
        self.assertEqual({item["name"] for item in preview["placeholders"]}, set(output_paths.OUTPUT_TEMPLATE_PLACEHOLDERS))


class NormedEncoder:
    """A CLIP-like tower whose hidden states differ per layer and whose final norm is visible."""

    config = types.SimpleNamespace(use_attention_mask=False)

    def __init__(self, width, layers=12, projected=False):
        self.width = width
        self.layers = layers
        self.projected = projected
        self.calls = []
        self.text_model = types.SimpleNamespace(final_layer_norm=lambda value: value * 10 + 0.5)

    def __call__(self, ids, output_hidden_states=False, **_kwargs):
        self.calls.append(output_hidden_states)
        base = ids.float().unsqueeze(-1) + torch.arange(1, self.width + 1, device=ids.device)
        states = [base * (layer + 1) for layer in range(self.layers + 1)]
        final = self.text_model.final_layer_norm(states[-1])
        first = final.mean(dim=1) * 3 if self.projected else final
        return _Output(first, states if output_hidden_states else None)


class _Output:
    def __init__(self, first, hidden_states):
        self.first = first
        self.hidden_states = hidden_states

    def __getitem__(self, index):
        if index == 0:
            return self.first
        raise IndexError(index)


class Tokenizer:
    bos_token_id = 1000
    eos_token_id = 1001
    pad_token_id = 0

    def __call__(self, text, **_kwargs):
        return {"input_ids": [10 + ord(character) % 89 for character in text if character != " "]}


def pipeline(sdxl=False):
    value = types.SimpleNamespace(tokenizer=Tokenizer(), text_encoder=NormedEncoder(3), _execution_device=torch.device("cpu"))
    if sdxl:
        value.tokenizer_2 = Tokenizer()
        value.text_encoder_2 = NormedEncoder(5, layers=32, projected=True)
    return value


class ClipSkipEncodingTests(unittest.TestCase):
    def test_tower_layers(self):
        self.assertEqual(tower_layers("sd"), forge_prompt.TowerLayer(1, False, False))
        self.assertEqual(tower_layers("sd", 2), forge_prompt.TowerLayer(2, True, False))
        self.assertEqual(tower_layers("sdxl"), forge_prompt.TowerLayer(2, False, True))
        self.assertEqual(tower_layers("sdxl", 3), forge_prompt.TowerLayer(3, False, True))
        with self.assertRaisesRegex(ValueError, "minimum effective CLIP skip of 2"):
            tower_layers("sdxl", 1)

    def test_sd_default_keeps_the_final_output_call(self):
        fake = pipeline()
        default = encode_forge_texts(fake, "sd", ["cat"], torch.device("cpu"))[0].cross_attention
        explicit = encode_forge_texts(fake, "sd", ["cat"], torch.device("cpu"), clip_skip=1)[0].cross_attention
        self.assertEqual(fake.text_encoder.calls, [False, False])
        self.assertTrue(torch.equal(default, explicit))

    def test_sd_skip_reads_a_deeper_layer_through_the_final_norm(self):
        fake = pipeline()
        skipped = encode_forge_texts(fake, "sd", ["cat"], torch.device("cpu"), clip_skip=2)[0].cross_attention
        ids = torch.tensor([forge_prompt.build_forge_token_batches(fake.tokenizer, "cat")[0][0]])
        base = ids.float().unsqueeze(-1) + torch.arange(1, 4)
        expected = (base * 12) * 10 + 0.5  # hidden_states[-2] of 13 states is layer index 11 -> x12
        self.assertTrue(torch.equal(skipped, expected))
        default = encode_forge_texts(fake, "sd", ["cat"], torch.device("cpu"))[0].cross_attention
        self.assertFalse(torch.equal(skipped, default))

    def test_sdxl_skip_changes_both_towers_but_not_the_pooled_vector(self):
        fake = pipeline(sdxl=True)
        default = encode_forge_texts(fake, "sdxl", ["cat"], torch.device("cpu"))[0]
        explicit = encode_forge_texts(fake, "sdxl", ["cat"], torch.device("cpu"), clip_skip=2)[0]
        deeper = encode_forge_texts(fake, "sdxl", ["cat"], torch.device("cpu"), clip_skip=3)[0]
        self.assertTrue(torch.equal(default.cross_attention, explicit.cross_attention))
        self.assertFalse(torch.equal(default.cross_attention[..., :3], deeper.cross_attention[..., :3]))
        self.assertFalse(torch.equal(default.cross_attention[..., 3:], deeper.cross_attention[..., 3:]))
        self.assertTrue(torch.equal(default.pooled, deeper.pooled))
        # No final norm on SDXL: layer -3 of CLIP-L is x11 of the base, unnormalised.
        ids = torch.tensor([forge_prompt.build_forge_token_batches(fake.tokenizer, "cat")[0][0]])
        base = ids.float().unsqueeze(-1) + torch.arange(1, 4)
        self.assertTrue(torch.equal(deeper.cross_attention[..., :3], base * 11))

    def test_skip_deeper_than_the_tower_is_refused(self):
        fake = pipeline()
        fake.text_encoder.layers = 4
        with self.assertRaisesRegex(ValueError, "deeper than"):
            encode_forge_texts(fake, "sd", ["cat"], torch.device("cpu"), clip_skip=5)

    def test_static_and_dynamic_plans_use_one_skip_for_every_branch(self):
        fake = pipeline()
        static = prepare_prompt_conditioning(fake, "sd", "cat", "dog", clip_skip=2)
        direct_negative = encode_forge_texts(fake, "sd", ["dog"], torch.device("cpu"), negative=True, clip_skip=2)[0]
        self.assertTrue(torch.equal(static.plan.negative.cross_attention, direct_negative.cross_attention))
        calls = []
        original = encode_forge_texts

        def spy(*args, **kwargs):
            calls.append(kwargs.get("clip_skip"))
            return original(*args, **kwargs)

        with mock.patch("backend.dynamic_prompt.encode_forge_texts", side_effect=spy):
            prepare_dynamic_prompt_conditioning(fake, "sd", "[cat|dog] AND bird", "[x:y:2]", timeline=PromptTimeline(4), clip_skip=3)
        self.assertTrue(calls)
        self.assertEqual(set(calls), {3})

    def test_comfy_oracle_refuses_a_non_default_skip(self):
        with self.assertRaisesRegex(ValueError, "does not implement CLIP skip"):
            prepare_prompt_conditioning(pipeline(), "sd", "cat", "", emphasis="comfy", clip_skip=2)


if __name__ == "__main__":
    unittest.main()
