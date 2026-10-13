"""QPlan2/P1 Phase D grid export: layout, budgets, downscale quality oracles, metadata, publication.

Pure: no torch. Sources are small synthetic PNGs written to a temporary output tree.
"""
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageCms, ImageDraw

from backend import xyz_grid
from backend.output_paths import OutputTarget, publish_exclusive
from backend.xyz_grid import (ExportCancelled, GridError, compute_layout, downscale_linear, estimate, export_grid,
                              image_target_size, open_source)

OPTIONS = {"scope": "all", "page": None, "mode": "original", "percent": 100, "theme": "light", "arrangement": "vertical", "captions": True}


def lanczos3(x):
    x = abs(x)
    if x >= 3:
        return 0.0
    if x == 0:
        return 1.0
    return (math.sin(math.pi * x) / (math.pi * x)) * (math.sin(math.pi * x / 3) / (math.pi * x / 3))


def reference_resize_1d(values, out_size):
    """Independent Lanczos3 with Pillow's documented coordinate convention (centre = (i + 0.5)·scale)."""
    in_size = values.shape[0]
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 3 * filterscale
    out = np.zeros((out_size,) + values.shape[1:], dtype=np.float64)
    for index in range(out_size):
        center = (index + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size)
        weights = np.array([lanczos3((x - center + 0.5) / filterscale) for x in range(xmin, xmax)])
        weights /= weights.sum()
        out[index] = np.tensordot(weights, values[xmin:xmax], axes=(0, 0))
    return out


def reference_resize(channel, size):
    width, height = size
    horizontal = reference_resize_1d(channel.astype(np.float64).T, width).T
    return reference_resize_1d(horizontal, height)


class GridFixture(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="xyz-d-grid-"))
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()

    def source(self, name, image):
        path = self.outputs / name
        image.save(path)
        return path

    def snapshot(self, sizes, statuses=None, x_labels=None, y_labels=None, z_labels=(), repeat=1, alpha=()):
        x_labels = x_labels or [f"X{index}" for index in range(len(sizes[0]))]
        y_labels = y_labels or [f"Y{index}" for index in range(len(sizes))]
        cells = []
        for z in range(max(1, len(z_labels))):
            for sample in range(repeat):
                for y, row in enumerate(sizes):
                    for x, (width, height) in enumerate(row):
                        cell_id = f"{x}:{y}:{z}:{sample}"
                        status = (statuses or {}).get(cell_id, "complete")
                        cell = {"id": cell_id, "coordinate": [x, y, z, sample], "labels": {"x": x_labels[x], "y": y_labels[y]}, "status": status,
                                "seed": str(1000 + x + 10 * y), "request": {"steps": 20}, "lora_references": [], "batch": None,
                                "relative": None, "asset_id": None, "width": None, "height": None, "alpha": cell_id in alpha}
                        if status == "complete":
                            mode = "RGBA" if cell_id in alpha else "RGB"
                            rng = np.random.default_rng(x * 31 + y * 7 + z * 3 + sample)
                            pixels = rng.integers(0, 256, size=(height, width, 4 if mode == "RGBA" else 3), dtype=np.uint8)
                            path = self.source(f"{cell_id.replace(':', '-')}.png", Image.fromarray(pixels, mode))
                            cell.update(path=path, relative=path.name, asset_id=path.name, width=width, height=height)
                        cells.append(cell)
        return {"experiment_id": "e" * 32, "digest": "d" * 64, "engine": "SD", "title": "XYZ 对比 · SD · 测试模型",
                "subtitle": "X：steps · Y：cfg · 测试", "baseline": {"width": 64}, "dimensions": {}, "repeat_count": repeat,
                "axes": {"x": {"field": "steps", "labels": x_labels}, "y": {"field": "cfg", "labels": y_labels},
                         "z": {"field": "prompt" if z_labels else None, "labels": list(z_labels)}}, "cells": cells}

    def export(self, snapshot, options, stem="XiriusA-XYZPlot-test", cancelled=lambda: False):
        target = OutputTarget(self.outputs, "", stem, True)
        return export_grid(snapshot, options, directory=self.outputs, stem=stem, export_id="x" * 32,
                           publish=lambda temporary: publish_exclusive(temporary, target), cancelled=cancelled)


class LayoutAndOriginalTests(GridFixture):
    def test_mixed_sizes_get_column_and_row_slots_without_enlargement(self):
        snapshot = self.snapshot([[(64, 96), (40, 40)], [(48, 64), (56, 32)]])
        layout = compute_layout(snapshot, OPTIONS)
        self.assertEqual(layout["columns"], [64, 56])
        self.assertEqual(layout["rows"], [96, 64])
        small = next(entry for entry in layout["ops"]["cells"] if entry["cell"] == "1:0:0:0")
        self.assertEqual(small["image"][2:], (40, 40))
        self.assertEqual(small["image"][:2], (small["slot"][0] + 8, small["slot"][1] + 28))

    def test_original_export_places_every_source_1to1_and_never_touches_source_bytes(self):
        snapshot = self.snapshot([[(64, 96), (40, 40)], [(48, 64), (56, 32)]], alpha={"1:1:0:0"})
        before = {cell["id"]: hashlib.sha256(cell["path"].read_bytes()).hexdigest() for cell in snapshot["cells"]}
        result = self.export(snapshot, OPTIONS)
        manifest = json.loads(result["manifest"].read_text(encoding="utf-8"))
        with Image.open(result["png"]) as exported:
            self.assertEqual(exported.mode, "RGBA")
            for cell in snapshot["cells"]:
                x, y, w, h = next(item["placed"]["rect"] for item in manifest["cells"] if item["id"] == cell["id"])
                with Image.open(cell["path"]) as source:
                    expected = np.asarray(source.convert("RGBA"))
                self.assertTrue(np.array_equal(np.asarray(exported.crop((x, y, x + w, y + h))), expected), cell["id"])
        self.assertEqual(before, {cell["id"]: hashlib.sha256(cell["path"].read_bytes()).hexdigest() for cell in snapshot["cells"]})
        self.assertIsNone(manifest["recipe"]["algorithm"])

    def test_scope_page_arrangements_and_partial_placeholders(self):
        snapshot = self.snapshot([[(32, 32), (32, 32)]], statuses={"1:0:0:0": "error", "0:0:1:0": "pending"}, z_labels=("a", "b", "c"))
        everything = compute_layout(snapshot, OPTIONS)
        self.assertEqual(everything["blocks"], [(0, 0), (1, 0), (2, 0)])
        page = compute_layout(snapshot, {**OPTIONS, "scope": "page", "page": {"z_index": 1, "sample_index": 0}})
        self.assertEqual(page["blocks"], [(1, 0)])
        wide = compute_layout(snapshot, {**OPTIONS, "arrangement": "horizontal"})
        self.assertGreater(wide["width"], everything["width"])
        self.assertLess(wide["height"], everything["height"])
        statuses = {entry["cell"]: (entry["status"], entry["image"]) for entry in everything["ops"]["cells"]}
        self.assertEqual(statuses["1:0:0:0"], ("error", None))
        self.assertEqual(statuses["0:0:1:0"], ("pending", None))
        with self.assertRaises(GridError):
            compute_layout(snapshot, {**OPTIONS, "scope": "page", "page": {"z_index": 9, "sample_index": 0}})
        empty = self.snapshot([[(32, 32)]], statuses={"0:0:0:0": "error"})
        with self.assertRaises(GridError) as raised:
            compute_layout(empty, OPTIONS)
        self.assertEqual(raised.exception.code, "no-complete")

    def test_unused_z_axis_with_the_planner_dash_label_gets_no_heading(self):
        snapshot = self.snapshot([[(32, 32)]])
        snapshot["axes"]["z"] = {"field": None, "labels": ["—"]}
        texts = [text["text"] for text in compute_layout(snapshot, OPTIONS)["ops"]["texts"]]
        self.assertFalse(any(text.startswith("Z1") for text in texts), texts)
        snapshot["repeat_count"] = 1
        named = self.snapshot([[(256, 256)]], z_labels=("夜晚",))
        self.assertIn("Z1 · 夜晚", [text["text"] for text in compute_layout(named, OPTIONS)["ops"]["texts"]])

    def test_chinese_labels_render_real_glyphs_with_the_bundled_font(self):
        face = xyz_grid.font(32)
        self.assertEqual(face.getname(), ("Noto Sans SC", "Regular"))
        self.assertEqual(hashlib.sha256(xyz_grid.FONT_PATH.read_bytes()).hexdigest(), xyz_grid.FONT_SHA256)
        glyphs = [bytes(face.getmask(text)) for text in ("步", "数", "")]
        self.assertNotEqual(glyphs[0], glyphs[1])
        self.assertNotEqual(glyphs[0], glyphs[2])
        self.assertTrue(any(glyphs[0]))


class ScaledQualityTests(GridFixture):
    def test_scaled_dimensions_single_pass_and_100_percent_skips_filtering(self):
        snapshot = self.snapshot([[(64, 96), (41, 33)]], alpha={"1:0:0:0"})
        layout = compute_layout(snapshot, {**OPTIONS, "mode": "scaled", "percent": 37})
        sizes = {entry["cell"]: entry["image"][2:] for entry in layout["ops"]["cells"]}
        self.assertEqual(sizes["0:0:0:0"], image_target_size(64, 96, 37))
        self.assertEqual(sizes["0:0:0:0"], (24, 36))
        self.assertEqual(sizes["1:0:0:0"], (15, 12))
        xyz_grid._RESIZE_CALLS[0] = 0
        self.export(snapshot, {**OPTIONS, "mode": "scaled", "percent": 37})
        self.assertEqual(xyz_grid._RESIZE_CALLS[0], 3 + 4)  # RGB: 3 channels, RGBA: 4 — one pass each
        xyz_grid._RESIZE_CALLS[0] = 0
        result = self.export(snapshot, {**OPTIONS, "mode": "scaled", "percent": 100}, stem="hundred")
        self.assertEqual(xyz_grid._RESIZE_CALLS[0], 0)
        self.assertIsNone(json.loads(result["manifest"].read_text(encoding="utf-8"))["recipe"]["algorithm"])

    def test_linear_light_averaging_of_fine_lines(self):
        stripes = np.zeros((64, 64, 3), dtype=np.uint8)
        stripes[:, ::2] = 255
        out = np.asarray(downscale_linear(Image.fromarray(stripes, "RGB"), (32, 32)))[8:24, 8:24]
        # Linear mean 0.5 is sRGB 187.5; gamma-space averaging would give ~128.
        self.assertTrue(np.all(np.abs(out.astype(int) - 188) <= 2), (out.min(), out.max()))
        naive = np.asarray(Image.fromarray(stripes, "RGB").resize((32, 32), Image.Resampling.LANCZOS))[8:24, 8:24]
        self.assertLess(abs(int(naive.mean()) - 128), 4)

    def test_premultiplied_alpha_never_leaks_hidden_rgb(self):
        image = np.zeros((96, 96, 4), dtype=np.uint8)
        image[..., 1] = 255  # hidden green under alpha 0
        canvas = Image.fromarray(image, "RGBA")
        ImageDraw.Draw(canvas).ellipse((20, 20, 76, 76), fill=(255, 0, 0, 255))
        out = np.asarray(downscale_linear(canvas, (38, 38)))
        visible = out[..., 3] > 0
        self.assertTrue(visible.any() and (out[..., 3] < 255).any())
        self.assertLessEqual(int(out[..., 1][visible].max()), 2)
        self.assertTrue(np.all(out[..., :3][out[..., 3] == 0] == 0))
        # A straight (non-premultiplied) per-channel resize shows the leak this guards against.
        naive = np.stack([np.asarray(Image.fromarray(image_channel).resize((38, 38), Image.Resampling.LANCZOS))
                          for image_channel in np.moveaxis(np.asarray(canvas), -1, 0)], axis=-1)
        self.assertGreater(int(naive[..., 1][(naive[..., 3] > 0) & (naive[..., 3] < 255)].max()), 40)

    def test_resize_matches_an_independent_lanczos3_reference(self):
        rng = np.random.default_rng(7)
        channel = rng.random((61, 83), dtype=np.float32)
        ours = xyz_grid._resize_channel(channel, (29, 23))
        reference = reference_resize(channel, (29, 23))
        self.assertLess(float(np.abs(ours - reference).max()), 1e-4)

    def test_overshoot_and_zero_alpha_are_safe(self):
        image = np.zeros((40, 40, 4), dtype=np.uint8)
        image[:, 20:] = (255, 255, 255, 255)
        image[:, 10:20, 3] = 255
        out = np.asarray(downscale_linear(Image.fromarray(image, "RGBA"), (13, 13)))
        self.assertFalse(np.isnan(out.astype(float)).any())
        self.assertTrue(np.all(out[..., :3][out[..., 3] == 0] == 0))

    def test_icc_profiles_are_validated_not_silently_converted(self):
        srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        path = self.outputs / "srgb.png"
        Image.new("RGB", (8, 8), (10, 20, 30)).save(path, icc_profile=srgb)
        self.assertEqual(open_source(path).mode, "RGB")
        other = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
        bad = self.outputs / "lab.png"
        Image.new("RGB", (8, 8)).save(bad, icc_profile=other)
        with self.assertRaises(GridError) as raised:
            open_source(bad)
        self.assertEqual(raised.exception.code, "icc")


class BudgetMetadataPublicationTests(GridFixture):
    def test_budget_refusal_offers_suggestions_and_never_applies_them(self):
        snapshot = self.snapshot([[(64, 64), (64, 64)], [(64, 64), (64, 64)]])
        full = estimate(snapshot, OPTIONS)
        self.assertTrue(full["budget"]["ok"])
        with mock.patch.object(xyz_grid, "EXPORT_MAX_PIXELS", full["width"] * full["height"] // 3):
            refused = estimate(snapshot, OPTIONS)
            self.assertFalse(refused["budget"]["ok"])
            self.assertEqual((refused["width"], refused["height"]), (full["width"], full["height"]))
            self.assertTrue(refused["suggestions"])
            for percent in refused["suggestions"]:
                self.assertTrue(estimate(snapshot, {**OPTIONS, "mode": "scaled", "percent": percent})["budget"]["ok"])
            with self.assertRaises(GridError) as raised:
                self.export(snapshot, OPTIONS)
            self.assertEqual(raised.exception.code, "budget")
        self.assertEqual(sorted(path.name for path in self.outputs.iterdir() if path.name.startswith(("Xirius", "."))), [])

    def test_tiny_and_long_labels_are_warned_never_dropped(self):
        long_label = "很长的标签" * 120
        snapshot = self.snapshot([[(64, 64), (64, 64)]], x_labels=[long_label, "短"])
        tiny = compute_layout(snapshot, {**OPTIONS, "mode": "scaled", "percent": 5})
        codes = {warning["code"] for warning in tiny["warnings"]}
        self.assertIn("tiny-labels", codes)
        long = compute_layout(snapshot, OPTIONS)
        self.assertIn("long-labels", {warning["code"] for warning in long["warnings"]})
        self.assertTrue(all(entry["image"][0] + entry["image"][2] <= long["width"] for entry in long["ops"]["cells"] if entry["image"]))

    def test_grid_metadata_is_independent_and_manifest_is_fingerprinted(self):
        snapshot = self.snapshot([[(48, 48), (48, 48)]])
        result = self.export(snapshot, {**OPTIONS, "mode": "scaled", "percent": 50, "theme": "dark"})
        with Image.open(result["png"]) as image:
            info = dict(image.info)
        self.assertNotIn("parameters", info)
        self.assertNotIn("workflow", info)
        record = json.loads(info["xyz_plot"])
        self.assertEqual(record["kind"], "xyz_plot_grid")
        self.assertEqual(record["manifest"]["sha256"], hashlib.sha256(result["manifest"].read_bytes()).hexdigest())
        self.assertEqual(record["recipe"]["percent"], 50)
        self.assertIn("Lanczos3", record["recipe"]["algorithm"])
        self.assertEqual(len(record["cells"]), 2)
        manifest = json.loads(result["manifest"].read_text(encoding="utf-8"))
        self.assertEqual({cell["id"] for cell in manifest["cells"]}, {"0:0:0:0", "1:0:0:0"})
        self.assertTrue(all(cell["asset"]["sha256"] and cell["request"] for cell in manifest["cells"]))
        with mock.patch.object(xyz_grid, "PNG_METADATA_LIMIT", 200):
            overflow = self.export(snapshot, OPTIONS, stem="overflow")
        self.assertTrue(overflow["metadata_overflow"])
        with Image.open(overflow["png"]) as image:
            self.assertIn("metadata_overflow", json.loads(image.info["xyz_plot"]))

    def test_publication_never_overwrites_and_cancellation_leaves_nothing(self):
        snapshot = self.snapshot([[(32, 32), (32, 32)]])
        first = self.export(snapshot, OPTIONS, stem="same")
        second = self.export(snapshot, OPTIONS, stem="same")
        self.assertEqual((first["png"].name, second["png"].name), ("same.png", "same-02.png"))
        self.assertEqual(second["manifest"].name, "same-02.json")
        before = sorted(path.name for path in self.outputs.iterdir())
        calls = []
        with self.assertRaises(ExportCancelled):
            self.export(snapshot, OPTIONS, stem="cancelled", cancelled=lambda: calls.append(1) or len(calls) > 1)
        self.assertEqual(sorted(path.name for path in self.outputs.iterdir()), before)


if __name__ == "__main__":
    unittest.main()


class AxisSummaryTextTests(unittest.TestCase):
    """The export summary names LoRAs for LoRA axes on X, Y and Z alike, never the internal field."""

    BASELINE = [{"instanceId": "iid-a", "name": "style/NK/ashima_v4.safetensors", "value": "style/NK/ashima_v4.safetensors"},
                {"instanceId": "iid-b", "name": "Kiriaina Anima Style", "value": "style/kiriaina.safetensors"},
                {"instanceId": "iid-g", "name": "Background", "value": "bg.safetensors"}]

    def test_every_lora_axis_kind_names_its_loras_on_each_axis(self):
        weight = {"field": "lora_weight", "target": "iid-a", "values": [{"value": 0.4}, {"value": 0.8}]}
        swap = {"field": "lora_model", "target": "iid-b", "values": [{"value": {"name": "x/one.safetensors", "value": "x/one.safetensors"}}, {"value": {"name": "Two", "value": "two.safetensors"}}]}
        profile = {"field": "lora_profile", "values": [
            {"value": {"mode": "keep_background", "members": [{"instanceId": "iid-a", "weight": "0.6"}], "test_targets": ["iid-a", "iid-b"]}},
            {"value": {"mode": "no_loras", "members": [], "test_targets": []}}]}
        multiplier = {"field": "lora_multiplier", "snapshot": {"mode": "full", "members": [{"instanceId": "iid-b", "name": "Kiriaina Anima Style"}, {"instanceId": "iid-g", "name": "Background"}]}, "values": [{"value": 0.5}]}
        for key in ("x", "y", "z"):
            label = key.upper()
            self.assertEqual(xyz_grid.axis_text(key, weight, self.BASELINE), f"{label}：LoRA 权重 · ashima_v4")
            self.assertEqual(xyz_grid.axis_text(key, swap, self.BASELINE), f"{label}：LoRA 替换 · Kiriaina Anima Style → one、Two")
            self.assertEqual(xyz_grid.axis_text(key, profile, self.BASELINE), f"{label}：LoRA 组合 · ashima_v4、Kiriaina Anima Style")
            self.assertEqual(xyz_grid.axis_text(key, multiplier, self.BASELINE), f"{label}：LoRA 组合倍率 · Kiriaina Anima Style、Background")
            self.assertNotIn("lora_", xyz_grid.axis_text(key, weight, self.BASELINE))

    def test_other_axes_and_long_lists(self):
        self.assertEqual(xyz_grid.axis_text("y", {"field": "steps"}), "Y：steps")
        many = {"field": "lora_model", "target": "missing-target", "values": [{"value": {"name": f"L{i}"}} for i in range(7)]}
        self.assertEqual(xyz_grid.axis_text("z", many, self.BASELINE), "Z：LoRA 替换 · missing-target → L0、L1、L2、L3 等 7 个")
