import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

import numpy as np
from PIL import Image, ImageDraw

try:
    from . import background_removal
    from .background_removal import (
        _validate_alpha,
        _onnx_alpha,
        _suppress_border_structures,
        extract_foreground,
        custom_models,
        model_catalog,
        parse_prompt_directives,
        transparent_conditioning_prompt,
    )
except ImportError:
    import background_removal
    from background_removal import (
        _validate_alpha,
        _onnx_alpha,
        _suppress_border_structures,
        extract_foreground,
        custom_models,
        model_catalog,
        parse_prompt_directives,
        transparent_conditioning_prompt,
    )


class PromptDirectiveTests(unittest.TestCase):
    def test_detects_and_removes_canonical_directive(self):
        prompt, directives = parse_prompt_directives("1girl, red dress, ({Transparent background})")
        self.assertEqual(prompt, "1girl, red dress")
        self.assertTrue(directives["transparent_background"])

    def test_accepts_case_and_internal_whitespace(self):
        prompt, directives = parse_prompt_directives("cat, ({  transparent   BACKGROUND  }), cute")
        self.assertEqual(prompt, "cat, cute")
        self.assertTrue(directives["transparent_background"])

    def test_removes_multiple_directives_without_damaging_separators(self):
        prompt, directives = parse_prompt_directives(
            "({Transparent background}), character, ({transparent background})"
        )
        self.assertEqual(prompt, "character")
        self.assertTrue(directives["transparent_background"])

    def test_plain_phrase_does_not_enable_feature(self):
        original = "product on a transparent background panel"
        prompt, directives = parse_prompt_directives(original)
        self.assertEqual(prompt, original)
        self.assertFalse(directives["transparent_background"])

    def test_removal_preserves_unrelated_prompt_formatting(self):
        prompt, directives = parse_prompt_directives("first,  second,, third, ({Transparent background})")
        self.assertEqual(prompt, "first,  second,, third")
        self.assertTrue(directives["transparent_background"])

    def test_conditioning_suffix_encourages_extractable_background(self):
        self.assertEqual(
            transparent_conditioning_prompt("a red fox"),
            "a red fox, isolated foreground subject, clean unobstructed silhouette, solid plain background, no background objects touching the subject",
        )


class ForegroundExtractionTests(unittest.TestCase):
    def synthetic_image(self):
        image = Image.new("RGB", (128, 96), (245, 245, 245))
        draw = ImageDraw.Draw(image)
        draw.ellipse((18, 17, 56, 76), fill=(196, 35, 55))
        draw.rectangle((75, 24, 110, 72), fill=(24, 70, 196))
        return image

    @patch.object(background_removal, "model_path", return_value=None)
    def test_algorithm_fallback_produces_rgba_and_preserves_rgb(self, _model_path):
        source = self.synthetic_image()
        output, diagnostics = extract_foreground(source)
        self.assertEqual(output.mode, "RGBA")
        self.assertEqual(output.size, source.size)
        self.assertEqual(diagnostics["method"], "algorithm")
        self.assertGreater(diagnostics["transparent_ratio"], 0.2)

        source_rgb = np.asarray(source)
        output_pixels = np.asarray(output)
        self.assertTrue(np.array_equal(output_pixels[:, :, :3], source_rgb))
        self.assertLessEqual(int(output_pixels[0, 0, 3]), 8)
        self.assertGreaterEqual(int(output_pixels[48, 36, 3]), 247)
        self.assertGreaterEqual(int(output_pixels[48, 92, 3]), 247)

    def test_rejects_effectively_empty_matte(self):
        with self.assertRaisesRegex(RuntimeError, "几乎完全透明"):
            _validate_alpha(np.zeros((32, 32), dtype=np.uint8))

    def test_rejects_effectively_opaque_matte(self):
        with self.assertRaisesRegex(RuntimeError, "几乎完全不透明"):
            _validate_alpha(np.full((16, 16), 255, dtype=np.uint8))

    def test_rejects_matte_without_confident_foreground(self):
        alpha = np.zeros((64, 64), dtype=np.uint8)
        alpha.flat[:40] = 32
        alpha[0, 0] = 255
        with self.assertRaisesRegex(RuntimeError, "未识别到可信主体"):
            _validate_alpha(alpha)

    def test_rejects_matte_with_only_midrange_alpha(self):
        with self.assertRaisesRegex(RuntimeError, "可信透明区域"):
            _validate_alpha(np.full((64, 64), 128, dtype=np.uint8))

    def test_rejects_matte_without_opaque_foreground(self):
        alpha = np.zeros((64, 64), dtype=np.uint8)
        alpha[16:48, 16:48] = 180
        with self.assertRaisesRegex(RuntimeError, "可信不透明前景"):
            _validate_alpha(alpha)

    @patch.object(background_removal, "_load_session")
    def test_probability_output_is_not_passed_through_sigmoid(self, load_session):
        class TensorInfo:
            name = "pixel_values"
            shape = [1, 3, 2, 2]
            type = "tensor(float)"

        class Session:
            def get_inputs(self):
                return [TensorInfo()]

            def run(self, _outputs, _inputs):
                return [np.asarray([[[[0.0, 0.25], [0.75, 1.0]]]], dtype=np.float32)]

        load_session.return_value = Session()
        alpha = _onnx_alpha(
            Image.new("RGB", (2, 2), "white"),
            Path("probability.onnx"),
            {"input_size": 2, "output": "probability"},
        )
        np.testing.assert_array_equal(alpha, np.asarray([[0, 64], [191, 255]], dtype=np.uint8))

    @patch.object(background_removal, "model_path", return_value=None)
    def test_algorithm_handles_maximum_canvas_without_prototype_broadcast(self, _model_path):
        image = Image.new("RGB", (2048, 2048), (248, 248, 248))
        ImageDraw.Draw(image).rectangle((512, 384, 1536, 1664), fill=(35, 68, 170))
        output, diagnostics = extract_foreground(image)
        self.assertEqual(output.size, (2048, 2048))
        self.assertEqual(diagnostics["method"], "algorithm")
        self.assertLessEqual(output.getpixel((0, 0))[3], 8)
        self.assertGreaterEqual(output.getpixel((1024, 1024))[3], 247)

    def test_catalog_prioritizes_complex_scene_model_over_lightweight_fallback(self):
        models = model_catalog()["models"]
        self.assertEqual(models[0]["id"], "birefnet-lite-fp16")
        self.assertEqual(models[0]["input_size"], 1024)
        self.assertEqual(models[0]["output"], "logits")
        self.assertEqual(models[1]["id"], "bria-rmbg-2-int8")
        self.assertEqual(models[1]["output"], "probability")
        self.assertEqual(models[2]["id"], "bria-rmbg-2-fp16")
        self.assertEqual(models[2]["output"], "probability")
        self.assertEqual(models[3]["id"], "u2netp-onnx")

    def test_custom_onnx_models_are_discovered_without_shadowing_catalog_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            original_directory = background_removal.MODEL_DIRECTORY
            background_removal.MODEL_DIRECTORY = Path(temporary)
            try:
                (Path(temporary) / "custom-matte.onnx").write_bytes(b"custom")
                (Path(temporary) / model_catalog()["models"][0]["filename"]).write_bytes(b"catalog")
                (Path(temporary) / "ignored.safetensors").write_bytes(b"weights")
                models = custom_models()
            finally:
                background_removal.MODEL_DIRECTORY = original_directory
        self.assertEqual([model["id"] for model in models], ["local:custom-matte.onnx"])
        self.assertEqual(models[0]["output"], "auto")
        self.assertTrue(models[0]["local"])

    def test_nested_custom_onnx_model_keeps_root_relative_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            original_directory = background_removal.MODEL_DIRECTORY
            background_removal.MODEL_DIRECTORY = Path(temporary)
            nested = Path(temporary) / "custom folder" / "deep"
            nested.mkdir(parents=True)
            (nested / "matte.onnx").write_bytes(b"custom")
            try:
                models = custom_models()
            finally:
                background_removal.MODEL_DIRECTORY = original_directory
        self.assertEqual(models[0]["id"], "local:custom folder/deep/matte.onnx")
        self.assertEqual(models[0]["filename"], "custom folder/deep/matte.onnx")

    @patch.object(background_removal, "_onnx_alpha")
    @patch.object(background_removal, "installed_models")
    def test_extract_foreground_prefers_first_installed_model(self, installed, predict):
        preferred = {
            "id": "birefnet-lite-fp16",
            "revision": "fixed",
            "input_size": 1024,
            "output": "logits",
        }
        fallback = {
            "id": "u2netp-onnx",
            "revision": "fallback",
            "input_size": 320,
            "output": "minmax",
        }
        installed.return_value = [(preferred, "preferred.onnx"), (fallback, "fallback.onnx")]
        predict.return_value = np.pad(np.full((64, 32), 255, dtype=np.uint8), ((0, 0), (16, 16)))
        with patch.object(background_removal, "background_removal_status", return_value={"runtime_available": True}):
            output, diagnostics = extract_foreground(Image.new("RGB", (64, 64), "white"))
        self.assertEqual(output.mode, "RGBA")
        self.assertEqual(diagnostics["method"], "birefnet-lite-fp16")
        predict.assert_called_once_with(ANY, "preferred.onnx", preferred)

    @patch.object(background_removal, "background_removal_status", return_value={"runtime_available": True})
    @patch.object(background_removal, "resolved_model_path", return_value="selected.onnx")
    @patch.object(background_removal, "model_by_id")
    @patch.object(background_removal, "installed_models")
    @patch.object(background_removal, "_onnx_alpha")
    def test_selected_custom_model_uses_only_hidden_fallback(self, predict, installed, selected_model, _resolved, _status):
        selected = {"id": "local:custom.onnx", "filename": "custom.onnx", "input_size": 1024, "output": "auto", "local": True}
        fallback = {"id": "u2netp-onnx", "revision": "fallback", "filename": "u2netp.onnx", "input_size": 320, "output": "minmax", "selectable": False}
        unrelated = {"id": "birefnet-lite-fp16", "revision": "other", "filename": "lite.onnx", "input_size": 1024, "output": "logits"}
        selected_model.return_value = selected
        installed.return_value = [(unrelated, "lite.onnx"), (fallback, "fallback.onnx")]
        predict.side_effect = [RuntimeError("incompatible"), np.pad(np.full((64, 32), 255, dtype=np.uint8), ((0, 0), (16, 16)))]
        output, diagnostics = extract_foreground(Image.new("RGB", (64, 64), "white"), selected["id"])
        self.assertEqual(output.mode, "RGBA")
        self.assertEqual(diagnostics["method"], "u2netp-onnx")
        self.assertEqual(diagnostics["requested_model"], selected["id"])
        self.assertIn("内置轻量模型回退", diagnostics["warning"])
        self.assertEqual([call.args[1] for call in predict.call_args_list], ["selected.onnx", "fallback.onnx"])

    def test_birefnet_cleanup_removes_thin_edge_structure_but_keeps_detached_subjects(self):
        alpha = np.zeros((128, 128), dtype=np.uint8)
        y, x = np.ogrid[:128, :128]
        alpha[(x - 64) ** 2 + (y - 70) ** 2 <= 26 ** 2] = 255
        alpha[:48, 62:66] = 255
        alpha[20:30, 104:114] = 255
        cleaned = _suppress_border_structures(alpha)
        self.assertLessEqual(int(cleaned[0, 64]), 8)
        self.assertGreaterEqual(int(cleaned[70, 64]), 247)
        self.assertGreaterEqual(int(cleaned[25, 109]), 247)


if __name__ == "__main__":
    unittest.main()
