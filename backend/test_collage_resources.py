import base64
import asyncio
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from PIL import GifImagePlugin, Image
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import inference_server


def image_data_url(image, image_format="PNG", **save_options):
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **save_options)
    return f"data:image/{image_format.lower()};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def small_png(color="red"):
    image = Image.new("RGBA", (8, 8), color)
    try:
        return image_data_url(image)
    finally:
        image.close()


def animated_gif():
    first = Image.new("RGBA", (8, 8), "red")
    second = Image.new("RGBA", (8, 8), "blue")
    try:
        return image_data_url(first, "GIF", save_all=True, append_images=[second], duration=[40, 80], loop=0)
    finally:
        first.close()
        second.close()


class CollageResourceBudgetTests(unittest.TestCase):
    def tearDown(self):
        inference_server._COLLAGE_SAVE_RESULTS.clear()

    def test_animated_source_and_output_byte_contracts_are_32_and_128_mib(self):
        mib = 1024 * 1024
        self.assertEqual(inference_server.COLLAGE_MAX_ANIMATED_SOURCE_BYTES, 32 * mib)
        self.assertEqual(inference_server.COLLAGE_MAX_ENCODED_OUTPUT_BYTES, 128 * mib)
        self.assertLessEqual(31 * mib, inference_server.COLLAGE_MAX_ANIMATED_SOURCE_BYTES)
        self.assertGreater(33 * mib, inference_server.COLLAGE_MAX_ANIMATED_SOURCE_BYTES)
        self.assertLessEqual(127 * mib, inference_server.COLLAGE_MAX_ENCODED_OUTPUT_BYTES)
        self.assertGreater(129 * mib, inference_server.COLLAGE_MAX_ENCODED_OUTPUT_BYTES)

    def test_static_collage_pixel_budget_and_decompression_bomb_map_to_413(self):
        request = inference_server.CollageInput(image_data=small_png(), name="budget.png")
        with patch.object(inference_server, "COLLAGE_MAX_PIXELS", 63):
            with self.assertRaises(HTTPException) as caught:
                inference_server.save_collage(request)
        self.assertEqual(caught.exception.status_code, 413)

        with patch.object(inference_server.Image, "open", side_effect=Image.DecompressionBombError("bomb")):
            with self.assertRaises(HTTPException) as caught:
                inference_server.save_collage(request)
        self.assertEqual(caught.exception.status_code, 413)

    def test_request_body_budget_rejects_declared_and_chunked_oversize_before_model_parsing(self):
        async def exercise(headers, bodies):
            sent = []
            queue = [{"type": "http.request", "body": body, "more_body": index < len(bodies) - 1} for index, body in enumerate(bodies)]

            async def downstream(_scope, receive, send):
                while True:
                    message = await receive()
                    if not message.get("more_body"):
                        break
                await send({"type": "http.response.start", "status": 204, "headers": []})
                await send({"type": "http.response.body", "body": b""})

            async def receive():
                return queue.pop(0)

            async def send(message):
                sent.append(message)

            middleware = inference_server.CollageRequestBodyLimitMiddleware(downstream, maximum_bytes=4)
            await middleware({"type": "http", "path": "/api/inference/collages", "headers": headers}, receive, send)
            return sent

        declared = asyncio.run(exercise([(b"content-length", b"5")], [b""]))
        chunked = asyncio.run(exercise([], [b"abc", b"de"]))
        self.assertEqual(declared[0]["status"], 413)
        self.assertEqual(chunked[0]["status"], 413)

    def test_static_collage_decoded_bytes_and_gif_frames_have_explicit_budgets(self):
        request = inference_server.CollageInput(image_data=small_png(), name="bytes.png")
        with patch.object(inference_server, "COLLAGE_MAX_DECODED_BYTES", 4):
            with self.assertRaises(HTTPException) as caught:
                inference_server.save_collage(request)
        self.assertEqual(caught.exception.status_code, 413)

        invalid = inference_server.CollageInput(image_data="data:image/png;base64," + "!" * 32, name="invalid.png")
        with self.assertRaises(HTTPException) as caught:
            inference_server.save_collage(invalid)
        self.assertEqual(caught.exception.status_code, 400)

        gif_request = inference_server.CollageInput(image_data=animated_gif(), name="frames.gif")
        with patch.object(inference_server, "COLLAGE_MAX_GIF_FRAMES", 1):
            with self.assertRaises(HTTPException) as caught:
                inference_server.save_collage(gif_request)
        self.assertEqual(caught.exception.status_code, 413)

    def test_manual_layout_rejects_unknown_version_external_sources_and_utf8_over_one_mib(self):
        placeholder = "data:image/png;base64," + "A" * 32
        external = {"version": 2, "layers": [{"kind": "image", "assetId": "x", "url": "https://example.test/a.png"}]}
        for layout in ({**external, "version": 3}, external):
            with self.assertRaises(ValidationError):
                inference_server.CollageInput(image_data=placeholder, manual_layout=layout)
        too_large = {"version": 2, "layers": [{"kind": "text", "text": "界" * 4000} for _ in range(100)]}
        with self.assertRaises(ValidationError) as caught:
            inference_server.CollageInput(image_data=placeholder, manual_layout=too_large)
        self.assertIn("1 MiB", str(caught.exception))

    def test_manual_layout_enforces_stroke_and_point_limits_before_metadata_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            path = Path(temporary) / "safe.png"
            Image.new("RGB", (8, 8)).save(path)
            asset_id = inference_server.history_asset_token(path)
            source = f"/api/inference/history/assets/{asset_id}"
            points = [{"x": 1, "y": 1}] * (inference_server.COLLAGE_MAX_STROKE_POINTS + 1)
            layout = {"version": 2, "layers": [{"kind": "image", "assetId": asset_id, "url": source, "paintStrokes": [{"tool": "brush", "points": points}]}]}
            try:
                with self.assertRaises(ValidationError):
                    inference_server.CollageInput(image_data="data:image/png;base64," + "A" * 32, manual_layout=layout)
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output

    def test_manual_layout_binds_asset_id_to_real_bounded_history_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            path = Path(temporary) / "source.png"
            Image.new("RGB", (8, 8), "red").save(path)
            asset_id = inference_server.history_asset_token(path)
            source = f"/api/inference/history/assets/{asset_id}"
            valid = {"version": 2, "layers": [{"kind": "image", "assetId": asset_id, "url": source}]}
            try:
                self.assertEqual(inference_server.CollageInput(image_data=small_png(), manual_layout=valid).manual_layout, valid)
                mismatched = {"version": 2, "layers": [{"kind": "image", "assetId": "different", "url": source}]}
                with self.assertRaises(ValidationError):
                    inference_server.CollageInput(image_data=small_png(), manual_layout=mismatched)
                with patch.object(inference_server, "COLLAGE_MAX_HISTORY_COPY_BYTES", 1):
                    with self.assertRaises(ValidationError):
                        inference_server.CollageInput(image_data=small_png(), manual_layout=valid)
                duplicated = {"version": 2, "layers": [valid["layers"][0], dict(valid["layers"][0])]}
                with patch.object(inference_server, "COLLAGE_MAX_LAYOUT_SOURCE_BYTES", path.stat().st_size):
                    with self.assertRaises(ValidationError):
                        inference_server.CollageInput(image_data=small_png(), manual_layout=duplicated)
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output

    def test_animated_inputs_enforce_decoded_bytes_pixels_and_source_frame_limits(self):
        static_request = inference_server.AnimatedCollageInput(layers=[{"url": small_png(), "x": 0, "y": 0, "width": 8, "height": 8}], width=8, height=8)
        with patch.object(inference_server, "COLLAGE_MAX_ANIMATED_INPUT_BYTES", 4):
            with self.assertRaises(inference_server.CollageResourceLimitError):
                inference_server.animated_collage_frames(static_request)
        with patch.object(inference_server, "COLLAGE_MAX_ANIMATED_SOURCE_PIXELS", 63):
            with self.assertRaises(inference_server.CollageResourceLimitError):
                inference_server.animated_collage_frames(static_request)

        gif_request = inference_server.AnimatedCollageInput(layers=[{"url": animated_gif(), "x": 0, "y": 0, "width": 8, "height": 8}], width=8, height=8)
        with patch.object(inference_server, "COLLAGE_MAX_ANIMATED_SOURCE_FRAMES", 1):
            with self.assertRaises(inference_server.CollageResourceLimitError):
                inference_server.animated_collage_frames(gif_request)

    def test_animated_aggregate_target_budget_rejects_clipped_8192_workload(self):
        layers = [{"url": small_png(), "x": 0, "y": 0, "width": 8192, "height": 8192, "clip": {"x": 0, "y": 0, "width": 1, "height": 1}} for _ in range(100)]
        with self.assertRaises(ValidationError) as caught:
            inference_server.AnimatedCollageInput(layers=layers, width=1, height=1)
        self.assertIn("aggregate target pixel", str(caught.exception))

    def test_animated_visible_intersection_precedes_resize_and_negative_clip_is_exact(self):
        request = inference_server.AnimatedCollageInput(
            layers=[{"url": small_png("red"), "x": 0, "y": 0, "width": 8192, "height": 8192, "clip": {"x": -2, "y": 0, "width": 3, "height": 1}}],
            width=3,
            height=1,
        )
        actual_resize = Image.Image.resize
        with patch.object(Image.Image, "resize", autospec=True, side_effect=lambda image, size, *args, **kwargs: actual_resize(image, size, *args, **kwargs)) as resize:
            frames, _durations = inference_server.animated_collage_frames(request)
        try:
            self.assertEqual(frames[0].getpixel((0, 0))[:3], (255, 0, 0))
            self.assertEqual(frames[0].getpixel((1, 0))[3], 0, "raw clip right is -2 + 3 = 1, not clamped-left + 3")
            self.assertNotIn((8192, 8192), [call.args[1] for call in resize.call_args_list], "the full target must never be resized before its 1px visible intersection")
        finally:
            for frame in frames:
                frame.close()

    def test_animated_composition_allocates_output_and_layer_sized_images_not_one_output_canvas_per_layer(self):
        actual_new = Image.new
        request = inference_server.AnimatedCollageInput(layers=[
            {"url": small_png("red"), "x": 0, "y": 0, "width": 8, "height": 8},
            {"url": small_png("blue"), "x": 8, "y": 8, "width": 8, "height": 8},
        ], width=64, height=64)
        with patch.object(inference_server.Image, "new", wraps=actual_new) as allocate:
            frames, _durations = inference_server.animated_collage_frames(request)
        try:
            output_allocations = [call for call in allocate.call_args_list if len(call.args) > 1 and call.args[1] == (64, 64)]
            self.assertEqual(len(output_allocations), 1)
            self.assertNotIn("placed = Image.new", Path(inference_server.__file__).read_text(encoding="utf-8"))
        finally:
            for frame in frames:
                frame.close()

    def test_hidden_sides_are_not_drawn_for_animated_shared_edges(self):
        image = Image.new("RGBA", (12, 12), (0, 0, 0, 0))
        try:
            inference_server.draw_collage_edge(image, {"x": 1, "y": 1, "width": 10, "height": 10, "hidden_sides": ["right"]}, {"enabled": True, "color": "#c8acfb", "style": "solid", "width": 1})
            self.assertEqual(image.getpixel((1, 5)), (200, 172, 251, 255))
            self.assertEqual(image.getpixel((10, 5)), (0, 0, 0, 0))
        finally:
            image.close()

        request = inference_server.AnimatedCollageInput(layers=[
            {"url": small_png("red"), "x": 0, "y": 0, "width": 8, "height": 8, "hidden_sides": ["right"]},
            {"url": small_png("blue"), "x": 8, "y": 0, "width": 8, "height": 8, "hidden_sides": ["left"]},
        ], width=16, height=8, edge_line={"enabled": True, "color": "#c8acfb", "style": "solid", "width": 1})
        frames, _durations = inference_server.animated_collage_frames(request)
        try:
            self.assertEqual(frames[0].getpixel((7, 4))[:3], (255, 0, 0))
            self.assertEqual(frames[0].getpixel((8, 4))[:3], (0, 0, 255))
            self.assertEqual(frames[0].getpixel((3, 0))[:3], (200, 172, 251))
        finally:
            for frame in frames:
                frame.close()

    def test_save_idempotency_key_returns_one_output_and_rejects_key_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            try:
                request = inference_server.CollageInput(image_data=small_png(), name="once.png", idempotency_key="test-once")
                first = inference_server.save_collage(request)
                second = inference_server.save_collage(request)
                self.assertEqual(first, second)
                self.assertEqual(len(list(Path(temporary).rglob("*.png"))), 1)
                changed = inference_server.CollageInput(image_data=small_png("blue"), name="once.png", idempotency_key="test-once")
                with self.assertRaises(HTTPException) as caught:
                    inference_server.save_collage(changed)
                self.assertEqual(caught.exception.status_code, 409)
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output

    def test_save_idempotency_cache_has_ttl_and_hard_cap(self):
        inference_server._COLLAGE_SAVE_RESULTS["expired"] = (9.0, "digest", {"id": "old"})
        for index in range(inference_server._COLLAGE_SAVE_RESULT_LIMIT + 3):
            inference_server._COLLAGE_SAVE_RESULTS[str(index)] = (20.0, "digest", {"id": str(index)})
        inference_server.prune_collage_save_results(now=10.0)
        self.assertNotIn("expired", inference_server._COLLAGE_SAVE_RESULTS)
        self.assertLessEqual(len(inference_server._COLLAGE_SAVE_RESULTS), inference_server._COLLAGE_SAVE_RESULT_LIMIT)

    def test_animated_render_is_process_single_flight_and_output_is_bounded_while_encoding(self):
        request = inference_server.AnimatedCollageInput(layers=[{"url": small_png(), "x": 0, "y": 0, "width": 8, "height": 8}], width=8, height=8)
        inference_server._COLLAGE_ANIMATED_RENDER_LOCK.acquire()
        try:
            with self.assertRaises(HTTPException) as caught:
                inference_server.render_animated_collage(request)
            self.assertEqual(caught.exception.status_code, 429)
        finally:
            inference_server._COLLAGE_ANIMATED_RENDER_LOCK.release()
        with patch.object(inference_server, "COLLAGE_MAX_ENCODED_OUTPUT_BYTES", 1):
            with self.assertRaises(HTTPException) as caught:
                inference_server.render_animated_collage(request)
        self.assertEqual(caught.exception.status_code, 413)

        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            try:
                with patch.object(inference_server, "COLLAGE_MAX_ENCODED_OUTPUT_BYTES", 1):
                    with self.assertRaises(HTTPException) as caught:
                        inference_server.save_collage(inference_server.CollageInput(image_data=small_png(), name="bounded.png"))
                self.assertEqual(caught.exception.status_code, 413)
                self.assertEqual([path for path in Path(temporary).rglob("*") if path.is_file()], [])
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output

    def test_history_clean_copy_budgets_and_normal_png_gif(self):
        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            png_path = Path(temporary) / "copy.png"
            gif_path = Path(temporary) / "copy.gif"
            Image.new("RGBA", (8, 8), "red").save(png_path)
            first = Image.new("RGBA", (8, 8), "red")
            second = Image.new("RGBA", (8, 8), "blue")
            first.save(gif_path, format="GIF", save_all=True, append_images=[second], duration=[40, 80], loop=0)
            first.close()
            second.close()
            try:
                png = inference_server.copy_history_asset({"asset_id": inference_server.history_asset_token(png_path)})
                gif = inference_server.copy_history_asset({"asset_id": inference_server.history_asset_token(gif_path)})
                self.assertEqual(png.media_type, "image/png")
                self.assertEqual(gif.media_type, "image/gif")
                with Image.open(io.BytesIO(gif.body)) as copied_gif:
                    self.assertEqual(copied_gif.n_frames, 2)
                with patch.object(inference_server, "COLLAGE_MAX_HISTORY_COPY_BYTES", 1):
                    with self.assertRaises(HTTPException) as caught:
                        inference_server.copy_history_asset({"asset_id": inference_server.history_asset_token(png_path)})
                self.assertEqual(caught.exception.status_code, 413)
                with patch.object(inference_server, "COLLAGE_MAX_ENCODED_OUTPUT_BYTES", 1):
                    with self.assertRaises(HTTPException) as caught:
                        inference_server.copy_history_asset({"asset_id": inference_server.history_asset_token(png_path)})
                self.assertEqual(caught.exception.status_code, 413)
                with patch.object(inference_server.Image, "open", side_effect=Image.DecompressionBombError("bomb")):
                    with self.assertRaises(HTTPException) as caught:
                        inference_server.copy_history_asset({"asset_id": inference_server.history_asset_token(png_path)})
                self.assertEqual(caught.exception.status_code, 413)
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output

    def test_gif_duration_normalization_is_total_and_bounded(self):
        normalize = inference_server.normalize_collage_gif_duration
        for value in (None, "40", -1, 0, float("nan"), float("inf"), True):
            self.assertEqual(normalize(value), 100)
        self.assertEqual(normalize(1), 20)
        self.assertEqual(normalize(20), 20)
        self.assertEqual(normalize(123.6), 124)
        self.assertEqual(normalize(60_001), 60_000)

    def test_malformed_gif_duration_is_safe_in_save_and_animated_composition(self):
        original_seek = GifImagePlugin.GifImageFile.seek

        def malformed_seek(image, frame):
            result = original_seek(image, frame)
            image.info["duration"] = "invalid"
            return result

        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            try:
                with patch.object(GifImagePlugin.GifImageFile, "seek", malformed_seek):
                    saved = inference_server.save_collage(inference_server.CollageInput(image_data=animated_gif(), name="duration.gif"))
                    request = inference_server.AnimatedCollageInput(layers=[{
                        "url": animated_gif(), "x": 0, "y": 0, "width": 8, "height": 8,
                    }], width=8, height=8)
                    frames, durations = inference_server.animated_collage_frames(request)
                try:
                    self.assertEqual(durations, [100, 100])
                finally:
                    for frame in frames:
                        frame.close()
                with Image.open(inference_server.history_asset_path(saved["id"])) as output:
                    self.assertEqual(output.info.get("duration"), 100)
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output

    def test_animation_allocation_error_maps_to_413(self):
        request = inference_server.AnimatedCollageInput(layers=[{"url": small_png(), "x": 0, "y": 0, "width": 8, "height": 8}], width=8, height=8)
        with patch.object(inference_server, "animated_collage_frames", side_effect=MemoryError("bounded failure")):
            with self.assertRaises(HTTPException) as caught:
                inference_server.render_animated_collage(request)
        self.assertEqual(caught.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
