import time
import unittest
from unittest.mock import patch

from backend import inference_server


class HealthScanCacheTests(unittest.TestCase):
    """The control plane calls /api/inference/health before proxying every
    /api/inference/* request, so this endpoint sits in front of every gallery
    image, job poll and catalog read. Three of its fields walk the model tree —
    tens of gigabytes of safetensors in a real installation — and recomputing
    them per request made opening a gallery card wait on a directory scan."""

    def setUp(self):
        inference_server._health_scan_cache = None
        inference_server._health_scan_at = 0.0

    tearDown = setUp

    def _counted(self):
        calls = {"detector": 0, "anima": 0, "upscaler": 0}

        def detector(_root):
            calls["detector"] += 1
            return ["face_yolov8n.pt"]

        def anima():
            calls["anima"] += 1
            return {"anima_available": True}

        def upscaler(_root):
            calls["upscaler"] += 1
            return {"models": []}

        return calls, detector, anima, upscaler

    def test_a_burst_of_requests_scans_the_model_tree_once(self):
        calls, detector, anima, upscaler = self._counted()
        with patch.object(inference_server, "discover_detector_models", detector), \
             patch.object(inference_server, "anima_health_fields", anima), \
             patch.object(inference_server, "upscaler_status", upscaler):
            first = inference_server.health_scan_fields()
            for _ in range(19):
                inference_server.health_scan_fields()
        self.assertEqual(calls, {"detector": 1, "anima": 1, "upscaler": 1})
        # Every caller still receives the complete payload, not a partial one.
        self.assertEqual(first["detector_models"], ["face_yolov8n.pt"])
        self.assertEqual(first["anima"], {"anima_available": True})
        self.assertEqual(first["upscalers"], {"models": []})

    def test_the_window_expires_so_a_new_model_becomes_visible(self):
        calls, detector, anima, upscaler = self._counted()
        with patch.object(inference_server, "discover_detector_models", detector), \
             patch.object(inference_server, "anima_health_fields", anima), \
             patch.object(inference_server, "upscaler_status", upscaler), \
             patch.object(inference_server, "HEALTH_SCAN_TTL_SECONDS", 0.05):
            inference_server.health_scan_fields()
            time.sleep(0.08)
            inference_server.health_scan_fields()
        self.assertEqual(calls["detector"], 2)
        self.assertLessEqual(inference_server.HEALTH_SCAN_TTL_SECONDS, 5.0,
                             "a long window would hide a freshly downloaded model for too long")

    def test_force_bypasses_the_window(self):
        calls, detector, anima, upscaler = self._counted()
        with patch.object(inference_server, "discover_detector_models", detector), \
             patch.object(inference_server, "anima_health_fields", anima), \
             patch.object(inference_server, "upscaler_status", upscaler):
            inference_server.health_scan_fields()
            inference_server.health_scan_fields(force=True)
        self.assertEqual(calls["detector"], 2)

    def test_health_reads_every_scanned_field_through_the_cache(self):
        """A field left calling the scanner directly would keep the per-request
        directory walk that this cache exists to remove."""
        source = inference_server.health.__wrapped__ if hasattr(inference_server.health, "__wrapped__") else inference_server.health
        import inspect

        body = inspect.getsource(source)
        for scanner in ("discover_detector_models(", "anima_health_fields(", "upscaler_status("):
            self.assertNotIn(scanner, body, f"health() must not call {scanner} directly")
        for field in ('scanned["detector_models"]', 'scanned["anima"]', 'scanned["upscalers"]'):
            self.assertIn(field, body)


if __name__ == "__main__":
    unittest.main()
