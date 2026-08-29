import unittest

from backend.memory_policy import (
    GIB,
    MIB,
    default_reserved_vram_bytes,
    error_looks_like_oom,
    estimate_inference_bytes,
    estimate_largest_component_bytes,
    normalize_memory_mode,
    reserved_vram_bytes,
    effective_vram_limit_bytes,
    select_memory_policy,
    vram_limit_bounds,
)


class MemoryPolicyTests(unittest.TestCase):
    def policy(self, total_gb, free_gb, weight_gb, family="sdxl", requested="auto", edge=1024, cfg=7.5):
        weight_bytes = int(weight_gb * GIB)
        return select_memory_policy(
            requested,
            total_bytes=int(total_gb * GIB),
            free_bytes=int(free_gb * GIB),
            runtime_weight_bytes=weight_bytes,
            inference_bytes=estimate_inference_bytes(family, edge, edge, cfg),
            largest_component_bytes=estimate_largest_component_bytes(weight_bytes, family),
            reserved_bytes=default_reserved_vram_bytes(int(total_gb * GIB), "nt"),
        )

    def test_windows_reservation_matches_comfyui_defaults(self):
        self.assertEqual(default_reserved_vram_bytes(8 * GIB, "nt"), 600 * MIB)
        self.assertEqual(default_reserved_vram_bytes(16 * GIB, "nt"), 700 * MIB)
        self.assertEqual(default_reserved_vram_bytes(16 * GIB, "posix"), 400 * MIB)

    def test_aliases_normalize_to_comfyui_names(self):
        self.assertEqual(normalize_memory_mode("high"), "high_vram")
        self.assertEqual(normalize_memory_mode("balanced"), "normal_vram")
        self.assertEqual(normalize_memory_mode("conservative"), "low_vram")
        self.assertEqual(normalize_memory_mode("sdxl_balanced"), "sdxl_balanced")

    def test_disabling_shared_memory_fallback_keeps_more_vram_headroom(self):
        total = 8 * GIB
        shared = reserved_vram_bytes(total, "nt", True)
        guarded = reserved_vram_bytes(total, "nt", False)
        self.assertEqual(shared, 600 * MIB)
        self.assertGreater(guarded, shared)
        self.assertLessEqual(guarded, 2 * GIB)

    def test_large_gpu_selects_resident_high_vram(self):
        self.assertEqual(self.policy(24, 22, 6.5)["mode"], "high_vram")

    def test_ultra_low_mode_never_promotes_itself(self):
        self.assertEqual(self.policy(24, 22, 6.5, requested="ultra_low_vram")["mode"], "ultra_low_vram")

    def test_eight_gb_sdxl_selects_dynamic_normal_vram_for_small_non_cfg_job(self):
        self.assertEqual(self.policy(8, 7, 6.46, edge=512, cfg=1)["mode"], "normal_vram")

    def test_cfg_and_resolution_select_low_vram_when_workspace_will_not_fit(self):
        self.assertEqual(self.policy(8, 7, 6.46, edge=1024, cfg=6.5)["mode"], "low_vram")

    def test_multi_image_batch_increases_inference_memory_budget(self):
        single = estimate_inference_bytes("sdxl", 1024, 1024, 7, 1)
        batch = estimate_inference_bytes("sdxl", 1024, 1024, 7, 4)
        self.assertGreater(batch, single)

    def test_unsafe_forced_high_mode_downgrades(self):
        policy = self.policy(8, 7, 6.46, requested="high", edge=512, cfg=1)
        self.assertEqual(policy["mode"], "normal_vram")
        self.assertFalse(policy["high_available"])

    def test_normal_vram_keeps_comfyui_component_load_headroom(self):
        largest = 4 * GIB
        inference = GIB
        reserve = 600 * MIB
        policy = select_memory_policy(
            "auto",
            total_bytes=16 * GIB,
            free_bytes=16 * GIB,
            runtime_weight_bytes=8 * GIB,
            inference_bytes=inference,
            largest_component_bytes=largest,
            reserved_bytes=reserve,
        )
        self.assertEqual(policy["normal_required_bytes"], int(largest * 1.10) + inference + reserve)

    def test_async_cuda_oom_descriptions_are_detected(self):
        self.assertTrue(error_looks_like_oom(RuntimeError("CUDA out of memory")))
        accelerator_error = RuntimeError("asynchronous accelerator failure")
        accelerator_error.error_code = 2
        self.assertTrue(error_looks_like_oom(accelerator_error))
        self.assertFalse(error_looks_like_oom(RuntimeError("shape mismatch")))

    def test_vram_wall_leaves_hardware_reserve_and_has_modest_floor(self):
        minimum, maximum, reserve = vram_limit_bounds(8 * GIB, "nt", True)
        self.assertEqual(reserve, 600 * MIB)
        self.assertEqual(maximum, 8 * GIB - reserve)
        self.assertEqual(minimum, 2 * GIB)
        self.assertEqual(effective_vram_limit_bytes(8 * GIB, 0, "nt", True), 8 * GIB)

    def test_fixed_vram_wall_is_clamped_to_safe_hardware_range(self):
        _minimum, maximum, _reserve = vram_limit_bounds(8 * GIB, "nt", True)
        self.assertEqual(effective_vram_limit_bytes(8 * GIB, 99, "nt", True), maximum)
        self.assertEqual(effective_vram_limit_bytes(8 * GIB, 1, "nt", True), 2 * GIB)


if __name__ == "__main__":
    unittest.main()
