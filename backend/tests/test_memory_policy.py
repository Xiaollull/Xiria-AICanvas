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


# The reference workload behind the SD/iL fix, in the numbers the machine reported: an RTX 4060
# Laptop (8188 MiB), `waiIllustriousSDXL_v170` at 6.461 GiB of measured weight bytes, its 4.78 GiB
# UNet, six LoRAs at 0.52 GiB, and the 1152x1920 hires redraw that admission sizes the whole chain
# against. Every figure here is one this workload actually produced, so the test moves when the
# hardware or the estimate does rather than when someone edits a round number.
REFERENCE_CARD_BYTES = 8585216000
REFERENCE_WEIGHT_BYTES = int(6.461 * GIB)
REFERENCE_UNET_BYTES = int(4.78 * GIB)
REFERENCE_LORA_BYTES = int(0.52 * GIB)
REFERENCE_REDRAW = (1152, 1920)


class MemoryPolicyTests(unittest.TestCase):
    def policy(self, total_gb, free_gb, weight_gb, family="sdxl", requested="auto", edge=1024, cfg=7.5,
               resident_first=False):
        weight_bytes = int(weight_gb * GIB)
        return select_memory_policy(
            requested,
            total_bytes=int(total_gb * GIB),
            free_bytes=int(free_gb * GIB),
            runtime_weight_bytes=weight_bytes,
            inference_bytes=estimate_inference_bytes(family, edge, edge, cfg),
            largest_component_bytes=estimate_largest_component_bytes(weight_bytes, family),
            reserved_bytes=default_reserved_vram_bytes(int(total_gb * GIB), "nt"),
            resident_first=resident_first,
        )

    def reference_policy(self, requested="sdxl_balanced", free_gb=6.9, total_bytes=REFERENCE_CARD_BYTES,
                         guidance_copies=1, resident_first=True):
        width, height = REFERENCE_REDRAW
        return select_memory_policy(
            requested,
            total_bytes=total_bytes,
            free_bytes=int(free_gb * GIB),
            runtime_weight_bytes=REFERENCE_WEIGHT_BYTES + REFERENCE_LORA_BYTES,
            inference_bytes=estimate_inference_bytes(
                "sdxl", width, height, 5.0, 1, guidance_copies=guidance_copies
            ),
            largest_component_bytes=REFERENCE_UNET_BYTES + REFERENCE_LORA_BYTES,
            reserved_bytes=default_reserved_vram_bytes(total_bytes, "nt"),
            resident_first=resident_first,
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

    def test_the_reference_illustrious_job_is_admitted_for_component_offload(self):
        # The bug this rule exists for: NORMAL used to ask for `largest x 1.10 + two CFG copies +
        # reserve` to be free before the first step, which on this card came to 9.49 GiB against a
        # 7.996 GiB device. No canvas could satisfy it, so every `sdxl_balanced` job was downgraded
        # to the per-submodule path and ran at 8.11 s/it.
        policy = self.reference_policy()
        self.assertEqual(policy["mode"], "normal_vram")
        self.assertEqual(policy["normal_admission_rule"], "resident_first")
        self.assertLessEqual(policy["normal_required_bytes"], REFERENCE_CARD_BYTES)

    def test_batched_cfg_is_what_the_reference_card_cannot_admit(self):
        # Serial CFG is not a nicety here: the same job budgeted for two forwards needs more than
        # the card has, which is what the halved activation peak buys back.
        batched = self.reference_policy(guidance_copies=2)
        self.assertEqual(batched["mode"], "low_vram")
        self.assertGreater(batched["normal_required_bytes"], REFERENCE_CARD_BYTES)

    def test_resident_first_is_scoped_and_leaves_the_upfront_rule_alone(self):
        # Without the flag the same numbers still choose the per-submodule path: the native engines
        # go through this function too, and their staged runtimes are written against that rule.
        unscoped = self.reference_policy(resident_first=False)
        self.assertEqual(unscoped["mode"], "low_vram")
        self.assertEqual(unscoped["normal_admission_rule"], "workspace_upfront")
        self.assertEqual(self.policy(8, 7, 6.46, edge=1024, cfg=6.5)["mode"], "low_vram")

    def test_resident_first_still_refuses_a_component_larger_than_the_card(self):
        # A 4 GB card cannot hold this UNet at all, and that is the condition LOW_VRAM is for.
        policy = self.reference_policy(total_bytes=4 * GIB, free_gb=3.8)
        self.assertEqual(policy["mode"], "low_vram")
        self.assertFalse(policy["normal_available"])

    def test_resident_first_refuses_a_move_that_free_memory_cannot_take(self):
        # The device is large enough, but another process owns most of it right now, so moving the
        # component in would fail where the arithmetic says it fits.
        policy = self.reference_policy(free_gb=2.0)
        self.assertEqual(policy["mode"], "low_vram")
        self.assertLessEqual(policy["normal_required_bytes"], REFERENCE_CARD_BYTES)
        self.assertGreater(policy["normal_residency_bytes"], int(2.0 * GIB))

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

    def test_the_anima_estimate_covers_its_measured_peaks_without_running_away(self):
        # Recorded on an RTX 4060 Laptop (8 GB) with anima_baseV10 BF16 at a 1120x1440 tile: the
        # transformer that shares the card at each moment is 3.644 GiB of DiT blocks and embedders,
        # so what the estimate has to cover is what was allocated beside it.
        beside_resident_transformer = 0.66  # VAE encode, the larger of encode and sampling
        decode_with_transformer_evicted = 1.29
        estimate = estimate_inference_bytes("anima", 1120, 1440, 5.0, 1, guidance_copies=1) / GIB
        self.assertGreater(estimate, beside_resident_transformer * 1.5)
        self.assertGreater(estimate, decode_with_transformer_evicted)
        self.assertLess(estimate, beside_resident_transformer * 2.5)

        # The point of the slope. A 1088x1408 base with a full-frame 2x resample unit is admitted
        # against the 2176x2816 redraw, and the generic 0.65 + 0.40/MP fallback made that alone
        # 2.99 GiB — enough to push NORMAL past what an 8 GB card physically has, which put the
        # cheap base pass on block streaming too. It has to stay well clear of that ceiling.
        eight_gb_headroom = (8 * GIB) - int(4.182e9 * 1.10) - default_reserved_vram_bytes(8 * GIB, "nt")
        self.assertLess(estimate_inference_bytes("anima", 2176, 2816, 5.0, 1, guidance_copies=1), eight_gb_headroom)

        # Still a canvas-sensitive estimate, not a flat one.
        self.assertGreater(
            estimate_inference_bytes("anima", 2176, 2816, 5.0, 1, guidance_copies=1),
            estimate_inference_bytes("anima", 1088, 1408, 5.0, 1, guidance_copies=1),
        )


if __name__ == "__main__":
    unittest.main()
