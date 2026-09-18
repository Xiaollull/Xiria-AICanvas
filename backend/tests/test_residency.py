"""The residency controller: which rung a stage lands on, and that the mechanism never doubles up.

The tier costs quoted here were measured on the reference card (RTX 4060 Laptop, 8188 MiB) with
`waiIllustriousSDXL_v170` and six LoRAs, 8 steps with the warm-up step excluded, and the four
sampling tiers were checked to produce bit-identical latents at both canvases. The numbers live in
`residency.py`'s module docstring; what is tested here is the arithmetic that chooses between them.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from backend import residency
from backend.residency import (
    GIB,
    T0_RESIDENT,
    T1_COMPONENT,
    T2_BLOCK_STREAM,
    T3_LEAF_STREAM,
    T4_SUBMODULE,
    ResidencyController,
    largest_block_group_bytes,
    largest_leaf_bytes,
    module_bytes,
    remove_group_offload,
    tier_for_memory_mode,
)

_device_patch = None


def setUpModule():
    # The controller touches the real allocator (`set_per_process_memory_fraction`, `empty_cache`)
    # whenever it believes a device is present. A test process must not have its allocator reshaped
    # as a side effect, so every test here runs as if there were none unless it says otherwise.
    global _device_patch
    _device_patch = patch.object(residency, "cuda_streams_available", return_value=False)
    _device_patch.start()


def tearDownModule():
    _device_patch.stop()


# The reference mount, as `pipeline_weight_sizes` measures it once the adapters are fused in.
UNET_BYTES = int(5.13 * GIB)
OTHER_BYTES = int(1.77 * GIB)        # two text towers and the autoencoder
BLOCK_GROUP_BYTES = int(2.45 * GIB)  # SDXL's `up_blocks[0]`, the largest single block
LEAF_BYTES = int(0.056 * GIB)
RESERVE = 600 * 1024**2


class FakeUnet(torch.nn.Module):
    pass


class FakeComponent(torch.nn.Module):
    """A text tower or autoencoder that records where it is sent instead of going there.

    A streaming rung places every component but the UNet by hand, and a real module sent to "cuda"
    would need a device — which a test machine may not have.
    """

    def __init__(self, events, name):
        super().__init__()
        self._events = events
        self._name = name

    def to(self, *args, **_kwargs):
        self._events.append(("component_to", self._name, str(args[0]) if args else None))
        return self


class FakePipeline:
    """Enough pipeline to exercise the controller without a device.

    It records the order in which mechanisms are installed and removed, which is the whole point of
    the plan's P1-3: the two are mutually exclusive and exactly one owner may touch them.
    """

    def __init__(self):
        self.events = []
        self.unet = FakeUnet()
        self.vae = FakeComponent(self.events, "vae")
        self.components = {"unet": self.unet, "vae": self.vae}
        self.unet.enable_group_offload = self._group_offload

    def _group_offload(self, **kwargs):
        self.events.append(("group_offload", kwargs["offload_type"], kwargs.get("num_blocks_per_group")))

    def to(self, device, **_kwargs):
        self.events.append(("to", str(device)))
        return self

    def enable_model_cpu_offload(self, device=None):
        self.events.append(("model_cpu_offload", str(device)))

    def enable_sequential_cpu_offload(self, device=None):
        self.events.append(("sequential_cpu_offload", str(device)))

    def remove_all_hooks(self):
        self.events.append(("remove_all_hooks", None))


def controller(free_gb=6.9, stream_capable=True, tier=T1_COMPONENT):
    pipeline = FakePipeline()
    free = [int(free_gb * GIB)]
    with patch.object(residency, "cuda_streams_available", return_value=False):
        instance = ResidencyController(
            pipeline, reserved_bytes=RESERVE, free_memory=lambda: free[0],
            tier=tier, stream_capable=stream_capable,
        )
    instance.unet_bytes = UNET_BYTES
    instance.other_bytes = OTHER_BYTES
    instance.total_weight_bytes = UNET_BYTES + OTHER_BYTES
    instance.block_group_bytes = BLOCK_GROUP_BYTES
    instance.leaf_bytes = LEAF_BYTES
    return instance, pipeline, free


class TierSelectionTests(unittest.TestCase):
    def test_the_reference_redraw_streams_where_it_used_to_offload_per_submodule(self):
        # 1152x1920, one physical forward: the UNet alone is more than the free memory can hold
        # beside the activations, and the old ladder's only answer below that was T4.
        instance, _pipeline, _free = controller(free_gb=6.9)
        redraw_inference = int(1.91 * GIB)
        self.assertEqual(instance.select(redraw_inference), T3_LEAF_STREAM)
        self.assertGreater(instance.requirement(T1_COMPONENT, redraw_inference), int(6.9 * GIB))
        self.assertLess(instance.requirement(T3_LEAF_STREAM, redraw_inference), int(6.9 * GIB))

    def test_a_card_with_room_keeps_the_model_resident(self):
        instance, _pipeline, _free = controller(free_gb=22.0)
        self.assertEqual(instance.select(int(1.91 * GIB)), T0_RESIDENT)

    def test_each_stage_is_placed_by_what_that_stage_needs(self):
        # The point of the phase: admission sizes the chain against its worst stage, so a base pass
        # with room to spare used to be placed by what the redraw needed.
        instance, _pipeline, _free = controller(free_gb=7.6)
        base = instance.select(int(1.27 * GIB))
        redraw = instance.select(int(1.91 * GIB))
        self.assertEqual(base, T1_COMPONENT)
        self.assertEqual(redraw, T3_LEAF_STREAM)

    def test_without_a_usable_stream_block_grouping_replaces_leaf_streaming(self):
        # Leaf-level without a stream loses its prefetch and becomes T4 with extra bookkeeping.
        instance, _pipeline, _free = controller(free_gb=6.9, stream_capable=False)
        self.assertNotIn(T3_LEAF_STREAM, instance.candidate_tiers())
        self.assertIn(T2_BLOCK_STREAM, instance.candidate_tiers())

    def test_a_card_that_cannot_hold_one_leaf_falls_all_the_way_to_the_last_resort(self):
        instance, _pipeline, _free = controller(free_gb=1.0)
        self.assertEqual(instance.select(int(1.91 * GIB)), T4_SUBMODULE)

    def test_the_streaming_rungs_budget_the_components_they_pin(self):
        # Group offload is installed on the UNet alone, so everything else is placed by hand and
        # has to be paid for — a card that cannot afford them belongs on T4, where accelerate
        # places them itself.
        instance, _pipeline, _free = controller()
        required = instance.requirement(T3_LEAF_STREAM, 0)
        self.assertEqual(required, OTHER_BYTES + 2 * LEAF_BYTES + RESERVE)

    def test_admission_modes_name_the_rung_they_leave_installed(self):
        self.assertEqual(tier_for_memory_mode("high_vram"), T0_RESIDENT)
        self.assertEqual(tier_for_memory_mode("normal_vram"), T1_COMPONENT)
        self.assertEqual(tier_for_memory_mode("low_vram"), T4_SUBMODULE)
        self.assertEqual(tier_for_memory_mode("ultra_low_vram"), T4_SUBMODULE)


class MechanismTests(unittest.TestCase):
    def test_a_switch_removes_the_old_mechanism_before_installing_the_new_one(self):
        # P1-3's second trap: `enable_model_cpu_offload` raises while group offload is live, and
        # `apply_group_offloading` refuses a module carrying accelerate hooks.
        instance, pipeline, _free = controller()
        pipeline.events.clear()
        instance.set_tier(T3_LEAF_STREAM, reason="test")
        kinds = [event[0] for event in pipeline.events]
        self.assertLess(kinds.index("remove_all_hooks"), kinds.index("group_offload"))
        self.assertIn(("group_offload", "leaf_level", None), pipeline.events)
        # Group offload covers the UNet alone; everything else is placed on the card by hand, or a
        # later prompt encode would meet CUDA inputs on a host-resident tower.
        self.assertIn(("component_to", "vae", "cuda"), pipeline.events)

    def test_setting_the_rung_already_installed_changes_nothing(self):
        instance, pipeline, _free = controller(tier=T3_LEAF_STREAM)
        pipeline.events.clear()
        self.assertFalse(instance.set_tier(T3_LEAF_STREAM, reason="test"))
        self.assertEqual(pipeline.events, [])

    def test_reinstall_puts_a_streaming_rung_back_rather_than_the_submodule_path(self):
        # Both streaming rungs report `low_vram`, so a restore that went through the mode ladder
        # would put a streaming job onto `enable_sequential_cpu_offload` — 5.6 s/it against 13.
        instance, pipeline, _free = controller(tier=T3_LEAF_STREAM)
        pipeline.events.clear()
        instance.reinstall()
        self.assertIn(("group_offload", "leaf_level", None), pipeline.events)
        self.assertNotIn("sequential_cpu_offload", [event[0] for event in pipeline.events])

    def test_a_block_rung_without_streams_keeps_the_requested_group_size(self):
        instance, pipeline, _free = controller(stream_capable=False)
        pipeline.events.clear()
        instance.set_tier(T2_BLOCK_STREAM, reason="test", blocks_per_group=3)
        self.assertIn(("group_offload", "block_level", 3), pipeline.events)
        self.assertEqual(instance.blocks_per_group, 3)


class TraceTests(unittest.TestCase):
    def test_a_job_records_every_move_with_the_free_memory_that_caused_it(self):
        instance, _pipeline, free = controller(free_gb=6.9)
        instance.begin_job()
        instance.plan_for_stage("base_sampling", int(1.27 * GIB), canvas=(768, 1280))
        instance.note_parked("base_sampling")
        instance.reinstall()
        instance.plan_for_stage("hires_sampling", int(1.91 * GIB), canvas=(1152, 1920))
        transitions = [entry for entry in instance.trace if entry["changed"]]
        self.assertGreaterEqual(len(transitions), 2)
        for entry in instance.trace:
            self.assertIn("free_bytes", entry)
            self.assertTrue(entry["reason"])
        self.assertEqual([entry["reason"] for entry in transitions][:2],
                         ["stage_plan_demote", "park_for_decode"])

    def test_each_job_gets_its_own_trace(self):
        instance, _pipeline, _free = controller()
        instance.begin_job()
        instance.plan_for_stage("base_sampling", int(1.27 * GIB))
        self.assertEqual(len(instance.trace), 1)
        instance.begin_job()
        self.assertEqual(instance.trace, [])

    def test_every_move_is_announced_as_it_happens_and_nothing_else_is(self):
        # P1-7: a transition has to be visible while the job runs, not only in the record after it.
        instance, _pipeline, _free = controller(free_gb=6.9)
        announced = []
        instance.on_transition = announced.append
        instance.plan_for_stage("base_sampling", int(1.91 * GIB))   # T1 -> T3
        instance.plan_for_stage("hires_sampling", int(1.91 * GIB))  # unchanged
        instance.note_parked("hires_sampling")                       # T3 -> parked
        self.assertEqual([(entry["from"], entry["to"]) for entry in announced],
                         [(T1_COMPONENT, T3_LEAF_STREAM), (T3_LEAF_STREAM, "parked")])

    def test_a_failing_announcement_never_fails_the_move(self):
        instance, _pipeline, _free = controller(free_gb=6.9)

        def broken(_entry):
            raise RuntimeError("console gone")

        instance.on_transition = broken
        instance.plan_for_stage("base_sampling", int(1.91 * GIB))
        self.assertEqual(instance.tier, T3_LEAF_STREAM)

    def test_the_console_line_names_both_rungs_the_free_memory_and_the_reason(self):
        from backend.progress_console import format_residency

        line = format_residency({
            "stage": "base_sampling", "from": "T1", "to": "T3",
            "free_bytes": int(5.19 * GIB), "reason": "stage_plan_demote",
        })
        self.assertIn("Residency · base sampling", line)
        self.assertIn("component → leaf stream", line)
        self.assertIn("stage plan demote", line)
        self.assertIn("free", line)
        self.assertIsNone(format_residency(None))

    def test_the_trace_cannot_grow_without_bound(self):
        instance, _pipeline, _free = controller()
        instance._trace_limit = 4
        for index in range(20):
            instance._record(f"stage_{index}", T1_COMPONENT, T3_LEAF_STREAM, 0, "test")
        self.assertEqual(len(instance.trace), 4)

    def test_the_snapshot_names_the_rung_and_counts_the_moves(self):
        instance, _pipeline, _free = controller(free_gb=6.9)
        instance.begin_job()
        instance.plan_for_stage("base_sampling", int(1.91 * GIB))
        snapshot = instance.snapshot()
        self.assertEqual(snapshot["tier"], T3_LEAF_STREAM)
        self.assertEqual(snapshot["offload_mode"], "leaf_stream")
        self.assertEqual(snapshot["memory_mode"], "low_vram")
        self.assertEqual(snapshot["transitions"], 1)


class HysteresisTests(unittest.TestCase):
    def test_a_promotion_needs_room_to_spare_and_happens_once_per_stage(self):
        instance, _pipeline, free = controller(free_gb=6.9, tier=T3_LEAF_STREAM)
        instance.plan_for_stage("base_sampling", int(1.27 * GIB))
        free[0] = int(22.0 * GIB)
        self.assertTrue(instance.promote_if_fits(int(1.27 * GIB)))
        self.assertEqual(instance.tier, T1_COMPONENT)
        self.assertFalse(instance.promote_if_fits(int(1.27 * GIB)))

    def test_a_promotion_that_only_just_fits_is_refused(self):
        instance, _pipeline, free = controller(tier=T3_LEAF_STREAM)
        instance.plan_for_stage("base_sampling", int(1.27 * GIB))
        # Exactly the requirement, with none of the 15% margin the plan asks for.
        free[0] = instance.requirement(T1_COMPONENT, int(1.27 * GIB))
        self.assertFalse(instance.promote_if_fits(int(1.27 * GIB)))

    def test_a_demotion_waits_until_free_memory_is_down_to_one_onload(self):
        instance, _pipeline, free = controller(tier=T1_COMPONENT)
        self.assertFalse(instance.demote_if_pressed(int(1.27 * GIB)))
        free[0] = LEAF_BYTES + 100 * 1024**2
        self.assertTrue(instance.demote_if_pressed(int(1.27 * GIB)))
        self.assertEqual(instance.tier, T3_LEAF_STREAM)


class EscalationTests(unittest.TestCase):
    """A mid-sampling OOM under the controller: one rung down its ladder, same noise replayed.

    The mode ladder cannot express this — T3 and T4 both report `low_vram` — and routing the retry
    through `reconfigure_memory_strategy` would have the controller reinstall the rung that had just
    run out of memory, so the retry would fail the same way and the job with it.
    """

    def strategy(self, mode, tier):
        return {
            "mode": mode, "label": "x", "reason": "r", "offload_mode": "model",
            "cfg_mode": "serial", "cfg_serial_eligible": True, "residency_tier": tier,
            "admission": {"fallback": None},
        }

    def oom_once(self, generator, draws):
        def call():
            draws.append(torch.rand(1, generator=generator).item())
            if len(draws) == 1:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return "image"
        return call

    def test_an_oom_steps_one_rung_down_and_replays_the_same_noise(self):
        from backend import inference_server

        instance, pipeline, _free = controller(tier=T1_COMPONENT)
        generator = torch.Generator().manual_seed(7)
        draws = []
        with patch.object(inference_server, "active_residency", instance), \
                patch.object(inference_server, "active_memory_strategy", self.strategy("normal_vram", T1_COMPONENT)):
            result = inference_server.run_with_offload_escalation(
                pipeline, {"generator": generator}, self.oom_once(generator, draws)
            )
            escalated = inference_server.active_memory_strategy
        self.assertEqual(result, "image")
        self.assertEqual(draws[0], draws[1])
        self.assertEqual(instance.tier, T3_LEAF_STREAM)
        self.assertEqual(escalated["offload_mode"], "leaf_stream")
        self.assertEqual(escalated["admission"]["fallback"]["to"], "leaf_stream")
        self.assertIn("oom_escalation", [entry["reason"] for entry in instance.trace])
        self.assertIn(("group_offload", "leaf_level", None), pipeline.events)

    def test_at_the_bottom_of_the_ladder_an_oom_falls_through_to_ultra_low(self):
        from backend import inference_server

        instance, pipeline, _free = controller(tier=T4_SUBMODULE)
        generator = torch.Generator().manual_seed(7)
        draws = []
        with patch.object(inference_server, "active_residency", instance), \
                patch.object(inference_server, "active_memory_strategy", self.strategy("low_vram", T4_SUBMODULE)), \
                patch.object(inference_server, "reconfigure_memory_strategy") as reconfigure:
            inference_server.run_with_offload_escalation(
                pipeline, {"generator": generator}, self.oom_once(generator, draws)
            )
        self.assertEqual(reconfigure.call_args[0][1]["mode"], "ultra_low_vram")
        self.assertEqual(instance.tier, T4_SUBMODULE)
        self.assertEqual(draws[0], draws[1])

    def test_the_ladder_knows_what_is_below_each_rung(self):
        instance, _pipeline, _free = controller(tier=T1_COMPONENT)
        self.assertEqual(instance.next_lower_tier(), T3_LEAF_STREAM)
        instance.tier = T4_SUBMODULE
        self.assertIsNone(instance.next_lower_tier())
        no_stream, _pipeline, _free = controller(tier=T1_COMPONENT, stream_capable=False)
        self.assertEqual(no_stream.next_lower_tier(), T2_BLOCK_STREAM)


class AllocatorCapTests(unittest.TestCase):
    """The cap that stops a streaming stage from spilling into host memory.

    Measured on the reference card: a redraw that ran at 4.1 s/it in a fresh process ran at 7.9 s/it
    on the next job, with the process pegged at 7.84 GiB of 8 GiB and the desktop's share on top.
    """

    def device(self, free_gb, reserved_gb, total_gb=8.0):
        fractions = []
        patches = (
            patch.object(residency, "cuda_streams_available", return_value=True),
            patch.object(residency.torch.cuda, "mem_get_info",
                         return_value=(int(free_gb * GIB), int(total_gb * GIB))),
            patch.object(residency.torch.cuda, "memory_reserved", return_value=int(reserved_gb * GIB)),
            patch.object(residency.torch.cuda, "get_device_properties",
                         return_value=SimpleNamespace(total_memory=int(total_gb * GIB))),
            patch.object(residency.torch.cuda, "set_per_process_memory_fraction",
                         side_effect=lambda fraction, _device: fractions.append(fraction)),
            patch.object(residency.torch.cuda, "empty_cache"),
        )
        return patches, fractions

    def test_the_cap_is_our_share_plus_what_is_free_less_a_margin(self):
        instance, _pipeline, _free = controller()
        patches, fractions = self.device(free_gb=5.0, reserved_gb=2.0)
        for item in patches:
            item.start()
        try:
            cap = instance.cap_allocator()
        finally:
            for item in patches:
                item.stop()
        self.assertEqual(cap, int(7.0 * GIB) - residency.ALLOCATOR_MARGIN_BYTES)
        self.assertAlmostEqual(fractions[-1], cap / int(8.0 * GIB))

    def test_the_cap_never_rises_above_a_configured_wall(self):
        instance, _pipeline, _free = controller()
        instance.allocator_ceiling_bytes = int(6.0 * GIB)
        patches, _fractions = self.device(free_gb=5.0, reserved_gb=2.0)
        for item in patches:
            item.start()
        try:
            self.assertEqual(instance.cap_allocator(), int(6.0 * GIB))
        finally:
            for item in patches:
                item.stop()

    def test_parking_gives_the_decoder_the_whole_ceiling_back(self):
        instance, _pipeline, _free = controller(tier=T3_LEAF_STREAM)
        patches, fractions = self.device(free_gb=5.0, reserved_gb=2.0)
        for item in patches:
            item.start()
        try:
            instance.cap_allocator()
            instance.note_parked("base_sampling")
        finally:
            for item in patches:
                item.stop()
        self.assertEqual(fractions[-1], 1.0)
        self.assertIsNone(instance.allocator_cap_bytes)

    def test_every_stage_plan_ends_capped(self):
        instance, _pipeline, _free = controller(free_gb=6.9)
        patches, fractions = self.device(free_gb=5.0, reserved_gb=2.0)
        for item in patches:
            item.start()
        try:
            instance.plan_for_stage("base_sampling", int(1.91 * GIB))
        finally:
            for item in patches:
                item.stop()
        self.assertIsNotNone(instance.allocator_cap_bytes)
        self.assertLess(fractions[-1], 1.0)


class MeasurementTests(unittest.TestCase):
    def test_block_and_leaf_sizes_come_from_the_module_tree(self):
        # A UNet's `ModuleList` children are what diffusers groups; everything else lands in one
        # unmatched group, and both are candidates for the largest single move.
        module = torch.nn.Module()
        module.blocks = torch.nn.ModuleList([torch.nn.Linear(100, 100), torch.nn.Linear(10, 10)])
        module.mid = torch.nn.Linear(50, 50)
        self.assertEqual(module_bytes(module.blocks[0]), (100 * 100 + 100) * 4)
        self.assertEqual(largest_block_group_bytes(module, 1), (100 * 100 + 100) * 4)
        self.assertEqual(largest_block_group_bytes(module, 2),
                         (100 * 100 + 100) * 4 + (10 * 10 + 10) * 4)
        self.assertEqual(largest_leaf_bytes(module), (100 * 100 + 100) * 4)

    def test_removing_group_offload_offloads_every_live_group_first(self):
        # Removing the hooks with a group still on the card strands that group's tensors there:
        # the hooks are the only thing that would have moved them back.
        offloaded = []

        class Group:
            def offload_(self):
                offloaded.append(True)

        class Registry:
            def __init__(self, hook):
                self._hook = hook
                self.removed = []

            def get_hook(self, name):
                return self._hook if name == "group_offloading" else None

            def remove_hook(self, name, recurse=False):
                self.removed.append(name)

        module = torch.nn.Linear(2, 2)
        module._diffusers_hook = Registry(SimpleNamespace(group=Group()))
        moved = []
        module.to = lambda device: moved.append(str(device))
        self.assertTrue(remove_group_offload(module))
        self.assertEqual(offloaded, [True])
        self.assertEqual(module._diffusers_hook.removed, list(residency.GROUP_OFFLOAD_HOOKS))
        self.assertEqual(moved, ["cpu"])

    def test_a_module_with_no_group_hooks_is_left_alone(self):
        self.assertFalse(remove_group_offload(torch.nn.Linear(2, 2)))
        self.assertFalse(remove_group_offload(None))


if __name__ == "__main__":
    unittest.main()
