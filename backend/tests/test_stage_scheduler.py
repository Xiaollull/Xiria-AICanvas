"""QPlan P8 — AUTO · 动态决策: stage-level VRAM scheduling across every engine.

The figures are the 2026-09-26 12 GB report's (`README/BUG/20260926-012814-...-e4ccf38c.log`):
`waiIllustriousSDXL_v140` with five LoRAs fused, 6,937,676,966 weight bytes, a 5,134,927,368-byte
UNet, a 768x1344 base and a 1.5x redraw at 1152x2016 on an RTX 5070 Ti Laptop (12,792,037,376
bytes). Every other band — 6, 8, 10, 11, 16, 24 GB — runs the same mount through the same engine.

Nothing here needs a device: the card is simulated by `FakeCard`, which answers the controller's
budget reads from the rung it is on, and the controller's allocator calls are switched off exactly
as the residency suite switches them off.
"""

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from backend import inference_server, residency, stage_scheduler
from backend.conditioning_reuse import StaticConditioningCache, lora_fusion_signature, tower_signature
from backend.memory_policy import estimate_inference_bytes, estimate_vae_decode_bytes
from backend.performance_settings import normalize_performance_settings
from backend.residency import (
    T0_RESIDENT,
    T1_COMPONENT,
    T2_BLOCK_STREAM,
    T3_LEAF_STREAM,
    T4_SUBMODULE,
    ResidencyController,
)
from backend.stage_scheduler import (
    AUTO_LABEL,
    BELOW_RECOMMENDED_WARNING,
    GIB,
    MIB,
    NATIVE_BLOCK,
    NATIVE_RESIDENT,
    StageBudget,
    StageMemoryError,
    allocator_safety_margin_bytes,
    below_recommended_vram,
    cap_invariant,
    choose_component_tier,
    choose_native_placement,
    decision_record,
    legacy_component_tier,
    png_schedule_note,
)

_device_patch = None


def setUpModule():
    # The controller reshapes the real allocator whenever it believes a device is present. A test
    # process must not have its allocator capped as a side effect, so none of it happens here.
    global _device_patch
    _device_patch = patch.object(residency, "cuda_streams_available", return_value=False)
    _device_patch.start()


def tearDownModule():
    _device_patch.stop()


# The reference mount, from the 12 GB report.
TOTAL_WEIGHT = 6_937_676_966
UNET = 5_134_927_368
OTHER = TOTAL_WEIGHT - UNET
LEAF = int(0.056 * GIB)
BLOCK = int(2.45 * GIB)
TEXT_TOWER = int(1.29 * GIB)
VAE = int(0.155 * GIB)
LOG_CARD = 12_792_037_376
BASE = (768, 1344)
REDRAW = (1152, 2016)
DETAIL = (1024, 1024)
CONTEXT = int(0.45 * GIB)       # the CUDA context and kernels: memory the allocator never sees
CONDITIONING = 64 * MIB         # tensors that stay allocated whatever the placement

BANDS_GB = (6, 8, 10, 11, 12, 16, 24)


def card_bytes(gb):
    return LOG_CARD if gb == 12 else int(gb * GIB)


def platform_reserve(total):
    return 700 * MIB if total > 15 * GIB else 600 * MIB


def inference(canvas, cfg=5.0):
    # Serial CFG: one physical forward, which is what these SDXL jobs are admitted for.
    return estimate_inference_bytes("sdxl", canvas[0], canvas[1], cfg, 1, 1)


class FakePipeline:
    def __init__(self):
        self.events = []
        self.unet = torch.nn.Module()
        self.unet.enable_group_offload = lambda **kwargs: self.events.append(("group_offload", kwargs["offload_type"]))
        self.vae = SimpleNamespace(use_tiling=False, use_slicing=False)
        self.components = {"unet": self.unet}

    def to(self, device, **_kwargs):
        self.events.append(("to", str(device)))
        return self

    def enable_model_cpu_offload(self, device=None):
        self.events.append(("model_cpu_offload", str(device)))

    def enable_sequential_cpu_offload(self, device=None):
        self.events.append(("sequential_cpu_offload", str(device)))

    def remove_all_hooks(self):
        self.events.append(("remove_all_hooks", None))


class FakeCard:
    """A device whose free memory follows where the controller has put the model.

    `other` is what other processes and the desktop hold. `after_first_read`, when set, replaces it
    for every read after the first — the Windows behaviour behind the 12 GB report, where the driver
    reported room for the whole model before it moved in and 0 bytes free after.
    """

    def __init__(self, total, other=int(0.9 * GIB), after_first_read=None):
        self.total = int(total)
        self.other = int(other)
        self.after_first_read = after_first_read
        self.reads = 0
        self.controller = None

    def resident(self):
        controller = self.controller
        if controller is None or controller.parked or controller.tier is None:
            return 0
        return controller.install_resident_bytes(controller.tier)

    def read(self, _resident_bytes=0):
        self.reads += 1
        if self.after_first_read is not None and self.reads > 1:
            self.other = self.after_first_read
        resident = self.resident()
        allocated = resident + CONDITIONING
        free = max(0, self.total - self.other - CONTEXT - allocated)
        return StageBudget(
            driver_free_bytes=free,
            physical_total_bytes=self.total,
            limit_bytes=self.total,
            allocated_bytes=allocated,
            reserved_bytes=allocated,
            active_bytes=allocated,
            resident_bytes=resident,
            platform_reserve_bytes=platform_reserve(self.total),
        )


def controller(card, tier, *, dynamic=True, stream_capable=True):
    pipeline = FakePipeline()
    instance = ResidencyController(
        pipeline,
        reserved_bytes=platform_reserve(card.total),
        free_memory=lambda: card.read().ambient_free_bytes,
        tier=tier,
        stream_capable=stream_capable,
        dynamic=dynamic,
        budget_reader=card.read,
        safety_margin_bytes=allocator_safety_margin_bytes(card.total),
    )
    instance.total_weight_bytes = TOTAL_WEIGHT
    instance.unet_bytes = UNET
    instance.other_bytes = OTHER
    instance.block_group_bytes = BLOCK
    instance.leaf_bytes = LEAF
    instance.text_encoder_bytes = TEXT_TOWER
    instance.vae_bytes = VAE
    instance.resident_bytes = card.resident
    card.controller = instance
    return instance, pipeline


def sampling(canvas, decode=False, pipeline=None):
    demand = {"kind": "sampling", "inference_bytes": inference(canvas)}
    if decode:
        demand["decode_bytes"] = estimate_vae_decode_bytes(canvas[0], canvas[1])
    return demand


LADDER = (T0_RESIDENT, T1_COMPONENT, T3_LEAF_STREAM, T4_SUBMODULE)


class BudgetTests(unittest.TestCase):
    def test_a_resident_model_is_not_charged_for_itself(self):
        # The pre-P8 reading: limit minus everything active, so 7 GB of resident weight made a
        # 12 GB card look as if it had 5 GB for a stage that needs the same 7 GB plus 2.
        budget = StageBudget(
            driver_free_bytes=int(4.2 * GIB), physical_total_bytes=LOG_CARD, limit_bytes=LOG_CARD,
            allocated_bytes=TOTAL_WEIGHT + CONDITIONING, reserved_bytes=TOTAL_WEIGHT + CONDITIONING,
            active_bytes=TOTAL_WEIGHT + CONDITIONING, resident_bytes=TOTAL_WEIGHT,
        )
        self.assertLess(budget.ambient_free_bytes, TOTAL_WEIGHT)
        self.assertEqual(budget.planning_free_bytes, int(4.2 * GIB) + TOTAL_WEIGHT)

    def test_the_log_cap_is_what_the_old_formula_produced(self):
        # 7,377,780,736 reserved + 0 free - 256 MiB = 6.62 GiB, the "allowed" in the OOM message.
        budget = StageBudget(
            driver_free_bytes=0, physical_total_bytes=LOG_CARD, limit_bytes=LOG_CARD,
            allocated_bytes=7_089_619_968, reserved_bytes=7_377_780_736, active_bytes=7_089_619_968,
        )
        self.assertAlmostEqual(budget.allocator_room_bytes / GIB, 6.62, places=2)
        invariant = cap_invariant(budget, inference(REDRAW), allocator_safety_margin_bytes(LOG_CARD))
        self.assertFalse(invariant["holds"])
        self.assertLess(invariant["headroom_bytes"], 0)

    def test_the_margin_follows_the_band_and_leaves_24_gb_unchanged(self):
        self.assertEqual(allocator_safety_margin_bytes(card_bytes(6)), int(0.75 * GIB))
        self.assertEqual(allocator_safety_margin_bytes(card_bytes(8)), int(0.75 * GIB))
        self.assertEqual(allocator_safety_margin_bytes(card_bytes(10)), GIB)
        self.assertEqual(allocator_safety_margin_bytes(card_bytes(12)), GIB)
        self.assertEqual(allocator_safety_margin_bytes(card_bytes(16)), GIB // 2)
        self.assertEqual(allocator_safety_margin_bytes(card_bytes(24)), residency.ALLOCATOR_MARGIN_BYTES)


class VramBandTests(unittest.TestCase):
    """Every band, every stage kind: AUTO lands on a rung of the ladder and never below it."""

    def plan(self, gb, tier, demand, other=int(0.9 * GIB)):
        card = FakeCard(card_bytes(gb), other=other)
        instance, _pipeline = controller(card, tier)
        chosen = instance.plan_for_stage("stage", demand["inference_bytes"], canvas=None, demand=demand)
        return instance, chosen

    def test_every_band_places_every_stage_on_the_ladder(self):
        for gb in BANDS_GB:
            for demand in (sampling(BASE), sampling(REDRAW), sampling(DETAIL, decode=True),
                           {"kind": "encode", "inference_bytes": stage_scheduler.TEXT_ENCODE_TRANSIENT_BYTES}):
                for start in (T0_RESIDENT, T1_COMPONENT, T3_LEAF_STREAM, T4_SUBMODULE):
                    with self.subTest(gb=gb, demand=demand, start=start):
                        instance, chosen = self.plan(gb, start, demand)
                        self.assertIn(chosen, LADDER)
                        entry = instance.last_decision
                        self.assertTrue(entry["cap_holds"] or entry["reason"] == "stage_floor_without_margin")
                        # Never the serial-CFG survival mode: that is not a rung AUTO can reach.
                        self.assertNotEqual(residency.TIER_MEMORY_MODES[chosen], "ultra_low_vram")

    def test_24_gb_keeps_every_stage_resident(self):
        for demand in (sampling(BASE), sampling(REDRAW), sampling(DETAIL, decode=True)):
            _instance, chosen = self.plan(24, T0_RESIDENT, demand)
            self.assertEqual(chosen, T0_RESIDENT)

    def test_16_gb_keeps_the_reference_chain_resident(self):
        for demand in (sampling(BASE), sampling(REDRAW)):
            self.assertEqual(self.plan(16, T0_RESIDENT, demand)[1], T0_RESIDENT)

    def test_12_gb_is_not_blanket_demoted(self):
        # With the desktop's ordinary share in use, the report's card holds the whole chain resident.
        self.assertEqual(self.plan(12, T0_RESIDENT, sampling(BASE))[1], T0_RESIDENT)
        self.assertEqual(self.plan(12, T0_RESIDENT, sampling(REDRAW))[1], T0_RESIDENT)

    def test_12_gb_under_pressure_steps_down_only_as_far_as_it_must(self):
        _instance, chosen = self.plan(12, T0_RESIDENT, sampling(REDRAW), other=int(2.5 * GIB))
        self.assertEqual(chosen, T1_COMPONENT)

    def test_8_and_6_gb_stream_the_unet_rather_than_offload_per_submodule(self):
        for gb in (6, 8):
            with self.subTest(gb=gb):
                chosen = self.plan(gb, T1_COMPONENT, sampling(REDRAW), other=int(0.5 * GIB))[1]
                self.assertEqual(chosen, T3_LEAF_STREAM)

    def test_a_card_without_streams_uses_block_grouping_instead(self):
        card = FakeCard(card_bytes(8), other=int(0.5 * GIB))
        instance, _pipeline = controller(card, T1_COMPONENT, stream_capable=False)
        chosen = instance.plan_for_stage("stage", inference(REDRAW), demand=sampling(REDRAW))
        self.assertIn(chosen, (T2_BLOCK_STREAM, T4_SUBMODULE))


class HysteresisTests(unittest.TestCase):
    def test_a_fitting_stage_is_never_demoted(self):
        card = FakeCard(card_bytes(12))
        instance, pipeline = controller(card, T0_RESIDENT)
        for _ in range(3):
            self.assertEqual(instance.plan_for_stage("stage", inference(BASE), demand=sampling(BASE)), T0_RESIDENT)
        self.assertEqual([event for event in pipeline.events if event[0] == "to"], [])

    def test_the_old_rule_demoted_the_same_resident_stage(self):
        # What the pre-P8 controller saw on this card: its own weights counted against it.
        card = FakeCard(card_bytes(12))
        instance, _pipeline = controller(card, T0_RESIDENT)
        budget = card.read()
        legacy = legacy_component_tier(
            ladder=LADDER, requirement=lambda tier: instance.requirement(tier, inference(BASE)),
            ambient_free=budget.ambient_free_bytes,
        )
        self.assertNotEqual(legacy, T0_RESIDENT)

    def test_promotion_needs_fifteen_percent_to_spare(self):
        demand = sampling(BASE)
        card = FakeCard(card_bytes(12))
        instance, _pipeline = controller(card, T3_LEAF_STREAM)
        need = instance.stage_requirement(T1_COMPONENT, demand)
        # Room for T1 but not for T1 plus 15%: the stage stays where it is.
        budget = StageBudget(
            driver_free_bytes=int(need * 1.1), physical_total_bytes=LOG_CARD, limit_bytes=LOG_CARD,
            allocated_bytes=0, reserved_bytes=0, active_bytes=0,
        )
        choice = choose_component_tier(
            ladder=(T1_COMPONENT, T3_LEAF_STREAM, T4_SUBMODULE), current=T3_LEAF_STREAM,
            requirement=lambda tier: instance.stage_requirement(tier, demand),
            resident_after=lambda tier: 0, workspace=lambda tier: 0, budget=budget, margin_bytes=0,
        )
        self.assertEqual(choice.tier, T3_LEAF_STREAM)
        budget = StageBudget(
            driver_free_bytes=int(need * 1.16), physical_total_bytes=LOG_CARD, limit_bytes=LOG_CARD,
            allocated_bytes=0, reserved_bytes=0, active_bytes=0,
        )
        choice = choose_component_tier(
            ladder=(T1_COMPONENT, T3_LEAF_STREAM, T4_SUBMODULE), current=T3_LEAF_STREAM,
            requirement=lambda tier: instance.stage_requirement(tier, demand),
            resident_after=lambda tier: 0, workspace=lambda tier: 0, budget=budget, margin_bytes=0,
        )
        self.assertEqual((choice.tier, choice.reason), (T1_COMPONENT, "stage_plan_promote"))

    def test_a_prompt_encode_never_promotes_but_a_sampling_stage_does(self):
        # Measured on the 8 GB card: the encode promoted T4 -> T1 and base sampling then moved to T3,
        # two installs where one would do.
        encode = {"kind": "encode", "inference_bytes": stage_scheduler.TEXT_ENCODE_TRANSIENT_BYTES}
        card = FakeCard(card_bytes(12))
        instance, pipeline = controller(card, T4_SUBMODULE)
        self.assertEqual(instance.plan_for_stage("prompt_encode", encode["inference_bytes"], demand=encode,
                                                 legacy_point=False), T4_SUBMODULE)
        self.assertEqual(pipeline.events, [])
        self.assertNotEqual(instance.plan_for_stage("base_sampling", inference(BASE), demand=sampling(BASE)),
                            T4_SUBMODULE)

    def test_a_requirement_exactly_at_the_budget_fits(self):
        budget = StageBudget(
            driver_free_bytes=5 * GIB, physical_total_bytes=8 * GIB, limit_bytes=8 * GIB,
            allocated_bytes=0, reserved_bytes=0, active_bytes=0,
        )
        choice = choose_component_tier(
            ladder=(T1_COMPONENT, T4_SUBMODULE), current=T4_SUBMODULE,
            requirement=lambda tier: 5 * GIB if tier == T1_COMPONENT else GIB,
            resident_after=lambda tier: 0, workspace=lambda tier: 0, budget=budget, margin_bytes=0,
            promotion_margin=1.0,
        )
        self.assertEqual(choice.tier, T1_COMPONENT)


class CapInvariantTests(unittest.TestCase):
    def test_the_12_gb_report_is_demoted_before_the_stage_runs(self):
        # The redraw plan sees room for the resident model; by the time it checks the cap, the
        # driver reports 0 bytes free beside 7 GB of its own weight — the 20 MiB cap of the report.
        hugging = LOG_CARD - CONTEXT - TOTAL_WEIGHT - CONDITIONING
        card = FakeCard(LOG_CARD, other=int(1.2 * GIB), after_first_read=hugging)
        instance, _pipeline = controller(card, T0_RESIDENT)
        chosen = instance.plan_for_stage("hires_sampling_1", inference(REDRAW), canvas=REDRAW, demand=sampling(REDRAW))
        entry = instance.last_decision
        self.assertNotEqual(chosen, T0_RESIDENT)
        self.assertEqual(entry["reason"], "cap_invariant_demote")
        self.assertIn(T0_RESIDENT, entry["cap_demotions"])
        self.assertTrue(entry["cap_holds"])
        self.assertGreaterEqual(entry["cap_bytes"], entry["required_cap_bytes"])
        # The stage it would have run into: a cap a few dozen MiB above what was allocated.
        self.assertGreater(entry["cap_bytes"] - entry["allocated_bytes"], inference(REDRAW))

    def test_a_stage_that_cannot_fit_even_on_the_floor_is_a_readable_error(self):
        card = FakeCard(int(4 * GIB), other=int(3.3 * GIB))
        instance, _pipeline = controller(card, T4_SUBMODULE)
        with self.assertRaises(StageMemoryError) as raised:
            instance.plan_for_stage("hires_sampling_1", inference(REDRAW), canvas=REDRAW, demand=sampling(REDRAW))
        message = str(raised.exception)
        self.assertIn("1152×2016", message)
        self.assertIn("重采样重绘", message)
        self.assertIn("未缩小画布", message)
        self.assertIn("未回退 CPU", message)
        self.assertEqual(raised.exception.diagnostics()["canvas"], [1152, 2016])

    def test_the_floor_runs_without_its_margin_when_the_stage_itself_fits(self):
        demand = sampling(BASE)
        workspace = demand["inference_bytes"] + LEAF
        margin = allocator_safety_margin_bytes(int(6 * GIB))
        # Room for the stage and a third of the margin.
        room = CONDITIONING + workspace + margin // 3
        other = int(6 * GIB) - CONTEXT - CONDITIONING - (room + residency.ALLOCATOR_MARGIN_BYTES - CONDITIONING)
        card = FakeCard(int(6 * GIB), other=other)
        instance, _pipeline = controller(card, T4_SUBMODULE)
        self.assertEqual(instance.plan_for_stage("base_sampling", demand["inference_bytes"], demand=demand), T4_SUBMODULE)
        self.assertEqual(instance.last_decision["reason"], "stage_floor_without_margin")


class LegacyTierTests(unittest.TestCase):
    """The autoencoder's tiling keeps following the rung the pre-P8 rule would have published."""

    def test_the_legacy_rung_moves_where_the_old_rule_would_have_moved(self):
        # A 10 GB card: the old first-fit rule promotes T1 to T0 with no margin; AUTO keeps T1
        # because T0 does not clear the 15% promotion margin.
        card = FakeCard(card_bytes(10), other=int(0.6 * GIB))
        instance, _pipeline = controller(card, T1_COMPONENT)
        self.assertEqual(instance.plan_for_stage("base_sampling", inference(BASE), demand=sampling(BASE)), T1_COMPONENT)
        self.assertEqual(instance.legacy_tier, T0_RESIDENT)

    def test_publishing_reports_the_real_rung_and_tiles_by_the_legacy_one(self):
        card = FakeCard(card_bytes(10), other=int(0.6 * GIB))
        instance, pipeline = controller(card, T1_COMPONENT)
        instance.plan_for_stage("base_sampling", inference(BASE), demand=sampling(BASE))
        strategy = {"mode": "normal_vram", "label": "x", "offload_mode": "model", "dynamic": True, "reason": "r"}
        with patch.object(inference_server, "active_residency", instance), \
                patch.object(inference_server, "active_memory_strategy", strategy):
            inference_server.publish_residency()
            published = inference_server.active_memory_strategy
        self.assertEqual(published["mode"], "normal_vram")
        self.assertEqual(published["legacy_mode"], "high_vram")
        self.assertEqual(published["label"], f"{AUTO_LABEL} · 组件调度")
        # `configure_vae_mode` reads the legacy mode: whole-image, exactly as before P8.
        vae = SimpleNamespace(
            calls=[],
            disable_slicing=lambda: vae.calls.append("-slice"),
            disable_tiling=lambda: vae.calls.append("-tile"),
            enable_slicing=lambda: vae.calls.append("+slice"),
            enable_tiling=lambda: vae.calls.append("+tile"),
        )
        with patch.dict(inference_server.performance_settings, {"vae_mode": "auto"}), \
                patch.object(inference_server, "active_vae_mode", "none"):
            inference_server.configure_vae_mode(SimpleNamespace(vae=vae), published)
        self.assertEqual(vae.calls, ["-slice", "-tile"])

    def test_a_manual_tier_keeps_the_old_rule_exactly(self):
        for gb in BANDS_GB:
            for free_other in (0.4, 1.5, 3.0):
                for start in (T0_RESIDENT, T1_COMPONENT, T3_LEAF_STREAM):
                    with self.subTest(gb=gb, other=free_other, start=start):
                        card = FakeCard(card_bytes(gb), other=int(free_other * GIB))
                        instance, _pipeline = controller(card, start, dynamic=False)
                        ambient = card.read().ambient_free_bytes
                        expected = legacy_component_tier(
                            ladder=instance.candidate_tiers(),
                            requirement=lambda tier: instance.requirement(tier, inference(REDRAW)),
                            ambient_free=ambient,
                        )
                        chosen = instance.plan_for_stage("hires_sampling", inference(REDRAW))
                        self.assertEqual(chosen, expected)
                        self.assertEqual(instance.legacy_tier, chosen)
                        self.assertIsNone(instance.last_decision)


class ConditioningReuseTests(unittest.TestCase):
    KEY = dict(
        family="sdxl", prompt="1girl, cherry blossoms", negative_prompt="lowres", clip_skip=2,
        emphasis="forge", model_signature=("ckpt", 1, 2), lora_signature=("clean", ("a",), None),
    )

    def test_every_part_of_the_key_invalidates(self):
        cache = StaticConditioningCache()
        base = cache.key(**self.KEY)
        for field, value in (
            ("prompt", "1girl"), ("negative_prompt", ""), ("clip_skip", 3), ("family", "sd"),
            ("model_signature", ("ckpt", 1, 3)), ("lora_signature", ("clean", ("b",), None)),
            ("emphasis", "comfy"),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(cache.key(**{**self.KEY, field: value}), base)
        self.assertEqual(cache.key(**self.KEY), base)

    def test_a_static_prompt_is_encoded_once_for_the_whole_chain(self):
        encodes = []

        def fake_encode(pipeline, family, prompt, negative_prompt, *, timeline=None, clip_skip=None, emphasis="forge"):
            encodes.append((prompt, negative_prompt, clip_skip))
            return {"prompt_embeds": object()}

        owner = SimpleNamespace(
            text_encoder=object(), text_encoder_2=object(),
            _xiriai_lora_fusion_state={"schema": 1, "state": "clean", "signature": ("a",)},
        )
        derived = SimpleNamespace(text_encoder=owner.text_encoder, text_encoder_2=owner.text_encoder_2)
        cache = StaticConditioningCache()
        with patch.object(inference_server, "prepare_dynamic_prompt_conditioning", side_effect=fake_encode), \
                patch.object(inference_server, "active_conditioning_cache", cache), \
                patch.object(inference_server, "loaded_pipeline", owner), \
                patch.object(inference_server, "loaded_checkpoint", "wai.safetensors"):
            results = [
                inference_server.stage_prompt_conditioning(
                    pipeline, "sdxl", "1girl, cherry blossoms", "lowres", timeline=None, clip_skip=2,
                )
                # base, the redraw (image-to-image) and three ADetailer units (inpainting)
                for pipeline in (owner, derived, derived, derived, derived)
            ]
            self.assertEqual(len(encodes), 1)
            self.assertTrue(all(result is results[0] for result in results))
            # A different CLIP skip is a different encoding.
            inference_server.stage_prompt_conditioning(owner, "sdxl", "1girl, cherry blossoms", "lowres", timeline=None, clip_skip=3)
            self.assertEqual(len(encodes), 2)
            # A dynamic prompt is never shared: every stage encodes its own timeline.
            for _ in range(2):
                inference_server.stage_prompt_conditioning(owner, "sdxl", "[a:b:5], 1girl", "lowres", timeline=None, clip_skip=2)
            self.assertEqual(len(encodes), 4)
            # A dirty adapter state cannot be trusted with a cached encoding.
            owner._xiriai_lora_fusion_state = {"schema": 1, "state": "dirty", "signature": ("a",)}
            inference_server.stage_prompt_conditioning(owner, "sdxl", "1girl, cherry blossoms", "lowres", timeline=None, clip_skip=2)
            self.assertEqual(len(encodes), 5)
        self.assertEqual(cache.stats()["encodes"], 2)
        self.assertEqual(cache.stats()["reused"], 4)

    def test_a_reused_encoding_plans_no_encode_stage(self):
        planned = []
        cache = StaticConditioningCache()
        owner = SimpleNamespace(text_encoder=object(), text_encoder_2=object(),
                                _xiriai_lora_fusion_state={"schema": 1, "state": "clean", "signature": ()})
        with patch.object(inference_server, "prepare_dynamic_prompt_conditioning", return_value={"x": 1}), \
                patch.object(inference_server, "active_conditioning_cache", cache), \
                patch.object(inference_server, "loaded_pipeline", owner), \
                patch.object(inference_server, "loaded_checkpoint", "wai.safetensors"):
            for _ in range(3):
                inference_server.stage_prompt_conditioning(
                    owner, "sdxl", "p", "n", timeline=None, clip_skip=2, before_encode=lambda: planned.append(1),
                )
        self.assertEqual(planned, [1])

    def test_the_signatures_follow_the_towers_and_the_adapter_state(self):
        towers = SimpleNamespace(text_encoder=object(), text_encoder_2=object())
        self.assertEqual(tower_signature(towers, "a"), tower_signature(SimpleNamespace(**vars(towers)), "a"))
        self.assertNotEqual(tower_signature(towers, "a"), tower_signature(towers, "b"))
        self.assertIsNone(lora_fusion_signature(SimpleNamespace(_xiriai_lora_fusion_state={"state": "dirty"})))

    def test_the_job_owns_its_cache(self):
        source = inspect.getsource(inference_server.run_generation)
        self.assertIn("active_conditioning_cache = StaticConditioningCache()", source)
        self.assertIn("active_conditioning_cache.clear()", source)


class NativePlacementTests(unittest.TestCase):
    TRANSFORMER = int(3.9 * GIB)
    GROUP = int(0.18 * GIB)

    def budget(self, free_gb, total_gb=8, resident=0):
        allocated = resident + CONDITIONING
        return StageBudget(
            driver_free_bytes=int(free_gb * GIB), physical_total_bytes=int(total_gb * GIB),
            limit_bytes=int(total_gb * GIB), allocated_bytes=allocated, reserved_bytes=allocated,
            active_bytes=allocated, resident_bytes=resident,
        )

    def choose(self, current, free_gb, locked=False, total_gb=8):
        inference_bytes = estimate_inference_bytes("anima", 1088, 1408, 4.0, 1, 1)
        return choose_native_placement(
            current=current,
            resident_required=int(self.TRANSFORMER * 1.1) + inference_bytes + 600 * MIB,
            block_required=int(self.GROUP * 1.1) + inference_bytes + 600 * MIB,
            budget=self.budget(free_gb, total_gb), margin_bytes=allocator_safety_margin_bytes(int(total_gb * GIB)),
            resident_workspace=inference_bytes, block_workspace=inference_bytes,
            transformer_bytes=self.TRANSFORMER, block_bytes=self.GROUP, locked=locked,
        )

    def test_a_stage_that_fits_resident_stays_resident(self):
        self.assertEqual(self.choose(NATIVE_RESIDENT, 7.0).tier, NATIVE_RESIDENT)

    def test_a_stage_that_does_not_fit_streams_a_block_at_a_time(self):
        self.assertEqual(self.choose(NATIVE_RESIDENT, 3.0).tier, NATIVE_BLOCK)

    def test_streaming_is_left_only_with_room_to_spare(self):
        need = int(self.TRANSFORMER * 1.1) + estimate_inference_bytes("anima", 1088, 1408, 4.0, 1, 1) + 600 * MIB
        self.assertEqual(self.choose(NATIVE_BLOCK, need * 1.05 / GIB, total_gb=24).tier, NATIVE_BLOCK)
        self.assertEqual(self.choose(NATIVE_BLOCK, need * 1.2 / GIB, total_gb=24).tier, NATIVE_RESIDENT)

    def test_a_locked_placement_is_kept(self):
        self.assertEqual(self.choose(NATIVE_RESIDENT, 3.0, locked=True).tier, NATIVE_RESIDENT)


class FakeRuntime:
    def __init__(self, group=False, **extra):
        self.weight_sizes = {
            "transformer": int(3.9 * GIB), "transformer_max_block": int(0.14 * GIB),
            "transformer_unmatched": int(0.04 * GIB), "total": int(5.2 * GIB),
        }
        self.components = {}
        self._group = group
        self.events = []
        for name, value in extra.items():
            setattr(self, name, value)

    @property
    def transformer_group_offload_enabled(self):
        return self._group

    def enable_transformer_group_offload(self, blocks):
        self.events.append(("group", blocks))
        self._group = True

    def _remove_transformer_group_offload(self):
        self.events.append(("ungroup",))
        self._group = False

    def configure_transformer_residency(self, keep):
        self.events.append(("keep", keep))


class NativeStageTests(unittest.TestCase):
    def strategy(self, **extra):
        return {
            "mode": "normal_vram", "label": "x", "reason": "r", "dynamic": True, "reserved_bytes": 600 * MIB,
            "physical_forward_copies": 1, "sampling_batch_size": 1, "transformer_blocks_per_group": 0,
            "keep_transformer_resident": False, "offload_mode": "staged_transformer_resident", **extra,
        }

    def run_plan(self, runtime, strategy, free_gb, total_gb=8, stage="base_sampling", canvas=(1088, 1408)):
        budget = StageBudget(
            driver_free_bytes=int(free_gb * GIB), physical_total_bytes=int(total_gb * GIB),
            limit_bytes=int(total_gb * GIB), allocated_bytes=CONDITIONING, reserved_bytes=CONDITIONING,
            active_bytes=CONDITIONING,
        )
        inference_server.reset_stage_schedule()
        with patch.object(inference_server, "active_memory_strategy", strategy), \
                patch.object(inference_server, "read_stage_budget", return_value=budget), \
                patch.object(inference_server.torch.cuda, "is_available", return_value=True), \
                patch.object(inference_server, "loaded_engine", "Anima"):
            target = inference_server.plan_native_stage(stage, "anima", canvas, 4.0, 1, runtime)
            schedule = list(inference_server.active_stage_schedule)
        return target, schedule

    def test_a_squeezed_stage_is_moved_to_block_streaming_and_back(self):
        runtime = FakeRuntime(_diffusion_lora_applied=True)
        strategy = self.strategy()
        target, schedule = self.run_plan(runtime, strategy, free_gb=2.5)
        self.assertEqual(target, NATIVE_BLOCK)
        self.assertEqual(runtime.events, [("keep", False), ("group", 1)])
        self.assertTrue(strategy["transformer_group_offload"])
        self.assertEqual(strategy["memory_path"], NATIVE_BLOCK)
        self.assertEqual(schedule[-1]["path"], NATIVE_BLOCK)
        target, _schedule = self.run_plan(runtime, strategy, free_gb=22, total_gb=24, stage="hires_sampling_1")
        self.assertEqual(target, NATIVE_RESIDENT)
        self.assertIn(("ungroup",), runtime.events)
        self.assertFalse(strategy["transformer_group_offload"])

    def test_a_compiled_transformer_or_a_pending_adapter_is_never_moved(self):
        for extra, strategy_extra in (
            ({"_diffusion_lora_applied": True}, {"acceleration": {"compile_active": True}}),
            ({"lora_plans": ["x"], "_diffusion_lora_applied": False}, {}),
        ):
            with self.subTest(extra=extra, strategy=strategy_extra):
                runtime = FakeRuntime(**extra)
                target, schedule = self.run_plan(runtime, self.strategy(**strategy_extra), free_gb=2.5)
                self.assertEqual(target, NATIVE_RESIDENT)
                self.assertEqual(runtime.events, [])
                self.assertEqual(schedule[-1]["reason"], "native_placement_locked")

    def test_a_stage_that_cannot_stream_even_one_block_is_a_readable_error(self):
        with self.assertRaises(StageMemoryError) as raised:
            self.run_plan(FakeRuntime(), self.strategy(), free_gb=0.5, stage="adetailer_region_1_1", canvas=(1024, 1024))
        self.assertIn("ADetailer 区域重绘", str(raised.exception))

    def test_manual_tiers_are_not_re_planned(self):
        runtime = FakeRuntime()
        target, schedule = self.run_plan(runtime, self.strategy(dynamic=False), free_gb=2.5)
        self.assertIsNone(target)
        self.assertEqual(runtime.events, [])
        self.assertEqual(schedule, [])


class StageMatrixTests(unittest.TestCase):
    """Every stage boundary of QPlan P8 §3.1 has a decision, for component and native engines."""

    def source(self, function):
        return inspect.getsource(function)

    def test_model_mount_is_recorded_by_every_loader(self):
        for loader in (inference_server.load_pipeline, inference_server.load_anima_pipeline,
                       inference_server.load_flux_pipeline):
            with self.subTest(loader=loader.__name__):
                source = self.source(loader)
                self.assertIn("record_mount_decision(", source)
                self.assertIn("reused=True", source)
                self.assertIn("reused=False", source)

    def test_prompt_base_sampling_and_decodes_are_planned_in_the_run(self):
        source = self.source(inference_server.run_generation)
        self.assertIn('"prompt_encode"', source)
        self.assertIn('"base_sampling", family, (request.width, request.height)', source)
        self.assertEqual(source.count('plan_native_stage(\n                    "base_sampling"'), 2)
        self.assertIn('decision_stage="base_decode"', source)
        self.assertIn("record_final_decode(", source)
        self.assertIn("stage_prompt_conditioning(", source)

    def test_every_resample_unit_is_planned_for_its_own_canvas(self):
        source = self.source(inference_server._apply_resample_unit_stages)
        self.assertIn('f"hires_upscale_{unit_number}"', source)
        self.assertIn('f"hires_latent_encode_{unit_number}"', source)
        # SD redraw, native full-frame redraw and native USDU tiles.
        self.assertEqual(source.count('f"hires_sampling_{unit_number}"'), 3)
        self.assertIn('decision_stage=f"hires_decode_{unit_number}"', source)
        self.assertIn("stage_prompt_conditioning(", source)

    def test_every_adetailer_region_is_planned_on_both_paths(self):
        source = self.source(inference_server.apply_adetailer_unit)
        self.assertEqual(source.count('f"adetailer_region_{unit_index + 1}_{index + 1}"'), 2)
        self.assertIn("stage_prompt_conditioning(", source)
        self.assertIn("before_encode=plan_detail_encode", source)

    def test_component_stages_record_one_vocabulary(self):
        card = FakeCard(card_bytes(12))
        instance, pipeline = controller(card, T0_RESIDENT)
        strategy = {"mode": "high_vram", "label": "x", "offload_mode": "none", "dynamic": True,
                    "reason": "r", "physical_forward_copies": 1}
        inference_server.reset_stage_schedule()
        stages = (
            ("prompt_encode", BASE, {"kind": "encode", "inference_bytes": stage_scheduler.TEXT_ENCODE_TRANSIENT_BYTES,
                                     "legacy_point": False}),
            ("base_sampling", BASE, {"decode_in_call": True}),
            ("hires_sampling_1", REDRAW, {}),
            ("adetailer_region_1_1", DETAIL, {"decode_in_call": True, "legacy_point": False}),
            ("adetailer_region_1_2", DETAIL, {"decode_in_call": True, "legacy_point": False}),
        )
        with patch.object(inference_server, "active_residency", instance), \
                patch.object(inference_server, "active_memory_strategy", strategy), \
                patch.object(inference_server, "loaded_pipeline", pipeline), \
                patch.object(inference_server, "loaded_engine", "iL"), \
                patch.object(inference_server, "pipeline_cpu_parked", False):
            for stage, canvas, options in stages:
                inference_server.plan_residency_for_stage(stage, "sdxl", canvas, 5.0, 1, None, **options)
            schedule = list(inference_server.active_stage_schedule)
        self.assertEqual([record["stage"] for record in schedule], [stage for stage, _canvas, _options in stages])
        keys = set(decision_record(stage="x", engine="iL", family="sdxl", path="resident", reason="r"))
        for record in schedule:
            self.assertTrue(keys <= set(record))
            self.assertEqual(record["requested_mode"], "auto")
            self.assertIn(record["path"], ("resident", "component", "block", "leaf", "submodule"))
        # Two regions of one size share one decision: the second changes nothing.
        self.assertFalse(schedule[-1]["changed"])


class OomRecoveryTests(unittest.TestCase):
    def strategy(self, tier):
        return {
            "mode": residency.TIER_MEMORY_MODES[tier], "label": "x", "reason": "r", "offload_mode": "model",
            "cfg_mode": "serial", "cfg_serial_eligible": True, "residency_tier": tier, "dynamic": True,
            "admission": {"fallback": None},
        }

    def call(self, generator, draws, failures):
        def run():
            draws.append(torch.rand(1, generator=generator).item())
            if len(draws) <= failures:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return "image"
        return run

    def escalate(self, tier, failures):
        card = FakeCard(card_bytes(8))
        instance, pipeline = controller(card, tier)
        generator = torch.Generator().manual_seed(11)
        draws = []
        with patch.object(inference_server, "active_residency", instance), \
                patch.object(inference_server, "active_memory_strategy", self.strategy(tier)), \
                patch.object(inference_server, "read_stage_budget", side_effect=card.read), \
                patch.object(inference_server, "reconfigure_memory_strategy") as reconfigure, \
                patch.object(inference_server, "residency_job_id", None):
            try:
                result = inference_server.run_with_offload_escalation(
                    pipeline, {"generator": generator}, self.call(generator, draws, failures)
                )
            finally:
                escalated = inference_server.active_memory_strategy
        return result, draws, instance, escalated, reconfigure

    def test_an_oom_retries_once_one_rung_lower_with_the_same_noise(self):
        result, draws, instance, escalated, reconfigure = self.escalate(T1_COMPONENT, failures=1)
        self.assertEqual(result, "image")
        self.assertEqual(draws[0], draws[1])
        self.assertEqual(instance.tier, T3_LEAF_STREAM)
        self.assertEqual(escalated["label"], f"{AUTO_LABEL} · 叶级流式")
        reconfigure.assert_not_called()

    def test_auto_never_falls_through_to_the_survival_mode(self):
        with self.assertRaises(StageMemoryError):
            self.escalate(T4_SUBMODULE, failures=1)

    def test_a_second_failure_is_a_stage_error_not_another_step_down(self):
        with self.assertRaises(StageMemoryError) as raised:
            self.escalate(T1_COMPONENT, failures=2)
        self.assertIn("降级一次重试后仍显存不足", str(raised.exception))


class ReportingTests(unittest.TestCase):
    def test_auto_reports_itself_and_the_path_never_a_manual_tier(self):
        strategy = {"mode": "high_vram", "label": f"{AUTO_LABEL} · 常驻", "reason": "r", "offload_mode": "none",
                    "model_resident": True, "dynamic": True, "memory_path": "resident", "residency_tier": "T0"}
        with patch.object(inference_server, "pipeline_cpu_parked", False):
            fields = inference_server.memory_job_fields(strategy)
        self.assertEqual(fields["memory_mode"], "auto")
        self.assertEqual(fields["memory_label"], f"{AUTO_LABEL} · 常驻")
        self.assertEqual(fields["memory_path"], "resident")
        self.assertEqual(fields["memory_path_label"], "常驻")
        self.assertNotIn("HIGH_VRAM", fields["memory_label"])

    def test_a_manual_tier_reports_exactly_as_before(self):
        strategy = {"mode": "normal_vram", "label": "NORMAL_VRAM 智能调度", "reason": "r",
                    "offload_mode": "model", "model_resident": False, "dynamic": False}
        with patch.object(inference_server, "pipeline_cpu_parked", False):
            fields = inference_server.memory_job_fields(strategy)
        self.assertEqual(fields["memory_mode"], "normal_vram")
        self.assertNotIn("memory_path", fields)
        self.assertNotIn("memory_schedule", fields)

    def test_only_the_literal_auto_setting_is_dynamic(self):
        for mode, dynamic in (("auto", True), ("sdxl_balanced", False), ("high_vram", False),
                              ("normal_vram", False), ("low_vram", False), ("ultra_low_vram", False)):
            with self.subTest(mode=mode), patch.dict(inference_server.performance_settings, {"memory_mode": mode}):
                self.assertEqual(inference_server.auto_dynamic_requested(), dynamic)

    def test_the_png_note_carries_the_path_and_nothing_machine_specific(self):
        records = [
            decision_record(stage="base_sampling", engine="iL", family="sdxl", path="resident", tier="T0",
                            reason="stage_plan_unchanged", required_bytes=9 * GIB),
            decision_record(stage="final_decode", engine="iL", family="sdxl", path="evicted", reason="x"),
        ]
        note = png_schedule_note(records)
        self.assertEqual(note["requested_mode"], "auto")
        self.assertEqual(note["stages"], [
            {"stage": "base_sampling", "path": "resident", "tier": "T0"},
            {"stage": "final_decode", "path": "evicted", "tier": None},
        ])
        self.assertNotIn("bytes", json.dumps(note))

    def test_below_six_gigabytes_warns_and_stays_selectable(self):
        self.assertTrue(below_recommended_vram(4 * GIB))
        self.assertTrue(below_recommended_vram(int(5.4 * GIB)))
        self.assertFalse(below_recommended_vram(6_442_450_944))   # a card sold as 6 GB
        self.assertFalse(below_recommended_vram(LOG_CARD))
        self.assertEqual(BELOW_RECOMMENDED_WARNING, "低于推荐配置，可能无法完成任务")
        self.assertEqual(inference_server.PerformanceInput(memory_mode="auto").memory_mode, "auto")

    def test_the_8g_manual_settings_file_is_kept_item_by_item(self):
        manual = {
            "memory_mode": "sdxl_balanced", "attention_backend": "auto", "compute_dtype": "fp16",
            "vae_mode": "tiled", "cuda_math": "balanced", "keep_model_cached": True,
            "allow_shared_memory": True, "calculate_model_hash": False, "staged_vae_decode": True,
            "compile_transformer": True, "unload_before_decode": True, "vram_limit_gb": 0.0,
        }
        self.assertEqual(normalize_performance_settings(dict(manual), vram_bytes=8585216000), manual)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "performance.json"
            path.write_text(json.dumps(manual), encoding="utf-8")
            self.assertEqual(inference_server.read_performance_settings(path, 8585216000), manual)


if __name__ == "__main__":
    unittest.main()
