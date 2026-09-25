import types
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

from backend import dynamic_prompt
from backend.dynamic_prompt import (
    MAX_AND_TERMS,
    PromptLeaseError,
    PromptPlanLimitError,
    PromptTimeline,
    StaticConditioningLease,
    compile_conditioning_syntax,
    prepare_dynamic_prompt_conditioning,
    prompt_plan_audit,
    sampling_timeline,
    static_prompt_plan,
)
from backend.forge_prompt import (
    ConditioningPlan,
    EncodedBranch,
    PreparedConditioning,
    prepare_prompt_conditioning,
)
from backend.tests.test_forge_prompt import fake_pipeline


@dataclass
class FakeOutput:
    sample: torch.Tensor


class FakeUNet:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def forward(self, sample, timestep=None, encoder_hidden_states=None, added_cond_kwargs=None, **_kwargs):
        if self.fail:
            raise RuntimeError("synthetic OOM")
        self.calls.append({
            "batch": sample.shape[0],
            "sequence": encoder_hidden_states.shape[1],
            "pooled": None if not added_cond_kwargs else added_cond_kwargs.get("text_embeds"),
            "time_ids": None if not added_cond_kwargs else added_cond_kwargs.get("time_ids"),
        })
        marker = torch.full_like(sample, float(encoder_hidden_states.shape[1]))
        return FakeOutput(marker)


class PredictionUNet:
    """A deterministic prediction oracle driven by sample and conditioning."""

    def __init__(self, fail_on_call=None):
        self.calls = []
        self.fail_on_call = fail_on_call

    def forward(self, sample, timestep=None, encoder_hidden_states=None, added_cond_kwargs=None, **_kwargs):
        call = len(self.calls)
        if self.fail_on_call == call:
            raise RuntimeError("synthetic OOM")
        marker = encoder_hidden_states.float().mean().item()
        self.calls.append({
            "marker": marker,
            "batch": sample.shape[0],
            "pooled": None if not added_cond_kwargs else added_cond_kwargs.get("text_embeds"),
            "time_ids": None if not added_cond_kwargs else added_cond_kwargs.get("time_ids"),
        })
        shape = (sample.shape[0], *sample.shape[1:])
        return FakeOutput(sample.float().reshape(shape) + marker)


class HookedUNet(torch.nn.Module):
    """An nn.Module whose instance `forward` is a hook layer, like Diffusers group offloading.

    With `displace_on_first_call`, the first call restores the bare forward the way
    `HookRegistry.remove_hook` does when lazy prefetch removes itself.
    """

    def __init__(self, displace_on_first_call=False):
        super().__init__()
        self.calls = []
        self.displace_on_first_call = displace_on_first_call
        self.forward = self.hook_forward

    def bare_forward(self, sample, timestep=None, encoder_hidden_states=None, **_kwargs):
        self.calls.append((sample.shape[0], encoder_hidden_states.shape[1]))
        return FakeOutput(torch.full_like(sample, float(encoder_hidden_states.shape[1])))

    def hook_forward(self, *args, **kwargs):
        output = self.bare_forward(*args, **kwargs)
        if self.displace_on_first_call:
            self.displace_on_first_call = False
            self.forward = self.bare_forward
        return output


def prepared(sdxl=False):
    positive = EncodedBranch(torch.ones(1, 231, 8), torch.ones(1, 4) if sdxl else None, {})
    negative = EncodedBranch(torch.ones(1, 77, 8), torch.full((1, 4), -1.0) if sdxl else None, {})
    plan = ConditioningPlan("sdxl" if sdxl else "sd", positive, negative)
    values = {"prompt_embeds": positive.cross_attention[:, :77], "negative_prompt_embeds": negative.cross_attention}
    return PreparedConditioning(values, plan)


class DynamicPromptLeaseTests(unittest.TestCase):
    def _pipeline(self, fail=False):
        unet = FakeUNet(fail)
        return types.SimpleNamespace(unet=unet), unet

    def test_cfg_splits_true_unequal_uc_for_batch_and_inpaint_channels(self):
        pipeline, unet = self._pipeline()
        original = unet.forward
        sample = torch.zeros(4, 9, 8, 8)
        dummy = torch.zeros(4, 77, 8)
        with StaticConditioningLease(pipeline, prepared(), 7.0) as lease:
            result = unet.forward(sample, torch.tensor(1), dummy)
            self.assertIs(unet.forward, lease.wrapper)
        self.assertEqual([(call["batch"], call["sequence"]) for call in unet.calls], [(2, 77), (2, 231)])
        self.assertEqual(result.sample[:, 0, 0, 0].tolist(), [77.0, 77.0, 231.0, 231.0])
        self.assertEqual(unet.forward, original)

    def test_cfg_zero_and_one_use_positive_only(self):
        for cfg in (0.0, 1.0):
            pipeline, unet = self._pipeline()
            with StaticConditioningLease(pipeline, prepared(), cfg):
                unet.forward(torch.zeros(3, 4), torch.tensor(1), torch.zeros(3, 77, 8))
            self.assertEqual([(call["batch"], call["sequence"]) for call in unet.calls], [(3, 231)])

    def test_sdxl_pooled_and_time_ids_follow_each_branch(self):
        pipeline, unet = self._pipeline()
        added = {"text_embeds": torch.zeros(4, 4), "time_ids": torch.arange(24).reshape(4, 6)}
        with StaticConditioningLease(pipeline, prepared(True), 5.0):
            unet.forward(torch.zeros(4, 4), torch.tensor(1), torch.zeros(4, 77, 8), added_cond_kwargs=added)
        self.assertTrue(torch.all(unet.calls[0]["pooled"] == -1))
        self.assertTrue(torch.all(unet.calls[1]["pooled"] == 1))
        torch.testing.assert_close(unet.calls[0]["time_ids"], added["time_ids"][:2])
        torch.testing.assert_close(unet.calls[1]["time_ids"], added["time_ids"][2:])

    def test_pag_layouts_are_kept_legal(self):
        pipeline, unet = self._pipeline()
        with StaticConditioningLease(pipeline, prepared(), 7.0, 3.0):
            result = unet.forward(torch.zeros(6, 4), torch.tensor(1), encoder_hidden_states=torch.zeros(6, 77, 8))
        self.assertEqual([(call["batch"], call["sequence"]) for call in unet.calls], [(6, 77), (6, 231)])
        self.assertEqual(result.sample.shape[0], 6)

        pipeline, unet = self._pipeline()
        with StaticConditioningLease(pipeline, prepared(), 1.0, 3.0):
            unet.forward(torch.zeros(4, 4), torch.tensor(1), encoder_hidden_states=torch.zeros(4, 77, 8))
        self.assertEqual([(call["batch"], call["sequence"]) for call in unet.calls], [(4, 231)])

    def test_exception_restores_identity_and_rebind_is_safe(self):
        pipeline, unet = self._pipeline(fail=True)
        original = unet.forward
        with self.assertRaisesRegex(RuntimeError, "OOM"):
            with StaticConditioningLease(pipeline, prepared(), 1.0):
                unet.forward(torch.zeros(1, 4), torch.tensor(1), encoder_hidden_states=torch.zeros(1, 77, 8))
        self.assertEqual(unet.forward, original)
        self.assertFalse(hasattr(pipeline, "_xiriai_prompt_lease_token"))

    def test_invalid_layout_fails_instead_of_padding_or_fallback(self):
        pipeline, unet = self._pipeline()
        with StaticConditioningLease(pipeline, prepared(), 7.0, 3.0):
            with self.assertRaises(PromptLeaseError):
                unet.forward(torch.zeros(5, 4), torch.tensor(1), encoder_hidden_states=torch.zeros(5, 77, 8))

    def test_lease_reseats_after_a_hook_restores_an_older_forward(self):
        # Diffusers lazy prefetch removes its hook after the first forward and
        # restores the forward it wrapped; the second step must still be leased.
        unet = HookedUNet(displace_on_first_call=True)
        pipeline = types.SimpleNamespace(unet=unet)
        with StaticConditioningLease(pipeline, prepared(), 7.0) as lease:
            for step in range(3):
                unet(torch.zeros(2, 4), torch.tensor(step), torch.zeros(2, 77, 8))
            self.assertEqual(lease.call_index, 3)
        self.assertEqual(unet.calls, [(1, 77), (1, 231)] * 3)
        self.assertEqual(unet.forward, unet.bare_forward)

    def test_a_call_around_the_lease_is_refused(self):
        unet = HookedUNet()
        pipeline = types.SimpleNamespace(unet=unet)
        with StaticConditioningLease(pipeline, prepared(), 7.0):
            unet.forward = unet.bare_forward
            with self.assertRaisesRegex(PromptLeaseError, "around the Forge conditioning lease"):
                unet(torch.zeros(2, 4), torch.tensor(1), torch.zeros(2, 77, 8))
        # The sentinel leaves with the lease.
        unet(torch.zeros(2, 4), torch.tensor(1), torch.zeros(2, 77, 8))
        self.assertEqual(unet.calls[-1], (2, 77))


class DynamicPromptParserTests(unittest.TestCase):
    """Goldens here are Forge f2.0.1 outputs recorded by the P4.1 differential oracle."""

    def states(self, source, total=4, timeline=None):
        plan = static_prompt_plan(source)
        timeline = timeline or PromptTimeline(total)
        return [plan.state_for(index, timeline)[0][0] for index in range(timeline.steps)]

    def test_schedule_keeps_before_through_its_threshold_step(self):
        self.assertEqual(self.states("[a:b:2]"), ["a", "a", "a", "b"])
        self.assertEqual(self.states("[a:b:.5]"), ["a", "a", "a", "b"])
        self.assertEqual(self.states("[a:b:1.0]"), ["a", "a", "a", "a"])
        self.assertEqual(self.states("[after:2]"), ["", "", "", "after"])
        self.assertEqual(self.states("[a:b:0]"), ["b", "b", "b", "b"])

    def test_threshold_numbers_follow_forge(self):
        # A dot makes a fraction of the pass; anything else counts steps.
        self.assertEqual(self.states("[a:b:1.5]"), ["a"] * 4)
        self.assertEqual(self.states("[a:b:-1]"), ["b"] * 4)
        self.assertEqual(self.states("[a:b:-0.5]"), ["b"] * 4)
        self.assertEqual(self.states("[a:b:1e-1]"), ["b"] * 4)
        self.assertEqual(self.states("[a:b: 2 ]"), ["a", "a", "a", "b"])

    def test_second_pass_continues_the_first_pass_timeline(self):
        hires = PromptTimeline(4, int_offset=4, float_offset=1.0)
        self.assertEqual(self.states("[a:b:0.5]", timeline=hires), ["b"] * 4)
        self.assertEqual(self.states("[a:b:1.5]", timeline=hires), ["a", "a", "a", "b"])
        self.assertEqual(self.states("[a:b:6]", timeline=hires), ["a", "a", "a", "b"])

    def test_alternation_phase_matches_forge(self):
        # Invocation 0 and 1 both render breakpoint 1.
        self.assertEqual(self.states("x [a|b]"), ["x a", "x a", "x b", "x a"])
        self.assertEqual(self.states("[fe|]male"), ["female", "female", "male", "female"])

    def test_malformed_bracket_stays_literal_without_disabling_later_schedules(self):
        source = "prefix [bad:no] and [a:b:2]"
        plan = static_prompt_plan(source)
        self.assertFalse(plan.fallback)
        self.assertTrue(plan.dynamic)
        self.assertEqual(self.states(source), ["prefix [bad:no] and a"] * 3 + ["prefix [bad:no] and b"])

    def test_top_level_pipe_makes_the_whole_prompt_static(self):
        source = "a | [b:c:1]"
        plan = static_prompt_plan(source)
        self.assertTrue(plan.fallback)
        self.assertFalse(plan.dynamic)
        self.assertEqual(self.states(source), [source] * 4)

    def test_dynamic_expands_before_square_attention(self):
        self.assertEqual(self.states("[[loud|quiet]]", 3), ["[loud]", "[loud]", "[quiet]"])

    def test_static_explicit_attention_does_not_hide_later_schedule(self):
        self.assertEqual(
            self.states("[(loud:1.2)] [a:b:1]", 3),
            ["[(loud:1.2)] a", "[(loud:1.2)] a", "[(loud:1.2)] b"],
        )

    def test_and_is_case_sensitive_weighted_ordered_and_allows_zero_negative(self):
        timeline = PromptTimeline(1)
        plan = static_prompt_plan("cat:0 AND dog:-1 AND cat:2")
        self.assertEqual(plan.state_for(0, timeline), (("cat", 0.0), ("dog", -1.0), ("cat", 2.0)))
        self.assertTrue(plan.composable)
        self.assertEqual(static_prompt_plan("cat and dog").state_for(0, timeline), (("cat and dog", 1.0),))
        single = static_prompt_plan("cat:0")
        self.assertEqual(single.state_for(0, timeline), (("cat", 0.0),))
        self.assertTrue(single.composable)

    def test_and_terms_follow_forge_weight_reader(self):
        timeline = PromptTimeline(1)
        self.assertEqual(static_prompt_plan("AND cat").state_for(0, timeline), (("", 1.0), ("cat", 1.0)))
        self.assertEqual(
            static_prompt_plan("cat:nan AND dog:1e1").state_for(0, timeline),
            (("cat:nan", 1.0), ("dog:1e1", 1.0)),
        )
        self.assertEqual(
            static_prompt_plan("x [a:b:0.5]:0.8").state_for(0, PromptTimeline(2)),
            (("x a", 0.8),),
        )

    def test_term_and_state_limits_are_explicit(self):
        with self.assertRaisesRegex(PromptPlanLimitError, str(MAX_AND_TERMS)):
            static_prompt_plan(" AND ".join(f"t{index}" for index in range(MAX_AND_TERMS + 1)))
        many = "[" + "|".join(f"t{index}" for index in range(300)) + "]"
        with self.assertRaisesRegex(PromptPlanLimitError, "256"):
            compile_conditioning_syntax(many, "", (PromptTimeline(300),))
        # Lock-stepped alternations only ever render two texts.
        compile_conditioning_syntax("".join("[a|b]" for _ in range(9)), "", (PromptTimeline(300),))

    def test_negative_prompt_does_not_split_and(self):
        plan = compile_conditioning_syntax("positive", "red AND blue")
        self.assertEqual(plan.negative.state_for(0, PromptTimeline(1)), (("red AND blue", 1.0),))

    def test_attention_is_validated_after_every_dynamic_expansion(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            compile_conditioning_syntax("[ok:(bad:" + "9" * 400 + "):2]", "", (PromptTimeline(4),))

    def test_audit_is_bounded_and_contains_no_prompt_text(self):
        audit = prompt_plan_audit(
            compile_conditioning_syntax("[secret_one|secret_two] AND c:2", "bad"), PromptTimeline(4)
        )
        self.assertEqual(audit["mode"], "dynamic")
        self.assertEqual(audit["positive_terms"], 2)
        self.assertEqual(audit["positive_states"], 3)
        self.assertNotIn("secret_one", repr(audit))

    def test_sampling_timeline_counts_denoiser_calls(self):
        heun = types.SimpleNamespace(scheduler=types.SimpleNamespace(order=2))
        euler = types.SimpleNamespace(scheduler=types.SimpleNamespace(order=1))
        base = sampling_timeline(heun, 10)
        self.assertEqual(base, PromptTimeline(20))
        self.assertEqual(sampling_timeline(euler, 6, continues=base), PromptTimeline(6, 20, 1.0))


class DynamicPromptEncodingTests(unittest.TestCase):
    def test_each_term_is_one_batch_and_identical_terms_share_it(self):
        calls = []

        def encode(_pipeline, _family, texts, _device, *, negative=False, clip_skip=None):
            self.assertIsNone(clip_skip)
            calls.append((tuple(texts), negative))
            return [EncodedBranch(torch.full((1, 77, 3), float(len(calls))), None, {}) for _ in texts]

        with mock.patch.object(dynamic_prompt, "encode_forge_texts", side_effect=encode):
            prepared_plan = prepare_dynamic_prompt_conditioning(
                types.SimpleNamespace(_execution_device=torch.device("cpu")),
                "sd",
                "cat AND [cat|dog] AND cat:2",
                "",
                timeline=PromptTimeline(3),
            )
        self.assertEqual(calls, [(("cat",), False), (("cat", "dog", "cat"), False), (("",), True)])
        state = prepared_plan.plan.state_for(0)
        self.assertEqual([term.text for term in state.positive], ["cat", "cat", "cat"])
        self.assertIs(state.positive[0].branch, state.positive[2].branch)
        self.assertIsNot(state.positive[0].branch, state.positive[1].branch)
        self.assertEqual([term.weight for term in state.positive], [1.0, 1.0, 2.0])

    def test_dynamic_prompt_requires_the_pass_timeline(self):
        with self.assertRaises(PromptLeaseError):
            prepare_dynamic_prompt_conditioning(fake_pipeline(), "sd", "[a|b]", "")

    def test_non_dynamic_path_is_the_verified_static_plan(self):
        direct = prepare_prompt_conditioning(fake_pipeline(), "sd", "ordinary [attention]", "")
        result = prepare_dynamic_prompt_conditioning(fake_pipeline(), "sd", "ordinary [attention]", "")
        self.assertIsInstance(result.plan, ConditioningPlan)
        torch.testing.assert_close(result.plan.positive.cross_attention, direct.plan.positive.cross_attention)
        torch.testing.assert_close(result.plan.negative.cross_attention, direct.plan.negative.cross_attention)

    def test_schedule_states_are_normalised_and_padded_together(self):
        timeline = PromptTimeline(2)
        result = prepare_dynamic_prompt_conditioning(fake_pipeline(), "sd", "[(ab:2):" + "c" * 80 + ":1]", "", timeline=timeline)
        encodings = result.plan.positive[0].encodings
        self.assertEqual({tuple(branch.cross_attention.shape) for branch in encodings.values()}, {(1, 154, 3)})
        alone = prepare_prompt_conditioning(fake_pipeline(), "sd", "(ab:2)", "").plan.positive
        self.assertFalse(torch.allclose(encodings["(ab:2)"].cross_attention[:, :77], alone.cross_attention))

    def test_sdxl_dynamic_negative_is_zero_only_when_every_state_is_empty(self):
        timeline = PromptTimeline(2)
        empty = prepare_dynamic_prompt_conditioning(fake_pipeline(True), "sdxl", "[a|b]", "", timeline=timeline)
        self.assertTrue(torch.count_nonzero(empty.plan.negative.encodings[""].cross_attention) == 0)
        mixed = prepare_dynamic_prompt_conditioning(fake_pipeline(True), "sdxl", "[a|b]", "[:x:1]", timeline=timeline)
        self.assertTrue(torch.count_nonzero(mixed.plan.negative.encodings[""].cross_attention) > 0)


class DynamicPredictionCompositionTests(unittest.TestCase):
    def prepared(self, prompt, negative="", sdxl=False, steps=1):
        return prepare_dynamic_prompt_conditioning(
            fake_pipeline(sdxl), "sdxl" if sdxl else "sd", prompt, negative, timeline=PromptTimeline(steps)
        )

    def pipeline(self, fail_on_call=None):
        unet = PredictionUNet(fail_on_call)
        return types.SimpleNamespace(unet=unet), unet

    def branch_marker(self, prepared_plan, text):
        plan = prepared_plan.plan
        for encodings in (*(term.encodings for term in plan.positive), plan.negative.encodings):
            if text in encodings:
                return encodings[text].cross_attention.float().mean().item()
        raise KeyError(text)

    def test_and_cfg_prediction_formula_and_batch_channels(self):
        encoded = self.prepared("a:2 AND b:-0.5", "n")
        pipeline, unet = self.pipeline()
        sample = torch.zeros(4, 9, 2, 2)
        with StaticConditioningLease(pipeline, encoded, 7.0):
            result = unet.forward(sample, torch.tensor(1), torch.zeros(4, 77, 3))
        n = self.branch_marker(encoded, "n")
        a = self.branch_marker(encoded, "a")
        b = self.branch_marker(encoded, "b")
        expected = n + 2 * (a - n) - 0.5 * (b - n)
        self.assertEqual([call["batch"] for call in unet.calls], [2, 2, 2])
        torch.testing.assert_close(result.sample[:2], torch.full_like(result.sample[:2], n))
        torch.testing.assert_close(result.sample[2:], torch.full_like(result.sample[2:], expected))

    def test_and_cfg_zero_and_one_compute_u_and_c_in_prediction_space(self):
        encoded = self.prepared("a AND b", "n")
        n = self.branch_marker(encoded, "n")
        a = self.branch_marker(encoded, "a")
        b = self.branch_marker(encoded, "b")
        for cfg, expected in ((0.0, n), (1.0, n + (a - n) + (b - n))):
            with self.subTest(cfg=cfg):
                pipeline, unet = self.pipeline()
                with StaticConditioningLease(pipeline, encoded, cfg):
                    result = unet.forward(torch.zeros(2, 4), torch.tensor(1), torch.zeros(2, 77, 3))
                self.assertEqual(len(unet.calls), 3)
                torch.testing.assert_close(result.sample, torch.full_like(result.sample, expected))

    def test_schedule_counter_advances_once_for_multiple_physical_forwards(self):
        encoded = self.prepared("[a|b] AND c", "n", steps=3)
        pipeline, unet = self.pipeline()
        with StaticConditioningLease(pipeline, encoded, 7.0) as lease:
            sample = torch.zeros(2, 4)
            dummy = torch.zeros(2, 77, 3)
            for expected_index in (1, 2, 3):
                unet.forward(sample, torch.tensor(1), dummy)
                self.assertEqual(lease.call_index, expected_index)
        # U, first term, second term for each logical invocation.
        self.assertEqual(len(unet.calls), 9)
        self.assertEqual(
            [unet.calls[index]["marker"] for index in (1, 4, 7)],
            [self.branch_marker(encoded, text) for text in ("a", "a", "b")],
        )

    def test_stage_timeline_resets_per_pipeline_invocation(self):
        encoded = self.prepared("[a:b:.5] AND c", "n", steps=4)
        pipeline, unet = self.pipeline()
        with StaticConditioningLease(pipeline, encoded, 7.0) as lease:
            lease.begin_invocation(object())
            dummy = torch.zeros(2, 77, 3)
            for _ in range(4):
                unet.forward(torch.zeros(2, 4), torch.tensor(1), dummy)
            self.assertAlmostEqual(unet.calls[-2]["marker"], self.branch_marker(encoded, "b"))
            self.assertAlmostEqual(unet.calls[-5]["marker"], self.branch_marker(encoded, "a"))
            lease.begin_invocation(object())
            unet.forward(torch.zeros(2, 4), torch.tensor(1), dummy)
            self.assertAlmostEqual(unet.calls[-2]["marker"], self.branch_marker(encoded, "a"))

    def test_pag_cfg_composes_cond_and_perturbed_slots_independently(self):
        encoded = self.prepared("a:2 AND b:-1", "n")
        pipeline, unet = self.pipeline()
        # External PAG layout is [U,C,P]. Sample values make C/P predictions distinct.
        sample = torch.tensor([[1.0], [3.0], [8.0]])
        with StaticConditioningLease(pipeline, encoded, 5.0, 3.0):
            result = unet.forward(sample, torch.tensor(1), encoder_hidden_states=torch.zeros(3, 77, 3))
        n = self.branch_marker(encoded, "n")
        a = self.branch_marker(encoded, "a")
        b = self.branch_marker(encoded, "b")
        u = 1 + n
        c = u + 2 * ((3 + a) - u) - ((3 + b) - u)
        p = u + 2 * ((8 + a) - u) - ((8 + b) - u)
        torch.testing.assert_close(result.sample[:, 0], torch.tensor([u, c, p]))
        self.assertEqual([call["batch"] for call in unet.calls], [3, 3, 3])

    def test_pag_without_outer_cfg_keeps_cfg_and_pag_deltas_independent(self):
        encoded = self.prepared("a AND b", "n")
        pipeline, unet = self.pipeline()
        sample = torch.tensor([[3.0], [8.0]])
        with StaticConditioningLease(pipeline, encoded, 0.0, 2.0):
            result = unet.forward(sample, torch.tensor(1), encoder_hidden_states=torch.zeros(2, 77, 3))
        n = self.branch_marker(encoded, "n")
        # CFG target at zero is U. P* is chosen so outer PAG adds both C-P deltas.
        target = 3 + n
        pag_delta = (3 - 8) + (3 - 8)
        torch.testing.assert_close(result.sample[:, 0], torch.tensor([target, target - pag_delta]))
        self.assertEqual([call["batch"] for call in unet.calls], [2, 2, 2])

    def test_dynamic_exception_restores_identity(self):
        encoded = self.prepared("a AND b", "n")
        pipeline, unet = self.pipeline(fail_on_call=1)
        original = unet.forward
        with self.assertRaisesRegex(RuntimeError, "OOM"):
            with StaticConditioningLease(pipeline, encoded, 7.0):
                unet.forward(torch.zeros(2, 4), torch.tensor(1), torch.zeros(2, 77, 3))
        self.assertEqual(unet.forward, original)
        self.assertFalse(hasattr(pipeline, "_xiriai_prompt_lease_token"))

    def test_dynamic_oom_rebind_resets_timeline_and_restores_identity(self):
        encoded = self.prepared("[a|b] AND c", "n", steps=3)
        pipeline, unet = self.pipeline(fail_on_call=1)
        original = unet.forward
        with StaticConditioningLease(pipeline, encoded, 7.0) as lease:
            with self.assertRaisesRegex(RuntimeError, "OOM"):
                unet.forward(torch.zeros(2, 4), torch.tensor(1), torch.zeros(2, 77, 3))
            self.assertEqual(unet.forward, original)
            unet.fail_on_call = None
            lease.begin_invocation(object())
            lease.ensure_bound()
            unet.forward(torch.zeros(2, 4), torch.tensor(1), torch.zeros(2, 77, 3))
            self.assertAlmostEqual(unet.calls[-2]["marker"], self.branch_marker(encoded, "a"))
        self.assertEqual(unet.forward, original)

    def test_dynamic_lease_bypasses_and_restores_serial_cfg_wrapper(self):
        encoded = self.prepared("a AND b", "n")
        pipeline, unet = self.pipeline()
        full_batch = unet.forward

        def serial_wrapper(*_args, **_kwargs):
            raise AssertionError("serial CFG must be mutually exclusive with the prompt lease")

        unet._xiriai_full_batch_forward = full_batch
        unet._xiriai_sequential_batch_forward = serial_wrapper
        unet.forward = serial_wrapper
        with StaticConditioningLease(pipeline, encoded, 7.0):
            unet.forward(torch.zeros(2, 4), torch.tensor(1), torch.zeros(2, 77, 3))
        self.assertIs(unet.forward, serial_wrapper)
        self.assertEqual(len(unet.calls), 3)

    def test_dynamic_sdxl_pooled_and_time_ids_follow_every_term(self):
        encoded = self.prepared("a AND b", "", sdxl=True)
        pipeline, unet = self.pipeline()
        added = {"text_embeds": torch.zeros(4, 5), "time_ids": torch.arange(24).reshape(4, 6)}
        with StaticConditioningLease(pipeline, encoded, 5.0):
            unet.forward(
                torch.zeros(4, 4), torch.tensor(1), torch.zeros(4, 77, 8), added_cond_kwargs=added
            )
        self.assertEqual(len(unet.calls), 3)
        self.assertTrue(torch.count_nonzero(unet.calls[0]["pooled"]) == 0)
        self.assertTrue(torch.count_nonzero(unet.calls[1]["pooled"]) > 0)
        self.assertTrue(torch.count_nonzero(unet.calls[2]["pooled"]) > 0)
        torch.testing.assert_close(unet.calls[0]["time_ids"], added["time_ids"][:2])
        torch.testing.assert_close(unet.calls[1]["time_ids"], added["time_ids"][2:])
        torch.testing.assert_close(unet.calls[2]["time_ids"], added["time_ids"][2:])


class DynamicPromptApiTests(unittest.TestCase):
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

    def request(self, prompt="cat", negative_prompt="", **overrides):
        fields = {
            "engine": "SD",
            "checkpoint": "model.safetensors",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": 512,
            "height": 512,
            "steps": 4,
            "cfg": 7,
            "denoise": 1,
            "seed": 1,
            "sampler": "euler",
            "scheduler": "normal",
        }
        return self.server.GenerateInput(**{**fields, **overrides})

    def test_create_job_dynamic_preflight_422_creates_no_job(self):
        before = set(self.server.jobs)
        with self.assertRaises(self.server.HTTPException) as caught:
            self.server.create_job(self.request(" AND ".join(["cat"] * (MAX_AND_TERMS + 1))))
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn(str(MAX_AND_TERMS), caught.exception.detail)
        self.assertEqual(set(self.server.jobs), before)

    def test_adetailer_unit_inline_tags_are_refused_before_a_job_exists(self):
        before = set(self.server.jobs)
        request = self.request(postprocess_chain=[{
            "type": "adetailer",
            "enabled": True,
            "units": [{"detector": "face_yolov8n.pt", "prompt": "detailed face, <lora:face:1>"}],
        }])
        with self.assertRaises(self.server.HTTPException) as caught:
            self.server.create_job(request)
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("ADetailer unit 1", caught.exception.detail)
        self.assertEqual(set(self.server.jobs), before)

    def test_request_timelines_cover_every_sampling_pass(self):
        request = self.request(
            steps=10,
            sampler="heun",
            postprocess_chain=[
                {"type": "resample", "enabled": True, "scale": 2, "denoise": 0.25, "steps": 7, "sampler": "euler"},
                {"type": "resample", "enabled": True, "scale": 1.5, "denoise": 0},
                {"type": "adetailer", "enabled": True,
                 "units": [{"detector": "face_yolov8n.pt", "use_steps": True, "steps": 6}]},
            ],
        )
        base, resample, adetailer = self.server.request_prompt_timelines(request)
        self.assertEqual(base, PromptTimeline(20))
        # The redraw continues the base timeline; the pure enlargement encodes nothing.
        self.assertEqual(resample, [PromptTimeline(7, 20, 1.0)])
        self.assertEqual(adetailer, [PromptTimeline(12)])

    def test_sampler_call_counts_match_the_configured_schedulers(self):
        from diffusers import EulerDiscreteScheduler

        for sampler in sorted(self.server.DIFFUSERS_RESAMPLE_SAMPLERS):
            with self.subTest(sampler=sampler):
                pipeline = types.SimpleNamespace(scheduler=EulerDiscreteScheduler())
                self.server.configure_scheduler(pipeline, sampler, "normal")
                self.assertEqual(self.server.sampler_calls_per_step(sampler), pipeline.scheduler.order)

    def test_job_keeps_submitted_prompt_while_worker_gets_stripped_runtime(self):
        submitted = self.request("raw <lora:item:1> [a|b]", "original negative")
        runtime = submitted.model_copy(update={"prompt": "runtime [a|b]"})
        captured = []
        with mock.patch.object(
            self.server,
            "preflight_inline_lora_request",
            return_value=(runtime, [{"id": "item.safetensors", "weight": 1.0}]),
        ), mock.patch.object(self.server.executor, "submit", side_effect=lambda *args: captured.append(args)):
            response = self.server.create_job(submitted)
        self.created.append(response["id"])
        record = self.server.jobs[response["id"]]
        self.assertEqual(record["prompt"], submitted.prompt)
        self.assertEqual(record["negative_prompt"], submitted.negative_prompt)
        self.assertEqual(record["prompt_encoding"]["base"]["mode"], "dynamic")
        self.assertEqual(record["prompt_encoding"]["base"]["timeline_steps"], 4)
        self.assertEqual(captured[0][2].prompt, "runtime [a|b]")
        self.assertEqual(captured[0][3].prompt, submitted.prompt)
        self.assertNotIn(str(Path.cwd()), json.dumps(self.server.public_job_view(record)))

    def test_png_metadata_retains_submitted_prompt_and_conditioning(self):
        request = self.request("submitted [a|b]", "submitted negative")
        image = Image.new("RGB", (8, 8), "white")
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(self.server, "OUTPUT_DIRECTORY", Path(directory)):
                output = self.server.save_image(
                    image,
                    "job",
                    request,
                    1.0,
                    1.0,
                    0.0,
                    conditioning_prompt="runtime a",
                )
            with Image.open(output) as saved:
                self.assertEqual(saved.text["prompt"], request.prompt)
                self.assertEqual(saved.text["negative_prompt"], request.negative_prompt)
                parameters = json.loads(saved.text["parameters"])
            self.assertEqual(parameters["conditioning_prompt"], "runtime a")

    def test_guided_pipeline_reseats_the_lease_after_every_step(self):
        unet = HookedUNet()
        pipeline = types.SimpleNamespace(unet=unet)
        conditioning = prepared()
        seen_steps = []

        class DisplacingActive:
            def __call__(self, **kwargs):
                for step in range(2):
                    unet(torch.zeros(2, 4), torch.tensor(step), kwargs["prompt_embeds"].expand(2, -1, -1))
                    # Residency work between steps hands the UNet another forward.
                    unet.forward = unet.bare_forward
                    kwargs["callback_on_step_end"](self, step, step, {"latents": None})
                return types.SimpleNamespace(images=[])

        def on_step_end(_pipeline, step, _timestep, callback_kwargs):
            seen_steps.append(step)
            return callback_kwargs

        self.server.run_guided_pipeline(
            pipeline,
            DisplacingActive(),
            {"guidance_scale": 7.0, "callback_on_step_end": on_step_end, **conditioning.as_dict()},
            prompt_conditioning=conditioning,
        )
        self.assertEqual(seen_steps, [0, 1])
        self.assertEqual(unet.calls, [(1, 77), (1, 231)] * 2)

    def test_cancelled_pipeline_call_finally_restores_dynamic_lease(self):
        encoded = prepare_dynamic_prompt_conditioning(fake_pipeline(), "sd", "a AND b", "n")
        unet = PredictionUNet()
        pipeline = types.SimpleNamespace(unet=unet)
        original = unet.forward

        class CancelActive:
            def __call__(self, **kwargs):
                unet.forward(
                    torch.zeros(2, 4),
                    torch.tensor(1),
                    kwargs["prompt_embeds"].expand(2, -1, -1),
                )
                raise self_error

        self_error = self.server.GenerationCancelled()
        with self.assertRaises(self.server.GenerationCancelled):
            self.server.run_guided_pipeline(
                pipeline,
                CancelActive(),
                {"guidance_scale": 7.0, **encoded.as_dict()},
                prompt_conditioning=encoded,
            )
        self.assertEqual(unet.forward, original)
        self.assertFalse(hasattr(pipeline, "_xiriai_prompt_lease_token"))


if __name__ == "__main__":
    unittest.main()
