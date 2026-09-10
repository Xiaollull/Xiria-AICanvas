"""A resident sampling pass that runs out of memory streams the transformer instead.

The prompt is part of the attention sequence and has no length limit, so a long enough one can
beat the admission estimate. Measured on a 24 GB card, Krea 2 at 1024x1472: sampling peaked at
2.36 GiB with a 153-token prompt and 12.59 GiB with a 13,849-token one, and the resident pass
stopped fitting somewhere between 6,929 and 10,379 tokens. Refusing the picture at that point
would put the ceiling back where it was; streaming the transformer produces it instead.

The stub stands in for a runtime because what is under test is the fallback's decisions — what it
restores, what it enables, what it records — none of which needs a GPU.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flux_pipeline import CUDA_OOM_ERRORS, sample_or_stream  # noqa: E402


class StubRuntime:
    """The residency surface FLUX.1, FLUX.2 and Krea 2 all present to the fallback."""

    def __init__(self, failures=1, offloaded=False, offload_raises=None):
        self.remaining_failures = failures
        self._offloaded = offloaded
        self.offload_raises = offload_raises
        self.keep_transformer_resident = True
        self.last_generation_metrics = {}
        self.parked = 0
        self.enabled_groups = []
        self.stages = []
        self._poisoned = False

    @property
    def transformer_group_offload_enabled(self):
        return self._offloaded

    def _run_cuda_stage(self, name, operation):
        self.stages.append(name)
        return operation()

    def _park_transformer_on_cpu(self):
        self.parked += 1

    def enable_transformer_group_offload(self, blocks_per_group=1):
        if self.offload_raises is not None:
            raise self.offload_raises
        self.enabled_groups.append(blocks_per_group)
        self._offloaded = True


class SamplingFallbackTests(unittest.TestCase):
    def generators(self, count=1, seed=7):
        return [torch.Generator(device="cpu").manual_seed(seed + index) for index in range(count)]

    def test_a_pass_that_fits_is_left_alone(self):
        runtime = StubRuntime(failures=0)
        result = sample_or_stream(runtime, self.generators(), lambda: "latents")
        self.assertEqual(result, "latents")
        self.assertEqual(runtime.enabled_groups, [])
        self.assertEqual(runtime.parked, 0)
        self.assertTrue(runtime.keep_transformer_resident)
        self.assertNotIn("sampling_fallback", runtime.last_generation_metrics)
        self.assertEqual(runtime.stages, ["sampling"])

    def test_an_out_of_memory_pass_is_retried_with_the_transformer_streamed(self):
        runtime = StubRuntime(failures=1)
        result = sample_or_stream(runtime, self.generators(), self._counting_sampler(runtime))
        self.assertEqual(result, "latents")
        self.assertEqual(runtime.parked, 1)
        self.assertEqual(runtime.enabled_groups, [1], "the retry streams one block at a time")
        self.assertFalse(runtime.keep_transformer_resident)
        self.assertEqual(runtime.stages, ["sampling", "sampling"])
        self.assertFalse(runtime._poisoned)

    def test_the_retry_draws_the_noise_the_first_attempt_drew(self):
        # Without restoring the generators the retry continues a half-consumed noise stream, and
        # the picture stops matching its seed — the failure would be silent and unreproducible.
        drawn = []
        generators = self.generators(count=2)
        runtime = StubRuntime(failures=1)

        def sample():
            drawn.append([int(torch.randint(0, 2**31, (1,), generator=g).item()) for g in generators])
            if runtime.remaining_failures > 0:
                runtime.remaining_failures -= 1
                raise torch.OutOfMemoryError("CUDA out of memory")
            return "latents"

        sample_or_stream(runtime, generators, sample)
        self.assertEqual(len(drawn), 2)
        self.assertEqual(drawn[0], drawn[1])

    def test_the_fallback_is_recorded_so_the_run_reports_the_mode_it_used(self):
        runtime = StubRuntime(failures=1)
        sample_or_stream(runtime, self.generators(), self._counting_sampler(runtime))
        self.assertEqual(
            runtime.last_generation_metrics["sampling_fallback"],
            {
                "from": "staged_transformer_resident",
                "to": "staged_transformer_group_offload",
                "reason": "cuda_oom",
                "stage": "sampling",
                "attempts": 1,
                "generator_states_restored": True,
            },
        )

    def test_a_stage_name_other_than_sampling_is_carried_into_the_record(self):
        runtime = StubRuntime(failures=1)
        sample_or_stream(runtime, self.generators(), self._counting_sampler(runtime), stage="refinement.sampling")
        self.assertEqual(runtime.stages, ["refinement.sampling", "refinement.sampling"])
        self.assertEqual(runtime.last_generation_metrics["sampling_fallback"]["stage"], "refinement.sampling")

    def test_an_already_streaming_transformer_has_nothing_left_to_give(self):
        # There is no third mode to fall back to, and retrying the identical pass would only spend
        # the time again, so the failure is reported.
        runtime = StubRuntime(failures=1, offloaded=True)
        with self.assertRaises(CUDA_OOM_ERRORS):
            sample_or_stream(runtime, self.generators(), self._counting_sampler(runtime))
        self.assertEqual(runtime.parked, 0)

    def test_a_second_failure_while_streaming_is_not_retried_again(self):
        runtime = StubRuntime(failures=2)
        with self.assertRaises(CUDA_OOM_ERRORS):
            sample_or_stream(runtime, self.generators(), self._counting_sampler(runtime))
        self.assertEqual(runtime.enabled_groups, [1])
        self.assertEqual(runtime.stages, ["sampling", "sampling"])

    def test_a_runtime_that_cannot_be_moved_is_poisoned_rather_than_reused(self):
        # If the switch itself fails, where the weights are is unknown, and handing the runtime to
        # the next job would corrupt that one instead.
        runtime = StubRuntime(failures=1, offload_raises=RuntimeError("hooks refused"))
        with self.assertRaises(RuntimeError):
            sample_or_stream(runtime, self.generators(), self._counting_sampler(runtime))
        self.assertTrue(runtime._poisoned)

    def test_a_failure_that_is_not_out_of_memory_is_not_treated_as_one(self):
        runtime = StubRuntime(failures=0)

        def sample():
            raise ValueError("a real bug")

        with self.assertRaises(ValueError):
            sample_or_stream(runtime, self.generators(), sample)
        self.assertEqual(runtime.enabled_groups, [])

    def test_both_spellings_of_the_torch_error_are_caught(self):
        # `torch.cuda.OutOfMemoryError` and `torch.OutOfMemoryError` are the same class today. If a
        # torch upgrade separates them, the fallback must not quietly stop firing.
        for error in {torch.OutOfMemoryError, torch.cuda.OutOfMemoryError}:
            with self.subTest(error=error.__name__):
                runtime = StubRuntime(failures=0)
                calls = []

                def sample():
                    calls.append(1)
                    if len(calls) == 1:
                        raise error("CUDA out of memory")
                    return "latents"

                self.assertEqual(sample_or_stream(runtime, self.generators(), sample), "latents")
                self.assertEqual(runtime.enabled_groups, [1])

    def _counting_sampler(self, runtime):
        def sample():
            if runtime.remaining_failures > 0:
                runtime.remaining_failures -= 1
                raise torch.OutOfMemoryError("CUDA out of memory")
            return "latents"

        return sample


if __name__ == "__main__":
    unittest.main()
