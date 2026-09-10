"""A prompt is encoded once per runtime, and the encoder crosses the bus once per stage that needs it.

Encoding moves a multi-billion-parameter text encoder onto the card and back. Every stage used to
pay for its own round trip, and every stage re-encoded prompts an earlier stage had already
encoded: a CFG generation moved the encoder twice, Hires moved it twice more for the same two
strings, and ADetailer and img2img chunking moved it twice per region and per chunk because both
call `refine_batch` in a loop.

Two things fix that and are tested here. The encoding is cached on the runtime, because the
encoder a runtime holds is fixed for its life — FLUX.1, FLUX.2 and Krea 2 all fuse LoRAs into the
transformer at load and never into the text encoder — so the same string always encodes to the same
tensor. And residency is claimed for a block but materialised only when something inside it
actually encodes, so a stage whose prompts are all cached moves nothing.

The stub stands in for a runtime because what is under test is the bookkeeping — how many passes
run, how many moves happen, what is evicted — none of which needs a GPU.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flux_pipeline import (  # noqa: E402
    CONDITIONING_CACHE_BYTES,
    cached_conditioning,
    discard_conditioning_cache,
    require_text_encoder,
    text_encoder_resident,
)


class StubModule:
    """Records every move, which is the cost the cache exists to remove."""

    def __init__(self, name="text_encoder"):
        self.name = name
        self.moves = []

    def to(self, device=None, dtype=None):
        self.moves.append("cpu" if device == "cpu" else "gpu")
        return self


class StubRuntime:
    """The residency surface all three native engines present to the cache."""

    text_encoder_attributes = ("text_encoder",)

    def __init__(self):
        self.text_encoder = StubModule()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.passes = []

    def encode(self, prompt, size=8):
        """One encoder pass, recorded, returning a tensor whose size the caller chooses."""
        self.passes.append(prompt)
        return torch.zeros(size, dtype=torch.float32)

    def conditioning(self, prompt, size=8):
        return cached_conditioning(self, prompt, lambda: self.encode(prompt, size))

    @property
    def moves(self):
        return self.text_encoder.moves


class PairRuntime(StubRuntime):
    """FLUX.1 stages a CLIP beside its T5 and caches the pair they produce together."""

    text_encoder_attributes = ("text_encoder", "text_encoder_2")

    def __init__(self):
        super().__init__()
        self.text_encoder_2 = StubModule("text_encoder_2")

    def encode(self, prompt, size=8):
        self.passes.append(prompt)
        return torch.zeros(size, dtype=torch.float32), torch.zeros(4, dtype=torch.float32)


class CacheTests(unittest.TestCase):
    def test_a_repeated_prompt_is_not_encoded_again(self):
        runtime = StubRuntime()
        first = runtime.conditioning("a lantern in the rain")
        second = runtime.conditioning("a lantern in the rain")
        self.assertEqual(runtime.passes, ["a lantern in the rain"])
        # The same tensor is handed back rather than a copy: an encoding can be hundreds of
        # megabytes, and callers only ever move and expand it.
        self.assertIs(first, second)

    def test_a_repeated_prompt_does_not_move_the_encoder(self):
        runtime = StubRuntime()
        runtime.conditioning("a lantern in the rain")
        self.assertEqual(runtime.moves, ["gpu", "cpu"])
        runtime.conditioning("a lantern in the rain")
        self.assertEqual(runtime.moves, ["gpu", "cpu"], "a cache hit must not touch the bus")

    def test_a_different_prompt_is_encoded(self):
        runtime = StubRuntime()
        runtime.conditioning("a lantern")
        runtime.conditioning("a lamp")
        self.assertEqual(runtime.passes, ["a lantern", "a lamp"])

    def test_reuse_is_reported(self):
        runtime = StubRuntime()
        runtime.conditioning("a lantern")
        self.assertFalse(runtime.last_conditioning_reused)
        runtime.conditioning("a lantern")
        self.assertTrue(runtime.last_conditioning_reused)


class ResidencyTests(unittest.TestCase):
    def test_a_cfg_pair_moves_the_encoder_once(self):
        """The prompt and its negative are two passes inside one residency, not two round trips."""
        runtime = StubRuntime()
        with text_encoder_resident(runtime):
            runtime.conditioning("a lantern")
            runtime.conditioning("blurry")
        self.assertEqual(runtime.passes, ["a lantern", "blurry"])
        self.assertEqual(runtime.moves, ["gpu", "cpu"])

    def test_a_stage_that_encodes_nothing_moves_nothing(self):
        """This is the Hires case: both prompts were encoded by the base pass."""
        runtime = StubRuntime()
        with text_encoder_resident(runtime):
            runtime.conditioning("a lantern")
            runtime.conditioning("blurry")
        runtime.text_encoder.moves.clear()
        with text_encoder_resident(runtime):
            runtime.conditioning("a lantern")
            runtime.conditioning("blurry")
        self.assertEqual(runtime.passes, ["a lantern", "blurry"])
        self.assertEqual(runtime.moves, [], "a fully cached stage must not stage the encoder")

    def test_residency_nests(self):
        runtime = StubRuntime()
        with text_encoder_resident(runtime):
            with text_encoder_resident(runtime):
                runtime.conditioning("a lantern")
            self.assertEqual(runtime.moves, ["gpu"], "the inner block must not return it early")
        self.assertEqual(runtime.moves, ["gpu", "cpu"])

    def test_the_encoder_returns_to_the_cpu_when_a_pass_raises(self):
        runtime = StubRuntime()

        def explode():
            raise RuntimeError("encoder failed")

        with self.assertRaises(RuntimeError):
            cached_conditioning(runtime, "a lantern", explode)
        self.assertEqual(runtime.moves, ["gpu", "cpu"])
        self.assertNotIn("a lantern", getattr(runtime, "_conditioning_cache", {}))

    def test_both_flux_encoders_are_staged_together(self):
        runtime = PairRuntime()
        runtime.conditioning("a lantern")
        self.assertEqual(runtime.text_encoder.moves, ["gpu", "cpu"])
        self.assertEqual(runtime.text_encoder_2.moves, ["gpu", "cpu"])
        runtime.conditioning("a lantern")
        self.assertEqual(runtime.text_encoder_2.moves, ["gpu", "cpu"])

    def test_require_is_idempotent_within_a_residency(self):
        runtime = StubRuntime()
        with text_encoder_resident(runtime):
            require_text_encoder(runtime)
            require_text_encoder(runtime)
        self.assertEqual(runtime.moves, ["gpu", "cpu"])


class EvictionTests(unittest.TestCase):
    """Prompts have no length ceiling, so the bound is bytes rather than a count of entries."""

    def entries(self, runtime):
        return list(getattr(runtime, "_conditioning_cache", {}))

    # Two fifths of the budget per entry, in float32 elements: two entries fit, a third pushes the
    # total over and costs exactly one eviction rather than collapsing the cache.
    SIZE = CONDITIONING_CACHE_BYTES // 10

    def test_the_cache_is_bounded_by_bytes(self):
        runtime = StubRuntime()
        runtime.conditioning("first", self.SIZE)
        runtime.conditioning("second", self.SIZE)
        runtime.conditioning("third", self.SIZE)
        self.assertEqual(self.entries(runtime), ["second", "third"])

    def test_the_newest_entry_is_never_evicted(self):
        """A single encoding can be larger than the whole budget; the caller still needs it."""
        runtime = StubRuntime()
        oversized = (CONDITIONING_CACHE_BYTES // 4) * 2
        runtime.conditioning("small", 8)
        result = runtime.conditioning("enormous", oversized)
        self.assertEqual(self.entries(runtime), ["enormous"])
        self.assertIs(runtime.conditioning("enormous", oversized), result)
        self.assertEqual(runtime.passes, ["small", "enormous"])

    def test_eviction_is_by_least_recent_use(self):
        """A negative prompt encoded first is asked for at every later stage, so it is not the
        entry to drop."""
        runtime = StubRuntime()
        runtime.conditioning("negative", self.SIZE)
        runtime.conditioning("positive", self.SIZE)
        runtime.conditioning("negative", self.SIZE)  # a hit, which makes it the most recent
        runtime.conditioning("hires", self.SIZE)
        self.assertEqual(self.entries(runtime), ["negative", "hires"])

    def test_closing_releases_every_encoding(self):
        runtime = StubRuntime()
        runtime.conditioning("a lantern")
        discard_conditioning_cache(runtime)
        self.assertEqual(self.entries(runtime), [])
        runtime.conditioning("a lantern")
        self.assertEqual(runtime.passes, ["a lantern", "a lantern"])


if __name__ == "__main__":
    unittest.main()
