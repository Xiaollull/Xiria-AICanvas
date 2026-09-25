import types
import unittest

import torch

from backend.forge_prompt import (
    BREAK,
    CHUNK_LENGTH,
    build_forge_token_batches,
    encode_forge_texts,
    parse_prompt_attention,
    prepare_prompt_conditioning,
)


class FakeTokenizer:
    bos_token_id = 1000
    eos_token_id = 1001
    pad_token_id = 0

    def __init__(self, expansion=1):
        self.expansion = expansion

    def __call__(self, text, **_kwargs):
        ids = []
        for character in text:
            if character == " ":
                continue
            token = 44 if character == "," else 10 + (ord(character) % 89)
            ids.extend([token] * self.expansion)
        return {"input_ids": ids}


class FakeOutput:
    def __init__(self, first, hidden_states=None):
        self.first = first
        self.hidden_states = hidden_states

    def __getitem__(self, index):
        if index == 0:
            return self.first
        raise IndexError(index)


class FakeEncoder:
    config = types.SimpleNamespace(use_attention_mask=False)

    def __init__(self, width, projected=False):
        self.width = width
        self.projected = projected

    def __call__(self, ids, output_hidden_states=False, **_kwargs):
        base = ids.float().unsqueeze(-1) + torch.arange(1, self.width + 1, device=ids.device)
        if output_hidden_states:
            pooled = base.mean(dim=1) if self.projected else base
            return FakeOutput(pooled, [base * 0.5, base, base * 2])
        return FakeOutput(base)


def fake_pipeline(sdxl=False):
    pipeline = types.SimpleNamespace(
        tokenizer=FakeTokenizer(),
        text_encoder=FakeEncoder(3),
        _execution_device=torch.device("cpu"),
    )
    if sdxl:
        pipeline.tokenizer_2 = FakeTokenizer(expansion=2)
        pipeline.text_encoder_2 = FakeEncoder(5, projected=True)
    return pipeline


def fragments(text):
    return [item if item is BREAK else (item.text, item.weight) for item in parse_prompt_attention(text)]


class ForgeParserTests(unittest.TestCase):
    def test_round_square_explicit_escape_and_unclosed(self):
        parsed = parse_prompt_attention(r"a (b [c]:2) d \(e\) (f")
        flattened = [(item.text, item.weight) for item in parsed]
        self.assertAlmostEqual(next(weight for text, weight in flattened if "b " in text), 2.0)
        self.assertAlmostEqual(next(weight for text, weight in flattened if text == "c"), 2.0 / 1.1)
        self.assertIn((" d (e) ", 1.0), flattened)
        self.assertAlmostEqual(flattened[-1][1], 1.1)

    def test_weights_follow_forge_number_reader(self):
        # Only digits and dots form an explicit weight; anything else is text.
        self.assertEqual(fragments("(a:inf)"), [("a:inf", 1.1)])
        self.assertEqual(fragments("(a:1e3)"), [("a:1e3", 1.1)])
        self.assertEqual(fragments("(a: 1.5 )"), [("a", 1.5)])
        self.assertEqual(fragments("x:1.2) y"), [("x:1.2) y", 1.0)])
        with self.assertRaisesRegex(ValueError, "finite"):
            parse_prompt_attention("(bad:" + "9" * 400 + ")")
        with self.assertRaisesRegex(ValueError, "not a number"):
            parse_prompt_attention("(bad:1.2.3)")

    def test_round_and_square_groups_close_independently(self):
        self.assertEqual(fragments("([a:1.2)"), [("a", 1.2 / 1.1)])
        self.assertEqual(fragments("a ] b ) c"), [("a ] b ) c", 1.0)])

    def test_equal_weights_join_before_tokenising(self):
        self.assertEqual(fragments("(a:1.2)(b:1.2)"), [("ab", 1.2)])

    def test_lone_backslash_is_dropped(self):
        self.assertEqual(fragments(r"a\b"), [("ab", 1.0)])

    def test_break_markers_follow_forge(self):
        self.assertEqual(fragments("a BREAK b"), [("a", 1.0), BREAK, ("b", 1.0)])
        # Inside a group the marker is weighted like text and stops being a break.
        self.assertEqual(fragments("(a BREAK b)")[1], ("BREAK", -1.1))


class ForgeChunkTests(unittest.TestCase):
    def test_break_forces_new_chunk(self):
        batches = build_forge_token_batches(FakeTokenizer(), "abc BREAK def")
        self.assertEqual(len(batches), 2)
        self.assertTrue(all(len(ids) == CHUNK_LENGTH for ids, _ in batches))

    def test_leading_and_doubled_break_emit_empty_chunks(self):
        leading = build_forge_token_batches(FakeTokenizer(), "BREAK a")
        self.assertEqual(len(leading), 2)
        self.assertEqual(leading[0][0][1], 1001)
        self.assertEqual(len(build_forge_token_batches(FakeTokenizer(), "a BREAK BREAK b")), 3)
        self.assertEqual(len(build_forge_token_batches(FakeTokenizer(), "a BREAK")), 1)

    def test_comma_fallback_moves_tail_to_next_chunk(self):
        text = "a" * 60 + "," + "b" * 30
        batches = build_forge_token_batches(FakeTokenizer(), text)
        first_content = batches[0][0][1:-1]
        # Padding follows the comma rather than splitting 14 tokens later.
        self.assertEqual(first_content[60], 44)
        self.assertEqual(first_content[61], 1001)

    def test_a_comma_arriving_at_a_full_chunk_does_not_backtrack(self):
        text = "a" * 60 + "," + "b" * 14 + ",c"
        batches = build_forge_token_batches(FakeTokenizer(), text)
        self.assertEqual(batches[0][0][1:76], [10 + ord("a") % 89] * 60 + [44] + [10 + ord("b") % 89] * 14)
        self.assertEqual(batches[1][0][1:3], [44, 10 + ord("c") % 89])


class ForgeEncodingTests(unittest.TestCase):
    def test_original_weight_normalization_oracle(self):
        prepared = prepare_prompt_conditioning(fake_pipeline(), "sd", "(ab:2)", "")
        actual = prepared.plan.positive.cross_attention[0, :CHUNK_LENGTH]
        ids, weights = build_forge_token_batches(FakeTokenizer(), "(ab:2)")[0]
        z = torch.tensor(ids).float().unsqueeze(-1) + torch.arange(1, 4)
        w = torch.tensor(weights).unsqueeze(-1)
        expected = z * w * z.mean() / (z * w).mean()
        torch.testing.assert_close(actual, expected)

    def test_texts_encoded_together_share_one_normalization_and_chunk_count(self):
        texts = ["(ab:2)", "cd" + "e" * 80]
        first, second = encode_forge_texts(fake_pipeline(), "sd", texts, torch.device("cpu"))
        self.assertEqual(first.cross_attention.shape[1], CHUNK_LENGTH * 2)
        self.assertEqual(second.cross_attention.shape[1], CHUNK_LENGTH * 2)
        rows = [build_forge_token_batches(FakeTokenizer(), text)[0] for text in texts]
        z = torch.stack([torch.tensor(ids).float().unsqueeze(-1) + torch.arange(1, 4) for ids, _ in rows])
        w = torch.stack([torch.tensor(weights).unsqueeze(-1) for _, weights in rows])
        expected = z * w * z.mean() / (z * w).mean()
        torch.testing.assert_close(first.cross_attention[0, :CHUNK_LENGTH], expected[0])
        alone = prepare_prompt_conditioning(fake_pipeline(), "sd", "(ab:2)", "").plan.positive
        self.assertFalse(torch.allclose(alone.cross_attention[0], first.cross_attention[0, :CHUNK_LENGTH]))
        # The shorter text's second chunk is Forge's empty filler chunk.
        empty = build_forge_token_batches(FakeTokenizer(), "")[0][0]
        torch.testing.assert_close(
            first.cross_attention[0, CHUNK_LENGTH:],
            torch.tensor(empty).float().unsqueeze(-1) + torch.arange(1, 4),
        )

    def test_positive_and_negative_keep_real_different_lengths(self):
        prepared = prepare_prompt_conditioning(fake_pipeline(), "sd", "x" * 160, "short")
        self.assertEqual(prepared.plan.positive.cross_attention.shape[1], CHUNK_LENGTH * 3)
        self.assertEqual(prepared.plan.negative.cross_attention.shape[1], CHUNK_LENGTH)
        self.assertEqual(prepared["prompt_embeds"].shape[1], CHUNK_LENGTH)
        self.assertEqual(prepared["negative_prompt_embeds"].shape[1], CHUNK_LENGTH)

    def test_sdxl_towers_pad_without_truncation_and_empty_negative_is_zero(self):
        prepared = prepare_prompt_conditioning(fake_pipeline(True), "sdxl", "x" * 60, "")
        positive = prepared.plan.positive
        self.assertEqual(positive.cross_attention.shape[1], CHUNK_LENGTH * 2)
        self.assertEqual(positive.diagnostics["clip_l_empty_padding"], 1)
        self.assertEqual(positive.diagnostics["clip_g_empty_padding"], 0)
        self.assertTrue(torch.count_nonzero(prepared.plan.negative.cross_attention) == 0)
        self.assertTrue(torch.count_nonzero(prepared.plan.negative.pooled) == 0)
        self.assertEqual(positive.pooled.shape, (1, 5))

    def test_empty_sdxl_positive_is_encoded_not_zeroed(self):
        prepared = prepare_prompt_conditioning(fake_pipeline(True), "sdxl", "", "")
        self.assertTrue(torch.count_nonzero(prepared.plan.positive.cross_attention) > 0)
        self.assertTrue(torch.count_nonzero(prepared.plan.positive.pooled) > 0)
        self.assertTrue(torch.count_nonzero(prepared.plan.negative.cross_attention) == 0)

    def test_comfy_oracle_remains_explicitly_available(self):
        prepared = prepare_prompt_conditioning(fake_pipeline(), "sd", "abc", "", emphasis="comfy")
        self.assertIsNone(prepared.plan)
        self.assertIn("prompt_embeds", prepared)


if __name__ == "__main__":
    unittest.main()
