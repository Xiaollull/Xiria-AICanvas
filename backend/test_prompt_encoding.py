import unittest

import torch

from backend.prompt_encoding import (
    build_weighted_token_batches,
    encode_weighted_prompt,
    parse_prompt_weights,
    prepare_prompt_conditioning,
    prompt_diagnostics,
    tokenize_weighted_prompt,
)


class FakeTokenizer:
    model_max_length = 6
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"

    def __call__(self, text, add_special_tokens=False, truncation=False):
        del add_special_tokens, truncation
        return {"input_ids": [ord(character) for character in text]}

    def build_inputs_with_special_tokens(self, token_ids):
        return [self.bos_token_id, *token_ids, self.eos_token_id]

    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):
        self.assert_not_special = already_has_special_tokens
        if already_has_special_tokens:
            return [int(token in {self.bos_token_id, self.eos_token_id}) for token in token_ids]
        return [1, *([0] * len(token_ids)), 1]


class FakeTextEncoder:
    class Config:
        use_attention_mask = False

    config = Config()

    def __call__(self, input_ids, **kwargs):
        del kwargs
        values = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        return (values,)


class FakePipeline:
    tokenizer = FakeTokenizer()
    text_encoder = FakeTextEncoder()
    _execution_device = "cpu"


class FakeXLResult:
    def __init__(self, input_ids, projected):
        hidden = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        self.hidden_states = (hidden, hidden + 1000)
        self.values = [(input_ids[:, :1].to(dtype=torch.float32) if projected else hidden)]

    def __getitem__(self, index):
        return self.values[index]


class FakeXLTextEncoder(FakeTextEncoder):
    def __init__(self, projected):
        self.projected = projected

    def __call__(self, input_ids, output_hidden_states=False, **kwargs):
        del output_hidden_states, kwargs
        return FakeXLResult(input_ids, self.projected)


class FakeXLPipeline:
    tokenizer = FakeTokenizer()
    tokenizer_2 = FakeTokenizer()
    text_encoder = FakeXLTextEncoder(projected=False)
    text_encoder_2 = FakeXLTextEncoder(projected=True)
    _execution_device = "cpu"


class PromptEncodingTests(unittest.TestCase):
    def test_nested_parentheses_multiply_weight(self):
        parsed = parse_prompt_weights("a ((cat))")
        self.assertEqual(parsed[0], ("a ", 1.0))
        self.assertEqual(parsed[1][0], "cat")
        self.assertAlmostEqual(parsed[1][1], 1.21)

    def test_explicit_weight_overrides_nesting(self):
        self.assertEqual(parse_prompt_weights("((cat:1.4))"), [("cat", 1.4)])

    def test_finite_signed_zero_and_scientific_weights(self):
        self.assertEqual(parse_prompt_weights("(zero:0) (negative:-2) (science:1e-2)"), [
            ("zero", 0.0), (" ", 1.0), ("negative", -2.0), (" ", 1.0), ("science", 0.01)
        ])

    def test_invalid_suffix_stays_literal_while_nonfinite_weights_fail(self):
        self.assertEqual(parse_prompt_weights("(cat:loud)"), [("cat:loud", 1.1)])
        self.assertEqual(parse_prompt_weights("(unclosed"), [("(unclosed", 1.0)])
        self.assertEqual(parse_prompt_weights("extra)"), [("extra)", 1.0)])
        for value in ("nan", "inf", "-inf", "Infinity"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                parse_prompt_weights(f"(cat:{value})")

    def test_pathological_nesting_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "nesting"):
            parse_prompt_weights("(" * 65 + "cat" + ")" * 65)

    def test_escaped_parentheses_stay_literal(self):
        self.assertEqual(parse_prompt_weights(r"a \(cat\)"), [("a (cat)", 1.0)])

    def test_non_parenthesis_grammars_remain_literal(self):
        self.assertEqual(parse_prompt_weights("[cat] BREAK <lora:name:1>"), [("[cat] BREAK <lora:name:1>", 1.0)])

    def test_fixed_length_weighted_tokenization_aligns_specials_padding_and_truncation(self):
        encoded = tokenize_weighted_prompt(FakeTokenizer(), "a(b:2)c", max_length=8)
        self.assertEqual(encoded["input_ids"].tolist(), [1, 97, 98, 99, 2, 0, 0, 0])
        self.assertEqual(encoded["attention_mask"].tolist(), [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(encoded["weights"].tolist(), [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(encoded["token_count"], 5)
        self.assertEqual(encoded["weighted_token_count"], 1)
        self.assertTrue(encoded["input_ids"].shape[0] == 8)

        truncated = tokenize_weighted_prompt(FakeTokenizer(), "(abcdef:2)", max_length=4)
        self.assertEqual(truncated["input_ids"].tolist(), [1, 97, 98, 2])
        self.assertEqual(truncated["weights"].tolist(), [1.0, 2.0, 2.0, 1.0])

    def test_long_prompt_uses_multiple_clip_blocks(self):
        batches = build_weighted_token_batches(FakeTokenizer(), "abcdefghi")
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0][0], [1, 97, 98, 99, 100, 2])
        self.assertEqual(batches[-1][0], [1, 105, 2, 0, 0, 0])
        self.assertEqual(prompt_diagnostics(FakeTokenizer(), "abcdefghi"), {"tokens": 9, "blocks": 3, "weighted_tokens": 0})

    def test_weights_blend_against_empty_conditioning(self):
        embeddings, _, _ = encode_weighted_prompt(FakeTokenizer(), FakeTextEncoder(), "(a:2)", "cpu")
        self.assertEqual(embeddings.shape, (1, 6, 1))
        self.assertEqual(embeddings[0, 1, 0].item(), 192.0)

    def test_conditioning_pads_the_shorter_prompt_without_truncating_the_longer_one(self):
        conditioning = prepare_prompt_conditioning(FakePipeline(), "sd", "abcdefghi", "x")
        self.assertEqual(conditioning["prompt_embeds"].shape, (1, 18, 1))
        self.assertEqual(conditioning["negative_prompt_embeds"].shape, (1, 18, 1))

    def test_sdxl_conditioning_uses_penultimate_embeddings_and_pooled_clip_g(self):
        conditioning = prepare_prompt_conditioning(FakeXLPipeline(), "sdxl", "abcdefghi", "x")
        self.assertEqual(conditioning["prompt_embeds"].shape, (1, 18, 2))
        self.assertEqual(conditioning["negative_prompt_embeds"].shape, (1, 18, 2))
        self.assertEqual(conditioning["pooled_prompt_embeds"].shape, (1, 1))
        self.assertEqual(conditioning["negative_pooled_prompt_embeds"].shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
