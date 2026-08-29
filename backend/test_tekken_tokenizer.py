import base64
import json
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tekken_tokenizer import (
    TEKKEN_BOS_ID,
    load_tekken_tokenizer,
    tekken_tokenizer_from_state_dict,
)

LLAMA_PATTERN = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)

SPECIAL_COUNT = 10
SPECIAL_TOKENS = ("<unk>", "<s>", "</s>", "[INST]", "[/INST]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]")


def tekken_document(merges=(), vocab_size=400):
    """A miniature tekken blob: every single byte, plus whatever merges a test needs."""
    entries = []
    for rank in range(256):
        entries.append({
            "rank": rank,
            "token_bytes": base64.b64encode(bytes([rank])).decode("ascii"),
            "token_str": chr(rank),
        })
    for offset, merge in enumerate(merges):
        entries.append({
            "rank": 256 + offset,
            "token_bytes": base64.b64encode(merge).decode("ascii"),
            "token_str": merge.decode("latin-1"),
        })
    return {
        "config": {
            "pattern": LLAMA_PATTERN,
            "default_vocab_size": vocab_size,
            "default_num_special_tokens": SPECIAL_COUNT,
        },
        "vocab": entries,
        "special_tokens": [
            {"rank": index, "token_str": text} for index, text in enumerate(SPECIAL_TOKENS)
        ],
    }


def tekken_blob(**kwargs):
    return json.dumps(tekken_document(**kwargs)).encode("utf-8")


class TekkenLoadingTests(unittest.TestCase):
    def test_ordinary_ranks_are_offset_past_the_special_block(self):
        tokenizer = load_tekken_tokenizer(tekken_blob())
        # Byte 0x41 has rank 65, so its id is 65 + the reserved control block.
        self.assertEqual(tokenizer.encode("A", add_bos=False), [65 + SPECIAL_COUNT])

    def test_the_vocabulary_is_cut_at_the_declared_size(self):
        merges = [b"ab", b"cd"]
        # A vocabulary size that leaves room for only the single bytes drops both merges.
        tokenizer = load_tekken_tokenizer(tekken_blob(merges=merges, vocab_size=256 + SPECIAL_COUNT))
        self.assertEqual(len(tokenizer.encode("ab", add_bos=False)), 2)
        roomy = load_tekken_tokenizer(tekken_blob(merges=merges))
        self.assertEqual(len(roomy.encode("ab", add_bos=False)), 1)

    def test_a_blob_without_its_single_bytes_is_refused(self):
        document = tekken_document()
        document["vocab"] = document["vocab"][:10]
        with self.assertRaises(ValueError) as error:
            load_tekken_tokenizer(json.dumps(document).encode("utf-8"))
        self.assertIn("single-byte", str(error.exception))

    def test_a_blob_without_a_pattern_is_refused(self):
        document = tekken_document()
        del document["config"]["pattern"]
        with self.assertRaises(ValueError):
            load_tekken_tokenizer(json.dumps(document).encode("utf-8"))

    def test_invalid_json_is_refused_with_a_sentence(self):
        with self.assertRaises(ValueError) as error:
            load_tekken_tokenizer(b"not json")
        self.assertIn("valid JSON", str(error.exception))

    def test_a_byte_tensor_from_a_checkpoint_is_accepted(self):
        blob = tekken_blob()
        state = {"tekken_model": torch.frombuffer(bytearray(blob), dtype=torch.uint8)}
        tokenizer = tekken_tokenizer_from_state_dict(state)
        self.assertEqual(tokenizer.encode("A", add_bos=False), [65 + SPECIAL_COUNT])

    def test_a_checkpoint_without_the_tokenizer_says_where_to_find_one(self):
        with self.assertRaises(ValueError) as error:
            tekken_tokenizer_from_state_dict({"model.embed_tokens.weight": torch.zeros(1)})
        self.assertIn("tekken.json", str(error.exception))


class TekkenEncodingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = load_tekken_tokenizer(tekken_blob(merges=[b"he", b"llo", b" wor", b"ld"]))

    def test_text_round_trips_through_encode_and_decode(self):
        for text in ("hello world", "a lantern in the rain", "多行\n文本", "punctuation!?"):
            with self.subTest(text=text):
                self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text, add_bos=False)), text)

    def test_a_beginning_of_sequence_token_is_prepended_on_request(self):
        with_bos = self.tokenizer.encode("hello", add_bos=True)
        without = self.tokenizer.encode("hello", add_bos=False)
        self.assertEqual(with_bos, [TEKKEN_BOS_ID, *without])

    def test_merges_are_applied_by_rank_rather_than_left_to_right(self):
        # "bc" outranks "ab", so a rank-driven merge yields [a][bc] where a greedy left-to-right
        # pass would yield [ab][c]. Getting this backwards would tokenise every prompt differently
        # from Mistral's own tokeniser while still round-tripping.
        document = tekken_document()
        document["vocab"].append({"rank": 300, "token_bytes": base64.b64encode(b"ab").decode("ascii")})
        document["vocab"].append({"rank": 260, "token_bytes": base64.b64encode(b"bc").decode("ascii")})
        tokenizer = load_tekken_tokenizer(json.dumps(document).encode("utf-8"))
        ids = tokenizer.encode("abc", add_bos=False)
        self.assertEqual(ids, [ord("a") + SPECIAL_COUNT, 260 + SPECIAL_COUNT])

    def test_special_tokens_become_single_ids(self):
        ids = self.tokenizer.encode("[INST]hello[/INST]", add_bos=False)
        self.assertEqual(ids[0], SPECIAL_TOKENS.index("[INST]"))
        self.assertEqual(ids[-1], SPECIAL_TOKENS.index("[/INST]"))

    def test_the_longest_special_token_wins(self):
        # `[/SYSTEM_PROMPT]` must not be matched as `[` plus text, which shorter-first would do.
        ids = self.tokenizer.encode("[/SYSTEM_PROMPT]", add_bos=False)
        self.assertEqual(ids, [SPECIAL_TOKENS.index("[/SYSTEM_PROMPT]")])

    def test_special_tokens_are_dropped_from_a_decode_by_default(self):
        ids = self.tokenizer.encode("[INST]hi[/INST]", add_bos=False)
        self.assertEqual(self.tokenizer.decode(ids), "hi")
        self.assertEqual(self.tokenizer.decode(ids, skip_special_tokens=False), "[INST]hi[/INST]")

    def test_a_multibyte_character_is_encoded_bytewise(self):
        ids = self.tokenizer.encode("é", add_bos=False)
        self.assertEqual(len(ids), 2)
        self.assertEqual(self.tokenizer.decode(ids), "é")


if __name__ == "__main__":
    unittest.main()
