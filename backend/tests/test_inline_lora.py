import json
import struct
import tempfile
import unittest
from pathlib import Path

from backend.inline_lora import (
    InlineLoraError,
    discover_candidates,
    parse_inline_tags,
    resolve_inline_loras,
)


def write_header(path: Path, output_name=None):
    metadata = {} if output_name is None else {"ss_output_name": output_name}
    payload = json.dumps({"__metadata__": metadata}, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


class InlineLoraTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        write_header(self.root / "styles" / "Alpha.safetensors", "metadata-alpha")
        write_header(self.root / "other" / "Beta.safetensors", "metadata-beta")
        self.candidates = discover_candidates(self.root, [], "SD", {".safetensors"})

    def tearDown(self):
        self.temp.cleanup()

    def test_parse_default_explicit_repeat_order_and_strip(self):
        result = resolve_inline_loras(
            "x <lora:styles/alpha:0.5> y <lora:Alpha>",
            "bad <lora:Beta:-2>",
            [],
            self.candidates,
        )
        self.assertNotIn("<lora:", result.prompt)
        self.assertNotIn("<lora:", result.negative_prompt)
        self.assertEqual([item.weight for item in result.matches], [0.5, 1.0, -2.0])
        self.assertEqual([item.source for item in result.matches], ["positive", "positive", "negative"])
        self.assertEqual([item.order for item in result.matches], [0, 1, 2])
        self.assertEqual(result.matches[0].match, "exact")
        self.assertEqual(result.matches[1].match, "stem")

    def test_metadata_alias_is_bounded_header_only(self):
        result = resolve_inline_loras("<lora:metadata-alpha>", "", [], self.candidates)
        self.assertEqual(result.matches[0].reference, "styles/Alpha.safetensors")
        self.assertEqual(result.matches[0].match, "ss_output_name")

    def test_manual_conflict_rejected(self):
        manual = [self.root / "styles" / "Alpha.safetensors"]
        with self.assertRaisesRegex(InlineLoraError, "conflicts"):
            resolve_inline_loras("<lora:Alpha>", "", manual, self.candidates)

    def test_ambiguous_stem_and_alias_rejected(self):
        duplicate = self.root / "duplicate" / "Alpha.safetensors"
        write_header(duplicate, "metadata-beta")
        # Different bytes from both Alpha and Beta, so neither match can collapse.
        duplicate.write_bytes(duplicate.read_bytes() + b"weights")
        candidates = discover_candidates(self.root, [], "SD", {".safetensors"})
        with self.assertRaisesRegex(InlineLoraError, "ambiguous"):
            resolve_inline_loras("<lora:Alpha>", "", [], candidates)
        with self.assertRaisesRegex(InlineLoraError, "ambiguous"):
            resolve_inline_loras("<lora:metadata-beta>", "", [], candidates)

    def test_byte_identical_copies_resolve_as_one_lora(self):
        # Style packs copied into two folders: same bytes under one stem and alias.
        (self.root / "packV7").mkdir()
        (self.root / "packV7" / "Alpha.safetensors").write_bytes((self.root / "styles" / "Alpha.safetensors").read_bytes())
        candidates = discover_candidates(self.root, [], "SD", {".safetensors"})
        for tag, match in (("<lora:Alpha:0.4>", "stem"), ("<lora:metadata-alpha>", "ss_output_name")):
            with self.subTest(tag=tag):
                result = resolve_inline_loras(tag, "", [], candidates)
                self.assertEqual(result.matches[0].reference, "packV7/Alpha.safetensors")
                self.assertEqual(result.matches[0].match, match)
                self.assertEqual(result.matches[0].audit()["identical_copies"], 1)
        # A manual mount of either copy is the same adapter mounted twice.
        with self.assertRaisesRegex(InlineLoraError, "conflicts"):
            resolve_inline_loras("<lora:Alpha>", "", [self.root / "styles" / "Alpha.safetensors"], candidates)

    def test_path_nonfinite_range_notfound_malformed_and_limit(self):
        bad = [
            "<lora:../Alpha>",
            "<lora:Alpha:nan>",
            "<lora:Alpha:6>",
            "<lora:missing>",
            "<lora:Alpha",
        ]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(InlineLoraError):
                resolve_inline_loras(text, "", [], self.candidates)
        with self.assertRaisesRegex(InlineLoraError, "at most"):
            resolve_inline_loras(" ".join(["<lora:Alpha>"] * 17), "", [], self.candidates)

    def test_square_brackets_in_filename_are_removed_before_attention(self):
        write_header(self.root / "[style].safetensors")
        candidates = discover_candidates(self.root, [], "SD", {".safetensors"})
        stripped, tags = parse_inline_tags("portrait <lora:[style]>", "positive")
        self.assertEqual(stripped, "portrait ")
        self.assertEqual(tags[0][0], "[style]")
        result = resolve_inline_loras("portrait <lora:[style]>", "", [], candidates)
        self.assertEqual(result.matches[0].reference, "[style].safetensors")

    def test_tag_separators_stay_in_the_conditioning_prompt(self):
        # Forge removes the tag span only; the reference PNG's seven tags leave
        # seven commas that CLIP still reads.
        prompt = "<lora:Alpha:1>,<lora:Beta:.6>,\nsubject"
        result = resolve_inline_loras(prompt, "", [], self.candidates)
        self.assertEqual(result.prompt, ",,\nsubject")
        self.assertEqual(parse_inline_tags("x, <lora:Alpha:1>, y", "positive")[0], "x, , y")

    def test_forge_argument_forms(self):
        _stripped, tags = parse_inline_tags("<lora:Alpha:0.5:0.5> <lora:Beta:te=0.3>", "positive")
        self.assertEqual([(name, weight) for name, weight, _ in tags], [("Alpha", 0.5), ("Beta", 0.3)])
        for text in ("<lora:Alpha:0.5:0.8>", "<lora:Alpha:1:1:4>", "<lora:Alpha:dyn=4>", "<lora:Alpha:x>"):
            with self.subTest(text=text), self.assertRaises(InlineLoraError):
                parse_inline_tags(text, "positive")

    def test_other_extra_network_tags_are_refused(self):
        for text in ("<hypernet:foo:1>", "<LORA:Alpha:1>", "<lyco:Alpha:1>"):
            with self.subTest(text=text), self.assertRaisesRegex(InlineLoraError, "unsupported"):
                parse_inline_tags(text, "positive")


if __name__ == "__main__":
    unittest.main()
