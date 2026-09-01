"""Every ggml block layout, checked against an independent transcription of the same reference.

The dequantisers in ``gguf_loader`` are vectorised: one expression handles a whole tensor, and the
indexing that makes that possible is exactly where a transcription slip hides.  So the references
below are deliberately written the other way — a scalar loop per block, in the order
``ggml-quants.c`` writes its output — and the two are compared value for value.  A reshape in the
wrong order fails here rather than producing a model that loads and paints noise.
"""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anima_pipeline
import gguf_loader
from gguf_loader import (
    GGUF_SUFFIX,
    SUPPORTED_GGML_TYPE_NAMES,
    gguf_quantization_summary,
    is_gguf_path,
    load_gguf_state_dict,
    read_gguf_header,
)

# The same list `scripts/gguf-header.mjs` publishes. The picker offers a file only when the runtime
# can read it, so the two sides have to agree; pinning the literal here is what makes a change on
# one side fail until it is made on the other.
EXPECTED_SUPPORTED_TYPES = (
    "F32", "F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
    "IQ4_NL", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "BF16",
)

TYPE_IDENTIFIERS = {entry.name: entry.identifier for entry in gguf_loader._GGML_TYPES.values()}


# --------------------------------------------------------------------------------------------
# A minimal GGUF writer, so the tests read files rather than hand-built dictionaries.
# --------------------------------------------------------------------------------------------


class RawValue:
    """A metadata value written with an explicit GGUF type id, for the types Python cannot infer."""

    def __init__(self, type_identifier, payload):
        self.type_identifier = type_identifier
        self.payload = payload


def _pack_string(text):
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _value_type(value):
    if isinstance(value, bool):
        return 7
    if isinstance(value, int):
        return 5
    if isinstance(value, float):
        return 6
    if isinstance(value, str):
        return 8
    raise TypeError(f"unsupported metadata value {value!r}")


def _value_payload(value):
    kind = _value_type(value)
    if kind == 7:
        return struct.pack("<B", 1 if value else 0)
    if kind == 5:
        return struct.pack("<i", value)
    if kind == 6:
        return struct.pack("<f", value)
    return _pack_string(value)


def _encode_value(value):
    if isinstance(value, RawValue):
        return struct.pack("<I", value.type_identifier) + value.payload
    if isinstance(value, (list, tuple)):
        element = _value_type(value[0])
        body = b"".join(_value_payload(item) for item in value)
        return struct.pack("<I", 9) + struct.pack("<I", element) + struct.pack("<Q", len(value)) + body
    return struct.pack("<I", _value_type(value)) + _value_payload(value)


def write_gguf(path, tensors, metadata=(), alignment=32, version=3, magic=b"GGUF", trim=0, damage=None):
    """Write a GGUF file.

    ``tensors`` is a sequence of ``(name, type_name, torch_shape, payload)``.  The shape is stated
    in torch order and reversed on the way out, which is the convention every real writer follows
    and the one the loader has to undo.  ``damage`` writes a deliberately wrong ``type``,
    ``offset`` or ``dimensions`` field for one tensor, which is how the malformed-file tests
    produce a broken record without patching bytes by hand.
    """
    damage = damage or {}
    header = bytearray()
    header += magic
    header += struct.pack("<I", version)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(metadata))
    for key, value in metadata:
        header += _pack_string(key)
        header += _encode_value(value)

    offset = 0
    payloads = []
    for name, type_name, shape, payload in tensors:
        broken = damage.get(name, {})
        header += _pack_string(name)
        header += struct.pack("<I", broken.get("dimensions", len(shape)))
        for dimension in reversed(shape):
            header += struct.pack("<Q", dimension)
        header += struct.pack("<I", broken.get("type", TYPE_IDENTIFIERS[type_name]))
        header += struct.pack("<Q", broken.get("offset", offset))
        payloads.append((offset, payload))
        offset += len(payload)
        offset = (offset + alignment - 1) // alignment * alignment

    data_start = (len(header) + alignment - 1) // alignment * alignment
    data = bytearray()
    for start, payload in payloads:
        data.extend(b"\0" * (start - len(data)))
        data.extend(payload)
    written = bytes(header) + b"\0" * (data_start - len(header)) + bytes(data)
    Path(path).write_bytes(written[: len(written) - trim] if trim else written)
    return path


def plain_tensor(name, tensor, type_name="F32"):
    dtype = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I32": torch.int32}[type_name]
    values = tensor.to(dtype).contiguous()
    return (name, type_name, tuple(values.shape), values.view(torch.uint8).reshape(-1).numpy().tobytes())


def quantize_q8_0(tensor):
    """The one quantiser the tests need, written the way ggml writes it: per 32 values, d = amax/127."""
    flat = tensor.reshape(-1).to(torch.float32)
    blocks = flat.reshape(-1, 32)
    payload = bytearray()
    for block in blocks:
        amax = float(block.abs().max())
        scale = amax / 127.0 if amax else 0.0
        half = torch.tensor([scale], dtype=torch.float16)
        payload += half.view(torch.uint8).numpy().tobytes()
        quantized = torch.zeros(32, dtype=torch.int8) if not scale else torch.round(block / scale).clamp(-127, 127).to(torch.int8)
        payload += quantized.numpy().tobytes()
    return bytes(payload)


def write_checkpoint_gguf(path, state, quantized="Q8_0"):
    """Write a state dict the way a real converter does.

    ggml quantises along the row, so a converter can only quantise a tensor whose last axis is a
    whole number of blocks and leaves everything else — norms, biases, the narrow projections — at
    F32. Reproducing that split is what makes this a checkpoint the loader will actually meet.
    """
    tensors = []
    for name, value in state.items():
        if quantized and value.dim() >= 2 and value.shape[-1] % 32 == 0:
            tensors.append((name, quantized, tuple(value.shape), quantize_q8_0(value)))
        else:
            tensors.append(plain_tensor(name, value))
    return write_gguf(path, tensors, metadata=[("general.architecture", "flux")])


def random_blocks(type_name, count, generator):
    """Random bytes, except that every float16 scale field is made a real, finite number."""
    entry = gguf_loader._GGML_TYPES_BY_NAME[type_name]
    blocks = torch.randint(0, 256, (count, entry.block_bytes), generator=generator, dtype=torch.uint8)
    for offset in HALF_SCALE_FIELDS.get(type_name, ()):
        halves = (torch.rand(count, generator=generator) * 6.0 - 3.0).to(torch.float16)
        blocks[:, offset : offset + 2] = halves.view(torch.uint8).reshape(count, 2)
    for offset in SINGLE_SCALE_FIELDS.get(type_name, ()):
        singles = torch.rand(count, generator=generator) * 6.0 - 3.0
        blocks[:, offset : offset + 4] = singles.view(torch.uint8).reshape(count, 4)
    return blocks


HALF_SCALE_FIELDS = {
    "Q4_0": (0,), "Q4_1": (0, 2), "Q5_0": (0,), "Q5_1": (0, 2), "Q8_0": (0,),
    "Q2_K": (80, 82), "Q3_K": (108,), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,),
    "IQ4_NL": (0,), "IQ4_XS": (0,),
}
SINGLE_SCALE_FIELDS = {"Q8_K": (0,)}


# --------------------------------------------------------------------------------------------
# Scalar references, one loop per block, in ggml's own output order.
# --------------------------------------------------------------------------------------------


def _half(block, offset):
    return struct.unpack_from("<e", block, offset)[0]


def _signed(value):
    return value - 256 if value > 127 else value


def _reference_q4_0(block):
    scale = _half(block, 0)
    quantized = block[2:18]
    return [((quantized[j] & 0x0F) - 8) * scale for j in range(16)] + [((quantized[j] >> 4) - 8) * scale for j in range(16)]


def _reference_q4_1(block):
    scale, minimum = _half(block, 0), _half(block, 2)
    quantized = block[4:20]
    return [(quantized[j] & 0x0F) * scale + minimum for j in range(16)] + [(quantized[j] >> 4) * scale + minimum for j in range(16)]


def _reference_q5_0(block):
    scale = _half(block, 0)
    high = struct.unpack_from("<I", block, 2)[0]
    quantized = block[6:22]
    low = [((quantized[j] & 0x0F) | (((high >> j) << 4) & 0x10)) - 16 for j in range(16)]
    top = [((quantized[j] >> 4) | ((high >> (j + 12)) & 0x10)) - 16 for j in range(16)]
    return [value * scale for value in low + top]


def _reference_q5_1(block):
    scale, minimum = _half(block, 0), _half(block, 2)
    high = struct.unpack_from("<I", block, 4)[0]
    quantized = block[8:24]
    low = [(quantized[j] & 0x0F) | (((high >> j) << 4) & 0x10) for j in range(16)]
    top = [(quantized[j] >> 4) | ((high >> (j + 12)) & 0x10) for j in range(16)]
    return [value * scale + minimum for value in low + top]


def _reference_q8_0(block):
    scale = _half(block, 0)
    return [_signed(block[2 + j]) * scale for j in range(32)]


def _reference_q2_k(block):
    scale, minimum = _half(block, 80), _half(block, 82)
    scales, quantized = block[0:16], block[16:80]
    values = []
    index = 0
    for group in range(2):
        base = group * 32
        for shift in range(0, 8, 2):
            for half in range(2):
                packed = scales[index]
                index += 1
                step, offset = scale * (packed & 0x0F), minimum * (packed >> 4)
                for element in range(16):
                    values.append(step * ((quantized[base + half * 16 + element] >> shift) & 3) - offset)
    return values


def _reference_q3_k(block):
    mask, quantized, packed = block[0:32], block[32:96], block[96:108]
    scale = _half(block, 108)
    scales = []
    for index in range(16):
        group, byte = index // 4, index % 4
        low = (packed[byte] if group == 0 else packed[4 + byte] if group == 1 else packed[byte] >> 4 if group == 2 else packed[4 + byte] >> 4) & 0x0F
        high = (packed[8 + byte] >> (2 * group)) & 0x03
        scales.append((low | (high << 4)) - 32)
    values = []
    bit = 0
    index = 0
    for group in range(2):
        base = group * 32
        for shift in range(0, 8, 2):
            for half in range(2):
                step = scale * scales[index]
                index += 1
                for element in range(16):
                    position = half * 16 + element
                    high = 0 if mask[position] & (1 << bit) else 4
                    values.append(step * (((quantized[base + position] >> shift) & 3) - high))
            bit += 1
    return values


def _k_scale_min(packed, index):
    if index < 4:
        return packed[index] & 63, packed[index + 4] & 63
    return (
        (packed[index + 4] & 0x0F) | ((packed[index - 4] >> 6) << 4),
        (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
    )


def _reference_q4_k(block):
    scale, minimum = _half(block, 0), _half(block, 2)
    packed, quantized = block[4:16], block[16:144]
    values = []
    for group in range(4):
        for half in range(2):
            factor, subtracted = _k_scale_min(packed, group * 2 + half)
            step, offset = scale * factor, minimum * subtracted
            for element in range(32):
                byte = quantized[group * 32 + element]
                values.append(step * ((byte & 0x0F) if half == 0 else (byte >> 4)) - offset)
    return values


def _reference_q5_k(block):
    scale, minimum = _half(block, 0), _half(block, 2)
    packed, high, quantized = block[4:16], block[16:48], block[48:176]
    values = []
    for group in range(4):
        for half in range(2):
            factor, subtracted = _k_scale_min(packed, group * 2 + half)
            step, offset = scale * factor, minimum * subtracted
            bit = group * 2 + half
            for element in range(32):
                byte = quantized[group * 32 + element]
                low = (byte & 0x0F) if half == 0 else (byte >> 4)
                values.append(step * (low + (16 if high[element] & (1 << bit) else 0)) - offset)
    return values


def _reference_q6_k(block):
    low, high, scales = block[0:128], block[128:192], block[192:208]
    scale = _half(block, 208)
    values = [0.0] * 256
    for group in range(2):
        for element in range(32):
            index = element // 16
            for which in range(4):
                byte = low[group * 64 + (which % 2) * 32 + element]
                nibble = (byte & 0x0F) if which < 2 else (byte >> 4)
                top = (high[group * 32 + element] >> (2 * which)) & 3
                quantized = (nibble | (top << 4)) - 32
                # `sc` advances eight scales per 128-value group, exactly as `ql` and `qh` advance.
                values[group * 128 + which * 32 + element] = scale * _signed(scales[group * 8 + index + 2 * which]) * quantized
    return values


def _reference_q8_k(block):
    scale = struct.unpack_from("<f", block, 0)[0]
    return [_signed(block[4 + index]) * scale for index in range(256)]


IQ4_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def _reference_iq4_nl(block):
    scale = _half(block, 0)
    quantized = block[2:18]
    return [IQ4_VALUES[quantized[j] & 0x0F] * scale for j in range(16)] + [IQ4_VALUES[quantized[j] >> 4] * scale for j in range(16)]


def _reference_iq4_xs(block):
    scale = _half(block, 0)
    high = struct.unpack_from("<H", block, 2)[0]
    low, quantized = block[4:8], block[8:136]
    values = []
    for group in range(8):
        packed = ((low[group // 2] >> (4 * (group % 2))) & 0x0F) | (((high >> (2 * group)) & 3) << 4)
        step = scale * (packed - 32)
        chunk = quantized[group * 16 : (group + 1) * 16]
        values.extend(step * IQ4_VALUES[byte & 0x0F] for byte in chunk)
        values.extend(step * IQ4_VALUES[byte >> 4] for byte in chunk)
    return values


REFERENCES = {
    "Q4_0": _reference_q4_0, "Q4_1": _reference_q4_1, "Q5_0": _reference_q5_0, "Q5_1": _reference_q5_1,
    "Q8_0": _reference_q8_0, "Q2_K": _reference_q2_k, "Q3_K": _reference_q3_k, "Q4_K": _reference_q4_k,
    "Q5_K": _reference_q5_k, "Q6_K": _reference_q6_k, "Q8_K": _reference_q8_k,
    "IQ4_NL": _reference_iq4_nl, "IQ4_XS": _reference_iq4_xs,
}


class BlockLayoutTests(unittest.TestCase):
    def test_every_supported_block_layout_matches_the_scalar_reference(self):
        generator = torch.Generator().manual_seed(20260901)
        for type_name, reference in REFERENCES.items():
            with self.subTest(type_name):
                entry = gguf_loader._GGML_TYPES_BY_NAME[type_name]
                blocks = random_blocks(type_name, 24, generator)
                produced = entry.dequantize(blocks).reshape(-1)
                expected = []
                for block in blocks:
                    expected.extend(reference(bytes(block.tolist())))
                self.assertEqual(produced.numel(), len(expected))
                self.assertTrue(
                    torch.allclose(produced, torch.tensor(expected, dtype=torch.float32), rtol=1e-5, atol=1e-6),
                    f"{type_name} differs from the reference",
                )

    def test_every_layout_named_as_supported_has_a_reference_or_is_a_plain_array(self):
        # A type must not be advertised in `SUPPORTED_GGML_TYPE_NAMES` without something pinning it.
        plain = {"F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64"}
        for name in SUPPORTED_GGML_TYPE_NAMES:
            with self.subTest(name):
                self.assertTrue(name in REFERENCES or name in plain)

    def test_the_supported_set_is_the_one_the_picker_publishes(self):
        self.assertEqual(sorted(SUPPORTED_GGML_TYPE_NAMES), sorted(EXPECTED_SUPPORTED_TYPES))


class ContainerTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def test_a_shape_is_reversed_out_of_ggml_order(self):
        weight = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        path = write_gguf(self.root / "shape.gguf", [plain_tensor("conv.weight", weight)])
        header = read_gguf_header(path)
        self.assertEqual(header.tensors[0].shape, (2, 3, 4))
        self.assertTrue(torch.equal(load_gguf_state_dict(path)["conv.weight"], weight))

    def test_plain_arrays_are_read_at_their_own_precision(self):
        values = torch.tensor([[1.5, -2.25, 0.75, 4.0]])
        path = write_gguf(self.root / "plain.gguf", [
            plain_tensor("a", values, "F32"),
            plain_tensor("b", values, "F16"),
            plain_tensor("c", values, "BF16"),
            plain_tensor("d", torch.tensor([[3, -4]]), "I32"),
        ])
        state = load_gguf_state_dict(path)
        self.assertEqual(state["a"].dtype, torch.float32)
        self.assertEqual(state["b"].dtype, torch.float16)
        self.assertEqual(state["c"].dtype, torch.bfloat16)
        self.assertEqual(state["d"].dtype, torch.int32)
        self.assertTrue(torch.equal(state["d"], torch.tensor([[3, -4]], dtype=torch.int32)))

    def test_the_requested_dtype_is_applied_to_floats_and_never_to_integers(self):
        path = write_gguf(self.root / "cast.gguf", [
            plain_tensor("weight", torch.tensor([[1.0, 2.0]]), "F32"),
            plain_tensor("count", torch.tensor([[7, 8]]), "I32"),
        ])
        state = load_gguf_state_dict(path, dtype=torch.bfloat16)
        self.assertEqual(state["weight"].dtype, torch.bfloat16)
        self.assertEqual(state["count"].dtype, torch.int32)

    def test_a_quantised_tensor_keeps_its_logical_shape(self):
        generator = torch.Generator().manual_seed(11)
        blocks = random_blocks("Q4_K", 8, generator)
        payload = bytes(blocks.reshape(-1).tolist())
        path = write_gguf(self.root / "quant.gguf", [("w", "Q4_K", (4, 512), payload)])
        header = read_gguf_header(path)
        self.assertEqual(header.tensors[0].shape, (4, 512))
        self.assertEqual(header.tensors[0].type_name, "Q4_K")
        self.assertEqual(load_gguf_state_dict(path)["w"].shape, (4, 512))

    def test_comfy_records_the_shape_ggml_could_not_hold(self):
        # ggml has four dimensions; ComfyUI's converter flattens anything wider and writes the
        # original shape into the metadata. Ignoring that would load a 5-D weight as a 4-D one.
        weight = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32)
        path = write_gguf(
            self.root / "wide.gguf",
            [("net.weight", "F32", (weight.numel(),), weight.numpy().tobytes())],
            metadata=[("comfy.gguf.orig_shape.net.weight", [2, 3, 4, 5, 6])],
        )
        self.assertEqual(read_gguf_header(path).tensors[0].shape, (2, 3, 4, 5, 6))
        self.assertEqual(load_gguf_state_dict(path)["net.weight"].shape, (2, 3, 4, 5, 6))

    def test_a_recorded_shape_that_does_not_hold_the_tensor_is_refused(self):
        weight = torch.zeros(24, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "original shape"):
            read_gguf_header(write_gguf(
                self.root / "bad-shape.gguf",
                [("net.weight", "F32", (24,), weight.numpy().tobytes())],
                metadata=[("comfy.gguf.orig_shape.net.weight", [2, 3])],
            ))

    def test_metadata_this_loader_does_not_need_is_walked_past_rather_than_kept(self):
        # A real file carries strings, arrays and scalars none of this is interested in. Getting
        # the skip length wrong would misread every tensor that follows.
        weight = torch.tensor([[2.0, 3.0]])
        path = write_gguf(self.root / "metadata.gguf", [plain_tensor("w", weight)], metadata=[
            ("general.architecture", "flux"),
            ("general.name", "a name"),
            ("general.quantization_version", 2),
            ("general.file_type", RawValue(4, struct.pack("<I", 7))),
            ("some.flag", True),
            ("some.value", 1.5),
            ("tokenizer.ggml.tokens", ["one", "two", "three"]),
            ("tokenizer.ggml.scores", [1.0, 2.0]),
            ("some.identifiers", [4, 5, 6]),
        ])
        header = read_gguf_header(path)
        self.assertEqual(header.architecture, "flux")
        self.assertTrue(torch.equal(load_gguf_state_dict(path)["w"], weight))

    def test_a_declared_alignment_moves_the_data_section(self):
        weight = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
        path = write_gguf(
            self.root / "aligned.gguf",
            [plain_tensor("w", weight)],
            metadata=[("general.alignment", RawValue(4, struct.pack("<I", 128)))],
            alignment=128,
        )
        header = read_gguf_header(path)
        self.assertEqual(header.alignment, 128)
        self.assertEqual(header.data_offset % 128, 0)
        self.assertTrue(torch.equal(load_gguf_state_dict(path)["w"], weight))

    def test_the_quantisation_summary_names_what_the_file_is_stored_as(self):
        generator = torch.Generator().manual_seed(3)
        blocks = bytes(random_blocks("Q6_K", 2, generator).reshape(-1).tolist())
        path = write_gguf(self.root / "summary.gguf", [
            ("a", "Q6_K", (1, 256), blocks[:210]),
            ("b", "Q6_K", (1, 256), blocks[210:]),
            plain_tensor("c", torch.zeros(4)),
        ])
        self.assertEqual(gguf_quantization_summary(read_gguf_header(path)), "Q6_K x 2, F32 x 1")

    def test_a_gguf_path_is_recognised_by_its_suffix(self):
        self.assertTrue(is_gguf_path("model.GGUF"))
        self.assertTrue(is_gguf_path(Path("a") / f"model{GGUF_SUFFIX}"))
        self.assertFalse(is_gguf_path("model.safetensors"))


class ComponentSlotTests(unittest.TestCase):
    """Which component slots a GGUF is allowed into, which is narrower than "the runtime reads GGUF"."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def test_a_diffusion_model_may_be_safetensors_or_gguf(self):
        for name in ("model.safetensors", "model.gguf", "model.GGUF"):
            with self.subTest(name):
                path = self.root / name
                path.write_bytes(b"weights")
                self.assertEqual(anima_pipeline._require_diffusion_weights(path, "Flux diffusion model"), path.resolve())

    def test_another_container_is_refused_by_name(self):
        path = self.root / "model.ckpt"
        path.write_bytes(b"weights")
        with self.assertRaisesRegex(ValueError, r"\.safetensors or \.gguf"):
            anima_pipeline._require_diffusion_weights(path, "Flux diffusion model")

    def test_a_missing_file_is_reported_as_missing_rather_than_as_a_bad_type(self):
        with self.assertRaises(FileNotFoundError):
            anima_pipeline._require_diffusion_weights(self.root / "absent.gguf", "Flux diffusion model")

    def test_every_other_component_stays_safetensors_only(self):
        # A GGUF text encoder is named in llama.cpp's key space, not the checkpoint's, so accepting
        # one here would mean listing a file that then fails thousands of tensors into a load.
        path = self.root / "encoder.gguf"
        path.write_bytes(b"weights")
        with self.assertRaisesRegex(ValueError, "safetensors"):
            anima_pipeline._require_safetensors(path, "Flux text encoder")

    def test_the_dispatcher_reads_a_gguf_and_leaves_safetensors_to_the_caller(self):
        weight = torch.tensor([[1.0, 2.0, 3.0]])
        path = write_gguf(self.root / "diffusion.gguf", [plain_tensor("img_in.weight", weight)])
        loaded = anima_pipeline._load_diffusion_state_dict(path, torch.bfloat16, {}, "Flux diffusion model")
        self.assertEqual(loaded["img_in.weight"].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(loaded["img_in.weight"].to(torch.float32), weight))

        sentinel = {"marker": torch.zeros(1)}
        deps = {"load_file": lambda path_value, device: sentinel}
        self.assertIs(
            anima_pipeline._load_diffusion_state_dict(self.root / "other.safetensors", torch.bfloat16, deps, "Flux"),
            sentinel,
        )


class MalformedFileTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def _tensors(self):
        return [plain_tensor("w", torch.tensor([[1.0, 2.0]]))]

    def test_a_file_that_is_not_a_gguf_is_refused_by_name(self):
        path = self.root / "not.gguf"
        path.write_bytes(b"GGUS" + b"\0" * 64)
        with self.assertRaisesRegex(ValueError, "magic"):
            read_gguf_header(path, "Flux diffusion model")

    def test_the_label_names_the_component_that_failed(self):
        path = self.root / "not.gguf"
        path.write_bytes(b"GGUS" + b"\0" * 64)
        with self.assertRaisesRegex(ValueError, "Flux2 diffusion model"):
            read_gguf_header(path, "Flux2 diffusion model")

    def test_an_unsupported_container_version_is_refused(self):
        with self.assertRaisesRegex(ValueError, "version 1"):
            read_gguf_header(write_gguf(self.root / "v1.gguf", self._tensors(), version=1))

    def test_a_ggml_type_this_loader_cannot_expand_is_refused_by_name(self):
        # IQ2_XXS needs a code book this loader does not carry. Naming it, and naming what would
        # work, is the difference between a user who can act and one who cannot.
        path = write_gguf(self.root / "iq2.gguf", [("w", "IQ2_XXS", (1, 256), b"\0" * 66)])
        header = read_gguf_header(path)
        self.assertEqual(header.tensors[0].type_name, "IQ2_XXS")
        with self.assertRaises(ValueError) as error:
            load_gguf_state_dict(path, label="Flux diffusion model")
        self.assertIn("IQ2_XXS", str(error.exception))
        self.assertIn("Q4_K", str(error.exception))
        self.assertIn("Flux diffusion model", str(error.exception))

    def test_an_unknown_ggml_type_is_refused_while_reading_the_header(self):
        path = write_gguf(self.root / "unknown.gguf", self._tensors(), damage={"w": {"type": 250}})
        with self.assertRaisesRegex(ValueError, "ggml type 250"):
            read_gguf_header(path)

    def test_a_row_that_is_not_a_whole_number_of_blocks_is_refused(self):
        with self.assertRaisesRegex(ValueError, "whole number of Q4_K blocks"):
            read_gguf_header(write_gguf(self.root / "ragged.gguf", [("w", "Q4_K", (2, 300), b"\0" * 400)]))

    def test_a_tensor_reaching_past_the_end_of_the_file_is_refused(self):
        with self.assertRaisesRegex(ValueError, "past the end"):
            read_gguf_header(write_gguf(self.root / "short.gguf", self._tensors(), trim=4))

    def test_a_truncated_header_is_refused(self):
        path = write_gguf(self.root / "cut.gguf", self._tensors())
        raw = path.read_bytes()
        path.write_bytes(raw[:20])
        with self.assertRaisesRegex(ValueError, "ends inside its header"):
            read_gguf_header(path)

    def test_two_tensors_sharing_a_name_are_refused(self):
        weight = torch.tensor([[1.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "share one name"):
            read_gguf_header(write_gguf(self.root / "twice.gguf", [plain_tensor("w", weight), plain_tensor("w", weight)]))

    def test_overlapping_tensors_are_refused(self):
        # Two tensors pointed at the same bytes is a corrupt file, not a compact one.
        payload = torch.tensor([[1.0, 2.0]]).numpy().tobytes()
        path = write_gguf(
            self.root / "overlap.gguf",
            [("a", "F32", (1, 2), payload), ("b", "F32", (1, 2), payload)],
            damage={"b": {"offset": 0}},
        )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            read_gguf_header(path)

    def test_an_unaligned_tensor_offset_is_refused(self):
        path = write_gguf(self.root / "unaligned.gguf", self._tensors(), damage={"w": {"offset": 4}})
        with self.assertRaisesRegex(ValueError, "not aligned"):
            read_gguf_header(path)

    def test_an_alignment_that_is_not_a_power_of_two_is_refused(self):
        with self.assertRaisesRegex(ValueError, "power of two"):
            read_gguf_header(write_gguf(
                self.root / "odd.gguf",
                self._tensors(),
                metadata=[("general.alignment", RawValue(4, struct.pack("<I", 48)))],
            ))

    def test_a_dimension_count_ggml_cannot_hold_is_refused(self):
        path = write_gguf(self.root / "dims.gguf", self._tensors(), damage={"w": {"dimensions": 9}})
        with self.assertRaisesRegex(ValueError, "9 dimensions"):
            read_gguf_header(path)


if __name__ == "__main__":
    unittest.main()
