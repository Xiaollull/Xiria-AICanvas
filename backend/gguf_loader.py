"""GGUF diffusion-model files for the native engines.

ComfyUI reads GGUF through the ComfyUI-GGUF nodes, which keep the packed blocks resident and
expand them inside their own linear op.  This module takes the other half of that design: it
expands every tensor once, while loading, into the compute dtype the rest of the runtime already
uses.  That is the same decision :func:`flux_pipeline.resolve_quantized_state_dict` already makes
for ComfyUI's scaled-fp8 and mixed-precision checkpoints, and it is what keeps a GGUF file
compatible with everything downstream — LoRA fusion on the host, ``_strict_assign`` into a
Diffusers module, block-level group offload — instead of forking all of it behind a second,
quantised runtime.

What that buys, and what it does not: a GGUF file is a much smaller download and needs far less
disk than the same weights at bf16, and a quantisation the publisher shipped is the only form some
models are released in.  Once loaded it occupies exactly as much memory as bf16 would, because
that is what it has been turned into.  VRAM is still bought by the memory policy and group
offload, not by the file format — the same is true of the fp8 files this runtime already reads.

The block layouts below are ggml's.  Each one is transcribed from ``ggml-quants.c`` and pinned in
``test_gguf_loader.py`` against an independent scalar implementation of the same reference, so a
transcription slip fails a test rather than silently loading a differently-scaled model.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

GGUF_SUFFIX = ".gguf"
GGUF_MAGIC = b"GGUF"
# Version 1 spelled its counts as 32-bit and predates the alignment key. No published diffusion
# model uses it, and guessing at the difference would be a way to misread a file rather than to
# read more of them.
GGUF_SUPPORTED_VERSIONS = (2, 3)
GGUF_DEFAULT_ALIGNMENT = 32
GGML_MAX_DIMENSIONS = 4

# A diffusion GGUF's header is tens of kilobytes. These caps are what stop a malformed or hostile
# file from asking for an unbounded allocation before a single field has been validated; they are
# the same idea as `MAX_SAFETENSORS_HEADER_BYTES` on the JavaScript side.
GGUF_MAX_HEADER_BYTES = 64 * 1024 * 1024
GGUF_MAX_TENSORS = 1 << 20
GGUF_MAX_METADATA_ENTRIES = 1 << 20
GGUF_MAX_STRING_BYTES = 1 << 24
GGUF_MAX_ELEMENTS = 1 << 40

GGUF_ARCHITECTURE_KEY = "general.architecture"
GGUF_ALIGNMENT_KEY = "general.alignment"
# ggml holds at most four dimensions, so ComfyUI's converter flattens anything wider and records
# what it flattened under this key. Restoring it is the difference between a working 5-D
# convolution and a shape error much later.
COMFY_ORIGINAL_SHAPE_PREFIX = "comfy.gguf.orig_shape."

_VALUE_UINT8 = 0
_VALUE_INT8 = 1
_VALUE_UINT16 = 2
_VALUE_INT16 = 3
_VALUE_UINT32 = 4
_VALUE_INT32 = 5
_VALUE_FLOAT32 = 6
_VALUE_BOOL = 7
_VALUE_STRING = 8
_VALUE_ARRAY = 9
_VALUE_UINT64 = 10
_VALUE_INT64 = 11
_VALUE_FLOAT64 = 12

_FIXED_VALUE_SIZES = {
    _VALUE_UINT8: 1,
    _VALUE_INT8: 1,
    _VALUE_UINT16: 2,
    _VALUE_INT16: 2,
    _VALUE_UINT32: 4,
    _VALUE_INT32: 4,
    _VALUE_FLOAT32: 4,
    _VALUE_BOOL: 1,
    _VALUE_UINT64: 8,
    _VALUE_INT64: 8,
    _VALUE_FLOAT64: 8,
}

_SIGNED_VALUE_TYPES = frozenset({_VALUE_INT8, _VALUE_INT16, _VALUE_INT32, _VALUE_INT64})


def _half(blocks: torch.Tensor, start: int) -> torch.Tensor:
    """Read one little-endian float16 out of every block as float32, shaped ``[n, 1]``."""
    return blocks[:, start : start + 2].contiguous().view(torch.float16).to(torch.float32)


def _shift(values: torch.Tensor, amounts: Sequence[int], dim: int) -> torch.Tensor:
    """Broadcast a set of right shifts over ``values`` along a new axis at ``dim``."""
    shifts = torch.tensor(list(amounts), dtype=values.dtype).reshape([-1 if index == dim else 1 for index in range(values.dim() + 1)])
    return values.unsqueeze(dim) >> shifts


def _dequantize_q4_0(blocks: torch.Tensor) -> torch.Tensor:
    scale = _half(blocks, 0)
    quantized = _shift(blocks[:, 2:18], (0, 4), 1).reshape(blocks.shape[0], 32) & 0x0F
    return (quantized.to(torch.float32) - 8.0) * scale


def _dequantize_q4_1(blocks: torch.Tensor) -> torch.Tensor:
    scale = _half(blocks, 0)
    minimum = _half(blocks, 2)
    quantized = _shift(blocks[:, 4:20], (0, 4), 1).reshape(blocks.shape[0], 32) & 0x0F
    return quantized.to(torch.float32) * scale + minimum


def _five_bit_high(blocks: torch.Tensor, start: int) -> torch.Tensor:
    """Expand the 32 packed fifth bits of a Q5_0/Q5_1 block into ``[n, 32]`` values of 0 or 16."""
    packed = blocks[:, start : start + 4].contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return ((packed >> torch.arange(32, dtype=torch.int64)) & 1) << 4


def _dequantize_q5_0(blocks: torch.Tensor) -> torch.Tensor:
    scale = _half(blocks, 0)
    low = (_shift(blocks[:, 6:22], (0, 4), 1).reshape(blocks.shape[0], 32) & 0x0F).to(torch.int64)
    return ((low | _five_bit_high(blocks, 2)).to(torch.float32) - 16.0) * scale


def _dequantize_q5_1(blocks: torch.Tensor) -> torch.Tensor:
    scale = _half(blocks, 0)
    minimum = _half(blocks, 2)
    low = (_shift(blocks[:, 8:24], (0, 4), 1).reshape(blocks.shape[0], 32) & 0x0F).to(torch.int64)
    return (low | _five_bit_high(blocks, 4)).to(torch.float32) * scale + minimum


def _dequantize_q8_0(blocks: torch.Tensor) -> torch.Tensor:
    scale = _half(blocks, 0)
    return blocks[:, 2:34].contiguous().view(torch.int8).to(torch.float32) * scale


def _dequantize_q2_k(blocks: torch.Tensor) -> torch.Tensor:
    count = blocks.shape[0]
    scales = blocks[:, 0:16]
    scale = _half(blocks, 80)
    minimum = _half(blocks, 82)
    step = (scale * (scales & 0x0F).to(torch.float32)).reshape(count, 16, 1)
    offset = (minimum * (scales >> 4).to(torch.float32)).reshape(count, 16, 1)
    quantized = (_shift(blocks[:, 16:80].reshape(count, 2, 32), (0, 2, 4, 6), 2) & 0x03).reshape(count, 16, 16)
    return (step * quantized.to(torch.float32) - offset).reshape(count, 256)


def _q3_k_scales(packed: torch.Tensor) -> torch.Tensor:
    """Unpack Q3_K's sixteen 6-bit scales: four low bits in bytes 0-7, two high bits in bytes 8-11."""
    count = packed.shape[0]
    low = _shift(packed[:, 0:8], (0, 4), 1).reshape(count, 16) & 0x0F
    high = _shift(packed[:, 8:12], (0, 2, 4, 6), 1).reshape(count, 16) & 0x03
    return (low | (high << 4)).to(torch.float32) - 32.0


def _dequantize_q3_k(blocks: torch.Tensor) -> torch.Tensor:
    count = blocks.shape[0]
    step = (_half(blocks, 108) * _q3_k_scales(blocks[:, 96:108])).reshape(count, 16, 1)
    low = (_shift(blocks[:, 32:96].reshape(count, 2, 32), (0, 2, 4, 6), 2) & 0x03).reshape(count, 16, 16)
    # The high bit is an inverted offset: set means add nothing, clear means subtract four.
    high = (_shift(blocks[:, 0:32].reshape(count, 1, 32), tuple(range(8)), 2) & 0x01).reshape(count, 16, 16)
    quantized = low.to(torch.float32) - ((high ^ 0x01) << 2).to(torch.float32)
    return (step * quantized).reshape(count, 256)


def _k_scale_min(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack the eight 6-bit scale/minimum pairs Q4_K and Q5_K share, from twelve bytes."""
    count = packed.shape[0]
    scale_bytes = packed[:, 0:4]
    minimum_bytes = packed[:, 4:8]
    high_bytes = packed[:, 8:12]
    scale = torch.cat([scale_bytes & 0x3F, (high_bytes & 0x0F) | ((scale_bytes >> 2) & 0x30)], dim=1)
    minimum = torch.cat([minimum_bytes & 0x3F, (high_bytes >> 4) | ((minimum_bytes >> 2) & 0x30)], dim=1)
    return scale.reshape(count, 8).to(torch.float32), minimum.reshape(count, 8).to(torch.float32)


def _dequantize_q4_k(blocks: torch.Tensor) -> torch.Tensor:
    count = blocks.shape[0]
    scale, minimum = _k_scale_min(blocks[:, 4:16])
    step = (_half(blocks, 0) * scale).reshape(count, 8, 1)
    offset = (_half(blocks, 2) * minimum).reshape(count, 8, 1)
    quantized = (_shift(blocks[:, 16:144].reshape(count, 4, 32), (0, 4), 2) & 0x0F).reshape(count, 8, 32)
    return (step * quantized.to(torch.float32) - offset).reshape(count, 256)


def _dequantize_q5_k(blocks: torch.Tensor) -> torch.Tensor:
    count = blocks.shape[0]
    scale, minimum = _k_scale_min(blocks[:, 4:16])
    step = (_half(blocks, 0) * scale).reshape(count, 8, 1)
    offset = (_half(blocks, 2) * minimum).reshape(count, 8, 1)
    low = (_shift(blocks[:, 48:176].reshape(count, 4, 32), (0, 4), 2) & 0x0F).reshape(count, 8, 32)
    high = (_shift(blocks[:, 16:48].reshape(count, 1, 32), tuple(range(8)), 2) & 0x01).reshape(count, 8, 32)
    return (step * (low | (high << 4)).to(torch.float32) - offset).reshape(count, 256)


def _dequantize_q6_k(blocks: torch.Tensor) -> torch.Tensor:
    count = blocks.shape[0]
    scales = blocks[:, 192:208].contiguous().view(torch.int8).to(torch.float32)
    step = (_half(blocks, 208) * scales).reshape(count, 16, 1)
    low = (_shift(blocks[:, 0:128].reshape(count, 2, 64), (0, 4), 2) & 0x0F).reshape(count, 8, 32)
    high = (_shift(blocks[:, 128:192].reshape(count, 2, 32), (0, 2, 4, 6), 2) & 0x03).reshape(count, 8, 32)
    quantized = (low | (high << 4)).to(torch.float32) - 32.0
    return (step * quantized.reshape(count, 16, 16)).reshape(count, 256)


def _dequantize_q8_k(blocks: torch.Tensor) -> torch.Tensor:
    scale = blocks[:, 0:4].contiguous().view(torch.float32)
    return blocks[:, 4:260].contiguous().view(torch.int8).to(torch.float32) * scale


# The non-linear code book both IQ4 layouts index with their 4-bit codes.
_IQ4_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def _iq4_lookup(codes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(_IQ4_VALUES, dtype=torch.float32)
    return table[codes.to(torch.long)]


def _dequantize_iq4_nl(blocks: torch.Tensor) -> torch.Tensor:
    scale = _half(blocks, 0)
    codes = _shift(blocks[:, 2:18], (0, 4), 1).reshape(blocks.shape[0], 32) & 0x0F
    return _iq4_lookup(codes) * scale


def _dequantize_iq4_xs(blocks: torch.Tensor) -> torch.Tensor:
    count = blocks.shape[0]
    scale = _half(blocks, 0)
    high = blocks[:, 2:4].contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    low = (_shift(blocks[:, 4:8], (0, 4), 2).reshape(count, 8) & 0x0F).to(torch.int64)
    high = (high >> torch.arange(0, 16, 2, dtype=torch.int64)) & 0x03
    scales = (low | (high << 4)).to(torch.float32) - 32.0
    step = (scale * scales).reshape(count, 8, 1)
    codes = (_shift(blocks[:, 8:136].reshape(count, 8, 16), (0, 4), 2) & 0x0F).reshape(count, 8, 32)
    return (step * _iq4_lookup(codes)).reshape(count, 256)


@dataclass(frozen=True)
class GgmlType:
    """One ggml storage type: how many elements a block holds, and how many bytes that block is."""

    identifier: int
    name: str
    block_elements: int
    block_bytes: int
    # Exactly one of these is set for a type this loader can read. `dtype` marks a plain array that
    # is reinterpreted in place; `dequantize` marks a block layout that has to be expanded.
    dtype: torch.dtype | None = None
    dequantize: Callable[[torch.Tensor], torch.Tensor] | None = None

    @property
    def supported(self) -> bool:
        return self.dtype is not None or self.dequantize is not None


_GGML_TYPES = {
    entry.identifier: entry
    for entry in (
        GgmlType(0, "F32", 1, 4, dtype=torch.float32),
        GgmlType(1, "F16", 1, 2, dtype=torch.float16),
        GgmlType(2, "Q4_0", 32, 18, dequantize=_dequantize_q4_0),
        GgmlType(3, "Q4_1", 32, 20, dequantize=_dequantize_q4_1),
        GgmlType(6, "Q5_0", 32, 22, dequantize=_dequantize_q5_0),
        GgmlType(7, "Q5_1", 32, 24, dequantize=_dequantize_q5_1),
        GgmlType(8, "Q8_0", 32, 34, dequantize=_dequantize_q8_0),
        GgmlType(9, "Q8_1", 32, 36),
        GgmlType(10, "Q2_K", 256, 84, dequantize=_dequantize_q2_k),
        GgmlType(11, "Q3_K", 256, 110, dequantize=_dequantize_q3_k),
        GgmlType(12, "Q4_K", 256, 144, dequantize=_dequantize_q4_k),
        GgmlType(13, "Q5_K", 256, 176, dequantize=_dequantize_q5_k),
        GgmlType(14, "Q6_K", 256, 210, dequantize=_dequantize_q6_k),
        GgmlType(15, "Q8_K", 256, 292, dequantize=_dequantize_q8_k),
        GgmlType(16, "IQ2_XXS", 256, 66),
        GgmlType(17, "IQ2_XS", 256, 74),
        GgmlType(18, "IQ3_XXS", 256, 98),
        GgmlType(19, "IQ1_S", 256, 50),
        GgmlType(20, "IQ4_NL", 32, 18, dequantize=_dequantize_iq4_nl),
        GgmlType(21, "IQ3_S", 256, 110),
        GgmlType(22, "IQ2_S", 256, 82),
        GgmlType(23, "IQ4_XS", 256, 136, dequantize=_dequantize_iq4_xs),
        GgmlType(24, "I8", 1, 1, dtype=torch.int8),
        GgmlType(25, "I16", 1, 2, dtype=torch.int16),
        GgmlType(26, "I32", 1, 4, dtype=torch.int32),
        GgmlType(27, "I64", 1, 8, dtype=torch.int64),
        GgmlType(28, "F64", 1, 8, dtype=torch.float64),
        GgmlType(29, "IQ1_M", 256, 56),
        GgmlType(30, "BF16", 1, 2, dtype=torch.bfloat16),
        GgmlType(34, "TQ1_0", 256, 54),
        GgmlType(35, "TQ2_0", 256, 66),
        GgmlType(39, "MXFP4", 32, 17),
        GgmlType(40, "NVFP4", 64, 36),
    )
}

_GGML_TYPES_BY_NAME = {entry.name: entry for entry in _GGML_TYPES.values()}
SUPPORTED_GGML_TYPE_NAMES = tuple(entry.name for entry in _GGML_TYPES.values() if entry.supported)


@dataclass(frozen=True)
class GgufTensorInfo:
    """One tensor's location and shape, with the shape already in torch order."""

    name: str
    shape: tuple[int, ...]
    type_name: str
    offset: int
    nbytes: int


@dataclass(frozen=True)
class GgufHeader:
    version: int
    alignment: int
    architecture: str | None
    data_offset: int
    tensors: tuple[GgufTensorInfo, ...]


class _HeaderReader:
    """A forward-only cursor over the header, bounded so a bad length cannot allocate the machine."""

    def __init__(self, handle, label: str, file_size: int):
        self._handle = handle
        self._label = label
        self._file_size = file_size
        self.offset = 0

    def fail(self, detail: str) -> None:
        raise ValueError(f"{self._label} is not a readable GGUF file: {detail}")

    def _advance(self, count: int) -> None:
        if count < 0 or self.offset + count > GGUF_MAX_HEADER_BYTES:
            self.fail("the header is larger than this loader accepts")
        if self.offset + count > self._file_size:
            self.fail("the file ends inside its header")
        self.offset += count

    def read(self, count: int) -> bytes:
        self._advance(count)
        data = self._handle.read(count)
        if len(data) != count:
            self.fail("the file ends inside its header")
        return data

    def skip(self, count: int) -> None:
        # Seeking past the end of a file is not an error, so the bound is checked rather than the
        # result: an oversized length has to be refused here or it becomes a wrong offset later.
        self._advance(count)
        self._handle.seek(count, 1)

    def integer(self, size: int, signed: bool = False) -> int:
        return int.from_bytes(self.read(size), "little", signed=signed)

    def uint32(self) -> int:
        return self.integer(4)

    def uint64(self) -> int:
        return self.integer(8)

    def string(self) -> str:
        length = self.uint64()
        if length > GGUF_MAX_STRING_BYTES:
            self.fail("a header string is longer than this loader accepts")
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError:
            self.fail("a header string is not valid UTF-8")


def _read_metadata_value(reader: _HeaderReader, value_type: int, keep: bool):
    """Read one metadata value, materialising it only when the caller asked to keep it.

    A GGUF's metadata can hold a tokeniser vocabulary — hundreds of thousands of strings this
    loader has no use for. Walking past those costs integer reads; building them costs memory.
    """
    if value_type in _FIXED_VALUE_SIZES:
        size = _FIXED_VALUE_SIZES[value_type]
        if not keep:
            reader.skip(size)
            return None
        if value_type == _VALUE_FLOAT32:
            return float(np.frombuffer(reader.read(4), dtype="<f4")[0])
        if value_type == _VALUE_FLOAT64:
            return float(np.frombuffer(reader.read(8), dtype="<f8")[0])
        raw = reader.integer(size, signed=value_type in _SIGNED_VALUE_TYPES)
        return bool(raw) if value_type == _VALUE_BOOL else raw
    if value_type == _VALUE_STRING:
        if not keep:
            length = reader.uint64()
            if length > GGUF_MAX_STRING_BYTES:
                reader.fail("a header string is longer than this loader accepts")
            reader.skip(length)
            return None
        return reader.string()
    if value_type == _VALUE_ARRAY:
        element_type = reader.uint32()
        count = reader.uint64()
        if element_type == _VALUE_ARRAY:
            reader.fail("a nested metadata array is not part of the format")
        if element_type in _FIXED_VALUE_SIZES and not keep:
            reader.skip(_FIXED_VALUE_SIZES[element_type] * count)
            return None
        if element_type not in _FIXED_VALUE_SIZES and element_type != _VALUE_STRING:
            reader.fail(f"metadata array element type {element_type} is unknown")
        return [_read_metadata_value(reader, element_type, keep) for _index in range(count)]
    reader.fail(f"metadata value type {value_type} is unknown")


def _wanted_metadata_key(key: str) -> bool:
    return key in (GGUF_ALIGNMENT_KEY, GGUF_ARCHITECTURE_KEY) or key.startswith(COMFY_ORIGINAL_SHAPE_PREFIX)


def _original_shape(metadata: Mapping[str, object], name: str, elements: int, reader: _HeaderReader):
    """ComfyUI's record of a shape ggml could not hold, when the converter had to flatten one."""
    recorded = metadata.get(f"{COMFY_ORIGINAL_SHAPE_PREFIX}{name}")
    if recorded is None:
        return None
    if not isinstance(recorded, list) or not recorded or not all(isinstance(value, int) and value > 0 for value in recorded):
        reader.fail(f"tensor {name!r} records an unusable original shape")
    shape = tuple(int(value) for value in recorded)
    total = 1
    for dimension in shape:
        total *= dimension
    if total != elements:
        reader.fail(f"tensor {name!r} records an original shape that does not hold its {elements} elements")
    return shape


def read_gguf_header(path: str | Path, label: str = "GGUF checkpoint") -> GgufHeader:
    """Parse the container: metadata, every tensor's shape and type, and where the data starts."""
    if sys.byteorder != "little":
        raise RuntimeError("Reading GGUF needs a little-endian host; this machine is big-endian")
    path = Path(path)
    file_size = path.stat().st_size
    with open(path, "rb") as handle:
        reader = _HeaderReader(handle, label, file_size)
        if reader.read(4) != GGUF_MAGIC:
            reader.fail("the file does not start with the GGUF magic bytes")
        version = reader.uint32()
        if version not in GGUF_SUPPORTED_VERSIONS:
            reader.fail(f"version {version} is not one of the supported versions {GGUF_SUPPORTED_VERSIONS}")
        tensor_count = reader.uint64()
        metadata_count = reader.uint64()
        if tensor_count > GGUF_MAX_TENSORS or metadata_count > GGUF_MAX_METADATA_ENTRIES:
            reader.fail("it declares more tensors or metadata entries than this loader accepts")

        metadata: dict[str, object] = {}
        for _index in range(metadata_count):
            key = reader.string()
            value_type = reader.uint32()
            value = _read_metadata_value(reader, value_type, _wanted_metadata_key(key))
            if _wanted_metadata_key(key):
                metadata[key] = value

        alignment = metadata.get(GGUF_ALIGNMENT_KEY, GGUF_DEFAULT_ALIGNMENT)
        if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
            reader.fail("its declared alignment is not a positive power of two")
        architecture = metadata.get(GGUF_ARCHITECTURE_KEY)

        tensors = []
        for _index in range(tensor_count):
            name = reader.string()
            dimension_count = reader.uint32()
            if dimension_count < 1 or dimension_count > GGML_MAX_DIMENSIONS:
                reader.fail(f"tensor {name!r} declares {dimension_count} dimensions")
            dimensions = [reader.uint64() for _axis in range(dimension_count)]
            type_identifier = reader.uint32()
            offset = reader.uint64()
            ggml_type = _GGML_TYPES.get(type_identifier)
            if ggml_type is None:
                reader.fail(f"tensor {name!r} uses ggml type {type_identifier}, which is not a known type")
            elements = 1
            for dimension in dimensions:
                if dimension < 1 or elements > GGUF_MAX_ELEMENTS // dimension:
                    reader.fail(f"tensor {name!r} declares an unusable shape")
                elements *= dimension
            if dimensions[0] % ggml_type.block_elements:
                reader.fail(
                    f"tensor {name!r} has a row of {dimensions[0]} elements, which is not a whole "
                    f"number of {ggml_type.name} blocks"
                )
            if offset % alignment:
                reader.fail(f"tensor {name!r} starts at an offset that is not aligned")
            nbytes = elements // ggml_type.block_elements * ggml_type.block_bytes
            # ggml stores the fastest-moving axis first; torch states it last.
            shape = _original_shape(metadata, name, elements, reader) or tuple(reversed(dimensions))
            tensors.append(GgufTensorInfo(name, shape, ggml_type.name, offset, nbytes))

        data_offset = (reader.offset + alignment - 1) // alignment * alignment
        if data_offset > file_size:
            reader.fail("the file ends before its tensor data begins")
        if len(set(tensor.name for tensor in tensors)) != len(tensors):
            reader.fail("two tensors share one name")
        occupied = sorted((tensor.offset, tensor.offset + tensor.nbytes, tensor.name) for tensor in tensors)
        previous_end = 0
        for start, end, name in occupied:
            if start < previous_end:
                reader.fail(f"tensor {name!r} overlaps the tensor before it")
            if data_offset + end > file_size:
                reader.fail(f"tensor {name!r} reaches past the end of the file")
            previous_end = end
    return GgufHeader(version, alignment, architecture if isinstance(architecture, str) else None, data_offset, tuple(tensors))


def gguf_quantization_summary(header: GgufHeader) -> str:
    """`Q4_K x 304, F32 x 476` — what a file is actually stored as, for a load report or an error."""
    counts: dict[str, int] = {}
    for tensor in header.tensors:
        counts[tensor.type_name] = counts.get(tensor.type_name, 0) + 1
    ordered = sorted(counts.items(), key=lambda entry: (-entry[1], entry[0]))
    return ", ".join(f"{name} x {count}" for name, count in ordered)


def _read_tensor(handle, header: GgufHeader, tensor: GgufTensorInfo, dtype: torch.dtype | None, label: str) -> torch.Tensor:
    ggml_type = _GGML_TYPES_BY_NAME[tensor.type_name]
    if not ggml_type.supported:
        raise ValueError(
            f"{label} stores {tensor.name!r} as {ggml_type.name}, which packs its weights in a layout "
            f"this loader cannot expand; use one of {', '.join(SUPPORTED_GGML_TYPE_NAMES)} instead"
        )
    handle.seek(header.data_offset + tensor.offset)
    raw = np.fromfile(handle, dtype=np.uint8, count=tensor.nbytes)
    if raw.size != tensor.nbytes:
        raise ValueError(f"{label} ends inside tensor {tensor.name!r}")
    packed = torch.from_numpy(raw)
    if ggml_type.dtype is not None:
        values = packed.view(ggml_type.dtype)
    else:
        # One tensor at a time, cast as it is produced. Holding the whole model at float32 before a
        # final cast would peak at four bytes per parameter — the same trap `dequantize_scaled_fp8`
        # documents, and the reason a 12 GB checkpoint can be loaded on a machine with 62 GB.
        values = ggml_type.dequantize(packed.reshape(-1, ggml_type.block_bytes)).reshape(-1)
    if dtype is not None and values.is_floating_point() and values.dtype != dtype:
        values = values.to(dtype=dtype)
    return values.reshape(tensor.shape).contiguous()


def load_gguf_state_dict(
    path: str | Path,
    dtype: torch.dtype | None = None,
    label: str = "GGUF checkpoint",
    header: GgufHeader | None = None,
) -> dict[str, torch.Tensor]:
    """Expand a GGUF file into an ordinary state dict, one tensor at a time.

    ``dtype`` is applied per tensor as each one is produced, so the peak is one expanded tensor
    rather than one expanded model.  Integer tensors keep their own type: a cast would corrupt
    them, and nothing in these checkpoints stores weights that way.
    """
    path = Path(path)
    header = header or read_gguf_header(path, label)
    state_dict: dict[str, torch.Tensor] = {}
    with open(path, "rb") as handle:
        for tensor in header.tensors:
            state_dict[tensor.name] = _read_tensor(handle, header, tensor, dtype, label)
    return state_dict


def is_gguf_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() == GGUF_SUFFIX
