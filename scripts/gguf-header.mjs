import { open } from "node:fs/promises";
import path from "node:path";

import { readSafetensorsHeader } from "./anima-models.mjs";

// The picker classifies a component by the tensors it holds, so a GGUF has to answer the same
// question a safetensors header answers: which tensors are in this file and what shape is each
// one. That is all this reader produces — names and shapes, never weights — which is why it can
// afford to run over every file in a folder.
//
// This is the JavaScript twin of `backend/gguf_loader.py`. The two must agree on which ggml types
// are readable or the picker will offer a file the runtime then refuses; `gguf-header.test.mjs`
// and `backend/test_gguf_loader.py` pin the same list on both sides.

export const GGUF_MAGIC = "GGUF";
export const GGUF_SUPPORTED_VERSIONS = new Set([2, 3]);
export const MAX_GGUF_HEADER_BYTES = 64 * 1024 * 1024;
export const GGML_MAX_DIMENSIONS = 4;
export const MAX_GGUF_TENSORS = 1 << 20;
export const MAX_GGUF_METADATA_ENTRIES = 1 << 20;
export const MAX_GGUF_STRING_BYTES = 1 << 24;

const COMFY_ORIGINAL_SHAPE_PREFIX = "comfy.gguf.orig_shape.";
const INITIAL_HEADER_BYTES = 1 << 20;

// Every ggml type `gguf_loader.py` can expand, by its type id, with the number of elements one
// stored block holds. A file using anything else — the code-book IQ layouts, the ternary and
// micro-float types — is left out of the picker rather than listed and then refused at load.
export const GGUF_SUPPORTED_TYPES = new Map([
  [0, { name: "F32", blockElements: 1 }],
  [1, { name: "F16", blockElements: 1 }],
  [2, { name: "Q4_0", blockElements: 32 }],
  [3, { name: "Q4_1", blockElements: 32 }],
  [6, { name: "Q5_0", blockElements: 32 }],
  [7, { name: "Q5_1", blockElements: 32 }],
  [8, { name: "Q8_0", blockElements: 32 }],
  [10, { name: "Q2_K", blockElements: 256 }],
  [11, { name: "Q3_K", blockElements: 256 }],
  [12, { name: "Q4_K", blockElements: 256 }],
  [13, { name: "Q5_K", blockElements: 256 }],
  [14, { name: "Q6_K", blockElements: 256 }],
  [15, { name: "Q8_K", blockElements: 256 }],
  [20, { name: "IQ4_NL", blockElements: 32 }],
  [23, { name: "IQ4_XS", blockElements: 256 }],
  [24, { name: "I8", blockElements: 1 }],
  [25, { name: "I16", blockElements: 1 }],
  [26, { name: "I32", blockElements: 1 }],
  [27, { name: "I64", blockElements: 1 }],
  [28, { name: "F64", blockElements: 1 }],
  [30, { name: "BF16", blockElements: 1 }],
]);

export const DIFFUSION_MODEL_EXTENSIONS = new Set([".safetensors", ".gguf"]);
const SAFETENSORS_ONLY_EXTENSIONS = new Set([".safetensors"]);

// Only the diffusion model widens. A text encoder or VAE offered as GGUF would be listed here and
// then refused by the runtime, which is the failure this whole change exists to remove.
export function componentFileExtensions(role) {
  return role === "diffusion_model" ? DIFFUSION_MODEL_EXTENSIONS : SAFETENSORS_ONLY_EXTENSIONS;
}

const FIXED_VALUE_SIZES = new Map([[0, 1], [1, 1], [2, 2], [3, 2], [4, 4], [5, 4], [6, 4], [7, 1], [10, 8], [11, 8], [12, 8]]);
const FIXED_VALUE_READERS = new Map([
  [0, (buffer, offset) => buffer.readUInt8(offset)],
  [1, (buffer, offset) => buffer.readInt8(offset)],
  [2, (buffer, offset) => buffer.readUInt16LE(offset)],
  [3, (buffer, offset) => buffer.readInt16LE(offset)],
  [4, (buffer, offset) => buffer.readUInt32LE(offset)],
  [5, (buffer, offset) => buffer.readInt32LE(offset)],
  [6, (buffer, offset) => buffer.readFloatLE(offset)],
  [7, (buffer, offset) => buffer.readUInt8(offset) !== 0],
  [10, (buffer, offset) => Number(buffer.readBigUInt64LE(offset))],
  [11, (buffer, offset) => Number(buffer.readBigInt64LE(offset))],
  [12, (buffer, offset) => buffer.readDoubleLE(offset)],
]);
const VALUE_STRING = 8;
const VALUE_ARRAY = 9;

const decoder = new TextDecoder("utf-8", { fatal: true });

function incomplete() {
  return Object.assign(new Error("GGUF header continues past the bytes read so far"), { code: "GGUF_NEED_MORE" });
}

class HeaderCursor {
  constructor(buffer) {
    this.buffer = buffer;
    this.offset = 0;
  }

  take(count) {
    if (count < 0 || !Number.isSafeInteger(count)) throw new Error("GGUF header declares an unusable length");
    const start = this.offset;
    const end = start + count;
    if (end > this.buffer.length) throw incomplete();
    this.offset = end;
    return start;
  }

  uint32() {
    return this.buffer.readUInt32LE(this.take(4));
  }

  uint64() {
    const value = this.buffer.readBigUInt64LE(this.take(8));
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("GGUF header declares an unusable length");
    return Number(value);
  }

  string() {
    const length = this.uint64();
    if (length > MAX_GGUF_STRING_BYTES) throw new Error("GGUF header holds a string longer than the limit");
    const start = this.take(length);
    return decoder.decode(this.buffer.subarray(start, start + length));
  }
}

function readMetadataValue(cursor, valueType, keep) {
  const fixed = FIXED_VALUE_SIZES.get(valueType);
  if (fixed !== undefined) {
    const start = cursor.take(fixed);
    return keep ? FIXED_VALUE_READERS.get(valueType)(cursor.buffer, start) : null;
  }
  if (valueType === VALUE_STRING) {
    if (keep) return cursor.string();
    const length = cursor.uint64();
    if (length > MAX_GGUF_STRING_BYTES) throw new Error("GGUF header holds a string longer than the limit");
    cursor.take(length);
    return null;
  }
  if (valueType !== VALUE_ARRAY) throw new Error(`GGUF metadata value type ${valueType} is unknown`);
  const elementType = cursor.uint32();
  const count = cursor.uint64();
  if (elementType === VALUE_ARRAY) throw new Error("GGUF metadata holds a nested array, which is not part of the format");
  const elementSize = FIXED_VALUE_SIZES.get(elementType);
  if (elementSize !== undefined && !keep) {
    cursor.take(elementSize * count);
    return null;
  }
  if (elementSize === undefined && elementType !== VALUE_STRING) {
    throw new Error(`GGUF metadata array element type ${elementType} is unknown`);
  }
  const values = [];
  for (let index = 0; index < count; index += 1) values.push(readMetadataValue(cursor, elementType, keep));
  return keep ? values : null;
}

function wantedMetadataKey(key) {
  return key.startsWith(COMFY_ORIGINAL_SHAPE_PREFIX);
}

function originalShape(metadata, name, elements) {
  const recorded = metadata.get(`${COMFY_ORIGINAL_SHAPE_PREFIX}${name}`);
  if (recorded === undefined) return null;
  if (!Array.isArray(recorded) || !recorded.length || !recorded.every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new Error(`GGUF tensor ${name} records an unusable original shape`);
  }
  if (recorded.reduce((total, dimension) => total * dimension, 1) !== elements) {
    throw new Error(`GGUF tensor ${name} records an original shape that does not hold its elements`);
  }
  return recorded;
}

export function parseGgufHeader(buffer, fileSize) {
  const cursor = new HeaderCursor(buffer);
  if (buffer.length < 4 || buffer.toString("latin1", 0, 4) !== GGUF_MAGIC) {
    if (buffer.length < 4) throw incomplete();
    throw new Error("File does not start with the GGUF magic bytes");
  }
  cursor.take(4);
  const version = cursor.uint32();
  if (!GGUF_SUPPORTED_VERSIONS.has(version)) throw new Error(`GGUF version ${version} is not supported`);
  const tensorCount = cursor.uint64();
  const metadataCount = cursor.uint64();
  if (tensorCount > MAX_GGUF_TENSORS || metadataCount > MAX_GGUF_METADATA_ENTRIES) {
    throw new Error("GGUF file declares more tensors or metadata entries than the limit");
  }

  const metadata = new Map();
  for (let index = 0; index < metadataCount; index += 1) {
    const key = cursor.string();
    const valueType = cursor.uint32();
    const keep = wantedMetadataKey(key);
    const value = readMetadataValue(cursor, valueType, keep);
    if (keep) metadata.set(key, value);
  }

  const header = {};
  for (let index = 0; index < tensorCount; index += 1) {
    const name = cursor.string();
    const dimensionCount = cursor.uint32();
    if (dimensionCount < 1 || dimensionCount > GGML_MAX_DIMENSIONS) {
      throw new Error(`GGUF tensor ${name} declares ${dimensionCount} dimensions`);
    }
    const dimensions = [];
    for (let axis = 0; axis < dimensionCount; axis += 1) dimensions.push(cursor.uint64());
    const typeIdentifier = cursor.uint32();
    cursor.uint64(); // The tensor's offset into the data section, which a header read never visits.
    if (dimensions.some((dimension) => dimension < 1)) throw new Error(`GGUF tensor ${name} declares an unusable shape`);
    const type = GGUF_SUPPORTED_TYPES.get(typeIdentifier);
    if (!type) throw new Error(`GGUF tensor ${name} uses ggml type ${typeIdentifier}, which this program cannot read`);
    if (dimensions[0] % type.blockElements) throw new Error(`GGUF tensor ${name} does not hold a whole number of ${type.name} blocks`);
    if (name in header) throw new Error(`GGUF file names the tensor ${name} twice`);
    const elements = dimensions.reduce((total, dimension) => total * dimension, 1);
    // ggml states the fastest-moving axis first; every classifier here reads torch's order.
    header[name] = { dtype: type.name, shape: originalShape(metadata, name, elements) || dimensions.slice().reverse() };
  }
  if (!Object.keys(header).length) throw new Error("GGUF file holds no tensors");
  if (cursor.offset > fileSize) throw new Error("GGUF header is truncated");
  return header;
}

export async function readGgufHeader(filePath, maximumHeaderBytes = MAX_GGUF_HEADER_BYTES) {
  const handle = await open(filePath, "r");
  try {
    const { size } = await handle.stat();
    const ceiling = Math.min(size, maximumHeaderBytes);
    // A diffusion GGUF's header is tens of kilobytes at the front of a file that can be twelve
    // gigabytes, so it is read in a small prefix and grown only if the tensor list runs past it.
    let length = Math.min(ceiling, INITIAL_HEADER_BYTES);
    for (;;) {
      const buffer = Buffer.alloc(length);
      const { bytesRead } = await handle.read(buffer, 0, length, 0);
      if (bytesRead !== length) throw new Error("GGUF file is shorter than its own header");
      try {
        return parseGgufHeader(buffer, size);
      } catch (error) {
        if (error?.code !== "GGUF_NEED_MORE") throw error;
        if (length >= ceiling) throw new Error("GGUF header is truncated or larger than the limit");
        length = Math.min(length * 8, ceiling);
      }
    }
  } finally {
    await handle.close();
  }
}

export async function readDiffusionModelHeader(filePath, maximumHeaderBytes) {
  return path.extname(filePath).toLowerCase() === ".gguf"
    ? readGgufHeader(filePath, maximumHeaderBytes)
    : readSafetensorsHeader(filePath, maximumHeaderBytes);
}
