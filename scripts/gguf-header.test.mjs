import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  DIFFUSION_MODEL_EXTENSIONS,
  GGUF_SUPPORTED_TYPES,
  componentFileExtensions,
  readDiffusionModelHeader,
  readGgufHeader,
} from "./gguf-header.mjs";

// The type ids ggml assigns. Only the ones this program reads are listed as supported; the rest
// are named here so the tests can prove a file using one is refused rather than offered.
const TYPES = { F32: 0, F16: 1, Q4_0: 2, Q8_0: 8, Q4_K: 12, Q6_K: 14, IQ2_XXS: 16, IQ4_XS: 23, BF16: 30 };

// The list `backend/test_gguf_loader.py` pins on the Python side. A type readable by only one of
// the two means the picker offers a file the runtime refuses, or hides one it could have loaded.
const EXPECTED_SUPPORTED_TYPES = [
  "BF16", "F16", "F32", "F64", "I16", "I32", "I64", "I8", "IQ4_NL", "IQ4_XS",
  "Q2_K", "Q3_K", "Q4_0", "Q4_1", "Q4_K", "Q5_0", "Q5_1", "Q5_K", "Q6_K", "Q8_0", "Q8_K",
];

function u32(value) {
  const buffer = Buffer.alloc(4);
  buffer.writeUInt32LE(value);
  return buffer;
}

function u64(value) {
  const buffer = Buffer.alloc(8);
  buffer.writeBigUInt64LE(BigInt(value));
  return buffer;
}

function i32(value) {
  const buffer = Buffer.alloc(4);
  buffer.writeInt32LE(value);
  return buffer;
}

function packString(text) {
  const data = Buffer.from(text, "utf8");
  return Buffer.concat([u64(data.length), data]);
}

function encodeValue(value) {
  if (typeof value === "string") return Buffer.concat([u32(8), packString(value)]);
  if (typeof value === "boolean") return Buffer.concat([u32(7), Buffer.from([value ? 1 : 0])]);
  if (typeof value === "number") return Buffer.concat([u32(5), i32(value)]);
  if (Array.isArray(value)) {
    const elementType = typeof value[0] === "string" ? 8 : 5;
    const body = value.map((item) => (elementType === 8 ? packString(item) : i32(item)));
    return Buffer.concat([u32(9), u32(elementType), u64(value.length), ...body]);
  }
  throw new Error(`unsupported metadata value ${value}`);
}

/** Write a GGUF file. `shape` is stated in torch order and reversed on the way out, as a real writer does. */
async function writeGguf(filePath, tensors, options = {}) {
  const { metadata = [], alignment = 32, version = 3, magic = "GGUF", trim = 0, damage = {} } = options;
  const parts = [Buffer.from(magic, "latin1"), u32(version), u64(tensors.length), u64(metadata.length)];
  for (const [key, value] of metadata) parts.push(packString(key), encodeValue(value));

  let offset = 0;
  const placements = [];
  for (const tensor of tensors) {
    const broken = damage[tensor.name] || {};
    parts.push(packString(tensor.name));
    parts.push(u32(broken.dimensions ?? tensor.shape.length));
    for (const dimension of [...tensor.shape].reverse()) parts.push(u64(dimension));
    parts.push(u32(broken.type ?? TYPES[tensor.type]));
    parts.push(u64(broken.offset ?? offset));
    placements.push([offset, tensor.payload]);
    offset = Math.ceil((offset + tensor.payload.length) / alignment) * alignment;
  }

  const header = Buffer.concat(parts);
  const padding = Buffer.alloc(Math.ceil(header.length / alignment) * alignment - header.length);
  const data = Buffer.alloc(offset);
  for (const [start, payload] of placements) payload.copy(data, start);
  const written = Buffer.concat([header, padding, data]);
  await writeFile(filePath, trim ? written.subarray(0, written.length - trim) : written);
  return filePath;
}

function floats(values) {
  const buffer = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => buffer.writeFloatLE(value, index * 4));
  return buffer;
}

async function temporaryDirectory(context) {
  const directory = await mkdtemp(path.join(os.tmpdir(), "xirai-gguf-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

test("the supported type list is the one the runtime publishes", () => {
  assert.deepEqual([...GGUF_SUPPORTED_TYPES.values()].map((entry) => entry.name).sort(), EXPECTED_SUPPORTED_TYPES);
});

test("a shape is reversed out of ggml order, and the quantisation is reported", async (context) => {
  const directory = await temporaryDirectory(context);
  const file = await writeGguf(path.join(directory, "flux.gguf"), [
    { name: "img_in.weight", type: "Q4_K", shape: [3072, 512], payload: Buffer.alloc(6 * 144) },
    { name: "img_in.bias", type: "F32", shape: [3072], payload: floats(new Array(3072).fill(0)) },
    { name: "conv.weight", type: "F16", shape: [2, 3, 4, 5], payload: Buffer.alloc(240) },
  ], { metadata: [["general.architecture", "flux"]] });

  const header = await readGgufHeader(file);
  assert.deepEqual(header["img_in.weight"], { dtype: "Q4_K", shape: [3072, 512] });
  assert.deepEqual(header["img_in.bias"], { dtype: "F32", shape: [3072] });
  assert.deepEqual(header["conv.weight"], { dtype: "F16", shape: [2, 3, 4, 5] });
});

test("a shape ggml could not hold is taken from ComfyUI's own record of it", async (context) => {
  const directory = await temporaryDirectory(context);
  const file = await writeGguf(path.join(directory, "wide.gguf"), [
    { name: "net.weight", type: "F32", shape: [720], payload: Buffer.alloc(720 * 4) },
  ], { metadata: [["comfy.gguf.orig_shape.net.weight", [2, 3, 4, 5, 6]]] });
  assert.deepEqual((await readGgufHeader(file))["net.weight"].shape, [2, 3, 4, 5, 6]);
});

test("a recorded shape that does not hold the tensor is refused", async (context) => {
  const directory = await temporaryDirectory(context);
  const file = await writeGguf(path.join(directory, "bad-shape.gguf"), [
    { name: "net.weight", type: "F32", shape: [720], payload: Buffer.alloc(720 * 4) },
  ], { metadata: [["comfy.gguf.orig_shape.net.weight", [2, 3]]] });
  await assert.rejects(readGgufHeader(file), /original shape/);
});

test("metadata this reader does not need is walked past rather than kept", async (context) => {
  // A real file carries a name, scalars and arrays none of this is interested in. Getting the skip
  // length wrong would misread every tensor record that follows.
  const directory = await temporaryDirectory(context);
  const file = await writeGguf(path.join(directory, "metadata.gguf"), [
    { name: "img_in.weight", type: "F32", shape: [2, 2], payload: floats([1, 2, 3, 4]) },
  ], {
    metadata: [
      ["general.architecture", "flux"],
      ["general.name", "a name with spaces"],
      ["general.quantization_version", 2],
      ["some.flag", true],
      ["tokenizer.ggml.tokens", ["one", "two", "three"]],
      ["some.identifiers", [4, 5, 6]],
    ],
  });
  assert.deepEqual((await readGgufHeader(file))["img_in.weight"], { dtype: "F32", shape: [2, 2] });
});

test("a header longer than the first read is grown rather than truncated", async (context) => {
  // The reader takes a small prefix of a file that can be twelve gigabytes, so the path that grows
  // that prefix is the one a real 780-tensor checkpoint depends on.
  const directory = await temporaryDirectory(context);
  const tensors = [];
  for (let index = 0; index < 9000; index += 1) {
    tensors.push({ name: `double_blocks.${index}.${"padding".repeat(14)}.weight`, type: "F32", shape: [1], payload: floats([index]) });
  }
  const file = await writeGguf(path.join(directory, "many.gguf"), tensors);
  const header = await readGgufHeader(file);
  assert.equal(Object.keys(header).length, 9000);
  assert.deepEqual(header[tensors[8999].name], { dtype: "F32", shape: [1] });
});

test("a file that is not a GGUF is refused", async (context) => {
  const directory = await temporaryDirectory(context);
  const file = path.join(directory, "not.gguf");
  await writeFile(file, Buffer.alloc(4096, 7));
  await assert.rejects(readGgufHeader(file), /magic bytes/);
});

test("an unsupported container version is refused", async (context) => {
  const directory = await temporaryDirectory(context);
  const file = await writeGguf(path.join(directory, "v1.gguf"), [
    { name: "w", type: "F32", shape: [1], payload: floats([1]) },
  ], { version: 1 });
  await assert.rejects(readGgufHeader(file), /version 1/);
});

test("a quantisation the runtime cannot expand is refused here rather than offered", async (context) => {
  // IQ2_XXS needs a code book `gguf_loader.py` does not carry. Listing the file would put it in the
  // picker and fail at load, which is the exact failure this reader exists to prevent.
  const directory = await temporaryDirectory(context);
  const file = await writeGguf(path.join(directory, "iq2.gguf"), [
    { name: "w", type: "IQ2_XXS", shape: [1, 256], payload: Buffer.alloc(66) },
  ]);
  await assert.rejects(readGgufHeader(file), /ggml type 16/);
});

test("an unknown ggml type is refused", async (context) => {
  const directory = await temporaryDirectory(context);
  const file = await writeGguf(path.join(directory, "unknown.gguf"), [
    { name: "w", type: "F32", shape: [1], payload: floats([1]) },
  ], { damage: { w: { type: 250 } } });
  await assert.rejects(readGgufHeader(file), /ggml type 250/);
});

test("a row that is not a whole number of blocks is refused", async (context) => {
  const directory = await temporaryDirectory(context);
  const file = await writeGguf(path.join(directory, "ragged.gguf"), [
    { name: "w", type: "Q4_K", shape: [2, 300], payload: Buffer.alloc(400) },
  ]);
  await assert.rejects(readGgufHeader(file), /whole number of Q4_K blocks/);
});

test("more dimensions than ggml holds is refused", async (context) => {
  const directory = await temporaryDirectory(context);
  const file = await writeGguf(path.join(directory, "dims.gguf"), [
    { name: "w", type: "F32", shape: [1], payload: floats([1]) },
  ], { damage: { w: { dimensions: 9 } } });
  await assert.rejects(readGgufHeader(file), /9 dimensions/);
});

test("two tensors sharing a name are refused", async (context) => {
  const directory = await temporaryDirectory(context);
  const file = await writeGguf(path.join(directory, "twice.gguf"), [
    { name: "w", type: "F32", shape: [1], payload: floats([1]) },
    { name: "w", type: "F32", shape: [1], payload: floats([2]) },
  ]);
  await assert.rejects(readGgufHeader(file), /names the tensor w twice/);
});

test("a truncated file is refused rather than read as a short header", async (context) => {
  const directory = await temporaryDirectory(context);
  const file = await writeGguf(path.join(directory, "cut.gguf"), [
    { name: "img_in.weight", type: "F32", shape: [2, 2], payload: floats([1, 2, 3, 4]) },
  ]);
  await writeFile(file, (await readFile(file)).subarray(0, 24));
  await assert.rejects(readGgufHeader(file), /truncated or larger than the limit/);
});

test("a file holding no tensors is refused", async (context) => {
  const directory = await temporaryDirectory(context);
  const file = await writeGguf(path.join(directory, "empty.gguf"), []);
  await assert.rejects(readGgufHeader(file), /no tensors/);
});

test("only the diffusion model slot widens past safetensors", () => {
  assert.deepEqual([...componentFileExtensions("diffusion_model")].sort(), [".gguf", ".safetensors"]);
  for (const role of ["text_encoder", "text_encoder_2", "vae"]) {
    assert.deepEqual([...componentFileExtensions(role)], [".safetensors"]);
  }
  assert.deepEqual([...DIFFUSION_MODEL_EXTENSIONS].sort(), [".gguf", ".safetensors"]);
});

test("the dispatcher reads each container with its own reader", async (context) => {
  const directory = await temporaryDirectory(context);
  const gguf = await writeGguf(path.join(directory, "model.gguf"), [
    { name: "img_in.weight", type: "BF16", shape: [4, 8], payload: Buffer.alloc(64) },
  ]);
  assert.deepEqual((await readDiffusionModelHeader(gguf))["img_in.weight"], { dtype: "BF16", shape: [4, 8] });

  const header = Buffer.from(JSON.stringify({ "img_in.weight": { dtype: "F32", shape: [1, 2], data_offsets: [0, 8] } }), "utf8");
  const safetensors = path.join(directory, "model.safetensors");
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(header.length));
  await writeFile(safetensors, Buffer.concat([length, header, Buffer.alloc(8)]));
  assert.deepEqual((await readDiffusionModelHeader(safetensors))["img_in.weight"].shape, [1, 2]);
});
