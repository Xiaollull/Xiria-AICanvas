import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  DIFFUSION_MODEL_EXTENSIONS,
  GGUF_SUPPORTED_TYPES,
  GGUF_TEXT_ENCODER_FAMILIES,
  GGUF_TEXT_ENCODER_KEY_MAPS,
  componentFileExtensions,
  ggufTextEncoderKey,
  readDiffusionModelHeader,
  readGgufHeader,
} from "./gguf-header.mjs";

// The type ids ggml assigns. Only the ones this program reads are listed as supported; the rest
// are named here so the tests can prove a file using one is refused rather than offered.
const TYPES = { F32: 0, F16: 1, Q4_0: 2, Q8_0: 8, Q4_K: 12, Q6_K: 14, IQ2_XXS: 16, IQ4_XS: 23, BF16: 30 };

// The list `backend/tests/test_gguf_loader.py` pins on the Python side. A type readable by only one of
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

test("every native component slot reads GGUF; anything else stays safetensors", () => {
  // P9: a GGUF text encoder or VAE is listed only when its tensors classify as that role, which a
  // llama.cpp encoder does only after `renameGgufTextEncoder` (see the tests below).
  for (const role of ["diffusion_model", "text_encoder", "text_encoder_2", "vae"]) {
    assert.deepEqual([...componentFileExtensions(role)].sort(), [".gguf", ".safetensors"], role);
  }
  assert.deepEqual([...componentFileExtensions("lora")], [".safetensors"]);
  assert.deepEqual([...DIFFUSION_MODEL_EXTENSIONS].sort(), [".gguf", ".safetensors"]);
});

// The same tables `backend/tests/test_gguf_loader.py` pins (`EXPECTED_TEXT_ENCODER_KEY_MAPS`).
const EXPECTED_TEXT_ENCODER_FAMILIES = { t5encoder: "t5", t5: "t5", qwen3: "qwen3", qwen3vl: "qwen3" };
const EXPECTED_TEXT_ENCODER_KEY_MAPS = {
  t5: {
    prefix: "enc.blk.",
    block: "encoder.block.{index}.",
    parts: {
      attn_q: "layer.0.SelfAttention.q", attn_k: "layer.0.SelfAttention.k",
      attn_v: "layer.0.SelfAttention.v", attn_o: "layer.0.SelfAttention.o",
      attn_rel_b: "layer.0.SelfAttention.relative_attention_bias", attn_norm: "layer.0.layer_norm",
      ffn_gate: "layer.1.DenseReluDense.wi_0", ffn_up: "layer.1.DenseReluDense.wi_1",
      ffn_down: "layer.1.DenseReluDense.wo", ffn_norm: "layer.1.layer_norm",
    },
    globals: { token_embd: "shared", "enc.output_norm": "encoder.final_layer_norm" },
    dropped: [],
  },
  qwen3: {
    prefix: "blk.",
    block: "model.layers.{index}.",
    parts: {
      attn_q: "self_attn.q_proj", attn_k: "self_attn.k_proj", attn_v: "self_attn.v_proj",
      attn_output: "self_attn.o_proj", attn_q_norm: "self_attn.q_norm", attn_k_norm: "self_attn.k_norm",
      attn_norm: "input_layernorm", ffn_norm: "post_attention_layernorm",
      ffn_gate: "mlp.gate_proj", ffn_up: "mlp.up_proj", ffn_down: "mlp.down_proj",
    },
    globals: { token_embd: "model.embed_tokens", output_norm: "model.norm" },
    dropped: ["output.weight"],
  },
};

test("the llama.cpp rename tables are the ones the runtime uses", () => {
  assert.deepEqual(Object.fromEntries(GGUF_TEXT_ENCODER_FAMILIES), EXPECTED_TEXT_ENCODER_FAMILIES);
  assert.deepEqual(GGUF_TEXT_ENCODER_KEY_MAPS, EXPECTED_TEXT_ENCODER_KEY_MAPS);
  assert.equal(ggufTextEncoderKey("enc.blk.0.attn_rel_b.weight", "t5"), "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");
  assert.equal(ggufTextEncoderKey("blk.35.attn_output.weight", "qwen3"), "model.layers.35.self_attn.o_proj.weight");
  assert.equal(ggufTextEncoderKey("blk.0.attn_q.bias", "qwen3"), "model.layers.0.self_attn.q_proj.bias");
  for (const unknown of ["blk.0.attn_qkv.weight", "blk.x.attn_q.weight", "rope_freqs.weight", "blk.0.attn_q.scale"]) {
    assert.equal(ggufTextEncoderKey(unknown, "qwen3"), null, unknown);
  }
});

function qwenTensors({ hidden = 2560, lmHead = true, extra = [] } = {}) {
  const tensors = [
    { name: "token_embd.weight", type: "F32", shape: [8, hidden], payload: Buffer.alloc(8 * hidden * 4) },
    { name: "blk.0.attn_q.weight", type: "F32", shape: [4, hidden], payload: Buffer.alloc(4 * hidden * 4) },
    { name: "blk.0.attn_q_norm.weight", type: "F32", shape: [128], payload: Buffer.alloc(512) },
    { name: "blk.0.ffn_norm.weight", type: "F32", shape: [hidden], payload: Buffer.alloc(hidden * 4) },
    { name: "output_norm.weight", type: "F32", shape: [hidden], payload: Buffer.alloc(hidden * 4) },
    ...extra,
  ];
  if (lmHead) tensors.push({ name: "output.weight", type: "F32", shape: [8, hidden], payload: Buffer.alloc(8 * hidden * 4) });
  return tensors;
}

test("a llama.cpp Qwen3 or Qwen3-VL encoder is read back in ComfyUI's names, head dropped", async (context) => {
  const directory = await temporaryDirectory(context);
  for (const architecture of ["qwen3", "qwen3vl"]) {
    const file = await writeGguf(path.join(directory, `${architecture}.gguf`), qwenTensors(), { metadata: [["general.architecture", architecture]] });
    const header = await readGgufHeader(file);
    assert.deepEqual(header.__metadata__, { "general.architecture": architecture });
    assert.deepEqual(Object.keys(header).filter((key) => key !== "__metadata__").sort(), [
      "model.embed_tokens.weight",
      "model.layers.0.post_attention_layernorm.weight",
      "model.layers.0.self_attn.q_norm.weight",
      "model.layers.0.self_attn.q_proj.weight",
      "model.norm.weight",
    ]);
    assert.deepEqual(header["model.layers.0.self_attn.q_proj.weight"].shape, [4, 2560]);
  }
});

test("a llama.cpp T5 encoder is read back in the FLUX.1 T5-XXL layout", async (context) => {
  const directory = await temporaryDirectory(context);
  const file = await writeGguf(path.join(directory, "t5.gguf"), [
    { name: "token_embd.weight", type: "F32", shape: [4, 8], payload: Buffer.alloc(128) },
    { name: "enc.blk.0.attn_rel_b.weight", type: "F32", shape: [32, 64], payload: Buffer.alloc(32 * 64 * 4) },
    { name: "enc.output_norm.weight", type: "F32", shape: [8], payload: Buffer.alloc(32) },
  ], { metadata: [["general.architecture", "t5encoder"]] });
  const header = await readGgufHeader(file);
  assert.deepEqual(header.__metadata__, { "general.architecture": "t5encoder" });
  assert.deepEqual(Object.keys(header).filter((key) => key !== "__metadata__").sort(), [
    "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
    "encoder.final_layer_norm.weight",
    "shared.weight",
  ]);
});

test("an encoder the runtime cannot map keeps llama.cpp's names, so no classifier lists it", async (context) => {
  const directory = await temporaryDirectory(context);
  // Mistral (`llama`) is refused by the runtime; an unnameable tensor refuses a whole Qwen3 file.
  const mistral = await writeGguf(path.join(directory, "mistral.gguf"), qwenTensors({ hidden: 5120 }), { metadata: [["general.architecture", "llama"]] });
  assert.ok(Object.hasOwn(await readGgufHeader(mistral), "blk.0.attn_q.weight"));
  const odd = await writeGguf(path.join(directory, "odd.gguf"), qwenTensors({
    extra: [{ name: "blk.0.attn_qkv.weight", type: "F32", shape: [4, 8], payload: Buffer.alloc(128) }],
  }), { metadata: [["general.architecture", "qwen3"]] });
  assert.ok(Object.hasOwn(await readGgufHeader(odd), "blk.0.attn_q.weight"));
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
