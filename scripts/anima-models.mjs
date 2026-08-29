import { createHash } from "node:crypto";
import { open, readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";

export const MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024;

export const ANIMA_RUNTIME_ARTIFACTS = Object.freeze({
  qwen_tokenizer: Object.freeze({
    filename: "anima-qwen3-tokenizer.json",
    size: 7334926,
    sha256: "47ec9be242d3ef39b9c97ac0a3f06c1752f061b234e295bc0a2842067a3fe4f9",
  }),
  qwen_tokenizer_config: Object.freeze({
    filename: "anima-qwen3-tokenizer-config.json",
    size: 9916,
    sha256: "7992a7924330571ac9b97d58e39d4a4993ccdb865335034cec29cf2c482fd460",
  }),
  t5_tokenizer: Object.freeze({
    filename: "anima-t5-tokenizer.json",
    size: 1389353,
    sha256: "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066",
  }),
});

export function bundledAnimaTokenizerDirectory(projectRoot) {
  return path.join(projectRoot, "backend", "resources", "anima-tokenizers");
}

async function hashRuntimeArtifact(file) {
  return createHash("sha256").update(await readFile(file.path)).digest("hex");
}

export async function animaRuntimeArtifactStatuses(directory, hashFile = hashRuntimeArtifact, artifacts = ANIMA_RUNTIME_ARTIFACTS) {
  const entries = await Promise.all(Object.entries(artifacts).map(async ([name, artifact]) => {
    const filePath = path.join(directory, artifact.filename);
    try {
      const fileStat = await stat(filePath);
      if (!fileStat.isFile() || fileStat.size !== artifact.size) {
        return [name, { path: filePath, installed: false, reason: "file size mismatch" }];
      }
      const sha256 = await hashFile({
        path: filePath,
        size: fileStat.size,
        mtimeMs: fileStat.mtimeMs,
        ctimeMs: fileStat.ctimeMs,
      });
      return sha256 === artifact.sha256
        ? [name, { path: filePath, installed: true, reason: null }]
        : [name, { path: filePath, installed: false, reason: "SHA-256 mismatch" }];
    } catch (error) {
      return [name, { path: filePath, installed: false, reason: error.code === "ENOENT" ? "not installed" : "validation failed" }];
    }
  }));
  return Object.fromEntries(entries);
}

export async function requireBundledAnimaTokenizers(projectRoot) {
  const statuses = await animaRuntimeArtifactStatuses(bundledAnimaTokenizerDirectory(projectRoot));
  const unavailable = Object.entries(statuses).filter(([, status]) => !status.installed);
  if (unavailable.length) {
    throw new Error(`Bundled Anima tokenizer resources are missing or corrupt: ${unavailable.map(([name, status]) => `${name} (${status.reason})`).join(", ")}`);
  }
  return statuses;
}

const animaPrefixes = ["net.", "model.diffusion_model.", "diffusion_model."];
const safetensorsDtypeBytes = new Map([
  ["BOOL", 1], ["U8", 1], ["I8", 1],
  ["U16", 2], ["I16", 2], ["F16", 2], ["BF16", 2],
  ["U32", 4], ["I32", 4], ["F32", 4],
  ["U64", 8], ["I64", 8], ["F64", 8],
  ["F8_E4M3", 1], ["F8_E5M2", 1],
]);

export function normalizeAnimaSafetensorsKey(key) {
  for (const prefix of animaPrefixes) {
    if (key.startsWith(prefix)) return key.slice(prefix.length);
  }
  return key;
}

function validTensorDescriptor(value) {
  if (
    !value
    || typeof value !== "object"
    || Array.isArray(value)
    || !safetensorsDtypeBytes.has(value.dtype)
    || !Array.isArray(value.shape)
    || !value.shape.every((dimension) => Number.isSafeInteger(dimension) && dimension >= 0)
    || !Array.isArray(value.data_offsets)
    || value.data_offsets.length !== 2
    || !value.data_offsets.every((offset) => Number.isSafeInteger(offset) && offset >= 0)
    || value.data_offsets[0] > value.data_offsets[1]
  ) return false;
  const elements = value.shape.reduce((total, dimension) => total * BigInt(dimension), 1n);
  const expectedBytes = elements * BigInt(safetensorsDtypeBytes.get(value.dtype));
  return BigInt(value.data_offsets[1] - value.data_offsets[0]) === expectedBytes;
}

function normalizedTensorMap(header, dataBytes = null) {
  if (!header || typeof header !== "object" || Array.isArray(header)) return null;
  const tensors = new Map();
  const ranges = [];
  for (const [key, descriptor] of Object.entries(header)) {
    if (key === "__metadata__") continue;
    if (!key || !validTensorDescriptor(descriptor)) return null;
    const normalized = normalizeAnimaSafetensorsKey(key);
    if (!normalized || tensors.has(normalized)) return null;
    tensors.set(normalized, descriptor);
    ranges.push(descriptor.data_offsets);
  }
  if (!tensors.size) return null;
  ranges.sort((first, second) => first[0] - second[0]);
  let expectedStart = 0;
  for (const [start, end] of ranges) {
    if (start !== expectedStart) return null;
    expectedStart = end;
  }
  if (dataBytes != null && expectedStart !== dataBytes) return null;
  return tensors;
}

function hasShape(tensors, key, shape) {
  const actual = tensors.get(key)?.shape;
  return Array.isArray(actual) && actual.length === shape.length && actual.every((dimension, index) => dimension === shape[index]);
}

function isAnimaDiffusion(tensors) {
  return tensors.size >= 680
    && hasShape(tensors, "llm_adapter.embed.weight", [32128, 1024])
    && hasShape(tensors, "llm_adapter.blocks.5.mlp.2.weight", [1024, 4096])
    && hasShape(tensors, "x_embedder.proj.1.weight", [2048, 68])
    && hasShape(tensors, "blocks.0.self_attn.q_proj.weight", [2048, 2048])
    && hasShape(tensors, "blocks.27.self_attn.q_proj.weight", [2048, 2048])
    && hasShape(tensors, "final_layer.linear.weight", [64, 2048]);
}

function isQwenTextEncoder(tensors) {
  const prefix = tensors.has("model.embed_tokens.weight") ? "model." : tensors.has("embed_tokens.weight") ? "" : null;
  if (prefix == null || !hasShape(tensors, `${prefix}embed_tokens.weight`, [151936, 1024])) return false;
  return tensors.size >= 300
    && hasShape(tensors, `${prefix}norm.weight`, [1024])
    && hasShape(tensors, `${prefix}layers.0.self_attn.q_proj.weight`, [2048, 1024])
    && hasShape(tensors, `${prefix}layers.27.self_attn.q_proj.weight`, [2048, 1024]);
}

function isDiffusersQwenImageVae(tensors) {
  return tensors.size >= 190
    && hasShape(tensors, "encoder.conv_in.weight", [96, 3, 3, 3, 3])
    && hasShape(tensors, "decoder.conv_out.weight", [3, 96, 3, 3, 3])
    && hasShape(tensors, "quant_conv.weight", [32, 32, 1, 1, 1])
    && hasShape(tensors, "post_quant_conv.weight", [16, 16, 1, 1, 1]);
}

function isComfyQwenImageVae(tensors) {
  return tensors.size >= 190
    && hasShape(tensors, "encoder.conv1.weight", [96, 3, 3, 3, 3])
    && hasShape(tensors, "decoder.head.2.weight", [3, 96, 3, 3, 3])
    && hasShape(tensors, "conv1.weight", [32, 32, 1, 1, 1])
    && hasShape(tensors, "conv1.bias", [32])
    && hasShape(tensors, "conv2.weight", [16, 16, 1, 1, 1])
    && tensors.has("encoder.middle.0.residual.0.gamma")
    && tensors.has("decoder.middle.0.residual.0.gamma");
}

export function classifyAnimaSafetensorsHeader(header) {
  const tensors = normalizedTensorMap(header);
  if (!tensors) return null;
  const matches = [
    ["diffusion_model", isAnimaDiffusion(tensors)],
    ["text_encoder", isQwenTextEncoder(tensors)],
    ["vae", isDiffusersQwenImageVae(tensors) || isComfyQwenImageVae(tensors)],
  ].filter(([, matchesSignature]) => matchesSignature);
  return matches.length === 1 ? matches[0][0] : null;
}

export async function readSafetensorsHeader(filePath, maximumHeaderBytes = MAX_SAFETENSORS_HEADER_BYTES) {
  const handle = await open(filePath, "r");
  try {
    const prefix = Buffer.alloc(8);
    const prefixRead = await handle.read(prefix, 0, prefix.length, 0);
    if (prefixRead.bytesRead !== prefix.length) throw new Error("Safetensors file is missing its header length");
    const headerLength = prefix.readBigUInt64LE(0);
    if (headerLength < 2n || headerLength > BigInt(maximumHeaderBytes) || headerLength > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error("Safetensors header length is invalid or exceeds the limit");
    }
    const fileStat = await handle.stat();
    if (8n + headerLength > BigInt(fileStat.size)) throw new Error("Safetensors header is truncated");
    const headerBuffer = Buffer.alloc(Number(headerLength));
    const headerRead = await handle.read(headerBuffer, 0, headerBuffer.length, 8);
    if (headerRead.bytesRead !== headerBuffer.length) throw new Error("Safetensors header is truncated");
    const text = new TextDecoder("utf-8", { fatal: true }).decode(headerBuffer);
    const header = JSON.parse(text);
    if (!header || typeof header !== "object" || Array.isArray(header)) throw new Error("Safetensors header must be a JSON object");
    const dataBytes = fileStat.size - 8 - Number(headerLength);
    if (!normalizedTensorMap(header, dataBytes)) throw new Error("Safetensors tensor descriptors or data section are invalid");
    return header;
  } finally {
    await handle.close();
  }
}

async function scanAnimaRoot(directory, role, relativeDirectory = "", readHeader = readSafetensorsHeader) {
  let entries;
  try {
    entries = await readdir(path.join(directory, relativeDirectory), { withFileTypes: true });
  } catch {
    return [];
  }
  const models = [];
  for (const entry of entries) {
    const relativePath = path.join(relativeDirectory, entry.name);
    if (entry.isDirectory()) {
      models.push(...await scanAnimaRoot(directory, role, relativePath, readHeader));
      continue;
    }
    if (!entry.isFile() || path.extname(entry.name).toLowerCase() !== ".safetensors") continue;
    try {
      const filePath = path.join(directory, relativePath);
      const [header, fileStat] = await Promise.all([readHeader(filePath), stat(filePath)]);
      if (classifyAnimaSafetensorsHeader(header) !== role) continue;
      const value = relativePath.split(path.sep).join("/");
      models.push({ name: value, value, size: fileStat.size, modifiedAt: fileStat.mtimeMs });
    } catch {
      // One corrupt or disappearing file must not hide the rest of the catalog.
    }
  }
  return models;
}

export async function discoverAnimaModels(roots, readHeader = readSafetensorsHeader) {
  const entries = await Promise.all(Object.entries(roots).map(async ([role, directory]) => {
    const models = await scanAnimaRoot(directory, role, "", readHeader);
    models.sort((first, second) => first.name.localeCompare(second.name, "en"));
    return [role, models];
  }));
  return Object.fromEntries(entries);
}
