import { readdir, stat } from "node:fs/promises";
import path from "node:path";

import { componentFileExtensions, readDiffusionModelHeader } from "./gguf-header.mjs";
import { normalizeFluxSafetensorsKey } from "./flux-models.mjs";

// Krea 2 mounts three component files out of the same shared directories Anima and both Flux
// engines use, so the picker cannot ask the user which file is which — it has to know. Every
// classifier below reads the tensors ComfyUI's `model_detection` and `sd.py` read, which is what
// keeps four engines' files apart in one folder without a filename heuristic.

function tensorMap(header) {
  if (!header || typeof header !== "object" || Array.isArray(header)) return null;
  const tensors = new Map();
  for (const [key, descriptor] of Object.entries(header)) {
    if (key === "__metadata__") continue;
    if (!descriptor || typeof descriptor !== "object" || !Array.isArray(descriptor.shape)) return null;
    tensors.set(normalizeFluxSafetensorsKey(key), descriptor);
  }
  return tensors.size ? tensors : null;
}

function shapeOf(tensors, key) {
  const shape = tensors.get(key)?.shape;
  return Array.isArray(shape) ? shape : null;
}

function hasShape(tensors, key, shape) {
  const actual = shapeOf(tensors, key);
  return Array.isArray(actual)
    && actual.length === shape.length
    && actual.every((dimension, index) => shape[index] === null || dimension === shape[index]);
}

// `model_detection.py` divides Krea 2's attention projections by a fixed 128, so the head width is
// part of the format rather than something read off the file.
const KREA2_ATTENTION_HEAD_DIM = 128;
// The conditioning is a twelve-layer Qwen3-VL-4B stack; `txtfusion.projector` collapses that axis.
const KREA2_TAP_COUNT = 12;
// Wan 2.1's latent: 16 channels at stride 8, which the transformer then patches 2x2.
const KREA2_LATENT_CHANNELS = 16;

function isKrea2Diffusion(tensors) {
  // `txtfusion.projector.weight` is what `model_detection` keys "krea2" on: the learned collapse
  // of the twelve text-encoder taps, which nothing else in the catalog has.
  const projector = shapeOf(tensors, "txtfusion.projector.weight");
  if (!projector || projector.length !== 2 || projector[0] !== 1 || projector[1] !== KREA2_TAP_COUNT) return false;
  const first = shapeOf(tensors, "first.weight");
  if (!first || first.length !== 2) return false;
  const [features, packed] = first;
  if (packed !== KREA2_LATENT_CHANNELS * 2 * 2) return false;
  const query = shapeOf(tensors, "blocks.0.attn.wq.weight");
  if (!query || query.length !== 2 || query[1] !== features || query[0] % KREA2_ATTENTION_HEAD_DIM) return false;
  return tensors.has("blocks.0.attn.wk.weight")
    && tensors.has("blocks.0.attn.gate.weight")
    && tensors.has("blocks.0.mod.lin")
    && tensors.has("txtfusion.layerwise_blocks.0.prenorm.scale")
    && tensors.has("txtfusion.refiner_blocks.0.prenorm.scale")
    && hasShape(tensors, "last.linear.weight", [packed, features]);
}

function languageModelPrefix(tensors) {
  for (const prefix of ["model.language_model.", "model.", ""]) {
    if (tensors.has(`${prefix}layers.0.post_attention_layernorm.weight`)) return prefix;
  }
  return null;
}

// Qwen3-VL-4B's language half is 2560 wide. FLUX.2 [klein] mounts a Qwen3 encoder of exactly that
// width, so the width alone cannot separate them: the DeepStack merger the vision tower carries is
// unique to Qwen3-VL, and `comfy/sd.py::detect_te_model` uses the same tensor to tell them apart.
const KREA2_TEXT_ENCODER_HIDDEN = 2560;
const KREA2_DEEPSTACK_KEYS = [
  "model.visual.deepstack_merger_list.0.norm.weight",
  "visual.deepstack_merger_list.0.norm.weight",
];
// The deepest conditioning tap is layer 35, so a shallower checkpoint cannot be a Krea 2 encoder.
const KREA2_DEEPEST_TAP = 35;

function isKrea2TextEncoder(tensors) {
  if (!KREA2_DEEPSTACK_KEYS.some((key) => tensors.has(key))) return false;
  const prefix = languageModelPrefix(tensors);
  if (prefix == null) return false;
  const norm = shapeOf(tensors, `${prefix}layers.0.post_attention_layernorm.weight`);
  if (!norm || norm.length !== 1 || norm[0] !== KREA2_TEXT_ENCODER_HIDDEN) return false;
  if (!tensors.has(`${prefix}layers.0.self_attn.q_norm.weight`)) return false;
  if (!tensors.has(`${prefix}layers.${KREA2_DEEPEST_TAP}.post_attention_layernorm.weight`)) return false;
  return hasShape(tensors, `${prefix}embed_tokens.weight`, [null, KREA2_TEXT_ENCODER_HIDDEN])
    && hasShape(tensors, `${prefix}layers.0.self_attn.q_proj.weight`, [null, KREA2_TEXT_ENCODER_HIDDEN])
    && hasShape(tensors, `${prefix}layers.0.mlp.gate_proj.weight`, [null, KREA2_TEXT_ENCODER_HIDDEN]);
}

function isKrea2Vae(tensors) {
  // Wan 2.1's residual middle block, and not Wan 2.2's upsample stack — the same two tests
  // `comfy/sd.py::VAE.__init__` makes. `encoder.head.2` is 32 wide because it emits mean and
  // log-variance for sixteen channels.
  if (!tensors.has("decoder.middle.0.residual.0.gamma")) return false;
  if (tensors.has("decoder.upsamples.0.upsamples.0.residual.2.weight")) return false;
  return hasShape(tensors, "encoder.head.2.weight", [KREA2_LATENT_CHANNELS * 2, null, null, null, null])
    && hasShape(tensors, "decoder.head.2.weight", [3, null, null, null, null])
    && tensors.has("decoder.head.0.gamma")
    && tensors.has("encoder.conv1.weight");
}

export function classifyKrea2SafetensorsHeader(header) {
  const tensors = tensorMap(header);
  if (!tensors) return null;
  const matches = [
    ["diffusion_model", isKrea2Diffusion(tensors)],
    ["text_encoder", isKrea2TextEncoder(tensors)],
    ["vae", isKrea2Vae(tensors)],
  ].filter(([, matched]) => matched);
  return matches.length === 1 ? matches[0][0] : null;
}

async function scanKrea2Root(directory, role, relativeDirectory = "", readHeader = readDiffusionModelHeader) {
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
      models.push(...await scanKrea2Root(directory, role, relativePath, readHeader));
      continue;
    }
    if (!entry.isFile() || !componentFileExtensions(role).has(path.extname(entry.name).toLowerCase())) continue;
    try {
      const filePath = path.join(directory, relativePath);
      const [header, fileStat] = await Promise.all([readHeader(filePath), stat(filePath)]);
      if (classifyKrea2SafetensorsHeader(header) !== role) continue;
      const value = relativePath.split(path.sep).join("/");
      models.push({ name: value, value, size: fileStat.size, modifiedAt: fileStat.mtimeMs });
    } catch {
      // One corrupt or disappearing file must not hide the rest of the catalog.
    }
  }
  return models;
}

export async function discoverKrea2Models(roots, readHeader = readDiffusionModelHeader) {
  const entries = await Promise.all(Object.entries(roots).map(async ([role, directory]) => {
    const models = await scanKrea2Root(directory, role, "", readHeader);
    models.sort((first, second) => first.name.localeCompare(second.name, "en"));
    return [role, models];
  }));
  return Object.fromEntries(entries);
}
