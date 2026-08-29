import { readdir, stat } from "node:fs/promises";
import path from "node:path";

import { readSafetensorsHeader } from "./anima-models.mjs";
import { normalizeFluxSafetensorsKey } from "./flux-models.mjs";

// FLUX.2 mounts three component files out of the same shared directories FLUX.1 and Anima use, so
// the picker cannot ask the user which file is which — it has to know. Every classifier below
// reads the tensors ComfyUI's `model_detection` and `sd.py` read, which is what lets one rule
// cover [dev] and both [klein] text encoders without a filename heuristic.

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

// The DiT consumes the autoencoder's already-packed 128-channel token, and carries no biases
// anywhere. Both are FLUX.2 properties FLUX.1 does not share.
const FLUX2_LATENT_CHANNELS = 128;

function isFlux2Diffusion(tensors) {
  // The shared modulation projections are what `model_detection` keys "flux2" on: FLUX.2 replaced
  // FLUX.1's per-block modulation with one set for the whole double stack and one for the single.
  const modulation = shapeOf(tensors, "double_stream_modulation_img.lin.weight");
  if (!modulation || modulation.length !== 2 || modulation[0] !== modulation[1] * 6) return false;
  const single = shapeOf(tensors, "single_stream_modulation.lin.weight");
  if (!single || single.length !== 2 || single[0] !== single[1] * 3) return false;
  if (tensors.has("img_in.bias")) return false;
  return hasShape(tensors, "double_stream_modulation_txt.lin.weight", [modulation[1] * 6, modulation[1]])
    && hasShape(tensors, "img_in.weight", [modulation[1], FLUX2_LATENT_CHANNELS])
    && hasShape(tensors, "txt_in.weight", [modulation[1], null])
    && tensors.has("double_blocks.0.img_attn.qkv.weight")
    && tensors.has("single_blocks.0.linear1.weight")
    && tensors.has("final_layer.linear.weight");
}

function languageModelPrefix(tensors) {
  for (const prefix of ["model.", ""]) {
    if (tensors.has(`${prefix}layers.0.post_attention_layernorm.weight`)) return prefix;
  }
  return null;
}

// Mistral-Small-3.1-24B for [dev]; Qwen3-4B and Qwen3-8B for [klein]. The hidden sizes are what
// `comfy/sd.py`'s own detection reads, and each is three times the conditioning width the
// transformer expects once the three tapped layers are concatenated.
const FLUX2_MISTRAL_HIDDEN = 5120;
const FLUX2_QWEN_HIDDEN = [2560, 4096];

function isFlux2TextEncoder(tensors) {
  const prefix = languageModelPrefix(tensors);
  if (prefix == null) return false;
  const norm = shapeOf(tensors, `${prefix}layers.0.post_attention_layernorm.weight`);
  if (!norm || norm.length !== 1) return false;
  const hidden = norm[0];
  if (!hasShape(tensors, `${prefix}embed_tokens.weight`, [null, hidden])) return false;
  if (!hasShape(tensors, `${prefix}layers.0.self_attn.q_proj.weight`, [null, hidden])) return false;
  if (!hasShape(tensors, `${prefix}layers.0.mlp.gate_proj.weight`, [null, hidden])) return false;
  // Qwen3 normalises queries and keys per head; Mistral does not. Neither shares a hidden size
  // with Anima's 1024-wide encoder, so the three never collide in one directory.
  if (tensors.has(`${prefix}layers.0.self_attn.q_norm.weight`)) {
    return FLUX2_QWEN_HIDDEN.includes(hidden);
  }
  return hidden === FLUX2_MISTRAL_HIDDEN;
}

function isFlux2Vae(tensors) {
  // Thirty-two latent channels at stride 8, plus the batch-normalisation statistics for the 2x2
  // pack that turns them into the 128-channel stride-16 token. `encoder.conv_out` is 64 wide
  // because it emits mean and log-variance.
  return hasShape(tensors, "bn.running_mean", [FLUX2_LATENT_CHANNELS])
    && hasShape(tensors, "bn.running_var", [FLUX2_LATENT_CHANNELS])
    && hasShape(tensors, "encoder.conv_in.weight", [128, 3, 3, 3])
    && hasShape(tensors, "encoder.conv_out.weight", [64, 512, 3, 3])
    && hasShape(tensors, "decoder.conv_in.weight", [512, 32, 3, 3])
    && hasShape(tensors, "decoder.conv_out.weight", [3, 128, 3, 3]);
}

export function classifyFlux2SafetensorsHeader(header) {
  const tensors = tensorMap(header);
  if (!tensors) return null;
  const matches = [
    ["diffusion_model", isFlux2Diffusion(tensors)],
    ["text_encoder", isFlux2TextEncoder(tensors)],
    ["vae", isFlux2Vae(tensors)],
  ].filter(([, matched]) => matched);
  return matches.length === 1 ? matches[0][0] : null;
}

async function scanFlux2Root(directory, role, relativeDirectory = "", readHeader = readSafetensorsHeader) {
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
      models.push(...await scanFlux2Root(directory, role, relativePath, readHeader));
      continue;
    }
    if (!entry.isFile() || path.extname(entry.name).toLowerCase() !== ".safetensors") continue;
    try {
      const filePath = path.join(directory, relativePath);
      const [header, fileStat] = await Promise.all([readHeader(filePath), stat(filePath)]);
      if (classifyFlux2SafetensorsHeader(header) !== role) continue;
      const value = relativePath.split(path.sep).join("/");
      models.push({ name: value, value, size: fileStat.size, modifiedAt: fileStat.mtimeMs });
    } catch {
      // One corrupt or disappearing file must not hide the rest of the catalog.
    }
  }
  return models;
}

export async function discoverFlux2Models(roots, readHeader = readSafetensorsHeader) {
  const entries = await Promise.all(Object.entries(roots).map(async ([role, directory]) => {
    const models = await scanFlux2Root(directory, role, "", readHeader);
    models.sort((first, second) => first.name.localeCompare(second.name, "en"));
    return [role, models];
  }));
  return Object.fromEntries(entries);
}
