import { readdir, stat } from "node:fs/promises";
import path from "node:path";

import { animaRuntimeArtifactStatuses } from "./anima-models.mjs";
import { readDiffusionModelHeader } from "./gguf-header.mjs";

// MOSS-Image mounts three component files out of the shared component directories every native
// engine uses, so the picker cannot ask which file is which. Each classifier reads the tensors the
// backend's strict loader requires (`backend/moss_model.py`, `moss_text_encoder.py`,
// `moss_pipeline.vae_signature`), which keeps MOSS's files apart from the other six engines' without a
// filename heuristic. Base and SFT share every key and shape; which of the two a file is, the backend
// decides from tensor content at load, never from its name.

const MOSS_PREFIXES = ["model.diffusion_model.", "diffusion_model."];
const MOSS_TEXT_DIM = 4096;
const MOSS_LATENT_CHANNELS = 128;
// Qwen3.5-9B: 4096 wide, 32 hybrid layers, the 248 320-row vocabulary MOSS's tokenizer addresses.
const QWEN35_9B_HIDDEN = 4096;
const QWEN35_VOCABULARY = 248320;
// The FLUX.2 full decoder MOSS validates; the HF bundle's small decoder is 384 wide and refused.
const FLUX2_FULL_DECODER_WIDTH = 512;

export const MOSS_RUNTIME_ARTIFACTS = Object.freeze({
  qwen35_tokenizer: Object.freeze({
    filename: "moss-qwen35-tokenizer.json.gz",
    size: 3760595,
    sha256: "70765620bd5db1fd13e40a868c10419c7c8de7dbb2b5abd3a43d69aaa6c6f5b0",
  }),
});

export function bundledMossTokenizerDirectory(projectRoot) {
  return path.join(projectRoot, "backend", "resources", "moss-tokenizers");
}

export function mossRuntimeArtifactStatuses(directory, hashFile) {
  return animaRuntimeArtifactStatuses(directory, hashFile, MOSS_RUNTIME_ARTIFACTS);
}

function tensorMap(header, prefixes = MOSS_PREFIXES) {
  if (!header || typeof header !== "object" || Array.isArray(header)) return null;
  const tensors = new Map();
  for (const [key, descriptor] of Object.entries(header)) {
    if (key === "__metadata__") continue;
    if (!descriptor || typeof descriptor !== "object" || !Array.isArray(descriptor.shape)) return null;
    const prefix = prefixes.find((candidate) => key.startsWith(candidate));
    tensors.set(prefix ? key.slice(prefix.length) : key, descriptor);
  }
  return tensors.size ? tensors : null;
}

function shapeOf(tensors, key) {
  const shape = tensors.get(key)?.shape;
  return Array.isArray(shape) ? shape : null;
}

function hasShape(tensors, key, shape) {
  const actual = shapeOf(tensors, key);
  return Array.isArray(actual) && actual.length === shape.length
    && actual.every((dimension, index) => shape[index] === null || dimension === shape[index]);
}

function isMossDiffusion(tensors) {
  // ComfyUI PR #17095's `model_detection` keys MOSS on these two; the rest is the plugin's
  // `model_config` signature, read under the fused (Comfy-Org) or split (HF) attention layout.
  if (!tensors.has("shared_block_modulation.weight") || !tensors.has("layers.0.proj.ffn_up_gate.weight")) return false;
  const embed = shapeOf(tensors, "video_embedder.weight");
  if (!embed || embed.length !== 2 || embed[1] !== MOSS_LATENT_CHANNELS) return false;
  const dim = embed[0];
  const fused = tensors.has("layers.0.proj.qkv.weight");
  const split = ["q", "k", "v"].every((part) => tensors.has(`layers.0.proj.${part}.weight`));
  if (fused === split) return false;
  return hasShape(tensors, "final_linear_video.weight", [MOSS_LATENT_CHANNELS, dim])
    && hasShape(tensors, "text_embedder.proj.weight", [dim, MOSS_TEXT_DIM])
    && tensors.has("text_embedder.blocks.0.qkv.weight")
    && hasShape(tensors, "video_head_modulation.weight", [dim * 2, dim])
    && tensors.has("time_embedding.0.weight")
    && tensors.has("layers.0.q_norm.weight");
}

function qwen35Prefix(tensors) {
  for (const prefix of ["model.language_model.", "language_model."]) {
    if (tensors.has(`${prefix}embed_tokens.weight`)) return prefix;
  }
  return null;
}

function isMossTextEncoder(tensors) {
  const prefix = qwen35Prefix(tensors);
  if (prefix == null) return false;
  // The hybrid layout only Qwen3.5 has: Gated DeltaNet on layer 0, gated full attention on layer 3.
  return hasShape(tensors, `${prefix}embed_tokens.weight`, [QWEN35_VOCABULARY, QWEN35_9B_HIDDEN])
    && tensors.has(`${prefix}layers.0.linear_attn.in_proj_qkv.weight`)
    && tensors.has(`${prefix}layers.0.linear_attn.A_log`)
    && tensors.has(`${prefix}layers.3.self_attn.q_norm.weight`)
    && tensors.has(`${prefix}layers.31.post_attention_layernorm.weight`)
    && !tensors.has(`${prefix}layers.32.post_attention_layernorm.weight`)
    && hasShape(tensors, `${prefix}norm.weight`, [QWEN35_9B_HIDDEN]);
}

function isMossVae(tensors) {
  return hasShape(tensors, "bn.running_mean", [MOSS_LATENT_CHANNELS])
    && hasShape(tensors, "bn.running_var", [MOSS_LATENT_CHANNELS])
    && hasShape(tensors, "decoder.conv_in.weight", [FLUX2_FULL_DECODER_WIDTH, 32, 3, 3]);
}

export function classifyMossSafetensorsHeader(header) {
  const tensors = tensorMap(header);
  if (!tensors) return null;
  const matches = [
    ["diffusion_model", isMossDiffusion(tensors)],
    ["text_encoder", isMossTextEncoder(tensors)],
    ["vae", isMossVae(tensors)],
  ].filter(([, matched]) => matched);
  return matches.length === 1 ? matches[0][0] : null;
}

async function scanMossRoot(directory, role, relativeDirectory = "", readHeader = readDiffusionModelHeader) {
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
      models.push(...await scanMossRoot(directory, role, relativePath, readHeader));
      continue;
    }
    // MOSS components are safetensors only: no GGUF build exists, and the loader refuses one.
    if (!entry.isFile() || path.extname(entry.name).toLowerCase() !== ".safetensors") continue;
    try {
      const filePath = path.join(directory, relativePath);
      const [header, fileStat] = await Promise.all([readHeader(filePath), stat(filePath)]);
      if (classifyMossSafetensorsHeader(header) !== role) continue;
      const value = relativePath.split(path.sep).join("/");
      models.push({ name: value, value, size: fileStat.size, modifiedAt: fileStat.mtimeMs });
    } catch {
      // One corrupt or disappearing file must not hide the rest of the catalog.
    }
  }
  return models;
}

export async function discoverMossModels(roots, readHeader = readDiffusionModelHeader) {
  const entries = await Promise.all(Object.entries(roots).map(async ([role, directory]) => {
    const models = await scanMossRoot(directory, role, "", readHeader);
    models.sort((first, second) => first.name.localeCompare(second.name, "en"));
    return [role, models];
  }));
  return Object.fromEntries(entries);
}
