import { readdir, stat } from "node:fs/promises";
import path from "node:path";

import { componentFileExtensions, readDiffusionModelHeader } from "./gguf-header.mjs";

// FLUX.1 mounts four component files out of one shared set of directories, so the picker cannot
// ask the user which file is which — it has to know. Every classifier below reads the same tensors
// ComfyUI's `model_detection` reads, which is what lets one rule cover [dev], [schnell],
// [Krea dev] and Fill without a filename heuristic.
const fluxPrefixes = ["model.diffusion_model.", "diffusion_model."];

export function normalizeFluxSafetensorsKey(key) {
  for (const prefix of fluxPrefixes) {
    if (key.startsWith(prefix)) return key.slice(prefix.length);
  }
  return key;
}

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

function hasShape(tensors, key, shape) {
  const actual = tensors.get(key)?.shape;
  return Array.isArray(actual)
    && actual.length === shape.length
    && actual.every((dimension, index) => shape[index] === null || dimension === shape[index]);
}

function isFluxDiffusion(tensors) {
  // The double-stream key norm plus an `img_in` projection is ComfyUI's own FLUX signature. The
  // two exclusions below are what keep FLUX.2 and Chroma — which share that signature — out.
  if (!tensors.has("double_blocks.0.img_attn.norm.key_norm.scale")) return false;
  if (tensors.has("double_stream_modulation_img.lin.weight")) return false;
  if ([...tensors.keys()].some((key) => key.startsWith("distilled_guidance_layer."))) return false;
  return hasShape(tensors, "img_in.weight", [null, null])
    && hasShape(tensors, "txt_in.weight", [null, 4096])
    && hasShape(tensors, "vector_in.in_layer.weight", [null, 768])
    && tensors.has("single_blocks.0.linear1.weight")
    && tensors.has("final_layer.linear.weight");
}

function isClipLTextEncoder(tensors) {
  return hasShape(tensors, "text_model.encoder.layers.0.mlp.fc1.weight", [3072, 768])
    && hasShape(tensors, "text_model.encoder.layers.11.mlp.fc1.weight", [3072, 768])
    && hasShape(tensors, "text_model.final_layer_norm.weight", [768])
    && hasShape(tensors, "text_model.embeddings.token_embedding.weight", [49408, 768]);
}

function isT5XxlTextEncoder(tensors) {
  const embedding = tensors.has("shared.weight") ? "shared.weight" : "encoder.embed_tokens.weight";
  return hasShape(tensors, embedding, [32128, 4096])
    && hasShape(tensors, "encoder.block.0.layer.0.SelfAttention.q.weight", [4096, 4096])
    && hasShape(tensors, "encoder.block.23.layer.1.DenseReluDense.wi_0.weight", [10240, 4096])
    && hasShape(tensors, "encoder.final_layer_norm.weight", [4096]);
}

function isFluxVae(tensors) {
  // Sixteen latent channels through a 2D autoencoder. The `encoder.conv_out` width is 32 because
  // it emits mean and log-variance; the Anima VAE in the same folder is 3D and never matches.
  return hasShape(tensors, "encoder.conv_in.weight", [128, 3, 3, 3])
    && hasShape(tensors, "encoder.conv_out.weight", [32, 512, 3, 3])
    && hasShape(tensors, "decoder.conv_in.weight", [512, 16, 3, 3])
    && hasShape(tensors, "decoder.conv_out.weight", [3, 128, 3, 3]);
}

export function classifyFluxSafetensorsHeader(header) {
  const tensors = tensorMap(header);
  if (!tensors) return null;
  const matches = [
    ["diffusion_model", isFluxDiffusion(tensors)],
    ["text_encoder", isClipLTextEncoder(tensors)],
    ["text_encoder_2", isT5XxlTextEncoder(tensors)],
    ["vae", isFluxVae(tensors)],
  ].filter(([, matched]) => matched);
  return matches.length === 1 ? matches[0][0] : null;
}

async function scanFluxRoot(directory, role, relativeDirectory = "", readHeader = readDiffusionModelHeader) {
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
      models.push(...await scanFluxRoot(directory, role, relativePath, readHeader));
      continue;
    }
    if (!entry.isFile() || !componentFileExtensions(role).has(path.extname(entry.name).toLowerCase())) continue;
    try {
      const filePath = path.join(directory, relativePath);
      const [header, fileStat] = await Promise.all([readHeader(filePath), stat(filePath)]);
      if (classifyFluxSafetensorsHeader(header) !== role) continue;
      const value = relativePath.split(path.sep).join("/");
      models.push({ name: value, value, size: fileStat.size, modifiedAt: fileStat.mtimeMs });
    } catch {
      // One corrupt or disappearing file must not hide the rest of the catalog.
    }
  }
  return models;
}

export async function discoverFluxModels(roots, readHeader = readDiffusionModelHeader) {
  const entries = await Promise.all(Object.entries(roots).map(async ([role, directory]) => {
    const models = await scanFluxRoot(directory, role, "", readHeader);
    models.sort((first, second) => first.name.localeCompare(second.name, "en"));
    return [role, models];
  }));
  return Object.fromEntries(entries);
}
