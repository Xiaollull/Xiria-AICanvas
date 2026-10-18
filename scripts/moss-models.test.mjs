import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { classifyAnimaSafetensorsHeader } from "./anima-models.mjs";
import { classifyFluxSafetensorsHeader } from "./flux-models.mjs";
import { classifyFlux2SafetensorsHeader } from "./flux2-models.mjs";
import { classifyKrea2SafetensorsHeader } from "./krea2-models.mjs";
import {
  MOSS_RUNTIME_ARTIFACTS,
  bundledMossTokenizerDirectory,
  classifyMossSafetensorsHeader,
  discoverMossModels,
  mossRuntimeArtifactStatuses,
} from "./moss-models.mjs";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const tensor = (shape = [1], dtype = "BF16") => ({ dtype, shape });

// The published MOSS-Image shape: 3072 wide, 42 layers, 128 latent channels (FLUX.2's 32 packed
// 2x2), a 4096-wide text projection. Only the tensors the classifier reads are listed.
const DIM = 3072;
const mossDiffusionHeader = ({ prefix = "", layout = "fused", latent = 128, text = 4096 } = {}) => {
  const header = {
    [`${prefix}shared_block_modulation.weight`]: tensor([DIM * 6, DIM]),
    [`${prefix}layers.0.proj.ffn_up_gate.weight`]: tensor([16384, DIM]),
    [`${prefix}video_embedder.weight`]: tensor([DIM, latent]),
    [`${prefix}final_linear_video.weight`]: tensor([latent, DIM]),
    [`${prefix}text_embedder.proj.weight`]: tensor([DIM, text]),
    [`${prefix}text_embedder.blocks.0.qkv.weight`]: tensor([3 * text, text]),
    [`${prefix}video_head_modulation.weight`]: tensor([DIM * 2, DIM]),
    [`${prefix}time_embedding.0.weight`]: tensor([DIM, 256]),
    [`${prefix}layers.0.q_norm.weight`]: tensor([128]),
  };
  if (layout === "fused" || layout === "both") header[`${prefix}layers.0.proj.qkv.weight`] = tensor([3 * DIM, DIM], "I8");
  if (layout === "split" || layout === "both") {
    for (const part of ["q", "k", "v"]) header[`${prefix}layers.0.proj.${part}.weight`] = tensor([DIM, DIM]);
  }
  return header;
};

const qwen35Header = ({ prefix = "model.language_model.", vocabulary = 248320, hidden = 4096, layers = 32 } = {}) => ({
  [`${prefix}embed_tokens.weight`]: tensor([vocabulary, hidden], "I8"),
  [`${prefix}layers.0.linear_attn.in_proj_qkv.weight`]: tensor([8192, hidden], "I8"),
  [`${prefix}layers.0.linear_attn.A_log`]: tensor([32], "F32"),
  [`${prefix}layers.3.self_attn.q_norm.weight`]: tensor([256]),
  [`${prefix}layers.${layers - 1}.post_attention_layernorm.weight`]: tensor([hidden]),
  [`${prefix}norm.weight`]: tensor([hidden]),
  "model.visual.blocks.0.attn.qkv.weight": tensor([3456, 1152]),
});

const flux2VaeHeader = (width = 512) => ({
  "bn.running_mean": tensor([128], "F32"),
  "bn.running_var": tensor([128], "F32"),
  "decoder.conv_in.weight": tensor([width, 32, 3, 3], "F32"),
  "decoder.conv_out.weight": tensor([3, 128, 3, 3], "F32"),
  "encoder.conv_in.weight": tensor([128, 3, 3, 3], "F32"),
  "encoder.conv_out.weight": tensor([64, 512, 3, 3], "F32"),
});

test("classifies each MOSS component from its own tensors", () => {
  for (const prefix of ["", "model.diffusion_model.", "diffusion_model."]) {
    assert.equal(classifyMossSafetensorsHeader(mossDiffusionHeader({ prefix })), "diffusion_model", prefix);
    assert.equal(classifyMossSafetensorsHeader(mossDiffusionHeader({ prefix, layout: "split" })), "diffusion_model", prefix);
  }
  assert.equal(classifyMossSafetensorsHeader(qwen35Header()), "text_encoder");
  assert.equal(classifyMossSafetensorsHeader(qwen35Header({ prefix: "language_model." })), "text_encoder");
  assert.equal(classifyMossSafetensorsHeader(flux2VaeHeader()), "vae");
});

test("refuses near-misses instead of guessing", () => {
  // A file carrying both attention layouts is neither; the backend's strict loader refuses it too.
  assert.equal(classifyMossSafetensorsHeader(mossDiffusionHeader({ layout: "both" })), null);
  assert.equal(classifyMossSafetensorsHeader(mossDiffusionHeader({ latent: 64 })), null);
  assert.equal(classifyMossSafetensorsHeader(mossDiffusionHeader({ text: 2560 })), null);
  // Another Qwen3.5 size, a 33-layer stack and a foreign vocabulary are not MOSS's encoder.
  assert.equal(classifyMossSafetensorsHeader(qwen35Header({ hidden: 2560 })), null);
  assert.equal(classifyMossSafetensorsHeader(qwen35Header({ layers: 33 })), null);
  assert.equal(classifyMossSafetensorsHeader(qwen35Header({ vocabulary: 151936 })), null);
  // The HF MOSS bundle's small decoder (384 wide) is a different autoencoder and is refused.
  assert.equal(classifyMossSafetensorsHeader(flux2VaeHeader(384)), null);
  assert.equal(classifyMossSafetensorsHeader(null), null);
  assert.equal(classifyMossSafetensorsHeader([]), null);
  assert.equal(classifyMossSafetensorsHeader({ "a.weight": { dtype: "F32" } }), null);
});

test("MOSS and the other engines never claim each other's diffusion model or encoder", () => {
  const others = [classifyAnimaSafetensorsHeader, classifyFluxSafetensorsHeader, classifyFlux2SafetensorsHeader, classifyKrea2SafetensorsHeader];
  for (const header of [mossDiffusionHeader(), mossDiffusionHeader({ layout: "split", prefix: "model.diffusion_model." }), qwen35Header()]) {
    for (const classify of others) assert.equal(classify(header), null, classify.name);
  }
  // The FLUX.2 full decoder is one file both engines mount: FLUX.2 and MOSS each list it.
  assert.equal(classifyFlux2SafetensorsHeader(flux2VaeHeader()), "vae");
  assert.equal(classifyMossSafetensorsHeader(flux2VaeHeader()), "vae");
});

test("discovers safetensors per role, recursively, skipping unreadable and non-safetensors files", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "moss-models-"));
  try {
    const diffusion = path.join(root, "diffusion_models");
    const encoders = path.join(root, "text_encoders");
    const vaes = path.join(root, "vae");
    await mkdir(path.join(diffusion, "moss"), { recursive: true });
    await mkdir(encoders, { recursive: true });
    await mkdir(vaes, { recursive: true });
    const headers = new Map();
    const place = async (file, header) => {
      await writeFile(file, "x");
      headers.set(file, header);
    };
    await place(path.join(diffusion, "moss", "sft_int8.safetensors"), mossDiffusionHeader({ prefix: "model.diffusion_model." }));
    await place(path.join(diffusion, "a_base_bf16.safetensors"), mossDiffusionHeader({ layout: "split" }));
    await place(path.join(diffusion, "moss.gguf"), mossDiffusionHeader());
    await place(path.join(diffusion, "broken.safetensors"), null);
    await place(path.join(encoders, "qwen3.5_9b_int8_convrot.safetensors"), qwen35Header());
    await place(path.join(encoders, "stray_vae.safetensors"), flux2VaeHeader());
    await place(path.join(vaes, "flux2-vae.safetensors"), flux2VaeHeader());
    await place(path.join(vaes, "small-decoder.safetensors"), flux2VaeHeader(384));
    const readHeader = async (file) => {
      const header = headers.get(file);
      if (header === null) throw new Error("corrupt");
      return header;
    };
    const catalog = await discoverMossModels({ diffusion_model: diffusion, text_encoder: encoders, vae: vaes, missing: path.join(root, "absent") }, readHeader);
    assert.deepEqual(catalog.diffusion_model.map((item) => item.value), ["a_base_bf16.safetensors", "moss/sft_int8.safetensors"]);
    assert.deepEqual(catalog.text_encoder.map((item) => item.value), ["qwen3.5_9b_int8_convrot.safetensors"]);
    assert.deepEqual(catalog.vae.map((item) => item.value), ["flux2-vae.safetensors"]);
    assert.deepEqual(catalog.missing, []);
    assert.equal(catalog.vae[0].size, 1);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("the bundled tokenizer archive matches its pinned size and digest", async () => {
  const directory = bundledMossTokenizerDirectory(projectRoot);
  const artifact = MOSS_RUNTIME_ARTIFACTS.qwen35_tokenizer;
  const bytes = await readFile(path.join(directory, artifact.filename));
  assert.equal(bytes.length, artifact.size);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), artifact.sha256);
  const hashFile = async ({ path: file }) => createHash("sha256").update(await readFile(file)).digest("hex");
  const statuses = await mossRuntimeArtifactStatuses(directory, hashFile);
  assert.deepEqual(statuses.qwen35_tokenizer, { path: path.join(directory, artifact.filename), installed: true, reason: null });
});

test("a tampered or missing tokenizer archive is reported, never trusted", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "moss-tokenizer-"));
  try {
    assert.equal((await mossRuntimeArtifactStatuses(root)).qwen35_tokenizer.reason, "not installed");
    const artifact = MOSS_RUNTIME_ARTIFACTS.qwen35_tokenizer;
    await writeFile(path.join(root, artifact.filename), Buffer.alloc(artifact.size));
    const statuses = await mossRuntimeArtifactStatuses(root, async () => "0".repeat(64));
    assert.deepEqual([statuses.qwen35_tokenizer.installed, statuses.qwen35_tokenizer.reason], [false, "SHA-256 mismatch"]);
    await writeFile(path.join(root, artifact.filename), "short");
    assert.equal((await mossRuntimeArtifactStatuses(root)).qwen35_tokenizer.reason, "file size mismatch");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
