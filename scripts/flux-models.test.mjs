import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { classifyFluxSafetensorsHeader, discoverFluxModels, normalizeFluxSafetensorsKey } from "./flux-models.mjs";

const tensor = (shape = [1]) => ({ dtype: "F32", shape });

const fluxDiffusionHeader = (prefix = "") => ({
  [`${prefix}double_blocks.0.img_attn.norm.key_norm.scale`]: tensor([128]),
  [`${prefix}img_in.weight`]: tensor([3072, 64]),
  [`${prefix}txt_in.weight`]: tensor([3072, 4096]),
  [`${prefix}vector_in.in_layer.weight`]: tensor([3072, 768]),
  [`${prefix}single_blocks.0.linear1.weight`]: tensor([21504, 3072]),
  [`${prefix}final_layer.linear.weight`]: tensor([64, 3072]),
});

const clipHeader = () => ({
  "text_model.embeddings.token_embedding.weight": tensor([49408, 768]),
  "text_model.encoder.layers.0.mlp.fc1.weight": tensor([3072, 768]),
  "text_model.encoder.layers.11.mlp.fc1.weight": tensor([3072, 768]),
  "text_model.final_layer_norm.weight": tensor([768]),
});

const t5Header = (embedding = "shared.weight") => ({
  [embedding]: tensor([32128, 4096]),
  "encoder.block.0.layer.0.SelfAttention.q.weight": tensor([4096, 4096]),
  "encoder.block.23.layer.1.DenseReluDense.wi_0.weight": tensor([10240, 4096]),
  "encoder.final_layer_norm.weight": tensor([4096]),
});

const fluxVaeHeader = () => ({
  "encoder.conv_in.weight": tensor([128, 3, 3, 3]),
  "encoder.conv_out.weight": tensor([32, 512, 3, 3]),
  "decoder.conv_in.weight": tensor([512, 16, 3, 3]),
  "decoder.conv_out.weight": tensor([3, 128, 3, 3]),
});

test("normalizes the FLUX.1 diffusion wrapper prefixes and nothing else", () => {
  assert.equal(normalizeFluxSafetensorsKey("model.diffusion_model.img_in.weight"), "img_in.weight");
  assert.equal(normalizeFluxSafetensorsKey("diffusion_model.img_in.weight"), "img_in.weight");
  assert.equal(normalizeFluxSafetensorsKey("img_in.weight"), "img_in.weight");
  // Anima's own `net.` prefix is not a FLUX prefix; stripping it would let one engine's checkpoint
  // be offered as the other's.
  assert.equal(normalizeFluxSafetensorsKey("net.blocks.0.weight"), "net.blocks.0.weight");
});

test("classifies each FLUX.1 component from its own tensors", () => {
  assert.equal(classifyFluxSafetensorsHeader(fluxDiffusionHeader()), "diffusion_model");
  assert.equal(classifyFluxSafetensorsHeader(fluxDiffusionHeader("model.diffusion_model.")), "diffusion_model");
  assert.equal(classifyFluxSafetensorsHeader(clipHeader()), "text_encoder");
  assert.equal(classifyFluxSafetensorsHeader(t5Header()), "text_encoder_2");
  assert.equal(classifyFluxSafetensorsHeader(t5Header("encoder.embed_tokens.weight")), "text_encoder_2");
  assert.equal(classifyFluxSafetensorsHeader(fluxVaeHeader()), "vae");
});

test("rejects the neighbouring architectures that share a FLUX.1 signature", () => {
  // FLUX.2 carries the same double-stream key norm but its own global modulation.
  assert.equal(classifyFluxSafetensorsHeader({
    ...fluxDiffusionHeader(),
    "double_stream_modulation_img.lin.weight": tensor([9216, 3072]),
  }), null);
  // Chroma replaces the modulation with a distilled guidance layer.
  assert.equal(classifyFluxSafetensorsHeader({
    ...fluxDiffusionHeader(),
    "distilled_guidance_layer.norms.0.scale": tensor([64]),
  }), null);
  // A four-channel Stable Diffusion autoencoder must never be offered as a FLUX.1 VAE.
  assert.equal(classifyFluxSafetensorsHeader({
    "encoder.conv_in.weight": tensor([128, 3, 3, 3]),
    "encoder.conv_out.weight": tensor([8, 512, 3, 3]),
    "decoder.conv_in.weight": tensor([512, 4, 3, 3]),
    "decoder.conv_out.weight": tensor([3, 128, 3, 3]),
  }), null);
  assert.equal(classifyFluxSafetensorsHeader({}), null);
  assert.equal(classifyFluxSafetensorsHeader(null), null);
});

test("an ambiguous header matching two roles is refused rather than guessed", () => {
  assert.equal(classifyFluxSafetensorsHeader({ ...clipHeader(), ...t5Header() }), null);
});

test("recursively discovers each FLUX.1 role in its configured root", async (context) => {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "xirai-flux-discovery-"));
  context.after(() => rm(temporary, { recursive: true, force: true }));
  const encoders = path.join(temporary, "text_encoders");
  const roots = {
    diffusion_model: path.join(temporary, "diffusion"),
    // Both encoders are listed out of the same directory, which is exactly how ComfyUI ships them.
    text_encoder: encoders,
    text_encoder_2: encoders,
    vae: path.join(temporary, "vae"),
  };
  await Promise.all([
    mkdir(path.join(roots.diffusion_model, "nested"), { recursive: true }),
    mkdir(encoders, { recursive: true }),
    mkdir(roots.vae, { recursive: true }),
  ]);
  await Promise.all([
    writeFile(path.join(roots.diffusion_model, "nested", "flux1-dev.safetensors"), "fixture"),
    writeFile(path.join(roots.diffusion_model, "anima.safetensors"), "fixture"),
    writeFile(path.join(encoders, "clip_l.safetensors"), "fixture"),
    writeFile(path.join(encoders, "t5xxl_fp16.safetensors"), "fixture"),
    writeFile(path.join(roots.vae, "ae.safetensors"), "fixture"),
    writeFile(path.join(roots.vae, "broken.safetensors"), "fixture"),
    writeFile(path.join(roots.diffusion_model, "flux1-dev.ckpt"), "not a safetensors file"),
  ]);

  const headers = new Map([
    ["flux1-dev.safetensors", fluxDiffusionHeader("model.diffusion_model.")],
    ["anima.safetensors", { "net.llm_adapter.embed.weight": tensor([32128, 1024]) }],
    ["clip_l.safetensors", clipHeader()],
    ["t5xxl_fp16.safetensors", t5Header()],
    ["ae.safetensors", fluxVaeHeader()],
  ]);
  const discovered = await discoverFluxModels(roots, async (filePath) => {
    if (path.basename(filePath) === "broken.safetensors") throw new Error("broken");
    return headers.get(path.basename(filePath));
  });

  assert.deepEqual(discovered.diffusion_model.map((model) => model.value), ["nested/flux1-dev.safetensors"]);
  assert.deepEqual(discovered.text_encoder.map((model) => model.value), ["clip_l.safetensors"]);
  assert.deepEqual(discovered.text_encoder_2.map((model) => model.value), ["t5xxl_fp16.safetensors"]);
  assert.deepEqual(discovered.vae.map((model) => model.value), ["ae.safetensors"]);
  assert.ok(discovered.diffusion_model[0].size > 0);
});
