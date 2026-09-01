import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { classifyFluxSafetensorsHeader } from "./flux-models.mjs";
import { classifyFlux2SafetensorsHeader, discoverFlux2Models } from "./flux2-models.mjs";

const tensor = (shape = [1]) => ({ dtype: "F32", shape });

const HIDDEN = 6144;
const MLP = HIDDEN * 3;

const flux2DiffusionHeader = (prefix = "") => ({
  [`${prefix}double_stream_modulation_img.lin.weight`]: tensor([HIDDEN * 6, HIDDEN]),
  [`${prefix}double_stream_modulation_txt.lin.weight`]: tensor([HIDDEN * 6, HIDDEN]),
  [`${prefix}single_stream_modulation.lin.weight`]: tensor([HIDDEN * 3, HIDDEN]),
  [`${prefix}img_in.weight`]: tensor([HIDDEN, 128]),
  [`${prefix}txt_in.weight`]: tensor([HIDDEN, 15360]),
  [`${prefix}double_blocks.0.img_attn.qkv.weight`]: tensor([HIDDEN * 3, HIDDEN]),
  [`${prefix}double_blocks.0.img_attn.norm.key_norm.scale`]: tensor([128]),
  [`${prefix}single_blocks.0.linear1.weight`]: tensor([HIDDEN * 3 + MLP * 2, HIDDEN]),
  [`${prefix}final_layer.linear.weight`]: tensor([128, HIDDEN]),
});

const languageModelHeader = ({ hidden, qwen = false, prefix = "model." } = {}) => ({
  [`${prefix}embed_tokens.weight`]: tensor([qwen ? 151936 : 131072, hidden]),
  [`${prefix}layers.0.post_attention_layernorm.weight`]: tensor([hidden]),
  [`${prefix}layers.0.self_attn.q_proj.weight`]: tensor([4096, hidden]),
  [`${prefix}layers.0.mlp.gate_proj.weight`]: tensor([hidden * 4, hidden]),
  ...(qwen ? { [`${prefix}layers.0.self_attn.q_norm.weight`]: tensor([128]) } : {}),
});

const flux2VaeHeader = () => ({
  "bn.running_mean": tensor([128]),
  "bn.running_var": tensor([128]),
  "encoder.conv_in.weight": tensor([128, 3, 3, 3]),
  "encoder.conv_out.weight": tensor([64, 512, 3, 3]),
  "decoder.conv_in.weight": tensor([512, 32, 3, 3]),
  "decoder.conv_out.weight": tensor([3, 128, 3, 3]),
});

test("classifies each FLUX.2 component from its own tensors", () => {
  assert.equal(classifyFlux2SafetensorsHeader(flux2DiffusionHeader()), "diffusion_model");
  assert.equal(classifyFlux2SafetensorsHeader(flux2DiffusionHeader("model.diffusion_model.")), "diffusion_model");
  // [dev] conditions on Mistral-Small-3.1-24B; [klein] on Qwen3-4B or Qwen3-8B.
  assert.equal(classifyFlux2SafetensorsHeader(languageModelHeader({ hidden: 5120 })), "text_encoder");
  assert.equal(classifyFlux2SafetensorsHeader(languageModelHeader({ hidden: 2560, qwen: true })), "text_encoder");
  assert.equal(classifyFlux2SafetensorsHeader(languageModelHeader({ hidden: 4096, qwen: true })), "text_encoder");
  assert.equal(classifyFlux2SafetensorsHeader(flux2VaeHeader()), "vae");
});

test("a text encoder is accepted with or without the wrapper prefix", () => {
  assert.equal(classifyFlux2SafetensorsHeader(languageModelHeader({ hidden: 5120, prefix: "" })), "text_encoder");
});

test("the two Flux generations never claim each other's files", () => {
  const flux1Diffusion = {
    "double_blocks.0.img_attn.norm.key_norm.scale": tensor([128]),
    "img_in.weight": tensor([3072, 64]),
    "txt_in.weight": tensor([3072, 4096]),
    "vector_in.in_layer.weight": tensor([3072, 768]),
    "single_blocks.0.linear1.weight": tensor([21504, 3072]),
    "final_layer.linear.weight": tensor([64, 3072]),
  };
  assert.equal(classifyFlux2SafetensorsHeader(flux1Diffusion), null);
  assert.equal(classifyFluxSafetensorsHeader(flux2DiffusionHeader()), null);

  // FLUX.1's 16-channel autoencoder and FLUX.2's 32-channel one differ in every rule that matters.
  const flux1Vae = {
    "encoder.conv_in.weight": tensor([128, 3, 3, 3]),
    "encoder.conv_out.weight": tensor([32, 512, 3, 3]),
    "decoder.conv_in.weight": tensor([512, 16, 3, 3]),
    "decoder.conv_out.weight": tensor([3, 128, 3, 3]),
  };
  assert.equal(classifyFlux2SafetensorsHeader(flux1Vae), null);
  assert.equal(classifyFluxSafetensorsHeader(flux2VaeHeader()), null);

  // FLUX.1's T5-XXL and CLIP-L are not language models with decoder layers.
  assert.equal(classifyFlux2SafetensorsHeader({
    "shared.weight": tensor([32128, 4096]),
    "encoder.block.0.layer.0.SelfAttention.q.weight": tensor([4096, 4096]),
  }), null);
});

test("neighbouring architectures in the same directory are refused", () => {
  // A FLUX.2 diffusion model carries no biases; a checkpoint that does is something else.
  assert.equal(classifyFlux2SafetensorsHeader({
    ...flux2DiffusionHeader(),
    "img_in.bias": tensor([HIDDEN]),
  }), null);
  // Anima's text encoder is a 1024-wide model; offering it as a FLUX.2 encoder would load a
  // conditioning three times the wrong width.
  assert.equal(classifyFlux2SafetensorsHeader(languageModelHeader({ hidden: 1024 })), null);
  // Qwen2.5 has no per-head query norm, so a 2560-wide model without one is not a Klein encoder.
  assert.equal(classifyFlux2SafetensorsHeader(languageModelHeader({ hidden: 2560 })), null);
  // A modulation projection that is not six times the hidden size is not FLUX.2's.
  assert.equal(classifyFlux2SafetensorsHeader({
    ...flux2DiffusionHeader(),
    "double_stream_modulation_img.lin.weight": tensor([HIDDEN * 3, HIDDEN]),
  }), null);
  assert.equal(classifyFlux2SafetensorsHeader({}), null);
  assert.equal(classifyFlux2SafetensorsHeader(null), null);
});

test("recursively discovers each FLUX.2 role in its configured root", async (context) => {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "xirai-flux2-discovery-"));
  context.after(() => rm(temporary, { recursive: true, force: true }));
  const roots = {
    diffusion_model: path.join(temporary, "diffusion"),
    text_encoder: path.join(temporary, "text_encoders"),
    vae: path.join(temporary, "vae"),
  };
  await Promise.all([
    mkdir(path.join(roots.diffusion_model, "nested"), { recursive: true }),
    mkdir(roots.text_encoder, { recursive: true }),
    mkdir(roots.vae, { recursive: true }),
  ]);
  await Promise.all([
    writeFile(path.join(roots.diffusion_model, "nested", "flux2-dev.safetensors"), "fixture"),
    writeFile(path.join(roots.diffusion_model, "flux1-dev.safetensors"), "fixture"),
    writeFile(path.join(roots.text_encoder, "flux2-text-encoder.safetensors"), "fixture"),
    writeFile(path.join(roots.text_encoder, "clip_l.safetensors"), "fixture"),
    writeFile(path.join(roots.vae, "flux2-vae.safetensors"), "fixture"),
    writeFile(path.join(roots.vae, "broken.safetensors"), "fixture"),
    writeFile(path.join(roots.diffusion_model, "flux2-dev.ckpt"), "not a safetensors file"),
    writeFile(path.join(roots.diffusion_model, "flux2-dev-Q4_K.gguf"), "fixture"),
    writeFile(path.join(roots.text_encoder, "flux2-text-encoder-Q8_0.gguf"), "fixture"),
  ]);

  const headers = new Map([
    ["flux2-dev.safetensors", flux2DiffusionHeader("model.diffusion_model.")],
    ["flux2-dev-Q4_K.gguf", flux2DiffusionHeader()],
    ["flux2-text-encoder-Q8_0.gguf", languageModelHeader({ hidden: 5120 })],
    ["flux1-dev.safetensors", {
      "double_blocks.0.img_attn.norm.key_norm.scale": tensor([128]),
      "img_in.weight": tensor([3072, 64]),
    }],
    ["flux2-text-encoder.safetensors", languageModelHeader({ hidden: 5120 })],
    ["clip_l.safetensors", { "text_model.final_layer_norm.weight": tensor([768]) }],
    ["flux2-vae.safetensors", flux2VaeHeader()],
  ]);
  const discovered = await discoverFlux2Models(roots, async (filePath) => {
    if (path.basename(filePath) === "broken.safetensors") throw new Error("broken");
    return headers.get(path.basename(filePath));
  });

  assert.deepEqual(discovered.diffusion_model.map((model) => model.value), ["flux2-dev-Q4_K.gguf", "nested/flux2-dev.safetensors"]);
  // The encoder GGUF classifies, and is still left out: only the diffusion slot reads one.
  assert.ok(!discovered.text_encoder.some((model) => model.value.endsWith(".gguf")));
  assert.deepEqual(discovered.text_encoder.map((model) => model.value), ["flux2-text-encoder.safetensors"]);
  assert.deepEqual(discovered.vae.map((model) => model.value), ["flux2-vae.safetensors"]);
  assert.ok(discovered.diffusion_model[0].size > 0);
});
