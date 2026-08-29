import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { classifyFluxSafetensorsHeader } from "./flux-models.mjs";
import { classifyFlux2SafetensorsHeader } from "./flux2-models.mjs";
import { classifyKrea2SafetensorsHeader, discoverKrea2Models } from "./krea2-models.mjs";

const tensor = (shape = [1]) => ({ dtype: "F32", shape });

// The published Krea 2 shape: 6144 wide, 48 heads of 128, 12 key/value heads, 16 latent channels
// packed 2x2 into the 64-wide input projection.
const FEATURES = 6144;
const HEAD_DIM = 128;
const HEADS = 48;
const KVHEADS = 12;
const TAPS = 12;
const TXTDIM = 2560;

const krea2DiffusionHeader = (prefix = "") => ({
  [`${prefix}txtfusion.projector.weight`]: tensor([1, TAPS]),
  [`${prefix}txtfusion.layerwise_blocks.0.prenorm.scale`]: tensor([TXTDIM]),
  [`${prefix}txtfusion.refiner_blocks.0.prenorm.scale`]: tensor([TXTDIM]),
  [`${prefix}first.weight`]: tensor([FEATURES, 64]),
  [`${prefix}blocks.0.attn.wq.weight`]: tensor([HEAD_DIM * HEADS, FEATURES]),
  [`${prefix}blocks.0.attn.wk.weight`]: tensor([HEAD_DIM * KVHEADS, FEATURES]),
  [`${prefix}blocks.0.attn.gate.weight`]: tensor([FEATURES, FEATURES]),
  [`${prefix}blocks.0.mod.lin`]: tensor([FEATURES * 6]),
  [`${prefix}last.linear.weight`]: tensor([64, FEATURES]),
});

const qwen3vlHeader = ({ hidden = TXTDIM, layers = 36, prefix = "model.language_model.", deepstack = true } = {}) => {
  const header = {
    [`${prefix}embed_tokens.weight`]: tensor([151936, hidden]),
    [`${prefix}layers.0.post_attention_layernorm.weight`]: tensor([hidden]),
    [`${prefix}layers.0.self_attn.q_proj.weight`]: tensor([4096, hidden]),
    [`${prefix}layers.0.self_attn.q_norm.weight`]: tensor([128]),
    [`${prefix}layers.0.mlp.gate_proj.weight`]: tensor([9728, hidden]),
    [`${prefix}layers.${layers - 1}.post_attention_layernorm.weight`]: tensor([hidden]),
  };
  if (deepstack) header["model.visual.deepstack_merger_list.0.norm.weight"] = tensor([hidden]);
  return header;
};

// Wan 2.1's autoencoder, which is the same file ComfyUI publishes as the Qwen-Image VAE.
const wan21VaeHeader = () => ({
  "decoder.middle.0.residual.0.gamma": tensor([384]),
  "decoder.head.0.gamma": tensor([96]),
  "decoder.head.2.weight": tensor([3, 96, 3, 3, 3]),
  "encoder.head.2.weight": tensor([32, 384, 3, 3, 3]),
  "encoder.conv1.weight": tensor([96, 3, 3, 3, 3]),
});

test("classifies each Krea 2 component from its own tensors", () => {
  assert.equal(classifyKrea2SafetensorsHeader(krea2DiffusionHeader()), "diffusion_model");
  assert.equal(classifyKrea2SafetensorsHeader(krea2DiffusionHeader("model.diffusion_model.")), "diffusion_model");
  assert.equal(classifyKrea2SafetensorsHeader(qwen3vlHeader()), "text_encoder");
  assert.equal(classifyKrea2SafetensorsHeader(wan21VaeHeader()), "vae");
});

test("a text encoder is accepted under either wrapper prefix ComfyUI accepts", () => {
  assert.equal(classifyKrea2SafetensorsHeader(qwen3vlHeader({ prefix: "model." })), "text_encoder");
  assert.equal(classifyKrea2SafetensorsHeader(qwen3vlHeader({ prefix: "" })), "text_encoder");
});

test("the vision tower is what separates Krea 2's encoder from FLUX.2 [klein]'s", () => {
  // Both are 2560 wide with per-head query norms; only Qwen3-VL carries a DeepStack merger, which
  // is the same tensor `comfy/sd.py::detect_te_model` keys on. The prefix here is `model.` because
  // that is the one both classifiers read, which is what makes this a claim about one file.
  const klein = qwen3vlHeader({ deepstack: false, prefix: "model." });
  assert.equal(classifyKrea2SafetensorsHeader(klein), null);
  assert.equal(classifyFlux2SafetensorsHeader(klein), "text_encoder");
});

test("an encoder shallower than the deepest conditioning tap is refused", () => {
  // The taps reach layer 35, so a 24-layer model cannot produce Krea 2's conditioning.
  assert.equal(classifyKrea2SafetensorsHeader(qwen3vlHeader({ layers: 24 })), null);
});

test("neither Flux generation claims a Krea 2 file, and Krea 2 claims neither of theirs", () => {
  const diffusion = krea2DiffusionHeader();
  assert.equal(classifyFluxSafetensorsHeader(diffusion), null);
  assert.equal(classifyFlux2SafetensorsHeader(diffusion), null);

  const flux1Diffusion = {
    "double_blocks.0.img_attn.norm.key_norm.scale": tensor([128]),
    "img_in.weight": tensor([3072, 64]),
    "txt_in.weight": tensor([3072, 4096]),
    "vector_in.in_layer.weight": tensor([3072, 768]),
    "single_blocks.0.linear1.weight": tensor([21504, 3072]),
    "final_layer.linear.weight": tensor([64, 3072]),
  };
  assert.equal(classifyKrea2SafetensorsHeader(flux1Diffusion), null);
  // FLUX.2's 32-channel autoencoder shares no rule with Wan 2.1's.
  assert.equal(classifyKrea2SafetensorsHeader({
    "bn.running_mean": tensor([128]),
    "bn.running_var": tensor([128]),
    "encoder.conv_in.weight": tensor([128, 3, 3, 3]),
    "encoder.conv_out.weight": tensor([64, 512, 3, 3]),
    "decoder.conv_in.weight": tensor([512, 32, 3, 3]),
    "decoder.conv_out.weight": tensor([3, 128, 3, 3]),
  }), null);
});

test("a Wan 2.2 autoencoder is not a Krea 2 one", () => {
  // Krea 2's latent format is Wan 2.1: 16 channels at stride 8. Wan 2.2 is 48 at stride 16, and
  // its upsample stack is the tensor `comfy/sd.py` tells the two apart by.
  assert.equal(classifyKrea2SafetensorsHeader({
    ...wan21VaeHeader(),
    "decoder.upsamples.0.upsamples.0.residual.2.weight": tensor([1]),
  }), null);
});

test("a projector collapsing the wrong number of taps is not Krea 2", () => {
  assert.equal(classifyKrea2SafetensorsHeader({
    ...krea2DiffusionHeader(),
    "txtfusion.projector.weight": tensor([1, 3]),
  }), null);
  assert.equal(classifyKrea2SafetensorsHeader({}), null);
  assert.equal(classifyKrea2SafetensorsHeader(null), null);
});

test("recursively discovers each Krea 2 role in its configured root", async (context) => {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "xirai-krea2-discovery-"));
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
    writeFile(path.join(roots.diffusion_model, "nested", "krea2_raw_bf16.safetensors"), "fixture"),
    writeFile(path.join(roots.diffusion_model, "flux2-dev.safetensors"), "fixture"),
    writeFile(path.join(roots.text_encoder, "qwen3vl_4b_bf16.safetensors"), "fixture"),
    writeFile(path.join(roots.text_encoder, "qwen3_4b_klein.safetensors"), "fixture"),
    writeFile(path.join(roots.vae, "qwen_image_vae.safetensors"), "fixture"),
    writeFile(path.join(roots.vae, "broken.safetensors"), "fixture"),
    writeFile(path.join(roots.diffusion_model, "krea2_raw_bf16.ckpt"), "not a safetensors file"),
  ]);

  const headers = new Map([
    ["krea2_raw_bf16.safetensors", krea2DiffusionHeader("model.diffusion_model.")],
    ["flux2-dev.safetensors", {
      "double_stream_modulation_img.lin.weight": tensor([6144 * 6, 6144]),
      "img_in.weight": tensor([6144, 128]),
    }],
    ["qwen3vl_4b_bf16.safetensors", qwen3vlHeader()],
    ["qwen3_4b_klein.safetensors", qwen3vlHeader({ deepstack: false })],
    ["qwen_image_vae.safetensors", wan21VaeHeader()],
  ]);
  const discovered = await discoverKrea2Models(roots, async (filePath) => {
    if (path.basename(filePath) === "broken.safetensors") throw new Error("broken");
    return headers.get(path.basename(filePath));
  });

  assert.deepEqual(discovered.diffusion_model.map((model) => model.value), ["nested/krea2_raw_bf16.safetensors"]);
  assert.deepEqual(discovered.text_encoder.map((model) => model.value), ["qwen3vl_4b_bf16.safetensors"]);
  assert.deepEqual(discovered.vae.map((model) => model.value), ["qwen_image_vae.safetensors"]);
  assert.ok(discovered.diffusion_model[0].size > 0);
});
