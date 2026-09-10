import assert from "node:assert/strict";
import test from "node:test";

import { DEFAULT_SETTINGS, normalizedSettings } from "../src/gallery-core.js";

// The Gallery normaliser accepted only SD, iL and Anima. A Flux, FLUX.2 or Krea 2 card was rewritten
// to the fallback engine every time it was loaded -- SD for a card, or whatever the workspace held --
// and applying it then asked that engine for files it never had: "精选卡片的 Anima 扩散模型未安装".

const krea2Card = {
  model: "Krea2",
  diffusionModel: "krea2_turbo_fp8_scaled.safetensors",
  textEncoder: "qwen3vl_4b_fp8_scaled.safetensors",
  vae: "qwen_image_vae.safetensors",
  steps: 8,
  cfg: 1,
};

test("a Krea 2 card keeps its engine and its files", () => {
  const settings = normalizedSettings(krea2Card, DEFAULT_SETTINGS);
  assert.equal(settings.model, "Krea2", "the card was rewritten to another engine");
  assert.equal(settings.diffusionModel, "krea2_turbo_fp8_scaled.safetensors");
  assert.equal(settings.textEncoder, "qwen3vl_4b_fp8_scaled.safetensors");
  assert.equal(settings.vae, "qwen_image_vae.safetensors");
});

test("every engine the workspace runs survives normalisation", () => {
  for (const model of ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"]) {
    assert.equal(normalizedSettings({ model }, DEFAULT_SETTINGS).model, model, model);
  }
  // An engine that does not exist still falls back rather than passing through.
  assert.equal(normalizedSettings({ model: "NotAnEngine" }, DEFAULT_SETTINGS).model, "SD");
});

test("a split-engine card without a sampler gets the split engines' default", () => {
  for (const model of ["Anima", "Flux", "Flux2", "Krea2"]) {
    const settings = normalizedSettings({ model, sampler: "nonsense", scheduler: "nonsense" }, { model });
    assert.equal(settings.sampler, "euler", model);
    assert.equal(settings.scheduler, "simple", model);
  }
  const sd = normalizedSettings({ model: "SD", sampler: "nonsense", scheduler: "nonsense" }, { model: "SD" });
  assert.equal(sd.sampler, "dpmpp_2m");
  assert.equal(sd.scheduler, "karras");
});

test("a card cannot carry guidance its engine refuses", () => {
  assert.equal(normalizedSettings({ model: "Flux", guidance: "pag" }, DEFAULT_SETTINGS).guidance, "none");
  assert.equal(normalizedSettings({ model: "Flux2", guidance: "cfg_zero_star" }, DEFAULT_SETTINGS).guidance, "none");
  assert.equal(normalizedSettings({ model: "Krea2", guidance: "pag" }, DEFAULT_SETTINGS).guidance, "none");
  assert.equal(normalizedSettings({ model: "Krea2", guidance: "cfg_zero_star" }, DEFAULT_SETTINGS).guidance, "cfg_zero_star");
  assert.equal(normalizedSettings({ model: "SD", guidance: "cfg_zero_star" }, DEFAULT_SETTINGS).guidance, "none");
  assert.equal(normalizedSettings({ model: "Anima", guidance: "pag" }, DEFAULT_SETTINGS).guidance, "pag");
});

test("only a FLUX.1 card keeps a second text encoder", () => {
  assert.equal(normalizedSettings({ model: "Flux", textEncoder2: "t5xxl.safetensors" }, DEFAULT_SETTINGS).textEncoder2, "t5xxl.safetensors");
  for (const model of ["Krea2", "Flux2", "Anima", "SD"]) {
    assert.equal(normalizedSettings({ model, textEncoder2: "t5xxl.safetensors" }, DEFAULT_SETTINGS).textEncoder2, "", model);
  }
});
