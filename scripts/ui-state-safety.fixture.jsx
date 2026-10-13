import React, { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "../src/styles.css";
import { SETTINGS_ENGINES } from "../src/engine-settings.js";

// Native HTTP is permitted ONLY for the isolated fixture's own UI owner.
// No formal backend, model/library route, generation, or external service is reachable.
const nativeFetch = window.fetch.bind(window);
const reply = (value, status = 200) => new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });
window.__fixtureOtherCalls = [];
window.__uiCalls = [];
window.__manualOpenCalls = [];
window.open = (...args) => { window.__manualOpenCalls.push(args); return null; };
const catalogGate = new URLSearchParams(location.search).has("hold-catalog")
  ? new Promise((resolve) => { window.__releaseCatalog = resolve; })
  : Promise.resolve();
window.__localPromptWrites = [];
const originalSet = Storage.prototype.setItem;
Storage.prototype.setItem = function (key, value) {
  if (key === "xirai-assistant-prompt-snapshot-v1") window.__localPromptWrites.push(value);
  return originalSet.call(this, key, value);
};
if (!localStorage.getItem("xirai-assistant-prompt-snapshot-v1")) originalSet.call(localStorage, "xirai-assistant-prompt-snapshot-v1", JSON.stringify({ positive: "LOCAL_KEEP", negative: "LOCAL_KEEP_NEG" }));
window.__localPromptBefore = localStorage.getItem("xirai-assistant-prompt-snapshot-v1");
window.fetch = async (input, options = {}) => {
  const url = typeof input === "string" ? input : input.url, method = options.method || "GET";
  if (url === "/api/ui-state") { window.__uiCalls.push({ method }); return nativeFetch(url, options); }
  const path = new URL(url, location.origin).pathname;
  window.__fixtureOtherCalls.push({ path, method });
  if (method !== "GET") return reply({ error: "No generation/library writes in this fixture" }, 409);
  const engine = new URL(url, location.origin).searchParams.get("engine") || "Anima";
  if (path === "/api/models") {
    await catalogGate;
    const choices = (name) => [{ value: `${engine}-${name}.safetensors`, name: `Fixture ${engine}` }];
    return reply(["SD", "iL"].includes(engine) ? { engine, model_type: "checkpoint", models: choices("checkpoint") } : { engine, model_type: "split", assets: Object.fromEntries(["diffusion_model", "text_encoder", "text_encoder_2", "vae"].map((k, i) => [k, { models: choices(["diffusion", "encoder", "encoder2", "vae"][i]) }])) });
  }
  if (path === "/api/loras") return reply({ engine, categories: [], directory: "fixture-only" });
  if (path === "/api/inference/health") return reply({ status: "ready", protocol: 39, workspace_id: "fixture", cuda: false, device: "Fixture CPU only", engines: Object.fromEntries(SETTINGS_ENGINES.map(e => [e, { available: true, features: { lora: true } }])), pipeline_configs: { sd: true, sdxl: true }, performance: {}, upscalers: { models: [], builtins: [] }, background_removal: { models: [], builtins: [] } });
  if (path === "/api/inference/jobs/active") return reply({ job: null });
  if (path === "/api/inference/jobs") return reply({ jobs: [], counts: {} });
  if (path === "/api/lora-relink/state") return reply({ revision: 5 });
  if (path === "/api/lora-cards") return reply({ store: { schemaVersion: 2, byEngine: {} } });
  if (path === "/api/inference/performance") return reply({ config: {}, capabilities: {}, active: {} });
  if (path === "/api/plugins") return reply({ plugins: [], diagnostics: [] });
  if (path === "/api/console") return reply({ entries: [], latest: 0 });
  if (path.includes("history")) return reply({ entries: [], images: [], batches: [] });
  if (path.endsWith("capabilities")) return reply({ executor_ready: true, max_concurrency: 1, protocol: 39 });
  if (path.includes("xyz-plots")) return reply({ experiments: [], unreadable: [] });
  return reply({ models: [], builtins: [], available: false, runtime_available: false });
};
const { default: App } = await import("../src/App.jsx");
createRoot(document.getElementById("root")).render(<StrictMode><App /></StrictMode>);
