import { engineSettingsDefaults, emptyEngineSettingsMap, SETTINGS_ENGINES } from "../src/engine-settings.js";
import { emptyMountedLoraMap } from "../src/lora-model-scope.js";
import { emptyLoraGroupMap } from "../src/lora-groups.js";
import { seededPromptPresetContainer } from "../src/prompt-presets.js";

window.__appFixture = true;
const settings = emptyEngineSettingsMap(), mounts = emptyMountedLoraMap(), groups = emptyLoraGroupMap();
const reference = (engine, index) => ({ name: `${engine} Portrait ${index}`, value: `${engine}-portrait-${index}.safetensors`, weight: 0.8137, precision: 4,
  enabled: true, identitySchema: 1, instanceId: `00000000-0000-4000-8000-${String(SETTINGS_ENGINES.indexOf(engine) * 10 + index + 1).padStart(12, "0")}`, contentHash: String(index + 1).repeat(64), contentRevision: 1 });
for (const engine of SETTINGS_ENGINES) {
  settings.byEngine[engine] = { ...engineSettingsDefaults(engine), positive: `Basic ${engine} exact prompt`, negative: "Basic exact negative", checkpoint: `${engine}-checkpoint.safetensors`,
    diffusionModel: `${engine}-diffusion.safetensors`, textEncoder: `${engine}-encoder.safetensors`, textEncoder2: `${engine}-encoder2.safetensors`, vae: `${engine}-vae.safetensors`,
    size: { width: 768, height: 1024 }, steps: 28, seed: "123456789", seedMode: "fixed", imageToImage: { ...engineSettingsDefaults(engine).imageToImage, prompt: `i2i ${engine} unchanged` } };
  mounts.byEngine[engine] = [reference(engine, 0)];
  groups.byEngine[engine] = [{ id: `g-${engine.toLowerCase()}`, name: `Library ${engine}`, enabled: false, collecting: false, presetPrompt: "group preset", members: [reference(engine, 1)] }];
}
let state = { workspace: { ...settings.byEngine.SD, model: "SD", engineSettingsByEngine: settings, mountedLorasByEngine: mounts, loraGroupsByEngine: groups,
  promptPresets: seededPromptPresetContainer(), canvasMaxEdge: 2048, generationMode: "basic" } };
window.__fixtureAppCalls = [];
window.__fixtureAppState = () => structuredClone(state);
const response = (payload, status = 200) => new Response(JSON.stringify(payload), { status, headers: { "Content-Type": "application/json", "X-XirAI-LoRA-Revision": "0", "X-XirAI-Document-Version": "fixture-only" } });
window.__appFetch = async (url, method, body) => {
  if (url.startsWith("/api/inference/xyz-plots")) return null;
  window.__fixtureAppCalls.push({ url, method, body });
  if (url.startsWith("/api/ui-state")) { if (method === "PUT") state = structuredClone(body); return response({ state }); }
  const engine = new URL(url, location.origin).searchParams.get("engine") || "SD";
  if (url.startsWith("/api/models?")) {
    const fields = engine === "Flux" ? ["diffusion_model", "text_encoder", "text_encoder_2", "vae"] : ["diffusion_model", "text_encoder", "vae"];
    const names = { diffusion_model: "diffusion", text_encoder: "encoder", text_encoder_2: "encoder2", vae: "vae" };
    const choices = (kind) => [{ value: `${engine}-${kind}.safetensors`, name: `${engine} ${kind}` }, { value: `${engine}-${kind}-other.safetensors`, name: `${engine} ${kind} other` }];
    return response(["SD", "iL"].includes(engine) ? { engine, model_type: "checkpoint", models: choices("checkpoint"), directory: "fixture/read-only" }
      : { engine, model_type: "split", assets: Object.fromEntries(fields.map((field) => [field, { models: choices(names[field]), directory: "fixture/read-only" }])), runtime: {} });
  }
  if (url.startsWith("/api/loras?")) return response({ engine, directory: "fixture/read-only", categories: [{ id: "character", label: "人物", models: [reference(engine, 0), reference(engine, 1)] }] });
  if (url.startsWith("/api/inference/health")) return response({ status: "ready", cuda: false, device: "Fixture · no GPU", protocol: 40,
    engines: Object.fromEntries(SETTINGS_ENGINES.map((e) => [e, { available: true, features: { lora: true, hires: true } }])),
    pipeline_configs: { sd: true, sdxl: true }, guidance: { pag: { available: true, engines: ["SD", "iL", "Anima"] }, cfg_zero_star: { available: true, engines: ["Anima", "Krea2"] }, },
    performance: {}, upscalers: { models: [], builtins: [] }, background_removal: { models: [], builtins: [] } });
  if (url.startsWith("/api/inference/jobs/active")) return response({ job: null });
  if (url === "/api/inference/jobs" && method !== "GET") return response({ detail: "Ordinary generation forbidden in this fixture" }, 409);
  if (url.startsWith("/api/inference/jobs")) return response({ jobs: window.__fixtureQueueParent ? [{ id: "old-exp", kind: "xyz_plot", engine: "SD", status: "paused", sequence: 1, active: false, xyz: { total: 8, counts: { complete: 4 } } }] : [], counts: {} });
  if (url.startsWith("/api/inference/performance")) return response({ config: {}, capabilities: {}, active: {} });
  if (url.startsWith("/api/lora-cards")) return response({ store: { schemaVersion: 2, byEngine: {} } });
  if (url.startsWith("/api/lora-relink/state")) return response({ revision: 0 });
  if (url.startsWith("/api/lora-relink/resolve")) {
    // Fixture-only stand-in for the authoritative service: strong references are echoed back as current evidence;
    // a legacy path-only reference is the service's untouched-legacy fallback, which can never authorise a snapshot.
    (window.__resolveCalls ||= []).push(body);
    const references = body.references.map((item) => item.instanceId ? { ...item, status: "resolved", kind: "current", originalValue: item.value } : { ...item, status: "unresolved" });
    return response({ revision: 0, references, loras: references.map((item) => ({ path: item.value, weight: item.weight })) });
  }
  if (url.startsWith("/api/console")) return response({ entries: [], latest: 0 });
  if (url.startsWith("/api/plugins")) return response({ plugins: [], diagnostics: [] });
  if (url.startsWith("/api/inference/adetailer/models") || url.startsWith("/api/inference/upscalers")) return response({ models: [], builtins: [], available: false, runtime_available: false });
  if (url.startsWith("/api/inference/history")) return response({ entries: [], images: [], batches: [] });
  if (url.includes("download/job") || url.endsWith("/download")) return response({ job: null });
  return response({}); // Deliberately never falls through to a network service.
};
await import("./xyz-ui-redesign.fixture.jsx");
