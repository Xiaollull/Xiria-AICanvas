import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "../src/styles.css";
import { createXYZDraft, xyzLocalSummary } from "../src/xyz-draft.js";
import { xyzRuntimeGuidanceForEngine } from "../src/xyz-axis-editor.js";
import XYZBaselineSidebar from "../src/XYZBaselineSidebar.jsx";
import GenerationModeSwitch from "../src/GenerationModeSwitch.jsx";

const idA = "11111111-1111-4111-8111-111111111111";
const idB = "22222222-2222-4222-8222-222222222222";
const frozenLora = { name: "Portrait LoRA", value: "portrait.safetensors", weight: "0.8", instanceId: idA, identitySchema: 1,
  contentHash: "a".repeat(64), contentRevision: 1, enabled: true };
const fixtureAsset = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jI4sAAAAASUVORK5CYII=";

const makeDraft = (engine) => createXYZDraft({
  parameters: { engine, checkpoint: "fixture-model.safetensors", diffusion_model: "fixture-diffusion.safetensors", text_encoder: "fixture-encoder.safetensors",
    text_encoder_2: "fixture-t5.safetensors", vae: "fixture-vae.safetensors", prompt: "soft portrait, warm light", negative_prompt: "blur, low quality",
    steps: 24, cfg: 6, width: 512, height: 512, seed: "18446744073709551615", sampler: "euler", scheduler: "normal", guidance: "none",
    pag: engine === "SD" ? null : { scale: 0.3, applied_layers: "mid" }, postprocess_chain: [] },
  seed_mode: "fixed", loras: [structuredClone(frozenLora)], group_snapshots: [{ id: "frozen-group", name: "Portrait profile", presetPrompt: "soft light", members: [structuredClone(frozenLora)] }],
  prompt_without_groups: "soft portrait, warm light", group_prompt: "", library_revision: 1,
});

function makeCells(experimentId) {
  return Array.from({ length: 8 }, (_, index) => {
    const coordinate = [index % 2, 0, Math.floor(index / 4), Math.floor((index % 4) / 2)];
    const id = coordinate.join(":");
    return { id, cell_id: id, coordinate, status: "complete", seed: String(index + 10), width: 512, height: 512, attempts: 1,
      asset_id: `fixture-asset-${experimentId}-${index}`, asset_url: fixtureAsset, thumbnail_url: fixtureAsset, output_name: `fixture-${index}.png`,
      labels: { x: `X${coordinate[0] + 1}`, y: "Y1", z: `Z${coordinate[2] + 1}` }, prompt: "soft portrait, warm light" };
  });
}
function makeDetail(id) {
  const historical = id === "old-exp";
  const cells = makeCells(id).map((cell, index) => historical && index >= 4 ? { ...cell, status: "pending", asset_id: null, asset_url: null, thumbnail_url: null } : cell);
  return { id, engine: "SD", created_at: historical ? 1791234567 : 1792234567, status: historical ? "paused" : "complete", counts: { complete: historical ? 4 : 8, error: 0, skipped: 0, missing: 0, cancelled: 0, pending: historical ? 4 : 0 }, total_cells: 8,
    requested_concurrency: 1, effective_concurrency: 1, concurrency_reason: "顺序生成", repeat_count: 2,
    baseline: { checkpoint: "frozen-old-model.safetensors", width: 768, height: 512, steps: 20, loras: 1 },
    axes: { x: { labels: ["步数 20", "步数 24"] }, y: { labels: ["—"] }, z: { labels: ["无测试 LoRA", "Portrait profile"] } }, cells, live: null,
    decision: null, error: null, exports: [], batch_log: [], foreign: false };
}

const experiments = new Map([["old-exp", makeDetail("old-exp")]]);
const makeList = () => ({ experiments: [...experiments.values()].map((item) => ({ id: item.id, engine: item.engine, created_at: item.created_at,
  status: item.status, counts: item.counts, total_cells: item.total_cells })), unreadable: [] });
const response = (payload, status = 200) => new Response(JSON.stringify(payload), { status, headers: { "Content-Type": "application/json" } });

window.__xyzCalls = [];
window.__xyzHostErrors = [];
window.addEventListener("error", (event) => window.__xyzHostErrors.push(event.message));
window.fetch = async (input, options = {}) => {
  const url = typeof input === "string" ? input : input.url;
  const method = String(options.method || "GET").toUpperCase();
  const body = options.body ? JSON.parse(options.body) : null;
  if (window.__appFetch) { const answer = await window.__appFetch(url, method, body); if (answer) return answer; }
  if (url.startsWith("/api/inference/xyz-plots")) window.__xyzCalls.push({ url, method, body });
  if (url.endsWith("/capabilities")) return response({ executor_ready: true, max_concurrency: 10, protocol: 40, phase: "E" });
  if (url.endsWith("/plan")) {
    window.__xyzPlanBodies ||= [];
    window.__xyzPlanBodies.push(body);
    if (window.__holdPlan) await new Promise((resolve) => { window.__releasePlan = resolve; });
    return response({ ok: true, dimensions: { x: 1, y: 1, z: 1 }, total_cells: 1, switches: { model: 0, loras: 0 }, seed_status: "resolved", requested_concurrency: 1,
      batches: { count: 1, sizes: { 1: 1 }, reasons: ["测试夹具，不是后端执行"] } });
  }
  if (url === "/api/inference/xyz-plots" && method === "POST") {
    window.__xyzSubmitBodies ||= [];
    window.__xyzSubmitBodies.push(body);
    if (window.__networkFailSubmit) { window.__networkFailSubmit -= 1; throw new TypeError("fixture network disconnect"); }
    const id = window.__holdSubmit ? "late-exp" : `fixture-exp-${window.__xyzSubmitBodies.length}`;
    if (window.__holdSubmit) await new Promise((resolve) => { window.__releaseSubmit = resolve; });
    if (window.__appFixture) {
      const summary = xyzLocalSummary(body.draft), p = body.draft.baseline.parameters, cells = [];
      for (let z = 0; z < summary.dimensions.z; z++) for (let sample = 0; sample < body.draft.repeat_count; sample++) for (let y = 0; y < summary.dimensions.y; y++) for (let x = 0; x < summary.dimensions.x; x++) {
        const coordinate = [x, y, z, sample];
        cells.push({ ...makeCells(id)[0], id: coordinate.join(":"), coordinate, width: p.width, height: p.height, labels: { x: String(x), y: String(y), z: String(z) } });
      }
      experiments.set(id, { ...makeDetail(id), engine: body.draft.engine, total_cells: cells.length, repeat_count: body.draft.repeat_count,
        baseline: { ...structuredClone(p), loras: body.draft.baseline.loras.length }, counts: { complete: cells.length, pending: 0, error: 0, missing: 0, skipped: 0, cancelled: 0 }, cells,
        axes: Object.fromEntries(["x", "y", "z"].map((key) => [key, { labels: Array.from({ length: summary.dimensions[key] }, (_, i) => `${key.toUpperCase()}${i + 1}`) }])) });
      window.__fixtureSubmitted = structuredClone(experiments.get(id));
    } else experiments.set(id, makeDetail(id));
    return response({ experiment: { id } }, 201);
  }
  if (url.startsWith("/api/inference/xyz-plots?") || url === "/api/inference/xyz-plots?engine=SD") return response(makeList());
  const detailMatch = url.match(/^\/api\/inference\/xyz-plots\/([^/?]+)$/);
  if (detailMatch) return response(experiments.get(decodeURIComponent(detailMatch[1])) || makeDetail(decodeURIComponent(detailMatch[1])));
  const cellMatch = url.match(/^\/api\/inference\/xyz-plots\/([^/]+)\/cells\/([^/?]+)$/);
  if (cellMatch) {
    const cell = makeDetail(decodeURIComponent(cellMatch[1])).cells.find((entry) => entry.id === decodeURIComponent(cellMatch[2])) || makeDetail("old-exp").cells[0];
    return response({ ...cell, engine: "SD", cell_id: cell.id, prompt_parts: { lead: "", rest: "soft portrait, warm light", removed: false },
      prompt_parts_summary: { lead: "", rest: "soft portrait, warm light", removed: false }, lora_references: [structuredClone(frozenLora)],
      request: { engine: "SD", checkpoint: "fixture-model.safetensors", prompt: "soft portrait, warm light", negative_prompt: "blur, low quality", steps: 24, cfg: 6,
        width: 512, height: 512, seed: "18446744073709551615", sampler: "euler", scheduler: "normal", guidance: "none", pag: null, loras: [structuredClone(frozenLora)], postprocess_chain: [] } });
  }
  const controlMatch = url.match(/^\/api\/inference\/xyz-plots\/([^/]+)\/(pause|resume|stop)$/);
  if (controlMatch) {
    const current = experiments.get(decodeURIComponent(controlMatch[1])) || makeDetail(decodeURIComponent(controlMatch[1]));
    const status = { pause: "paused", resume: "running", stop: "cancelled" }[controlMatch[2]];
    experiments.set(current.id, { ...current, status });
    return response({ ok: true });
  }
  if (url.includes("/export/estimate")) {
    const percent = body?.percent ?? 100;
    return response({ width: Math.round(2048 * percent / 100), height: Math.round(1024 * percent / 100), percent, blocks: 2,
      cells: { complete: 8 }, budget: { ok: percent <= 75, memory_bytes: percent <= 75 ? 32_000_000 : 2_000_000_000, problems: percent <= 75 ? [] : ["夹具预算超限"] }, warnings: [], suggestions: [50] });
  }
  if (url.endsWith("/exports")) return response({ export: { id: "fixture-export", status: "complete", progress: { done: 8, total: 8 }, output: { asset_id: "fixture-grid", asset_url: fixtureAsset, name: "fixture-grid.png", width: 1024, height: 512, bytes: 1024 }, manifest_url: fixtureAsset } }, 201);
  if (url.includes("/exports/fixture-export")) return response({ id: "fixture-export", status: "complete", progress: { done: 8, total: 8 }, output: { asset_id: "fixture-grid", asset_url: fixtureAsset, name: "fixture-grid.png", width: 1024, height: 512, bytes: 1024 }, manifest_url: fixtureAsset });
  if (method === "POST") return response({ ok: true });
  return response({ detail: `Fixture route not found: ${method} ${url}` }, 404);
};

async function boot() {
  if (window.__appFixture) {
    const { default: App } = await import("../src/App.jsx");
    createRoot(document.getElementById("root")).render(<App />);
    return;
  }
  const { default: XYZPlotWorkspace } = await import("../src/XYZPlotWorkspace.jsx");
  const defaultEngine = "SD";
  function Fixture() {
    const [engine, setEngine] = useState(defaultEngine);
    const [mode, setMode] = useState("xyz_plot"), [expanded, setExpanded] = useState(false);
    const [runtimeGuidance, setRuntimeGuidance] = useState(() => xyzRuntimeGuidanceForEngine({
      engines: { Anima: { features: { pag: true } }, Flux: {}, Flux2: {}, Krea2: {} },
      guidance: { pag: { available: true, implementation: "diffusers_pag", implementations: {
        SD: "diffusers_pag", iL: "diffusers_pag", Anima: "native_cosmos_identity_self_attention",
      }, engines: ["SD", "iL", "Anima"] }, cfg_zero_star: { available: true, engines: ["Anima", "Krea2"] } },
    }, defaultEngine));
    const [drafts, setDrafts] = useState(() => ({ SD: makeDraft("SD"), iL: makeDraft("iL"), Anima: makeDraft("Anima"), Flux: makeDraft("Flux"), Flux2: makeDraft("Flux2"), Krea2: makeDraft("Krea2") }));
    const [basic, setBasic] = useState({ prompt: "basic stays separate", checkpoint: "basic-model.safetensors", lora: "Basic Portrait" });
    window.__fixtureInspect = () => ({ engine, draft: structuredClone(drafts[engine]), basic: structuredClone(basic), runtimeGuidance: structuredClone(runtimeGuidance) });
    window.__fixtureSetRuntimeGuidance = (value) => setRuntimeGuidance(structuredClone(value));
    window.__fixturePatchDraft = (targetEngine, patch) => setDrafts((current) => ({ ...current, [targetEngine]: { ...current[targetEngine], ...patch } }));
    const draft = drafts[engine];
    const componentChoices = { checkpoint: [{ value: "fixture-model.safetensors", label: "Fixture checkpoint" }],
      diffusion_model: [{ value: "fixture-diffusion.safetensors", label: "Fixture diffusion" }], text_encoder: [{ value: "fixture-encoder.safetensors", label: "Fixture encoder" }],
      text_encoder_2: [{ value: "fixture-t5.safetensors", label: "Fixture T5" }], vae: [{ value: "fixture-vae.safetensors", label: "Fixture VAE" }] };
    const onDraft = (next) => setDrafts((old) => ({ ...old, [engine]: next }));
    const catalog = { choices: componentChoices, loading: false, loras: [{ ...frozenLora, instanceId: idB, name: "Second Portrait", value: "second.safetensors", contentRevision: 2 }] };
    return <div className="fixture-frame">
      <div className="fixture-toolbar"><strong>独立 UI 夹具</strong><span>当前源码组件 · Fixture APIs · 不连接推理服务</span></div>
      <main className="workspace xyz-mode fixture-workspace">
        <aside className="left-panel panel fixture-left xyz-sidebar"><div className="panel-scroll">
          <label>引擎<select aria-label="夹具引擎" value={engine} onChange={(event) => setEngine(event.target.value)}>{["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"].map((value) => <option key={value}>{value}</option>)}</select></label>
          <GenerationModeSwitch value={mode} onChange={setMode} />
          <button type="button" className="xyz-sidebar-mobile-toggle" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>XYZ 基线 · {engine}<span>{expanded ? "收起" : "展开编辑"}</span></button>
          <div className={`xyz-sidebar-content${expanded ? " expanded" : ""}`}>
          {mode === "xyz_plot" ? <XYZBaselineSidebar key={engine} draft={draft} onChange={onDraft} catalog={catalog} runtimeGuidance={["none", "pag", "cfg_zero_star"]} groups={draft.baseline.group_snapshots} /> : <label>Basic Prompt<input aria-label="Basic Prompt fixture" value={basic.prompt} onChange={(event) => setBasic({ ...basic, prompt: event.target.value })} /></label>}
          </div>
        </div></aside>
        {mode === "xyz_plot" && <XYZPlotWorkspace key={engine} draft={draft} engine={engine} componentChoices={componentChoices} runtimeGuidance={runtimeGuidance} captureBaseline={() => structuredClone(makeDraft(engine).baseline)}
          onChange={onDraft} integration={{ onOpenViewer: (value) => { window.__viewerSource = value; },
            onAddToGallery: (value) => { window.__gallerySource = value; }, onApplyToBasic: (settings, groups) => { window.__applySource = { settings, groups }; },
            promptContext: { groupPromptFor: () => "" } }} />}
      </main>
    </div>;
  }
  createRoot(document.getElementById("root")).render(<Fixture />);
}
void boot();
