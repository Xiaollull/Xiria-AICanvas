// QPlan2/P2 browser fixture: the real App plus an in-page stand-in for the LoRA mix-search routes and the XYZ detail of
// search rounds. It exercises the UI wiring only; the authoritative rules, reservations and admission are the Python
// service (backend/tests/test_lora_mix_service.py). Nothing here reaches a real service, model or GPU.
import { analyzeMembers, parseUnits } from "../src/lora-mix-rules.js";

await import("./xyz-full-ui-stage1.fixture.jsx");
const base = window.__appFetch;
const PIXEL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jI4sAAAAASUVORK5CYII=";
const state = window.__mixState = { experiments: new Map(), calls: [], fail: new Set(), exports: [], xyzLists: [] };
const json = (payload, status = 200) => new Response(JSON.stringify(payload), { status, headers: { "Content-Type": "application/json" } });
const hex = (text) => [...text].reduce((value, char) => (value * 31 + char.charCodeAt(0)) >>> 0, 7).toString(16).padStart(8, "0");

function weightsOf(draft) {
  const analysis = analyzeMembers(draft.mix.members);
  if (!analysis.free.length) return [{ id: "C01", source: "baseline", weights: Object.fromEntries(analysis.rows.map((row) => [draft.mix.members[row.index].ref.instanceId, row.parsed.base])) }];
  return draft.mix.candidates.map((candidate) => ({ id: candidate.id, source: candidate.source, weights: Object.fromEntries(analysis.rows.map((row) => {
    const id = draft.mix.members[row.index].ref.instanceId;
    return [id, row.locked ? row.parsed.base : parseUnits(candidate.raw[id]).units];
  })) }));
}
const keyOf = (weights) => Object.entries(weights).map(([id, units]) => `${id}:${units}`).join(",");
const statesOf = (experiment) => experiment.origin.candidates.map((candidate) => {
  const cells = experiment.cells.filter((cell) => cell.coordinate[0] === candidate.x_index);
  const complete = cells.filter((cell) => cell.status === "complete").length;
  return { ...candidate, state: complete === cells.length ? "tested" : experiment.released ? "failed" : experiment.status === "running" ? "reserved" : "failed", cells: cells.map((cell) => cell.status), complete, samples: cells.length };
});
const summary = (experiment) => ({ id: experiment.id, engine: "Anima", created_at: experiment.created_at, status: experiment.status, total_cells: experiment.cells.length, repeat_count: experiment.repeat_count, counts: counts(experiment),
  origin: { kind: "lora_mix_search", known: true, session_id: experiment.origin.session_id, round: experiment.origin.round, context_hash: "c".repeat(64), candidates: experiment.origin.candidates.length } });
const counts = (experiment) => { const value = { pending: 0, running: 0, complete: 0, error: 0, skipped: 0, cancelled: 0, missing: 0 }; for (const cell of experiment.cells) value[cell.status] += 1; return value; };
const sessionView = (sessionId) => {
  const rounds = [...state.experiments.values()].filter((item) => item.origin.session_id === sessionId).sort((a, b) => a.origin.round - b.origin.round);
  return { id: sessionId, revision: rounds.length, rounds: rounds.map((item) => ({ round: item.origin.round, experiment_id: item.id, created_at: item.created_at, status: item.status, context_hash: "c".repeat(64),
    algorithm: item.origin.algorithm, seeds: item.origin.seeds, rules: item.origin.rules, global_loras: item.origin.global_loras, released: Boolean(item.released), candidates: statesOf(item) })) };
};

function confirm(body) {
  const draft = body.draft, sessionId = draft.mix.session.id || hex(body.submission_id).repeat(4), existing = [...state.experiments.values()].filter((item) => item.origin.session_id === sessionId);
  if (draft.mix.session.revision !== existing.length) return json({ detail: { code: "stale-session", message: `搜索会话已有 ${existing.length} 轮`, revision: existing.length } }, 409);
  const used = new Set([...state.experiments.values()].flatMap((item) => statesOf(item).filter((c) => c.state !== "failed").map((c) => keyOf(c.weights))));
  const candidates = weightsOf(draft);
  const clash = candidates.filter((candidate) => used.has(keyOf(candidate.weights)));
  if (clash.length) return json({ detail: { code: "candidate-used", message: `${clash.length} 个候选向量已测试或已预留`, errors: clash.map((c) => ({ candidate: c.id, message: `${c.id}：已测试` })), error_count: clash.length } }, 409);
  const id = body.submission_id.replaceAll("-", ""), samples = draft.repeat_count;
  const seeds = Array.from({ length: samples }, (_, i) => String(1000 + i));
  const origin = { kind: "lora_mix_search", schema_version: 1, session_id: sessionId, round: existing.length + 1, context_hash: "c".repeat(64), seeds, algorithm: { name: "lora-mix-stratified-fps-v1", sampling_seed: draft.mix.sampling_seed, cursor: draft.mix.cursor },
    rules: { members: draft.mix.members, max_displacement: draft.mix.max_displacement }, global_loras: draft.baseline.loras,
    candidates: candidates.map((candidate, index) => ({ ...candidate, key: hex(keyOf(candidate.weights)).repeat(8), x_index: index, displacement: analyzeMembers(draft.mix.members).free.reduce((total, row) => total + Math.abs(candidate.weights[draft.mix.members[row.index].ref.instanceId] - row.parsed.base), 0) })) };
  const cells = [];
  candidates.forEach((candidate, x) => { for (let sample = 0; sample < samples; sample += 1) {
    const failed = state.fail.has(candidate.id), cellId = `${x}:0:0:${sample}`;
    cells.push({ id: cellId, coordinate: [x, 0, 0, sample], labels: { x: candidate.id, y: "—", z: "—" }, status: failed ? "error" : "complete", attempts: 1, seed: seeds[sample], error: failed ? "fixture failure" : null,
      ...(failed ? {} : { asset_id: `asset-${id}-${cellId}`, asset_url: PIXEL, thumbnail_url: PIXEL, output_name: `${candidate.id}-${sample}.png`, width: 512, height: 512 }) });
  } });
  const experiment = { id, origin, cells, status: cells.some((cell) => cell.status === "error") ? "complete" : "complete", repeat_count: samples, created_at: Date.now() / 1000 + state.experiments.size, released: false };
  state.experiments.set(id, experiment);
  return json({ experiment: summary(experiment), session: { id: sessionId, revision: origin.round }, replayed: false }, 202);
}

function mixRoute(url, method, body) {
  const path = url.split("?")[0].replace("/api/inference/lora-mix", "");
  if (path === "/capabilities") return json({ phase: "C", executor_ready: true, max_concurrency: 10, limits: { body_bytes: 1048576, total_cells: 1024, axis_values: 256, repeat_count: 32, requested_concurrency: 10, plan_bytes: 16777216 }, algorithm: "lora-mix-stratified-fps-v1" });
  if (path === "/used") return json({ context_hash: "c".repeat(64), used: [...state.experiments.values()].flatMap((item) => statesOf(item).map((c) => ({ candidate_id: c.id, state: c.state, weights: c.weights, round: item.origin.round, session_id: item.origin.session_id, experiment_id: item.id }))) });
  if (path === "/plan") { const candidates = weightsOf(body.draft); return json({ ok: true, errors: [], error_count: 0, summary: { profiles: candidates.length, samples: body.draft.repeat_count, cells: candidates.length * body.draft.repeat_count, new: candidates.length, duplicates: [], context_hash: "c".repeat(64) } }); }
  if (path === "/sessions" && method === "POST") return confirm(body);
  if (path === "/sessions") { const ids = [...new Set([...state.experiments.values()].map((item) => item.origin.session_id))]; return json({ sessions: ids.map((sid) => { const rounds = [...state.experiments.values()].filter((item) => item.origin.session_id === sid); const latest = rounds.sort((a, b) => b.created_at - a.created_at)[0]; return { id: sid, engine: "Anima", rounds: rounds.length, created_at: latest.created_at, updated_at: latest.created_at, latest_experiment: latest.id }; }) }); }
  let match = /^\/sessions\/([0-9a-f]{32})$/.exec(path);
  if (match) return json(sessionView(match[1]));
  match = /^\/experiments\/([0-9a-f]{32})\/release$/.exec(path);
  if (match) { const item = state.experiments.get(match[1]); item.released = true; return json({ experiment: summary(item) }); }
  match = /^\/experiments\/([0-9a-f]{32})\/export\/estimate$/.exec(path);
  if (match) return json({ width: 512 * (body.columns || 4), height: 1200, percent: 100, alpha: false, blocks: 1, cells: { complete: state.experiments.get(match[1]).cells.filter((c) => c.status === "complete").length }, warnings: [], suggestions: [], budget: { ok: true, memory_bytes: 1e6, problems: [] } });
  match = /^\/experiments\/([0-9a-f]{32})\/exports$/.exec(path);
  if (match) { const record = { id: `export-${state.exports.length + 1}`, status: "complete", options: body, progress: { done: 1, total: 1 }, output: { asset_id: "grid-asset", asset_url: PIXEL, name: "grid.png", width: 2048, height: 1200 } }; state.exports.push({ experiment: match[1], body, record }); return json({ export: record }, 202); }
  return json({ detail: "fixture: unknown search route" }, 404);
}

function xyzRoute(id, rest, method) {
  const item = state.experiments.get(id);
  if (!rest) return json({ ...summary(item), origin: item.origin, mix_released: item.released ? { at: 1 } : null, requested_concurrency: 1, effective_concurrency: 1, decision: null, exports: [], batch_log: [], foreign: false,
    axes: { x: { field: "lora_profile", labels: item.origin.candidates.map((c) => c.id) }, y: { field: null, labels: ["—"] }, z: { field: null, labels: ["—"] } }, baseline: { width: 512, height: 512, steps: 20 }, cells: item.cells, live: null });
  let match = /^\/cells\/(.+)$/.exec(rest);
  if (match) { const cell = item.cells.find((entry) => entry.id === decodeURIComponent(match[1])); return json({ experiment_id: id, cell_id: cell.id, coordinate: { x_index: cell.coordinate[0], y_index: 0, z_index: 0, sample_index: cell.coordinate[3] }, labels: cell.labels,
    request: { engine: "Anima", prompt: "fixture prompt", negative_prompt: "", seed: Number(cell.seed), steps: 20, cfg: 4, width: 512, height: 512, sampler: "euler", scheduler: "normal", loras: [] }, lora_references: [], prompt_parts: { prompt: "fixture prompt", negative_prompt: "", rest: "fixture prompt", removed: false }, status: cell.status, output: cell.asset_id ? { asset_id: cell.asset_id, asset_url: PIXEL, name: cell.output_name, width: 512, height: 512 } : null }); }
  match = /^\/exports\/(.+)$/.exec(rest);
  if (match) return json(state.exports.find((entry) => entry.record.id === match[1])?.record || { status: "error" });
  return json({ detail: "fixture: unsupported xyz route" }, 404);
}

// Like the real service: the XYZ list holds every experiment (search rounds carry `origin`) unless origin=none is asked.
const plainXYZ = { id: "old-exp", engine: "Anima", created_at: 1791234567, status: "paused", counts: { complete: 4, pending: 4, running: 0, error: 0, skipped: 0, cancelled: 0, missing: 0 }, total_cells: 8, origin: null };
function xyzList(url) {
  state.xyzLists.push(url);
  const all = [plainXYZ, ...[...state.experiments.values()].map(summary)].sort((a, b) => b.created_at - a.created_at);
  const experiments = new URLSearchParams(url.split("?")[1] || "").get("origin") === "none" ? all.filter((item) => !item.origin) : all;
  return json({ experiments, total: experiments.length, unreadable: [] });
}

window.__appFetch = async (url, method, body) => {
  if (url.startsWith("/api/inference/lora-mix/")) { state.calls.push({ url, method, body }); return mixRoute(url, method, body); }
  if (url.split("?")[0] === "/api/inference/xyz-plots" && method === "GET") return xyzList(url);
  const match = /^\/api\/inference\/xyz-plots\/([0-9a-f]{32})(\/.*)?$/.exec(url.split("?")[0]);
  if (match && state.experiments.has(match[1])) return xyzRoute(match[1], match[2] || "", method);
  if (url.startsWith("/api/inference/jobs") && method === "GET" && window.__mixQueue) {
    const item = [...state.experiments.values()][0];
    return json({ jobs: [{ id: item.id, kind: "xyz_plot", engine: "Anima", status: "complete", sequence: 1, active: false, total_images: item.cells.length, xyz: { total: item.cells.length, counts: counts(item) }, origin: summary(item).origin }], counts: {} });
  }
  return base(url, method, body);
};
