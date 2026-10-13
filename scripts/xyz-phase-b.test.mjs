import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { xyzLoraReferences, xyzPlotMiddleware } from "./xyz-plot-http.mjs";
import { xyzApi, xyzCoordinateText, xyzErrorText, xyzGridPage, xyzIssues, xyzSubmissionId, xyzExperimentActive } from "../src/xyz-client.js";
import { GENERATION_MODES } from "../src/generation-modes.js";

const A = randomUUID(), B = randomUUID(), C = randomUUID();
const ref = (instanceId, value, weight = "1", extra = {}) => ({ value, weight, precision: 4, identitySchema: 1, instanceId, contentHash: "a".repeat(64), contentRevision: 1, ...extra });
const draft = (overrides = {}) => ({ schema_version: 1, kind: "xyz_draft", engine: "SD", baseline: { parameters: { engine: "SD" }, loras: [] }, axes: { x: null, y: null, z: null }, ...overrides });

test("LoRA reference projection delegates to the authoritative private planner, never declaration union", async () => {
  const value = draft({
    baseline: { parameters: {}, loras: [ref(A, "a.safetensors"), ref(B, "b.safetensors", "0"), ref(C, "c.safetensors", "1", { enabled: false })] },
    axes: {
      x: { field: "lora_weight", target: B, values: [{ value: "0.5" }] },
      y: { field: "lora_profile", values: [{ value: { mode: "full", members: [ref(A, "a.safetensors"), ref(randomUUID(), "d.safetensors")] } }, { value: "raw invalid json" }] },
      z: { field: "lora_model", target: A, values: [{ value: ref(randomUUID(), "e.safetensors") }, { value: 3 }] },
    },
  });
  const result = { effectiveReferences: [ref(B, 'effective.safetensors')], references: [ref(B, 'effective.safetensors')], revision: 1 };
  assert.equal(await xyzLoraReferences(value, { resolveXYZ: async (input) => { assert.equal(input, value); return result; } }), result);
});

function harness(service) {
  // Old middleware unit fixtures supply already-projected effective sets. The delta suite
  // below uses the REAL service, lease, stdin owner and Python planner for semantic coverage.
  if (!service.resolveXYZ) service.resolveXYZ = async (draft) => {
    const effectiveReferences = (draft.baseline.loras || []).filter((item) => item.enabled !== false && Number(item.weight) !== 0);
    const resolved = effectiveReferences.length ? await service.resolve(draft.engine, effectiveReferences) : { revision: 0, references: [] };
    return { ...resolved, effectiveReferences };
  };
  const forwarded = [];
  const route = xyzPlotMiddleware({ service, readJson: async (request) => request.body, admit: (request) => { if (request.headers.origin === "http://evil") throw Object.assign(new Error("cross"), { statusCode: 403 }); },
    forward: async (pathname, payload, headers, _request, response) => { forwarded.push({ pathname, payload, headers }); response.statusCode = 202; response.end("{}"); } });
  const invoke = async (url, body, headers = {}, method = "POST") => {
    const response = { statusCode: 200, headersSent: false, setHeader() {}, end(text) { this.text = text; } };
    let passed = false;
    await route({ url, method, headers, body }, response, () => { passed = true; });
    return { response, passed };
  };
  return { forwarded, invoke };
}

test("Node XYZ admission: resolves every reference, stamps the revision, refuses stale/relocated before Python", async () => {
  const calls = [];
  const service = { resolve: async (engine, references) => {
    calls.push({ engine, references });
    return { revision: 7, references: references.map((item) => ({ ...item, value: item.value === "moved.safetensors" ? "concept/moved.safetensors" : item.value })), loras: [] };
  } };
  const { forwarded, invoke } = harness(service);
  assert.equal((await invoke("/api/inference/xyz-plots/abc", {}, {}, "GET")).passed, true);
  assert.equal((await invoke("/api/inference/xyz-plots/abc/stop", {})).passed, true);
  const withLora = draft({ baseline: { parameters: {}, loras: [ref(A, "a.safetensors")] } });
  let result = await invoke("/api/inference/xyz-plots/plan", withLora, { "x-xirai-lora-revision": "6" });
  assert.equal(result.response.statusCode, 409);
  assert.equal(forwarded.length, 0);
  result = await invoke("/api/inference/xyz-plots/plan", withLora, { "x-xirai-lora-revision": "7", "x-xirai-lora-context": "[{}]" });
  assert.equal(forwarded.length, 1);
  assert.equal(forwarded[0].headers["x-xirai-lora-revision"], "7");
  assert.equal(forwarded[0].headers["x-xirai-lora-context"], undefined);
  assert.equal(calls.at(-1).engine, "SD");
  const submission = { submission_id: randomUUID(), draft: draft({ baseline: { parameters: {}, loras: [ref(A, "moved.safetensors")] } }) };
  result = await invoke("/api/inference/xyz-plots", submission, { "x-xirai-lora-revision": "7" });
  assert.equal(result.response.statusCode, 409);
  assert.match(JSON.parse(result.response.text).detail, /同步 LoRA/);
  assert.equal(forwarded.length, 1);
  result = await invoke("/api/inference/xyz-plots", { submission_id: randomUUID(), draft: draft() });
  assert.equal(result.response.statusCode, 202);
  assert.equal(forwarded.length, 2);
  assert.equal(calls.length, 3, "no resolve for a draft without LoRA references");
  assert.equal((await invoke("/api/inference/xyz-plots", [])).response.statusCode, 400);
  assert.equal((await invoke("/api/inference/xyz-plots", { draft: { engine: 3 } })).response.statusCode, 400);
  assert.equal((await invoke("/api/inference/xyz-plots", { draft: draft() }, { origin: "http://evil" })).response.statusCode, 403);
  const failing = harness({ resolve: async () => { throw new Error("internal"); } });
  const failed = await failing.invoke("/api/inference/xyz-plots/plan", withLora, { "x-xirai-lora-revision": "7" });
  assert.equal(failed.response.statusCode, 503);
  assert.equal(failing.forwarded.length, 0);
});

test("client: getRandomValues UUIDv4 submission id (plain-HTTP safe), never randomUUID", () => {
  let called = 0;
  const id = xyzSubmissionId({ getRandomValues: (bytes) => { called += 1; bytes.fill(0xab); return bytes; }, randomUUID: () => assert.fail("randomUUID must not be used") });
  assert.equal(called, 1);
  assert.match(id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  assert.notEqual(xyzSubmissionId(), xyzSubmissionId());
});

test("client: error shapes, coordinates and grid pages from server snapshots", () => {
  assert.equal(xyzErrorText({ detail: "plain" }), "plain");
  assert.equal(xyzErrorText({ detail: { message: "planned", errors: [] } }), "planned");
  assert.equal(xyzErrorText({ detail: [{ msg: "a" }, { msg: "b" }] }), "a；b");
  assert.equal(xyzIssues({ detail: { errors: [{ code: "x" }] } }).length, 1);
  assert.equal(xyzCoordinateText({ coordinate: { x_index: 1, y_index: 0, z_index: 0, sample_index: 2 } }), "X2 · Y1 · Z1 · 样本 3");
  assert.equal(xyzCoordinateText({ axis: "y" }), "Y 轴");
  const detail = { axes: { x: { labels: ["a", "b"] }, y: { labels: ["r"] }, z: { labels: [] } }, cells: [
    { id: "1:0:0:0", coordinate: [1, 0, 0, 0] }, { id: "0:0:0:0", coordinate: [0, 0, 0, 0] }, { id: "0:0:0:1", coordinate: [0, 0, 0, 1] }] };
  assert.deepEqual(xyzGridPage(detail, 0, 0)[0].map((cell) => cell.id), ["0:0:0:0", "1:0:0:0"]);
  assert.deepEqual(xyzGridPage(detail, 0, 1)[0].map((cell) => cell?.id ?? null), ["0:0:0:1", null]);
  assert.equal(xyzExperimentActive({ status: "paused" }), true);
  assert.equal(xyzExperimentActive({ status: "interrupted" }), false);
});

test("client: submit carries the published revision and one submission id; reads are no-store", async () => {
  const requests = [];
  const fetcher = async (path, options) => { requests.push({ path, options }); return { ok: true, json: async () => ({ experiment: { id: "e" } }) }; };
  const api = xyzApi(fetcher);
  const id = randomUUID();
  await api.submit({ engine: "SD" }, { submissionId: id, onError: "continue" });
  const body = JSON.parse(requests[0].options.body);
  assert.deepEqual(Object.keys(body).sort(), ["draft", "kind", "on_error", "schema_version", "submission_id"]);
  assert.equal(body.submission_id, id);
  assert.equal(body.on_error, "continue");
  assert.ok("X-XirAI-LoRA-Revision" in requests[0].options.headers);
  await api.detail("abc");
  assert.equal(requests[1].options.cache, "no-store");
  assert.equal(requests[1].options.method, "GET");
  await api.continue("abc", "failed");
  assert.deepEqual(JSON.parse(requests[2].options.body), { retry: "failed" });
  const failing = xyzApi(async () => ({ ok: false, status: 422, json: async () => ({ detail: { message: "bad", errors: [{ code: "x" }] } }) }));
  await assert.rejects(failing.plan({}), (error) => error.status === 422 && error.message === "bad" && xyzIssues(error.payload).length === 1);
});

test("wiring: registry executor B, workspace never posts ordinary jobs, App never adopts an experiment, Vite admits XYZ after the identity gate", async () => {
  assert.equal(GENERATION_MODES[1].executorReady, true);
  const [app, workspace, vite] = await Promise.all(["../src/App.jsx", "../src/XYZPlotWorkspace.jsx", "../vite.config.js"].map((file) => readFile(new URL(file, import.meta.url), "utf8")));
  assert.doesNotMatch(workspace, /\/api\/inference\/jobs|randomSeed\(|fetch\(/);
  assert.match(workspace, /api\.submit\(snapshot, \{ submissionId, onError: policy \}\)/);
  assert.match(workspace, /xyzSubmissionKey\(submittedKey, policy\)/);
  assert.match(app, /const generate = async \(\) => \{\s*if \(selectedGenerationMode !== "basic"\) return/);
  assert.match(app, /job\.active && job\.id !== generationJob && job\.kind !== "xyz_plot"/);
  assert.match(app, /if \(job\.kind === "xyz_plot"\) return <XYZQueueRow/);
  const gate = vite.indexOf("const registerInferenceIdentityGate"), job = vite.indexOf("server.middlewares.use(loraJobMiddleware("), xyz = vite.indexOf("server.middlewares.use(xyzPlotMiddleware(");
  assert.ok(gate > 0 && job > gate && xyz > job, "XYZ admission is registered inside the identity gate block, after the job route");
  assert.match(vite.slice(xyz, xyz + 600), /admit: requireSameOrigin/);
});
