import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { loraMixMiddleware, loraMixReferences } from "./lora-mix-http.mjs";

// QPlan2/P2: Node admission for the two search routes that admit work. Fake service; the real-service identity
// checks live in xyz-group-resolution-service.test.mjs and backend/tests/test_lora_mix_service.py.
const ref = (id, value, extra = {}) => ({ value, name: value, identitySchema: 1, instanceId: id, contentHash: "a".repeat(64), contentRevision: 1, ...extra });
const G = randomUUID(), A = randomUUID(), B = randomUUID();
const draft = (overrides = {}) => ({ engine: "Anima", baseline: { parameters: {}, loras: [{ ...ref(G, "g.safetensors"), weight: "0.5", enabled: true }, { ...ref(randomUUID(), "off.safetensors"), weight: "1", enabled: false }] },
  mix: { members: [{ ref: ref(A, "a.safetensors"), rule: {} }, { ref: ref(B, "b.safetensors"), rule: {} }] }, ...overrides });

function harness(service) {
  const forwarded = [];
  const route = loraMixMiddleware({ service, readJson: async (request) => request.body, admit: (request) => { if (request.headers.origin === "http://evil") throw Object.assign(new Error("cross-origin"), { statusCode: 403 }); },
    forward: async (pathname, payload, headers, _request, response) => { forwarded.push({ pathname, payload, headers }); response.statusCode = 202; response.end("{}"); } });
  const invoke = async (url, body, headers = {}, method = "POST") => {
    const response = { statusCode: 200, headersSent: false, setHeader() {}, end(text) { this.text = text; } };
    let passed = false;
    await route({ url, method, headers, body }, response, () => { passed = true; });
    return { response, passed };
  };
  return { forwarded, invoke };
}

test("the reference set is the enabled globals then every test member, and malformed shapes are left to Python", () => {
  const references = loraMixReferences(draft());
  assert.deepEqual(references.map((item) => item.instanceId), [G, A, B], "a disabled global is not executed, so it is not resolved");
  assert.ok(references.slice(1).every((item) => item.enabled === true && item.weight === "1"), "members resolve as bindings, never as a weight claim");
  assert.equal(loraMixReferences({ baseline: {}, mix: {} })?.length, 0);
  for (const bad of [null, {}, { baseline: [], mix: {} }, draft({ mix: { members: [{ ref: "x" }] } }), draft({ mix: { members: [{ ref: { value: 3 } }] } }), draft({ baseline: { loras: ["x"] } })]) assert.equal(loraMixReferences(bad), null);
});

test("only plan and confirm are admitted; every other search route is an ordinary proxied call", async () => {
  const { forwarded, invoke } = harness({ resolveReferences: async () => { throw new Error("must not be called"); } });
  for (const [url, method] of [["/api/inference/lora-mix/capabilities", "GET"], ["/api/inference/lora-mix/used", "POST"], ["/api/inference/lora-mix/sessions", "GET"],
    ["/api/inference/lora-mix/sessions/" + "a".repeat(32), "GET"], ["/api/inference/lora-mix/sessions/" + "a".repeat(32) + "/remaining", "GET"], ["/api/inference/lora-mix/experiments/x/release", "POST"], ["/api/inference/xyz-plots", "POST"]]) {
    assert.equal((await invoke(url, {}, {}, method)).passed, true, `${method} ${url}`);
  }
  assert.equal(forwarded.length, 0);
});

test("admission resolves globals and members under the lease, stamps the revision and refuses stale, moved or malformed requests", async () => {
  const calls = [];
  const service = { resolveReferences: async (engine, references) => {
    calls.push({ engine, references });
    return { revision: 7, references: references.map((item) => ({ ...item, value: item.value === "a.safetensors" && globalThis.__moveA ? "concept/a.safetensors" : item.value })), effectiveReferences: references };
  } };
  const { forwarded, invoke } = harness(service);
  let result = await invoke("/api/inference/lora-mix/plan", { draft: draft() }, { "x-xirai-lora-revision": "6" });
  assert.equal(result.response.statusCode, 409); assert.equal(forwarded.length, 0, "a stale revision never reaches Python");
  result = await invoke("/api/inference/lora-mix/plan", { draft: draft() }, { "x-xirai-lora-revision": "7", "x-xirai-lora-context": "[{}]", "x-xirai-lora-chain": "[]" });
  assert.equal(result.response.statusCode, 202); assert.equal(forwarded.length, 1);
  assert.equal(forwarded[0].pathname, "/api/inference/lora-mix/plan");
  assert.equal(forwarded[0].headers["x-xirai-lora-revision"], "7");
  assert.equal(forwarded[0].headers["x-xirai-lora-context"], undefined); assert.equal(forwarded[0].headers["x-xirai-lora-chain"], undefined);
  assert.equal(calls.at(-1).engine, "Anima"); assert.equal(calls.at(-1).references.length, 3);
  const submission = { schema_version: 1, kind: "lora_mix_submit", submission_id: randomUUID(), draft: draft(), on_error: "pause" };
  result = await invoke("/api/inference/lora-mix/sessions", submission, { "x-xirai-lora-revision": "7" });
  assert.equal(result.response.statusCode, 202); assert.equal(forwarded.length, 2);
  assert.deepEqual(forwarded[1].payload, submission, "the body is forwarded untouched");
  globalThis.__moveA = true;
  result = await invoke("/api/inference/lora-mix/sessions", submission, { "x-xirai-lora-revision": "7" });
  delete globalThis.__moveA;
  assert.equal(result.response.statusCode, 409); assert.match(result.response.text, /已被修复重定位或替换/); assert.equal(forwarded.length, 2);
  for (const body of [null, {}, { draft: 3 }, { draft: { mix: {} } }]) {
    result = await invoke("/api/inference/lora-mix/plan", body, { "x-xirai-lora-revision": "7" });
    assert.equal(result.response.statusCode, 400, JSON.stringify(body));
  }
  result = await invoke("/api/inference/lora-mix/plan", { draft: draft() }, { origin: "http://evil", "x-xirai-lora-revision": "7" });
  assert.equal(result.response.statusCode, 403);
});

test("a service failure keeps the draft: unresolved, forged or unavailable identities are refused before Python", async () => {
  const { forwarded, invoke } = harness({ resolveReferences: async () => { throw Object.assign(new Error("LoRA reference is unresolved; choose an explicit target"), { statusCode: 422 }); } });
  let result = await invoke("/api/inference/lora-mix/sessions", { draft: draft() }, { "x-xirai-lora-revision": "1" });
  assert.equal(result.response.statusCode, 422); assert.equal(forwarded.length, 0);
  const down = harness({ resolveReferences: async () => { throw new Error("disk exploded"); } });
  result = await down.invoke("/api/inference/lora-mix/plan", { draft: draft() }, { "x-xirai-lora-revision": "1" });
  assert.equal(result.response.statusCode, 503); assert.match(result.response.text, /search draft retained/); assert.doesNotMatch(result.response.text, /disk exploded/);
});

test("the search admission is registered inside the identity gate, after the job route", async () => {
  const vite = await readFile(new URL("../vite.config.js", import.meta.url), "utf8");
  const gate = vite.indexOf("const registerInferenceIdentityGate"), job = vite.indexOf("server.middlewares.use(loraJobMiddleware("), mix = vite.indexOf("server.middlewares.use(loraMixMiddleware(");
  assert.ok(gate > 0 && job > gate && mix > job, "ordered after the ordinary job route inside the identity gate block");
  assert.match(vite.slice(mix, mix + 600), /admit: requireSameOrigin/);
});
