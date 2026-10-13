import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import os from "node:os"; import path from "node:path";
import { assertUiStateValue, admitUiStateResponse } from "../src/ui-state-authority.js";
import { LoraPublishedStore } from "./lora-published-store.mjs";

test("authority distinguishes explicit absence, legal empty/legacy/schema3, bad root/payload/future map", () => {
  assert.equal(admitUiStateResponse({ state: null, exists: false }).status, "confirmed-absent");
  for (const state of [{}, { theme: {} }, { workspace: {} }, { workspace: { model: "Anima", positive: "legacy" } }, { workspace: { engineSettingsByEngine: { schemaVersion: 3, byEngine: {} } } }]) assert.equal(admitUiStateResponse({ state }).status, "loaded");
   for (const payload of [null, [], {}, { state: null }, { state: [] }, { state: 1 }, { state: {} , exists: false }, { state: { unrelated: true } }, { state: { workspace: null } }, { state: { workspace: [] } }, { state: { workspace: { engineSettingsByEngine: { schemaVersion: 4, byEngine: {} } } } }, { state: { workspace: { engineSettingsByEngine: { schemaVersion: 3, byEngine: { Anima: [] } } } } }]) assert.throws(() => admitUiStateResponse(payload));
  const fatalRaw = { workspace: { mountedLorasByEngine: { schemaVersion: 999, raw: "retain" }, promptPresets: { schemaVersion: 999, raw: "retain" } } };
  assert.equal(assertUiStateValue(fatalRaw), fatalRaw, "individual fatal containers remain raw for their owners");
});
test("observe-only retry validates fresh GET but preserves the original CAS/local basis", async () => {
const native = globalThis.fetch;
  try {
    const client = await import('../src/lora-library-client.js?observe-only');
    const original = { workspace: { positive: "basis" } };
    globalThis.fetch = async () => new Response(JSON.stringify({ state: original }), { headers: { "X-XirAI-LoRA-Revision": "0", "X-XirAI-Document-Version": "old" } });
    await client.loraFetch('/api/ui-state');
    globalThis.fetch = async () => new Response(JSON.stringify({ state: { workspace: { positive: "other user" } } }), { headers: { "X-XirAI-LoRA-Revision": "5", "X-XirAI-Document-Version": "fresh" } });
    await client.loraFetch('/api/ui-state', { loraObserveOnly: true });
    assert.deepEqual(client.publishedUiSnapshot(), original); assert.equal(client.currentLoraRevision(), 0);
    let headers;
    globalThis.fetch = async (_url, options) => { headers = options.headers; assert.equal('loraObserveOnly' in options, false); return new Response('{}', { status: 409 }); };
    await client.loraFetch('/api/ui-state', { method: 'PUT', body: '{"workspace":{}}' });
    assert.equal(headers.get('X-XirAI-Document-Version'), 'old');
    await assert.rejects(client.loraFetch('/api/ui-state', { method: 'PUT', loraObserveOnly: true }));
  } finally { globalThis.fetch = native; }
});
test("hydration opens the write gate only after the committed React snapshot, never inside GET settlement", async () => {
  const source = await readFile(new URL('../src/App.jsx', import.meta.url), 'utf8');
  const load = source.slice(source.indexOf('const timeout = window.setTimeout(() => controller.abort(), 10000)'), source.indexOf('}, [uiStateLoadAttempt]);'));
  assert.doesNotMatch(load, /uiStateWritableRef\.current = true/);
  assert.match(source, /useEffect\(\(\) => \{\s*uiStateWritableRef\.current = uiStateReady;/);
});
test("late JSON body and cancelled GET cannot replace a newer admitted receipt", async () => {
  const native = globalThis.fetch;
  try {
    const client = await import('../src/lora-library-client.js?late-receipt');
    let release, started;
    const entered = new Promise(r => started = r);
    globalThis.fetch = async () => ({ ok: true, headers: new Headers(), clone: () => ({ json: async () => { started(); return new Promise(r => release = r); } }) });
    const old = client.loraFetch('/api/ui-state'); const rejection = assert.rejects(old, /迟到/); await entered;
    globalThis.fetch = async () => new Response('{"state":{"workspace":{"positive":"new"}}}');
    await client.loraFetch('/api/ui-state'); release({ state: { workspace: { positive: 'old' } } }); await rejection;
    assert.equal(client.publishedUiSnapshot().workspace.positive, 'new');
    const controller = new AbortController(); controller.abort();
    await assert.rejects(client.loraFetch('/api/ui-state', { signal: controller.signal }));
    assert.equal(client.publishedUiSnapshot().workspace.positive, 'new');
  } finally { globalThis.fetch = native; }
});
test("real published read only marks ENOENT absent; bad JSON/null/array/wrong shape remain unchanged", async () => {
  const dir = await mkdtemp(path.join(process.env.UI_STATE_EVIDENCE_DIR || path.join(os.tmpdir(), "opencode"), "ui-state-owner-"));
  const store = new LoraPublishedStore({ stateDirectory: dir }), file = path.join(dir, "ui-state.json");
  assert.equal((await store.read("uiState")).exists, false);
  for (const text of ['{"broken":', 'null', '[]', '{"unrelated":true}', '{"workspace":[]}']) {
    await writeFile(file, text); await assert.rejects(store.read("uiState"));
    await assert.rejects(store.save("uiState", { workspace: {} })); assert.equal(await readFile(file, "utf8"), text);
  }
  await writeFile(file, '{}'); assert.deepEqual((await store.read("uiState")).value, {});
});
test("readJsonIfPresent explicitly propagates EACCES rather than absence (mock I/O)", async () => {
  const text = await readFile(new URL("./lora-coordinator.mjs", import.meta.url), "utf8");
  const part = text.slice(text.indexOf("export async function readJsonIfPresent"), text.indexOf("export function coordinationPaths"));
  const read = new Function("readFile", part.replace("export ", "") + ";return readJsonIfPresent;")(async () => { throw Object.assign(Error("denied"), { code: "EACCES" }); });
  await assert.rejects(read("fixture", "absent"), { code: "EACCES" });
});
test("transport refuses invalid receipt/version, preserves ordinary path-only resolve and acknowledgements", async () => {
  const native = globalThis.fetch;
  try {
    for (const body of ['{"state":', '{}', '{"state":null}', '{"state":[]}']) {
      const client = await import(`../src/lora-library-client.js?invalid=${encodeURIComponent(body)}`);
      globalThis.fetch = async () => new Response(body, { headers: { "X-XirAI-LoRA-Revision": "5", "X-XirAI-Document-Version": "fixture-version" } });
      await assert.rejects(client.loraFetch("/api/ui-state"));
      assert.equal(client.publishedUiSnapshot(), undefined); assert.equal(client.currentLoraRevision(), 0);
      let submitted;
      globalThis.fetch = async (_url, opts) => { submitted = opts.headers; return new Response('{"error":"conflict"}', { status: 409 }); };
      await client.loraFetch("/api/ui-state", { method: "PUT", body: '{"workspace":{}}' });
      assert.equal(submitted.get("X-XirAI-Document-Version"), null);
    }
    const client = await import('../src/lora-library-client.js?valid-receipt');
    const legacy = { workspace: { positive: "sentinel", size: { width: 768, height: 1344 } } };
    globalThis.fetch = async () => new Response(JSON.stringify({ state: legacy }), { headers: { "X-XirAI-LoRA-Revision": "5", "X-XirAI-Document-Version": "v" } });
    await client.loraFetch("/api/ui-state"); assert.deepEqual(client.publishedUiSnapshot(), legacy);
    globalThis.fetch = async () => new Response(JSON.stringify({ revision: 5, references: [{ value: "style/A", status: "unresolved" }], loras: [] }));
    assert.equal((await client.resolveLoraReferences("Anima", [{ value: "style/A" }])).references[0].status, "unresolved");
  } finally { globalThis.fetch = native; }
});
