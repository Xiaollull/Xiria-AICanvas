import assert from "node:assert/strict";
import { test } from "node:test";
import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { spawnSync } from "node:child_process";
import { LoraRelinkService } from "./lora-relink-service.mjs";
import { LoraPublishedStore } from "./lora-published-store.mjs";
import { galleryOwnerParticipant } from "./lora-gallery-owner.mjs";
import { documentDigest, durableJson, publishedLibraryVersion } from "./lora-coordinator.mjs";
import { emptyLoraCardStore, loraCardFor } from "../src/lora-cards.js";
import { rewriteExactLoraTags } from "../src/lora-relink.js";
import { loraJobMiddleware } from "./lora-job-http.mjs";
import { pythonDependencyOptions } from "./python-dependency.test.helper.mjs";

const python = process.env.P10_PYTHON || "python", cwd = path.resolve(import.meta.dirname, "..");
const galleryOptions = pythonDependencyOptions(python, cwd, ["PIL"]);
const galleryTest = (name, body) => test(name, galleryOptions, body);
const parent = process.env.P10_EVIDENCE_DIR || path.join(os.tmpdir(), "opencode", "p10-short-c-fixtures");
const json = async file => JSON.parse(await readFile(file, "utf8"));
function py(code, args) {
  const child = spawnSync(python, ["-B", "-c", code, ...args], { cwd, encoding: "utf8", windowsHide: true });
  assert.equal(child.status, 0, child.stderr); return JSON.parse(child.stdout);
}
async function fixture({ gallery = true, fault } = {}) {
  await mkdir(parent, { recursive: true });
  const directory = await mkdtemp(path.join(parent, "c-service-")), stateDirectory = path.join(directory, "state");
  const root = { engine: "Anima", id: "local-Anima", directory: path.join(directory, "models") };
  await mkdir(path.join(root.directory, "style"), { recursive: true });
  for (const name of ["A", "B"]) await writeFile(path.join(root.directory, `style/${name}.safetensors`), "same complete bytes");
  const entry = { value: "style/A.safetensors", weight: .8137, precision: 4, enabled: true, groupId: "g-a" };
  const state = { theme: { accent: "keep" }, workspace: { model: "Anima", positive: "keep prompt", loras: [entry],
    loraGroupsByEngine: { schemaVersion: 1, byEngine: { Anima: [{ id: "g-a", enabled: false, presetPrompt: "group text", members: [entry] }] } } } };
  const cards = emptyLoraCardStore(); cards.byEngine.Anima.loras[entry.value] = { title: "A presentation", note: "keep", cover: "a".repeat(32) };
  await durableJson(path.join(stateDirectory, "ui-state.json"), state);
  await durableJson(path.join(stateDirectory, "lora-cards/cards.json"), cards);
  const card = gallery ? py("import json,sys; from backend.gallery import GalleryStore; s=GalleryStore(sys.argv[1]); s.create_collection('test'); print(json.dumps(s.create_card('test',title='title',settings=json.loads(sys.argv[2]))))", [stateDirectory, JSON.stringify({ model: "Anima", positive: "gallery text", loras: [entry], loraGroups: [{ id: "run-snapshot" }] })]) : null;
  let busy = false;
  const service = new LoraRelinkService({ stateDirectory, gallery: galleryOwnerParticipant(stateDirectory, python), getRoots: async () => [root],
    admitIdle: async () => { if (busy) throw Object.assign(new Error("queued/active"), { statusCode: 409 }); }, validatePath: async () => {}, fault });
  const store = new LoraPublishedStore({ stateDirectory, ensureRecovered: () => service.ensureRecovered() });
  const initial = await service.scan({ engines: ["Anima"] });
  await service.commit({ scanId: initial.scanId, registerBaseline: true, selections: [], operationId: randomUUID() });
  const registry = await json(path.join(stateDirectory, "lora-library/registry.json"));
  const A = Object.values(registry.instances).find(item => item.value === entry.value), B = Object.values(registry.instances).find(item => item.value.includes("B."));
  await mkdir(path.join(root.directory, "concept"));
  await rename(path.join(root.directory, A.value), path.join(root.directory, "concept/A.safetensors"));
  const scan = await service.scan({ engines: ["Anima"] });
  const payload = { scanId: scan.scanId, operationId: randomUUID(), selections: [{ sourceId: A.instanceId,
    target: { value: "concept/A.safetensors", rootId: root.id, kind: "relocate" }, confirmedOrigin: true, scope: { all: true, history: true } }] };
  return { directory, stateDirectory, root, A, B, state, cards, card, service, store, payload, setBusy: value => { busy = value; } };
}

galleryTest("C whole-document saves merge only unrelated fields and reject stale publication/card reset", async () => {
  const f = await fixture({ gallery: false });
  const basis = await f.store.read("uiState"), left = structuredClone(basis.value), right = structuredClone(basis.value);
  left.theme.accent = "later accent"; right.workspace.positive = "later prompt";
  await f.store.save("uiState", left, basis);
  const merged = await f.store.save("uiState", right, basis);
  assert.equal(merged.value.theme.accent, "later accent"); assert.equal(merged.value.workspace.positive, "later prompt");
  const result = await f.service.commit(f.payload);
  assert.equal(result.status, "published");
  await assert.rejects(f.store.save("uiState", basis.value, basis), /revision changed/);
  const cardBasis = await f.store.read("cards"), raw = await readFile(f.store.file("cards"));
  await assert.rejects(f.store.save("cards", {}, cardBasis), /Unrecognized card/);
  assert.deepEqual(await readFile(f.store.file("cards")), raw);
  const conflictA = structuredClone(merged.value); conflictA.workspace.positive = "competing";
  // Even an unchanged publication needs an actual document version after repair.
  await assert.rejects(f.store.save("uiState", conflictA, { revision: result.revision }), /Versioned save/);
});

galleryTest("C selected card edits invalidate the draft; busy refusal writes no operation", async () => {
  const f = await fixture();
  f.setBusy(true); await assert.rejects(f.service.commit(f.payload), /queued/); f.setBusy(false);
  await assert.rejects(readFile(path.join(f.stateDirectory, `lora-library/operations/${f.payload.operationId}.json`)), { code: "ENOENT" });
  const basis = await f.store.read("cards"), changed = structuredClone(basis.value);
  changed.byEngine.Anima.loras[f.A.value].title = "later card edit";
  await f.store.save("cards", changed, basis);
  await assert.rejects(f.service.preview(f.payload), /Selected user card changed/);
  await assert.rejects(f.service.commit(f.payload), /Selected user card changed/);
  assert.equal((await f.store.read("cards")).value.byEngine.Anima.loras[f.A.value].title, "later card edit");
});

galleryTest("C all-owner undo retains later presentation/weights/prompt/title and keeps card identity index coherent", async () => {
  const f = await fixture(), result = await f.service.commit(f.payload);
  await assert.rejects(f.service.commit({ ...f.payload, selections: [] }), /different draft/);
  assert.equal((await f.service.commit(f.payload)).replay, true);
  const stateBasis = await f.store.read("uiState"), state = structuredClone(stateBasis.value);
  state.workspace.positive = "later user prompt"; state.workspace.loras[0].weight = -.25;
  await f.store.save("uiState", state, stateBasis);
  const cardBasis = await f.store.read("cards"), cards = structuredClone(cardBasis.value), card = cards.byEngine.Anima.loras["concept/A.safetensors"];
  card.title = "later presentation"; cards.byEngine.Anima.byInstance[f.A.instanceId] = card;
  await f.store.save("cards", cards, cardBasis);
  py("import json,sys; from backend.gallery import GalleryStore; s=GalleryStore(sys.argv[1]); c=s._connect(); card=s._fetch_card(c,sys.argv[2]); c.close(); settings=card['settings']; settings['positive']='later gallery'; print(json.dumps(s.update_card(sys.argv[2],title='later title',settings=settings)))", [f.stateDirectory, f.card.id]);
  const payload = { operationId: result.operationId, undoOperationId: randomUUID() }, undone = await f.service.undo(payload);
  assert.deepEqual(undone.conflicts, []); assert.equal((await f.service.undo(payload)).replay, true);
  await assert.rejects(f.service.undo({ ...payload, operationId: randomUUID() }), /different operation/);
  const latest = (await f.store.read("uiState")).value.workspace;
  assert.equal(latest.loras[0].value, f.A.value); assert.equal(latest.loras[0].weight, -.25); assert.equal(latest.positive, "later user prompt");
  const normalized = (await f.store.read("cards")).value;
  assert.equal(loraCardFor(normalized, "Anima", f.A.value).title, "later presentation");
  assert.equal(normalized.byEngine.Anima.byInstance[f.A.instanceId], undefined);
  const gallery = await f.service.access(token => f.service.gallery.list(token));
  assert.equal(gallery.cards[0].title, "later title"); assert.equal(gallery.cards[0].settings.positive, "later gallery");
  assert.deepEqual(gallery.cards[0].settings.loraGroups, [{ id: "run-snapshot" }]);
  assert.equal(gallery.cards[0].settings.loras[0].value, f.A.value);
});

galleryTest("C publication-before-receipt stays closed and ensureRecovered finalizes before another commit", async () => {
  const f = await fixture({ gallery: false });
  f.service.coordinator.fault = async stage => { if (stage === "publication") throw new Error("receipt gap"); };
  await assert.rejects(f.service.commit(f.payload), /receipt gap/);
  await assert.rejects(publishedLibraryVersion(f.stateDirectory), /Unpublished/);
  f.service.coordinator.fault = async () => {};
  assert.equal((await f.store.read("uiState")).revision, 2);
  const record = await json(path.join(f.stateDirectory, `lora-library/operations/${f.payload.operationId}.json`));
  assert.equal(record.phase, "published");
  const nextScan = await f.service.scan({ engines: ["Anima"] });
  assert.equal((await f.service.commit({ scanId: nextScan.scanId, selections: [], registerBaseline: true, operationId: randomUUID() })).revision, 3);
});

galleryTest("C future mirror identity cannot be overwritten by whole UI save or repair snapshot", async () => {
  const f = await fixture({ gallery: false }), file = f.store.file("uiState"), value = await json(file);
  value.workspace.loras[0].identitySchema = 999;
  await durableJson(file, value); const raw = await readFile(file), basis = await f.store.read("uiState");
  await assert.rejects(f.store.save("uiState", f.state, basis), /mirror identity/);
  await assert.rejects(f.service.scan({ engines: ["Anima"] }), /mirror identity/);
  assert.deepEqual(await readFile(file), raw);
  for (const malformed of [[], { workspace: { loraGroupsByEngine: { schemaVersion: "future", byEngine: {} } } },
    { workspace: { mountedLorasByEngine: { schemaVersion: 1, byModel: { legacy: [{ value: "style/A.safetensors", identitySchema: 999 }] } } } }]) {
    await durableJson(file, malformed); const before = await readFile(file);
    if (Array.isArray(malformed)) {
      await assert.rejects(f.store.read("uiState"), /原数据已保留/);
      await assert.rejects(f.store.save("uiState", f.state, { revision: 1 }), /原数据已保留/);
    } else {
      const basis = await f.store.read("uiState");
      await assert.rejects(f.store.save("uiState", f.state, basis), /original state retained/);
    }
    assert.deepEqual(await readFile(file), before);
  }
});

galleryTest("C undo reports later card association conflict and does not restore a whole old card", async () => {
  const f = await fixture({ gallery: false }), result = await f.service.commit(f.payload);
  const basis = await f.store.read("cards"), cards = structuredClone(basis.value), current = cards.byEngine.Anima.loras["concept/A.safetensors"];
  current.originalReference.value = "user-changed-original.safetensors";
  current.note = "later note"; cards.byEngine.Anima.byInstance[f.A.instanceId] = current;
  await f.store.save("cards", cards, basis);
  const undone = await f.service.undo({ operationId: result.operationId, undoOperationId: randomUUID() });
  assert.ok(undone.conflicts.some(item => item.reason === "card-association-edited"));
  const latest = (await f.store.read("cards")).value;
  assert.equal(latest.byEngine.Anima.loras[f.A.value], undefined);
  assert.equal(latest.byEngine.Anima.loras["concept/A.safetensors"].note, "later note");
});

galleryTest("C scoped replacement and owned history undo preserve later rules for another content revision", async () => {
  const f = await fixture({ gallery: false });
  const selected = { ...f.payload, selections: [{ sourceId: f.A.instanceId, target: { value: f.B.value, rootId: f.root.id, kind: "replace" }, scope: { groupIds: ["g-a"] } }] };
  const scoped = await f.service.commit(selected);
  assert.equal((await json(path.join(f.stateDirectory, "lora-library/registry.json"))).substitutions.length, 0);
  const current = (await f.store.read("uiState")).value.workspace;
  assert.equal(current.loras[0].value, f.A.value); assert.equal(current.loraGroupsByEngine.byEngine.Anima[0].members[0].instanceId, f.B.instanceId);
  await f.service.undo({ operationId: scoped.operationId, undoOperationId: randomUUID() });
  const scan = await f.service.scan({ engines: ["Anima"] });
  const all = await f.service.commit({ scanId: scan.scanId, operationId: randomUUID(), selections: [{ sourceId: f.A.instanceId,
    target: { value: f.B.value, rootId: f.root.id, kind: "replace" }, scope: { all: true, history: true } }] });
  const file = path.join(f.stateDirectory, "lora-library/registry.json"), registry = await json(file);
  const later = { engine: "Anima", sourceId: f.A.instanceId, sourceHash: "f".repeat(64), targetId: f.B.instanceId, targetHash: f.B.contentHash, kind: "replace" };
  registry.substitutions.push(later); await durableJson(file, registry);
  const undone = await f.service.undo({ operationId: all.operationId, undoOperationId: randomUUID() });
  assert.deepEqual(undone.conflicts, []);
  assert.deepEqual((await json(file)).substitutions, [later]);
});

test("C unused corrupt registry does not disable ordinary legacy path resolution before first publication", async () => {
  await mkdir(parent, { recursive: true }); const directory = await mkdtemp(path.join(parent, "c-legacy-"));
  await durableJson(path.join(directory, "lora-library/registry.json"), { schemaVersion: 999 });
  const service = new LoraRelinkService({ stateDirectory: directory, gallery: {}, getRoots: async () => [], admitIdle: async () => {}, validatePath: async () => {} });
  assert.equal((await service.resolve("SD", [{ value: "style/ordinary.safetensors", weight: .8 }])).loras[0].path, "style/ordinary.safetensors");
  await assert.rejects(service.resolve("SD", [{ value: "style/ordinary.safetensors", instanceId: randomUUID() }]), /Unsupported/);
});

galleryTest("C explicit recovery refuses future publication without claiming success or rewriting evidence", async () => {
  const f = await fixture({ gallery: false }), file = path.join(f.stateDirectory, "lora-library/publication.json");
  await durableJson(file, { schemaVersion: 999, phase: "published", revision: 1 });
  const raw = await readFile(file);
  await assert.rejects(f.service.coordinator.recover(), /publication/);
  await assert.rejects(f.store.read("uiState"), /publication/);
  assert.deepEqual(await readFile(file), raw);
});

test("D exact tag spans cover named/equal weights without changing whitespace, ambiguous names or invalid capabilities", () => {
  const source = "style/A.safetensors", target = "concept/Renamed.safetensors";
  const input = "notes style/A <lora: style/A :te=0.8137:unet=0.8137> + <lora:A:.8> + <lora:style/A:te=1:unet=2> + <lora:style/A:dyn=4>";
  const result = rewriteExactLoraTags(input, source, target);
  assert.equal(result.spans.length, 1);
  assert.equal(result.text, input.replace("<lora: style/A :", "<lora: concept/Renamed :"));
});

test("C Node job route stamps current resolved evidence; stale/strong-identity conflicts never reach Python", async () => {
  const A = randomUUID(); let submitted = 0, captured;
  const service = { resolve: async (_engine, refs) => {
    if (refs[0].instanceId && refs[0].instanceId !== A) throw Object.assign(new Error("identity-conflicted"), { statusCode: 409 });
    return { revision: 4, loras: [{ path: "concept/A.safetensors", weight: refs[0].weight }], references: [{ instanceId: A, contentHash: "a".repeat(64) }] };
  } };
  const route = loraJobMiddleware({ service, readJson: async req => req.body, admit: () => {}, submit: async (body, headers, _req, res) => { submitted++; captured = { body, headers }; res.end("accepted"); } });
  const invoke = async (headers = {}, loras = [{ path: "style/A.safetensors", weight: .8137 }]) => {
    const response = { statusCode: 202, setHeader() {}, end(text) { this.text = text; } };
    await route({ url: "/api/inference/jobs", method: "POST", headers, body: { engine: "Anima", loras } }, response, () => assert.fail("job route bypassed"));
    return response;
  };
  assert.equal((await invoke()).statusCode, 409); assert.equal(submitted, 0);
  assert.equal((await invoke({ "x-xirai-lora-revision": "4", "x-xirai-lora-context": JSON.stringify([{ instanceId: randomUUID() }]) })).statusCode, 409);
  await invoke({ "x-xirai-lora-revision": "4", "x-xirai-lora-context": JSON.stringify([{ instanceId: A, contentHash: "a".repeat(64) }]) });
  assert.equal(submitted, 1); assert.equal(captured.body.loras[0].path, "concept/A.safetensors"); assert.equal(captured.body.loras[0].weight, .8137);
  assert.equal(JSON.parse(captured.headers["x-xirai-lora-context"])[0].instanceId, A);
  await invoke({}, []); assert.equal(submitted, 2);
});
