import assert from "node:assert/strict";
import { test } from "node:test";
import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, readFile, rename, link, writeFile } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { spawnSync } from "node:child_process";
import { pythonDependencyOptions } from "./python-dependency.test.helper.mjs";
import { scanLoraInventory, registerInitialInventory, registerLegacyReferences, emptyLoraRegistry, matchLoraInventory, assertRegistry, observeLoraFile } from "./lora-library.mjs";
import { planLoraRelink } from "./lora-relink-plan.mjs";
import { LoraCoordinator, documentDigest, durableJson, publishedLibraryVersion, withPublishedLibrary, withLibraryLease } from "./lora-coordinator.mjs";
import { galleryOwnerParticipant } from "./lora-gallery-owner.mjs";
import { createRelinkDraft, editRelinkRow, mergeRelinkSuggestions, selectedRelinkRows, patchLoraList, undoWorkspacePatches } from "../src/lora-relink.js";
import { normalizeMountedLoras, mountedLoraResourcesReady } from "../src/lora-state.js";
import { normalizeLoraGroupMap } from "../src/lora-groups.js";
import { emptyLoraCardStore, normalizeLoraCardStore } from "../src/lora-cards.js";
import { readLoraCardStore, writeLoraCardStore } from "./lora-cards.mjs";
import { emptyMountedLoraMap, normalizeMountedLoraMap, reconcileMountedLoraScan, READY_LORA_ENGINES } from "../src/lora-model-scope.js";
import { resolveLibraryReference } from "../src/lora-identity.js";

const evidence = process.env.P10_EVIDENCE_DIR || path.join(os.tmpdir(), "opencode", "p10-fixtures");
const python = process.env.P10_PYTHON || "python";
const cwd = path.resolve(import.meta.dirname, "..");
async function scratch(label) { await mkdir(evidence, { recursive: true }); return mkdtemp(path.join(evidence, `${label}-`)); }
async function file(root, name, bytes = "same bytes") { const location = path.join(root, name); await mkdir(path.dirname(location), { recursive: true }); await writeFile(location, bytes); }
async function inventory(engine = "Anima") {
  const directory = await scratch("inventory");
  const root = { engine, directory, id: `local-${engine}` };
  for (const name of ["style/Test1/IDA.safetensors", "style/Test2/IDA.safetensors", "style/Test3/IDA.safetensors"]) await file(directory, name);
  const scan = await scanLoraInventory([root]);
  const registry = registerInitialInventory(emptyLoraRegistry(), scan);
  const instances = Object.values(registry.instances);
  return { root, scan, registry, A: instances.find((item) => item.value.includes("Test1")), B: instances.find((item) => item.value.includes("Test2")), C: instances.find((item) => item.value.includes("Test3")) };
}
function workspace(A, B) {
  const mounts = emptyMountedLoraMap();
  const entry = (instance, weight) => ({ value: instance.value, name: "custom", category: "style", weight, precision: 4, enabled: true, groupId: "g-a", instanceId: instance.instanceId, contentHash: instance.contentHash, identitySchema: 1 });
  mounts.byEngine[A.engine] = [entry(A, 0.8137), entry(B, -0.25)];
  return { theme: { accent: "unchanged" }, workspace: { model: A.engine, positive: "keep <lora:IDA:0.8> text", mountedLorasByEngine: mounts, loras: structuredClone(mounts.byEngine[A.engine]),
    loraGroupsByEngine: { schemaVersion: 1, byEngine: { [A.engine]: [{ id: "g-a", enabled: true, presetPrompt: "keep", members: [entry(A, 0.8137)] }, { id: "g-b", enabled: false, members: [entry(A, 1)] }] } } } };
}

test("L01/L02 six-engine registration isolates copies without inspecting weight format", async () => {
  for (const engine of READY_LORA_ENGINES) {
    const { A, B, registry } = await inventory(engine);
    assert.notEqual(A.instanceId, B.instanceId);
    assert.equal(A.contentHash, B.contentHash);
    assert.equal(resolveLibraryReference(registry, engine, A).value, A.value);
    assert.equal(resolveLibraryReference(registry, engine === "SD" ? "iL" : "SD", A).status, "unresolved");
  }
});
test("L07/L08 legacy missing paths and claimed metadata hashes do not manufacture file provenance", async () => {
  const { scan } = await inventory();
  const legacy = registerLegacyReferences(emptyLoraRegistry(), [{ engine: "Anima", rootId: "local-Anima", value: "old/IDA.safetensors", contentHash: "a".repeat(64) }], scan);
  const source = Object.values(legacy.registry.instances)[0];
  assert.equal(source.status, "missing"); assert.equal(source.contentHash, null); assert.equal(source.contentRevision, 0);
  assert.equal(legacy.references[0].contentHash, undefined);
  assert.equal(matchLoraInventory(legacy.registry, scan)[0].confidence, "unresolved");
  assert.equal(matchLoraInventory(legacy.registry, scan)[0].recommendation, null);
  assertRegistry(legacy.registry);
});
test("L03/L04 scan observes root/category/rename moves without committing any baseline", async () => {
  const { root, registry, A, B } = await inventory();
  const original = JSON.stringify(registry);
  let previous = A.value;
  for (const next of ["style/IDA.safetensors", "style/Test4/renamed.safetensors", "concept/IDA.safetensors"]) {
    await mkdir(path.dirname(path.join(root.directory, next)), { recursive: true });
    await rename(path.join(root.directory, previous), path.join(root.directory, next));
    const scan = await scanLoraInventory([root]);
    const row = matchLoraInventory(registry, scan).find((item) => item.sourceId === A.instanceId);
    assert.equal(row.recommendation.value, next);
    if (process.platform === "win32") assert.notEqual(row.confidence, "filesystem-and-content"); // Windows default provider absent.
    assert.equal(matchLoraInventory(registry, scan).find((item) => item.sourceId === B.instanceId).status, "unchanged");
    assert.equal(JSON.stringify(registry), original);
    previous = next;
  }
});
test("L05/L12 local replacement leaves A missing, C separate, disabled/unselected group and cards untouched", async () => {
  const { root, registry, A, B, C } = await inventory();
  await rename(path.join(root.directory, A.value), path.join(root.directory, "outside-library.bin"));
  const scan = await scanLoraInventory([root]);
  const state = workspace(A, B), cards = emptyLoraCardStore();
  cards.byEngine.Anima.loras[A.value] = { title: "A", cover: "a".repeat(32) };
  cards.byEngine.Anima.loras[C.value] = { title: "C" };
  const plan = planLoraRelink({ registry, scan, state, cards, selections: [{ sourceId: A.instanceId, target: { value: C.value, rootId: C.rootId, kind: "replace" }, scope: { groupIds: ["g-a"] } }] });
  assert.equal(plan.registry.instances[A.instanceId].status, "missing");
  assert.equal(plan.registry.instances[C.instanceId].instanceId, C.instanceId);
  assert.equal(plan.state.workspace.loraGroupsByEngine.byEngine.Anima[0].members[0].instanceId, C.instanceId);
  assert.deepEqual(plan.state.workspace.loraGroupsByEngine.byEngine.Anima[1], state.workspace.loraGroupsByEngine.byEngine.Anima[1]);
  assert.deepEqual(plan.state.workspace.loras, state.workspace.loras);
  assert.deepEqual(plan.cards, cards);
  assert.deepEqual(plan.registry.substitutions, []);
  assert.deepEqual(plan.conflicts, []);
});
test("L06 user target/scope/skip survives late recommendations and epoch changes", async () => {
  const row = { sourceId: randomUUID(), recommendation: { value: "a" }, status: "missing" };
  const draft = createRelinkDraft({ scanId: "scan", complete: true }, [row], 1);
  const edited = editRelinkRow(draft, row.sourceId, { target: { value: "c" }, skipped: true, scope: { groupIds: ["g"] } });
  const late = mergeRelinkSuggestions(edited, createRelinkDraft({ scanId: "later", complete: true }, [{ ...row, recommendation: { value: "b" } }], 2));
  assert.equal(late.rows[0].target.value, "c");
  assert.equal(late.rows[0].chosenBy, "user");
  assert.equal(late.rows[0].needsReview, true);
  assert.deepEqual(selectedRelinkRows(late), []);
});
test("L07 multiple same-byte candidates, hardlinks and simulated reused FS ID cannot imply origin", async () => {
  const { root, registry, A } = await inventory();
  registry.instances[A.instanceId].fsIdentity = null; // Explicit unavailable-provider branch.
  await rename(path.join(root.directory, A.value), path.join(root.directory, "style/moved.safetensors"));
  await file(root.directory, "concept/duplicate.safetensors");
  let scan = await scanLoraInventory([root]);
  let row = matchLoraInventory(registry, scan).find((item) => item.sourceId === A.instanceId);
  assert.equal(row.confidence, "ambiguous-content"); assert.equal(row.recommendation, null);
  await link(path.join(root.directory, "style/moved.safetensors"), path.join(root.directory, "style/hardlink.safetensors"));
  scan = await scanLoraInventory([root]);
  const fake = structuredClone(registry); fake.instances[A.instanceId].fsIdentity = { key: "reused", creation: "old", links: 1 };
  scan.observations.forEach((item) => { item.fsIdentity = { key: "reused", creation: "new", links: 2 }; });
  row = matchLoraInventory(fake, scan).find((item) => item.sourceId === A.instanceId);
  assert.notEqual(row.confidence, "filesystem-and-content");
});
test("L08/L09 old path reoccupied with different bytes never overrides instance/hash", async () => {
  const { root, registry, A } = await inventory();
  await rename(path.join(root.directory, A.value), path.join(root.directory, "style/moved.safetensors"));
  await file(root.directory, A.value, "DIFFERENT!");
  await file(root.directory, "style/fake.civitai.info", JSON.stringify({ hash: A.contentHash }));
  const scan = await scanLoraInventory([root]);
  const row = matchLoraInventory(registry, scan).find((item) => item.sourceId === A.instanceId);
  assert.equal(row.status, "identity-conflict"); assert.equal(row.recommendation, null);
  assert.notEqual(scan.observations.find((item) => item.value === A.value).contentHash, A.contentHash);
  const conflict = structuredClone(registry); const id = randomUUID();
  conflict.instances[A.instanceId].status = "missing";
  conflict.instances[id] = { ...A, instanceId: id, contentHash: scan.observations.find((item) => item.value === A.value).contentHash };
  assert.equal(resolveLibraryReference(conflict, A.engine, A).status, "missing");
  assert.equal(resolveLibraryReference(conflict, A.engine, { value: A.value }).status, "ambiguous-path");
});
test("L09 same-byte replacement at old path without native provider is a review conflict, not path proof", async () => {
  const { root, registry, A } = await inventory();
  registry.instances[A.instanceId].fsIdentity = null;
  await rename(path.join(root.directory, A.value), path.join(root.directory, "style/moved.safetensors"));
  await file(root.directory, A.value);
  const scan = await scanLoraInventory([root]);
  const row = matchLoraInventory(registry, scan).find((item) => item.sourceId === A.instanceId);
  assert.equal(row.status, "identity-conflict"); assert.equal(row.recommendation, null);
});
test("L10 scan reconciliation never prunes mounts or rewrites group definitions", () => {
  const container = emptyMountedLoraMap(); container.byEngine.SD = [{ value: "style/missing.safetensors", weight: .937, enabled: true, precision: 4, groupId: "g-a" }];
  const result = reconcileMountedLoraScan({ container, categories: [], scanScopeKey: "SD", requestToken: 1, latestToken: 1, capturedRevision: 0, latestRevision: 0, responseEngine: "SD", activeEngine: "SD" });
  assert.equal(result.applied, true); assert.equal(result.changed, false);
  assert.equal(result.container, container); assert.equal(result.loras[0].value, "style/missing.safetensors");
  assert.equal(mountedLoraResourcesReady(result.loras, []), false);
  assert.equal(mountedLoraResourcesReady(result.loras.map((item) => ({ ...item, enabled: false })), []), true);
  assert.equal(mountedLoraResourcesReady([], [], { identitySupported: false }), false);
});
test("L11 incomplete/cancel/budget/disabled shared roots are not deletion and cannot authorize repair", async () => {
  const { root, registry, A, B } = await inventory();
  for (const options of [{ maxBytes: 1 }, { maxFiles: 1 }, { signal: AbortSignal.abort() }]) {
    const scan = await scanLoraInventory([root], options);
    assert.equal(scan.complete, false); assert.ok(scan.issues.length);
    assert.throws(() => planLoraRelink({ registry, scan, state: workspace(A, B), selections: [] }), /Incomplete/);
  }
  const scan = await scanLoraInventory([{ ...root, id: "123456abcdef", shared: true, enabled: false }]);
  assert.equal(scan.complete, false); assert.equal(scan.issues[0].reason, "root-disabled");
  const missing = await scanLoraInventory([{ ...root, directory: path.join(root.directory, "unavailable") }]);
  assert.equal(missing.complete, false);
});
test("L12 all consumers include mirror, disabled groups, Gallery, but never prompts/run snapshots", async () => {
  const { root, registry, A, B, C } = await inventory();
  await rename(path.join(root.directory, A.value), path.join(root.directory, "not-a-lora.bin"));
  const state = workspace(A, B);
  const gallery = [{ id: "card", settings: { model: A.engine, loras: state.workspace.loras, positive: "untouched", loraGroups: [{ id: "historic" }], imageSeeds: ["42"] } }];
  const plan = planLoraRelink({ registry, scan: await scanLoraInventory([root]), state, galleryCards: gallery, selections: [{ sourceId: A.instanceId, target: { value: C.value, rootId: C.rootId, kind: "replace" }, scope: { all: true, history: true } }] });
  assert.equal(plan.state.workspace.loras[0].instanceId, C.instanceId);
  assert.equal(plan.state.workspace.mountedLorasByEngine.byEngine.Anima[0].instanceId, C.instanceId);
  assert.equal(plan.state.workspace.loraGroupsByEngine.byEngine.Anima[1].members[0].instanceId, C.instanceId);
  assert.equal(plan.state.workspace.positive, state.workspace.positive);
  assert.deepEqual(plan.galleryPatches[0].after.loraGroups, gallery[0].settings.loraGroups);
  assert.equal(plan.registry.substitutions.length, 1);
});
test("L13 duplicate target keeps both entries and exact weights; reports conflict instead of dedup/sum", async () => {
  const { A, B } = await inventory();
  const state = workspace(A, B);
  const patched = patchLoraList(state.workspace.loras, { source: A, target: B, kind: "replace" });
  assert.equal(patched.items.length, 2); assert.equal(patched.items[0].weight, .8137);
  assert.equal(patched.items[1].weight, -.25); assert.equal(patched.conflicts.length, 2);
});
test("L13/L14 same-instance relocation carries own card/cover, no A/C metadata merge", async () => {
  const { root, registry, A, B } = await inventory();
  const next = "concept/renamed.safetensors";
  await mkdir(path.dirname(path.join(root.directory, next)), { recursive: true });
  await rename(path.join(root.directory, A.value), path.join(root.directory, next));
  const cards = emptyLoraCardStore(); cards.byEngine.Anima.loras[A.value] = { title: "user", prompt: "keep", cover: "a".repeat(32) };
  const options = { registry, scan: await scanLoraInventory([root]), state: workspace(A, B), cards,
    selections: [{ sourceId: A.instanceId, target: { value: next, rootId: A.rootId, kind: "relocate" }, confirmedOrigin: true, scope: { all: true, history: true } }] };
  const plan = planLoraRelink(options);
  assert.equal(plan.registry.instances[A.instanceId].value, next);
  assert.equal(plan.cards.byEngine.Anima.loras[next].cover, "a".repeat(32));
  assert.equal(plan.cards.byEngine.Anima.loras[next].prompt, "keep");
  options.cards.byEngine.Anima.loras[next] = { title: "occupied" };
  assert.ok(planLoraRelink(options).conflicts.some((conflict) => conflict.reason === "target-user-card-exists"));
});
test("L14 valid identity round-trips through mounts/groups/cards; future identity retains raw and blocks save", async () => {
  const { A, B } = await inventory();
  const state = workspace(A, B);
  assert.equal(normalizeMountedLoras(state.workspace.loras)[0].instanceId, A.instanceId);
  assert.equal(normalizeLoraGroupMap(state.workspace.loraGroupsByEngine).container.byEngine.Anima[0].members[0].instanceId, A.instanceId);
  const bad = structuredClone(state.workspace.mountedLorasByEngine); bad.byEngine.Anima[0].identitySchema = 999;
  const inspected = normalizeMountedLoraMap(bad); assert.equal(inspected.fatal, true); assert.deepEqual(inspected.raw, bad);
  const directory = await scratch("future-cards"); const store = emptyLoraCardStore(); store.schemaVersion = 999;
  await file(directory, "lora-cards/cards.json", JSON.stringify(store));
  assert.equal((await readLoraCardStore(directory)).fatal, true);
  assert.throws(() => assertRegistry({ schemaVersion: 999 }), /Unsupported/);
  await assert.rejects(writeLoraCardStore(directory, emptyLoraCardStore()), /saving disabled/);
  assert.equal(normalizeLoraCardStore(store).fatal, true);
  assert.equal(JSON.parse(await readFile(path.join(directory, "lora-cards/cards.json"), "utf8")).schemaVersion, 999);
});
test("L16 fresh target rehash detects equal-length bytes changed (no metadata/hash-cache authority)", async () => {
  const { root } = await inventory(); const relative = "style/Test1/IDA.safetensors";
  const first = await observeLoraFile(root, relative);
  await file(root.directory, relative, "evil bytes");
  const second = await observeLoraFile(root, relative, { force: true });
  assert.notEqual(first.contentHash, second.contentHash);
  await assert.rejects(observeLoraFile(root, "../outside.safetensors"), /Invalid/);
  await assert.rejects(observeLoraFile({ ...root, engine: "Bad" }, relative), /Invalid/);
});
test("L18 owned typed undo preserves unrelated edits and detects later weight edits", async () => {
  const { A, B, C } = await inventory(); const original = workspace(A, B);
  const { patchWorkspaceLoras } = await import("../src/lora-relink.js");
  const plan = patchWorkspaceLoras(original, { source: A, target: C, kind: "replace", scope: { all: true } });
  plan.state.workspace.positive = "later prompt";
  const undone = undoWorkspacePatches(plan.state, plan.patches);
  assert.equal(undone.conflicts.length, 0); assert.equal(undone.state.workspace.positive, "later prompt");
  assert.deepEqual(undone.state.workspace.loras, original.workspace.loras);
  plan.state.workspace.loras[0].weight = .5;
  assert.ok(undoWorkspacePatches(plan.state, plan.patches).conflicts.length);
});

function pythonJson(code, args = []) {
  const result = spawnSync(python, ["-B", "-c", code, ...args], { cwd, encoding: "utf8", windowsHide: true });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}
async function transactionFixture() {
  const stateDirectory = await scratch("transaction");
  const before = { model: "Anima", positive: "keep", loras: [{ value: "style/a.safetensors", weight: .8137 }], loraGroups: [{ id: "old-run" }] };
  const card = pythonJson("import json,sys; from backend.gallery import GalleryStore; s=GalleryStore(sys.argv[1]); s.create_collection('test'); print(json.dumps(s.create_card('test', settings=json.loads(sys.argv[2]))))", [stateDirectory, JSON.stringify(before)]);
  const state = { theme: { accent: "custom" }, workspace: { model: "Anima", loras: before.loras, positive: "keep" } };
  await durableJson(path.join(stateDirectory, "ui-state.json"), state);
  const registry = emptyLoraRegistry(), cards = emptyLoraCardStore();
  await durableJson(path.join(stateDirectory, "lora-library", "registry.json"), registry);
  await durableJson(path.join(stateDirectory, "lora-cards", "cards.json"), cards);
  const nextRegistry = { ...registry, revision: 1 };
  const nextCards = structuredClone(cards); nextCards.byEngine.Anima.loras["concept/a.safetensors"] = { title: "own card", prompt: "keep", note: "note", tags: [], cover: "" };
  const after = structuredClone(before); after.loras[0].value = "concept/a.safetensors";
  const nextState = structuredClone(state); nextState.workspace.loras = after.loras;
  const request = { operationId: randomUUID(), expectedRevision: 0,
    files: { registry: { beforeDigest: documentDigest(registry), after: nextRegistry }, uiState: { beforeDigest: documentDigest(state), after: nextState }, cards: { beforeDigest: documentDigest(cards), after: nextCards } }, galleryPatches: [{ cardId: card.id, before, after }] };
  const create = (fault = async () => {}) => new LoraCoordinator({ stateDirectory, gallery: galleryOwnerParticipant(stateDirectory, python), admitIdle: async () => {}, validate: async () => {}, fault });
  return { stateDirectory, request, create, card, before, after };
}
test("L17/L18 real Python Gallery owner + JSON publication, idempotency and stale writer gate", pythonDependencyOptions(python, cwd, ["PIL"]), async () => {
  const fixture = await transactionFixture();
  const coordinator = fixture.create();
  const result = await coordinator.commit(fixture.request);
  assert.equal(result.status, "published"); assert.equal(result.revision, 1);
  assert.deepEqual(await coordinator.commit(fixture.request), result);
  await assert.rejects(coordinator.commit({ ...fixture.request, files: { ...fixture.request.files, cards: { after: {} } } }), /different selection/);
  await assert.rejects(withPublishedLibrary(fixture.stateDirectory, { write: true, expectedRevision: 0 }, async () => assert.fail("late save ran")), /Stale/);
  const card = pythonJson("import json,sys; from backend.gallery import GalleryStore; s=GalleryStore(sys.argv[1]); c=s._connect(); result=s._fetch_card(c,sys.argv[2]); c.close(); print(json.dumps(result))", [fixture.stateDirectory, fixture.card.id]);
  assert.equal(card.settings.loras[0].value, "concept/a.safetensors"); assert.equal(card.settings.loraLibraryRevision, 1);
  assert.deepEqual(card.settings.loraGroups, fixture.before.loraGroups);
  const stale = spawnSync(python, ["-B", "-c", "import json,sys; from backend.gallery import GalleryStore; GalleryStore(sys.argv[1]).update_card(sys.argv[2], settings=json.loads(sys.argv[3]))", fixture.stateDirectory, fixture.card.id, JSON.stringify(fixture.before)], { cwd, encoding: "utf8" });
  assert.notEqual(stale.status, 0); assert.match(stale.stderr, /Stale Gallery/);
});
for (const stage of ["intent", "barrier", "prepared", "file:registry", "file:uiState", "file:cards", "gallery", "publication", "receipt"]) {
  test(`L18 fault/recreate coordinator at ${stage} retains evidence and recovers full publication`, pythonDependencyOptions(python, cwd, ["PIL"]), async () => {
    const fixture = await transactionFixture();
    await assert.rejects(fixture.create(async (at) => { if (at === stage) throw new Error(`fault:${stage}`); }).commit(fixture.request), /fault:/);
    if (!["publication", "receipt"].includes(stage)) await assert.rejects(publishedLibraryVersion(fixture.stateDirectory), /recovery|Unpublished/);
    const recovered = await fixture.create().recover();
    assert.equal(recovered.status, "published"); assert.equal(recovered.revision, 1);
    assert.equal((await fixture.create().commit(fixture.request)).revision, 1);
    assert.equal((JSON.parse(await readFile(path.join(fixture.stateDirectory, "ui-state.json"), "utf8"))).workspace.loras[0].value, "concept/a.safetensors");
  });
}
test("L16/L19 stale CAS / idle and validation refusals happen before journal or publication", pythonDependencyOptions(python, cwd, ["PIL"]), async () => {
  const fixture = await transactionFixture();
  const bad = structuredClone(fixture.request); bad.files.uiState.beforeDigest = "bad";
  await assert.rejects(fixture.create().commit(bad), /Stale uiState/);
  assert.equal(await publishedLibraryVersion(fixture.stateDirectory), 0);
  const blocked = new LoraCoordinator({ stateDirectory: fixture.stateDirectory, gallery: galleryOwnerParticipant(fixture.stateDirectory, python), admitIdle: async () => { throw new Error("queued job"); }, validate: async () => {} });
  await assert.rejects(blocked.commit(fixture.request), /queued/);
  const unavailable = new LoraCoordinator({ stateDirectory: fixture.stateDirectory, gallery: galleryOwnerParticipant(fixture.stateDirectory, python), admitIdle: async () => {}, validate: async () => { throw new Error("target changed"); } });
  await assert.rejects(unavailable.commit(fixture.request), /target changed/);
  const noOp = await fixture.create().commit({ operationId: randomUUID(), expectedRevision: 0, files: {}, galleryPatches: [] });
  assert.equal(noOp.status, "no-op"); assert.equal(noOp.operationId, null);
});
test("L17 cross-process owner refuses independent Gallery reads/writes while Node holds lease", pythonDependencyOptions(python, cwd, ["PIL"]), async () => {
  const fixture = await transactionFixture();
  await withLibraryLease(fixture.stateDirectory, async () => {
    const probe = spawnSync(python, ["-B", "-c", "import sys; from backend.gallery import GalleryStore; GalleryStore(sys.argv[1])", fixture.stateDirectory], { cwd, encoding: "utf8" });
    assert.notEqual(probe.status, 0); assert.match(probe.stderr, /busy or recovering/);
  });
});

test("L03/L07 real native OS File ID proves bounded unique rename evidence, hardlink falls back", async () => {
  const { root } = await inventory();
  const fileIdentity = async (file) => pythonJson("import json,sys; from backend.lora_file_identity import file_identity; print(json.dumps(file_identity(sys.argv[1])))", [file]);
  const initial = await scanLoraInventory([root], { fileIdentity });
  const registry = registerInitialInventory(emptyLoraRegistry(), initial);
  const A = Object.values(registry.instances).find((item) => item.value.includes("Test1"));
  assert.ok(A.fsIdentity);
  const target = "concept/renamed.safetensors";
  await mkdir(path.dirname(path.join(root.directory, target)), { recursive: true });
  await rename(path.join(root.directory, A.value), path.join(root.directory, target));
  let scan = await scanLoraInventory([root], { fileIdentity });
  assert.equal(matchLoraInventory(registry, scan).find((item) => item.sourceId === A.instanceId).confidence, "filesystem-and-content");
  await link(path.join(root.directory, target), path.join(root.directory, "style/hardlinked.safetensors"));
  scan = await scanLoraInventory([root], { fileIdentity });
  assert.notEqual(matchLoraInventory(registry, scan).find((item) => item.sourceId === A.instanceId).confidence, "filesystem-and-content");
});
for (const stage of ["intent", "barrier", "prepared", "file:registry", "file:uiState", "file:cards", "gallery", "publication", "receipt"]) {
  test(`L18 actual child process exit at ${stage}: dead lease retained/quarantined and restarted recovery`, pythonDependencyOptions(python, cwd, ["PIL"]), async () => {
    const fixture = await transactionFixture();
    const fixtureFile = path.join(fixture.stateDirectory, "crash-request.json");
    await durableJson(fixtureFile, { stateDirectory: fixture.stateDirectory, request: fixture.request });
    const crash = spawnSync(process.execPath, ["scripts/lora-relink-crash-probe.mjs"], { cwd, encoding: "utf8", windowsHide: true,
      env: { ...process.env, P10_CRASH_REQUEST: fixtureFile, P10_CRASH_STAGE: stage, P10_PYTHON: python } });
    await durableJson(path.join(fixture.stateDirectory, "crash-result.json"), { stage, status: crash.status, stdout: crash.stdout, stderr: crash.stderr });
    assert.equal(crash.status, 73, crash.stderr);
    const recovered = await fixture.create().recover();
    assert.equal(recovered.revision, 1); assert.equal(await publishedLibraryVersion(fixture.stateDirectory), 1);
    assert.equal((await fixture.create().commit(fixture.request)).revision, 1);
  });
}
