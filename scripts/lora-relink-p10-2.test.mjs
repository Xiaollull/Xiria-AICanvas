import assert from "node:assert/strict";
import { test } from "node:test";
import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, readFile, rename, rm, writeFile, symlink, readdir } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { readLoraRegistry, registerObservedInventory, scanLoraInventory, emptyLoraRegistry, registerInitialInventory } from "./lora-library.mjs";
import { LoraRelinkService } from "./lora-relink-service.mjs";
import { galleryOwnerParticipant } from "./lora-gallery-owner.mjs";
import { durableJson, publishedLibraryVersion } from "./lora-coordinator.mjs";
import { LoraPublishedStore } from "./lora-published-store.mjs";
import { resolveRepairDirectory, listRepairDirectories, withinPrefix } from "./lora-repair-scope.mjs";
import { planScopedReset } from "./lora-maintenance-reset.mjs";
import { createRelinkDraft, relinkDraftSummary, selectedRelinkRows } from "../src/lora-relink.js";
import { createLoraRepairController } from "../src/lora-repair-controller.js";
import { emptyMountedLoraMap } from "../src/lora-model-scope.js";
import { pythonDependencyOptions } from "./python-dependency.test.helper.mjs";

// P10.2 S01–S25 contracts on isolated fixtures (never the user's files).
const python = process.env.P10_PYTHON || "python", cwd = path.resolve(import.meta.dirname, "..");
const galleryOptions = pythonDependencyOptions(python, cwd, ["PIL"]);
const parent = process.env.P10_EVIDENCE_DIR || path.join(os.tmpdir(), "opencode", "p10.2-fixtures");
const SHARED = "0a1b2c3d4e5f";
const NAMES = ["ashima_v4", "galaxy", "joezunzun_v2", "kiriaina_anima_style-000030", "yusano_v5_step1500"];
async function scratch(label) { await mkdir(parent, { recursive: true }); return mkdtemp(path.join(parent, `${label}-`)); }
async function put(root, name, bytes) { const location = path.join(root, name); await mkdir(path.dirname(location), { recursive: true }); await writeFile(location, bytes); }
function py(code, args) { const child = spawnSync(python, ["-B", "-c", code, ...args], { cwd, encoding: "utf8", windowsHide: true }); assert.equal(child.status, 0, child.stderr); return JSON.parse(child.stdout); }
const galleryCard = (stateDirectory, settings) => py("import json,sys; from backend.gallery import GalleryStore; s=GalleryStore(sys.argv[1]); s.create_collection('t'); print(json.dumps(s.create_card('t',title='card title',settings=json.loads(sys.argv[2]))))", [stateDirectory, JSON.stringify(settings)]);
const readCard = (stateDirectory, id) => py("import json,sys; from backend.gallery import GalleryStore; s=GalleryStore(sys.argv[1]); c=s._connect(); card=s._fetch_card(c,sys.argv[2]); c.close(); print(json.dumps({'title':card['title'],'settings':card['settings']}))", [stateDirectory, id]);
const entry = (value, weight = 0.5, extra = {}) => ({ value, name: value.split("/").at(-1).replace(".safetensors", ""), category: value.split("/")[0], weight, precision: 4, enabled: true, ...extra });

/** Files back in style/, saved references still in style/NKStyleV3/ (the user's manual-test state). */
async function backInStyle({ gallery = false, extraRefs = [] } = {}) {
  const directory = await scratch("scope"), stateDirectory = path.join(directory, "state"), project = path.join(directory, "project");
  const local = { engine: "Anima", id: "local-Anima", directory: path.join(project, "models", "loras", "anima") };
  const flux = { engine: "Flux", id: "local-Flux", directory: path.join(project, "models", "loras", "flux") };
  const shared = { engine: "Anima", id: SHARED, directory: path.join(directory, "comfy"), shared: true, label: "ComfyUI LoRA" };
  for (const name of NAMES) { await put(local.directory, `style/${name}.safetensors`, `bytes ${name}`); await put(shared.directory, `Anima/Style/${name}.safetensors`, `bytes ${name}`); }
  await put(local.directory, "style/4x2style_v2_epoch32.safetensors", "unrelated sixth");
  await mkdir(path.join(local.directory, "style", "NKStyleV3"), { recursive: true });
  await put(local.directory, "concept/keep.safetensors", "keep");
  await put(flux.directory, "style/fluxonly.safetensors", "flux");
  const mounts = emptyMountedLoraMap();
  mounts.byEngine.Anima = [...NAMES.map((name, index) => entry(`style/NKStyleV3/${name}.safetensors`, [0.8137, -0.25, 1, 0.5, 0.65][index])), entry("concept/keep.safetensors", 0.3), ...extraRefs];
  const state = { theme: { accent: "keep" }, workspace: { model: "Anima", positive: "keep prompt", mountedLorasByEngine: mounts, loras: structuredClone(mounts.byEngine.Anima),
    loraGroupsByEngine: { schemaVersion: 1, byEngine: { Anima: [{ id: "g1", name: "Style mix", enabled: true, members: [entry("style/NKStyleV3/galaxy.safetensors", -0.25), entry("concept/keep.safetensors", 0.3), ...extraRefs] }] } } } };
  await durableJson(path.join(stateDirectory, "ui-state.json"), state);
  const card = gallery ? galleryCard(stateDirectory, { model: "Anima", positive: "gallery text", loras: [entry("style/NKStyleV3/galaxy.safetensors", -0.25), ...extraRefs] }) : null;
  const roots = [local, flux, shared];
  const service = new LoraRelinkService({ stateDirectory, projectRoot: project, gallery: galleryOwnerParticipant(stateDirectory, python), getRoots: async () => roots, admitIdle: async () => {}, validatePath: async () => {} });
  return { directory, stateDirectory, project, local, flux, shared, roots, state, card, service };
}
const pending = (scan) => scan.rows.filter((row) => row.status !== "unchanged" && row.lifecycle === "pending");
const nameOf = (value) => value.split("/").at(-1).replace(".safetensors", "");
const select = (draft) => draft.rows.map((row) => ({ sourceId: row.sourceId, target: row.target, scope: row.scope, skipped: row.skipped, confirmedOrigin: row.target?.kind === "relocate" }));

test("S01/S02/S03 directories normalize from engine-relative, project-relative, absolute and picked forms; errors never switch engines", async () => {
  const f = await backInStyle();
  const options = { roots: f.roots, engines: ["Anima"], projectRoot: f.project };
  const expectEntry = (resolved, prefix) => assert.deepEqual(resolved.entries.map(({ engine, rootId, prefix: p }) => [engine, rootId, p]), [["Anima", "local-Anima", prefix]]);
  expectEntry(await resolveRepairDirectory("style/", { ...options, role: "target" }), "style");
  expectEntry(await resolveRepairDirectory("style\\NKStyleV3\\", { ...options, role: "target" }), "style/NKStyleV3");
  expectEntry(await resolveRepairDirectory("models/loras/anima/style", { ...options, role: "target" }), "style");
  expectEntry(await resolveRepairDirectory(path.join(f.local.directory, "style"), { ...options, role: "target" }), "style");
  expectEntry(await resolveRepairDirectory({ rootId: "local-Anima", path: "style" }, { ...options, role: "target" }), "style");
  const sharedPick = await resolveRepairDirectory(`shared:${SHARED}/Anima/Style`, { ...options, role: "target" });
  assert.deepEqual(sharedPick.entries.map((item) => [item.rootId, item.prefix, item.label]), [[SHARED, "Anima/Style", "ComfyUI LoRA"]]);
  // Missing source folders are legal (moved away); missing targets are not.
  assert.equal((await resolveRepairDirectory("style/OldFolder", { ...options, role: "source" })).entries[0].exists, false);
  await assert.rejects(resolveRepairDirectory("style/OldFolder", { ...options, role: "target" }), /不存在/);
  await assert.rejects(resolveRepairDirectory(path.join(f.flux.directory, "style"), { ...options, role: "target" }), /属于 Flux.*不属于 Anima/);
  await assert.rejects(resolveRepairDirectory("style/../../etc", { ...options, role: "source" }), /无效/);
  await assert.rejects(resolveRepairDirectory("\\\\server\\share\\loras", { ...options, role: "target" }), /网络或设备/);
  await assert.rejects(resolveRepairDirectory(os.tmpdir(), { ...options, role: "target" }), /不在已登记/);
  await assert.rejects(resolveRepairDirectory({ rootId: "local-Flux", path: "style" }, { ...options, role: "target" }), /不适用/);
  const checked = await f.service.scope({ engines: ["Anima"], target: "models/loras/flux/style" });
  assert.equal(checked.target.ok, false); assert.match(checked.target.error, /属于 Flux/);
  assert.equal((await f.service.scope({ engines: ["Anima"], source: "style/Gone" })).source.entries[0].exists, false);
  assert.equal(withinPrefix("style2/a.safetensors", "style"), false);
  assert.equal(withinPrefix("style/NKStyleV3/a.safetensors", "style"), true);
  const listed = await listRepairDirectories(f.roots, ["Anima"]);
  assert.ok(listed.find((root) => root.rootId === "local-Anima").directories.includes("style/NKStyleV3"), "empty folders are navigable");
  assert.ok(!listed.some((root) => root.engine === "Flux"));
  const all = await resolveRepairDirectory("style", { ...options, engines: ["Anima", "Flux"], role: "target" });
  assert.deepEqual(all.entries.map((item) => item.engine).sort(), ["Anima", "Flux"], "All engines expands through each real root");
  if (process.platform === "linux") {
    await symlink(os.tmpdir(), path.join(f.local.directory, "escape"));
    await assert.rejects(resolveRepairDirectory("escape", { ...options, role: "target" }), /链接|越出/);
  }
});

test("S05/S06/S07/S10 one scoped scan prefills the five inside the target folder only; coverage is narrowed; nothing written", galleryOptions, async () => {
  const f = await backInStyle();
  const before = await readFile(path.join(f.stateDirectory, "ui-state.json"));
  const scan = await f.service.scan({ engines: ["Anima"], source: "style/NKStyleV3", target: "style" });
  assert.equal(scan.domain.scoped, true); assert.equal(scan.domain.fullRoots, false);
  assert.ok(scan.roots.every((root) => root.rootId !== SHARED), "shared root outside the target is not observed");
  const rows = pending(scan);
  assert.deepEqual(rows.map((row) => nameOf(row.sourceValue)).sort(), [...NAMES].sort());
  for (const row of rows) {
    assert.equal(row.recommendation.value, `style/${nameOf(row.sourceValue)}.safetensors`);
    assert.ok(row.candidates.every((item) => item.rootId === "local-Anima" && item.value.startsWith("style/")), "no shared/outside candidates");
  }
  assert.ok(!scan.rows.some((row) => row.sourceValue.includes("4x2style") && row.status !== "unchanged"), "unrelated sixth file is not a source");
  assert.ok(!scan.rows.some((row) => row.sourceValue === "concept/keep.safetensors"), "outside the source folder is not a row");
  assert.ok(scan.observations.every((item) => item.value.startsWith("style/")), "only the declared domain was observed");
  assert.deepEqual(await readFile(path.join(f.stateDirectory, "ui-state.json")), before);
  await assert.rejects(readFile(path.join(f.stateDirectory, "lora-library/registry.json")), { code: "ENOENT" });
  const draft = createRelinkDraft(scan, scan.rows, scan.revision);
  assert.equal(relinkDraftSummary(draft).unselected, 0);
  // S07: another compatible LoRA in the folder is allowed; outside the folder is refused, typed or picked.
  const galaxy = draft.rows.find((row) => nameOf(row.sourceValue) === "galaxy");
  const choose = (target) => ({ scanId: scan.scanId, selections: select(draft).map((item) => item.sourceId === galaxy.sourceId ? { ...item, target } : item) });
  const preview = await f.service.preview(choose({ value: "style/4x2style_v2_epoch32.safetensors", rootId: "local-Anima", kind: "replace" }));
  assert.deepEqual(preview.conflicts, []);
  await assert.rejects(f.service.preview(choose({ value: "concept/keep.safetensors", rootId: "local-Anima", kind: "replace" })), /目标不在所选目标目录内/);
  await assert.rejects(f.service.preview(choose({ value: `shared:${SHARED}/Anima/Style/galaxy.safetensors`, rootId: SHARED, kind: "replace" })), /目标不在所选目标目录内/);
  await assert.rejects(f.service.preview({ ...choose(galaxy.target), registerBaseline: true }), /完整观察基线需要覆盖全部根目录/);
  // Blank target keeps library-wide ranking with shared alternatives (P10.1 behaviour).
  const wide = await f.service.scan({ engines: ["Anima"], source: "style/NKStyleV3" });
  assert.ok(pending(wide).every((row) => row.candidates.some((item) => item.rootId === SHARED)));
});

test("S10 a narrowed folder scan never marks outside instances missing; baseline refresh only from full roots", galleryOptions, async () => {
  const f = await backInStyle();
  const full = await f.service.scan({ engines: ["Anima"] });
  await f.service.commit({ scanId: full.scanId, selections: [], registerBaseline: true, operationId: randomUUID() });
  const keep = Object.values((await readLoraRegistry(f.stateDirectory)).instances).find((item) => item.value === "concept/keep.safetensors");
  const narrow = await f.service.scan({ engines: ["Anima"], source: "style/NKStyleV3", target: "style" });
  const session = f.service.session(narrow.scanId);
  const refreshed = registerObservedInventory(session.registry, session.scan);
  assert.equal(refreshed.instances[keep.instanceId].status, "present", "outside the observed prefixes stays as it was");
  assert.ok(!narrow.rows.some((row) => row.sourceId === keep.instanceId));
});

test("S08 changing engine/source/target invalidates preview and confirmation until a rescan", async () => {
  const calls = [];
  const response = (value) => ({ ok: true, json: async () => value });
  const controller = createLoraRepairController({ request: async (url, options) => {
    calls.push([url, options?.body && JSON.parse(options.body)]);
    if (url.endsWith("/state")) return response({ available: true, contractVersion: 1 });
    if (url.endsWith("/scan")) return response({ scanId: `s${calls.length}`, complete: true, revision: 1, observations: [], rows: [{ sourceId: "a", status: "missing", lifecycle: "pending", recommendation: { value: "style/A", rootId: "local-Anima", kind: "replace" }, impacts: { mounts: 1, groups: [], cards: [] } }] });
    if (url.endsWith("/preview")) return response({ conflicts: [], results: [{ evidence: "unverified-source" }], affected: { workspace: 1, cards: 0, gallery: 0 } });
    return response({ operationId: "op", status: "published", revision: 2 });
  } });
  await controller.availability();
  const params = { engines: ["Anima"], source: "style/NKStyleV3", target: { rootId: "local-Anima", path: "style" } };
  controller.setParams(params); await controller.scan(params);
  assert.deepEqual(calls.find(([url]) => url.endsWith("/scan"))[1], params);
  await controller.preview(); assert.ok(controller.state().preview);
  controller.setParams({ ...params, target: "" });
  assert.equal(controller.state().staleParams, true); assert.equal(controller.state().preview, null);
  await controller.preview(); await controller.commit();
  assert.ok(!calls.some(([url]) => url.endsWith("/commit")), "no confirmation with stale parameters");
  controller.setParams(params);
  assert.equal(controller.state().staleParams, false, "returning to the scanned parameters is consistent again");
});

test("S12/S13 settled sources leave the pending list after repair, restart and entry change; partial repair keeps real consumers; a moved target becomes the primary issue", galleryOptions, async () => {
  const f = await backInStyle({ gallery: true });
  let scan = await f.service.scan({ engines: ["Anima"], source: "style/NKStyleV3", target: "style" });
  let draft = createRelinkDraft(scan, scan.rows, scan.revision);
  // Partial for galaxy: mounts only; its group and Gallery card still use the old path.
  const galaxy = draft.rows.find((row) => nameOf(row.sourceValue) === "galaxy");
  const selections = select(draft).map((item) => item.sourceId === galaxy.sourceId ? { ...item, scope: { mounts: true } } : item);
  const receipt = await f.service.commit({ scanId: scan.scanId, selections, operationId: randomUUID() });
  assert.equal(receipt.status, "published");
  const restarted = new LoraRelinkService({ stateDirectory: f.stateDirectory, projectRoot: f.project, gallery: galleryOwnerParticipant(f.stateDirectory, python), getRoots: async () => f.roots, admitIdle: async () => {}, validatePath: async () => {} });
  for (const service of [f.service, restarted]) {
    scan = await service.scan({ engines: ["Anima"], source: "style/NKStyleV3", target: "style" });
    const open = pending(scan);
    assert.deepEqual(open.map((row) => nameOf(row.sourceValue)), ["galaxy"], "only the partial repair remains pending");
    assert.equal(open[0].impacts.groups.length + open[0].impacts.cards.length, 2);
    draft = createRelinkDraft(scan, scan.rows, scan.revision);
    assert.equal(draft.processed.filter((row) => row.lifecycle === "settled").length, 4);
    assert.equal(relinkDraftSummary(draft).sources, 1);
  }
  // Whole-library scan with no folders agrees (entry surfaces send different scopes).
  const wide = await restarted.scan({ engines: ["Anima"] });
  assert.deepEqual(pending(wide).map((row) => nameOf(row.sourceValue)), ["galaxy"]);
  // Move a repaired target away: the target instance is now the primary pending problem; its old source is history.
  await mkdir(path.join(f.local.directory, "style", "Moved"), { recursive: true });
  await rename(path.join(f.local.directory, "style", "ashima_v4.safetensors"), path.join(f.local.directory, "style", "Moved", "ashima_v4.safetensors"));
  const moved = await restarted.scan({ engines: ["Anima"] });
  const ashima = moved.rows.filter((row) => nameOf(row.sourceValue) === "ashima_v4" && row.status !== "unchanged");
  assert.deepEqual(ashima.filter((row) => row.lifecycle === "pending").map((row) => row.sourceValue), ["style/ashima_v4.safetensors"]);
  assert.equal(ashima.find((row) => row.sourceValue.includes("NKStyleV3")).lifecycle, "history");
  assert.equal(readCard(f.stateDirectory, f.card.id).settings.loras[0].value, "style/NKStyleV3/galaxy.safetensors", "out-of-scope Gallery use untouched");
});

test("S11/S17/S18/S19 confirmed deletion cleans only the chosen LoRA, archives it, survives rescans, refuses incomplete scans and undoes owned changes", galleryOptions, async () => {
  const gone = entry("style/gone_v1.safetensors", 0.9);
  const f = await backInStyle({ gallery: true, extraRefs: [gone] });
  let scan = await f.service.scan({ engines: ["Anima"] });
  const row = pending(scan).find((item) => item.sourceValue === gone.value);
  assert.equal(row.candidateCount, 0);
  const deletion = { scanId: scan.scanId, selections: [{ sourceId: row.sourceId, target: { kind: "delete" }, scope: { all: true } }] };
  const preview = await f.service.preview(deletion);
  assert.deepEqual(preview.results.map((item) => [item.kind, item.evidence, item.removed]), [["delete", "confirmed-deleted", 4]]);
  assert.deepEqual(JSON.parse(await readFile(path.join(f.stateDirectory, "ui-state.json"))).workspace.loras.at(-1).value, gone.value, "preview writes nothing");
  const receipt = await f.service.commit({ ...deletion, operationId: randomUUID() });
  const ws = JSON.parse(await readFile(path.join(f.stateDirectory, "ui-state.json"))).workspace;
  for (const list of [ws.mountedLorasByEngine.byEngine.Anima, ws.loras, ws.loraGroupsByEngine.byEngine.Anima[0].members]) assert.ok(!list.some((item) => item.value === gone.value));
  assert.deepEqual(ws.loraGroupsByEngine.byEngine.Anima[0].members.map((item) => [item.value, item.weight]), [["style/NKStyleV3/galaxy.safetensors", -0.25], ["concept/keep.safetensors", 0.3]]);
  assert.equal(ws.loraGroupsByEngine.byEngine.Anima[0].name, "Style mix"); assert.equal(ws.positive, "keep prompt");
  const card = readCard(f.stateDirectory, f.card.id);
  assert.deepEqual(card.settings.loras.map((item) => item.value), ["style/NKStyleV3/galaxy.safetensors"]); assert.equal(card.title, "card title"); assert.equal(card.settings.positive, "gallery text");
  const registry = await readLoraRegistry(f.stateDirectory);
  assert.equal(registry.instances[row.sourceId].retired.operationId, receipt.operationId);
  scan = await f.service.scan({ engines: ["Anima"] });
  assert.ok(!pending(scan).some((item) => item.sourceValue === gone.value), "no dead entry in the default list");
  assert.equal(scan.rows.find((item) => item.sourceId === row.sourceId).lifecycle, "archived");
  // A stale whole-document save cannot resurrect the cleaned binding.
  const store = new LoraPublishedStore({ stateDirectory: f.stateDirectory, ensureRecovered: () => f.service.ensureRecovered() });
  await assert.rejects(store.save("uiState", f.state, { revision: 0 }), /revision changed/);
  await assert.rejects(store.save("uiState", f.state, { revision: 1 }), /Versioned save|revision/);
  // A new file with that name is discovered as a new instance, never blacklisted.
  await put(f.local.directory, "style/gone_v1.safetensors", "a new file");
  scan = await f.service.scan({ engines: ["Anima"] });
  const reborn = scan.observations.find((item) => item.value === gone.value);
  assert.ok(reborn && reborn.instanceId !== row.sourceId);
  // Owned undo restores the members and the tombstone without overwriting later edits elsewhere.
  const undo = await f.service.undo({ operationId: receipt.operationId, undoOperationId: randomUUID() });
  assert.deepEqual(undo.conflicts, []);
  const restored = JSON.parse(await readFile(path.join(f.stateDirectory, "ui-state.json"))).workspace;
  assert.equal(restored.loras.at(-1).value, gone.value);
  assert.equal((await readLoraRegistry(f.stateDirectory)).instances[row.sourceId].retired, undefined);
  // S11: an incomplete scan (disabled shared root) cannot authorize cleanup.
  const disabled = new LoraRelinkService({ stateDirectory: f.stateDirectory, projectRoot: f.project, gallery: galleryOwnerParticipant(f.stateDirectory, python), admitIdle: async () => {}, validatePath: async () => {},
    getRoots: async () => f.roots.map((root) => root.shared ? { ...root, enabled: false } : root) });
  await rm(path.join(f.local.directory, "style/gone_v1.safetensors"));
  const partial = await disabled.scan({ engines: ["Anima"] });
  assert.equal(partial.complete, false);
  const again = partial.rows.find((item) => item.sourceValue === gone.value);
  await assert.rejects(disabled.preview({ scanId: partial.scanId, selections: [{ sourceId: again.sourceId, target: { kind: "delete" }, scope: { all: true } }] }), /incomplete/i);
});

test("S20/S21/S22 scoped reset plan touches only the manifest's local chains and keeps business paths", () => {
  const id = () => randomUUID(), hash = (seed) => seed.repeat(64).slice(0, 64);
  const instances = {};
  const add = (value, extra = {}) => { const instanceId = id(); instances[instanceId] = { instanceId, engine: "Anima", rootId: "local-Anima", value, contentHash: hash("a"), contentRevision: 1, history: [], status: "present", ...extra }; return instanceId; };
  const oldIds = NAMES.map((name) => add(`style/${name}.safetensors`, { contentHash: null, contentRevision: 0, status: "missing", evidence: "legacy-unverified" }));
  const nkIds = NAMES.map((name) => add(`style/NKStyleV3/${name}.safetensors`));
  const sixth = add("style/4x2style_v2_epoch32.safetensors");
  const sharedId = id(); instances[sharedId] = { instanceId: sharedId, engine: "Anima", rootId: SHARED, value: `shared:${SHARED}/Anima/Style/galaxy.safetensors`, contentHash: hash("a"), contentRevision: 1, history: [], status: "present" };
  const fluxId = id(); instances[fluxId] = { instanceId: fluxId, engine: "Flux", rootId: "local-Flux", value: "style/galaxy.safetensors", contentHash: hash("b"), contentRevision: 1, history: [], status: "present" };
  const registry = { schemaVersion: 1, revision: 3, instances, substitutions: oldIds.map((sourceId, index) => ({ engine: "Anima", sourceId, targetId: nkIds[index], kind: "replace", sourceHash: null, targetHash: hash("a"), tagsConfirmed: false })) };
  const strong = (index, weight) => ({ ...entry(`style/NKStyleV3/${NAMES[index]}.safetensors`, weight), identitySchema: 1, instanceId: nkIds[index], contentHash: hash("a"), contentRevision: 1, originalReference: { value: `style/${NAMES[index]}.safetensors`, lastBoundReference: { value: `style/${NAMES[index]}.safetensors` } } });
  const mounts = emptyMountedLoraMap();
  mounts.byEngine.Anima = [strong(0, 0.8137), strong(1, -0.25), { ...entry("style/4x2style_v2_epoch32.safetensors"), identitySchema: 1, instanceId: sixth, contentHash: hash("a"), contentRevision: 1 }];
  mounts.byEngine.Flux = [{ ...entry("style/galaxy.safetensors"), identitySchema: 1, instanceId: fluxId, contentHash: hash("b"), contentRevision: 1 }];
  const state = { workspace: { model: "Anima", mountedLorasByEngine: mounts, loras: structuredClone(mounts.byEngine.Anima), loraGroupsByEngine: { schemaVersion: 1, byEngine: { Anima: [{ id: "g", name: "G", members: [strong(1, -0.25)] }] } } } };
  const cards = { schemaVersion: 2, byEngine: { Anima: { loras: { "style/NKStyleV3/galaxy.safetensors": { title: "keep title", cover: "c".repeat(32), identitySchema: 1, instanceId: nkIds[1] } }, byInstance: { [nkIds[1]]: { title: "keep title" } } } } };
  const galleryCards = [{ id: "card", settings: { model: "Anima", positive: "p", loras: [strong(1, -0.25), entry("concept/keep.safetensors")] } }];
  const operations = [{ operationId: id(), text: oldIds[0] }, { operationId: id(), text: "unrelated" }];
  const values = NAMES.flatMap((name) => [`style/${name}.safetensors`, `style/NKStyleV3/${name}.safetensors`]);
  const plan = planScopedReset({ registry, state, cards, galleryCards, manifest: { engine: "Anima", rootId: "local-Anima", values }, operations, operationId: id(), at: new Date().toISOString() });
  assert.deepEqual(new Set(plan.report.removedInstances.map((item) => item.instanceId)), new Set([...oldIds, ...nkIds]));
  for (const kept of [sixth, sharedId, fluxId]) assert.ok(plan.registry.instances[kept], "sixth file, shared copy and other engine stay");
  assert.deepEqual(plan.registry.substitutions, []);
  assert.equal(plan.registry.revision, 4); assert.deepEqual(plan.registry.maintenance.at(-1).supersedes, [operations[0].operationId]);
  const after = plan.state.workspace.mountedLorasByEngine.byEngine;
  assert.deepEqual(after.Anima.map((item) => [item.value, item.weight, item.instanceId || null]), [["style/NKStyleV3/ashima_v4.safetensors", 0.8137, null], ["style/NKStyleV3/galaxy.safetensors", -0.25, null], ["style/4x2style_v2_epoch32.safetensors", 0.5, sixth]]);
  assert.ok(!("originalReference" in after.Anima[0]) && after.Anima[0].precision === 4 && after.Anima[0].enabled === true);
  assert.equal(after.Flux[0].instanceId, fluxId);
  assert.equal(plan.state.workspace.loraGroupsByEngine.byEngine.Anima[0].members[0].instanceId, undefined);
  assert.deepEqual(plan.cards.byEngine.Anima.loras["style/NKStyleV3/galaxy.safetensors"], { title: "keep title", cover: "c".repeat(32) });
  assert.deepEqual(plan.cards.byEngine.Anima.byInstance, {});
  assert.deepEqual(plan.galleryPatches[0].after.loras.map((item) => [item.value, item.instanceId || null]), [["style/NKStyleV3/galaxy.safetensors", null], ["concept/keep.safetensors", null]]);
  assert.equal(plan.galleryPatches[0].after.positive, "p");
});

test("S20/S22 reset CLI on isolated state: dry run, backups, coordinated publication, stale scan/undo refused, then a clean scan", galleryOptions, async () => {
  const f = await backInStyle({ gallery: true });
  // Reproduce the user's history: repair into NKStyleV3, then files moved back to style/.
  for (const name of NAMES) await rename(path.join(f.local.directory, "style", `${name}.safetensors`), path.join(f.local.directory, "style", "NKStyleV3", `${name}.safetensors`));
  const ws = JSON.parse(await readFile(path.join(f.stateDirectory, "ui-state.json")));
  for (const list of [ws.workspace.mountedLorasByEngine.byEngine.Anima, ws.workspace.loras]) for (const item of list) item.value = item.value.replace("style/NKStyleV3/", "style/");
  ws.workspace.loraGroupsByEngine.byEngine.Anima[0].members[0].value = "style/galaxy.safetensors";
  await durableJson(path.join(f.stateDirectory, "ui-state.json"), ws);
  let scan = await f.service.scan({ engines: ["Anima"] });
  const repaired = await f.service.commit({ scanId: scan.scanId, selections: select(createRelinkDraft(scan, scan.rows, scan.revision)), operationId: randomUUID() });
  for (const name of NAMES) await rename(path.join(f.local.directory, "style", "NKStyleV3", `${name}.safetensors`), path.join(f.local.directory, "style", `${name}.safetensors`));
  const stale = await f.service.scan({ engines: ["Anima"] });
  const manifest = path.join(f.directory, "manifest.json"), backup = path.join(f.directory, "backup");
  await writeFile(manifest, JSON.stringify({ engine: "Anima", rootId: "local-Anima", values: NAMES.flatMap((name) => [`style/${name}.safetensors`, `style/NKStyleV3/${name}.safetensors`]) }));
  const run = (extra) => spawnSync(process.execPath, ["scripts/lora-maintenance-reset.mjs", "--state", f.stateDirectory, "--manifest", manifest, "--backup", backup, "--python", python, "--inference", "http://127.0.0.1:65530", ...extra], { cwd, encoding: "utf8", windowsHide: true });
  const revisionBefore = await publishedLibraryVersion(f.stateDirectory);
  const dry = run([]);
  assert.equal(dry.status, 0, dry.stderr);
  assert.equal(await publishedLibraryVersion(f.stateDirectory), revisionBefore, "dry run publishes nothing");
  const applied = run(["--apply"]);
  assert.equal(applied.status, 0, applied.stderr);
  const result = JSON.parse(applied.stdout);
  assert.equal(result.summary.removedInstances, 10); assert.ok(result.summary.supersedes.includes(repaired.operationId));
  assert.equal(await publishedLibraryVersion(f.stateDirectory), revisionBefore + 1);
  const files = await readdir(backup);
  for (const name of ["dry-run.json", "reset-plan.json", "backup-manifest.json", "registry.json", "ui-state.json", "gallery.sqlite3", "result.json"]) assert.ok(files.includes(name), name);
  const after = JSON.parse(await readFile(path.join(f.stateDirectory, "ui-state.json"))).workspace;
  assert.deepEqual(after.loras.slice(0, 5).map((item) => [item.value, item.instanceId || null]), NAMES.map((name) => [`style/NKStyleV3/${name}.safetensors`, null]), "business paths kept, identity cleared");
  assert.equal(readCard(f.stateDirectory, f.card.id).settings.loras[0].instanceId, undefined);
  await assert.rejects(f.service.preview({ scanId: stale.scanId, selections: [] }), /changed|Scan/);
  await assert.rejects(f.service.undo({ operationId: repaired.operationId, undoOperationId: randomUUID() }), /维护重置/);
  scan = await f.service.scan({ engines: ["Anima"], source: "style/NKStyleV3", target: "style" });
  assert.deepEqual(pending(scan).map((row) => row.recommendation?.value).sort(), NAMES.map((name) => `style/${name}.safetensors`).sort(), "manual-test state: NKStyleV3 sources suggest style/");
  assert.ok(!scan.rows.some((row) => row.status === "identity-conflict"));
});
