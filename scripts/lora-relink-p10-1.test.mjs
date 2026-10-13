import assert from "node:assert/strict";
import { test } from "node:test";
import { createHash, randomUUID } from "node:crypto";
import { mkdir, mkdtemp, readFile, rename, rm, writeFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { scanLoraInventory, registerInitialInventory, registerLegacyReferences, emptyLoraRegistry, matchLoraInventory, readLoraRegistry } from "./lora-library.mjs";
import { planLoraRelink } from "./lora-relink-plan.mjs";
import { LoraRelinkService } from "./lora-relink-service.mjs";
import { galleryOwnerParticipant } from "./lora-gallery-owner.mjs";
import { durableJson } from "./lora-coordinator.mjs";
import { createRelinkDraft, editRelinkRow, mergeRelinkSuggestions, restoreRelinkSuggestion, relinkDraftSummary, selectedRelinkRows } from "../src/lora-relink.js";
import { repairRootLabels, describeRepairFile, repairPickerSections, repairBrowseTree, repairFolderChoices, PICKER_SEARCH_LIMIT } from "../src/lora-repair-picker.js";
import { emptyMountedLoraMap } from "../src/lora-model-scope.js";
import { emptyLoraCardStore } from "../src/lora-cards.js";
import { createLoraRepairController } from "../src/lora-repair-controller.js";
import { pythonDependencyOptions } from "./python-dependency.test.helper.mjs";

// P10.1 R01–R11/R15/R20 ranking, draft and picker contracts. Fixtures are
// isolated temp roots; example names are data, never matching rules.
const python = process.env.P10_PYTHON || "python", cwd = path.resolve(import.meta.dirname, "..");
const galleryOptions = pythonDependencyOptions(python, cwd, ["PIL"]);
const parent = process.env.P10_EVIDENCE_DIR || path.join(os.tmpdir(), "opencode", "p10.1-fixtures");
const SHARED = "0a1b2c3d4e5f";
const NAMES = ["ashima_v4", "galaxy", "joezunzun_v2", "kiriaina_anima_style-000030", "yusano_v5_step1500"];
async function scratch(label) { await mkdir(parent, { recursive: true }); return mkdtemp(path.join(parent, `${label}-`)); }
async function put(root, name, bytes) { const location = path.join(root, name); await mkdir(path.dirname(location), { recursive: true }); await writeFile(location, bytes); }
function py(code, args) { const child = spawnSync(python, ["-B", "-c", code, ...args], { cwd, encoding: "utf8", windowsHide: true }); assert.equal(child.status, 0, child.stderr); return JSON.parse(child.stdout); }
const legacyRoot = (directory, engine = "Anima") => ({ engine, id: `local-${engine}`, directory });

async function incident({ gallery = false } = {}) {
  const directory = await scratch("incident"), stateDirectory = path.join(directory, "state");
  const local = legacyRoot(path.join(directory, "models"));
  const shared = { engine: "Anima", id: SHARED, directory: path.join(directory, "comfy-loras"), shared: true, label: "loras" };
  await put(local.directory, "style/.gitkeep", "");
  await put(local.directory, "concept/keep.safetensors", "unmoved bytes");
  for (const name of NAMES) {
    await put(local.directory, `style/NKStyleV3/${name}.safetensors`, `bytes ${name}`);
    await put(shared.directory, `Anima/Style/${name}.safetensors`, `bytes ${name}`);
  }
  const entry = (name, index) => ({ value: `style/${name}.safetensors`, name, category: "style", weight: [0.8137, -0.25, 1, 0.5, 0.65][index], precision: 4, enabled: index !== 2 });
  const mounts = emptyMountedLoraMap();
  mounts.byEngine.Anima = [...NAMES.map(entry), { value: "concept/keep.safetensors", name: "keep", category: "concept", weight: 0.3, precision: 2, enabled: true }];
  const state = { theme: { accent: "unchanged" }, workspace: { model: "Anima", positive: "keep prompt", mountedLorasByEngine: mounts, loras: structuredClone(mounts.byEngine.Anima),
    loraGroupsByEngine: { schemaVersion: 1, byEngine: { Anima: [{ id: "g-on", name: "on", enabled: true, members: [entry(NAMES[1], 1), entry(NAMES[3], 3)] }, { id: "g-off", name: "off", enabled: false, members: [entry(NAMES[0], 0)] }] } } } };
  await durableJson(path.join(stateDirectory, "ui-state.json"), state);
  const card = gallery ? py("import json,sys; from backend.gallery import GalleryStore; s=GalleryStore(sys.argv[1]); s.create_collection('test'); print(json.dumps(s.create_card('test',title='gallery',settings=json.loads(sys.argv[2]))))",
    [stateDirectory, JSON.stringify({ model: "Anima", positive: "gallery text", loras: [entry(NAMES[1], 1)] })]) : null;
  const service = new LoraRelinkService({ stateDirectory, gallery: galleryOwnerParticipant(stateDirectory, python), getRoots: async () => [local, shared],
    admitIdle: async () => {}, validatePath: async () => {} });
  return { directory, stateDirectory, local, shared, state, card, service };
}
const missingRows = (scan) => scan.rows.filter((row) => row.status === "missing");
const byName = (rows) => Object.fromEntries(rows.map((row) => [row.sourceValue.split("/").at(-1).replace(".safetensors", ""), row]));
const shape = (row) => ({ recommendation: row.recommendation, suggestion: row.suggestion, candidates: row.candidates.map(({ rootId, value, basis, reasons }) => ({ rootId, value, basis, reasons })) });
const selections = (draft) => draft.rows.map((row) => ({ sourceId: row.sourceId, target: row.target, scope: row.scope, skipped: row.skipped, confirmedOrigin: row.target?.kind === "relocate" }));

test("R01 zero registry + five moved same-root files: all prefilled to the new local folder, nothing written", galleryOptions, async () => {
  const f = await incident();
  const before = await readFile(path.join(f.stateDirectory, "ui-state.json"));
  const scan = await f.service.scan({ engines: ["Anima"] });
  assert.equal(scan.initial, true); assert.equal(scan.complete, true);
  const rows = missingRows(scan);
  assert.equal(rows.length, 5);
  for (const [name, row] of Object.entries(byName(rows))) {
    assert.deepEqual([row.recommendation.value, row.recommendation.rootId, row.recommendation.kind, row.recommendation.basis], [`style/NKStyleV3/${name}.safetensors`, "local-Anima", "replace", "name-location"]);
    assert.equal(row.confidence, "unresolved"); assert.equal(row.sourceContentHash, null); assert.equal(row.suggestion.state, "suggested");
    for (const reason of ["same-name", "same-root", "descendant-folder", "directory-group", "provenance-unconfirmed"]) assert.ok(row.recommendation.reasons.includes(reason), reason);
    assert.deepEqual(row.recommendation.group, { from: "style", to: "style/NKStyleV3", count: 5, of: 5 });
    // The shared same-named copy stays an alternative, below the local one.
    assert.deepEqual(row.candidates.map((item) => item.rootId), ["local-Anima", SHARED]);
    assert.ok(row.candidates[1].reasons.includes("other-root"));
  }
  assert.ok(!scan.rows.some((row) => row.sourceValue === "concept/keep.safetensors" && row.status !== "unchanged"));
  const draft = createRelinkDraft(scan, scan.rows, scan.revision);
  assert.deepEqual(relinkDraftSummary(draft), { sources: 5, selected: 5, suggested: 5, manual: 0, unselected: 0, skipped: 0, tied: 0, review: 0, deletions: 0, processed: 0 });
  assert.ok(draft.rows.every((row) => row.target.kind === "replace" && row.chosenBy === "suggestion"));
  assert.deepEqual(await readFile(path.join(f.stateDirectory, "ui-state.json")), before);
  await assert.rejects(readFile(path.join(f.stateDirectory, "lora-library/registry.json")), { code: "ENOENT" });
  assert.equal(scan.metrics.files, 11); assert.ok(scan.metrics.hashedBytes > 0);
  assert.equal(scan.roots.find((root) => root.rootId === SHARED).label, "loras");
  // Cached rescan reads no weight bytes again.
  const again = await f.service.scan({ engines: ["Anima"] });
  assert.equal(again.metrics.hashedBytes, 0); assert.equal(again.metrics.cachedFiles, 11);
  await writeFile(path.join(parent, `R01-scan-summary-${Date.now()}.json`), JSON.stringify({ initial: scan.initial, complete: scan.complete, metrics: scan.metrics, again: again.metrics,
    rows: rows.map((row) => ({ source: row.sourceValue, status: row.status, confidence: row.confidence, recommendation: row.recommendation, suggestion: row.suggestion, candidates: row.candidates })), summary: relinkDraftSummary(draft) }, null, 2));
});

test("R02/R15/R20 registered targets keep R01 order; explicit rebinding audit, untouched params, deterministic rescan", galleryOptions, async () => {
  const reference = await incident(), r01 = byName(missingRows(await reference.service.scan({ engines: ["Anima"] })));
  const f = await incident({ gallery: true });
  const initial = await f.service.scan({ engines: ["Anima"] });
  await f.service.commit({ scanId: initial.scanId, selections: [], registerBaseline: true, operationId: randomUUID() });
  const registered = await readLoraRegistry(f.stateDirectory);
  assert.equal(Object.values(registered.instances).filter((item) => item.status === "present").length, 11);
  const scan = await f.service.scan({ engines: ["Anima"] });
  assert.equal(scan.initial, false);
  const rows = byName(missingRows(scan));
  for (const name of NAMES) {
    assert.deepEqual(shape(rows[name]), shape(r01[name]), name);
    assert.ok(rows[name].candidates.every((item) => item.existing), "registered targets are existing instances");
  }
  assert.equal(rows.galaxy.impacts.mounts, 1); assert.equal(rows.galaxy.impacts.groups.length, 1); assert.equal(rows.galaxy.impacts.cards.length, 1);
  const draft = createRelinkDraft(scan, scan.rows, scan.revision);
  const payload = { scanId: scan.scanId, selections: selections(draft), registerBaseline: false };
  const preview = await f.service.preview(payload);
  assert.deepEqual(preview.conflicts, []);
  assert.deepEqual(preview.results.map((item) => [item.kind, item.evidence]), NAMES.map(() => ["replace", "unverified-source"]));
  const beforeState = JSON.parse(await readFile(path.join(f.stateDirectory, "ui-state.json"), "utf8"));
  const receipt = await f.service.commit({ ...payload, operationId: randomUUID() });
  assert.equal(receipt.status, "published"); assert.ok(receipt.results.every((item) => item.evidence === "unverified-source" && item.kind === "replace"));
  const after = JSON.parse(await readFile(path.join(f.stateDirectory, "ui-state.json"), "utf8"));
  const compare = (list, previous) => list.forEach((item, index) => {
    const old = previous[index];
    assert.equal(item.value, old.value.startsWith("style/") ? old.value.replace("style/", "style/NKStyleV3/") : old.value);
    for (const key of ["weight", "precision", "enabled", "name"]) assert.equal(item[key], old[key], key);
  });
  compare(after.workspace.mountedLorasByEngine.byEngine.Anima, beforeState.workspace.mountedLorasByEngine.byEngine.Anima);
  compare(after.workspace.loras, beforeState.workspace.loras);
  for (const [index, group] of after.workspace.loraGroupsByEngine.byEngine.Anima.entries()) compare(group.members, beforeState.workspace.loraGroupsByEngine.byEngine.Anima[index].members);
  assert.deepEqual(after.workspace.mountedLorasByEngine.byEngine.Anima.at(-1), beforeState.workspace.mountedLorasByEngine.byEngine.Anima.at(-1));
  assert.equal(after.theme.accent, "unchanged"); assert.equal(after.workspace.positive, "keep prompt");
  const card = py("import json,sys; from backend.gallery import GalleryStore; s=GalleryStore(sys.argv[1]); c=s._connect(); card=s._fetch_card(c,sys.argv[2]); c.close(); print(json.dumps(card['settings']))", [f.stateDirectory, f.card.id]);
  assert.equal(card.loras[0].value, "style/NKStyleV3/galaxy.safetensors"); assert.equal(card.loras[0].weight, -0.25); assert.equal(card.positive, "gallery text");
  const registry = await readLoraRegistry(f.stateDirectory);
  for (const row of Object.values(rows)) {
    const source = registry.instances[row.sourceId];
    assert.equal(source.status, "missing"); assert.equal(source.contentHash, null); assert.equal(source.evidence, "legacy-unverified");
    const rule = registry.substitutions.find((item) => item.sourceId === row.sourceId);
    assert.equal(rule.kind, "replace"); assert.equal(rule.sourceHash, null);
    assert.equal(registry.instances[rule.targetId].value, row.recommendation.value);
  }
  // R20: same suggestions after restart; settled rows default to skipped, no implicit commit.
  const restarted = new LoraRelinkService({ stateDirectory: f.stateDirectory, gallery: galleryOwnerParticipant(f.stateDirectory, python), getRoots: async () => [f.local, f.shared], admitIdle: async () => {}, validatePath: async () => {} });
  const rescan = await restarted.scan({ engines: ["Anima"] });
  const again = byName(missingRows(rescan));
  for (const name of NAMES) {
    assert.deepEqual(again[name].recommendation, rows[name].recommendation);
    assert.equal(again[name].existingRule.targetValue, `style/NKStyleV3/${name}.safetensors`);
  }
  const redraft = createRelinkDraft(rescan, rescan.rows, rescan.revision);
  assert.equal(selectedRelinkRows(redraft).length, 0);
  assert.equal((await restarted.commit({ scanId: rescan.scanId, selections: selections(redraft), registerBaseline: false, operationId: randomUUID() })).status, "no-op");
});

async function baselineInventory(files, { identity } = {}) {
  const directory = await scratch("rank"), root = legacyRoot(directory);
  for (const [name, bytes] of Object.entries(files)) await put(directory, name, bytes);
  const options = identity ? { fileIdentity: identity } : {};
  const scan = await scanLoraInventory([root], options);
  return { directory, root, options, registry: registerInitialInventory(emptyLoraRegistry(), scan) };
}
// Test-only provider: a stable key per byte content stands in for an inode.
const contentIdentity = async (file) => ({ kind: "test", key: createHash("sha256").update(readFileSync(file)).digest("hex").slice(0, 16), links: 1, creation: "c" });
const rowFor = (rows, registry, value) => rows.find((row) => row.sourceId === Object.values(registry.instances).find((item) => item.value === value).instanceId);

test("R03 trusted SHA + unique filesystem identity stays proven relocation; equal SHA alone is only equal content", async () => {
  const proven = await baselineInventory({ "style/A.safetensors": "A bytes", "style/B.safetensors": "B bytes" }, { identity: contentIdentity });
  await put(proven.directory, "concept/renamed.safetensors", "A bytes"); await rm(path.join(proven.directory, "style/A.safetensors"));
  let rows = matchLoraInventory(proven.registry, await scanLoraInventory([proven.root], proven.options));
  let row = rowFor(rows, proven.registry, "style/A.safetensors");
  assert.equal(row.confidence, "filesystem-and-content");
  assert.deepEqual([row.recommendation.kind, row.recommendation.basis, row.recommendation.value], ["relocate", "proven-relocation", "concept/renamed.safetensors"]);
  assert.ok(!row.recommendation.reasons.includes("provenance-unconfirmed"));
  const plain = await baselineInventory({ "style/A.safetensors": "A bytes", "style/B.safetensors": "B bytes" });
  await put(plain.directory, "concept/renamed.safetensors", "A bytes"); await rm(path.join(plain.directory, "style/A.safetensors"));
  rows = matchLoraInventory(plain.registry, await scanLoraInventory([plain.root]));
  row = rowFor(rows, plain.registry, "style/A.safetensors");
  assert.equal(row.confidence, "same-content-unproven");
  assert.deepEqual([row.recommendation.kind, row.recommendation.basis], ["replace", "equal-content"]);
  assert.ok(row.recommendation.reasons.includes("provenance-unconfirmed"));
});

function legacyScan(scan, values, engine = "Anima") {
  return registerLegacyReferences(emptyLoraRegistry(), values.map((value) => ({ engine, rootId: `local-${engine}`, value })), scan).registry;
}
test("R04 substantive ties stay ambiguous; group support is a visible reason; competing sources get no second best", async () => {
  const directory = await scratch("ties"), root = legacyRoot(directory);
  for (const name of ["style/X/A.safetensors", "style/Y/A.safetensors"]) await put(directory, name, name);
  let scan = await scanLoraInventory([root]);
  let rows = matchLoraInventory(legacyScan(scan, ["style/A.safetensors"]), scan);
  assert.equal(rows[0].recommendation, null); assert.deepEqual(rows[0].suggestion, { state: "tied", basis: "name-location", ties: 2 });
  // Partial group: X explains both sources, Y only one -> X is preferred, with the reason shown.
  await put(directory, "style/X/B.safetensors", "B");
  scan = await scanLoraInventory([root]);
  rows = matchLoraInventory(legacyScan(scan, ["style/A.safetensors", "style/B.safetensors"]), scan);
  for (const row of rows) { assert.match(row.recommendation.value, /^style\/X\//); assert.deepEqual(row.recommendation.group, { from: "style", to: "style/X", count: 2, of: 2 }); }
  // Equal group support everywhere -> both remain tied, alphabetical order never selects.
  await put(directory, "style/Y/B.safetensors", "B2");
  scan = await scanLoraInventory([root]);
  rows = matchLoraInventory(legacyScan(scan, ["style/A.safetensors", "style/B.safetensors"]), scan);
  assert.ok(rows.every((row) => row.recommendation === null && row.suggestion.state === "tied"));
  // Two old folders competing for the only same-named file: neither is prefilled.
  const contest = await scratch("contest"), contestRoot = legacyRoot(contest);
  await put(contest, "style/c/M.safetensors", "M");
  scan = await scanLoraInventory([contestRoot]);
  rows = matchLoraInventory(legacyScan(scan, ["style/a/M.safetensors", "style/b/M.safetensors"]), scan);
  assert.ok(rows.every((row) => row.recommendation === null && row.suggestion.state === "contested" && row.candidates.length === 1));
});

test("R05 different trusted SHA is never a recovery preference; deliberate replacement audits different content", async () => {
  const { directory, root, registry } = await baselineInventory({ "style/A.safetensors": "version one" });
  const A = Object.values(registry.instances)[0];
  await rm(path.join(directory, "style/A.safetensors"));
  await put(directory, "style/new/A.safetensors", "version two");
  await put(directory, "style/new/A.civitai.info", JSON.stringify({ hash: A.contentHash }));
  const scan = await scanLoraInventory([root]);
  const row = matchLoraInventory(registry, scan)[0];
  assert.equal(row.recommendation, null); assert.equal(row.suggestion.state, "none");
  assert.equal(row.candidates[0].basis, "different-content"); assert.ok(row.candidates[0].reasons.includes("different-content"));
  const plan = planLoraRelink({ registry, scan, state: { workspace: { model: "Anima", loras: [{ value: A.value, instanceId: A.instanceId, weight: 0.5 }] } }, cards: emptyLoraCardStore(),
    selections: [{ sourceId: A.instanceId, target: { value: "style/new/A.safetensors", rootId: root.id, kind: "replace" }, scope: { all: true, history: true } }] });
  assert.equal(plan.results[0].evidence, "different-content");
  assert.equal(plan.registry.substitutions[0].sourceHash, A.contentHash);
  assert.throws(() => planLoraRelink({ registry, scan, state: {}, cards: emptyLoraCardStore(), selections: [{ sourceId: A.instanceId, target: { value: "style/new/A.safetensors", rootId: root.id, kind: "relocate" }, confirmedOrigin: true, scope: { all: true } }] }), /verified equal bytes/);
});

test("R06 existing equal-content copy is an explicit substitute with its own identity; no cross-engine candidates", async () => {
  const directory = await scratch("copies");
  const anima = legacyRoot(path.join(directory, "anima")), sd = legacyRoot(path.join(directory, "sd"), "SD");
  for (const name of ["style/A.safetensors", "style/B.safetensors"]) await put(anima.directory, name, "same bytes");
  await put(sd.directory, "style/A.safetensors", "same bytes");
  const registry = registerInitialInventory(emptyLoraRegistry(), await scanLoraInventory([anima, sd]));
  await rm(path.join(anima.directory, "style/A.safetensors"));
  const scan = await scanLoraInventory([anima, sd]);
  const rows = matchLoraInventory(registry, scan);
  const row = rows.find((item) => item.engine === "Anima" && item.status === "missing");
  const B = Object.values(registry.instances).find((item) => item.engine === "Anima" && item.value === "style/B.safetensors");
  assert.equal(row.confidence, "unresolved", "an existing copy is not provenance");
  assert.deepEqual([row.recommendation.value, row.recommendation.kind, row.recommendation.basis], ["style/B.safetensors", "replace", "equal-content"]);
  assert.ok(row.recommendation.reasons.includes("existing-copy"));
  assert.ok(row.candidates.every((item) => item.engine === "Anima"));
  const plan = planLoraRelink({ registry, scan, state: {}, cards: emptyLoraCardStore(), selections: [{ sourceId: row.sourceId, target: { value: "style/B.safetensors", rootId: anima.id, kind: "replace" }, scope: { all: true } }] });
  assert.equal(plan.results[0].targetId, B.instanceId); assert.notEqual(plan.results[0].targetId, row.sourceId);
  assert.equal(plan.registry.instances[row.sourceId].status, "missing");
});

test("R07 arbitrary folders, cross-category, renamed-with-SHA and partial moves rank sensibly; unmoved stay out", async () => {
  const directory = await scratch("general"), root = legacyRoot(directory);
  await put(directory, "zz-archive/2026/q3/portrait_pack-v7.safetensors", "p");
  let scan = await scanLoraInventory([root]);
  let rows = matchLoraInventory(legacyScan(scan, ["characters/portrait_pack-v7.safetensors", "characters/portrait_pack-v8.safetensors"]), scan);
  const [v7, v8] = rows;
  assert.equal(v7.recommendation.value, "zz-archive/2026/q3/portrait_pack-v7.safetensors"); assert.ok(v7.recommendation.reasons.includes("other-folder"));
  assert.equal(v8.recommendation, null, "different version suffix is never equated");
  const { directory: moved, root: movedRoot, registry } = await baselineInventory({ "old/one.safetensors": "1", "old/two.safetensors": "2", "old/three.safetensors": "3" });
  await mkdir(path.join(moved, "fresh"), { recursive: true });
  await rename(path.join(moved, "old/one.safetensors"), path.join(moved, "fresh/one.safetensors"));
  await rename(path.join(moved, "old/two.safetensors"), path.join(moved, "fresh/second-name.safetensors"));
  scan = await scanLoraInventory([movedRoot]);
  rows = matchLoraInventory(registry, scan);
  assert.equal(rows.filter((row) => row.status !== "unchanged").length, 2);
  assert.equal(rowFor(rows, registry, "old/three.safetensors").status, "unchanged");
  assert.equal(rowFor(rows, registry, "old/one.safetensors").recommendation.value, "fresh/one.safetensors");
  assert.equal(rowFor(rows, registry, "old/two.safetensors").recommendation.value, "fresh/second-name.safetensors");
  // Windows has no default native identity provider; POSIX inodes prove the rename.
  assert.equal(rowFor(rows, registry, "old/two.safetensors").recommendation.basis, process.platform === "win32" ? "equal-content" : "proven-relocation");
});

test("R08 manual targets, skips and scopes survive late rankings; Restore suggestion resets only the target", () => {
  const rows = [{ sourceId: "a", status: "missing", recommendation: { value: "x/A", rootId: "local-SD", kind: "replace", basis: "name-location" }, suggestion: { state: "suggested" } },
    { sourceId: "b", status: "missing", recommendation: { value: "x/B", rootId: "local-SD", kind: "replace" }, suggestion: { state: "suggested" } },
    { sourceId: "c", status: "missing", recommendation: { value: "x/C", rootId: "local-SD", kind: "candidate" }, suggestion: { state: "suggested" } }];
  let draft = createRelinkDraft({ scanId: "s", complete: true }, rows, 1);
  assert.equal(draft.rows[2].target.kind, "replace", "ranking kind is normalized before the planner");
  draft = editRelinkRow(draft, "a", { target: { value: "y/A", rootId: "local-SD", kind: "replace" } });
  draft = editRelinkRow(draft, "b", { skipped: true, scope: { all: true, history: false } });
  const late = createRelinkDraft({ scanId: "t", complete: true }, rows.map((row) => ({ ...row, recommendation: { ...row.recommendation, value: `late/${row.sourceId}` } })), 1);
  let merged = mergeRelinkSuggestions(draft, late);
  assert.equal(merged.rows[0].target.value, "y/A"); assert.equal(merged.rows[0].recommendation.value, "late/a");
  assert.equal(merged.rows[1].skipped, true); assert.equal(merged.rows[1].scope.history, false);
  assert.equal(merged.rows[2].target.value, "late/c", "untouched suggestions may follow newer ranking");
  merged = restoreRelinkSuggestion(merged, "b");
  assert.equal(merged.rows[1].target.value, "late/b"); assert.equal(merged.rows[1].skipped, true, "restore never unskips");
  assert.deepEqual(merged.rows[1].scope, { all: true, history: false }, "restore never re-enables a disabled scope");
  assert.equal(merged.rows[1].chosenBy, "suggestion");
});

// R09 (after-scan "match by directory") was replaced in P10.2 by the pre-scan
// target directory; see lora-relink-p10-2.test.mjs S05–S07.

test("R10/R11 picker: few relevant candidates with friendly roots first; search and browse reach every legal file", () => {
  const roots = [{ engine: "Anima", rootId: "local-Anima" }, { engine: "Anima", rootId: SHARED, shared: true, label: "loras" }, { engine: "Anima", rootId: "ffffeeeeaaaa", shared: true, label: "loras" }, { engine: "SD", rootId: "local-SD" }];
  const labels = repairRootLabels(roots);
  assert.equal(labels.get("local-Anima").label, "本地");
  assert.notEqual(labels.get(SHARED).label, labels.get("ffffeeeeaaaa").label, "identically named roots are disambiguated");
  const observations = [];
  for (let index = 0; index < 500; index += 1) observations.push({ engine: "Anima", rootId: index % 2 ? SHARED : "local-Anima", value: `${index % 2 ? `shared:${SHARED}/` : ""}bulk/f${index}/item-${index}.safetensors` });
  observations.push({ engine: "Anima", rootId: "local-Anima", value: "style/NK/galaxy.safetensors" }, { engine: "Anima", rootId: SHARED, value: `shared:${SHARED}/Anima/Style/galaxy.safetensors` },
    { engine: "Anima", rootId: "local-Anima", value: "style/NK/galaxy.ckpt-like.safetensors" }, { engine: "SD", rootId: "local-SD", value: "style/galaxy.safetensors" });
  const row = { engine: "Anima", sourceValue: "style/galaxy.safetensors", rootId: "local-Anima", target: { value: "bulk/f2/item-2.safetensors", rootId: "local-Anima" },
    recommendation: { value: "style/NK/galaxy.safetensors", rootId: "local-Anima" },
    candidates: [{ value: "style/NK/galaxy.safetensors", rootId: "local-Anima", basis: "name-location", reasons: [] }, { value: `shared:${SHARED}/Anima/Style/galaxy.safetensors`, rootId: SHARED, basis: "name-location", reasons: [] }] };
  const initial = repairPickerSections(row, observations);
  assert.equal(initial.recommended.value, "style/NK/galaxy.safetensors");
  assert.deepEqual(initial.alternatives.map((item) => item.rootId), [SHARED]);
  assert.equal(initial.selected.value, "bulk/f2/item-2.safetensors", "manual selection stays visible");
  const shared = describeRepairFile(initial.alternatives[0], labels);
  assert.equal(shared.folder, "Anima/Style/"); assert.ok(!shared.folder.includes("shared:") && !shared.rootLabel.includes(SHARED));
  assert.equal(shared.displayName, "galaxy");
  const search = repairPickerSections(row, observations, { query: "item-49" });
  assert.ok(search.results.length <= PICKER_SEARCH_LIMIT && search.total >= 11);
  assert.ok(search.results.some((item) => item.rootId === SHARED));
  assert.ok(!repairPickerSections(row, observations, { query: "galaxy" }).results.some((item) => item.engine === "SD"));
  const tree = repairBrowseTree(observations, "Anima", labels);
  assert.equal(tree[0].rootId, "local-Anima");
  assert.equal(tree.reduce((sum, root) => sum + root.folders.reduce((count, folder) => count + folder.items.length, 0), 0), 503);
  assert.ok(repairFolderChoices(observations, ["Anima"], labels).some((item) => item.rootId === SHARED && item.folder === "Anima/Style"));
});

test("R17 a refused commit keeps the draft; a rescan never retries the stale payload", async () => {
  const response = (value, ok = true) => ({ ok, status: ok ? 200 : 409, json: async () => value });
  const commits = [];
  let scans = 0, refuse = true;
  const controller = createLoraRepairController({ request: async (url, options) => {
    if (url.endsWith("/state")) return response({ available: true, contractVersion: 1 });
    if (url.endsWith("/scan")) { scans += 1; return response({ scanId: `scan-${scans}`, complete: true, revision: 1, observations: [],
      rows: [{ sourceId: "a", status: "missing", recommendation: { value: "new/A", rootId: "local-SD", kind: "replace" }, impacts: { groups: [], cards: [], mounts: 1 } }] }); }
    if (url.endsWith("/preview")) return response({ conflicts: [], results: [{ evidence: "unverified-source" }], affected: { workspace: 1, cards: 0, gallery: 0 } });
    const body = JSON.parse(options.body); commits.push(body);
    if (refuse) return response({ error: "Target renamed, removed or unreadable; review the draft" }, false);
    return response({ operationId: body.operationId, status: "published", revision: 2 });
  } });
  await controller.availability(); await controller.scan(["SD"]); await controller.preview(); await controller.commit();
  assert.equal(controller.state().phase, "editing"); assert.match(controller.state().error, /Target renamed/);
  assert.equal(controller.state().draft.rows[0].target.value, "new/A", "draft retained");
  refuse = false;
  await controller.scan(["SD"], { discard: true }); await controller.preview(); await controller.commit();
  assert.equal(commits.length, 2); assert.equal(commits[1].scanId, "scan-2"); assert.notEqual(commits[1].operationId, commits[0].operationId);
  assert.equal(controller.state().phase, "result");
});

test("rescans reuse the native identity of an unchanged file; a changed file is asked again", async () => {
  const directory = await scratch("identity-cache"), root = legacyRoot(directory);
  for (const name of ["a/one.safetensors", "a/two.safetensors"]) await put(directory, name, name);
  let calls = 0;
  const fileIdentity = async (file) => { calls += 1; return contentIdentity(file); };
  const hashCache = new Map();
  const first = await scanLoraInventory([root], { fileIdentity, hashCache });
  assert.equal(calls, 2);
  const again = await scanLoraInventory([root], { fileIdentity, hashCache });
  assert.equal(calls, 2, "no provider call for unchanged fingerprints");
  assert.deepEqual(again.observations.map((item) => item.fsIdentity), first.observations.map((item) => item.fsIdentity));
  await put(directory, "a/two.safetensors", "changed bytes");
  const changed = await scanLoraInventory([root], { fileIdentity, hashCache });
  assert.equal(calls, 3);
  assert.notDeepEqual(changed.observations.find((item) => item.value === "a/two.safetensors").fsIdentity, first.observations.find((item) => item.value === "a/two.safetensors").fsIdentity);
});
