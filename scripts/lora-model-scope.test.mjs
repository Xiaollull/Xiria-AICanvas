import assert from "node:assert/strict";
import test from "node:test";
import { applyMountedLoraSync, canStartMountedLoraScan, emptyMountedLoraMap, engineScopeKey, galleryMountedLorasForTarget, mountedLorasForScope, nextMountedLoraRevision, normalizeMountedLoraMap, reconcileMountedLoraScan, shouldApplyMountedLoraScan, transitionMountedLoraScope, withMountedLorasForScope } from "../src/lora-model-scope.js";
import { readFile } from "node:fs/promises";
const lora = (value, extra = {}) => ({ value, name: value, category: "character", weight: 1, precision: 1, enabled: true, ...extra });
const v1key = (engine, assets) => `xirai-lora-scope-v1.${Buffer.from(JSON.stringify({ v: 1, engine, assets })).toString("base64url")}`;

test("engine container preserves exact iL A to B to A and Anima split asset changes", () => {
  let map = withMountedLorasForScope(emptyMountedLoraMap(), "iL", [lora("il-A", { weight: .25, enabled: false })]);
  assert.deepEqual(mountedLorasForScope(map, "iL"), [lora("il-A", { weight: .25, enabled: false })]);
  // Checkpoint/split assets do not participate in the runtime address.
  map = withMountedLorasForScope(map, engineScopeKey("Anima"), [lora("anima")]);
  assert.deepEqual(mountedLorasForScope(map, "iL").map((x) => x.value), ["il-A"]);
  assert.deepEqual(mountedLorasForScope(map, "Anima").map((x) => x.value), ["anima"]);
});
test("SD and iL same base path isolate; engine switching restores", () => {
  let map = withMountedLorasForScope(emptyMountedLoraMap(), "SD", [lora("same-path")]);
  map = withMountedLorasForScope(map, "iL", [lora("same-path", { weight: 2 })]);
  assert.equal(mountedLorasForScope(map, "SD")[0].weight, 1);
  assert.equal(mountedLorasForScope(map, "iL")[0].weight, 2);
  assert.equal(engineScopeKey("future"), null);
});
test("iL base-asset changes do not re-scope or replace its engine list", () => {
  const initial = withMountedLorasForScope(emptyMountedLoraMap(), "iL", [lora("iL-only", { weight: .25 })]);
  const afterBaseA = transitionMountedLoraScope(initial, { sourceEngine: "iL", sourceLoras: mountedLorasForScope(initial, "iL"), targetEngine: "iL" });
  const afterBaseB = transitionMountedLoraScope(afterBaseA.container, { sourceEngine: "iL", sourceLoras: afterBaseA.activeLoras, targetEngine: "iL" });
  assert.equal(afterBaseA.changed, false);
  assert.equal(afterBaseB.changed, false);
  assert.deepEqual(mountedLorasForScope(afterBaseB.container, "iL"), [lora("iL-only", { weight: .25 })]);
  assert.deepEqual(mountedLorasForScope(afterBaseB.container, "SD"), []);
});
test("v1 migration uses legacy current engine, deterministic winner, and never merges", () => {
  const a = v1key("SD", ["A"]), b = v1key("SD", ["B"]), il = v1key("iL", ["I"]);
  const migrated = normalizeMountedLoraMap({ schemaVersion: 1, byModel: { [b]: [lora("B")], [a]: [lora("A")], [il]: [lora("I")] } }, { activeEngine: "SD", activeModelIdentity: { engine: "SD", assets: ["A"] }, legacyLoras: [lora("visible")] });
  assert.deepEqual(migrated.container.byEngine.SD.map((x) => x.value), ["visible"]);
  assert.deepEqual(migrated.container.byEngine.iL.map((x) => x.value), ["I"]);
  const noLegacy = normalizeMountedLoraMap({ schemaVersion: 1, byModel: { [b]: [lora("B")], [a]: [lora("A")] } }, { activeEngine: "SD", activeModelIdentity: { engine: "SD", assets: ["A"] } });
  assert.deepEqual(noLegacy.container.byEngine.SD.map((x) => x.value), ["A"]);
  const empty = normalizeMountedLoraMap({ schemaVersion: 2, byEngine: { SD: [], iL: [], Anima: [] } }, { activeEngine: "SD", legacyLoras: [lora("no-revive")] });
  assert.deepEqual(empty.activeLoras, []);
});
test("v1 malformed keys and shapes fail closed, retain raw identity, and redact paths", () => {
  const futureV1Key = `xirai-lora-scope-v1.${Buffer.from(JSON.stringify({ v: 2, engine: "SD", assets: ["A"] })).toString("base64url")}`;
  const cases = [
    { schemaVersion: 1, byModel: { "not-a-v1-key": [lora("secret-path")] } },
    { schemaVersion: 1, byModel: { [v1key("SD", ["A"])]: "not-an-array" } },
    { schemaVersion: 1, byModel: [] },
    { schemaVersion: 1, byModel: { [futureV1Key]: [lora("secret-path")] } },
  ];
  for (const raw of cases) {
    const result = normalizeMountedLoraMap(raw, { activeEngine: "SD" });
    assert.equal(result.fatal, true);
    assert.strictEqual(result.raw, raw);
    assert.equal(result.warning.includes("secret-path"), false);
    assert.deepEqual(result.container, emptyMountedLoraMap());
  }
});
test("valid empty v1 migrates once while valid empty v2 never revives", () => {
  const v1 = { schemaVersion: 1, byModel: {} };
  const migrated = normalizeMountedLoraMap(v1, { activeEngine: "iL", legacyLoras: [lora("legacy")] });
  assert.equal(migrated.fatal, false); assert.equal(migrated.migrated, true);
  assert.deepEqual(migrated.container.byEngine.iL.map((item) => item.value), ["legacy"]);
  const v2 = { schemaVersion: 2, byEngine: { SD: [], iL: [], Anima: [] } };
  const retainedEmpty = normalizeMountedLoraMap(v2, { activeEngine: "iL", legacyLoras: [lora("must-not-revive")] });
  assert.equal(retainedEmpty.fatal, false); assert.equal(retainedEmpty.migrated, false); assert.deepEqual(retainedEmpty.activeLoras, []);
});
test("future and malformed v2 containers fail closed and preserve raw without path leakage", () => {
  for (const raw of [
    { schemaVersion: 3, byEngine: { SD: [lora("secret-path")], iL: [], Anima: [] } },
    { schemaVersion: 2, byEngine: { SD: [], iL: [], Anima: "bad" } },
    { schemaVersion: 2, byEngine: { SD: [], iL: [], Anima: [], future: [] } },
  ]) {
    const result = normalizeMountedLoraMap(raw, { activeEngine: "SD" });
    assert.equal(result.fatal, true); assert.strictEqual(result.raw, raw);
    assert.equal(result.warning.includes("secret-path"), false);
  }
});
test("engine transition API preserves source and restores target; same-engine reconnect does not clobber", () => {
  let map = withMountedLorasForScope(emptyMountedLoraMap(), "iL", [lora("il")]);
  map = withMountedLorasForScope(map, "SD", [lora("sd")]);
  const toSd = transitionMountedLoraScope(map, { sourceEngine: "iL", sourceLoras: [lora("il", { weight: .4 })], targetEngine: "SD" });
  assert.deepEqual(toSd.activeLoras.map((item) => item.value), ["sd"]);
  assert.equal(mountedLorasForScope(toSd.container, "iL")[0].weight, .4);
  const reconnect = transitionMountedLoraScope(toSd.container, { sourceEngine: "SD", sourceLoras: toSd.activeLoras, targetEngine: "SD" });
  assert.deepEqual(reconnect.activeLoras.map((item) => item.value), ["sd"]);
});
test("mounted-map revision advances exactly once only for semantic SD/iL/Anima changes", () => {
  const revision = 40;
  let map = withMountedLorasForScope(emptyMountedLoraMap(), "SD", [lora("shared")]);
  map = withMountedLorasForScope(map, "iL", [lora("shared")]);
  map = withMountedLorasForScope(map, "Anima", [lora("anima")]);

  // Selecting/reconnecting the same engine and crossing iL -> SD with equal
  // lists still update the active mirror, but cannot invalidate a scan.
  const sameEngine = transitionMountedLoraScope(map, { sourceEngine: "iL", sourceLoras: mountedLorasForScope(map, "iL"), targetEngine: "iL" });
  const equalCrossEngine = transitionMountedLoraScope(map, { sourceEngine: "iL", sourceLoras: mountedLorasForScope(map, "iL"), targetEngine: "SD" });
  assert.equal(sameEngine.scopeChanged, false);
  assert.equal(equalCrossEngine.scopeChanged, true);
  assert.equal(nextMountedLoraRevision(revision, map, sameEngine.container), revision);
  assert.equal(nextMountedLoraRevision(revision, map, equalCrossEngine.container), revision);

  // A normalized restore of the already-initial map is likewise a no-op.
  const restoredSame = normalizeMountedLoraMap(map, { activeEngine: "SD" }).container;
  assert.equal(nextMountedLoraRevision(revision, map, restoredSame), revision);

  const local = withMountedLorasForScope(map, "SD", [lora("local")]);
  assert.equal(nextMountedLoraRevision(revision, map, local), revision + 1, "local edit");

  const synced = applyMountedLoraSync(map, { engine: "iL", loras: [lora("synced")] }, { activeEngine: "SD" });
  assert.equal(nextMountedLoraRevision(revision, map, synced.container), revision + 1, "cross-window sync");

  const gallery = withMountedLorasForScope(map, "Anima", galleryMountedLorasForTarget(map, {
    targetEngine: "Anima", sourceLoras: [lora("gallery")], applyLoras: true,
  }).loras);
  assert.equal(nextMountedLoraRevision(revision, map, gallery), revision + 1, "Gallery target replacement");

  // Source save and target initialization are folded into one logical map
  // commit, so this transition can only produce revision +1, never +2.
  const transition = transitionMountedLoraScope(map, {
    sourceEngine: "iL", sourceLoras: [lora("source-save")], targetEngine: "SD",
  });
  const transitionWithTarget = withMountedLorasForScope(transition.container, "SD", [lora("target-init")]);
  assert.equal(nextMountedLoraRevision(revision, map, transitionWithTarget), revision + 1, "engine transition");

  const migrated = normalizeMountedLoraMap(undefined, {
    fieldMissing: true, activeEngine: "Anima", legacyLoras: [lora("migrated")],
  }).container;
  assert.equal(nextMountedLoraRevision(revision, emptyMountedLoraMap(), migrated), revision + 1, "populated restore/migration");

  const pruneMap = withMountedLorasForScope(map, "iL", [lora("shared"), lora("removed")]);
  const pruned = reconcileMountedLoraScan({
    container: pruneMap, categories: [{ id: "character", models: [lora("shared")] }], scanScopeKey: "iL",
    requestToken: 6, latestToken: 6, capturedRevision: revision, latestRevision: revision,
    responseEngine: "iL", activeEngine: "iL",
  });
  assert.equal(pruned.changed, true);
  assert.equal(nextMountedLoraRevision(revision, pruneMap, pruned.container), revision + 1, "real scan prune");

  const noPrune = reconcileMountedLoraScan({
    container: map, categories: [{ id: "character", models: [lora("shared")] }], scanScopeKey: "SD",
    requestToken: 7, latestToken: 7, capturedRevision: revision, latestRevision: revision,
    responseEngine: "SD", activeEngine: "SD",
  });
  assert.equal(noPrune.changed, false);
  assert.equal(nextMountedLoraRevision(revision, map, noPrune.container), revision, "scan with nothing to prune");
});
test("a stale scan survives a semantic no-op but loses after a real mutation", () => {
  const revision = 12;
  const map = withMountedLorasForScope(emptyMountedLoraMap(), "iL", [lora("existing")]);
  const noOp = transitionMountedLoraScope(map, { sourceEngine: "iL", sourceLoras: mountedLorasForScope(map, "iL"), targetEngine: "SD" });
  const noOpRevision = nextMountedLoraRevision(revision, map, noOp.container);
  assert.equal(shouldApplyMountedLoraScan({ requestToken: 3, latestToken: 3, capturedRevision: revision, latestRevision: noOpRevision, responseEngine: "iL", activeEngine: "iL" }), true);
  const changed = withMountedLorasForScope(map, "iL", [lora("existing"), lora("new")]);
  const changedRevision = nextMountedLoraRevision(revision, map, changed);
  assert.equal(changedRevision, revision + 1);
  assert.equal(shouldApplyMountedLoraScan({ requestToken: 3, latestToken: 3, capturedRevision: revision, latestRevision: changedRevision, responseEngine: "iL", activeEngine: "iL" }), false);
});
test("scan gate permits idle, rejects running or locks before completion, then permits one unlock retry", () => {
  const idle = { uiStateReady: true, status: "idle", modelSwitching: false, workspaceLocked: false, shouldPersist: true };
  assert.equal(canStartMountedLoraScan(idle), true);
  assert.equal(shouldApplyMountedLoraScan({ ...idle, requestToken: 7, latestToken: 7, capturedRevision: 4, latestRevision: 4, responseEngine: "iL", activeEngine: "iL" }), true);
  const lockedBeforeCompletion = { ...idle, workspaceLocked: true };
  assert.equal(canStartMountedLoraScan(lockedBeforeCompletion), false);
  assert.equal(shouldApplyMountedLoraScan({ ...lockedBeforeCompletion, requestToken: 7, latestToken: 7, capturedRevision: 4, latestRevision: 4, responseEngine: "iL", activeEngine: "iL" }), false);
  assert.equal(canStartMountedLoraScan({ ...idle, status: "running" }), false);
  assert.equal(canStartMountedLoraScan({ ...idle, modelSwitching: true }), false);
  assert.equal(canStartMountedLoraScan({ ...idle, shouldPersist: false }), false);
  assert.equal(shouldApplyMountedLoraScan({ ...idle, requestToken: 8, latestToken: 8, capturedRevision: 5, latestRevision: 5, responseEngine: "iL", activeEngine: "iL" }), true);
});
test("cold active-job recovery gate blocks every engine until active request settles", () => {
  const recovery = { uiStateReady: true, activeJobRecoveryPending: true, status: "idle", modelSwitching: false, workspaceLocked: false, shouldPersist: true };
  assert.equal(canStartMountedLoraScan(recovery), false);
  assert.equal(shouldApplyMountedLoraScan({ ...recovery, requestToken: 1, latestToken: 1, capturedRevision: 0, latestRevision: 0, responseEngine: "SD", activeEngine: "SD" }), false);
  assert.equal(shouldApplyMountedLoraScan({ ...recovery, requestToken: 2, latestToken: 2, capturedRevision: 0, latestRevision: 0, responseEngine: "iL", activeEngine: "iL" }), false, "engine change cannot bypass pending recovery");
  const noActive = { ...recovery, activeJobRecoveryPending: false };
  assert.equal(canStartMountedLoraScan(noActive), true, "settled no-active response permits one normal scan");
  const active = { ...noActive, status: "running" };
  assert.equal(canStartMountedLoraScan(active), false, "settled active response stays locked by running status");
});
test("App wiring uses named engine transition fields, preserves asset changes, and locks refresh while running", async () => {
  const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(app, /sourceEngine: activeLoraScopeRef\.current,\s+sourceLoras: lorasRef\.current,\s+targetEngine: targetScopeKey/);
  assert.doesNotMatch(app, /sourceScopeKey:|targetScopeKey:/);
  assert.match(app, /const refreshLoras = async \(\) => \{\s+if \(lorasRefreshing \|\| !canStartMountedLoraScan\(loraScanGate\.current\)\) return;/);
  assert.match(app, /disabled=\{status === "running" \|\| lorasRefreshing \|\| modelSwitching \|\| loraWorkspaceLocked \|\| !shouldPersistMountedLoras\}/);
  assert.match(app, /if \(!canStartMountedLoraScan\(\{ uiStateReady, \.\.\.loraScanGate\.current \}\)\) \{\s+setLoraLoading\(false\);\s+setLorasRefreshing\(false\);\s+return undefined;/);
  assert.match(app, /const \[activeJobRecoveryPending, setActiveJobRecoveryPending\] = useState\(true\);/);
  assert.match(app, /loraScanGate\.current = \{ activeJobRecoveryPending,/);
  assert.match(app, /const activeJobRecoveryToken = useRef\(0\);/);
  assert.match(app, /const recoveryToken = \+\+activeJobRecoveryToken\.current;/);
  assert.match(app, /\.finally\(\(\) => \{ if \(activeJobRecoveryToken\.current === recoveryToken\) setActiveJobRecoveryPending\(false\); \}\);/);
  assert.match(app, /const gate = loraScanGate\.current;/);
  assert.match(app, /\.\.\.loraScanGate\.current/);
  assert.match(app, /shouldPersistMountedLoras: !mountedLoraMap\.fatal/);
  assert.match(app, /if \(\(promptPresetLibraryError && !shouldPersistPromptPresets\) \|\| !shouldPersistMountedLoras\) return;/);
});
test("scan race, sync engine and Gallery target list are engine keyed", () => {
  let map = withMountedLorasForScope(emptyMountedLoraMap(), "SD", [lora("sd")]);
  assert.equal(shouldApplyMountedLoraScan({ requestToken: 2, latestToken: 2, capturedRevision: 3, latestRevision: 3, responseEngine: "iL", activeEngine: "iL" }), true);
  assert.equal(shouldApplyMountedLoraScan({ requestToken: 2, latestToken: 2, capturedRevision: 3, latestRevision: 3, responseEngine: "SD", activeEngine: "iL" }), false);
  const synced = applyMountedLoraSync(map, { engine: "iL", loras: [lora("il")] }, { activeEngine: "SD" });
  assert.deepEqual(synced.activeLoras.map((x) => x.value), ["sd"]);
  assert.deepEqual(galleryMountedLorasForTarget(synced.container, { targetEngine: "iL", applyLoras: false }).loras.map((x) => x.value), ["il"]);
  assert.deepEqual(galleryMountedLorasForTarget(synced.container, { targetEngine: "iL", sourceLoras: [lora("gallery")], applyLoras: true }).loras.map((x) => x.value), ["gallery"]);
});

test("deferred automatic and manual scans cannot roll back newer local or cross-window iL edits", () => {
  const scannedCategories = [{ id: "character", models: [lora("existing")] }];
  const request = { requestToken: 9, latestToken: 9, capturedRevision: 41, responseEngine: "iL", activeEngine: "iL" };
  let map = withMountedLorasForScope(emptyMountedLoraMap(), "SD", [lora("sd-keep")]);
  map = withMountedLorasForScope(map, "iL", [lora("existing")]);

  // A deferred automatic response began at revision 41. A local iL mount at
  // 42 is absent from its listing, but the completion cannot prune it.
  const localEdit = withMountedLorasForScope(map, "iL", [lora("existing"), lora("local-new")]);
  let staleMapReads = 0;
  const automatic = reconcileMountedLoraScan({ getCurrentContainer: () => { staleMapReads += 1; return localEdit; }, categories: scannedCategories, scanScopeKey: "iL", latestRevision: 42, ...request });
  assert.equal(automatic.applied, false);
  assert.equal(staleMapReads, 0, "revision rejection reads no current map");
  assert.equal(automatic.container, undefined);
  assert.deepEqual(mountedLorasForScope(localEdit, "iL").map((item) => item.value), ["existing", "local-new"]);
  assert.deepEqual(mountedLorasForScope(localEdit, "SD").map((item) => item.value), ["sd-keep"]);

  // The same protection applies to an incoming BroadcastChannel replacement
  // and to manual refresh; neither stale completion has any apply/prune result.
  const incoming = applyMountedLoraSync(map, { engine: "iL", loras: [lora("existing"), lora("sync-new", { weight: .5 })] }, { activeEngine: "iL" });
  assert.equal(incoming.applied, true);
  const incomingSync = incoming.container;
  const crossWindow = reconcileMountedLoraScan({ container: incomingSync, categories: scannedCategories, scanScopeKey: "iL", latestRevision: 42, ...request });
  assert.equal(crossWindow.applied, false);
  assert.deepEqual(mountedLorasForScope(crossWindow.container, "iL").map((item) => item.value), ["existing", "sync-new"]);
  const manual = reconcileMountedLoraScan({ container: localEdit, categories: scannedCategories, scanScopeKey: "iL", latestRevision: 42, ...request });
  assert.equal(manual.applied, false);
  assert.equal(manual.categories, null, "stale refresh cannot replace library UI");
});

test("same-revision scan prunes only its engine, while newest token and revision must both match", () => {
  let map = withMountedLorasForScope(emptyMountedLoraMap(), "SD", [lora("sd-keep")]);
  map = withMountedLorasForScope(map, "iL", [lora("existing"), lora("removed")]);
  const categories = [{ id: "character", models: [lora("existing")] }];
  const current = reconcileMountedLoraScan({
    container: map, categories, scanScopeKey: "iL", requestToken: 12, latestToken: 12,
    capturedRevision: 7, latestRevision: 7, responseEngine: "iL", activeEngine: "iL",
  });
  assert.equal(current.applied, true);
  assert.equal(current.changed, true);
  assert.deepEqual(mountedLorasForScope(current.container, "iL").map((item) => item.value), ["existing"]);
  assert.deepEqual(mountedLorasForScope(current.container, "SD").map((item) => item.value), ["sd-keep"]);
  assert.equal(reconcileMountedLoraScan({ container: map, categories, scanScopeKey: "iL", requestToken: 11, latestToken: 12, capturedRevision: 7, latestRevision: 7, responseEngine: "iL", activeEngine: "iL" }).applied, false, "older token loses even at the current revision");
  assert.equal(reconcileMountedLoraScan({ container: map, categories, scanScopeKey: "iL", requestToken: 12, latestToken: 12, capturedRevision: 6, latestRevision: 7, responseEngine: "iL", activeEngine: "iL" }).applied, false, "latest token loses after a mutation");
});

test("App has one semantic map commit boundary for the non-persisted scan revision", async () => {
  const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(app, /const mountedLoraRevisionRef = useRef\(0\);/);
  assert.match(app, /const commitMountedLoraMap = \(container, \{ rescan = true \} = \{\}\) => \{\s+const previous = mountedLorasMapRef\.current;\s+const nextRevision = nextMountedLoraRevision\(mountedLoraRevisionRef\.current, previous, container\);/);
  assert.match(app, /if \(changed\) \{\s+mountedLoraRevisionRef\.current = nextRevision;/);
  assert.doesNotMatch(app, /scopeTransition|forceRevision|mountedLoraRevisionRef\.current \+=/);
  assert.match(app, /const result = updateMountedLorasForScope[\s\S]*?commitMountedLoraMap\(result\.container\)/);
  assert.match(app, /activeLoraScopeRef\.current = targetScopeKey;\s+lorasRef\.current = nextLoras;\s+commitMountedLoraMap\(container\);\s+setLoras\(nextLoras\);/);
  assert.match(app, /commitMountedLoraMap\(workspace\.mountedLorasByEngine\);/);
  assert.match(app, /commitMountedLoraMap\(synced\.container\)/);
  assert.match(app, /commitMountedLoraMap\(scan\.container, \{ rescan: false \}\)/);
  assert.equal((app.match(/mountedLorasMapRef\.current =/g) || []).length, 1, "map ref has no bypass outside the commit boundary");
  assert.equal((app.match(/setMountedLorasByEngine\(/g) || []).length, 1, "map state has no bypass outside the commit boundary");
  assert.match(app, /capturedRevision: scanRevision,\s+latestRevision: mountedLoraRevisionRef\.current/);
  assert.match(app, /const galleryTarget = galleryMountedLorasForTarget\(mountedLorasMapRef\.current/);
  assert.match(app, /commitMountedLoras\(\(\) => galleryTargetLoras, targetLoraScopeKey\)/);
  assert.doesNotMatch(app, /mountedLorasByModel/);
});
