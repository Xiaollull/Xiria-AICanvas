import assert from "node:assert/strict";
import { test } from "node:test";
import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, writeFile, readFile, rename, link } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { LoraRelinkService } from "./lora-relink-service.mjs";
import { galleryOwnerParticipant } from "./lora-gallery-owner.mjs";
import { resolveLibraryReference } from "../src/lora-identity.js";
import { readLoraRegistry, registerObservedInventory, emptyLoraRegistry } from "./lora-library.mjs";
import { pythonDependencyOptions } from "./python-dependency.test.helper.mjs";

const python = process.env.P10_PYTHON || "python";
const galleryOptions = pythonDependencyOptions(python, path.resolve(import.meta.dirname, ".."), ["PIL"]);
const galleryTest = (name, body) => test(name, galleryOptions, body);

const parent = process.env.P10_EVIDENCE_DIR || path.join(os.tmpdir(), "opencode", "p10-l09");
async function fixture() {
  await mkdir(parent, { recursive: true });
  const dir = await mkdtemp(path.join(parent, "l09-")), stateDirectory = path.join(dir, "state");
  const root = { engine: "SD", id: "local-SD", directory: path.join(dir, "models") };
  await mkdir(root.directory); await mkdir(stateDirectory);
  await writeFile(path.join(root.directory, "A.safetensors"), "full bytes A");
  const service = new LoraRelinkService({ stateDirectory, getRoots: async () => [root],
    gallery: galleryOwnerParticipant(stateDirectory, python), admitIdle: async () => {}, validatePath: async () => {} });
  const baseline = async () => {
    const scan = await service.scan({ engines: ["SD"] });
    const payload = { scanId: scan.scanId, selections: [], registerBaseline: true, operationId: randomUUID() };
    const result = await service.commit(payload);
    return { scan, payload, result };
  };
  await baseline();
  const A = Object.values((await readLoraRegistry(stateDirectory)).instances)[0];
  await rename(path.join(root.directory, A.value), path.join(root.directory, "M.safetensors"));
  const scan = await service.scan({ engines: ["SD"] });
  const moved = await service.commit({ scanId: scan.scanId, operationId: randomUUID(), selections: [{ sourceId: A.instanceId,
    target: { value: "M.safetensors", rootId: root.id, kind: "relocate" }, confirmedOrigin: true, scope: { all: true, history: true } }] });
  return { dir, stateDirectory, root, service, baseline, A, moved };
}

for (const sameBytes of [false, true]) galleryTest(`L09 ${sameBytes ? "equal-byte" : "different-byte"} new occupant: draft UUID, independent baseline, strong references and owned undo`, async () => {
  const f = await fixture();
  await writeFile(path.join(f.root.directory, f.A.value), sameBytes ? "full bytes A" : "new occupant C");
  const ref = { value: f.A.value, weight: .75 };
  await assert.rejects(f.service.resolve("SD", [ref], { history: true }), error => error.statusCode === 409 && error.resolution.auditKind === "path-reuse");
  const before = await readFile(path.join(f.stateDirectory, "lora-library/registry.json"));
  const scan = await f.service.scan({ engines: ["SD"] });
  const draft = scan.observations.find(item => item.value === f.A.value);
  assert.ok(draft.instanceId); assert.notEqual(draft.instanceId, f.A.instanceId);
  assert.deepEqual(await readFile(path.join(f.stateDirectory, "lora-library/registry.json")), before);
  const payload = { scanId: scan.scanId, selections: [], registerBaseline: true, operationId: randomUUID() };
  await f.service.preview(payload); await f.service.preview(payload);
  await f.service.commit(payload); assert.equal((await f.service.commit(payload)).replay, true);
  await assert.rejects(f.service.commit({ ...payload, registerBaseline: false }), /different draft/);
  let registry = await readLoraRegistry(f.stateDirectory);
  const C = registry.instances[draft.instanceId];
  assert.equal(C.value, f.A.value); assert.equal(registry.instances[f.A.instanceId].value, "M.safetensors");
  assert.equal(Object.keys(registry.instances).length, 2);
  const verdict = resolveLibraryReference(registry, "SD", ref);
  assert.equal(verdict.status, "ambiguous-path"); assert.equal(verdict.candidates.length, 2);
  await assert.rejects(f.service.resolve("SD", [ref]), /ambiguous-path/);
  for (const [id, value] of [[f.A.instanceId, "M.safetensors"], [C.instanceId, C.value]]) {
    assert.equal((await f.service.resolve("SD", [{ ...ref, instanceId: id }])).loras[0].path, value);
  }
  await assert.rejects(f.service.resolve("iL", [{ ...ref, instanceId: C.instanceId }]), /unresolved/);
  if (!sameBytes) assert.equal((await f.service.resolve("SD", [{ ...ref, contentHash: f.A.contentHash }], { history: true })).loras[0].path, "M.safetensors");
  else await assert.rejects(f.service.resolve("SD", [{ ...ref, contentHash: f.A.contentHash }]), /ambiguous/);
  const repeated = await f.baseline();
  registry = await readLoraRegistry(f.stateDirectory);
  assert.deepEqual(Object.keys(registry.instances).sort(), [C.instanceId, f.A.instanceId].sort());
  await f.service.undo({ operationId: f.moved.operationId, undoOperationId: randomUUID() });
  await f.service.undo({ operationId: repeated.result.operationId, undoOperationId: randomUUID() });
  assert.deepEqual((await readLoraRegistry(f.stateDirectory)).instances, registry.instances);
});

galleryTest("L09 changed baseline target returns 409 with unchanged registry; skipped draft is no-op", async () => {
  const f = await fixture(); await writeFile(path.join(f.root.directory, f.A.value), "C");
  const scan = await f.service.scan({ engines: ["SD"] });
  const payload = { scanId: scan.scanId, selections: [], registerBaseline: true, operationId: randomUUID() };
  await f.service.preview(payload);
  const before = await readFile(path.join(f.stateDirectory, "lora-library/registry.json"));
  assert.equal((await f.service.commit({ ...payload, registerBaseline: false })).status, "no-op");
  await writeFile(path.join(f.root.directory, f.A.value), "C changed");
  await assert.rejects(f.service.commit(payload), /changed/);
  assert.deepEqual(await readFile(path.join(f.stateDirectory, "lora-library/registry.json")), before);
});

galleryTest("L09 changed current bytes create an independent missing-source/current-occupant pair", async () => {
  const f = await fixture();
  await writeFile(path.join(f.root.directory, "M.safetensors"), "different current bytes");
  const { scan } = await f.baseline();
  const C = scan.observations.find(item => item.value === "M.safetensors");
  assert.notEqual(C.instanceId, f.A.instanceId);
  const registry = await readLoraRegistry(f.stateDirectory);
  assert.equal(registry.instances[f.A.instanceId].status, "missing");
  assert.equal(registry.instances[f.A.instanceId].contentHash, f.A.contentHash);
  await assert.rejects(f.service.resolve("SD", [{ value: "M.safetensors" }]), /ambiguous/);
  await assert.rejects(f.service.resolve("SD", [{ value: "M.safetensors", instanceId: f.A.instanceId }]), /missing/);
  assert.equal((await f.service.resolve("SD", [{ value: "M.safetensors", instanceId: C.instanceId }])).loras[0].path, "M.safetensors");
});

galleryTest("L09 hardlink occupants remain separate IDs and explicit existing-target local replacement adds no alias", async () => {
  const f = await fixture(); await link(path.join(f.root.directory, "M.safetensors"), path.join(f.root.directory, f.A.value));
  const { scan } = await f.baseline();
  const C = scan.observations.find(item => item.value === f.A.value);
  assert.notEqual(C.instanceId, f.A.instanceId);
  await writeFile(path.join(f.stateDirectory, "ui-state.json"), JSON.stringify({ workspace: { model: "SD", loras: [{ value: "M.safetensors", instanceId: f.A.instanceId, weight: .4 }],
    loraGroupsByEngine: { schemaVersion: 1, byEngine: { SD: [{ id: "only", enabled: false, members: [{ value: "M.safetensors", instanceId: f.A.instanceId, weight: .8 }] }] } } } }));
  const next = await f.service.scan({ engines: ["SD"] });
  const payload = { scanId: next.scanId, operationId: randomUUID(), selections: [{ sourceId: f.A.instanceId,
    target: { value: C.value, rootId: f.root.id, kind: "replace" }, scope: { groupIds: ["only"] } }] };
  const preview = await f.service.preview(payload); assert.equal(preview.results[0].targetId, C.instanceId);
  const result = await f.service.commit(payload); assert.equal(result.results[0].targetId, C.instanceId);
  const ws = JSON.parse(await readFile(path.join(f.stateDirectory, "ui-state.json"))).workspace;
  assert.equal(ws.loras[0].instanceId, f.A.instanceId); assert.equal(ws.loraGroupsByEngine.byEngine.SD[0].members[0].instanceId, C.instanceId);
  assert.deepEqual((await readLoraRegistry(f.stateDirectory)).substitutions, []);
});

test("L09 current+historical indexing is order independent, Windows-folded only on request, and full hash is not copy provenance", () => {
  const idA = randomUUID(), idC = randomUUID();
  const record = (id, value, history, hash) => ({ instanceId: id, engine: "SD", rootId: "local-SD", value, history, contentHash: hash.repeat(64), contentRevision: 1, status: "present" });
  for (const records of [[record(idA, "M.safetensors", ["A.safetensors"], "a"), record(idC, "A.safetensors", [], "a")],
    [record(idC, "A.safetensors", [], "a"), record(idA, "M.safetensors", ["A.safetensors"], "a")]]) {
    const registry = { ...emptyLoraRegistry(), instances: Object.fromEntries(records.map(x => [x.instanceId, x])) };
    assert.equal(resolveLibraryReference(registry, "SD", { value: "a.safetensors" }, { caseInsensitive: true }).status, "ambiguous-path");
    assert.equal(resolveLibraryReference(registry, "SD", { value: "a.safetensors" }).status, "unresolved");
    assert.equal(resolveLibraryReference(registry, "SD", { value: "A.safetensors", contentHash: "a".repeat(64) }).status, "ambiguous-path");
    assert.equal(resolveLibraryReference(registry, "SD", { value: "unknown.safetensors", contentHash: "a".repeat(64) }).status, "unresolved");
  }
  assert.throws(() => registerObservedInventory(emptyLoraRegistry(), { complete: false }), /Complete/);
});
