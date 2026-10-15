import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { createLoraRepairController } from "../src/lora-repair-controller.js";
import { pathOnlyFamilyLoras, provenIdentity, registerEngineLoraIdentities, resolveFamilyLoraIdentities, withFamilyLoraIdentities } from "../src/xyz-lora-identity.js";
import { xyzStrongLora } from "../src/xyz-sidebar-context.js";
import { galleryOwnerParticipant } from "./lora-gallery-owner.mjs";
import { pythonDependencyOptions } from "./python-dependency.test.helper.mjs";

const root = path.resolve(import.meta.dirname, "..");
const ID = "0f8fad5b-d9cb-469f-a165-70867728950e", HASH = "a".repeat(64);
const identity = { identitySchema: 1, instanceId: ID, contentHash: HASH, contentRevision: 1 };
const strong = { value: "style/strong.safetensors", weight: "1", ...identity, instanceId: "7c9e6679-7425-40de-944b-e07fc1f90ae7" };
// The shape xiria-39 saved: mounts synced into the draft before the library was ever registered.
const draft = () => ({
  kind: "xyz_draft", engine: "Krea2",
  baseline: {
    loras: [{ value: "style/A.safetensors", name: "A", weight: "1", enabled: true }, strong, { value: "style/B.safetensors", weight: "0.5", groupId: "g1" }],
    group_snapshots: [{ id: "g1", enabled: true, members: [{ value: "style/B.safetensors", weight: "0.5" }] }],
  },
});
const resolvedAs = (value, extra = {}) => ({ value, status: "resolved", kind: "current", instanceId: ID, contentHash: HASH, contentRevision: 1, ...extra });

test("path-only references are found once each, strong ones are left alone", () => {
  assert.deepEqual(pathOnlyFamilyLoras(draft()), [{ value: "style/A.safetensors" }, { value: "style/B.safetensors" }]);
  const mix = { kind: "lora_mix_draft", baseline: { loras: [] }, mix: { members: [{ ref: { value: "style/C.safetensors" } }, { ref: strong }] } };
  assert.deepEqual(pathOnlyFamilyLoras(mix), [{ value: "style/C.safetensors" }]);
});

test("an identity is added only to the path-only references of its own path", () => {
  const before = draft();
  const after = withFamilyLoraIdentities(before, new Map([["style/B.safetensors", identity]]));
  assert.equal(after.baseline.loras[0].instanceId, undefined, "an unanswered path stays path-only");
  assert.deepEqual(after.baseline.loras[1], strong, "an existing identity is never rewritten");
  assert.deepEqual(after.baseline.loras[2], { value: "style/B.safetensors", weight: "0.5", groupId: "g1", ...identity });
  assert.ok(xyzStrongLora(after.baseline.loras[2]));
  assert.deepEqual(after.baseline.group_snapshots[0].members[0], { value: "style/B.safetensors", weight: "0.5", ...identity });
  assert.equal(before.baseline.loras[2].instanceId, undefined, "the input draft is not mutated");
  assert.equal(withFamilyLoraIdentities(before, new Map([["style/unknown.safetensors", identity]])), before, "nothing to apply returns the same draft");
  const noGroups = { kind: "lora_mix_draft", baseline: { loras: [{ value: "style/A.safetensors" }] }, mix: { members: [] } };
  assert.equal(Object.hasOwn(withFamilyLoraIdentities(noGroups, new Map([["style/A.safetensors", identity]])).baseline, "group_snapshots"), false);
});

test("only a same-path, byte-proven resolution counts as an identity", () => {
  assert.deepEqual(provenIdentity(resolvedAs("style/A.safetensors"), "style/A.safetensors"), identity);
  assert.equal(provenIdentity(resolvedAs("concept/A.safetensors", { kind: "relocate" }), "style/A.safetensors"), null, "a relocation needs an explicit sync");
  assert.equal(provenIdentity({ value: "style/A.safetensors", status: "unresolved" }, "style/A.safetensors"), null);
  assert.equal(provenIdentity(resolvedAs("style/A.safetensors", { contentHash: null }), "style/A.safetensors"), null);
});

test("one unresolvable path does not cost the others their identity", async () => {
  const calls = [];
  const resolve = async (engine, references) => {
    calls.push(references.map((item) => item.value));
    if (references.some((item) => item.value === "style/moved.safetensors")) throw new Error("LoRA reference is missing; choose an explicit target");
    return { references: references.map((item) => resolvedAs(item.value)) };
  };
  const identities = await resolveFamilyLoraIdentities("Krea2", [{ value: "style/A.safetensors" }, { value: "style/moved.safetensors" }, { value: "style/B.safetensors" }], { resolve });
  assert.deepEqual([...identities.keys()], ["style/A.safetensors", "style/B.safetensors"]);
  assert.equal(calls.length, 4, "the batch, then each reference on its own");
});

function fakeLibrary({ scans, commitStatus = 200 }) {
  const requests = [];
  const request = async (url, options) => {
    const body = JSON.parse(options.body);
    requests.push({ action: url.split("/").at(-1), body });
    if (url.endsWith("/scan")) return { ok: true, json: async () => scans.shift() };
    return { ok: commitStatus === 200, status: commitStatus, json: async () => (commitStatus === 200 ? { revision: 4, operationId: body.operationId } : { error: "LoRA library is busy" }) };
  };
  return { request, requests };
}
const fullScan = { scanId: "s2", complete: true, issues: [], domain: { fullRoots: true }, observations: [{}, {}, {}] };

test("registration is a full engine scan, repeated while only the byte budget ran out, then a baseline commit", async () => {
  const library = fakeLibrary({ scans: [{ scanId: "s1", complete: false, issues: [{ reason: "observation-413", relative: "style/big.safetensors" }], domain: { fullRoots: true } }, fullScan] });
  const progress = [];
  const result = await registerEngineLoraIdentities("Krea2", { request: library.request, newId: () => ID, onProgress: (text) => progress.push(text) });
  assert.deepEqual(result, { files: 3, revision: 4 });
  assert.deepEqual(library.requests.map((item) => item.action), ["scan", "scan", "commit"]);
  assert.deepEqual(library.requests[0].body, { engines: ["Krea2"], maxBytes: 32 * 1024 ** 3 });
  assert.deepEqual(library.requests[2].body, { scanId: "s2", selections: [], registerBaseline: true, operationId: ID });
  assert.equal(progress.length, 3);
});

test("registration refuses an incomplete scan and surfaces a refused commit", async () => {
  const unreadable = fakeLibrary({ scans: [{ scanId: "s1", complete: false, issues: [{ reason: "EACCES", relative: "style/locked.safetensors" }], domain: { fullRoots: true } }] });
  await assert.rejects(registerEngineLoraIdentities("iL", { request: unreadable.request }), /style\/locked\.safetensors：EACCES/);
  assert.deepEqual(unreadable.requests.map((item) => item.action), ["scan"], "nothing is committed from a partial scan");
  const busy = fakeLibrary({ scans: [fullScan], commitStatus: 409 });
  await assert.rejects(registerEngineLoraIdentities("iL", { request: busy.request }), /LoRA library is busy/);
});

test("the repair panel commits on a plain-HTTP page, where crypto.randomUUID does not exist", async () => {
  Object.defineProperty(globalThis.crypto, "randomUUID", { value: undefined, configurable: true, writable: true });
  try {
    let committed;
    const response = (value) => ({ ok: true, json: async () => value });
    const controller = createLoraRepairController({ request: async (url, options) => {
      if (url.endsWith("/state")) return response({ available: true, contractVersion: 1 });
      if (url.endsWith("/scan")) return response({ scanId: "scan", complete: true, revision: 1, rows: [{ sourceId: "a", recommendation: { value: "concept/A", kind: "relocate" }, impacts: { groups: [], cards: [], mounts: 1 } }] });
      if (url.endsWith("/preview")) return response({ conflicts: [], affected: { workspace: 1, cards: 0, gallery: 0 } });
      committed = JSON.parse(options.body);
      return response({ operationId: committed.operationId, status: "published", revision: 2 });
    } });
    await controller.availability(); await controller.scan(["SD"]); await controller.preview(); await controller.commit();
    assert.equal(controller.state().error, "");
    assert.equal(controller.state().phase, "result");
    assert.match(committed.operationId, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  } finally {
    delete globalThis.crypto.randomUUID;
  }
});

test("a refused first scan is shown instead of silently returning to the empty state", async () => {
  const panel = await readFile(path.join(root, "src", "LoraRepairPanel.jsx"), "utf8");
  assert.match(panel, /!state\.scan && state\.phase !== "scanning" && state\.error && <div className="lora-repair-error" role="alert">/);
});

test("LoRA management admits same-origin callers, so remote instances can register", async () => {
  const config = await readFile(path.join(root, "vite.config.js"), "utf8");
  const plugin = config.slice(config.indexOf("function loraRelinkApiPlugin"), config.indexOf("async function uiStateApi"));
  assert.match(plugin, /admit: \(request\) => \{ requireSameOrigin\(request\);/);
  assert.match(plugin, /sec-fetch-site"\] === "cross-site"/);
  assert.doesNotMatch(plugin, /requireLocalRequest/);
});

const python = process.env.P10_PYTHON || "python";
test("the Gallery owner names the bound that stopped it", pythonDependencyOptions(python, root, ["PIL"]), async () => {
  const stateDirectory = await mkdtemp(path.join(os.tmpdir(), "xirai-owner-bound-"));
  try {
    // No gallery.sqlite3: the owner answers an empty listing (34 bytes).
    assert.deepEqual(await galleryOwnerParticipant(stateDirectory, python).list(null), { cards: [], collections: [] });
    await assert.rejects(galleryOwnerParticipant(stateDirectory, python, { maximumReplyBytes: 10 }).list(null),
      (error) => error.statusCode === 503 && error.message === "Gallery owner stopped (list): reply larger than 10 bytes");
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("the Gallery owner's reply bound fits a real Gallery (144 cards answered 2.85 MB)", async () => {
  const source = await readFile(path.join(root, "scripts", "lora-gallery-owner.mjs"), "utf8");
  const bound = Number(eval(/const MAXIMUM_OWNER_REPLY_BYTES = ([\d *]+);/.exec(source)[1]));
  assert.ok(bound >= 64 * 1024 * 1024, `bound ${bound} is too small for a Gallery of a few thousand cards`);
});
