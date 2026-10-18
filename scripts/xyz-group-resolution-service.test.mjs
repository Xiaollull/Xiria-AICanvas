import assert from "node:assert/strict";
import test from "node:test";
import http from "node:http";
import { randomUUID } from "node:crypto";
import { spawnSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { LoraRelinkService } from "./lora-relink-service.mjs";
import { loraRelinkMiddleware } from "./lora-relink-http.mjs";
import { galleryOwnerParticipant } from "./lora-gallery-owner.mjs";
import { readLoraRegistry } from "./lora-library.mjs";
import { READY_LORA_ENGINES } from "../src/lora-model-scope.js";
import { resolveLoraReferences } from "../src/lora-library-client.js";
import { analyzeXYZGroupCopy, projectResolvedGroup, resolveGroupMembers } from "../src/xyz-group-resolution.js";
import { copyXYZManagerGroup } from "../src/xyz-manager-context.js";
import { createXYZDraft } from "../src/xyz-draft.js";
import { captureXYZPromptSource } from "../src/xyz-prompt-source.js";
import { pythonDependencyOptions } from "./python-dependency.test.helper.mjs";

// QPlan2/P2.1 G01-G06/G14: the real HTTP route, the real client and the real service verify
// registered shared/local groups. Isolated temp roots only; the example names are data.
const python = process.env.P10_PYTHON || "python", cwd = path.resolve(import.meta.dirname, "..");
const galleryOptions = pythonDependencyOptions(python, cwd, ["PIL"]);
const parent = process.env.P10_EVIDENCE_DIR || path.join(os.tmpdir(), "opencode", "p2.1-fixtures");
const SHARED = "5738b73b9323";
const NAMES = ["ashima_v4", "yusano_v5_step1500", "kiriaina_anima_style-000030", "extra_a", "extra_b"];

async function put(root, name, bytes) { const location = path.join(root, name); await mkdir(path.dirname(location), { recursive: true }); await writeFile(location, bytes); }
async function fixture() {
  await mkdir(parent, { recursive: true });
  const dir = await mkdtemp(path.join(parent, "g-")), stateDirectory = path.join(dir, "state");
  const local = { engine: "Anima", id: "local-Anima", directory: path.join(dir, "models") };
  const shared = { engine: "Anima", id: SHARED, directory: path.join(dir, "comfy"), shared: true, label: "loras" };
  await mkdir(stateDirectory, { recursive: true });
  // Equal bytes in both roots: the shared and local copies of one name are different instances with one hash.
  for (const name of NAMES) {
    await put(local.directory, `style/NekoX/${name}.safetensors`, `bytes ${name}`);
    await put(shared.directory, `Anima/Style/${name}.safetensors`, `bytes ${name}`);
  }
  let roots = [local, shared];
  const service = new LoraRelinkService({ stateDirectory, gallery: galleryOwnerParticipant(stateDirectory, python), getRoots: async () => roots, admitIdle: async () => {}, validatePath: async () => {} });
  const scan = await service.scan({ engines: ["Anima"] });
  await service.commit({ scanId: scan.scanId, selections: [], registerBaseline: true, operationId: randomUUID() });
  const middleware = loraRelinkMiddleware({ service, admit: () => {}, readJson: (request) => new Promise((resolve, reject) => {
    const chunks = []; request.on("data", (chunk) => chunks.push(chunk)); request.on("end", () => { try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8"))); } catch (error) { reject(error); } });
  }) });
  const server = http.createServer((request, response) => middleware(request, response, () => { response.statusCode = 404; response.end("{}"); }));
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}`, realFetch = globalThis.fetch;
  globalThis.fetch = (url, init) => realFetch(typeof url === "string" && url.startsWith("/") ? base + url : url, init);
  return { dir, stateDirectory, local, shared, service, setRoots: (next) => { roots = next; },
    close: async () => { globalThis.fetch = realFetch; await new Promise((resolve) => server.close(resolve)); await rm(dir, { recursive: true, force: true }); } };
}
const legacy = (value, name, extra = {}) => ({ value, name, category: "style", weight: 0.6, precision: 4, enabled: true, ...extra });
const sharedMember = (name, extra = {}) => ({ value: `shared:${SHARED}/Anima/Style/${name}.safetensors`, name, category: "style", weight: 0.6, precision: 4, enabled: true, ...extra });
const localMember = (name, extra = {}) => ({ value: `style/NekoX/${name}.safetensors`, name, category: "style", weight: 0.4, precision: 2, enabled: true, ...extra });
const group = (id, members) => ({ id, name: id, enabled: false, collecting: false, presetPrompt: "words", members });
const baseline = () => ({ parameters: { engine: "Anima", diffusion_model: "d.safetensors", text_encoder: "t.safetensors", vae: "v.safetensors", prompt: "raw", negative_prompt: "", steps: 20, cfg: 4,
  width: 512, height: 512, seed: "1", sampler: "euler", scheduler: "normal", guidance: "none", pag: { scale: 0.3, applied_layers: "mid" }, postprocess_chain: [] },
  loras: [], group_snapshots: [], prompt_without_groups: "raw", group_prompt: "", seed_mode: "fixed", library_revision: 0 });
const fresh = () => createXYZDraft(captureXYZPromptSource(baseline(), { positive: "raw", negative: "", records: [], active: { positive: [], negative: [] } }));
const verify = async (source) => { const resolution = await resolveGroupMembers("Anima", source, { resolve: resolveLoraReferences }); return { resolution, projection: projectResolvedGroup(source, resolution) }; };

test("G01/G04/G14 registered legacy shared members resolve through the real route; disabled and zero members survive", galleryOptions, async () => {
  const f = await fixture();
  try {
    const source = group("g-shared", NAMES.map((name, index) => sharedMember(name, index === 1 ? { enabled: false } : index === 2 ? { weight: 0 } : index === 3 ? { weight: -1.2345 } : {})));
    const before = JSON.stringify(source), registryBefore = await readFile(path.join(f.stateDirectory, "lora-library/registry.json"));
    const { resolution, projection } = await verify(source);
    assert.equal(resolution.stale, false);
    assert.equal(projection.ok, true, JSON.stringify(projection.issues));
    assert.equal(JSON.stringify(source), before);
    assert.equal(projection.group.members.length, 5);
    const registry = await readLoraRegistry(f.stateDirectory);
    projection.group.members.forEach((member, index) => {
      const record = registry.instances[member.instanceId];
      assert.equal(record.value, source.members[index].value);
      assert.equal(member.contentHash, record.contentHash);
      assert.equal(member.contentRevision, record.contentRevision);
      assert.equal(member.identitySchema, 1);
      for (const field of ["weight", "precision", "enabled", "name", "category"]) assert.equal(member[field], source.members[index][field]);
    });
    const draft = copyXYZManagerGroup(fresh(), projection.group);
    assert.deepEqual(draft.baseline.loras.map((item) => [item.weight, item.enabled]), [[0.6, true], [0.6, false], [0, true], [-1.2345, true], [0.6, true]]);
    assert.deepEqual(await readFile(path.join(f.stateDirectory, "lora-library/registry.json")), registryBefore, "resolution registered and changed nothing");
    // The service's own filter would have dropped the disabled member: that is why references are binding-only.
    assert.equal((await f.service.resolve("Anima", [{ value: source.members[1].value, enabled: false }])).references.length, 0);
  } finally { await f.close(); }
});

test("G02/G06 the equal-byte local copy resolves to a different instance and conflicts with the enabled shared snapshot", galleryOptions, async () => {
  const f = await fixture();
  try {
    const shared = (await verify(group("g-shared", NAMES.slice(0, 3).map((name) => sharedMember(name))))).projection.group;
    const local = (await verify(group("g-local", NAMES.slice(0, 3).map((name) => localMember(name))))).projection.group;
    shared.members.forEach((member, index) => {
      assert.notEqual(member.instanceId, local.members[index].instanceId);
      assert.equal(member.contentHash, local.members[index].contentHash);
    });
    const withShared = copyXYZManagerGroup(fresh(), shared);
    const analysis = analyzeXYZGroupCopy(withShared, local);
    assert.deepEqual(analysis.actions, { "copy-only": true, enable: false, replace: false });
    assert.equal(analysis.enableBlockers.length, 3);
    assert.throws(() => copyXYZManagerGroup(withShared, local), /内容相同/);
    const kept = copyXYZManagerGroup(withShared, local, { mode: "copy-only" });
    assert.deepEqual(kept.baseline.loras.map((item) => item.groupId), ["g-shared", "g-shared", "g-shared"]);
  } finally { await f.close(); }
});

test("G03/G14 unregistered, forged, changed, removed and wrong-engine data never become a strong snapshot", galleryOptions, async () => {
  const f = await fixture();
  try {
    await put(f.local.directory, "style/NekoX/new_unscanned.safetensors", "never scanned");
    const registryBefore = await readFile(path.join(f.stateDirectory, "lora-library/registry.json"));
    // Untouched legacy fallback: HTTP succeeds but the verdict is unresolved, and nothing is registered.
    const fallback = await verify(group("g-new", [localMember("new_unscanned")]));
    assert.equal(fallback.projection.ok, false);
    assert.equal(fallback.projection.issues[0].code, "unresolved");
    assert.deepEqual(await readFile(path.join(f.stateDirectory, "lora-library/registry.json")), registryBefore);

    const good = (await verify(group("g-ok", [localMember("ashima_v4")]))).projection.group.members[0];
    const forged = await verify(group("g-forged", [{ ...good, instanceId: randomUUID() }]));
    assert.equal(forged.projection.ok, false);
    const wrongHash = await verify(group("g-hash", [{ ...good, contentHash: "f".repeat(64) }]));
    assert.equal(wrongHash.projection.ok, false);

    await writeFile(path.join(f.local.directory, "style/NekoX/ashima_v4.safetensors"), "changed bytes!");
    const changed = await verify(group("g-changed", [good]));
    assert.equal(changed.projection.ok, false);
    assert.match(changed.projection.issues[0].message, /changed|内容/);
    await rm(path.join(f.local.directory, "style/NekoX/ashima_v4.safetensors"));
    assert.equal((await verify(group("g-removed", [good]))).projection.ok, false);

    const other = await resolveGroupMembers("SD", group("g-engine", [localMember("yusano_v5_step1500")]), { resolve: resolveLoraReferences });
    assert.equal(projectResolvedGroup(group("g-engine", [localMember("yusano_v5_step1500")]), other).ok, false);

    f.setRoots([f.local]);
    const gone = await verify(group("g-root", [sharedMember("extra_a")]));
    assert.equal(gone.projection.ok, false);
    assert.equal((await readLoraRegistry(f.stateDirectory)).revision, JSON.parse(registryBefore.toString()).revision);
  } finally { await f.close(); }
});

// The real backend planner is the final judge: projected members must pass its strict reference model and
// registry identity check, and an equal-byte pair must be refused there exactly as the UI refuses it.
function planner(entries, engine, stateDirectory) {
  const code = "import json,sys\nfrom backend.xyz_planner import checked_loras, PlanningError\nentries,engine,registry=json.load(open(sys.argv[1],encoding='utf-8')),sys.argv[2],json.load(open(sys.argv[3],encoding='utf-8'))\n"
    + "try:\n    print(json.dumps({'ok':True,'count':len(checked_loras(entries,engine,registry))}))\nexcept PlanningError as error:\n    print(json.dumps({'ok':False,'code':error.code}))\n";
  const input = path.join(stateDirectory, `planner-${randomUUID()}.json`);
  return writeFile(input, JSON.stringify(entries)).then(() => {
    const child = spawnSync(python, ["-B", "-c", code, input, engine, path.join(stateDirectory, "lora-library/registry.json")], { cwd, encoding: "utf8", windowsHide: true });
    assert.equal(child.status, 0, child.stderr);
    return JSON.parse(child.stdout);
  });
}

test("G05/G06 the backend planner accepts projected strong members and refuses the equal-byte pair", pythonDependencyOptions(python, cwd, ["PIL", "pydantic"]), async () => {
  const f = await fixture();
  try {
    const shared = (await verify(group("g-shared", NAMES.slice(0, 3).map((name) => sharedMember(name))))).projection.group;
    const local = (await verify(group("g-local", NAMES.slice(0, 3).map((name) => localMember(name))))).projection.group;
    assert.deepEqual(await planner(shared.members, "Anima", f.stateDirectory), { ok: true, count: 3 });
    assert.deepEqual(await planner(local.members, "Anima", f.stateDirectory), { ok: true, count: 3 });
    assert.deepEqual(await planner([...shared.members, ...local.members], "Anima", f.stateDirectory), { ok: false, code: "duplicate-content" });
    // A dormant (disabled) copy is not fused, so the planner agrees with the UI's dormant rule.
    const dormant = local.members.map((member) => ({ ...member, enabled: false }));
    assert.deepEqual(await planner([...shared.members, ...dormant], "Anima", f.stateDirectory), { ok: true, count: 3 });
    // A legacy path-only member (what the old copy would have mounted) has no identity and is refused.
    assert.deepEqual(await planner([{ ...sharedMember("ashima_v4"), weight: "0.5" }], "Anima", f.stateDirectory), { ok: false, code: "lora-identity" });
  } finally { await f.close(); }
});

test("G02 every ready engine resolves its own legacy local group and refuses the others", galleryOptions, async () => {
  await mkdir(parent, { recursive: true });
  const dir = await mkdtemp(path.join(parent, "six-")), stateDirectory = path.join(dir, "state");
  await mkdir(stateDirectory, { recursive: true });
  const roots = READY_LORA_ENGINES.map((engine) => ({ engine, id: `local-${engine}`, directory: path.join(dir, "models", engine) }));
  for (const root of roots) for (const name of ["one", "two"]) await put(root.directory, `style/${root.engine}-${name}.safetensors`, `${root.engine} ${name}`);
  const service = new LoraRelinkService({ stateDirectory, gallery: galleryOwnerParticipant(stateDirectory, python), getRoots: async () => roots, admitIdle: async () => {}, validatePath: async () => {} });
  const scan = await service.scan({ engines: [...READY_LORA_ENGINES] });
  await service.commit({ scanId: scan.scanId, selections: [], registerBaseline: true, operationId: randomUUID() });
  const middleware = loraRelinkMiddleware({ service, admit: () => {}, readJson: (request) => new Promise((resolve, reject) => {
    const chunks = []; request.on("data", (chunk) => chunks.push(chunk)); request.on("end", () => { try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8"))); } catch (error) { reject(error); } });
  }) });
  const server = http.createServer((request, response) => middleware(request, response, () => { response.statusCode = 404; response.end("{}"); }));
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}`, realFetch = globalThis.fetch;
  globalThis.fetch = (url, init) => realFetch(typeof url === "string" && url.startsWith("/") ? base + url : url, init);
  try {
    assert.equal(READY_LORA_ENGINES.length, 7);
    for (const engine of READY_LORA_ENGINES) {
      const source = group(`g-${engine}`, ["one", "two"].map((name, index) => legacy(`style/${engine}-${name}.safetensors`, `${engine} ${name}`, { weight: index ? -0.25 : 0.8137, precision: index ? 2 : 4, enabled: index === 0 })));
      const own = projectResolvedGroup(source, await resolveGroupMembers(engine, source, { resolve: resolveLoraReferences }));
      assert.equal(own.ok, true, `${engine}: ${JSON.stringify(own.issues)}`);
      assert.deepEqual(own.group.members.map((item) => [item.weight, item.precision, item.enabled]), [[0.8137, 4, true], [-0.25, 2, false]]);
      const registry = await readLoraRegistry(stateDirectory);
      assert.ok(own.group.members.every((item) => registry.instances[item.instanceId].engine === engine));
      const other = READY_LORA_ENGINES.find((name) => name !== engine);
      assert.equal(projectResolvedGroup(source, await resolveGroupMembers(other, source, { resolve: resolveLoraReferences })).ok, false, `${engine} members are not ${other} members`);
    }
  } finally { globalThis.fetch = realFetch; await new Promise((resolve) => server.close(resolve)); await rm(dir, { recursive: true, force: true }); }
});
