// Controlled local maintenance (P10.2 §8): a SCOPED reset of LoRA technical
// identity for an explicit manifest of engine/root/paths. Not an HTTP route and
// never a startup behavior. Business `value` paths, weights, order, notes,
// Prompt text, Gallery images and weights on disk are never changed.
//
//   node scripts/lora-maintenance-reset.mjs --manifest m.json [--state dir] [--apply] [--backup dir]
//   m.json: { "engine": "Anima", "rootId": "local-Anima", "values": ["style/a.safetensors", ...] }
import { createHash, randomUUID } from "node:crypto";
import { copyFile, mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { assertRegistry, libraryError, libraryPaths, validRelative } from "./lora-library.mjs";
import { LoraCoordinator, withLibraryLease, publishedLibraryVersion, readJsonIfPresent, documentDigest, coordinationPaths } from "./lora-coordinator.mjs";
import { galleryOwnerParticipant } from "./lora-gallery-owner.mjs";
import { READY_LORA_ENGINES } from "../src/lora-model-scope.js";

export const IDENTITY_FIELDS = ["identitySchema", "instanceId", "contentHash", "contentRevision", "originalReference"];

/** Pure plan. Instances whose current or historical path is in the manifest
 * (same engine + root) are removed with every rule touching them; consumers
 * keep their item but lose only the stale technical identity fields. */
export function planScopedReset({ registry, state, cards, galleryCards = [], manifest, operations = [], operationId, at }) {
  assertRegistry(registry);
  const { engine, rootId, values } = manifest || {};
  if (!READY_LORA_ENGINES.includes(engine) || typeof rootId !== "string" || !Array.isArray(values) || !values.length || !values.every(validRelative)) throw libraryError("Invalid reset manifest", 400);
  const valueSet = new Set(values);
  const affected = new Set(Object.values(registry.instances).filter((item) => item.engine === engine && item.rootId === rootId
    && (valueSet.has(item.value) || item.history.some((value) => valueSet.has(value)))).map((item) => item.instanceId));
  const next = structuredClone(registry);
  const removedRules = next.substitutions.filter((rule) => affected.has(rule.sourceId) || affected.has(rule.targetId));
  next.substitutions = next.substitutions.filter((rule) => !removedRules.includes(rule));
  const removedInstances = [...affected].map((id) => ({ instanceId: id, value: registry.instances[id].value, status: registry.instances[id].status, contentHash: registry.instances[id].contentHash }));
  for (const id of affected) delete next.instances[id];
  const touched = (item) => item && typeof item.value === "string" && ((item.instanceId && affected.has(item.instanceId))
    || (valueSet.has(item.value) && IDENTITY_FIELDS.some((key) => Object.hasOwn(item, key))));
  const strip = (item) => Object.fromEntries(Object.entries(item).filter(([key]) => !IDENTITY_FIELDS.includes(key)));
  const items = [];
  const clean = (list, address) => (list || []).map((item, index) => {
    if (!touched(item)) return item;
    items.push({ address: [...address, index], value: item.value, removedFields: IDENTITY_FIELDS.filter((key) => Object.hasOwn(item, key)) });
    return strip(item);
  });
  const nextState = structuredClone(state || {});
  const ws = nextState.workspace || {};
  if (ws.mountedLorasByEngine?.byEngine?.[engine]) ws.mountedLorasByEngine.byEngine[engine] = clean(ws.mountedLorasByEngine.byEngine[engine], ["workspace", "mountedLorasByEngine", "byEngine", engine]);
  if (ws.model === engine && Array.isArray(ws.loras)) ws.loras = clean(ws.loras, ["workspace", "loras"]);
  (ws.loraGroupsByEngine?.byEngine?.[engine] || []).forEach((group, index) => { group.members = clean(group.members, ["workspace", "loraGroupsByEngine", "byEngine", engine, index, "members"]); });
  const nextCards = cards ? structuredClone(cards) : cards, cardEntries = [];
  const scope = nextCards?.byEngine?.[engine];
  if (scope?.loras) for (const [key, entry] of Object.entries(scope.loras)) {
    if (!touched({ ...entry, value: key })) continue;
    cardEntries.push({ key, removedFields: IDENTITY_FIELDS.filter((field) => Object.hasOwn(entry, field)) });
    scope.loras[key] = strip(entry);
  }
  if (scope?.byInstance) for (const id of Object.keys(scope.byInstance)) if (affected.has(id)) { delete scope.byInstance[id]; cardEntries.push({ byInstance: id, removed: true }); }
  const galleryPatches = [];
  for (const card of galleryCards) {
    if (card.settings?.model !== engine || !(card.settings.loras || []).some(touched)) continue;
    galleryPatches.push({ cardId: card.id, before: card.settings, after: { ...card.settings, loras: card.settings.loras.map((item) => touched(item) ? strip(item) : item) } });
  }
  const ids = [...affected];
  const supersedes = operations.filter((operation) => ids.some((id) => JSON.stringify(operation).includes(id))).map((operation) => operation.operationId);
  if (affected.size || removedRules.length) {
    next.revision += 1;
    next.maintenance = [...(registry.maintenance || []), { operationId, at, engine, rootId, values, supersedes }];
  }
  return { registry: next, state: nextState, cards: nextCards, galleryPatches,
    changed: { registry: Boolean(affected.size || removedRules.length), uiState: items.length > 0, cards: cardEntries.length > 0 },
    report: { manifest: { engine, rootId, values }, removedInstances, removedRules: removedRules.map((rule) => ({ sourceId: rule.sourceId, targetId: rule.targetId, sourceHash: rule.sourceHash, kind: rule.kind })),
      uiStateItems: items, cardEntries, galleryCards: galleryPatches.map((patch) => ({ cardId: patch.cardId, items: patch.before.loras.filter(touched).map((item) => item.value) })), supersedes } };
}

const sha256 = async (file) => createHash("sha256").update(await readFile(file)).digest("hex");
function argsOf(argv) {
  const out = { apply: false };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--apply") out.apply = true;
    else if (key.startsWith("--")) out[key.slice(2)] = argv[++index];
  }
  return out;
}

async function main() {
  const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const args = argsOf(process.argv.slice(2));
  const stateDirectory = path.resolve(projectRoot, args.state || process.env.XIRAI_STATE_DIR || "state-cache");
  const manifest = JSON.parse(await readFile(path.resolve(args.manifest), "utf8"));
  const python = args.python || path.join(projectRoot, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
  const inference = args.inference || `http://127.0.0.1:${process.env.INFERENCE_PORT || 8718}`;
  const backup = path.resolve(args.backup || path.join(projectRoot, "..", "Temp", `lora-reset-${new Date().toISOString().replace(/[:.]/g, "-")}`));
  const gallery = galleryOwnerParticipant(stateDirectory, python);
  const operationId = randomUUID(), at = new Date().toISOString();
  const admitIdle = async (leaseToken) => {
    let response;
    try { response = await fetch(`${inference}/api/inference/lora-library/admission`, { method: "POST", headers: { "Content-Type": "application/json", Origin: inference }, body: JSON.stringify({ leaseToken }), signal: AbortSignal.timeout(5000) }); }
    catch (error) {
      // No inference process listening means no job can be active; anything else (timeout, reset) is refused.
      const refused = (cause, depth = 0) => Boolean(cause) && depth < 5 && (cause.code === "ECONNREFUSED" || refused(cause.cause, depth + 1) || (cause.errors || []).some((item) => refused(item, depth + 1)));
      if (refused(error)) return;
      throw libraryError("Inference owner did not answer admission; not resetting", 503);
    }
    if (!response.ok) throw libraryError(response.status === 409 ? "Active, queued or recovering generation; reset postponed" : `Inference owner refused admission (${response.status}); is it using this state directory?`, response.status);
  };
  const coordinator = new LoraCoordinator({ stateDirectory, gallery, admitIdle, validate: async (request) => {
    if (request.validation?.type !== "maintenance-reset") throw libraryError("Unexpected validation in maintenance tool", 503);
  } });
  await mkdir(backup, { recursive: true });
  const result = await withLibraryLease(stateDirectory, async (token) => {
    const revision = await publishedLibraryVersion(stateDirectory); // refuses unpublished intents / recovery pending
    const paths = { registry: libraryPaths(stateDirectory).registry, uiState: path.join(stateDirectory, "ui-state.json"), cards: path.join(stateDirectory, "lora-cards", "cards.json") };
    const registry = await readJsonIfPresent(paths.registry), state = await readJsonIfPresent(paths.uiState), cards = await readJsonIfPresent(paths.cards);
    if (!registry) throw libraryError("No LoRA identity registry; nothing to reset", 409);
    const operations = [];
    for (const name of await readdir(coordinationPaths(stateDirectory).operations).catch(() => [])) if (/^[0-9a-f-]{36}\.json$/.test(name)) operations.push(await readJsonIfPresent(path.join(coordinationPaths(stateDirectory).operations, name)));
    const galleryCards = (await gallery.list(token)).cards || [];
    const plan = planScopedReset({ registry, state, cards, galleryCards, manifest, operations, operationId, at });
    await writeFile(path.join(backup, args.apply ? "reset-plan.json" : "dry-run.json"), JSON.stringify({ stateDirectory, revision, operationId, at, ...plan.report }, null, 2));
    if (!args.apply) return { dryRun: true, revision, report: plan.report };
    // Backups (state files, journals, consistent SQLite copy) with a hash manifest.
    const copies = [];
    const keep = async (source, name) => { try { await copyFile(source, path.join(backup, name)); copies.push({ name, sha256: await sha256(path.join(backup, name)) }); } catch (error) { if (error.code !== "ENOENT") throw error; } };
    await keep(paths.registry, "registry.json"); await keep(coordinationPaths(stateDirectory).publication, "publication.json");
    await keep(paths.uiState, "ui-state.json"); await keep(paths.cards, "cards.json");
    await mkdir(path.join(backup, "operations"), { recursive: true });
    for (const operation of operations) await keep(path.join(coordinationPaths(stateDirectory).operations, `${operation.operationId}.json`), `operations/${operation.operationId}.json`);
    const snapshot = spawnSync(python, ["-B", "-c", "import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); d=sqlite3.connect(sys.argv[2]); s.backup(d); d.close(); s.close()", path.join(stateDirectory, "gallery.sqlite3"), path.join(backup, "gallery.sqlite3")], { encoding: "utf8", windowsHide: true });
    if (snapshot.status !== 0) throw libraryError(`Gallery snapshot failed: ${snapshot.stderr.slice(-300)}`, 503);
    copies.push({ name: "gallery.sqlite3", sha256: await sha256(path.join(backup, "gallery.sqlite3")) });
    await writeFile(path.join(backup, "backup-manifest.json"), JSON.stringify({ stateDirectory, revision, at, copies }, null, 2));
    const files = {};
    if (plan.changed.registry) files.registry = { beforeDigest: documentDigest(registry), after: plan.registry };
    if (plan.changed.uiState) files.uiState = { beforeDigest: documentDigest(state), after: plan.state };
    if (plan.changed.cards) files.cards = { beforeDigest: documentDigest(cards), after: plan.cards };
    const receipt = await coordinator.commit({ operationId, expectedRevision: revision, files, galleryPatches: plan.galleryPatches,
      validation: { type: "maintenance-reset", scope: plan.report.manifest, supersedes: plan.report.supersedes } }, { token });
    return { revision, receipt, report: plan.report };
  });
  await writeFile(path.join(backup, "result.json"), JSON.stringify(result, null, 2));
  console.log(JSON.stringify({ backup, ...result, report: undefined, summary: { removedInstances: result.report.removedInstances.length, removedRules: result.report.removedRules.length,
    uiStateItems: result.report.uiStateItems.length, cardEntries: result.report.cardEntries.length, galleryCards: result.report.galleryCards.length, supersedes: result.report.supersedes } }, null, 2));
}
if (import.meta.url === pathToFileURL(process.argv[1] || "").href) main().catch((error) => { console.error(error.message); process.exit(1); });
