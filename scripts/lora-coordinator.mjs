import { createHash, randomUUID } from "node:crypto";
import { mkdir, open, readFile, readdir, rename, rm, stat } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { assertRegistry, libraryError, libraryPaths } from "./lora-library.mjs";
import { CONTENT_HASH, INSTANCE_ID } from "../src/lora-identity.js";

export function documentDigest(value) { return createHash("sha256").update(JSON.stringify(value)).digest("hex"); }
const operationPattern = INSTANCE_ID;
function assertPublication(value) {
  if (value !== null && (!value || value.schemaVersion !== 1 || !Number.isSafeInteger(value.revision) || value.revision < 0
    || !["preparing", "published"].includes(value.phase) || (value.operationId !== undefined && !operationPattern.test(value.operationId)))) throw libraryError("Unsupported/damaged LoRA publication; original evidence retained", 503);
  return value;
}
function assertJournal(record) {
  if (!record || record.schemaVersion !== 1 || !operationPattern.test(record.operationId || "")
    || !CONTENT_HASH.test(record.requestDigest || "") || !Number.isSafeInteger(record.baseRevision) || record.baseRevision < 0
    || record.revision !== record.baseRevision + 1 || !["intent", "prepared", "applied", "published"].includes(record.phase)
    || !record.files || typeof record.files !== "object" || Array.isArray(record.files)
    || Object.keys(record.files).some((kind) => !["registry", "uiState", "cards"].includes(kind))
    || !Array.isArray(record.galleryPatches)) throw libraryError("Unsupported/damaged repair journal; original evidence retained", 503);
  return record;
}
export async function durableJson(file, value) {
  await mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.${randomUUID()}.tmp`;
  const handle = await open(temporary, "wx");
  try { await handle.writeFile(`${JSON.stringify(value, null, 2)}\n`, "utf8"); await handle.sync(); }
  finally { await handle.close(); }
  // On error keep the synced temporary as evidence, never remove a journal.
  await rename(temporary, file);
  // fsync of directory metadata is supported on POSIX, not Windows.
  if (process.platform !== "win32") {
    const directory = await open(path.dirname(file), "r");
    try { await directory.sync(); } finally { await directory.close(); }
  }
}
export async function readJsonIfPresent(file, fallback = null) {
  try { return JSON.parse(await readFile(file, "utf8")); }
  catch (error) { if (error.code === "ENOENT") return fallback; throw error; }
}
export function coordinationPaths(stateDirectory) {
  const { directory, operations } = libraryPaths(stateDirectory);
  return { directory, operations, lock: path.join(stateDirectory, ".lora-coordination.lock"), publication: path.join(directory, "publication.json") };
}

/** Common Node/Python exclusive lock. No timestamp/TTL-based lock theft. */
export async function withLibraryLease(stateDirectory, action) {
  const locations = coordinationPaths(stateDirectory);
  await mkdir(stateDirectory, { recursive: true });
  const deadline = Date.now() + 2000;
  for (;;) {
    try { await mkdir(locations.lock); break; }
    catch (error) {
      if (error.code !== "EEXIST") throw error;
      if (Date.now() >= deadline) throw libraryError("LoRA library is busy or requires abandoned-lock recovery");
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
  }
  const token = randomUUID();
  const identity = await stat(locations.lock, { bigint: true });
  try {
    await durableJson(path.join(locations.lock, "lease.json"), { token, pid: process.pid, host: os.hostname().toLowerCase(), owner: "node", createdAt: new Date().toISOString() });
    return await action(token);
  } finally {
    const now = await stat(locations.lock, { bigint: true }).catch(() => null);
    const lease = await readJsonIfPresent(path.join(locations.lock, "lease.json"));
    if (now && now.dev === identity.dev && now.ino === identity.ino && lease?.token === token) await rm(locations.lock, { recursive: true });
  }
}

/** Explicit startup recovery: only reclaim a recorded, provably dead local PID.
 * A live/reused PID or an incomplete lease is NOT stolen; retain evidence.
 * Reaper interruption is fail-closed for manual inspection, not TTL deletion.
 */
export async function recoverAbandonedLibraryLease(stateDirectory) {
  const locations = coordinationPaths(stateDirectory);
  const lease = await readJsonIfPresent(path.join(locations.lock, "lease.json"));
  if (!lease) return false;
  if (lease.host !== os.hostname().toLowerCase()) throw libraryError("Library lease is from an unknown host; manual inspection required", 503);
  if (!Number.isSafeInteger(lease.pid) || lease.pid < 1) throw libraryError("Invalid abandoned lease; manual inspection required", 503);
  try { process.kill(lease.pid, 0); throw libraryError("Library lease owner is still alive (or PID reused)"); }
  catch (error) { if (error.code !== "ESRCH") throw error; }
  const reaper = path.join(locations.directory, "recovery.lock");
  await mkdir(locations.directory, { recursive: true });
  try { await mkdir(reaper); }
  catch (error) { if (error.code === "EEXIST") throw libraryError("Library lease reaper requires inspection", 503); throw error; }
  try {
    const current = await readJsonIfPresent(path.join(locations.lock, "lease.json"));
    if (current?.token !== lease.token) throw libraryError("Abandoned lease changed during recovery");
    await mkdir(locations.operations, { recursive: true });
    await rename(locations.lock, path.join(locations.operations, `abandoned-${randomUUID()}`));
    return true;
  } finally { await rm(reaper, { recursive: true }); }
}

/** Fail closed for pending/corrupt publication; absent manifest preserves legacy read behavior. */
export async function publishedLibraryVersion(stateDirectory) {
  const locations = coordinationPaths(stateDirectory);
  const value = assertPublication(await readJsonIfPresent(locations.publication));
  for (const name of await readdir(locations.operations).catch((error) => { if (error.code === "ENOENT") return []; throw error; })) {
    if (!/^[0-9a-f-]{36}\.json$/.test(name)) continue;
    const record = assertJournal(await readJsonIfPresent(path.join(locations.operations, name)));
     if (record.phase !== "published") throw libraryError("Unpublished LoRA intent requires recovery", 503);
  }
  if (!value) return 0;
  if (value.schemaVersion !== 1 || !Number.isSafeInteger(value.revision) || value.revision < 0 || value.phase !== "published") throw libraryError("LoRA repair recovery barrier is closed", 503);
  return value.revision;
}

/** All authority access in the management core uses this gate, including stale saves. */
export async function withPublishedLibrary(stateDirectory, { expectedRevision, write = false } = {}, action) {
  return withLibraryLease(stateDirectory, async (token) => {
    const revision = await publishedLibraryVersion(stateDirectory);
    if (write && (expectedRevision ?? 0) !== revision) throw libraryError("Stale LoRA published revision; refetch before saving");
    return action({ revision, token });
  });
}

const authorityPaths = (stateDirectory) => ({
  registry: libraryPaths(stateDirectory).registry,
  uiState: path.join(stateDirectory, "ui-state.json"),
  cards: path.join(stateDirectory, "lora-cards", "cards.json"),
});

/**
 * One coordinator; Gallery is a transactional Python owner participant.
 * This core is intentionally NOT a public management route yet. The caller
 * must supply actual idle admission and fresh source/target validation under
 * the lease. No module import/constructor initializes or rewrites user state.
 * Recovery finishes a durable intent forward; undo uses owned typed patches,
 * not journal before-images. Ordinary readers cannot observe prepare/publish.
 */
export class LoraCoordinator {
  constructor({ stateDirectory, gallery, admitIdle, validate, fault = async () => {} }) {
    if (!stateDirectory || !gallery || typeof admitIdle !== "function" || typeof validate !== "function") throw new Error("Coordinator requires all owners and admission/validation gates");
    Object.assign(this, { stateDirectory, gallery, admitIdle, validate, fault });
  }
  async commit(request, { token: delegated } = {}) {
    if (!operationPattern.test(request.operationId || "")) throw libraryError("Invalid idempotent operation id", 400);
    if (!Object.keys(request.files || {}).length && !(request.galleryPatches || []).length) return { status: "no-op", operationId: null };
    const locations = coordinationPaths(this.stateDirectory);
    const apply = async (token) => {
      const journalPath = path.join(locations.operations, `${request.operationId}.json`);
      const old = await readJsonIfPresent(journalPath);
      const requestDigest = documentDigest({ expectedRevision: request.expectedRevision, files: request.files, galleryPatches: request.galleryPatches || [], validation: request.validation || null });
      if (old) {
        assertJournal(old);
        if (old.requestDigest !== requestDigest) throw libraryError("Operation id already belongs to a different selection");
        const publication = assertPublication(await readJsonIfPresent(locations.publication));
        if (old.phase === "published") return { status: "published", operationId: old.operationId, revision: old.revision };
        if (old.phase === "intent" && (!publication || (publication.phase === "published" && publication.revision === old.baseRevision))) {
          await durableJson(locations.publication, { schemaVersion: 1, revision: old.baseRevision, phase: "preparing", operationId: old.operationId });
        } else if (publication?.operationId !== old.operationId) throw libraryError("Operation recovery does not own the barrier", 503);
        return this.finish(old, journalPath, token);
      }
      const baseRevision = await publishedLibraryVersion(this.stateDirectory);
      if (request.expectedRevision !== baseRevision) throw libraryError("Stale repair baseline");
      await this.admitIdle(token);
      await this.validate(request, token);
      const files = {};
      for (const [kind, patch] of Object.entries(request.files || {})) {
        const paths = authorityPaths(this.stateDirectory);
        if (!Object.hasOwn(paths, kind)) throw libraryError("Unknown authority participant", 400);
        const file = paths[kind];
        const current = await readJsonIfPresent(file);
        if (documentDigest(current) !== patch.beforeDigest) throw libraryError(`Stale ${kind} consumer version`);
        if (!patch.after || typeof patch.after !== "object" || Array.isArray(patch.after)) throw libraryError("Invalid authority document", 400);
        if (kind === "registry") assertRegistry(patch.after);
        files[kind] = { beforeDigest: patch.beforeDigest, after: patch.after };
      }
      // Preview SQLite CAS before any intent/barrier. Prepare repeats the CAS.
      await this.gallery.check(request.galleryPatches || [], token);
      const journal = { schemaVersion: 1, operationId: request.operationId, requestDigest,
        baseRevision, revision: baseRevision + 1, phase: "intent", files,
        galleryPatches: request.galleryPatches || [], validation: request.validation || null };
      await durableJson(journalPath, journal);
      await this.fault("intent");
      await durableJson(locations.publication, { schemaVersion: 1, revision: baseRevision, phase: "preparing", operationId: journal.operationId });
      await this.fault("barrier");
      return this.finish(journal, journalPath, token);
    };
    return delegated ? apply(delegated) : withLibraryLease(this.stateDirectory, apply);
  }
  async finish(journal, journalPath, token) {
    assertJournal(journal);
    const locations = coordinationPaths(this.stateDirectory);
    // No target substitution into an existing job. Pending recovery waits for
    // actual safe idle, retaining the barrier and original request snapshots.
    await this.admitIdle(token);
    await this.validate({ validation: journal.validation, files: journal.files, galleryPatches: journal.galleryPatches, recovering: true }, token);
    await this.gallery.prepare(journal.operationId, journal.galleryPatches, token);
    journal.phase = "prepared";
    await durableJson(journalPath, journal);
    await this.fault("prepared");
    for (const [kind, patch] of Object.entries(journal.files)) {
      const file = authorityPaths(this.stateDirectory)[kind];
      const current = await readJsonIfPresent(file);
      if (documentDigest(current) !== documentDigest(patch.after)) {
        if (documentDigest(current) !== patch.beforeDigest) throw libraryError(`Recovery conflict in ${kind}; barrier retained`, 503);
        await durableJson(file, patch.after);
      }
      await this.fault(`file:${kind}`);
    }
    await this.gallery.publish(journal.operationId, token);
    journal.phase = "applied";
    await durableJson(journalPath, journal);
    await this.fault("gallery");
    await durableJson(locations.publication, { schemaVersion: 1, revision: journal.revision, phase: "published", operationId: journal.operationId });
    await this.fault("publication");
    journal.phase = "published";
    await durableJson(journalPath, journal);
    await this.fault("receipt");
    return { status: "published", operationId: journal.operationId, revision: journal.revision };
  }
  async recover() {
    await recoverAbandonedLibraryLease(this.stateDirectory);
    return withLibraryLease(this.stateDirectory, async (token) => {
      const locations = coordinationPaths(this.stateDirectory);
      const publication = assertPublication(await readJsonIfPresent(locations.publication));
      const pending = [];
      for (const name of await readdir(locations.operations).catch((error) => { if (error.code === "ENOENT") return []; throw error; })) {
        if (!/^[0-9a-f-]{36}\.json$/.test(name)) continue;
        const record = assertJournal(await readJsonIfPresent(path.join(locations.operations, name)));
        if (record.phase !== "published") pending.push(record);
      }
      if (!pending.length && !publication?.operationId) return { status: "no-recovery" };
      if (pending.length > 1) throw libraryError("Multiple pending repair intents; barrier retained", 503);
      const pendingIntent = pending[0];
      if (pendingIntent && (!publication || (publication.phase === "published" && publication.revision === pendingIntent.baseRevision))) {
        await durableJson(locations.publication, { schemaVersion: 1, revision: pendingIntent.baseRevision, phase: "preparing", operationId: pendingIntent.operationId });
        return this.finish(pendingIntent, path.join(locations.operations, `${pendingIntent.operationId}.json`), token);
      }
      const journalPath = path.join(locations.operations, `${publication.operationId}.json`);
      const journal = assertJournal(await readJsonIfPresent(journalPath));
      if (!journal || journal.schemaVersion !== 1 || journal.operationId !== publication.operationId) throw libraryError("Missing/corrupt recovery intent; barrier retained", 503);
      if (journal.phase === "published") {
        if (publication.phase !== "published" || publication.revision !== journal.revision) throw libraryError("Publication/receipt mismatch; original evidence retained", 503);
        return { status: "published", revision: journal.revision, operationId: journal.operationId };
      }
      return this.finish(journal, journalPath, token);
    });
  }
}
