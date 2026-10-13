import path from "node:path";
import { documentDigest, durableJson, readJsonIfPresent, publishedLibraryVersion, withLibraryLease } from "./lora-coordinator.mjs";
import { libraryError } from "./lora-library.mjs";
import { normalizeMountedLoraMap, READY_LORA_ENGINES } from "../src/lora-model-scope.js";
import { normalizeLoraGroupMap, LORA_GROUPS_SCHEMA_VERSION } from "../src/lora-groups.js";
import { normalizeLoraCardStore, emptyLoraCardStore } from "../src/lora-cards.js";
import { inspectMountedLoras } from "../src/lora-state.js";
import { assertUiStateValue } from "../src/ui-state-authority.js";

const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const object = (a) => a && typeof a === "object" && !Array.isArray(a);
export function assertWorkspaceIdentity(value) {
  if (value != null && !object(value)) throw libraryError("Unreadable workspace document; original state retained");
  const ws = value?.workspace || {};
  if (inspectMountedLoras(ws.loras).fatal) throw libraryError("Unrecognized mirror identity; original state retained");
  if (Object.values(ws.mountedLorasByEngine?.byModel || {}).some((list) => inspectMountedLoras(list).fatal)) throw libraryError("Unrecognized legacy map identity; original state retained");
  if (ws.mountedLorasByEngine && normalizeMountedLoraMap(ws.mountedLorasByEngine).fatal) throw libraryError("Unrecognized mounted identity; original state retained");
  const groups = ws.loraGroupsByEngine;
  if (groups != null && (!object(groups) || !Number.isInteger(groups.schemaVersion) || groups.schemaVersion < 1 || groups.schemaVersion > LORA_GROUPS_SCHEMA_VERSION || !object(groups.byEngine)
    || Object.entries(groups.byEngine).some(([engine, list]) => !READY_LORA_ENGINES.includes(engine) || !Array.isArray(list) || list.some((group) => !object(group) || (group.members !== undefined && !Array.isArray(group.members)))))) throw libraryError("Unrecognized group container; original state retained");
  if (normalizeLoraGroupMap(ws.loraGroupsByEngine).fatal) throw libraryError("Unrecognized group identity; original state retained");
}
/** Three-way edits: unchanged client fields retain the latest server value.
 * Arrays are ordered units, never merged/deduplicated/weight-summed.
 */
export function mergePublishedEdits(base, incoming, current, address = [], conflicts = []) {
  if (equal(incoming, base)) return structuredClone(current);
  if (equal(current, base) || equal(current, incoming)) return structuredClone(incoming);
  if (object(base) && object(incoming) && object(current)) {
    const next = {};
    for (const key of new Set([...Object.keys(base), ...Object.keys(incoming), ...Object.keys(current)])) {
      if (["__proto__", "constructor", "prototype"].includes(key)) throw libraryError("Invalid state key", 400);
      const value = mergePublishedEdits(base[key], incoming[key], current[key], [...address, key], conflicts);
      if (value !== undefined) next[key] = value;
    }
    return next;
  }
  conflicts.push(address);
  return structuredClone(current);
}

export class LoraPublishedStore {
  constructor({ stateDirectory, ensureRecovered = async () => {} }) {
    Object.assign(this, { stateDirectory, ensureRecovered });
    this.snapshots = new Map();
  }
  file(kind) {
    if (!new Set(["uiState", "cards"]).has(kind)) throw libraryError("Invalid document owner", 400);
    return path.join(this.stateDirectory, kind === "cards" ? "lora-cards/cards.json" : "ui-state.json");
  }
  remember(kind, value) {
    const version = documentDigest(value);
    if (this.snapshots.size >= 128) this.snapshots.delete(this.snapshots.keys().next().value);
    this.snapshots.set(`${kind}:${version}`, structuredClone(value));
    return version;
  }
  async access(action) {
    await this.ensureRecovered();
    return withLibraryLease(this.stateDirectory, async (token) => action({ token, revision: await publishedLibraryVersion(this.stateDirectory) }));
  }
  async read(kind) {
    return this.access(async ({ revision }) => {
      // Only ENOENT yields this private sentinel; stored null/bad JSON/EACCES
      // must not masquerade as an absent workspace eligible for initialization.
      const absent = Symbol("absent");
      const stored = await readJsonIfPresent(this.file(kind), absent);
      const exists = stored !== absent;
      const value = exists ? stored : kind === "cards" ? emptyLoraCardStore() : null;
      if (kind === "uiState" && exists) assertUiStateValue(value);
      return { value, exists, revision, version: this.remember(kind, value) };
    });
  }
  async save(kind, incoming, { revision = 0, version } = {}, afterSave = async () => {}) {
    return this.access(async (gate) => {
      if (Number(revision) !== gate.revision) throw libraryError("Published LoRA revision changed; refetch before saving");
      const absent = Symbol("absent");
      const stored = await readJsonIfPresent(this.file(kind), kind === "cards" ? emptyLoraCardStore() : absent);
      const current = stored === absent ? null : stored;
      if (kind === "uiState" && stored !== absent) assertUiStateValue(current);
      if (!object(incoming)) throw libraryError("Saved document must be an object", 400);
      let next = incoming;
      if (version && version !== documentDigest(current)) {
        const base = this.snapshots.get(`${kind}:${version}`);
        if (base === undefined) throw libraryError("Expired document version; refetch before saving");
        const conflicts = [];
        next = mergePublishedEdits(base, incoming, current, [], conflicts);
        if (conflicts.length) throw Object.assign(libraryError("Concurrent document fields were edited; refetch and review"), { conflicts });
      } else if (!version && gate.revision) throw libraryError("Versioned save required after LoRA publication");
      if (kind === "uiState") {
        assertUiStateValue(next);
        for (const value of [current, next]) {
          assertWorkspaceIdentity(value);
        }
      } else {
        if (normalizeLoraCardStore(current).fatal || normalizeLoraCardStore(next).fatal || normalizeLoraCardStore(current).reset || normalizeLoraCardStore(next).reset) throw libraryError("Unrecognized card identity; original data retained");
        next = normalizeLoraCardStore(next).store;
      }
      await durableJson(this.file(kind), next);
      await afterSave(next, gate); // Prune/asset ownership stays under the same lease.
      return { value: next, revision: gate.revision, version: this.remember(kind, next) };
    });
  }
}
