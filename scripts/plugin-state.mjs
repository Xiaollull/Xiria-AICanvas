import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { validPluginId } from "./plugin-registry.mjs";

// Persisted enabled-state for the plugin port.
//
// `enabled` is a user preference, not an execution grant. Plugin code still never runs, so turning a
// plugin on records intent and nothing more. The store is opt-in — an id absent from the file is
// disabled — so a plugin dropped on disk is never live by default, which is the behaviour a future
// sandboxed execution host has to inherit.
//
// Discovery stays read-only: every write lives in this module.

export const PLUGIN_STATE_SCHEMA_VERSION = 1;
export const PLUGIN_STATE_FILENAME = "plugins.json";
export const MAXIMUM_ENABLED_PLUGINS = 512;

export function pluginStatePathFor(stateDirectory) {
  return path.join(stateDirectory, PLUGIN_STATE_FILENAME);
}

/**
 * Validate a persisted document. Returns a sorted, de-duplicated id list, or `null` when the
 * document is malformed or written by a newer schema. `null` always means "fail closed": treat
 * every plugin as disabled and refuse to overwrite whatever is on disk.
 */
export function normalizePluginState(document) {
  if (!document || typeof document !== "object" || Array.isArray(document)) return null;
  const keys = Object.keys(document);
  if (keys.some((key) => key !== "schemaVersion" && key !== "enabled")) return null;
  if (document.schemaVersion !== PLUGIN_STATE_SCHEMA_VERSION) return null;
  if (!Array.isArray(document.enabled) || document.enabled.length > MAXIMUM_ENABLED_PLUGINS) return null;
  if (!document.enabled.every((id) => validPluginId(id))) return null;
  return [...new Set(document.enabled)].sort();
}

export function serializePluginState(enabledIds) {
  return { schemaVersion: PLUGIN_STATE_SCHEMA_VERSION, enabled: [...new Set(enabledIds)].sort() };
}

/**
 * Apply one toggle to an id list. Pure, so admission can be unit-tested without touching disk.
 */
export function applyPluginEnabled(enabledIds, id, enabled) {
  const next = new Set(enabledIds);
  if (enabled) next.add(id);
  else next.delete(id);
  return [...next].sort();
}

/**
 * Read the enabled set.
 *
 * Never throws. A missing file is the normal first-run state. A malformed or future-schema file
 * reports `readable: false`, which disables every plugin and blocks writes so the user's data is
 * preserved rather than silently rewritten.
 */
export async function readPluginState({ statePath } = {}) {
  let contents;
  try {
    contents = await readFile(statePath, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return { enabled: [], readable: true, present: false };
    return { enabled: [], readable: false, present: true };
  }
  let document;
  try {
    document = JSON.parse(contents);
  } catch {
    return { enabled: [], readable: false, present: true };
  }
  const enabled = normalizePluginState(document);
  if (!enabled) return { enabled: [], readable: false, present: true };
  return { enabled, readable: true, present: true };
}

/**
 * Write the enabled set atomically, mirroring the UI-state writer: exclusive temporary file then
 * rename, with the temporary file always removed.
 */
export async function writePluginState({ stateDirectory, statePath, enabled } = {}) {
  const document = serializePluginState(enabled);
  if (document.enabled.length > MAXIMUM_ENABLED_PLUGINS) {
    throw Object.assign(new Error("Too many enabled plugins"), { statusCode: 400, code: "too_many_enabled_plugins" });
  }
  await mkdir(stateDirectory, { recursive: true });
  const temporaryPath = path.join(stateDirectory, `${randomUUID()}.tmp`);
  try {
    await writeFile(temporaryPath, `${JSON.stringify(document, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    await rename(temporaryPath, statePath);
  } finally {
    await rm(temporaryPath, { force: true });
  }
  return document.enabled;
}

/**
 * Decide whether a toggle may be applied, given the plugin's current registry entry.
 *
 * Enabling is admitted only for a healthy `discovered` plugin: an invalid, incompatible, or blocked
 * manifest can never be turned on. Disabling is always admitted, so a plugin that became unhealthy
 * while enabled can still be switched off.
 */
export function pluginToggleAdmission(entry, enabled) {
  if (!entry) return { allowed: false, statusCode: 404, code: "plugin_not_found", message: "Plugin was not found" };
  if (!enabled) return { allowed: true };
  if (entry.state === "blocked") {
    return { allowed: false, statusCode: 409, code: "permissions_not_supported", message: "This plugin declares permissions, which are not supported" };
  }
  if (entry.state !== "discovered") {
    return { allowed: false, statusCode: 409, code: "plugin_not_enableable", message: "Only a valid, compatible plugin can be enabled" };
  }
  return { allowed: true };
}
