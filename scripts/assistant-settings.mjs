import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import { normalizeAssistantSettings, redactAssistantSettings, validateAssistantSettings } from "../src/ai-assistant-providers.js";
import {
  activeAssistantProfile,
  normalizeAssistantProfileStore,
  redactAssistantProfileStore,
} from "../src/assistant-profiles.js";

// Persistence for the AI assistant's service configuration.
//
// The file holds a *set* of named profiles with one of them active (`src/assistant-profiles.js`),
// so switching between a cloud vendor and a local runtime is a selection rather than a retype. A
// v1 file — a bare settings object — reads as a single profile, which is the whole of the
// migration; it is rewritten in the new shape on the first save.
//
// The API keys live here and only here. They are written to the local state directory, read
// server-side when a request is proxied, and never returned to the browser: `redactAssistantSettings`
// replaces each with a presence flag and a four-character tail. Keeping the secrets out of the page
// means they also stay out of localStorage, out of the popped-out tab, and out of any error surface
// that echoes a request.

export const ASSISTANT_SETTINGS_FILENAME = "assistant-settings.json";

export function assistantSettingsPath(stateDirectory) {
  return path.join(stateDirectory, ASSISTANT_SETTINGS_FILENAME);
}

// A corrupt or absent file falls back to defaults rather than throwing: the settings page has to
// stay reachable, otherwise a bad write would lock the user out of fixing it. Normalisation is
// deterministic, so the ids handed out for an unwritten file are the same on every read and a save
// addressed to one of them still lands.
export async function readAssistantProfileStore(file) {
  try {
    const parsed = JSON.parse(await readFile(file, "utf8"));
    return normalizeAssistantProfileStore(parsed);
  } catch (error) {
    if (error.code !== "ENOENT") return { ...normalizeAssistantProfileStore(null), corrupt: true };
    return normalizeAssistantProfileStore(null);
  }
}

// The live configuration, in the exact shape every consumer already expects. Nothing downstream of
// "which settings are active" knows profiles exist.
export async function readAssistantSettings(file) {
  const store = await readAssistantProfileStore(file);
  const settings = activeAssistantProfile(store).settings;
  return store.corrupt === true ? { ...settings, corrupt: true } : settings;
}

async function writeJsonAtomically(stateDirectory, file, payload) {
  await mkdir(stateDirectory, { recursive: true });
  const temporaryPath = path.join(stateDirectory, `${randomUUID()}.tmp`);
  try {
    await writeFile(temporaryPath, `${JSON.stringify(payload, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    await rename(temporaryPath, file);
  } finally {
    await rm(temporaryPath, { force: true });
  }
  return payload;
}

// Normalises before writing, so a caller cannot persist a transient flag such as `corrupt`, an
// unknown key, or a store whose `activeId` points at nothing.
export async function writeAssistantProfileStore(stateDirectory, store) {
  const normalized = normalizeAssistantProfileStore(store);
  await writeJsonAtomically(stateDirectory, assistantSettingsPath(stateDirectory), normalized);
  return normalized;
}

// Replaces the *active* profile's settings and leaves every other profile untouched. Read-modify-
// write rather than a whole-file overwrite: saving the service you are using must not discard the
// other configurations stored beside it.
export async function writeAssistantSettings(stateDirectory, settings) {
  const file = assistantSettingsPath(stateDirectory);
  const store = normalizeAssistantProfileStore(await readAssistantProfileStore(file));
  const normalized = normalizeAssistantSettings(settings);
  const now = new Date().toISOString();
  await writeAssistantProfileStore(stateDirectory, {
    ...store,
    profiles: store.profiles.map((profile) => profile.id === store.activeId
      ? { ...profile, settings: normalized, updatedAt: now }
      : profile),
  });
  return normalized;
}

// PUT semantics for the secret, decided by presence rather than value:
//   key absent      -> keep whatever is stored (the form never re-sends a key it only shows masked)
//   key non-empty   -> replace
//   key empty string-> clear
//
// Returns the merged input *unnormalized*. Validation has to see what the user actually sent, so
// repairing the values here would hide exactly the mistakes the caller needs to report.
export function mergeAssistantSettings(stored, incoming) {
  const provided = incoming && typeof incoming === "object" && !Array.isArray(incoming) ? incoming : {};
  const carriesKey = Object.prototype.hasOwnProperty.call(provided, "apiKey");
  const apiKey = carriesKey ? (typeof provided.apiKey === "string" ? provided.apiKey.trim() : "") : (stored?.apiKey || "");
  return { ...provided, apiKey };
}

// Single gate every outbound request passes. Returning the reasons rather than a bare boolean lets
// the chat surface tell the user which field is wrong instead of failing with a generic error.
// `settings` is the normalized form to store, valid only when `ready` is true.
export function assistantReadiness(settings) {
  const { valid, errors, settings: normalized } = validateAssistantSettings(settings);
  return { ready: valid, errors, settings: normalized };
}

export { normalizeAssistantSettings, redactAssistantSettings, redactAssistantProfileStore, validateAssistantSettings };
