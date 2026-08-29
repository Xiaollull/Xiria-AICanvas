// Named configuration profiles for the AI assistant.
//
// The assistant used to hold exactly one service configuration, which meant switching between a
// paid cloud model and a local Ollama — or between two keys for the same vendor — was a matter of
// retyping four fields and pasting a secret back in. A profile is that same configuration with an
// identity: it can be selected, renamed, duplicated and deleted, and exactly one of them is active
// at a time.
//
// The settings object inside a profile is byte-for-byte the shape this file's callers already use
// (`normalizeAssistantSettings` / `validateAssistantSettings` / `redactAssistantSettings` apply to
// it unchanged), so nothing downstream of "which settings are live" had to learn about profiles.
// That is also what makes the migration a pure wrap: a stored v1 file *is* a profile's settings.

import {
  DEFAULT_PROVIDER_ID,
  defaultAssistantSettings,
  normalizeAssistantSettings,
  providerProfile,
  redactAssistantSettings,
} from "./ai-assistant-providers.js";

export const ASSISTANT_PROFILES_SCHEMA_VERSION = 2;

// A ceiling rather than a design constraint: the picker is a list, not a grid, and a hundred
// near-identical rows would be worse than none. Twelve covers "every vendor I have a key for".
export const MAXIMUM_PROFILES = 12;
export const MAXIMUM_PROFILE_NAME = 40;

// Ids are generated where a profile is created and appear in a URL path, so the accepted shape is
// deliberately narrow rather than "whatever the client sent". Same pattern as session ids.
export const PROFILE_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

export function validProfileId(value) {
  return typeof value === "string" && PROFILE_ID_PATTERN.test(value);
}

// Only ever a local identifier — never a token, never a filename — so the fallback exists to keep
// this module usable wherever `crypto` is not exposed, not to provide unpredictability.
export function generateProfileId() {
  const uuid = globalThis.crypto?.randomUUID;
  if (typeof uuid === "function") return globalThis.crypto.randomUUID();
  const hex = (length) => Array.from({ length }, () => Math.floor(Math.random() * 16).toString(16)).join("");
  return `${hex(8)}-${hex(4)}-4${hex(3)}-${(8 + Math.floor(Math.random() * 4)).toString(16)}${hex(3)}-${hex(12)}`;
}

function isoTimestamp(value, fallback) {
  if (typeof value === "string") {
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) return new Date(parsed).toISOString();
  }
  return fallback;
}

// Vendor label plus model, because that is how a user thinks about which configuration is which.
// Falls back to the vendor alone before a model has been chosen.
export function defaultProfileName(settings) {
  const provider = providerProfile(settings?.provider) || providerProfile(DEFAULT_PROVIDER_ID);
  const model = typeof settings?.model === "string" ? settings.model.trim() : "";
  return (model ? `${provider.label} · ${model}` : provider.label).slice(0, MAXIMUM_PROFILE_NAME);
}

// Suffixes rather than refusing: a duplicate name is a naming inconvenience, not a data problem,
// and blocking the create would be a worse answer than "DeepSeek · x 2".
export function uniqueProfileName(name, taken) {
  const existing = new Set((taken || []).map((entry) => String(entry).trim()));
  const base = String(name || "").trim().slice(0, MAXIMUM_PROFILE_NAME) || "未命名配置";
  if (!existing.has(base)) return base;
  for (let index = 2; index <= MAXIMUM_PROFILES + 1; index += 1) {
    const suffix = ` ${index}`;
    const candidate = `${base.slice(0, MAXIMUM_PROFILE_NAME - suffix.length)}${suffix}`;
    if (!existing.has(candidate)) return candidate;
  }
  return base;
}

export function normalizeProfileName(value, settings) {
  const raw = typeof value === "string" ? value.trim().replace(/\s+/g, " ") : "";
  return raw ? raw.slice(0, MAXIMUM_PROFILE_NAME) : defaultProfileName(settings);
}

export function normalizeAssistantProfile(value, { id, now } = {}) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  // A v1 file is a bare settings object, so a profile that carries no nested `settings` is read as
  // one. This is the whole of the migration.
  const settings = normalizeAssistantSettings(
    source.settings && typeof source.settings === "object" ? source.settings : source,
  );
  const createdAt = isoTimestamp(source.createdAt, now || new Date().toISOString());
  return {
    id: validProfileId(source.id) ? source.id : (validProfileId(id) ? id : generateProfileId()),
    name: normalizeProfileName(source.name, settings),
    createdAt,
    updatedAt: isoTimestamp(source.updatedAt, createdAt),
    settings,
  };
}

// Reading must be *stable*, not merely non-throwing. A v1 file, a hand-edited one, or no file at
// all still has to answer "which profile is this?" with the same id on every read — the settings
// page lists profiles and then edits one by id, and an id minted freshly per read would 404 the
// save. So repairs during normalisation draw from a deterministic sequence; only an explicit
// create mints a random id, and that path writes the file.
function placeholderProfileId(index) {
  return `00000000-0000-4000-8000-${String(index).padStart(12, "0")}`;
}

export function emptyAssistantProfileStore({ now } = {}) {
  const stamp = now || new Date().toISOString();
  const profile = normalizeAssistantProfile(
    { settings: defaultAssistantSettings() },
    { id: placeholderProfileId(0), now: stamp },
  );
  return { schemaVersion: ASSISTANT_PROFILES_SCHEMA_VERSION, activeId: profile.id, profiles: [profile] };
}

// Never throws, and never yields an empty list. A store with no profiles would leave the chat
// surface with nothing to read and the settings page with nothing to edit, so a damaged file
// degrades to one default profile rather than to a dead end.
export function normalizeAssistantProfileStore(value, { now } = {}) {
  const stamp = now || new Date().toISOString();
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  // A v1 file is a bare settings object with no `profiles` array, and reads as a single profile.
  const rows = Array.isArray(source.profiles) ? source.profiles : [source];
  const seen = new Set();
  const profiles = [];
  let placeholder = 0;
  const nextPlaceholder = () => {
    let candidate = placeholderProfileId(placeholder);
    while (seen.has(candidate)) candidate = placeholderProfileId((placeholder += 1));
    placeholder += 1;
    return candidate;
  };
  for (const row of rows) {
    if (profiles.length >= MAXIMUM_PROFILES) break;
    const stated = validProfileId(row?.id) ? row.id : "";
    const profile = normalizeAssistantProfile(row, { id: stated || nextPlaceholder(), now: stamp });
    // A duplicated id would make "edit this one" ambiguous, so the later row is re-identified
    // rather than dropped: it still holds a configuration the user wrote.
    if (seen.has(profile.id)) profile.id = nextPlaceholder();
    seen.add(profile.id);
    profiles.push(profile);
  }
  if (!profiles.length) return emptyAssistantProfileStore({ now: stamp });
  const activeId = validProfileId(source.activeId) && seen.has(source.activeId) ? source.activeId : profiles[0].id;
  return { schemaVersion: ASSISTANT_PROFILES_SCHEMA_VERSION, activeId, profiles };
}

export function assistantProfileAt(store, id) {
  return normalizeAssistantProfileStore(store).profiles.find((profile) => profile.id === id) || null;
}

// The one function the rest of the app needs: which configuration is live. Everything that used to
// read the single settings file reads this instead.
export function activeAssistantProfile(store) {
  const normalized = normalizeAssistantProfileStore(store);
  return normalized.profiles.find((profile) => profile.id === normalized.activeId) || normalized.profiles[0];
}

export function canRemoveAssistantProfile(store) {
  return normalizeAssistantProfileStore(store).profiles.length > 1;
}

export function canCreateAssistantProfile(store) {
  return normalizeAssistantProfileStore(store).profiles.length < MAXIMUM_PROFILES;
}

// A new profile is created active. Creating one and then having to select it separately reads as a
// failed create, and the user's next act is always to configure the thing they just made.
export function createAssistantProfile(store, seed, { id, now } = {}) {
  const stamp = now || new Date().toISOString();
  const normalized = normalizeAssistantProfileStore(store, { now: stamp });
  if (normalized.profiles.length >= MAXIMUM_PROFILES) return normalized;
  const source = seed && typeof seed === "object" ? seed : {};
  const settings = normalizeAssistantSettings(source.settings || source);
  const profile = normalizeAssistantProfile({
    id: validProfileId(id) ? id : generateProfileId(),
    name: uniqueProfileName(
      normalizeProfileName(source.name, settings),
      normalized.profiles.map((entry) => entry.name),
    ),
    createdAt: stamp,
    updatedAt: stamp,
    settings,
  }, { now: stamp });
  return { ...normalized, activeId: profile.id, profiles: [...normalized.profiles, profile] };
}

// `patch` carries only what changed: a name-only rename must not reset the settings, and a
// settings-only save must not rename the profile.
export function updateAssistantProfile(store, id, patch, { now } = {}) {
  const stamp = now || new Date().toISOString();
  const normalized = normalizeAssistantProfileStore(store, { now: stamp });
  const changes = patch && typeof patch === "object" ? patch : {};
  return {
    ...normalized,
    profiles: normalized.profiles.map((profile) => profile.id !== id ? profile : normalizeAssistantProfile({
      ...profile,
      ...(Object.prototype.hasOwnProperty.call(changes, "name") ? { name: normalizeProfileName(changes.name, profile.settings) } : {}),
      ...(changes.settings ? { settings: normalizeAssistantSettings(changes.settings) } : {}),
      updatedAt: stamp,
    }, { now: stamp })),
  };
}

// Removing the active profile promotes its neighbour rather than leaving nothing selected, and
// prefers the one before it so deleting down a list does not walk the selection to the end.
export function removeAssistantProfile(store, id) {
  const normalized = normalizeAssistantProfileStore(store);
  const index = normalized.profiles.findIndex((profile) => profile.id === id);
  if (index < 0 || normalized.profiles.length <= 1) return normalized;
  const profiles = normalized.profiles.filter((profile) => profile.id !== id);
  const activeId = normalized.activeId === id
    ? profiles[Math.max(0, index - 1)].id
    : normalized.activeId;
  return { ...normalized, activeId, profiles };
}

export function activateAssistantProfile(store, id) {
  const normalized = normalizeAssistantProfileStore(store);
  if (!normalized.profiles.some((profile) => profile.id === id)) return normalized;
  return { ...normalized, activeId: id };
}

// Copies the secret too. The point of duplicating is "same service, different model", and a copy
// that silently dropped the key would look broken in a way the form cannot explain.
export function duplicateAssistantProfile(store, id, { id: newId, now } = {}) {
  const stamp = now || new Date().toISOString();
  const normalized = normalizeAssistantProfileStore(store, { now: stamp });
  const source = normalized.profiles.find((profile) => profile.id === id);
  if (!source || normalized.profiles.length >= MAXIMUM_PROFILES) return normalized;
  return createAssistantProfile(normalized, { name: source.name, settings: source.settings }, { id: newId, now: stamp });
}

// What the control plane may hand back. `settings` goes through the same redaction as before, so a
// profile list can never become a second way to read the key out of the state directory.
export function redactAssistantProfile(profile, activeId) {
  const normalized = normalizeAssistantProfile(profile);
  return {
    id: normalized.id,
    name: normalized.name,
    createdAt: normalized.createdAt,
    updatedAt: normalized.updatedAt,
    active: normalized.id === activeId,
    settings: redactAssistantSettings(normalized.settings),
  };
}

export function redactAssistantProfileStore(store) {
  const normalized = normalizeAssistantProfileStore(store);
  return {
    schemaVersion: normalized.schemaVersion,
    activeId: normalized.activeId,
    profiles: normalized.profiles.map((profile) => redactAssistantProfile(profile, normalized.activeId)),
  };
}
