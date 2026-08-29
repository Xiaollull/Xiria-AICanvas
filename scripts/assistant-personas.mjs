import { randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  MAXIMUM_SYSTEM_PROMPT_CHARACTERS,
  PERSONA_SCHEMA_VERSION,
  normalizeAssistantPersona,
  personaSections,
  sortAssistantPersonas,
  uniquePersonaName,
  validUserPersonaId,
} from "../src/assistant-persona.js";

// Character storage for the AI assistant.
//
// Characters come from two places and the difference is deliberate:
//
//   assistant/personas/*.json          shipped with the app, read-only, id is the filename
//   state-cache/assistant-personas/    written by the character interface, one uuid.json each
//
// Keeping the two apart is what makes both halves safe. An app update may replace the shipped
// directory wholesale without touching anything the user wrote, and the interface never has to
// edit a file that the next update would overwrite. A built-in is duplicated, not edited.
//
// Characters are data, never code. Nothing here imports, evaluates, or spawns a character file,
// and the composed system prompt is the only part that ever reaches a provider.

export { PERSONA_SCHEMA_VERSION, MAXIMUM_SYSTEM_PROMPT_CHARACTERS };

export const PERSONA_DIRECTORY_NAME = path.join("assistant", "personas");
export const USER_PERSONA_DIRECTORY_NAME = "assistant-personas";
export const PERSONA_MANIFEST_MAXIMUM_BYTES = 64 * 1024;
export const MAXIMUM_PERSONAS = 64;
export const MAXIMUM_USER_PERSONAS = 48;

const BUILT_IN_ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const BUILT_IN_ID_MAXIMUM_LENGTH = 64;

// Stable machine-readable codes. These reach the browser, so they must never carry absolute paths
// or file contents.
export const PERSONA_DIAGNOSTIC_CODES = Object.freeze([
  "personas_root_unavailable",
  "persona_unreadable",
  "persona_too_large",
  "persona_not_json",
  "persona_invalid_shape",
  "persona_id_mismatch",
  "duplicate_id",
]);

function diagnostic(file, code) {
  return { file, code };
}

// Returns null when the record is unusable. Callers skip those rather than failing the listing: one
// malformed file must not hide every other character the user has installed. "Unusable" means the
// id is wrong for its filename, or nothing at all would be sent — a record whose only content was
// a name would select silently and do nothing.
export function normalizePersona(value, expectedId, { builtIn = true } = {}) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const stated = typeof value.id === "string" ? value.id.trim().toLowerCase() : "";
  if (!stated) return null;
  if (builtIn && (stated.length > BUILT_IN_ID_MAXIMUM_LENGTH || !BUILT_IN_ID_PATTERN.test(stated))) return null;
  if (!builtIn && !validUserPersonaId(stated)) return null;
  if (expectedId && stated !== expectedId) return null;
  const persona = normalizeAssistantPersona(value, { id: stated, builtIn });
  if (!persona.id || !personaSections(persona).length) return null;
  return persona;
}

async function readPersonaFile(directory, name, { builtIn }) {
  const filePath = path.join(directory, name);
  const expectedId = name.slice(0, -".json".length).toLowerCase();
  let raw;
  try {
    const info = await stat(filePath);
    if (!info.isFile()) return { skipped: true };
    if (info.size > PERSONA_MANIFEST_MAXIMUM_BYTES) return { code: "persona_too_large" };
    raw = await readFile(filePath, "utf8");
  } catch {
    return { code: "persona_unreadable" };
  }
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { code: "persona_not_json" };
  }
  const persona = normalizePersona(parsed, expectedId, { builtIn });
  if (!persona) {
    return { code: parsed?.id && parsed.id !== expectedId ? "persona_id_mismatch" : "persona_invalid_shape" };
  }
  return { persona };
}

async function readPersonaFolder(directory, { builtIn, limit }) {
  const personas = [];
  const diagnostics = [];
  let entries;
  try {
    entries = await readdir(directory, { withFileTypes: true });
  } catch (error) {
    return { personas, diagnostics, available: false, code: error.code || null };
  }
  const files = entries
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(".json"))
    .map((entry) => entry.name)
    .sort((first, second) => first.localeCompare(second))
    .slice(0, limit);
  for (const name of files) {
    const result = await readPersonaFile(directory, name, { builtIn });
    if (result.skipped) continue;
    if (result.code) {
      diagnostics.push(diagnostic(name, result.code));
      continue;
    }
    personas.push(result.persona);
  }
  return { personas, diagnostics, available: true, code: null };
}

export function personaDirectory(root) {
  return path.join(root, PERSONA_DIRECTORY_NAME);
}

export function userPersonaDirectory(stateDirectory) {
  return path.join(stateDirectory, USER_PERSONA_DIRECTORY_NAME);
}

// The id becomes the filename, so it is re-validated here even though the route already checked
// it. A traversal sequence must never reach `path.join`.
function userPersonaFile(stateDirectory, id) {
  if (!validUserPersonaId(id)) throw Object.assign(new Error("Invalid persona id"), { statusCode: 400 });
  return path.join(userPersonaDirectory(stateDirectory), `${id}.json`);
}

// Both halves of the library, merged. `stateDirectory` is optional so a caller that only wants the
// shipped set — or a test with no state directory — keeps working.
export async function readPersonaDirectory(root, stateDirectory = "") {
  const shipped = await readPersonaFolder(personaDirectory(root), { builtIn: true, limit: MAXIMUM_PERSONAS });
  const authored = stateDirectory
    ? await readPersonaFolder(userPersonaDirectory(stateDirectory), { builtIn: false, limit: MAXIMUM_USER_PERSONAS })
    : { personas: [], diagnostics: [], available: false, code: null };

  const diagnostics = [...authored.diagnostics, ...shipped.diagnostics];
  // A missing shipped directory is a normal state (the user deleted it); it is reported, not
  // thrown, so the assistant still opens. A missing *user* directory is the ordinary case before
  // the first character is created and is not worth a diagnostic.
  if (!shipped.available) diagnostics.unshift(diagnostic(null, "personas_root_unavailable"));

  const seen = new Set();
  const personas = [];
  // Authored characters are admitted first: on the (impossible by construction, but cheap to
  // decide) event of an id collision, the file the user wrote is the one that wins.
  for (const persona of [...authored.personas, ...shipped.personas]) {
    if (personas.length >= MAXIMUM_PERSONAS) break;
    if (seen.has(persona.id)) {
      diagnostics.push(diagnostic(`${persona.id}.json`, "duplicate_id"));
      continue;
    }
    seen.add(persona.id);
    personas.push(persona);
  }
  return {
    personas: sortAssistantPersonas(personas),
    diagnostics,
    available: shipped.available || authored.available,
    code: shipped.code,
  };
}

async function writePersonaFile(stateDirectory, persona) {
  const directory = userPersonaDirectory(stateDirectory);
  await mkdir(directory, { recursive: true });
  const target = userPersonaFile(stateDirectory, persona.id);
  const temporaryPath = path.join(directory, `${randomUUID()}.tmp`);
  try {
    await writeFile(temporaryPath, `${JSON.stringify(persona, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    await rename(temporaryPath, target);
  } finally {
    await rm(temporaryPath, { force: true });
  }
  return persona;
}

export async function listUserPersonas(stateDirectory) {
  return (await readPersonaFolder(userPersonaDirectory(stateDirectory), { builtIn: false, limit: MAXIMUM_USER_PERSONAS })).personas;
}

export async function readUserPersona(stateDirectory, id) {
  if (!validUserPersonaId(id)) return null;
  try {
    const file = userPersonaFile(stateDirectory, id);
    const info = await stat(file);
    if (!info.isFile() || info.size > PERSONA_MANIFEST_MAXIMUM_BYTES) return null;
    return normalizeAssistantPersona(JSON.parse(await readFile(file, "utf8")), { id, builtIn: false });
  } catch {
    return null;
  }
}

// Stored without `builtIn`: that flag describes where a record came from, and a file in the user
// directory is authored by definition. Persisting it would let a hand-edit claim otherwise.
function storable(persona) {
  const { builtIn, ...rest } = persona;
  return rest;
}

export async function createUserPersona(stateDirectory, seed) {
  const existing = await listUserPersonas(stateDirectory);
  if (existing.length >= MAXIMUM_USER_PERSONAS) {
    throw Object.assign(new Error(`最多只能保存 ${MAXIMUM_USER_PERSONAS} 个自定义角色。`), { statusCode: 409 });
  }
  const id = randomUUID();
  const persona = normalizeAssistantPersona({
    ...(seed && typeof seed === "object" ? seed : {}),
    id,
    name: uniquePersonaName(seed?.name, existing.map((entry) => entry.name)),
  }, { id, builtIn: false });
  await writePersonaFile(stateDirectory, storable(persona));
  return persona;
}

export async function saveUserPersona(stateDirectory, id, incoming) {
  const existing = await readUserPersona(stateDirectory, id);
  if (!existing) return null;
  const persona = normalizeAssistantPersona({ ...incoming, id }, { id, builtIn: false });
  await writePersonaFile(stateDirectory, storable(persona));
  return persona;
}

export async function deleteUserPersona(stateDirectory, id) {
  try {
    await rm(userPersonaFile(stateDirectory, id), { force: true });
    return true;
  } catch {
    return false;
  }
}

// Resolution used at request time. An unknown or blank id yields null so the assistant sends only
// the output protocol instead of silently substituting a character the user did not pick.
export function selectPersona(personas, personaId) {
  const id = typeof personaId === "string" ? personaId.trim().toLowerCase() : "";
  if (!id) return null;
  return (Array.isArray(personas) ? personas : []).find((persona) => persona.id === id) || null;
}
