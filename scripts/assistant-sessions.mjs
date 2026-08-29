import { randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  MAXIMUM_SESSIONS,
  normalizeSession,
  sessionSummary,
  sortSessions,
  validSessionId,
} from "../src/ai-assistant-protocol.js";

// Chat session storage: one JSON file per conversation under `state-cache/assistant-sessions/`.
//
// One file per session rather than one array file, because appending a message would otherwise
// rewrite every stored conversation on every turn. The directory listing is the index, so there is
// no separate manifest that can disagree with what is on disk.

export const SESSION_DIRECTORY_NAME = "assistant-sessions";
export const SESSION_FILE_MAXIMUM_BYTES = 4 * 1024 * 1024;

export function sessionDirectory(stateDirectory) {
  return path.join(stateDirectory, SESSION_DIRECTORY_NAME);
}

// The id is the filename, so it is re-validated here even though the route already checked it.
// A traversal sequence must never reach `path.join`.
function sessionFile(stateDirectory, id) {
  if (!validSessionId(id)) throw Object.assign(new Error("Invalid session id"), { statusCode: 400 });
  return path.join(sessionDirectory(stateDirectory), `${id}.json`);
}

async function writeSessionFile(stateDirectory, session) {
  const directory = sessionDirectory(stateDirectory);
  await mkdir(directory, { recursive: true });
  const target = sessionFile(stateDirectory, session.id);
  const temporaryPath = path.join(directory, `${randomUUID()}.tmp`);
  try {
    await writeFile(temporaryPath, `${JSON.stringify(session, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    await rename(temporaryPath, target);
  } finally {
    await rm(temporaryPath, { force: true });
  }
  return session;
}

export async function readSession(stateDirectory, id) {
  const file = sessionFile(stateDirectory, id);
  try {
    const info = await stat(file);
    if (!info.isFile() || info.size > SESSION_FILE_MAXIMUM_BYTES) return null;
    return normalizeSession(JSON.parse(await readFile(file, "utf8")), { id });
  } catch (error) {
    if (error.code === "ENOENT") return null;
    // A corrupt file still opens as an empty session rather than making its slot permanently
    // unreachable; the user can then delete it from the picker.
    return normalizeSession(null, { id });
  }
}

// Returns metadata only. Reading every transcript to render a sidebar would scale with total chat
// history rather than with the number of conversations.
export async function listSessions(stateDirectory) {
  const directory = sessionDirectory(stateDirectory);
  let entries;
  try {
    entries = await readdir(directory, { withFileTypes: true });
  } catch {
    return [];
  }
  const ids = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
    .map((entry) => entry.name.slice(0, -".json".length))
    .filter(validSessionId);
  const sessions = [];
  for (const id of ids) {
    const session = await readSession(stateDirectory, id);
    if (session) sessions.push(sessionSummary(session));
  }
  return sortSessions(sessions);
}

export async function createSession(stateDirectory, seed = null) {
  const now = new Date().toISOString();
  const session = normalizeSession(seed, { id: randomUUID(), now });
  await writeSessionFile(stateDirectory, session);
  await pruneSessions(stateDirectory);
  return session;
}

export async function saveSession(stateDirectory, id, incoming) {
  const existing = await readSession(stateDirectory, id);
  if (!existing) return null;
  const now = new Date().toISOString();
  const session = normalizeSession({
    ...incoming,
    id,
    createdAt: existing.createdAt,
    updatedAt: now,
    // An explicit title is kept; otherwise it is re-derived, so a conversation that has just
    // received its first message stops being called 新对话.
    title: typeof incoming?.title === "string" && incoming.title.trim() ? incoming.title : "",
  }, { id, now });
  return writeSessionFile(stateDirectory, session);
}

export async function deleteSession(stateDirectory, id) {
  const file = sessionFile(stateDirectory, id);
  try {
    await rm(file, { force: true });
    return true;
  } catch {
    return false;
  }
}

// Keeps stored history bounded. Trims the least recently updated, which is also the bottom of the
// picker, so the cap never silently removes the conversation the user is looking at.
export async function pruneSessions(stateDirectory) {
  const sessions = await listSessions(stateDirectory);
  if (sessions.length <= MAXIMUM_SESSIONS) return [];
  const removed = sessions.slice(MAXIMUM_SESSIONS);
  for (const session of removed) await deleteSession(stateDirectory, session.id);
  return removed.map((session) => session.id);
}
