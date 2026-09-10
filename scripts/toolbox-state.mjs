import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  TOOLBOX_ASSET_PATTERN,
  emptyToolboxState,
  normalizeToolboxState,
  referencedToolboxAssets,
} from "../src/toolbox-state.js";

// Where the Toolbox's saved state lives. The shape is decided in `src/toolbox-state.js`, which the
// browser reads too; this file only decides where the bytes go and how they get there safely.
//
// It is a separate store from `/api/ui-state` on purpose. The workspace state is rewritten on
// almost every interaction in the generate page and is capped at 16 MB for the whole document; an
// uploaded picture inside it would be re-encoded as base64 and rewritten on every slider drag.
// Here the picture is a file of its own, referenced by a hash of its contents, and the JSON beside
// it stays a few kilobytes.

// The reader itself accepts a 64 MB upload. Persisting one would mean an 85 MB base64 request on
// every change, so anything past this is still read and shown in the session and simply is not
// saved: the reader reports that on restore rather than pretending the picture is still there.
export const MAXIMUM_TOOLBOX_ASSET_BYTES = 12 * 1024 * 1024;
const ASSET_TYPES = { "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp" };
const EXTENSION_TYPES = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp" };
const DATA_URL_PATTERN = /^data:(image\/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=\s]+)$/;

export function toolboxPaths(stateDirectory) {
  const directory = path.join(stateDirectory, "toolbox");
  return {
    directory,
    statePath: path.join(directory, "image-info.json"),
    assetsDirectory: path.join(directory, "assets"),
  };
}

/**
 * What the bytes are, rather than what the upload claimed they are.
 *
 * The store serves these files back with a content type taken from their file name, so the name
 * has to be decided by the content: otherwise a caller would be choosing the response header.
 */
export function sniffToolboxImageType(buffer) {
  if (!Buffer.isBuffer(buffer) || buffer.length < 12) return "";
  if (buffer.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) return "image/png";
  if (buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) return "image/jpeg";
  if (buffer.subarray(0, 4).toString("latin1") === "RIFF" && buffer.subarray(8, 12).toString("latin1") === "WEBP") return "image/webp";
  return "";
}

export function decodeToolboxImage(dataUrl) {
  const match = typeof dataUrl === "string" ? dataUrl.match(DATA_URL_PATTERN) : null;
  if (!match) throw Object.assign(new Error("Toolbox images must be a base64 PNG, JPEG or WebP data URL"), { statusCode: 400 });
  const [, declaredType, encoded] = match;
  const compact = encoded.replace(/\s+/g, "");
  // Base64 expands by 4/3, so an oversized upload is refused from its length rather than after
  // being materialised as a buffer.
  if (Math.floor(compact.length / 4) * 3 > MAXIMUM_TOOLBOX_ASSET_BYTES + 3) {
    throw Object.assign(new Error("Toolbox images must be 12 MB or smaller to be remembered"), { statusCode: 413 });
  }
  const buffer = Buffer.from(compact, "base64");
  if (!buffer.length) throw Object.assign(new Error("Toolbox image is empty"), { statusCode: 400 });
  if (buffer.length > MAXIMUM_TOOLBOX_ASSET_BYTES) {
    throw Object.assign(new Error("Toolbox images must be 12 MB or smaller to be remembered"), { statusCode: 413 });
  }
  const sniffed = sniffToolboxImageType(buffer);
  if (!sniffed || sniffed !== declaredType) {
    throw Object.assign(new Error("Toolbox image content does not match its declared image type"), { statusCode: 400 });
  }
  return { buffer, contentType: sniffed, extension: ASSET_TYPES[sniffed] };
}

/** Content-addressed, so reading the same picture twice reuses the file it already has. */
export function toolboxAssetId(buffer) {
  return createHash("sha256").update(buffer).digest("hex").slice(0, 32);
}

async function writeFileAtomically(directory, target, contents) {
  await mkdir(directory, { recursive: true });
  const temporaryPath = path.join(directory, `${randomUUID()}.tmp`);
  try {
    await writeFile(temporaryPath, contents, { flag: "wx" });
    await rename(temporaryPath, target);
  } finally {
    await rm(temporaryPath, { force: true });
  }
}

export async function readToolboxState(stateDirectory) {
  const { statePath } = toolboxPaths(stateDirectory);
  try {
    const saved = JSON.parse(await readFile(statePath, "utf8"));
    return { state: normalizeToolboxState(saved), reset: false };
  } catch (error) {
    // A missing file is the normal first run. An unreadable one costs the user a tool selection
    // they can make again, so neither is worth failing the request over.
    if (error.code === "ENOENT") return { state: emptyToolboxState(), reset: false };
    return { state: emptyToolboxState(), reset: true };
  }
}

export async function writeToolboxState(stateDirectory, state) {
  const { directory, statePath } = toolboxPaths(stateDirectory);
  const normalized = normalizeToolboxState(state);
  await writeFileAtomically(directory, statePath, `${JSON.stringify(normalized, null, 2)}\n`);
  return normalized;
}

export async function writeToolboxAsset(stateDirectory, dataUrl) {
  const { assetsDirectory } = toolboxPaths(stateDirectory);
  const decoded = decodeToolboxImage(dataUrl);
  const id = toolboxAssetId(decoded.buffer);
  await writeFileAtomically(assetsDirectory, path.join(assetsDirectory, `${id}${decoded.extension}`), decoded.buffer);
  return { id, contentType: decoded.contentType, bytes: decoded.buffer.length };
}

export async function readToolboxAsset(stateDirectory, id) {
  if (!TOOLBOX_ASSET_PATTERN.test(id || "")) throw Object.assign(new Error("Invalid toolbox image id"), { statusCode: 400 });
  const { assetsDirectory } = toolboxPaths(stateDirectory);
  for (const extension of Object.keys(EXTENSION_TYPES)) {
    try {
      // The id is 32 hex characters and the extension comes from a fixed table, so the joined
      // name cannot escape the assets directory by construction.
      const buffer = await readFile(path.join(assetsDirectory, `${id}${extension}`));
      return { buffer, contentType: EXTENSION_TYPES[extension] };
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
  }
  throw Object.assign(new Error("Toolbox image not found"), { statusCode: 404 });
}

/**
 * Deletes image files the saved state no longer points at. Called after each write, so reading a
 * second picture reclaims the first instead of leaving it behind forever.
 */
export async function pruneToolboxAssets(stateDirectory, state) {
  const { assetsDirectory } = toolboxPaths(stateDirectory);
  const referenced = referencedToolboxAssets(state);
  let entries;
  try {
    entries = await readdir(assetsDirectory);
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
  const removed = [];
  for (const entry of entries) {
    const extension = path.extname(entry).toLowerCase();
    const id = path.basename(entry, extension);
    if (!EXTENSION_TYPES[extension] || !TOOLBOX_ASSET_PATTERN.test(id) || referenced.has(id)) continue;
    await rm(path.join(assetsDirectory, entry), { force: true });
    removed.push(entry);
  }
  return removed;
}
