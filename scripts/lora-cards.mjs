import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  CARD_ASSET_PATTERN,
  emptyLoraCardStore,
  normalizeLoraCardStore,
  referencedCardAssets,
} from "../src/lora-cards.js";

// Storage for the user-owned LoRA card store. The shape and every rule about
// what a card may contain live in `src/lora-cards.js`, which the browser reads
// too; this file only decides where the bytes go and how they get there safely.

export const MAXIMUM_ASSET_BYTES = 4 * 1024 * 1024;
export const CARD_ASSET_TYPES = { "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp" };
const EXTENSION_TYPES = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp" };
const DATA_URL_PATTERN = /^data:(image\/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=\s]+)$/;

export function loraCardPaths(stateDirectory) {
  const directory = path.join(stateDirectory, "lora-cards");
  return { directory, indexPath: path.join(directory, "cards.json"), assetsDirectory: path.join(directory, "assets") };
}

/**
 * What the bytes actually are, rather than what the upload claimed. The store
 * serves these files back with a content type derived from their name, so the
 * name has to be decided by the content or a caller could pick the header.
 */
export function sniffCardImageType(buffer) {
  if (!Buffer.isBuffer(buffer) || buffer.length < 12) return "";
  if (buffer.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) return "image/png";
  if (buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) return "image/jpeg";
  if (buffer.subarray(0, 4).toString("latin1") === "RIFF" && buffer.subarray(8, 12).toString("latin1") === "WEBP") return "image/webp";
  return "";
}

/**
 * Decodes one `data:image/...;base64,` URL into bytes. The declared type has to
 * match the sniffed one: an upload that says PNG and carries something else is
 * refused rather than stored under a name that would misdescribe it later.
 */
export function decodeCardImage(dataUrl) {
  const match = typeof dataUrl === "string" ? dataUrl.match(DATA_URL_PATTERN) : null;
  if (!match) throw Object.assign(new Error("Card images must be a base64 PNG, JPEG or WebP data URL"), { statusCode: 400 });
  const [, declaredType, encoded] = match;
  const compact = encoded.replace(/\s+/g, "");
  // Base64 expands by 4/3, so this rejects an oversized upload before it is
  // materialised as a buffer instead of after.
  if (Math.floor(compact.length / 4) * 3 > MAXIMUM_ASSET_BYTES + 3) {
    throw Object.assign(new Error("Card images must be 4 MB or smaller"), { statusCode: 413 });
  }
  const buffer = Buffer.from(compact, "base64");
  if (!buffer.length) throw Object.assign(new Error("Card image is empty"), { statusCode: 400 });
  if (buffer.length > MAXIMUM_ASSET_BYTES) throw Object.assign(new Error("Card images must be 4 MB or smaller"), { statusCode: 413 });
  const sniffed = sniffCardImageType(buffer);
  if (!sniffed || sniffed !== declaredType) {
    throw Object.assign(new Error("Card image content does not match its declared image type"), { statusCode: 400 });
  }
  return { buffer, contentType: sniffed, extension: CARD_ASSET_TYPES[sniffed] };
}

/** Content-addressed, so re-uploading the same picture reuses the file it already has. */
export function cardAssetId(buffer) {
  return createHash("sha256").update(buffer).digest("hex").slice(0, 32);
}

export function cardAssetFileName(id, extension) {
  if (!CARD_ASSET_PATTERN.test(id || "")) throw Object.assign(new Error("Invalid card image id"), { statusCode: 400 });
  if (!EXTENSION_TYPES[extension]) throw Object.assign(new Error("Invalid card image type"), { statusCode: 400 });
  return `${id}${extension}`;
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

export async function readLoraCardStore(stateDirectory) {
  const { indexPath } = loraCardPaths(stateDirectory);
  let saved;
  try {
    saved = JSON.parse(await readFile(indexPath, "utf8"));
  } catch (error) {
    // A missing file is the normal first run; an unreadable one is a card store
    // the user can rebuild by typing, so neither is worth failing the request.
    if (error.code === "ENOENT") return { store: emptyLoraCardStore(), rejected: 0, reset: false };
    return { store: emptyLoraCardStore(), rejected: 0, reset: true };
  }
  return normalizeLoraCardStore(saved);
}

export async function writeLoraCardStore(stateDirectory, store) {
  const { directory, indexPath } = loraCardPaths(stateDirectory);
  const normalized = normalizeLoraCardStore(store).store;
  await writeFileAtomically(directory, indexPath, `${JSON.stringify(normalized, null, 2)}\n`);
  return normalized;
}

export async function writeCardAsset(stateDirectory, dataUrl) {
  const { assetsDirectory } = loraCardPaths(stateDirectory);
  const decoded = decodeCardImage(dataUrl);
  const id = cardAssetId(decoded.buffer);
  const fileName = cardAssetFileName(id, decoded.extension);
  await writeFileAtomically(assetsDirectory, path.join(assetsDirectory, fileName), decoded.buffer);
  return { id, fileName, contentType: decoded.contentType, bytes: decoded.buffer.length };
}

export async function readCardAsset(stateDirectory, id) {
  if (!CARD_ASSET_PATTERN.test(id || "")) throw Object.assign(new Error("Invalid card image id"), { statusCode: 400 });
  const { assetsDirectory } = loraCardPaths(stateDirectory);
  for (const extension of Object.keys(EXTENSION_TYPES)) {
    try {
      // The id is 32 hex characters and the extension comes from a fixed table,
      // so the joined name cannot escape the assets directory by construction.
      const buffer = await readFile(path.join(assetsDirectory, `${id}${extension}`));
      return { buffer, contentType: EXTENSION_TYPES[extension] };
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
  }
  throw Object.assign(new Error("Card image not found"), { statusCode: 404 });
}

/**
 * Deletes image files no card points at any more. Called after each write, so
 * replacing a cover reclaims the old one instead of leaving it behind forever.
 */
export async function pruneCardAssets(stateDirectory, store) {
  const { assetsDirectory } = loraCardPaths(stateDirectory);
  const referenced = referencedCardAssets(store);
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
    // Anything that is not a well-formed asset name is left alone: this walks a
    // directory inside the user's state folder and must not become a cleaner.
    if (!EXTENSION_TYPES[extension] || !CARD_ASSET_PATTERN.test(id) || referenced.has(id)) continue;
    await rm(path.join(assetsDirectory, entry), { force: true });
    removed.push(id);
  }
  return removed;
}
