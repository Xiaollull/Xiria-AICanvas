import { randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";

import { cardAssetId, decodeCardImage } from "./lora-cards.mjs";
import { PROMPT_PRESET_PREVIEW_PATTERN, PROMPT_PRESET_SCHEMA_VERSION, referencedPromptPresetPreviews } from "../src/prompt-presets.js";

// Preview pictures of prompt preset cards.
//
// The preset library lives in the workspace document, which is rewritten on nearly every change,
// so a card holds only the content-hash id of its picture and the bytes live here, one file each,
// decoded and type-checked exactly like a LoRA card cover. A picture is uploaded while its card is
// still being edited — before any saved card names it — so pruning spares anything written in the
// last hour: an unsaved upload survives the autosaves that happen around it, and an abandoned one is
// collected later.

export const PROMPT_PRESET_PREVIEW_GRACE_MS = 60 * 60 * 1000;
const EXTENSION_TYPES = { ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp" };

export function promptPresetPreviewDirectory(stateDirectory) {
  return path.join(stateDirectory, "prompt-presets", "previews");
}

export async function writePromptPresetPreview(stateDirectory, dataUrl) {
  const decoded = decodeCardImage(dataUrl);
  const id = cardAssetId(decoded.buffer);
  const directory = promptPresetPreviewDirectory(stateDirectory);
  await mkdir(directory, { recursive: true });
  const target = path.join(directory, `${id}${decoded.extension}`);
  const temporaryPath = path.join(directory, `${randomUUID()}.tmp`);
  try {
    await writeFile(temporaryPath, decoded.buffer, { flag: "wx" });
    // Re-uploading the same picture replaces the file with identical bytes, which also refreshes its
    // age, so a picture re-attached to a card is not pruned as a stale orphan.
    await rename(temporaryPath, target);
  } finally {
    await rm(temporaryPath, { force: true });
  }
  return { id, contentType: decoded.contentType, bytes: decoded.buffer.length };
}

export async function readPromptPresetPreview(stateDirectory, id) {
  if (!PROMPT_PRESET_PREVIEW_PATTERN.test(id || "")) throw Object.assign(new Error("Invalid preview id"), { statusCode: 400 });
  const directory = promptPresetPreviewDirectory(stateDirectory);
  for (const [extension, contentType] of Object.entries(EXTENSION_TYPES)) {
    try {
      // The id is 32 hex characters and the extension comes from a fixed table, so the joined name
      // cannot leave the preview directory.
      return { buffer: await readFile(path.join(directory, `${id}${extension}`)), contentType };
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
  }
  throw Object.assign(new Error("Preview not found"), { statusCode: 404 });
}

/**
 * The preview ids a saved workspace document still references, or `null` when its library is not a
 * schema this build understands — a damaged or newer library is kept verbatim by the page, and its
 * pictures must not be collected on the strength of a reading this build cannot make.
 */
export function referencedPreviewsInUiState(state) {
  const library = state?.workspace?.promptPresets;
  if (!library || typeof library !== "object" || library.schemaVersion !== PROMPT_PRESET_SCHEMA_VERSION || !Array.isArray(library.records)) return null;
  return referencedPromptPresetPreviews(library);
}

/** Deletes previews nothing references that are older than the grace period. */
export async function prunePromptPresetPreviews(stateDirectory, referenced, { now = Date.now(), graceMs = PROMPT_PRESET_PREVIEW_GRACE_MS } = {}) {
  if (!(referenced instanceof Set)) return [];
  const directory = promptPresetPreviewDirectory(stateDirectory);
  let entries;
  try {
    entries = await readdir(directory);
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
  const removed = [];
  for (const entry of entries) {
    const extension = path.extname(entry).toLowerCase();
    const id = path.basename(entry, extension);
    // Only well-formed preview names are ever touched: this walks a folder in the user's state
    // directory and must not become a general cleaner.
    if (!EXTENSION_TYPES[extension] || !PROMPT_PRESET_PREVIEW_PATTERN.test(id) || referenced.has(id)) continue;
    const file = path.join(directory, entry);
    try {
      if (now - (await stat(file)).mtimeMs < graceMs) continue;
      await rm(file, { force: true });
      removed.push(id);
    } catch {}
  }
  return removed;
}
