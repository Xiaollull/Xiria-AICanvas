// The Toolbox's own saved state, and every rule about what it may contain.
//
// The Toolbox is mounted lazily and unmounted the moment the user navigates away, so everything it
// held in React state was lost on the way out: the selected tool, the picture the image reader had
// open, and the metadata it had just parsed. Coming back landed on an empty reader, and so did a
// refresh or a restart.
//
// This module is the shared half of the fix. It is imported by the browser and by the storage
// layer in `scripts/toolbox-state.mjs`, so the shape is described once and both ends agree on it.
// Storage decides where bytes go; this decides what is allowed to be there.
//
// Two things are deliberately absent. Nothing here is transient: `loading`, `dragActive` and the
// in-flight request token are rebuilt on restore rather than saved, because a snapshot taken
// mid-request would otherwise come back as a reader stuck in a loading state that nothing will
// finish. And no image bytes are here either — a picture is referenced by an opaque asset id and
// lives in its own file, so the state stays small enough to write on every change.

export const TOOLBOX_STATE_VERSION = 1;
// Kept in step with the rail in `ToolboxPage`. The first entry is what the Toolbox opens on when
// nothing has been saved yet.
export const TOOLBOX_TOOLS = ["downloader", "image-info"];
export const TOOLBOX_ASSET_PATTERN = /^[a-f0-9]{32}$/;
// A ComfyUI export can carry a sixty-node graph, and the reader shows all of it. This bounds one
// saved record without truncating anything a normal picture produces.
export const MAXIMUM_INFO_BYTES = 512 * 1024;
export const MAXIMUM_BATCH_FILES = 2000;

export function emptyImageInfoState() {
  return {
    mode: "single",
    assetId: "",
    assetName: "",
    directory: "",
    scannedDirectory: "",
    files: [],
    index: 0,
    truncated: false,
    info: null,
  };
}

export function emptyToolboxState() {
  return { version: TOOLBOX_STATE_VERSION, activeTool: TOOLBOX_TOOLS[0], imageInfo: emptyImageInfoState() };
}

function text(value, limit = 4096) {
  return typeof value === "string" ? value.slice(0, limit) : "";
}

function boundedIndex(value, count) {
  const numeric = Number.isFinite(Number(value)) ? Math.trunc(Number(value)) : 0;
  if (!count) return 0;
  return Math.max(0, Math.min(count - 1, numeric));
}

/**
 * Whether a parsed metadata record is small enough to keep.
 *
 * Measured by serialising it, because the cost that matters is the cost of writing the file, and a
 * graph with a thousand nodes is cheap to hold in memory and expensive to write on every change.
 */
export function infoWithinLimit(info) {
  if (info === null || info === undefined) return true;
  try {
    return JSON.stringify(info).length <= MAXIMUM_INFO_BYTES;
  } catch {
    return false;
  }
}

export function normalizeImageInfoState(value) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const files = Array.isArray(source.files)
    ? source.files.slice(0, MAXIMUM_BATCH_FILES)
      .map((file) => (file && typeof file === "object" ? { name: text(file.name, 512), bytes: Number.isFinite(Number(file.bytes)) ? Number(file.bytes) : 0 } : null))
      .filter((file) => file && file.name)
    : [];
  const mode = source.mode === "batch" && files.length ? "batch" : "single";
  const assetId = TOOLBOX_ASSET_PATTERN.test(source.assetId || "") ? source.assetId : "";
  return {
    mode,
    // An upload and a directory selection are alternatives, never both: keeping the losing one
    // would let a restore show one picture and the other one's metadata.
    assetId: mode === "single" ? assetId : "",
    assetName: mode === "single" && assetId ? text(source.assetName, 512) : "",
    directory: text(source.directory, 4096),
    scannedDirectory: mode === "batch" ? text(source.scannedDirectory, 4096) : "",
    files: mode === "batch" ? files : [],
    index: mode === "batch" ? boundedIndex(source.index, files.length) : 0,
    truncated: mode === "batch" && source.truncated === true,
    info: infoWithinLimit(source.info) && source.info && typeof source.info === "object" ? source.info : null,
  };
}

export function normalizeToolboxState(value) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  return {
    version: TOOLBOX_STATE_VERSION,
    activeTool: TOOLBOX_TOOLS.includes(source.activeTool) ? source.activeTool : TOOLBOX_TOOLS[0],
    imageInfo: normalizeImageInfoState(source.imageInfo),
  };
}

/** The asset ids a saved state still points at, so storage can delete the rest. */
export function referencedToolboxAssets(state) {
  const normalized = normalizeToolboxState(state);
  return new Set(normalized.imageInfo.assetId ? [normalized.imageInfo.assetId] : []);
}

/**
 * The preview URL for a saved reader state, or "" when there is nothing to show.
 *
 * A directory picture is addressed by the route that serves it, so it needs no stored copy and
 * survives a restart for as long as the file does. An uploaded one is addressed by its asset id.
 * Neither form puts a filesystem path in front of the browser.
 */
export function toolboxImageUrl(imageInfo) {
  const state = normalizeImageInfoState(imageInfo);
  if (state.mode === "batch") {
    const name = state.files[state.index]?.name;
    if (!state.scannedDirectory || !name) return "";
    return `/api/image-info/preview?directory=${encodeURIComponent(state.scannedDirectory)}&name=${encodeURIComponent(name)}`;
  }
  return state.assetId ? `/api/toolbox/state/asset?id=${state.assetId}` : "";
}
