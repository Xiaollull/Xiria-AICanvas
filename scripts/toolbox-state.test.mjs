import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  MAXIMUM_INFO_BYTES,
  emptyToolboxState,
  normalizeToolboxState,
  referencedToolboxAssets,
  toolboxImageUrl,
} from "../src/toolbox-state.js";
import {
  MAXIMUM_TOOLBOX_ASSET_BYTES,
  decodeToolboxImage,
  pruneToolboxAssets,
  readToolboxAsset,
  readToolboxState,
  toolboxPaths,
  writeToolboxAsset,
  writeToolboxState,
} from "./toolbox-state.mjs";

// The Toolbox is mounted lazily and unmounted on the way out, so everything the image reader held
// used to be lost on navigation, refresh and restart. These tests are about the store that fixes
// that: what it accepts, what it round-trips, and what it refuses to keep.

const PNG = Buffer.from(
  "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a4944415478da6360000002000100"
  + "05fe02fea70000000049454e44ae426082", "hex");
const PNG_DATA_URL = `data:image/png;base64,${PNG.toString("base64")}`;

async function temporaryState() {
  const directory = await mkdtemp(path.join(tmpdir(), "xirai-toolbox-"));
  return { directory, cleanup: () => rm(directory, { recursive: true, force: true }) };
}

test("an unsaved toolbox opens on the first tool rather than on nothing", async () => {
  const { directory, cleanup } = await temporaryState();
  try {
    const { state, reset } = await readToolboxState(directory);
    assert.equal(reset, false);
    assert.deepEqual(state, emptyToolboxState());
    assert.equal(state.activeTool, "downloader");
  } finally {
    await cleanup();
  }
});

test("the selected tool survives a write and a read, which is what a restart is", async () => {
  const { directory, cleanup } = await temporaryState();
  try {
    await writeToolboxState(directory, { activeTool: "image-info", imageInfo: { directory: "D:/pictures" } });
    const { state } = await readToolboxState(directory);
    assert.equal(state.activeTool, "image-info");
    assert.equal(state.imageInfo.directory, "D:/pictures");
  } finally {
    await cleanup();
  }
});

test("an unreadable state file costs the selection and never the page", async () => {
  const { directory, cleanup } = await temporaryState();
  try {
    const { directory: storeDirectory, statePath } = toolboxPaths(directory);
    await mkdir(storeDirectory, { recursive: true });
    await writeFile(statePath, "{ not json", "utf8");
    const { state, reset } = await readToolboxState(directory);
    assert.equal(reset, true, "the caller is told the file was discarded");
    assert.deepEqual(state, emptyToolboxState());
  } finally {
    await cleanup();
  }
});

test("an uploaded picture round-trips through the asset store by content hash", async () => {
  const { directory, cleanup } = await temporaryState();
  try {
    const first = await writeToolboxAsset(directory, PNG_DATA_URL);
    assert.match(first.id, /^[a-f0-9]{32}$/);
    assert.equal(first.contentType, "image/png");
    // Content-addressed: reading the same picture twice must not accumulate files.
    const second = await writeToolboxAsset(directory, PNG_DATA_URL);
    assert.equal(second.id, first.id);
    const { assetsDirectory } = toolboxPaths(directory);
    assert.equal((await readdir(assetsDirectory)).length, 1);

    const asset = await readToolboxAsset(directory, first.id);
    assert.deepEqual(asset.buffer, PNG);
    assert.equal(asset.contentType, "image/png");
  } finally {
    await cleanup();
  }
});

test("bytes that are not the type they claim are refused rather than stored under that name", async () => {
  // The store serves files back with a content type taken from their name, so letting a caller
  // declare the type would let it choose the response header.
  const lying = `data:image/png;base64,${Buffer.from("this is not a png at all, not even close").toString("base64")}`;
  assert.throws(() => decodeToolboxImage(lying), /does not match its declared image type/);
  assert.throws(() => decodeToolboxImage("not a data url"), /base64 PNG, JPEG or WebP/);
});

test("an oversized upload is refused from its length, before it becomes a buffer", () => {
  const oversized = `data:image/png;base64,${"A".repeat(Math.ceil((MAXIMUM_TOOLBOX_ASSET_BYTES + 4096) / 3) * 4)}`;
  assert.throws(() => decodeToolboxImage(oversized), (error) => error.statusCode === 413);
});

test("an asset id that is not an id cannot reach the filesystem", async () => {
  const { directory, cleanup } = await temporaryState();
  try {
    for (const id of ["../../secrets", "..", "", "abc", "/etc/passwd", "a".repeat(31)]) {
      await assert.rejects(readToolboxAsset(directory, id), (error) => error.statusCode === 400, `accepted ${id}`);
    }
  } finally {
    await cleanup();
  }
});

test("a picture the state no longer points at is deleted rather than kept forever", async () => {
  const { directory, cleanup } = await temporaryState();
  try {
    const kept = await writeToolboxAsset(directory, PNG_DATA_URL);
    const other = Buffer.concat([PNG, Buffer.from([0])]);
    const dropped = await writeToolboxAsset(directory, `data:image/png;base64,${other.toString("base64")}`);
    const state = await writeToolboxState(directory, {
      activeTool: "image-info",
      imageInfo: { mode: "single", assetId: kept.id, assetName: "kept.png" },
    });
    const removed = await pruneToolboxAssets(directory, state);
    assert.equal(removed.length, 1);
    await assert.rejects(readToolboxAsset(directory, dropped.id), (error) => error.statusCode === 404);
    assert.ok(await readToolboxAsset(directory, kept.id), "the referenced picture is still there");
  } finally {
    await cleanup();
  }
});

test("the saved document never contains image bytes", async () => {
  const { directory, cleanup } = await temporaryState();
  try {
    const asset = await writeToolboxAsset(directory, PNG_DATA_URL);
    await writeToolboxState(directory, {
      activeTool: "image-info",
      imageInfo: { mode: "single", assetId: asset.id, assetName: "shot.png", info: { name: "shot.png" } },
    });
    const { statePath } = toolboxPaths(directory);
    const written = await readFile(statePath, "utf8");
    assert.ok(!written.includes("data:image"), "a data URL reached the state document");
    assert.ok(!written.includes(PNG.toString("base64")), "image bytes reached the state document");
    assert.ok(written.includes(asset.id), "the picture is referenced by its opaque id");
  } finally {
    await cleanup();
  }
});

test("transient state is not part of the record, so a restore cannot come back stuck", () => {
  const state = normalizeToolboxState({
    activeTool: "image-info",
    imageInfo: { mode: "single", loading: true, dragActive: true, requestToken: 7, imageUrl: "blob:x" },
  });
  for (const key of ["loading", "dragActive", "requestToken", "imageUrl"]) {
    assert.ok(!(key in state.imageInfo), `${key} was persisted`);
  }
});

test("a batch position is clamped to the list it belongs to", () => {
  const files = Array.from({ length: 3 }, (_, index) => ({ name: `${index}.png`, bytes: 10 }));
  assert.equal(normalizeToolboxState({ imageInfo: { mode: "batch", files, index: 99 } }).imageInfo.index, 2);
  assert.equal(normalizeToolboxState({ imageInfo: { mode: "batch", files, index: -4 } }).imageInfo.index, 0);
  // A batch with nothing in it is not a batch.
  assert.equal(normalizeToolboxState({ imageInfo: { mode: "batch", files: [] } }).imageInfo.mode, "single");
});

test("an upload and a directory selection never both survive", () => {
  const state = normalizeToolboxState({
    imageInfo: {
      mode: "batch",
      files: [{ name: "a.png" }],
      scannedDirectory: "D:/pictures",
      assetId: "a".repeat(32),
      assetName: "upload.png",
    },
  });
  // Keeping the losing one would let a restore show one picture and the other one's metadata.
  assert.equal(state.imageInfo.assetId, "");
  assert.equal(state.imageInfo.assetName, "");
});

test("a metadata record too large to write is dropped rather than written on every change", () => {
  const huge = { nodes: "x".repeat(MAXIMUM_INFO_BYTES + 1) };
  assert.equal(normalizeToolboxState({ imageInfo: { info: huge } }).imageInfo.info, null);
  const ordinary = { name: "shot.png", prompt: "a cat" };
  assert.deepEqual(normalizeToolboxState({ imageInfo: { info: ordinary } }).imageInfo.info, ordinary);
});

test("the preview URL is derived, and is never a filesystem path", () => {
  const upload = normalizeToolboxState({ imageInfo: { mode: "single", assetId: "b".repeat(32), assetName: "a.png" } });
  assert.equal(toolboxImageUrl(upload.imageInfo), `/api/toolbox/state/asset?id=${"b".repeat(32)}`);

  const batch = normalizeToolboxState({
    imageInfo: { mode: "batch", files: [{ name: "one.png" }, { name: "two.png" }], index: 1, scannedDirectory: "D:/pics" },
  });
  const url = toolboxImageUrl(batch.imageInfo);
  assert.match(url, /^\/api\/image-info\/preview\?directory=/);
  assert.ok(url.includes(encodeURIComponent("two.png")), "the URL addresses the selected file");
  assert.equal(toolboxImageUrl(emptyToolboxState().imageInfo), "");
});

test("the referenced set is what pruning trusts, and it follows the normalised state", () => {
  const id = "c".repeat(32);
  assert.deepEqual([...referencedToolboxAssets({ imageInfo: { mode: "single", assetId: id } })], [id]);
  // Dropped by normalisation, so nothing keeps the file alive.
  assert.deepEqual([...referencedToolboxAssets({ imageInfo: { mode: "single", assetId: "nope" } })], []);
});
