import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// The contract between the Toolbox page, the image reader and the store behind them. The store
// itself is exercised for real in `toolbox-state.test.mjs`; this file pins the wiring that decides
// whether any of it is reached.

const read = (file) => readFile(new URL(`../src/${file}`, import.meta.url), "utf8");

test("the toolbox holds its selection in the project, not in component state", async () => {
  const page = await read("ToolboxPage.jsx");
  // `useState` here was the bug: the page is lazily mounted and unmounted the moment the user
  // navigates away, so the selected tool went with it.
  assert.match(page, /useToolboxState\(\)/);
  assert.doesNotMatch(page, /useState\(TOOLBOX_TOOLS\[0\]\.id\)/);
  assert.match(page, /onClick=\{\(\) => setActiveTool\(tool\.id\)\}/);
});

test("no tool mounts before the saved state has arrived", async () => {
  const page = await read("ToolboxPage.jsx");
  // A tool that mounted empty and was handed its state afterwards would flash the empty view and
  // then have to reconcile the restore against whatever the user had already typed.
  assert.match(page, /!ready \?/);
  assert.match(page, /savedState=\{state\.imageInfo\} onStateChange=\{setImageInfo\}/);
});

test("the reader is seeded from the snapshot rather than restored into afterwards", async () => {
  const reader = await read("ImageInfoReader.jsx");
  for (const field of ["mode", "directory", "files", "index", "scannedDirectory", "truncated", "info"]) {
    assert.match(reader, new RegExp(`useState\\(restored\\.${field}\\)`), `${field} is not seeded`);
  }
  assert.match(reader, /useState\(\(\) => toolboxImageUrl\(restored\)\)/);
});

test("transient state is rebuilt on restore instead of being carried across", async () => {
  const reader = await read("ImageInfoReader.jsx");
  // A snapshot taken mid-request would otherwise come back as a reader stuck loading forever.
  assert.match(reader, /const \[loading, setLoading\] = useState\(false\)/);
  assert.match(reader, /const \[dragActive, setDragActive\] = useState\(false\)/);
  const reported = reader.match(/reportRef\.current\?\.\(\{([^}]*)\}\)/);
  assert.ok(reported, "the reader never reports its state");
  for (const key of ["loading", "dragActive", "imageUrl", "detailsOpen", "applyOpen"]) {
    assert.ok(!reported[1].includes(key), `${key} is reported and must not be`);
  }
});

test("an uploaded picture is stored by the project, and a directory one is not", async () => {
  const reader = await read("ImageInfoReader.jsx");
  assert.match(reader, /storeUploadedImage\(dataUrl\)/);
  assert.match(reader, /"\/api\/toolbox\/state\/asset"/);
  // A directory picture is addressed by the route that serves it, so the upload's stored copy is
  // dropped rather than left pointing at a picture that is no longer on screen.
  const directoryRead = reader.slice(reader.indexOf("const readFromDirectory"), reader.indexOf("const scanDirectory"));
  assert.match(directoryRead, /setAssetId\(""\)/);
});

test("failing to keep the picture never becomes a failure to read it", async () => {
  const reader = await read("ImageInfoReader.jsx");
  const store = reader.slice(reader.indexOf("async function storeUploadedImage"), reader.indexOf("async function readJsonResponse"));
  assert.match(store, /if \(!response\.ok\) return null;/);
  assert.match(store, /catch \{\s*return null;\s*\}/);
});

test("a source that has gone away reports itself instead of showing a broken picture", async () => {
  const reader = await read("ImageInfoReader.jsx");
  assert.match(reader, /onError=\{\(\) => \{ setImageUrl\(""\); setError\(/);
  assert.match(reader, /原图已不在原位置/);
});

test("the toolbox keeps nothing in browser storage", async () => {
  for (const file of ["ToolboxPage.jsx", "ImageInfoReader.jsx", "use-toolbox-state.js", "toolbox-state.js"]) {
    const source = await read(file);
    assert.ok(!/localStorage|sessionStorage|indexedDB/i.test(source), `${file} reaches for browser storage`);
  }
});

test("saves are debounced and flushed on the way out", async () => {
  const hook = await read("use-toolbox-state.js");
  // Typing a directory path is one change per keystroke, so an unthrottled save would rewrite the
  // file dozens of times a second.
  assert.match(hook, /window\.setTimeout\(flush, SAVE_DELAY\)/);
  // Leaving the Toolbox is exactly when the state most needs to have been written.
  assert.match(hook, /useEffect\(\(\) => \(\) => \{\s*window\.clearTimeout\(timerRef\.current\);\s*void flush\(\);/);
});

test("nothing is written before the first read has come back", async () => {
  const hook = await read("use-toolbox-state.js");
  // Otherwise the mount-time save from a tool's own initialisation overwrites the saved state
  // with an empty one, which is a data-loss bug rather than a cosmetic one.
  assert.match(hook, /if \(!loadedRef\.current\) return;/);
});

test("the reader parses every response defensively, on every path", async () => {
  const reader = await read("ImageInfoReader.jsx");
  // A 502 from the proxy has an HTML body. Parsing before checking `ok` turned that into
  // "Unexpected token '<'" instead of telling the user the request had failed.
  assert.doesNotMatch(reader, /await response\.json\(\);/);
  assert.equal((reader.match(/await readJsonResponse\(response, /g) || []).length, 4,
    "upload, directory read, directory scan and the model re-check must all go through the safe parser");
  const parser = reader.slice(reader.indexOf("async function readJsonResponse"), reader.indexOf("export default function"));
  assert.match(parser, /await response\.text\(\)/);
  assert.match(parser, /HTTP \$\{response\.status\}/);
});
