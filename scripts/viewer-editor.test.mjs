import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {
  appendStrokePoints,
  admitViewerUndoSnapshot,
  assertViewerByteBudget,
  boundedViewerUndoStack,
  clientPointToLayer,
  clipLineToLayer,
  clipStrokeSamples,
  cloneViewerLayer,
  hasViewerEdits,
  historyAssetIdFromUrl,
  mapViewerConcurrent,
  normalizeManualLayout,
  normalizePaintStroke,
  normalizeViewerEdgeLine,
  persistedManualLayout,
  readViewerFileAsDataUrl,
  resolvedCollageEntries,
  serializeViewerLayer,
  viewerCanvasDimensions,
  viewerAnimatedBatchAdmission,
  viewerClipboardPasteIntent,
  viewerDataUrlBytes,
  viewerEditorLayerBounds,
  viewerFileBatchAdmission,
  viewerLayerSourceByteCount,
  viewerSafeResizeHandles,
  viewerStrokeCount,
  viewerStrokePointCount,
  viewerUndoSnapshotMetrics,
  viewerUndoStackMetrics,
  VIEWER_DEFAULT_COLOR,
  MAX_EXPORT_PIXELS,
  VIEWER_MAX_LAYOUT_POINTS,
  VIEWER_MAX_LAYOUT_STROKES,
  VIEWER_MAX_ANIMATED_SOURCE_BYTES,
  VIEWER_MAX_OUTPUT_BYTES,
  VIEWER_MAX_UNDO_POINTS,
  VIEWER_MAX_UNDO_SOURCE_BYTES,
} from "../src/viewer-editor.js";

test("a rotated layer uses a rotation-aware AABB", () => {
  const layer = { naturalWidth: 100, naturalHeight: 40, x: 10, y: -5, scale: 2, rotation: 90 };
  const quarterTurn = viewerEditorLayerBounds(layer);
  assert.ok(Math.abs(quarterTurn.left + 30) < 1e-9);
  assert.ok(Math.abs(quarterTurn.right - 50) < 1e-9);
  assert.equal(quarterTurn.top, -105);
  assert.equal(quarterTurn.bottom, 95);
  assert.ok(Math.abs(quarterTurn.width - 80) < 1e-9);
  assert.equal(quarterTurn.height, 200);
  const diagonal = viewerEditorLayerBounds({ ...layer, scale: 1, rotation: 45 });
  assert.ok(Math.abs(diagonal.width - 98.99494936611666) < 1e-9);
  assert.ok(Math.abs(diagonal.width - diagonal.height) < 1e-9);
});

test("client coordinates invert canvas center, pan, zoom, layer rotation, and layer scale", () => {
  const local = clientPointToLayer(
    { clientX: 370, clientY: 250 },
    { left: 10, top: 20, width: 400, height: 300 },
    { x: 40, y: -20 },
    2,
    { x: 50, y: 20, scale: 2, rotation: 90, naturalWidth: 100, naturalHeight: 80 },
  );
  assert.ok(Math.abs(local.x - 65) < 1e-9);
  assert.ok(Math.abs(local.y - 35) < 1e-9);
});

test("stroke clipping records boundary crossings but never smears outside motion along an edge", () => {
  assert.deepEqual(clipLineToLayer({ x: 5, y: 5 }, { x: 15, y: 5 }, 10, 10), [{ x: 5, y: 5 }, { x: 10, y: 5 }]);
  const exited = clipStrokeSamples([{ x: 8, y: 5 }, { x: 20, y: 5 }], { width: 10, height: 10 });
  assert.deepEqual(exited.points, [{ x: 8, y: 5 }, { x: 10, y: 5 }]);
  const stayedOutside = clipStrokeSamples([{ x: 20, y: 8 }, { x: 20, y: 2 }], { width: 10, height: 10 }, exited.previousRawPoint);
  assert.deepEqual(stayedOutside.points, []);
  const reentered = clipStrokeSamples([{ x: 7, y: 2 }], { width: 10, height: 10 }, stayedOutside.previousRawPoint);
  assert.deepEqual(reentered.points, [{ x: 10, y: 2 }, { x: 7, y: 2 }]);
  assert.equal(normalizePaintStroke({ points: [{ x: -1, y: 5 }, { x: 4, y: 5 }] }, { width: 10, height: 10 }).points.length, 1);
});

test("paint strokes remain immutable and cloned layers do not share point arrays", () => {
  const stroke = normalizePaintStroke({ id: "one", color: "#ABC", size: 999, opacity: 75, points: [{ x: 1, y: 2 }] }, { width: 20, height: 20 });
  assert.equal(stroke.color, "#aabbcc");
  assert.equal(stroke.size, 300);
  assert.equal(stroke.opacity, .75);
  const appended = appendStrokePoints(stroke, [{ x: 3, y: 4 }], { width: 20, height: 20 });
  assert.notEqual(appended, stroke);
  assert.equal(stroke.points.length, 1);
  const original = { id: "image", paintStrokes: [appended] };
  const copied = cloneViewerLayer(original);
  copied.paintStrokes[0].points[0].x = 99;
  assert.equal(original.paintStrokes[0].points[0].x, 1);
});

test("manualLayout v1 remains readable and v2 round-trips rotation and paint without text", () => {
  const historyUrl = "/api/inference/history/assets/asset-1";
  const v1 = normalizeManualLayout({ version: 1, layers: [{ assetId: "asset-1", url: historyUrl, x: 3, y: 4, scale: .5 }] });
  assert.equal(v1.version, 1);
  assert.equal(v1.layers[0].x, 3);
  const v2 = normalizeManualLayout({ version: 2, layers: [
    { kind: "image", assetId: "asset-1", url: historyUrl, originalUrl: historyUrl, naturalWidth: 20, naturalHeight: 10, x: 1, y: 2, scale: 1, rotation: 25, paintStrokes: [{ tool: "brush", points: [{ x: 2, y: 3 }], color: "#c8acfb", size: 4, opacity: 1 }] },
    { kind: "text", text: "semantic", font: "Arial", size: 24, weight: 600, align: "center", lineHeight: 1.4, x: 8, y: 9, scale: 1.2, rotation: -30 },
  ] });
  assert.equal(v2.version, 2);
  assert.equal(v2.layers[0].paintStrokes[0].points.length, 1);
  assert.equal(v2.layers[0].rotation, 25);
  // Text layers were withdrawn: a layout saved while they existed keeps its pictures and loses only
  // the text, rather than refusing to reopen at all.
  assert.equal(v2.layers.length, 1);
  assert.deepEqual(normalizeManualLayout(v2), v2);
  const persisted = persistedManualLayout(v2);
  assert.equal(persisted.layout.version, 2);
  assert.ok(persisted.bytes > 0);
  assert.equal(persisted.pointCount, 1);
  assert.equal(persistedManualLayout({ ...v2, layers: [{ ...v2.layers[0], assetId: "", url: "data:image/png;base64,AA" }] }).layout, null);
});

test("manualLayout applies one global point budget across strokes", () => {
  const points = Array.from({ length: 10_000 }, (_, index) => ({ x: index % 10, y: index % 10 }));
  const valid = normalizeManualLayout({ version: 2, layers: Array.from({ length: 5 }, (_, index) => ({
    kind: "image", assetId: String(index), url: `/api/inference/history/assets/${index}`, naturalWidth: 20, naturalHeight: 20,
    paintStrokes: [{ id: String(index), points, size: 2 }],
  })) });
  assert.equal(viewerStrokePointCount(valid.layers), VIEWER_MAX_LAYOUT_POINTS);
  assert.equal(normalizeManualLayout({ ...valid, layers: [...valid.layers, valid.layers[0]] }), null, "oversized point collections are rejected before cloning rather than silently truncated");
  const tooManyStrokes = [{ kind: "image", assetId: "asset", url: "/api/inference/history/assets/asset", paintStrokes: Array.from({ length: VIEWER_MAX_LAYOUT_STROKES + 1 }, () => ({ points: [{ x: 0, y: 0 }] })) }];
  assert.equal(viewerStrokeCount(tooManyStrokes), VIEWER_MAX_LAYOUT_STROKES + 1);
  assert.equal(normalizeManualLayout({ version: 2, layers: tooManyStrokes }), null);
});

test("canvas allocation guard supports 8K but rejects unsafe edges and areas", () => {
  assert.deepEqual(viewerCanvasDimensions(8192, 8192), { width: 8192, height: 8192, pixels: MAX_EXPORT_PIXELS });
  assert.throws(() => viewerCanvasDimensions(8193, 8193, "测试画布"), /64 MP/);
  assert.throws(() => viewerCanvasDimensions(24577, 1), /24576/);
  assert.throws(() => viewerCanvasDimensions(NaN, 10), /尺寸无效/);
});

test("animated source and encoded output byte ceilings match the backend contracts", () => {
  const MiB = 1024 * 1024;
  assert.equal(VIEWER_MAX_ANIMATED_SOURCE_BYTES, 32 * MiB);
  assert.equal(VIEWER_MAX_OUTPUT_BYTES, 128 * MiB);
  assert.doesNotThrow(() => assertViewerByteBudget(31 * MiB, VIEWER_MAX_ANIMATED_SOURCE_BYTES, "GIF 拼图单个源图"));
  assert.throws(() => assertViewerByteBudget(33 * MiB, VIEWER_MAX_ANIMATED_SOURCE_BYTES, "GIF 拼图单个源图"), /32 MiB/);
  assert.doesNotThrow(() => assertViewerByteBudget(127 * MiB, VIEWER_MAX_OUTPUT_BYTES, "拼图保存数据"));
  assert.throws(() => assertViewerByteBudget(129 * MiB, VIEWER_MAX_OUTPUT_BYTES, "拼图保存数据"), /128 MiB/);
});

test("manualLayout versions and trust boundary are fail closed", () => {
  const external = { version: 2, layers: [{ kind: "image", assetId: "x", url: "https://example.test/image.png" }] };
  assert.equal(normalizeManualLayout({ ...external, version: 3 }), null);
  assert.equal(normalizeManualLayout(external), null);
  assert.equal(normalizeManualLayout(external, { trustedCurrentSession: true }).layers.length, 1);
  // A layout that was nothing but text has nothing left to restore.
  assert.equal(normalizeManualLayout({ version: 2, layers: [{ kind: "text", text: "x" }] }), null);
  assert.equal(normalizeManualLayout({ version: 2, layers: [{ kind: "text", text: "x" }] }, { trustedCurrentSession: true }), null);
  const oversizedSessionLayout = { version: 2, layers: Array.from({ length: 100 }, (_, index) => ({ kind: "image", assetId: `a${index}`, url: "https://example.test/image.png", name: "界".repeat(200) })) };
  assert.equal(normalizeManualLayout(oversizedSessionLayout), null, "untrusted layouts with unprotected sources are rejected");
  const trustedOversized = normalizeManualLayout(oversizedSessionLayout, { trustedCurrentSession: true });
  assert.equal(trustedOversized.layers.length, 100, "bounded current-session state can remain editable without becoming server metadata");
  assert.equal(persistedManualLayout(trustedOversized).layout, null);
  assert.equal(historyAssetIdFromUrl("/api/inference/history/assets/asset-a"), "asset-a");
  assert.equal(normalizeManualLayout({ version: 2, layers: [{ kind: "image", assetId: "asset-b", url: "/api/inference/history/assets/asset-a" }] }), null, "history URL token and assetId must bind exactly");
  assert.equal(normalizeManualLayout({ version: 2, layers: [{ kind: "image", assetId: "asset-a", url: "/api/inference/history/assets/asset-a", originalUrl: "/api/inference/history/assets/asset-b" }] }), null, "every persisted source alias must bind to the same assetId");
});

test("viewer input, source, and undo budgets are aggregate rather than per-operation only", () => {
  const MiB = 1024 * 1024;
  assert.equal(viewerFileBatchAdmission([{ size: 128 * MiB }, { size: 128 * MiB }]).ok, true);
  assert.match(viewerFileBatchAdmission([{ size: 128 * MiB + 1 }]).reason, /单张/);
  assert.match(viewerFileBatchAdmission([{ size: 128 * MiB }, { size: 128 * MiB }, { size: 1 }]).reason, /本批/);
  const dataUrl = "data:image/png;base64," + "A".repeat(400);
  assert.equal(viewerDataUrlBytes(dataUrl), 300);
  assert.equal(viewerDataUrlBytes("data:image/png;base64,TQ=="), 1);
  assert.equal(viewerDataUrlBytes("data:image/png;base64,TWE="), 2);
  assert.equal(viewerDataUrlBytes("data:image/png;base64,TWFu"), 3);
  assert.equal(viewerLayerSourceByteCount([{ sourceBytes: 9 }, { url: dataUrl }]), 309);
  const maximumAnimatedSources = Array.from({ length: 4 }, () => ({ sourceBytes: 32 * MiB }));
  assert.equal(viewerAnimatedBatchAdmission(maximumAnimatedSources).ok, true);
  assert.match(viewerAnimatedBatchAdmission([...maximumAnimatedSources, { sourceBytes: 1 }]).reason, /总量/);
  assert.equal(viewerLayerSourceByteCount([{ kind: "image", url: "/unknown.png" }]), Infinity, "unmeasured existing sources fail closed");

  const points = Array.from({ length: 50_000 }, (_, index) => ({ x: index, y: index }));
  const snapshot = { layers: [{ kind: "image", paintStrokes: [{ points }] }], snappedLayers: [], activeLayer: "x" };
  let stack = [];
  for (let index = 0; index < 5; index += 1) stack = boundedViewerUndoStack(stack, snapshot);
  assert.ok(stack.length < 5, "old undo snapshots must be evicted before 50 × 50k points accumulate");
  assert.ok(stack.reduce((sum, item) => sum + viewerStrokePointCount(item.layers), 0) <= VIEWER_MAX_UNDO_POINTS);
});

test("undo source accounting deduplicates identical sources but evicts different large sources", () => {
  const MiB = 1024 * 1024;
  const snapshot = (id, sourceBytes = 128 * MiB) => ({ layers: [{ id, kind: "image", url: `/source-${id}.png`, sourceBytes, paintStrokes: [] }], snappedLayers: [] });
  let stack = [];
  for (const id of ["a", "b", "c"]) stack = admitViewerUndoSnapshot(stack, snapshot(id)).stack;
  assert.deepEqual(stack.map((entry) => entry.layers[0].id), ["b", "c"], "oldest distinct source is evicted at the 256 MiB source ceiling");
  assert.equal(viewerUndoStackMetrics(stack).sourceBytes, VIEWER_MAX_UNDO_SOURCE_BYTES);

  stack = [];
  for (let index = 0; index < 50; index += 1) stack = admitViewerUndoSnapshot(stack, snapshot("same")).stack;
  assert.equal(stack.length, 50);
  assert.equal(viewerUndoStackMetrics(stack).sourceBytes, 128 * MiB, "the same immutable URL is charged once across all snapshots");

  const oversized = admitViewerUndoSnapshot(stack, snapshot("too-large", VIEWER_MAX_UNDO_SOURCE_BYTES + 1));
  assert.equal(oversized.saved, false);
  assert.match(oversized.reason, /不会加入撤销历史/);
  assert.equal(oversized.stack, stack, "a rejected single step cannot mutate the existing undo stack");
});

test("undo metrics include local data URLs and bounded manualLayout contents", () => {
  const local = "data:image/png;base64,QUJDRA==";
  const plain = viewerUndoSnapshotMetrics({ layers: [{ kind: "image", url: local, paintStrokes: [] }] });
  assert.ok(plain.sourceBytes >= local.length);
  const withLayout = viewerUndoSnapshotMetrics({ layers: [{
    kind: "image", url: local, paintStrokes: [], manualLayout: { version: 2, layers: [
      { kind: "image", assetId: "", url: local, paintStrokes: [{ points: [{ x: 1, y: 2 }] }] },
    ] },
  }] });
  assert.equal(withLayout.sourceBytes, plain.sourceBytes, "the same data URL in the layer and manualLayout is charged once");
  assert.ok(withLayout.structuralBytes > plain.structuralBytes);
  assert.equal(withLayout.points, plain.points + 1);
});

test("bounded file reads abort a hanging FileReader instead of merely ignoring its result", async () => {
  class HangingReader {
    static latest;
    constructor() { HangingReader.latest = this; }
    readAsDataURL() {}
    abort() { this.aborted = true; this.onabort?.(); }
  }
  const controller = new AbortController();
  const reading = readViewerFileAsDataUrl({ size: 12 }, { signal: controller.signal, FileReaderClass: HangingReader });
  controller.abort();
  await assert.rejects(reading, (error) => error.name === "AbortError");
  assert.equal(HangingReader.latest.aborted, true);
});

test("restore worker pool never exceeds four concurrent source loads", async () => {
  let active = 0;
  let peak = 0;
  const results = await mapViewerConcurrent(Array.from({ length: 17 }, (_, index) => index), 4, async (value) => {
    active += 1;
    peak = Math.max(peak, active);
    await new Promise((resolve) => setTimeout(resolve, 1));
    active -= 1;
    return value * 2;
  });
  assert.equal(peak, 4);
  assert.equal(results[16], 32);
});

test("one concurrent worker failure aborts all sibling loads and waits out late commits", async () => {
  const controller = new AbortController();
  let aborted = 0;
  let lateCommits = 0;
  const work = mapViewerConcurrent([0, 1, 2, 3], 4, async (value, _index, { signal }) => {
    if (value === 0) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      throw new Error("first worker failed");
    }
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => { lateCommits += 1; resolve(); }, 40);
      signal.addEventListener("abort", () => {
        clearTimeout(timer);
        aborted += 1;
        reject(signal.reason);
      }, { once: true });
    });
  }, { signal: controller.signal, controller });
  await assert.rejects(work, /first worker failed/);
  assert.equal(controller.signal.aborted, true);
  assert.equal(aborted, 3);
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.equal(lateCommits, 0);
});

test("mapViewerConcurrent links an external parent abort into its worker signal", async () => {
  const parent = new AbortController();
  let workerAborted = false;
  const work = mapViewerConcurrent([1], 1, (_value, _index, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener("abort", () => { workerAborted = true; reject(signal.reason); }, { once: true });
  }), { signal: parent.signal });
  await Promise.resolve();
  parent.abort();
  await assert.rejects(work, (error) => error.name === "AbortError");
  assert.equal(workerAborted, true);
});

test("rotated images expose no axis-unsafe resize handles while toolbar scale remains independent", () => {
  assert.deepEqual(viewerSafeResizeHandles({ kind: "image", rotation: 30 }), []);
  assert.equal(viewerSafeResizeHandles({ kind: "image", rotation: 0 }).length, 8);
  assert.equal(viewerSafeResizeHandles({ rotation: 0 }).length, 8, "every layer is a picture now");
});

test("slot confirmation replaces placeholder metadata with decoded intrinsic dimensions", () => {
  const entries = [{ asset: { url: "data:image/png;base64,AA", width: 1, height: 1 }, scale: 1 }];
  const resolved = resolvedCollageEntries(entries, [{ naturalWidth: 2048, naturalHeight: 1024 }]);
  assert.deepEqual([resolved[0].asset.width, resolved[0].asset.height, resolved[0].asset.naturalWidth], [2048, 1024, 2048]);
  assert.equal(entries[0].asset.width, 1, "resolution is immutable and cannot mutate slot state behind React");
});

test("clipboard paste requires the exact fresh marker or actual image files", () => {
  const copied = { layer: { id: "layer-1" }, copiedAt: 1000 };
  assert.equal(viewerClipboardPasteIntent({ copied, marker: "external text", now: 2000 }), "none");
  assert.equal(viewerClipboardPasteIntent({ copied, marker: "", now: 2000 }), "none", "old in-memory copy never substitutes for external text");
  assert.equal(viewerClipboardPasteIntent({ copied, marker: "XIRAI_LAYER:layer-1", now: 2000 }), "internal-layer");
  assert.equal(viewerClipboardPasteIntent({ copied, marker: "external text", hasImages: true, now: 2000 }), "images");
  assert.equal(viewerClipboardPasteIntent({ copied, marker: "XIRAI_LAYER:layer-1", now: 400_001 }), "none");
});

test("edge line validation owns the purple default and the full 50px contract", () => {
  assert.equal(VIEWER_DEFAULT_COLOR, "#c8acfb");
  assert.deepEqual(normalizeViewerEdgeLine({ enabled: true, color: "bad", style: "script", width: 500 }), { enabled: true, color: "#c8acfb", style: "solid", width: 50 });
  assert.equal(serializeViewerLayer({ assetId: "a" }).kind, "image");
});

test("App gesture transactions keep cancel separate from one-shot undo commits", async () => {
  const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const finish = app.slice(app.indexOf("const finishViewerPointer ="), app.indexOf("const finishViewerResizeForDisable ="));
  assert.match(finish, /if \(cancelled\)[\s\S]*restoreViewerSnapshot\(drag\.undoSnapshot\)[\s\S]*return;[\s\S]*if \(drag\?\.changed\) saveViewerUndo\(drag\.undoSnapshot\)/);
  assert.match(app, /onPointerCancel=\{\(event\) => finishViewerPointer\(event, true\)\}/);
  assert.match(app, /onLostPointerCapture=\{loseViewerPointer\}/);
  assert.match(app, /getCoalescedEvents/);
  assert.match(app, /event\.button !== 0 \|\| event\.isPrimary === false/);
});

test("render wiring separates ordinary images, raster replay, and semantic text without edging text", async () => {
  const [app, raster] = await Promise.all([
    readFile(new URL("../src/App.jsx", import.meta.url), "utf8"),
    readFile(new URL("../src/ViewerRasterLayer.jsx", import.meta.url), "utf8"),
  ]);
  assert.match(app, /painted[\s\S]{0,120}<ViewerRasterLayer layer=\{layer\}/);
  assert.match(app, /viewerEdgeLine\.enabled && \["top", "right", "bottom", "left"\]/);
  assert.match(app, /rotate\(\$\{normalizeRotation\(layer\.rotation\)\}deg\)/);
  assert.doesNotMatch(raster.match(/useEffect\([\s\S]*?\}, \[[^\]]*\]\);/)?.[0] || "", /onError\]/);
  assert.match(raster, /onErrorRef\.current/);
  assert.match(raster, /\}, \[layer\.originalUrl, layer\.url\]\);/, "stroke replay must not decode the source image again");
  assert.match(raster, /\[loadedImage, layer\.naturalWidth, layer\.naturalHeight, layer\.paintStrokes\]/);
});

test("static composition draws replayed layers in array order and GIF editing is explicitly static", async () => {
  const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const collage = app.slice(app.indexOf("const createManualCollage ="), app.indexOf("const selectGeneratedOutput ="));
  assert.match(collage, /sourceLayers = cloneViewerLayers\(viewerLayers\)/);
  assert.match(collage, /const source = await viewerLayerBitmap\(layer, \{ signal: token\.signal \}\)/);
  assert.match(collage, /nodes\.forEach\(\(item\) => \{[\s\S]*drawViewerLayer\(context, item\.layer, item\.source/);
  assert.match(collage, /hasAnimatedSource && !hasViewerEdits\(sourceLayers\)/);
  assert.match(collage, /GIF 与编辑内容已静态合成为 PNG/);
  assert.match(collage, /version: 2/);
  assert.match(collage, /persistedManualLayout\(manualLayout\)/);
});

test("App guards every editor-sized canvas and sends shared-edge metadata to GIF rendering", async () => {
  const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  for (const purpose of ["图片编辑画布", "图层复制画布", "图片复制画布", "模板拼图画布", "手动编辑导出画布", "GIF 拼图画布"]) {
    assert.ok(app.includes(`viewerCanvasDimensions(`) && app.includes(`"${purpose}"`), `${purpose} needs a pre-allocation guard`);
  }
  assert.match(app, /hidden_sides: slotEdges\[index\] \|\| \[\]/);
  assert.match(app, /hidden_sides: layerEdges\[item\.layer\.id\] \|\| \[\]/);
  assert.match(app, /renderAnimatedCollage\([\s\S]*\{ signal: token\.signal \}/);
});

test("App admits bytes before reads and threads operation abort signals through every editor decoder", async () => {
  const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(app, /viewerFileBatchAdmission\(imageFiles\)[\s\S]{0,500}beginAppend\("drop"\)/);
  assert.match(app, /readImageFile\(file, \{ signal: token\.signal \}\)/);
  assert.match(app, /function loadBrowserImage\(source, \{ signal \} = \{\}\)/);
  assert.match(app, /imageSourceDataUrl\([^\n]+\{ signal: token\.signal \}/);
  assert.match(app, /viewerLayerBitmap\(layer, \{ signal \} = \{\}\)/);
  assert.match(app, /mapViewerConcurrent\(sourceLayers, 4/);
  assert.match(app, /VIEWER_MAX_LAYER_SOURCE_BYTES/);
  assert.match(app, /admitViewerUndoSnapshot/);
  assert.match(app, /controller: token\.controller/);
  assert.match(app, /VIEWER_MAX_ANIMATED_SOURCE_BYTES/);
  assert.match(app, /VIEWER_MAX_OUTPUT_BYTES/);
});


test("text layers stay withdrawn from the editor, the model and the saved layout", async () => {
  const [app, editor, css, backend] = await Promise.all([
    readFile(new URL("../src/App.jsx", import.meta.url), "utf8"),
    readFile(new URL("../src/viewer-editor.js", import.meta.url), "utf8"),
    readFile(new URL("../src/styles.css", import.meta.url), "utf8"),
    readFile(new URL("../backend/inference_server.py", import.meta.url), "utf8"),
  ]);
  // The tool, its gestures, its layer model and its stylesheet are gone, not merely hidden.
  assert.doesNotMatch(app, /viewerTextDefaults|editingViewerText|beginViewerTextEdit|createViewerText|viewerTextBox|measureViewerText|viewerTextMarquee|viewerLayerKind|viewerPaintOrder/);
  assert.doesNotMatch(editor, /normalizeTextLayer|drawTextLayer|measureTextLayer|wrapTextLines|viewerTextBox|VIEWER_TEXT_STYLES/);
  assert.doesNotMatch(css, /viewer-text-editor|viewer-text-content|viewer-text-marquee|tool-text|layer-rotate/);
  assert.doesNotMatch(app, /文字工具|文字图层|文字属性/);
  // Three tools remain, and the keyboard offers exactly those.
  assert.match(app, /\{ v: "move", b: "brush", e: "eraser" \}/);
  assert.match(editor, /VIEWER_TOOLS = Object\.freeze\(\["move", "brush", "eraser"\]\)/);
  // Every layer is a picture, so an upright one offers all eight handles and none can be rotated.
  assert.equal(viewerSafeResizeHandles({ rotation: 0 }).length, 8);
  assert.doesNotMatch(app, /kind: "rotate"|layer-rotate-handle/);
  // Nothing may submit a text layer any more, and a stored one is dropped on the way back in.
  assert.match(backend, /manual_layout cannot contain text layers/);
  assert.doesNotMatch(backend, /_COLLAGE_TEXT_LAYER_KEYS|COLLAGE_MAX_LAYOUT_TEXT/);
  assert.match(editor, /if \(source\.kind === "text" \|\| source\.type === "text"\) continue;/);
});

test("every name App.jsx imports from the editor module is really exported by it", async () => {
  // A deleted export leaves a bare identifier behind: the bundler treats it as a global and builds
  // happily, so the page only fails when the user reaches that line. This catches it at test time.
  const [app, module] = await Promise.all([
    readFile(new URL("../src/App.jsx", import.meta.url), "utf8"),
    import("../src/viewer-editor.js"),
  ]);
  const block = app.slice(app.indexOf("import {"), app.indexOf('} from "./viewer-editor.js";'));
  const imported = block.split("\n").map((line) => line.trim().replace(/,$/, "")).filter((name) => /^[A-Za-z_$][\w$]*$/.test(name));
  assert.ok(imported.length > 20, `parsed only ${imported.length} imported names`);
  const missing = imported.filter((name) => !(name in module));
  assert.deepEqual(missing, [], "App.jsx imports names the editor module does not export");
});
