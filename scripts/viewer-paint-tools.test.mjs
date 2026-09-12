import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { viewerBrushCursorSize, VIEWER_BRUSH_CURSOR_MAX_PX, VIEWER_BRUSH_CURSOR_MIN_PX } from "../src/viewer-geometry.js";
import { persistedManualLayout, raiseViewerLayer, serializeViewerLayer, viewerRestoredUndoStack, VIEWER_MAX_UNDO_STEPS } from "../src/viewer-editor.js";

test("the brush ring is the size of the pixels the stroke covers, at any zoom or layer scale", () => {
  // A stroke is stored in the layer's own pixels, so what the user sees painted is the brush size
  // taken through that layer's scale and then through the viewer's zoom.
  assert.equal(viewerBrushCursorSize(40, 1, 1), 40);
  assert.equal(viewerBrushCursorSize(40, 2, 1), 80);
  assert.equal(viewerBrushCursorSize(40, 2, 0.5), 40);
  assert.equal(viewerBrushCursorSize(300, 8, 8), VIEWER_BRUSH_CURSOR_MAX_PX);
  // A one-pixel brush zoomed far out still has to be aimable.
  assert.equal(viewerBrushCursorSize(1, 0.05, 1), VIEWER_BRUSH_CURSOR_MIN_PX);
  for (const broken of [[NaN, 1, 1], [40, NaN, 1], [0, 1, 1], [-40, 1, 1], [40, 1, 0]]) {
    assert.ok(viewerBrushCursorSize(...broken) >= VIEWER_BRUSH_CURSOR_MIN_PX);
  }
});

test("the canvas wires a pointer-following ring and the middle button, and offers no text tool", () => {
  const app = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  const css = fs.readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
  // The ring is a DOM circle, not a `cursor:` image: CSS cursors stop at 128px and a 300px brush
  // zoomed in needs far more. It is driven through a ref so a stroke does not repaint the app.
  assert.match(app, /viewerBrushCursorSize\(viewerBrush\.size, viewerZoom, at\.scale\)/);
  assert.match(app, /node\.style\.width = `\$\{diameter\}px`/);
  assert.match(app, /className=\{`viewer-brush-cursor \$\{viewerTool\} \$\{viewerPointerOverCanvas \? "" : "away"\}`\}/);
  assert.match(app, /const moveViewerImage = \(event\) => \{\s*moveViewerBrushCursor\(event\);/);
  for (const tool of ["brush", "eraser"]) assert.match(css, new RegExp(`\\.image-viewer-canvas\\.tool-${tool}[^\\n]*cursor: none;`));
  assert.match(css, /\.viewer-brush-cursor\.away \{ display: none; \}/);
  // The middle button moves layers and pans whichever tool is selected, and the browser must not
  // claim it for autoscroll or for a primary-selection paste.
  assert.match(app, /const middleButton = event\.button === 1;/);
  assert.match(app, /if \(!middleButton && \(event\.button !== 0 \|\| event\.isPrimary === false\)\) return;/);
  assert.match(app, /if \(event\.button === 1\) \{\s*startViewerPan\(event\);/);
  assert.match(app, /const suppressViewerMiddleClick = \(event\) => \{\s*if \(event\.button === 1\) event\.preventDefault\(\);/);
  assert.match(app, /onMouseDown=\{suppressViewerMiddleClick\}/);
  assert.match(app, /onAuxClick=\{suppressViewerMiddleClick\}/);
  // Pressing bare canvas still drops the selection, so a picture stops framing itself.
  assert.match(app, /const startViewerDrag = \(event\) => \{[\s\S]{0,400}setActiveViewerLayer\(""\);/);
  // No text tool remains to sweep a box or to write into one.
  assert.doesNotMatch(app, /startViewerTextMarquee|viewer-text-marquee|text-box/);
  assert.doesNotMatch(css, /viewer-text-marquee|tool-text/);
});

test("painting brings the picture to the front, keeping every other layer's order", () => {
  const layers = [{ id: "a" }, { id: "b" }, { id: "c" }];
  // Array order is the stacking order, so the painted picture moving last is it coming to the front.
  assert.deepEqual(raiseViewerLayer(layers, "a").map((layer) => layer.id), ["b", "c", "a"]);
  assert.deepEqual(raiseViewerLayer(layers, "c").map((layer) => layer.id), ["a", "b", "c"]);
  // The changes ride along with the raise, so a stroke and its restacking are one update.
  const painted = raiseViewerLayer(layers, "b", { paintStrokes: [{ id: "s" }] });
  assert.deepEqual(painted.map((layer) => layer.id), ["a", "c", "b"]);
  assert.deepEqual(painted.at(-1), { id: "b", paintStrokes: [{ id: "s" }] });
  // Sources are never mutated, and an unknown id changes nothing.
  assert.deepEqual(layers.map((layer) => layer.id), ["a", "b", "c"]);
  assert.equal(raiseViewerLayer(layers, "missing"), layers);
  assert.deepEqual(raiseViewerLayer(null, "a"), []);
});

test("drawing offers one Save, which composes and writes the picture it just composed", () => {
  const app = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  // The button appears because there are strokes to keep, and it replaces the collage's apply
  // button rather than sitting beside it, so only one primary action is ever offered.
  assert.match(app, /const viewerHasPaint = viewerLayers\.some\(hasLayerPaint\);/);
  assert.match(app, /!activeCollage && !collageResult && viewerHasPaint && <div className="viewer-toolbar-group viewer-toolbar-results"[\s\S]{0,400}保存绘制/);
  assert.match(app, /!activeCollage && !collageResult && !viewerHasPaint && \(viewerLayers\.length > 1 \|\| viewerHasEditableContent\)/);
  // Compose hands its result straight to save: reading it back from React state would save the
  // previous composition, or nothing at all on the first click.
  assert.match(app, /const composed = await createManualCollage\(\{ quiet: true \}\);\s*if \(!composed\) return;\s*const saved = await saveCollage\(composed\);/);
  assert.match(app, /const saveCollage = async \(explicitResult = null\) => \{\s*const result = explicitResult && explicitResult\.dataUrl \? explicitResult : collageResult;/);
  // The strokes land in the file, and the layout that can peel them off again is saved with it.
  assert.match(app, /manual_layout: result\.persistedManualLayout/);
  // The canvas survives the save: composing flattens it, so the layers are put back afterwards and
  // the user can carry on drawing on the same layers their undo history refers to.
  assert.match(app, /const before = cloneViewerLayers\(viewerLayers\);[\s\S]{0,600}setViewerLayers\(before\);\s*setActiveViewerLayer\(activeBefore\);\s*setCollageResult\(null\);/);
  // The stroke itself raises its picture as it is added.
  assert.match(app, /raiseViewerLayer\(current, layer\.id, \{ originalUrl: painted\.originalUrl \|\| painted\.url, paintStrokes:/);
});

test("reopening a saved picture brings its strokes back as strokes", () => {
  const app = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  // Opening from the history restores the saved layout first; only if that fails does the flat
  // file get shown, so a deleted source degrades to today's behaviour instead of an error.
  assert.match(app, /const layout = viewerAssetLayout\(asset\);\s*if \(layout && await restoreManualCollage\(layout, \{ trustedCurrentSession: asset\?\.manualLayoutTrusted === true \}\)\) \{/);
  assert.match(app, /const viewerAssetLayout = \(asset\) => asset\?\.manual_layout \|\| asset\?\.manualLayout \|\| null;/);
  // Every history entry point funnels through that one function.
  assert.match(app, /const openHistoryCard = \(card\) => \{[\s\S]{0,320}focusViewerAsset\(card\.preview\)/);
  // A restored layer carries its strokes, which is what makes undo and redrawing possible.
  assert.match(app, /paintStrokes: Array\.isArray\(asset\.paintStrokes\) \? cloneViewerLayer\(asset\)\.paintStrokes : \[\]/);
});

test("a picture drawn on is addressed so that its strokes can be saved as strokes", () => {
  const strokes = [{ tool: "brush", color: "#c8acfb", size: 4, opacity: 1, points: [{ x: 1, y: 1 }, { x: 2, y: 2 }] }];
  const layer = (url) => serializeViewerLayer({ kind: "image", assetId: "abc-123", url, originalUrl: url, naturalWidth: 8, naturalHeight: 8, paintStrokes: strokes });
  // A saved layout may only name protected history routes, and that is what makes the strokes
  // recoverable when the picture is reopened.
  const history = persistedManualLayout({ version: 2, layers: [layer("/api/inference/history/assets/abc-123")] });
  assert.ok(history.layout, `a history-addressed picture must persist: ${history.reason}`);
  assert.equal(history.layout.layers[0].paintStrokes[0].points.length, 2);
  // The job route a freshly generated picture is served from carries a cache-buster, so a layout
  // naming it is refused and the strokes would be flattened into the file for good.
  const job = persistedManualLayout({ version: 2, layers: [layer("/api/inference/jobs/j1/images/0?v=42")] });
  assert.equal(job.layout, null);
  // So the viewer opens a generated picture by its history token, which names the same file.
  const app = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(app, /url: selectedOutput\?\.asset_id \? `\/api\/inference\/history\/assets\/\$\{selectedOutput\.asset_id\}` : generatedImage,/);
});

test("a reopened picture can undo the strokes it was saved with", () => {
  const stroke = (id) => ({ id, tool: "brush", points: [{ x: 1, y: 1 }] });
  const layers = [{ id: "a", paintStrokes: [stroke("s1"), stroke("s2")] }, { id: "b", paintStrokes: [stroke("s3")] }];
  const stack = viewerRestoredUndoStack(layers, { activeLayer: "b" });
  // One step per stroke, oldest first, which is the order an undo stack pops from: the first undo
  // returns to "s3 not yet drawn", the next to "s2 not yet drawn", and so on back to the bare
  // pictures.
  assert.equal(stack.length, 3);
  assert.deepEqual(stack.map((state) => state.layers.map((layer) => layer.paintStrokes.length)), [[0, 0], [1, 0], [2, 0]]);
  assert.equal(stack.at(-1).activeLayer, "b");
  // Nothing drawn, nothing to undo.
  assert.deepEqual(viewerRestoredUndoStack([{ id: "a", paintStrokes: [] }]), []);
  assert.deepEqual(viewerRestoredUndoStack(null), []);
  // The stack stops at the same depth a session's own history does.
  assert.equal(viewerRestoredUndoStack([{ id: "a", paintStrokes: Array.from({ length: 80 }, (_, index) => stroke(`s${index}`)) }]).length, VIEWER_MAX_UNDO_STEPS);
  // Snapshots are copies: undoing must not hand back arrays the canvas is still drawing from.
  assert.notEqual(stack[1].layers[0].paintStrokes, layers[0].paintStrokes);
  const app = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(app, /viewerUndo\.current = viewerRestoredUndoStack\(restored, \{ activeLayer: restored\.at\(-1\)\.id \}\);/);
});
